#!/usr/bin/env python3
"""DDP entry point for Exp 18 Test 5b onward — lean bare-SSM training path.

Sibling to ``runner_exp18.py``. The only functional difference is that
training goes through ``chaoscontrol.train_ssm.train_ssm_for_budget``
(chunked LM-head backward, manual DDP all-reduce) instead of
``chaoscontrol.training.train_chaoscontrol_for_budget``. Everything
else — launch convention, rank-0 eval, result-JSON schema — is
identical so ``_harness.run_parallel_ddp_matrix`` treats results from
either runner uniformly.

Why a sibling rather than a flag: ``runner_exp18.py`` and
``training.train_chaoscontrol_for_budget`` are frozen for
reproducibility of every prior Exp 18 test. Threading a trainer flag
through the old runner would touch the frozen path; a sibling does
not.

Config contract additions vs runner_exp18.py:
    - ``optimizer`` is required (no default). Only {"adamw","muon","lamb"}
      are recognized and the training call doesn't construct the
      optimizer itself (that's ``train_ssm_for_budget``'s contract).
    - ``chunk_size`` is required — the time-axis chunk size for the
      LM-head chunked backward. 64 is the Exp 18 default; smaller
      values reduce peak logits-grad memory at the cost of per-step
      Python overhead.

Rejected configs (hard error at entry, matches
``train_ssm._reject_unsupported``):
    wernicke, outer_model, semantic_tier, posterior, bucket_prototypes.

Expected launch:
    torchrun --standalone --nproc_per_node=N \\
        experiments/18_throughput_levers/runner_exp18_ssm.py \\
        --config <path.yaml> --data-path <fineweb-dir> \\
        --sp-model-path <sp.model> --output-json <result.json>

The script also runs cleanly as a single process via plain
``python``, in which case it resolves to rank=0, world_size=1 and
does not touch ``torch.distributed``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "17_local_attn_sidecar"))

from chaoscontrol.core import verify_diag_recurrence  # noqa: E402
from chaoscontrol.data import (  # noqa: E402
    build_lm_starts,
    choose_eval_starts,
    resolve_device,
    resolve_param_dtype,
)
from chaoscontrol.optim.lamb import LAMB  # noqa: E402
from chaoscontrol.optim.muon import Muon  # noqa: E402
from chaoscontrol.train_ssm import (  # noqa: E402
    _reject_unsupported,
    train_ssm_for_budget,
)

# Reuse model + eval machinery from runner_exp17 (same as runner_exp18).
from runner_exp17 import (  # noqa: E402
    build_model,
    build_sentencepiece_luts,
    evaluate_bpb_sp,
    load_sp_data,
)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    return int(val)


def _init_distributed(world_size_override: int | None) -> tuple[int, int, int]:
    """Initialize the process group if required by the launch context.

    Identical semantics to ``runner_exp18._init_distributed`` — returns
    (rank, world_size, local_rank). Single-process launch returns
    (0, 1, 0) without touching ``torch.distributed``.
    """
    env_world = _env_int("WORLD_SIZE", 1)
    target_world = world_size_override if world_size_override is not None else env_world
    if target_world <= 1:
        return 0, 1, 0

    if not (dist.is_available() and dist.is_initialized()):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = _env_int("LOCAL_RANK", 0)
    return rank, world_size, local_rank


def _pick_device(local_rank: int, config_device: str) -> torch.device:
    if torch.cuda.is_available() and config_device in ("auto", "cuda"):
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return resolve_device(config_device)


def _build_optimizer(
    optimizer_name: str,
    model: torch.nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Construct the optimizer for the bare-SSM config.

    Matches the optimizer wiring in
    ``chaoscontrol.training.train_chaoscontrol_for_budget`` so Test 5b
    is comparable to the original Test 5 at the regime level — same
    optimizer class, same hyperparameters, different training loop.

    No ``structured_proj`` / ``tokenizer`` aux params because the
    bare-SSM path does not support MC / metabolic / codebook
    alignment (those configs are rejected by
    ``train_ssm._reject_unsupported``).
    """
    params = list(model.parameters())
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    if optimizer_name == "muon":
        optimizer = Muon(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
            adamw_lr=base_lr,
            adamw_weight_decay=weight_decay,
        )
        # Bind param names so Muon's ndim-based classifier can key on
        # tensor names if a future config hand-specifies matrix-param
        # overrides. Identical to training.py's binding pattern.
        optimizer.bind_param_names(list(model.named_parameters()))
        return optimizer
    if optimizer_name == "lamb":
        return LAMB(params, lr=base_lr, weight_decay=weight_decay)
    raise ValueError(
        f"Unknown optimizer: {optimizer_name!r}. "
        "Expected one of {'adamw', 'muon', 'lamb'}."
    )


def _shard_train_starts(
    train_starts: list[int],
    *,
    rank: int,
    world_size: int,
) -> list[int]:
    """Stride-shard ``train_starts`` so each rank owns a disjoint shard.

    Matches ``training.train_chaoscontrol_for_budget``'s rank-aware
    sampler (``[s for i, s in enumerate(train_starts) if i % world_size
    == rank]``). ``train_ssm_for_budget`` itself only diverges the
    per-rank sampling RNG via ``seed + rank`` — the caller is expected
    to pre-shard if strict disjoint-window semantics matter. For Exp 18
    comparability we want strict sharding because the original Test 5
    runs under strict sharding.
    """
    if world_size <= 1:
        return list(train_starts)
    sharded = [s for i, s in enumerate(train_starts) if i % world_size == rank]
    if len(sharded) < 1:
        raise RuntimeError(
            f"rank {rank} has no train_starts after stride-sharding "
            f"({len(train_starts)} total starts across world_size={world_size}). "
            "Either the corpus is too small for this DDP world_size or "
            "the caller is constructing train_starts incorrectly."
        )
    return sharded


def run_ddp(
    config: dict[str, Any],
    *,
    data_path: str,
    sp_model_path: str,
    budget_seconds: float,
    output_json: str | None,
    world_size_override: int | None,
) -> dict[str, Any]:
    """Run one Exp 18 training condition via the lean train_ssm path."""
    rank, world_size, local_rank = _init_distributed(world_size_override)
    is_rank0 = rank == 0
    ddp_active = world_size > 1

    device = _pick_device(local_rank, config.get("device", "auto"))
    param_dtype = resolve_param_dtype(config.get("dtype", "bf16"), device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    seed = int(config.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Per-rank compile warmup — each rank resolves its own chunked-scan
    # backend before the first training step so no rank lags into the
    # collective with a cold compile state. Same role as in runner_exp18.
    verify_diag_recurrence(device)

    train_tokens, val_tokens, _ = load_sp_data(data_path, config["vocab_size"])

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, config["vocab_size"], device
    )

    seq_len = int(config["seq_len"])
    stride = int(config.get("stride", seq_len // 2))
    batch_size = int(config["batch_size"])
    eval_batches = int(config.get("eval_batches", 16))

    train_starts_all = build_lm_starts(int(train_tokens.numel()), seq_len, stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), seq_len, stride)
    eval_starts = choose_eval_starts(
        val_starts, batch_size=batch_size, eval_batches=eval_batches, seed=seed,
    )
    # Pre-shard so each rank sees strictly disjoint training windows —
    # matches the frozen training loop's sampler. train_ssm_for_budget
    # itself does not shard.
    train_starts = _shard_train_starts(
        train_starts_all, rank=rank, world_size=world_size,
    )

    model = build_model(config, device, param_dtype)
    # Fail fast at runner entry — not deep inside the training loop.
    # train_ssm_step rejects these too, but by then we've already
    # allocated VRAM, sharded data, and constructed the optimizer.
    # Catching a misconfigured wernicke/outer_model/posterior here
    # saves a few minutes of pod time per bad seed.
    _reject_unsupported(model)
    precision = str(config.get("precision", "bf16"))
    if precision == "fp8":
        from chaoscontrol.precision import maybe_promote_linears_to_te
        n_promoted = maybe_promote_linears_to_te(model, enabled=True)
        if is_rank0:
            print(
                f"[rank 0] promoted {n_promoted} nn.Linear -> te.Linear for fp8",
                flush=True,
            )
    model_params = sum(p.numel() for p in model.parameters())

    if is_rank0:
        print(
            f"[rank {rank}/{world_size}] model dim={config['model_dim']} "
            f"layers={config['num_layers']} params={model_params:,}",
            flush=True,
        )

    # Build optimizer before calling train_ssm — train_ssm_for_budget
    # takes an optimizer instance, unlike train_chaoscontrol_for_budget
    # which constructs one internally from an optimizer-name string.
    optimizer_name = str(config.get("optimizer", "")).strip()
    if not optimizer_name:
        raise ValueError(
            "runner_exp18_ssm requires config['optimizer'] to be set "
            "explicitly (one of {'adamw', 'muon', 'lamb'}). The bare-SSM "
            "training path does not default like the frozen runner does."
        )
    base_lr = float(config.get("base_lr", 2e-3))
    weight_decay = float(config.get("weight_decay", 1e-2))
    optimizer = _build_optimizer(
        optimizer_name, model, base_lr=base_lr, weight_decay=weight_decay,
    )

    chunk_size = int(config.get("chunk_size", 64))

    train_result = train_ssm_for_budget(
        model,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        optimizer=optimizer,
        budget_seconds=budget_seconds,
        chunk_size=chunk_size,
        grad_clip_norm=float(config.get("grad_clip_norm", 1.0)),
        seed=seed,
        rank=rank,
        world_size=world_size,
        precision=precision,
    )

    # train_ssm_for_budget installs its own teardown barrier when
    # ddp_active, so by the time it returns all ranks are lockstep.
    # An extra barrier here is redundant but harmless and matches the
    # runner_exp18 shape.
    if ddp_active:
        dist.barrier()

    # Rank-0-only eval on the full val set. Same rationale as runner_exp18:
    # the model is small enough that single-rank eval dominated by the
    # harness's barrier cost is the right trade-off vs per-rank eval
    # with an all-reduce.
    if is_rank0:
        eval_result = evaluate_bpb_sp(
            model,
            tokens=val_tokens,
            eval_starts=eval_starts,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
    else:
        eval_result = {}

    if ddp_active:
        dist.barrier()

    history_final = train_result["history"][-1] if train_result["history"] else {}
    # Read peak_vram_mb from the training result, not from a
    # post-eval measurement. The frozen runner reads it the same way
    # (runner_exp18.py:250 <- training.py:911), so both paths' JSONs
    # report "training-only peak memory" and are directly comparable.
    train_summary = {
        "steps": int(train_result["steps"]),
        "elapsed_s": float(train_result["elapsed_s"]),
        "steps_per_second": float(
            train_result["steps"] / max(train_result["elapsed_s"], 1e-9)
        ),
        "final_loss": float(history_final.get("loss", float("nan"))),
        "peak_vram_mb": float(train_result.get("peak_vram_mb", 0.0)),
        "ddp_rank": int(train_result.get("rank", rank)),
        "ddp_world_size": int(train_result.get("world_size", world_size)),
    }

    result = {
        "config": config,
        "params": model_params,
        "train": train_summary,
        "eval": eval_result,
    }

    # Fail closed on numerically poisoned runs — matches runner_exp18's
    # contract so result_is_finite() in the orchestrator treats either
    # runner's outputs the same way.
    if is_rank0:
        violations: list[str] = []
        if not math.isfinite(train_summary["final_loss"]):
            violations.append(
                f"train.final_loss={train_summary['final_loss']} is not finite"
            )
        for key in ("bpb", "loss"):
            if key in eval_result:
                val = float(eval_result[key])
                if not math.isfinite(val):
                    violations.append(f"eval.{key}={val} is not finite")
        if violations:
            raise RuntimeError(
                "runner_exp18_ssm: refusing to write poisoned result JSON — "
                + "; ".join(violations)
            )

    if is_rank0 and output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(result, indent=2, default=str))
        tmp_path.rename(out_path)

    if is_rank0:
        bpb_display = eval_result.get("bpb", float("nan"))
        print(
            f"[rank {rank}/{world_size}] done: bpb={bpb_display:.4f} "
            f"steps={train_summary['steps']} "
            f"steps/s={train_summary['steps_per_second']:.2f} "
            f"peak_vram={train_summary['peak_vram_mb']:.1f} MB",
            flush=True,
        )

    if ddp_active and dist.is_initialized():
        dist.destroy_process_group()

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 5b+ runner — lean train_ssm path"
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help=(
            "Override the DDP world size (defaults to WORLD_SIZE env var set "
            "by torchrun). Use 1 to force single-device even inside a "
            "torchrun launch."
        ),
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    run_ddp(
        config,
        data_path=args.data_path,
        sp_model_path=args.sp_model_path,
        budget_seconds=args.budget,
        output_json=args.output_json,
        world_size_override=args.world_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
