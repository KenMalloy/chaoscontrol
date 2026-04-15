#!/usr/bin/env python3
"""DDP entry point for Experiment 18 (throughput lever stack).

This is the minimal DDP runner: init process group, build model, call
``train_chaoscontrol_for_budget`` (which wraps the model in DDP internally),
run rank-0-only eval, tear down. Expected launch convention:

    torchrun --standalone --nproc_per_node=8 \
        experiments/18_throughput_levers/runner_exp18.py \
        --config <path.yaml> \
        --data-path <fineweb-dir> \
        --sp-model-path <sp8192.model> \
        --output-json <result.json>

The script also runs cleanly as a single process via plain ``python``, in
which case it resolves to rank=0, world_size=1 and does not touch
``torch.distributed`` at all.

Design notes:
    - Only imports ``build_model`` / ``build_sentencepiece_luts`` / data
      helpers from the Exp 17 runner. The rest of the training loop lives in
      ``chaoscontrol.training`` so both experiments share the same kernel.
    - Eval runs on rank 0 only over the full val set. Other ranks wait at a
      barrier. This is the simplest correct strategy given that the full val
      set fits comfortably on a single GPU for the ChaosControl model sizes
      targeted by Exp 18.
    - ``verify_diag_recurrence(device)`` is called per-rank before the first
      training step to pre-resolve the chunked-scan backend (and its compile
      fallback, if any) so no rank stalls on first-step compile while others
      start the collective.
"""
from __future__ import annotations

import argparse
import json
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
from chaoscontrol.training import train_chaoscontrol_for_budget  # noqa: E402

# Reuse only the minimum necessary from runner_exp17: model builder and the
# SP competition-bpb evaluator. Everything else (arg parsing, rank wiring,
# etc.) is Exp-18-local.
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

    Returns (rank, world_size, local_rank).

    Behavior:
        - If WORLD_SIZE > 1 in the environment (torchrun-style launch), call
          ``init_process_group`` with NCCL on CUDA or GLOO on CPU. The
          process group must be torn down via ``destroy_process_group``
          before exit.
        - Otherwise, return (0, 1, 0) and do not touch torch.distributed.
        - ``--world-size N`` (when N > 1) lets the user force DDP even with
          only env vars partially set; this primarily exists for
          troubleshooting and should rarely be used directly (torchrun is
          the supported launch path).
    """
    env_world = _env_int("WORLD_SIZE", 1)
    target_world = world_size_override if world_size_override is not None else env_world
    if target_world <= 1:
        return 0, 1, 0

    if not (dist.is_available() and dist.is_initialized()):
        # Pick a backend based on the device the runner will use. CUDA -> NCCL
        # is the standard path; GLOO is for CPU-only fallbacks (e.g. CI). The
        # caller ensures RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT are set;
        # torchrun sets them automatically.
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = _env_int("LOCAL_RANK", 0)
    return rank, world_size, local_rank


def _pick_device(local_rank: int, config_device: str) -> torch.device:
    """Choose the device for this rank.

    On CUDA with DDP: pin to ``cuda:local_rank``. On CPU / single-device:
    delegate to ``resolve_device``. This ensures each DDP rank binds to its
    own GPU and nothing steps on rank 0's device by accident.
    """
    if torch.cuda.is_available() and config_device in ("auto", "cuda"):
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return resolve_device(config_device)


def run_ddp(
    config: dict[str, Any],
    *,
    data_path: str,
    sp_model_path: str,
    budget_seconds: float,
    output_json: str | None,
    world_size_override: int | None,
) -> dict[str, Any]:
    """Run one Exp 18 training condition under DDP."""
    rank, world_size, local_rank = _init_distributed(world_size_override)
    is_rank0 = rank == 0
    ddp_active = world_size > 1

    device = _pick_device(local_rank, config.get("device", "auto"))
    param_dtype = resolve_param_dtype(config.get("dtype", "bf16"), device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Identical model-init seed across ranks so DDP broadcasts a consistent
    # starting point. The data RNG inside train_chaoscontrol_for_budget adds
    # rank to diverge sample streams.
    seed = int(config.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Per-rank compile warmup — each rank resolves its own backend before the
    # first training step so no rank lags into the collective with a cold
    # compile state.
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

    train_starts = build_lm_starts(int(train_tokens.numel()), seq_len, stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), seq_len, stride)
    eval_starts = choose_eval_starts(
        val_starts, batch_size=batch_size, eval_batches=eval_batches, seed=seed
    )

    model = build_model(config, device, param_dtype)
    model_params = sum(p.numel() for p in model.parameters())

    if is_rank0:
        print(
            f"[rank {rank}/{world_size}] model dim={config['model_dim']} "
            f"layers={config['num_layers']} params={model_params:,}",
            flush=True,
        )

    train_result = train_chaoscontrol_for_budget(
        model,
        train_tokens=train_tokens,
        train_starts=train_starts,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        param_dtype=param_dtype,
        budget_seconds=budget_seconds,
        base_lr=config.get("base_lr", 2e-3),
        weight_decay=config.get("weight_decay", 1e-2),
        grad_clip_norm=config.get("grad_clip_norm", 1.0),
        seed=seed,
        crit_reg_alpha=config.get("crit_reg_alpha", 0.01),
        crit_reg_beta=config.get("crit_reg_beta", 0.001),
        crit_target_coupling=config.get("crit_target_coupling", 0.92),
        optimizer=config.get("optimizer", "adamw"),
        # Pass resolved context explicitly so train_chaoscontrol_for_budget
        # does not re-resolve from env vars (redundant but defensive — avoids
        # surprises if callers monkey-patch the env between these lines).
        rank=rank,
        world_size=world_size,
    )

    # Barrier before eval so all ranks finish the final training step and
    # release any collective waits cleanly before rank 0 runs eval alone.
    if ddp_active:
        dist.barrier()

    # Rank-0-only eval on the full val set. For Exp 18 the model is small
    # enough (~13 MB artifact, per the throughput-levers plan) that running
    # eval on one rank while others idle at the barrier is the correct
    # trade-off vs splitting data across ranks and all-reducing loss/bytes.
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
    train_summary = {
        "steps": int(train_result["steps"]),
        "elapsed_s": float(train_result["elapsed_s"]),
        "steps_per_second": float(
            train_result["steps"] / max(train_result["elapsed_s"], 1e-9)
        ),
        "final_loss": float(history_final.get("loss", float("nan"))),
        "peak_vram_mb": float(train_result.get("peak_vram_mb", 0.0)),
        "ddp_rank": int(train_result.get("ddp_rank", rank)),
        "ddp_world_size": int(train_result.get("ddp_world_size", world_size)),
    }

    result = {
        "config": config,
        "params": model_params,
        "train": train_summary,
        "eval": eval_result,
    }

    # Fail closed on numerically poisoned runs. A training step that
    # produced NaN/Inf loss or an eval that returned non-finite bpb
    # indicates the run diverged silently — the model is garbage and
    # its "result" is not a datapoint any summary gate should ingest.
    # Raise on rank 0 so the subprocess exits non-zero and the
    # orchestrator hard-fails the run with its log tail visible.
    import math
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
                "runner_exp18: refusing to write poisoned result JSON — "
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
        description="Exp 18 DDP runner (throughput lever stack)"
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
