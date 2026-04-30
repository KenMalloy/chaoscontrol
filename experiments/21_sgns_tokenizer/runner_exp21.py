#!/usr/bin/env python3
"""DDP entry point for Exp 21 — SSM × modded-NanoGPT × {random, SGNS init}.

Sibling to ``experiments/18_throughput_levers/runner_exp18_ssm.py``. The
only functional differences:

1. Local ``build_model`` that dispatches on ``config['model_type']``
   (``ssm_exp18_t4b`` → bare ``CareStudentLM``, ``transformer_nanogpt_lean``
   → ``NanoGPTLeanLM``). ``runner_exp18_ssm.py`` imports ``build_model``
   from ``runner_exp17`` which is hardcoded to ``CareStudentLM``; threading
   a flag through the frozen runner would touch the reproducibility path.

2. ``_apply_embed_init`` honors ``config['embed_init_path']`` — loads a
   saved ``(vocab_size, d_model)`` tensor and copies into
   ``model.embed.weight`` under ``no_grad``. Runs after ``build_model``
   moved the model to its training device/dtype so the init is cast into
   the right dtype on load.

Everything else — launch convention, DDP init, rank-0 eval, result-JSON
schema — matches ``runner_exp18_ssm``.

Expected launch:
    torchrun --standalone --nproc_per_node=N \\
        experiments/21_sgns_tokenizer/runner_exp21.py \\
        --config <path.yaml> --data-path <fineweb-dir> \\
        --sp-model-path <sp.model> --output-json <result.json>
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
from chaoscontrol.model import CareStudentLM  # noqa: E402
from chaoscontrol.optim.lamb import LAMB  # noqa: E402
from chaoscontrol.optim.muon import Muon  # noqa: E402
from chaoscontrol.train_ssm import (  # noqa: E402
    _reject_unsupported,
    train_ssm_for_budget,
)

from runner_exp17 import (  # noqa: E402
    build_sentencepiece_luts,
    evaluate_bpb_sp,
    load_sp_data,
)


def _ssm_constructor_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the exact CareStudentLM(**kwargs) needed to reconstruct the
    SSM-arm model from a runner training config.

    Single source of truth for the SSM arm: ``build_model`` consumes this
    dict to instantiate the live training model, and ``--output-ckpt``
    serializes the same dict so ``scripts/run_exp20_eval.py::_build_model``
    can do ``CareStudentLM(**cfg)`` and reconstruct the matching shape.

    Keys must match ``CareStudentLM.__init__`` parameter names — not YAML
    config keys (e.g. ``dim`` not ``model_dim``). Carries every kwarg that
    affects parameter shape; defaults from ``CareStudentLM`` for absent
    fields stay implicit and the class supplies them on reload.
    """
    crct_enabled = bool(config.get("crct_enabled", False))
    kwargs = dict(
        vocab_size=int(config["vocab_size"]),
        dim=int(config["model_dim"]),
        num_layers=int(config["num_layers"]),
        ff_mult=int(config.get("ff_mult", 2)),
        a_mode=config.get("a_mode", "diag"),
        a_full_rank=int(config.get("a_full_rank", 8)),
        a_full_gamma=float(config.get("a_full_gamma", 0.05)),
        ssm_delta_rank=int(config.get("ssm_delta_rank", 0)),
        outer_model_dim=(
            int(config.get("outer_model_dim", 64)) if crct_enabled else 0
        ),
        outer_model_type=str(
            config.get("outer_model_type", "multislot" if crct_enabled else "single")
        ),
        outer_max_slots=int(config.get("outer_max_slots", 4096 if crct_enabled else 64)),
        outer_compress_ratio=int(config.get("outer_compress_ratio", 2)),
        buffer_mode=str(
            config.get("buffer_mode", "append_only" if crct_enabled else "legacy")
        ),
        retrieval_mode=str(config.get("retrieval_mode", "softmax_all")),
        retrieval_k=int(config.get("retrieval_k", 16 if crct_enabled else 8)),
        wernicke_enabled=False,
        local_attn_window=int(config.get("local_attn_window", 0)),
        local_attn_heads=int(config.get("local_attn_heads", 1)),
        local_attn_dim=int(config.get("local_attn_dim", 64)),
        local_attn_topk=int(config.get("local_attn_topk", 0)),
        local_attn_topk_random=bool(config.get("local_attn_topk_random", False)),
        activation_checkpoint=bool(config.get("activation_checkpoint", False)),
    )
    if crct_enabled and kwargs["outer_model_dim"] <= 0:
        raise ValueError(
            "crct_enabled=True requires outer_model_dim > 0; otherwise "
            "memory_mode='force_on' is identical to memory_mode='off'."
        )
    if crct_enabled and kwargs["buffer_mode"] != "append_only":
        raise ValueError(
            "crct_enabled=True requires buffer_mode='append_only' so the "
            "fast runner can populate memory from encode() hidden states."
        )
    if crct_enabled and kwargs["outer_model_type"] != "multislot":
        raise ValueError(
            "crct_enabled=True requires outer_model_type='multislot' for "
            "append-only memory writes."
        )
    return kwargs


def _transformer_constructor_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the exact NanoGPTLeanLM(**kwargs) needed to reconstruct the
    transformer-arm model from a runner training config.

    Same single-source-of-truth contract as ``_ssm_constructor_kwargs``.
    Note: ``max_seq_len`` is intentionally omitted — ``build_model`` does
    not pass it, so reload uses ``NanoGPTLeanLM``'s 2048 default which is
    what training also used.
    """
    model_dim = int(config["model_dim"])
    n_head = int(config.get("n_head", max(1, model_dim // 64)))
    return dict(
        vocab_size=int(config["vocab_size"]),
        d_model=model_dim,
        n_head=n_head,
        n_layer=int(config["num_layers"]),
        ffn_mult=int(config.get("ff_mult", 4)),
        activation_checkpoint=bool(config.get("activation_checkpoint", False)),
    )


def build_model(
    config: dict[str, Any],
    device: torch.device,
    param_dtype: torch.dtype,
) -> torch.nn.Module:
    """Build the model for an Exp 21 training condition.

    Dispatches on ``config['model_type']``:
      - ``'transformer_nanogpt_lean'`` → ``NanoGPTLeanLM`` (cells A, B, Phase 0)
      - otherwise → ``CareStudentLM`` bare-SSM (cells C, D, controls)

    The bare-SSM branch matches the field reads in
    ``runner_exp17.build_model`` so SSM cells C/D are bit-identical to
    Test 4b at the model-construction level.
    """
    model_type = str(config.get("model_type", "")).strip()
    if model_type == "transformer_nanogpt_lean":
        from chaoscontrol.baselines_nanogpt_lean import NanoGPTLeanLM

        model: torch.nn.Module = NanoGPTLeanLM(
            **_transformer_constructor_kwargs(config)
        )
    else:
        model = CareStudentLM(**_ssm_constructor_kwargs(config))
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(dtype=param_dtype)
    return model


def _apply_embed_init(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    """Load ``config['embed_init_path']`` into ``model.embed.weight``.

    No-op if ``embed_init_path`` is absent or None. Validates shape to
    catch vocab-size mismatches at runner entry (before budget-seconds
    of GPU time is spent). Casts into the model's existing dtype so a
    bf16 model arm sees a bf16 init even when the saved tensor is fp32.
    """
    path = config.get("embed_init_path")
    if not path:
        return
    weights = torch.load(str(path), map_location=device)
    expected = tuple(model.embed.weight.shape)
    got = tuple(weights.shape)
    assert got == expected, (
        f"embed_init_path shape mismatch: got {got}, expected {expected}"
    )
    weights = weights.to(
        device=model.embed.weight.device, dtype=model.embed.weight.dtype
    )
    with torch.no_grad():
        model.embed.weight.data.copy_(weights)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    return int(val)


def _init_distributed(world_size_override: int | None) -> tuple[int, int, int]:
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
    if world_size <= 1:
        return list(train_starts)
    sharded = [s for i, s in enumerate(train_starts) if i % world_size == rank]
    if len(sharded) < 1:
        raise RuntimeError(
            f"rank {rank} has no train_starts after stride-sharding "
            f"({len(train_starts)} total starts across world_size={world_size})."
        )
    return sharded


def _save_output_ckpt(
    output_ckpt: str,
    model: torch.nn.Module,
    config: dict[str, Any],
    episodic_cache: Any | None = None,
    online_eval_state: dict[str, Any] | None = None,
) -> None:
    """Write a downstream-loadable checkpoint to ``output_ckpt``.

    Format is ``{"model": state_dict, "config": <constructor_kwargs>}`` by
    default. Cache-aware runs may additionally carry ``"episodic_cache"`` as
    an ``EpisodicCache.to_dict()`` payload consumed by
    ``scripts/run_exp20_eval.py``.

    Both arms are saved with the right kwargs dict, but the consumer
    currently hardcodes ``CareStudentLM(**cfg)`` and will only successfully
    reconstruct the SSM arm. Transformer-arm checkpoints save correctly but
    require a future ``_build_model`` extension to load — left intentional
    since the submission target is SSM.

    Saves model weights on CPU so the artifact is portable across
    GPU/CPU and bf16/fp32 reload contexts.
    """
    model_type = str(config.get("model_type", "")).strip()
    if model_type == "transformer_nanogpt_lean":
        ctor_kwargs = _transformer_constructor_kwargs(config)
    else:
        ctor_kwargs = _ssm_constructor_kwargs(config)
    cpu_state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    blob: dict[str, Any] = {"model": cpu_state, "config": ctor_kwargs}
    if episodic_cache is not None:
        if isinstance(episodic_cache, dict):
            blob["episodic_cache"] = episodic_cache
        else:
            to_dict = getattr(episodic_cache, "to_dict", None)
            if not callable(to_dict):
                raise TypeError(
                    "episodic_cache must be a dict or expose to_dict() when "
                    "passed to _save_output_ckpt"
                )
            blob["episodic_cache"] = to_dict()
    if isinstance(online_eval_state, dict) and online_eval_state:
        blob["online_eval_state"] = online_eval_state
    out_path = Path(output_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(blob, tmp_path)
    tmp_path.rename(out_path)


def run_ddp(
    config: dict[str, Any],
    *,
    data_path: str,
    sp_model_path: str,
    budget_seconds: float,
    output_json: str | None,
    world_size_override: int | None,
    output_ckpt: str | None = None,
) -> dict[str, Any]:
    """Run one Exp 21 training condition via the lean train_ssm path.

    Mirrors ``runner_exp18_ssm.run_ddp`` — same launch convention, same
    rank-0 eval, same result-JSON schema — with the local dispatching
    ``build_model`` and ``_apply_embed_init`` hook swapped in.
    """
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
    train_starts = _shard_train_starts(
        train_starts_all, rank=rank, world_size=world_size,
    )

    model = build_model(config, device, param_dtype)
    _apply_embed_init(model, config, device)
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
    elif precision == "fp8_fused":
        from chaoscontrol.precision import maybe_promote_linears_to_fused_fp8
        n_promoted = maybe_promote_linears_to_fused_fp8(model, enabled=True)
        if is_rank0:
            print(
                f"[rank 0] promoted {n_promoted} nn.Linear -> FusedFP8Linear for fp8_fused",
                flush=True,
            )
    model_params = sum(p.numel() for p in model.parameters())

    if is_rank0:
        print(
            f"[rank {rank}/{world_size}] model_type={config.get('model_type','ssm')} "
            f"dim={config['model_dim']} layers={config['num_layers']} "
            f"params={model_params:,}",
            flush=True,
        )

    optimizer_name = str(config.get("optimizer", "")).strip()
    if not optimizer_name:
        raise ValueError(
            "runner_exp21 requires config['optimizer'] to be set explicitly "
            "(one of {'adamw', 'muon', 'lamb'})."
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

    if ddp_active:
        dist.barrier()

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
        "ddp_rank": int(train_result.get("rank", rank)),
        "ddp_world_size": int(train_result.get("world_size", world_size)),
    }

    result = {
        "config": config,
        "params": model_params,
        "train": train_summary,
        "eval": eval_result,
    }

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
        # Default: poisoned results are blocking — runner refuses to write a
        # NaN/Inf JSON that could silently contaminate downstream summaries.
        # Controls that study divergence on purpose (e.g. zero-init floor)
        # set ``allow_nonfinite: true`` in the config so the run is
        # preserved as a datapoint and a ``nonfinite`` flag is recorded.
        if violations:
            if bool(config.get("allow_nonfinite", False)):
                result["nonfinite"] = {
                    "flag": True,
                    "violations": violations,
                }
                print(
                    "[rank 0] allow_nonfinite=true — preserving divergent "
                    f"run as datapoint: {'; '.join(violations)}",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    "runner_exp21: refusing to write poisoned result JSON — "
                    + "; ".join(violations)
                )

    if is_rank0 and output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(result, indent=2, default=str))
        tmp_path.rename(out_path)

    # Optional checkpoint save AFTER the JSON write — a save failure here
    # does not lose the result of an already-completed training run.
    if is_rank0 and output_ckpt:
        _save_output_ckpt(output_ckpt, model, config)

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
        description="Exp 21 runner — SSM × NanoGPT-lean × {random, SGNS init}"
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--output-ckpt",
        default=None,
        help=(
            "Optional path to save a checkpoint loadable by "
            "scripts/run_exp20_eval.py::_build_model. Format: "
            '{"model": state_dict, "config": constructor_kwargs}.'
        ),
    )
    parser.add_argument("--world-size", type=int, default=None)
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    run_ddp(
        config,
        data_path=args.data_path,
        sp_model_path=args.sp_model_path,
        budget_seconds=args.budget,
        output_json=args.output_json,
        world_size_override=args.world_size,
        output_ckpt=args.output_ckpt,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
