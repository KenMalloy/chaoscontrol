#!/usr/bin/env python3
"""Single-run engine for Experiment 17 Phase A.

Trains a fast SP8192 SSM with an optional local-attention sidecar in the
top block, then evaluates competition-correct bpb.
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
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.data import (
    batch_from_starts,
    build_lm_starts,
    choose_eval_starts,
    load_fineweb_tokens,
    maybe_autocast,
    resolve_device,
    resolve_param_dtype,
)
from chaoscontrol.core import verify_diag_recurrence
from chaoscontrol.evaluation import compute_bpb
from chaoscontrol.model import ChaosSSMBlock, ChaosSSMHybridBlock, ChaosStudentLM
from chaoscontrol.training import train_chaoscontrol_for_budget

MIN_PYTHON = (3, 10)


def _resolve_rank_world() -> tuple[int, int]:
    """Resolve (rank, world_size) from torch.distributed or env vars.

    Returns (0, 1) when distributed is not set up. This keeps the
    single-device Exp 17 path bit-identical: no branches taken, no
    new env lookups fail, nothing changes.
    """
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    env_rank = os.environ.get("RANK")
    env_world = os.environ.get("WORLD_SIZE")
    if env_rank is not None and env_world is not None:
        return int(env_rank), int(env_world)
    return 0, 1


def resolve_visible_cuda_devices(env: dict[str, str] | None = None) -> list[str]:
    env_map = os.environ if env is None else env
    mask = env_map.get("CUDA_VISIBLE_DEVICES", "").strip()
    if mask:
        return [piece.strip() for piece in mask.split(",") if piece.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def validate_gpu_concurrency(num_gpus: int, env: dict[str, str] | None = None) -> list[str]:
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    visible = resolve_visible_cuda_devices(env)
    if not visible:
        raise RuntimeError(
            "No visible CUDA devices. Check the pod allocation, CUDA_VISIBLE_DEVICES, and driver/runtime setup."
        )
    if num_gpus > len(visible):
        raise RuntimeError(
            f"Requested num_gpus={num_gpus}, but only {len(visible)} CUDA slots are visible "
            f"({','.join(visible)})."
        )
    return visible


def build_child_env(
    *,
    gpu_slot: int | None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    if gpu_slot is None:
        return env
    visible = resolve_visible_cuda_devices(env)
    if gpu_slot < 0 or gpu_slot >= len(visible):
        raise RuntimeError(
            f"GPU slot {gpu_slot} is out of range for visible CUDA devices {visible}."
        )
    env["CUDA_VISIBLE_DEVICES"] = visible[gpu_slot]
    return env


def assert_runtime_compatibility(
    *,
    device: torch.device,
    sp_model_path: str,
) -> dict[str, Any]:
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            f"Experiment 17 requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, "
            f"found {sys.version_info.major}.{sys.version_info.minor}."
        )
    info: dict[str, Any] = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "diag_recurrence": verify_diag_recurrence(device),
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")
        visible = resolve_visible_cuda_devices()
        info["visible_cuda_devices"] = visible
        info["cuda_device_count"] = torch.cuda.device_count()
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        info["bf16_supported"] = bf16_supported
        if not bf16_supported:
            raise RuntimeError(
                "CUDA is visible, but torch reports bf16 is unsupported. "
                "Check the pod image, PyTorch build, and driver/runtime compatibility."
            )

    try:
        import sentencepiece as spm
    except Exception as exc:
        raise RuntimeError("sentencepiece is required for Experiment 17, but import failed.") from exc
    sp = spm.SentencePieceProcessor()
    loaded = sp.Load(sp_model_path)
    if not loaded:
        raise RuntimeError(f"Failed to load SentencePiece model: {sp_model_path}")
    info["sentencepiece_model"] = sp_model_path
    return info


def build_sentencepiece_luts(
    sp,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate the competition byte LUT logic used in Exp 15/16."""
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_sp_data(data_dir: str, vocab_size: int = 8192) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load pre-tokenized SP shards and clamp header contamination."""
    train_tokens, val_tokens = load_fineweb_tokens(data_dir)
    train_tokens = train_tokens.clamp(0, vocab_size - 1)
    val_tokens = val_tokens.clamp(0, vocab_size - 1)
    test_tokens = train_tokens[:0]
    rank, _ = _resolve_rank_world()
    if rank == 0:
        print(f"  SP data: train={train_tokens.numel():,} val={val_tokens.numel():,} tokens", flush=True)
    return train_tokens, val_tokens, test_tokens


def evaluate_bpb_sp(
    model: Any,
    *,
    tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> dict[str, float]:
    """Competition-correct SP bpb computation."""
    was_training = model.training
    model.eval()
    total_ce_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    vocab_size = model.vocab_size

    with torch.no_grad():
        for idx in range(0, len(eval_starts), batch_size):
            batch_starts = eval_starts[idx : idx + batch_size]
            inputs, targets = batch_from_starts(tokens, batch_starts, seq_len, device)
            autocast_dtype = next(model.parameters()).dtype if device.type == "cuda" else torch.float32
            with maybe_autocast(device, autocast_dtype):
                out = model(inputs)
                logits = out["logits"]
            batch_ce = float(
                F.cross_entropy(
                    logits.float().reshape(-1, vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_ce_nats += batch_ce
            total_tokens += int(targets.numel())

            prev_ids = inputs.reshape(-1)
            tgt_ids = targets.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            total_bytes += int(token_bytes.to(torch.int64).sum().item())

    if was_training:
        model.train()

    return {
        "loss": float(total_ce_nats / max(total_tokens, 1)),
        "bpb": compute_bpb(total_ce_nats, total_bytes),
        "tokens": float(total_tokens),
        "total_ce_nats": total_ce_nats,
        "total_scored_bytes": total_bytes,
    }


def compute_gate_stats(
    model: Any,
    *,
    tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> list[dict[str, float]]:
    """Compute gate statistics for each hybrid block.

    Records final gate_bias and mean sigmoid(gate_proj(x_ssm) + gate_bias)
    on a single eval batch. Used to distinguish "gate never opened"
    (gate stays near sigmoid(-4) = 0.018) from "attention learned to
    activate" cases when interpreting top-k results.
    """
    hybrid_layers = [
        (i, layer)
        for i, layer in enumerate(model.layers)
        if isinstance(layer, ChaosSSMHybridBlock)
    ]
    if not hybrid_layers:
        return []

    # Grab first eval batch for a representative gate sample
    batch_starts = eval_starts[:batch_size]
    if not batch_starts:
        return [
            {
                "layer_idx": i,
                "final_gate_bias": float(layer.gate_bias.item()),
                "mean_gate_value": float("nan"),
            }
            for i, layer in hybrid_layers
        ]
    inputs, _ = batch_from_starts(tokens, batch_starts, seq_len, device)

    stats: list[dict[str, float]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        # Run forward through early layers to compute x_ssm at each hybrid block
        x = model.embed(inputs)
        for i, layer in enumerate(model.layers):
            if isinstance(layer, ChaosSSMHybridBlock):
                normed = layer.input_norm(x)
                ssm_out = layer.core.forward(normed)
                x_ssm = x + ssm_out
                gate = torch.sigmoid(
                    layer.gate_proj(x_ssm) + layer.gate_bias
                )
                stats.append({
                    "layer_idx": i,
                    "final_gate_bias": float(layer.gate_bias.item()),
                    "mean_gate_value": float(gate.mean().item()),
                    "max_gate_value": float(gate.max().item()),
                })
                # Continue propagation through this layer for downstream hybrid blocks
                x = layer(x)
            else:
                x = layer(x)
    if was_training:
        model.train()
    return stats


def build_model(config: dict[str, Any], device: torch.device, param_dtype: torch.dtype) -> ChaosStudentLM:
    """Build bare or hybrid fast SP-SSM for Exp 17."""
    model = ChaosStudentLM(
        vocab_size=config["vocab_size"],
        dim=config["model_dim"],
        num_layers=config["num_layers"],
        ff_mult=config.get("ff_mult", 2),
        a_mode=config.get("a_mode", "diag"),
        a_full_rank=config.get("a_full_rank", 8),
        a_full_gamma=config.get("a_full_gamma", 0.05),
        outer_model_dim=0,
        wernicke_enabled=False,
        local_attn_window=int(config.get("local_attn_window", 0)),
        local_attn_heads=int(config.get("local_attn_heads", 1)),
        local_attn_dim=int(config.get("local_attn_dim", 64)),
        local_attn_topk=int(config.get("local_attn_topk", 0)),
        local_attn_topk_random=bool(config.get("local_attn_topk_random", False)),
        activation_checkpoint=bool(config.get("activation_checkpoint", False)),
    )
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(dtype=param_dtype)
    return model


def summarize_train_result(train_result: dict[str, Any]) -> dict[str, float]:
    """Keep per-run JSON compact; omit long loss histories."""
    history = train_result.get("history", [])
    final_loss = float(history[-1]["loss"]) if history else float("nan")
    elapsed_s = float(train_result["elapsed_s"])
    steps = int(train_result["steps"])
    return {
        "steps": steps,
        "elapsed_s": elapsed_s,
        "steps_per_second": float(steps / max(elapsed_s, 1e-9)),
        "final_loss": final_loss,
        "peak_vram_mb": float(train_result.get("peak_vram_mb", 0.0)),
    }


def run_single(
    config: dict[str, Any],
    *,
    data_path: str,
    budget_seconds: float,
    sp_model_path: str,
    output_json: str | None = None,
) -> dict[str, Any]:
    """Run one Exp 17 train/eval condition.

    DDP-safe: when this script is launched via torchrun with world_size > 1
    (or with explicit RANK/WORLD_SIZE env vars), the runner will apply
    rank-aware seeding, guard prints and file writes to rank 0, and insert
    distributed barriers around the eval phase. In the default single-device
    path (no distributed env vars, no init_process_group call) the behavior is
    bit-identical to the pre-DDP version — rank=0, world_size=1, no barriers,
    no new branches taken.
    """
    rank, world_size = _resolve_rank_world()
    is_rank0 = rank == 0
    ddp_active = world_size > 1

    device = resolve_device(config.get("device", "auto"))
    param_dtype = resolve_param_dtype(config.get("dtype", "bf16"), device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    runtime = assert_runtime_compatibility(device=device, sp_model_path=sp_model_path)

    # Model-init seeds must match across ranks so DDP broadcasts a consistent
    # initial state. The data loader RNG inside train_chaoscontrol_for_budget
    # picks up seed + rank itself; we only need the process-level init seed
    # (torch/numpy/random) to be identical across ranks.
    seed = int(config.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Per-rank compile warmup — must run on every rank before the first
    # training step to avoid torch.compile stagger (one rank compiling while
    # others start the collective and block). verify_diag_recurrence is
    # idempotent and fast; calling it here costs nothing but guarantees the
    # chunked-scan backend is resolved before DDP initialization effects.
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
    eval_starts = choose_eval_starts(val_starts, batch_size=batch_size, eval_batches=eval_batches, seed=seed)

    model = build_model(config, device, param_dtype)
    model_params = sum(p.numel() for p in model.parameters())
    artifact_bytes = model.artifact_bytes()
    if is_rank0:
        print(
            f"Model: dim={config['model_dim']} | layers={config['num_layers']} | "
            f"window={config.get('local_attn_window', 0)} | params={model_params:,} | "
            f"artifact={artifact_bytes:,} bytes ({artifact_bytes / 1e6:.1f} MB)",
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
    )
    train_summary = summarize_train_result(train_result)

    # Eval is identical across ranks (data-parallel eval on the same val_starts
    # would be a waste — rank 0 runs the full eval while other ranks wait at
    # the barrier). Wrap the eval phase in barriers so stragglers don't
    # overlap the collective state when we return.
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
        gate_stats = compute_gate_stats(
            model,
            tokens=val_tokens,
            eval_starts=eval_starts,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
    else:
        eval_result = {}
        gate_stats = []

    if ddp_active:
        dist.barrier()

    result = {
        "config": config,
        "params": model_params,
        "artifact_bytes": artifact_bytes,
        "train": train_summary,
        "eval": eval_result,
        "runtime": runtime,
        "gate_stats": gate_stats,
        "model_shape": {
            "hybrid_enabled": bool(config.get("local_attn_window", 0) > 0),
            "num_hybrid_layers": int(sum(isinstance(layer, ChaosSSMHybridBlock) for layer in model.layers)),
            "num_pure_ssm_layers": int(sum(isinstance(layer, ChaosSSMBlock) for layer in model.layers)),
        },
    }

    if is_rank0 and output_json:
        out_path = Path(output_json)
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(result, indent=2, default=str))
        tmp_path.rename(out_path)

    if is_rank0:
        bpb_display = eval_result.get("bpb", float("nan"))
        print(
            f"Done: bpb={bpb_display:.4f} | steps={train_summary['steps']} | "
            f"steps/s={train_summary['steps_per_second']:.2f} | peak_vram={train_summary['peak_vram_mb']:.1f} MB",
            flush=True,
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 17 runner: local attention sidecar on fast SP-SSM")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.preflight_only:
        device = resolve_device(config.get("device", "auto"))
        info = assert_runtime_compatibility(device=device, sp_model_path=args.sp_model_path)
        print(json.dumps(info, indent=2))
    else:
        run_single(
            config,
            data_path=args.data_path,
            budget_seconds=args.budget,
            sp_model_path=args.sp_model_path,
            output_json=args.output_json,
        )
