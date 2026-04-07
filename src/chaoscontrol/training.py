"""Training loop, matrix runner, and CLI for ChaosControl experiments."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from chaoscontrol.config import ChaosControlConfig
from chaoscontrol.core import criticality_loss
from chaoscontrol.data import batch_from_starts, maybe_autocast, maybe_sync_cuda
from chaoscontrol.metabolic import metabolic_fork
from chaoscontrol.memory import MultiSlotOuterModel


def train_chaoscontrol_for_budget(
    model: Any,
    *,
    train_tokens: torch.Tensor,
    train_starts: list[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
    param_dtype: torch.dtype,
    budget_seconds: float,
    base_lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    seed: int,
    crit_reg_alpha: float = 0.01,
    crit_reg_beta: float = 0.001,
    crit_target_coupling: float = 0.88,
    metabolic_gate: bool = False,
    metabolic_k: int = 4,
    metabolic_threshold: float = 0.1,
    metabolic_threshold_mode: str = "fixed",
    metabolic_score: str = "memory_consistency",
    metabolic_noise_std: float = 0.01,
) -> dict[str, Any]:
    """Train ChaosStudentLM for a time budget with optional criticality regularization."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()
    steps = 0
    use_crit = crit_reg_alpha > 0 or crit_reg_beta > 0
    vocab_size = model.vocab_size
    loss_ema = 2.0  # for metabolic gate surprise calculation
    fork_count = 0
    current_threshold = metabolic_threshold  # adaptive mode will adjust this
    last_forked = False
    pre_fork_loss: float = 0.0  # loss BEFORE the fork, for adaptive comparison
    model.train()

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= budget_seconds and steps > 0:
            break

        batch_starts = [train_starts[rng.randrange(len(train_starts))] for _ in range(batch_size)]
        inputs, targets = batch_from_starts(train_tokens, batch_starts, seq_len, device)

        # Check metabolic gate: should we fork?
        surprise_ratio = abs(loss_ema - (history[-1]["loss"] if history else loss_ema)) / max(loss_ema, 1e-6) if history else 0.0
        use_fork = metabolic_gate and surprise_ratio > current_threshold

        # Adaptive threshold: compare current loss against the loss BEFORE
        # the last fork. If loss improved since then, fork helped -- lower
        # threshold. If not, raise it.
        if metabolic_threshold_mode == "adaptive" and last_forked and history:
            fork_helped = ce_val_prev < pre_fork_loss if steps > 0 else False  # noqa: F821
            if fork_helped:
                current_threshold = max(0.01, current_threshold * 0.95)
            else:
                current_threshold = min(1.0, current_threshold * 1.05)

        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(device, param_dtype):
            if use_fork:
                # Expensive path: generation + selection
                out = metabolic_fork(
                    model, inputs,
                    k=metabolic_k,
                    noise_std=metabolic_noise_std,
                    score_mode=metabolic_score,
                )
                fork_count += 1
                # Forked path doesn't produce jacobian_stats, so run a cheap
                # stats-only pass on the winning output for consistent L_crit
                if use_crit:
                    with torch.no_grad():
                        stats_out = model(inputs, return_jacobian_stats=True)
                        if "jacobian_stats" in stats_out:
                            out["jacobian_stats"] = stats_out["jacobian_stats"]
            else:
                # Cheap path: single deterministic pass
                out = model(inputs, return_jacobian_stats=use_crit)
            ce_loss = F.cross_entropy(out["logits"].reshape(-1, vocab_size), targets.reshape(-1))
            loss = ce_loss
            if use_crit and "jacobian_stats" in out:
                loss = loss + criticality_loss(
                    out["jacobian_stats"], alpha=crit_reg_alpha, beta=crit_reg_beta,
                    target_log_sv=math.log(crit_target_coupling),
                )
            # Wernicke balance loss
            if "balance_loss" in out and model.wernicke is not None:
                loss = loss + model.wernicke_balance_weight * out["balance_loss"]

        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        ce_val_prev = history[-1]["loss"] if history else loss_ema  # noqa: F841
        ce_val = float(ce_loss.detach().cpu())
        loss_ema = 0.99 * loss_ema + 0.01 * ce_val
        last_forked = use_fork
        if use_fork:
            pre_fork_loss = ce_val_prev  # save loss BEFORE this fork for next step's comparison
        step_record: dict[str, Any] = {"step": float(steps), "loss": ce_val, "forked": use_fork}

        # Outer model consolidation (after optimizer step)
        if model.outer_model is not None:
            with torch.no_grad():
                per_sample_ce = F.cross_entropy(
                    out["logits"].detach().reshape(-1, vocab_size),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape(inputs.size(0), -1).mean(dim=1)  # (batch,)
            hidden = out["hidden"][:, -1, :].detach()  # (batch, dim)
            # Determine dominant bucket_id from Wernicke layer for typed writes
            dominant_bucket: int | None = None
            if "bucket_ids" in out:
                bids = out["bucket_ids"].detach()  # (batch, seq)
                flat_ids = bids.reshape(-1)
                dominant_bucket = int(flat_ids.mode().values.item())
            surprise = model.outer_model.consolidation_step(
                hidden, current_loss=ce_val, per_sample_weights=per_sample_ce,
                bucket_id=dominant_bucket,
            )
            step_record["surprise"] = float(surprise)

        history.append(step_record)
        steps += 1

    maybe_sync_cuda(device)
    return {
        "steps": int(steps),
        "history": history,
        "elapsed_s": float(time.perf_counter() - start_time),
        "fork_count": fork_count,
    }


def build_chaoscontrol_matrix() -> list[dict[str, Any]]:
    """Build the 24-cell test matrix."""
    a_modes = ["diag", "paired", "full"]
    rows = [
        {"rich_b_mode": "none", "outer_model_dim": 0, "label": "base"},
        {"rich_b_mode": "nn", "outer_model_dim": 0, "label": "+nn"},
        {"rich_b_mode": "hub", "outer_model_dim": 0, "label": "+hub"},
        {"rich_b_mode": "assembly", "outer_model_dim": 0, "label": "+assembly"},
        {"rich_b_mode": "hybrid", "outer_model_dim": 0, "label": "+hybrid"},
        {"rich_b_mode": "none", "outer_model_dim": 64, "label": "+outer"},
        {"rich_b_mode": "nn", "outer_model_dim": 64, "label": "+nn+outer"},
        {"rich_b_mode": "hub", "outer_model_dim": 64, "label": "+hub+outer"},
    ]
    cells = []
    for a_mode in a_modes:
        for row in rows:
            cell = dict(row)
            cell["a_mode"] = a_mode
            cells.append(cell)
    return cells


def run_chaoscontrol_matrix(
    cfg: ChaosControlConfig,
    *,
    device: torch.device,
    param_dtype: torch.dtype,
    train_tokens: torch.Tensor,
    train_starts: list[int],
    eval_tokens: torch.Tensor,
    eval_starts: list[int],
) -> list[dict[str, Any]]:
    """Iterate matrix cells, build model per cell, train, eval, collect results.

    Prints progress per cell and checkpoints results to output_json (if set)
    after every cell, so partial results survive pod crashes.
    """
    from chaoscontrol.model import ChaosStudentLM
    from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb

    cells = build_chaoscontrol_matrix()
    results: list[dict[str, Any]] = []
    total = len(cells)

    for i, cell in enumerate(cells):
        model = ChaosStudentLM(
            vocab_size=cfg.vocab_size,
            dim=cfg.model_dim,
            num_layers=cfg.num_layers,
            ff_mult=cfg.ff_mult,
            a_mode=cell["a_mode"],
            a_full_rank=cfg.a_full_rank,
            a_full_gamma=cfg.a_full_gamma,
            rich_b_mode=cell["rich_b_mode"],
            rich_b_bottleneck=cfg.rich_b_bottleneck,
            rich_b_num_subnets=cfg.rich_b_num_subnets,
            rich_b_settling_steps=cfg.rich_b_settling_steps,
            outer_model_dim=cell["outer_model_dim"],
            consolidation_mode=cfg.consolidation_mode,
            consolidation_ema_decay=cfg.consolidation_ema_decay,
            consolidation_trigger=cfg.consolidation_trigger,
            consolidation_window=cfg.consolidation_window,
            outer_model_type=cfg.outer_model_type,
            outer_max_slots=cfg.outer_max_slots,
            outer_compress_ratio=cfg.outer_compress_ratio,
            wernicke_enabled=cfg.wernicke_enabled,
            wernicke_k_max=cfg.wernicke_k_max,
            wernicke_window=cfg.wernicke_window,
            wernicke_router=cfg.wernicke_router,
            wernicke_balance_weight=cfg.wernicke_balance_weight,
        ).to(device)

        train_result = train_chaoscontrol_for_budget(
            model,
            train_tokens=train_tokens,
            train_starts=train_starts,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            device=device,
            param_dtype=param_dtype,
            budget_seconds=cfg.budget_seconds,
            base_lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
            grad_clip_norm=cfg.grad_clip_norm,
            seed=cfg.seed,
            crit_reg_alpha=cfg.crit_reg_alpha,
            crit_reg_beta=cfg.crit_reg_beta,
            crit_target_coupling=cfg.crit_target_coupling,
            metabolic_gate=cfg.metabolic_gate,
            metabolic_k=cfg.metabolic_k,
            metabolic_threshold=cfg.metabolic_threshold,
            metabolic_threshold_mode=cfg.metabolic_threshold_mode,
            metabolic_score=cfg.metabolic_score,
            metabolic_noise_std=cfg.metabolic_noise_std,
        )

        eval_result = evaluate_chaoscontrol_bpb(
            model,
            tokens=eval_tokens,
            eval_starts=eval_starts,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            device=device,
        )

        result = {
            **cell,
            "params": sum(p.numel() for p in model.parameters()),
            "artifact_bytes": model.artifact_bytes(),
            "train_steps": train_result["steps"],
            "train_elapsed_s": train_result["elapsed_s"],
            **eval_result,
        }
        results.append(result)
        print(
            f"[{i+1}/{total}] {cell['a_mode']:>6} {cell['label']:<20} "
            f"bpb={eval_result['bpb']:.4f}  steps={train_result['steps']}  "
            f"elapsed={train_result['elapsed_s']:.1f}s"
        )

        # Checkpoint after every cell so partial results survive crashes
        if cfg.output_json:
            _checkpoint_path = Path(cfg.output_json)
            _checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_checkpoint_path, "w") as _f:
                json.dump(results, _f, indent=2, default=str)

    return results


def parse_chaoscontrol_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ChaosControl SSM-native experiment")
    # Data / device
    p.add_argument("--enwik8-path", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output-json", default=None)
    # Model sizing
    p.add_argument("--model-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batches", type=int, default=8)
    # A parameterization
    p.add_argument("--a-mode", default="diag", choices=["diag", "paired", "full"])
    p.add_argument("--a-full-rank", type=int, default=8)
    p.add_argument("--a-full-gamma", type=float, default=0.05)
    # Rich B
    p.add_argument("--rich-b-mode", default="none", choices=["none", "nn", "hub", "assembly", "hybrid"])
    p.add_argument("--rich-b-bottleneck", type=int, default=32)
    p.add_argument("--rich-b-num-subnets", type=int, default=4)
    p.add_argument("--rich-b-settling-steps", type=int, default=2)
    # Outer model
    p.add_argument("--outer-model-dim", type=int, default=0)
    p.add_argument("--consolidation-mode", default="symmetric", choices=["symmetric", "pain_biased", "learned"])
    p.add_argument("--consolidation-ema-decay", type=float, default=0.99)
    p.add_argument("--consolidation-trigger", default="immediate", choices=["immediate", "resolution", "windowed"])
    p.add_argument("--consolidation-window", type=int, default=8)
    p.add_argument("--outer-model-type", default="single", choices=["single", "multislot"])
    p.add_argument("--outer-max-slots", type=int, default=64)
    p.add_argument("--outer-compress-ratio", type=int, default=2)
    # Wernicke layer
    p.add_argument("--wernicke-enabled", action="store_true", default=False)
    p.add_argument("--wernicke-k-max", type=int, default=16)
    p.add_argument("--wernicke-window", type=int, default=8)
    p.add_argument("--wernicke-router", default="vq", choices=["vq", "moe"])
    p.add_argument("--wernicke-balance-weight", type=float, default=0.01)
    # Metabolic gate
    p.add_argument("--metabolic-gate", action="store_true", default=False)
    p.add_argument("--metabolic-k", type=int, default=4)
    p.add_argument("--metabolic-threshold", type=float, default=0.1)
    p.add_argument("--metabolic-threshold-mode", default="fixed", choices=["fixed", "adaptive"])
    p.add_argument("--metabolic-score", default="memory_consistency",
                    choices=["memory_consistency", "loss_lookahead", "ensemble_agreement"])
    p.add_argument("--metabolic-noise-std", type=float, default=0.01)
    # Criticality
    p.add_argument("--crit-reg-alpha", type=float, default=0.01)
    p.add_argument("--crit-reg-beta", type=float, default=0.001)
    p.add_argument("--crit-target-coupling", type=float, default=0.88)
    # Training
    p.add_argument("--base-lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--budget-seconds", type=float, default=60.0)
    # Mode
    p.add_argument("--run-matrix", action="store_true")
    return p.parse_args(argv)
