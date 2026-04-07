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
from chaoscontrol.metabolic import metabolic_fork, metabolic_monte_carlo, StructuredProjections
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
    generation_mode: str = "noise",
    metabolic_mode: str = "fork",
    mcts_horizon: int = 8,
    mcts_ucb_c: float = 1.41,
    consolidation_write: str = "last",
    latent_persistence: bool = False,
    cfr_enabled: bool = False,
) -> dict[str, Any]:
    """Train ChaosStudentLM for a time budget with optional criticality regularization."""
    # Set up structured projections if requested (before optimizer so its params are included)
    structured_proj = None
    if generation_mode == "structured":
        model_dim = model.embed.weight.shape[1]
        structured_proj = StructuredProjections(
            dim=model_dim,
            k=metabolic_k,
        ).to(device)

    all_params = list(model.parameters())
    if structured_proj is not None:
        all_params += list(structured_proj.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=base_lr, weight_decay=weight_decay)
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
    ce_val_prev: float = loss_ema  # previous step's loss, for adaptive threshold
    spectral_snapshots: list[dict[str, Any]] = []
    bucket_snapshots: list[dict[str, Any]] = []
    spectral_log_interval = 50

    # CFR regret tracking
    regret_table = None
    if cfr_enabled:
        from chaoscontrol.regret import RegretTable
        k_max = model.wernicke.k_max if getattr(model, "wernicke", None) else 16
        regret_table = RegretTable(n_buckets=k_max, n_actions=metabolic_k)

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
            fork_helped = ce_val_prev < pre_fork_loss if steps > 0 else False
            if fork_helped:
                current_threshold = max(0.01, current_threshold * 0.95)
            else:
                current_threshold = min(1.0, current_threshold * 1.05)

        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(device, param_dtype):
            if use_fork and metabolic_mode == "fork":
                # Pick-best path: generation + selection
                out = metabolic_fork(
                    model, inputs,
                    k=metabolic_k,
                    noise_std=metabolic_noise_std,
                    score_mode=metabolic_score,
                    generation_mode=generation_mode,
                    structured_proj=structured_proj,
                )
                fork_count += 1
                if use_crit:
                    with torch.no_grad():
                        stats_out = model(inputs, return_jacobian_stats=True)
                        if "jacobian_stats" in stats_out:
                            out["jacobian_stats"] = stats_out["jacobian_stats"]
            elif use_fork and metabolic_mode == "mcts":
                # Micro-MCTS path: tree search using SSM as world model
                from chaoscontrol.metabolic import micro_mcts
                out = micro_mcts(
                    model, inputs,
                    n_rollouts=metabolic_k,
                    horizon=mcts_horizon,
                    ucb_c=mcts_ucb_c,
                )
                fork_count += 1
                if use_crit:
                    with torch.no_grad():
                        stats_out = model(inputs, return_jacobian_stats=True)
                        if "jacobian_stats" in stats_out:
                            out["jacobian_stats"] = stats_out["jacobian_stats"]
            else:
                # Cheap deterministic pass (always — MC mode uses this as the base)
                out = model(inputs, return_jacobian_stats=use_crit)

            # Monte Carlo path: sample the possibility space on surprise,
            # use distributional stats to weight the gradient — no winner picked
            mc_stats = None
            if use_fork and metabolic_mode == "monte_carlo":
                fork_count += 1
                with torch.no_grad():
                    mc_out = metabolic_monte_carlo(
                        model, inputs,
                        k=metabolic_k,
                        noise_std=metabolic_noise_std,
                        generation_mode=generation_mode,
                        structured_proj=structured_proj,
                    )
                    mc_stats = mc_out["mc_stats"]

            # Compute loss — MC mode weights per-token CE by uncertainty
            if mc_stats is not None:
                # Per-token CE weighted by uncertainty map
                per_token_ce = F.cross_entropy(
                    out["logits"].reshape(-1, vocab_size), targets.reshape(-1), reduction="none",
                ).reshape(inputs.size(0), -1)  # (batch, seq)
                umap = mc_stats["uncertainty_map"]  # (batch, seq)
                # Scale: mean weight = 1.0, uncertain tokens get up to 2x
                weights = 1.0 + umap / (umap.mean() + 1e-8)
                ce_loss = (per_token_ce * weights).mean()
            else:
                ce_loss = F.cross_entropy(out["logits"].reshape(-1, vocab_size), targets.reshape(-1))
            loss = ce_loss
            if use_crit and "jacobian_stats" in out:
                if "per_layer_jacobian_stats" in out:
                    # Dynamic per-layer criticality: linearly spaced targets
                    n_layers = len(out["per_layer_jacobian_stats"])
                    spread = 0.04  # +/- around global target
                    for li, layer_stats in enumerate(out["per_layer_jacobian_stats"]):
                        layer_target = crit_target_coupling - spread + 2 * spread * li / max(n_layers - 1, 1)
                        loss = loss + criticality_loss(
                            layer_stats, alpha=crit_reg_alpha / n_layers, beta=crit_reg_beta / n_layers,
                            target_log_sv=math.log(layer_target),
                        )
                else:
                    loss = loss + criticality_loss(
                        out["jacobian_stats"], alpha=crit_reg_alpha, beta=crit_reg_beta,
                        target_log_sv=math.log(crit_target_coupling),
                    )
            # Wernicke balance loss
            if "balance_loss" in out and model.wernicke is not None:
                loss = loss + model.wernicke_balance_weight * out["balance_loss"]

        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(all_params, grad_clip_norm)
        optimizer.step()

        ce_val_prev = history[-1]["loss"] if history else loss_ema  # noqa: F841
        ce_val = float(ce_loss.detach().cpu())
        loss_ema = 0.99 * loss_ema + 0.01 * ce_val
        last_forked = use_fork
        if use_fork:
            pre_fork_loss = ce_val_prev  # save loss BEFORE this fork for next step's comparison
        step_record: dict[str, Any] = {"step": float(steps), "loss": ce_val, "forked": use_fork}
        if metabolic_gate:
            step_record["threshold"] = current_threshold
            step_record["surprise_ratio"] = surprise_ratio
            step_record["metabolic_mode"] = metabolic_mode
        if mc_stats is not None:
            step_record["mc_variance"] = float(mc_stats["logits_var"].mean().cpu())
            step_record["mc_entropy"] = float(mc_stats["entropy"].mean().cpu())
            step_record["mc_divergence"] = float(mc_stats["candidate_divergence"].cpu())

        # Spectral logging: FFT power spectrum + Jacobian stats snapshot
        if steps % spectral_log_interval == 0 and "hidden" in out and not use_fork:
            with torch.no_grad():
                h = out["hidden"].detach().float()  # (batch, seq, dim)
                spec = torch.fft.rfft(h, dim=1)  # (batch, seq//2+1, dim)
                power = spec.abs().pow(2).mean(dim=(0, 2))  # (seq//2+1,)
                snapshot: dict[str, Any] = {
                    "step": int(steps),
                    "power_spectrum": power.cpu().tolist(),
                    "dominant_freq": int(power[1:].argmax().item()) + 1,
                }
                if "jacobian_stats" in out:
                    snapshot["lambda_max"] = float(out["jacobian_stats"]["lambda_max"].cpu())
                    snapshot["sv_log_var"] = float(out["jacobian_stats"]["sv_log_var"].cpu())
                if "per_layer_jacobian_stats" in out:
                    snapshot["per_layer_lambda_max"] = [
                        float(s["lambda_max"].cpu()) for s in out["per_layer_jacobian_stats"]
                    ]
                # Semantic/episodic divergence snapshot
                if (
                    hasattr(model, "semantic_tier") and model.semantic_tier is not None
                    and hasattr(model, "outer_model") and model.outer_model is not None
                    and hasattr(model.outer_model, "_slots") and model.outer_model._slots
                ):
                    semantic_norm = float(model.semantic_tier.bases.norm().cpu())
                    episodic_norm = float(torch.cat(model.outer_model._slots, dim=0).norm().cpu())
                    snapshot["semantic_norm"] = semantic_norm
                    snapshot["episodic_norm"] = episodic_norm
                spectral_snapshots.append(snapshot)

        # Bucket distribution logging (Wernicke typed composition)
        if steps % spectral_log_interval == 0 and "bucket_ids" in out:
            with torch.no_grad():
                bids = out["bucket_ids"].detach().reshape(-1)
                counts = torch.bincount(bids, minlength=model.wernicke.k_max if model.wernicke else 1)
                bucket_snapshots.append({
                    "step": int(steps),
                    "bucket_counts": counts.cpu().tolist(),
                    "active_buckets": int((counts > 0).sum().item()),
                })

        # Outer model consolidation (after optimizer step)
        dominant_bucket: int | None = None
        surprise_ratio_for_latent = 0.0
        if model.outer_model is not None:
            with torch.no_grad():
                per_sample_ce = F.cross_entropy(
                    out["logits"].detach().reshape(-1, vocab_size),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape(inputs.size(0), -1).mean(dim=1)  # (batch,)
            hidden = out["hidden"][:, -1, :].detach()  # (batch, dim)
            # Determine dominant bucket_id from Wernicke layer for typed writes
            # Only pass bucket_id when typed_storage is enabled
            dominant_bucket = None
            if "bucket_ids" in out and getattr(model, "typed_storage", False):
                bids = out["bucket_ids"].detach()  # (batch, seq)
                flat_ids = bids.reshape(-1)
                dominant_bucket = int(flat_ids.mode().values.item())

            if consolidation_write == "full_sequence" and hasattr(model.outer_model, 'write_sequence'):
                # Full-sequence write: manual surprise check + write_sequence
                running_avg = model.outer_model.loss_ema.item()
                signal = model.outer_model.compute_consolidation_signal(ce_val, running_avg)
                if running_avg > 0 and signal / running_avg > 0.01:
                    model.outer_model.write_sequence(
                        out["hidden"].detach(),
                        per_sample_weights=per_sample_ce,
                        bucket_id=dominant_bucket,
                    )
                model.outer_model.update_survival(ce_val)
                model.outer_model.loss_ema = model.outer_model.ema_decay * model.outer_model.loss_ema + (1 - model.outer_model.ema_decay) * ce_val
                step_record["surprise"] = float(signal)
                surprise_ratio_for_latent = signal / max(running_avg, 1e-6)
            else:
                surprise = model.outer_model.consolidation_step(
                    hidden, current_loss=ce_val, per_sample_weights=per_sample_ce,
                    bucket_id=dominant_bucket,
                )
                step_record["surprise"] = float(surprise)
                running_avg = model.outer_model.loss_ema.item()
                surprise_ratio_for_latent = float(surprise) / max(running_avg, 1e-6)

            # Latent persistence: reactivate compressed slot traces on high surprise
            if latent_persistence and hasattr(model.outer_model, 'try_reactivate'):
                if dominant_bucket is not None and surprise_ratio_for_latent > current_threshold:
                    model.outer_model.try_reactivate(bucket_id=dominant_bucket, surprise=surprise_ratio_for_latent)

            # Compression consequence: feed merge quality back to Wernicke
            # Only when compression_consequence flag is explicitly enabled
            if (
                getattr(model, "compression_consequence", False)
                and hasattr(model.outer_model, "_compression_consequences")
                and model.outer_model._compression_consequences
                and model.wernicke is not None
            ):
                for bucket_id, quality_delta in model.outer_model._compression_consequences:
                    model.wernicke.compression_consequence_update(bucket_id, quality_delta)
                model.outer_model._compression_consequences.clear()
            elif hasattr(model.outer_model, "_compression_consequences"):
                model.outer_model._compression_consequences.clear()

        # CFR regret update: estimate counterfactual values via short lookahead
        if regret_table is not None and use_fork and dominant_bucket is not None:
            actual_value = -ce_val  # negative CE (higher is better)

            # Get the top-K candidate tokens from last position
            with torch.no_grad():
                last_logits = out["logits"][:, -1, :].detach()
                k_actions = min(metabolic_k, last_logits.size(-1))
                _, top_tokens = last_logits.mean(dim=0).topk(k_actions)

                # Build state at the last position by stepping through input
                rollout_state = model.init_state(inputs.size(0))
                for t in range(inputs.size(1)):
                    _, _, rollout_state = model.step(inputs[:, t:t+1], rollout_state)

                # Short lookahead (2 steps) for each candidate token
                counterfactual_values = []
                for a in range(k_actions):
                    token = top_tokens[a].unsqueeze(0).expand(inputs.size(0)).unsqueeze(-1)
                    cf_state = [s.clone() for s in rollout_state]
                    cf_logits, _, cf_state = model.step(token, cf_state)
                    # One more step with greedy continuation
                    next_token = cf_logits.argmax(dim=-1, keepdim=True)
                    cf_logits2, _, _ = model.step(next_token, cf_state)
                    # Value = negative mean CE of 2-step lookahead
                    cf_probs = F.softmax(cf_logits2, dim=-1)
                    cf_value = cf_probs.max(dim=-1).values.mean().item()
                    counterfactual_values.append(cf_value)

            action_taken = 0
            if "mcts_stats" in out and "visit_counts" in out["mcts_stats"]:
                action_taken = int(out["mcts_stats"]["visit_counts"].argmax().item())

            regret_table.update(
                bucket_id=dominant_bucket % regret_table.n_buckets,
                action_taken=action_taken,
                counterfactual_values=counterfactual_values,
                actual_value=actual_value,
            )

        # Semantic tier consolidation: extract gist from recent episodic slots
        if (
            hasattr(model, "semantic_tier")
            and model.semantic_tier is not None
            and hasattr(model, "outer_model")
            and model.outer_model is not None
            and hasattr(model.outer_model, "_slots")
            and model.outer_model._slots
        ):
            if getattr(model, "typed_consolidation", False) and dominant_bucket is not None:
                # Type-aware: only consolidate slots matching dominant bucket
                matching = [
                    s for s, b in zip(model.outer_model._slots, model.outer_model._slot_buckets)
                    if b == dominant_bucket
                ]
                slots_for_consolidation = matching[-5:] if matching else model.outer_model._slots[-5:]
            else:
                # Untyped: use all recent slots
                slots_for_consolidation = model.outer_model._slots[-5:]
            recent_slots = torch.cat(slots_for_consolidation, dim=0)
            recent_decoded = model.outer_model.decoder(recent_slots)
            model.semantic_tier.consolidate_from_episodes(recent_decoded)

        history.append(step_record)
        steps += 1

    maybe_sync_cuda(device)
    return {
        "steps": int(steps),
        "history": history,
        "elapsed_s": float(time.perf_counter() - start_time),
        "fork_count": fork_count,
        "extra_params": sum(p.numel() for p in structured_proj.parameters()) if structured_proj else 0,
        "structured_proj": structured_proj,
        "spectral_snapshots": spectral_snapshots,
        "bucket_snapshots": bucket_snapshots,
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
            semantic_tier_bases=cfg.semantic_tier_bases,
            typed_storage=cfg.typed_storage,
            typed_consolidation=cfg.typed_consolidation,
            compression_consequence=cfg.compression_consequence,
            cue_projection=cfg.cue_projection,
            dynamic_crit_per_layer=cfg.dynamic_crit_per_layer,
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
            generation_mode=cfg.generation_mode,
            metabolic_mode=cfg.metabolic_mode,
            mcts_horizon=cfg.mcts_horizon,
            mcts_ucb_c=cfg.mcts_ucb_c,
            consolidation_write=cfg.consolidation_write,
            latent_persistence=cfg.latent_persistence,
            cfr_enabled=cfg.cfr_enabled,
        )

        eval_result = evaluate_chaoscontrol_bpb(
            model,
            tokens=eval_tokens,
            eval_starts=eval_starts,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            device=device,
            warmup=cfg.eval_warmup,
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
