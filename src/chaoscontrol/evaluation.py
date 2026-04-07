"""Evaluation utilities for ChaosControl models."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from chaoscontrol.data import batch_from_starts, maybe_autocast
from chaoscontrol.metabolic import metabolic_fork, StructuredProjections


def compute_bpb(total_ce_nats: float, total_raw_bytes: int) -> float:
    """Compute bits-per-byte. Tokenizer-agnostic.

    Args:
        total_ce_nats: Sum of cross-entropy loss (in nats) across all predicted tokens.
        total_raw_bytes: Count of raw bytes in the evaluation text.
            This is a property of the text, independent of the model's tokenizer.

    Returns:
        Bits per byte. Lower is better.
    """
    if total_raw_bytes <= 0:
        return 0.0
    return total_ce_nats / total_raw_bytes / math.log(2.0)


def evaluate_chaoscontrol_bpb(
    model: Any,
    *,
    tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    metabolic_gate: bool = False,
    metabolic_k: int = 4,
    metabolic_score: str = "memory_consistency",
    metabolic_noise_std: float = 0.01,
    metabolic_mode: str = "fork",
    generation_mode: str = "noise",
    structured_proj: Any = None,
    warmup: bool = False,
    warmup_write_mode: str = "last",
    warmup_latent: bool = False,
    warmup_cold_start: bool = False,
    total_raw_bytes: int | None = None,
) -> dict[str, float]:
    """Evaluate ChaosStudentLM, returning loss and bits-per-byte.

    When metabolic_gate=True, runs both a plain forward pass and a gate-aware
    pass, returning both so experiments can compare.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_loss_gated = 0.0
    total_tokens = 0
    vocab_size = model.vocab_size

    # Save outer model state before eval warmup so memory writes don't persist
    saved_outer_state = None
    if warmup and getattr(model, "outer_model", None) is not None:
        om = model.outer_model
        saved_outer_state = {
            "loss_ema": om.loss_ema.clone(),
            # Trigger state
            "_spike_seen": om._spike_seen,
            "_steps_since_spike": om._steps_since_spike,
            "_pre_spike_loss": om._pre_spike_loss,
        }
        # MultiSlotOuterModel fields (not present on single-slot OuterModel)
        if hasattr(om, "_slots"):
            saved_outer_state["slots"] = [s.clone() for s in om._slots]
        if hasattr(om, "_survival"):
            saved_outer_state["survival"] = list(om._survival)
        if hasattr(om, "_slot_buckets"):
            saved_outer_state["slot_buckets"] = list(om._slot_buckets)
        if hasattr(om, "_retrieval_weights"):
            saved_outer_state["_retrieval_weights"] = om._retrieval_weights
        if hasattr(om, "_compression_consequences"):
            saved_outer_state["_compression_consequences"] = list(om._compression_consequences)
        if hasattr(om, "_latent_traces"):
            saved_outer_state["latent_traces"] = [
                {"bucket_id": t["bucket_id"], "centroid_contrib": t["centroid_contrib"].clone()}
                for t in om._latent_traces
            ]
        if hasattr(om, "_compress_rng"):
            saved_outer_state["_compress_rng_state"] = om._compress_rng.getstate()
        # Single-slot OuterModel state field
        if hasattr(om, "state"):
            saved_outer_state["state"] = om.state.clone()
        if hasattr(om, "consolidation_w"):
            saved_outer_state["consolidation_w"] = om.consolidation_w.clone()
        if hasattr(om, "_last_signal_was_pain"):
            saved_outer_state["_last_signal_was_pain"] = om._last_signal_was_pain.clone()
            saved_outer_state["_last_loss"] = om._last_loss.clone()
            saved_outer_state["_last_wrote"] = om._last_wrote.clone()

        # Cold start: wipe all memory before eval loop
        if warmup_cold_start:
            if hasattr(om, "_slots"):
                om._slots = []
            if hasattr(om, "_survival"):
                om._survival = []
            if hasattr(om, "_slot_buckets"):
                om._slot_buckets = []
            if hasattr(om, "_latent_traces"):
                om._latent_traces = []
            if hasattr(om, "state"):
                om.state.zero_()

    try:
        with torch.no_grad():
            for idx in range(0, len(eval_starts), batch_size):
                batch_starts = eval_starts[idx : idx + batch_size]
                inputs, targets = batch_from_starts(tokens, batch_starts, seq_len, device)
                autocast_dtype = next(model.parameters()).dtype if device.type == "cuda" else torch.float32
                with maybe_autocast(device, autocast_dtype):
                    # Standard deterministic eval
                    out = model(inputs)
                    logits = out["logits"]

                    # Warmup: write to episodic memory for future batches
                    if warmup and getattr(model, "outer_model", None) is not None:
                        batch_loss = F.cross_entropy(
                            logits.float().reshape(-1, vocab_size),
                            targets.reshape(-1),
                        ).item()

                        if warmup_write_mode == "full_sequence" and hasattr(model.outer_model, "write_sequence"):
                            # Full-sequence write: mirrors training's full_sequence path
                            running_avg = model.outer_model.loss_ema.item()
                            signal = model.outer_model.compute_consolidation_signal(batch_loss, running_avg)
                            if running_avg > 0 and signal / running_avg > 0.01:
                                model.outer_model.write_sequence(
                                    out["hidden"].detach(),
                                    bucket_id=None,
                                )
                            model.outer_model.update_survival(batch_loss)
                            model.outer_model.loss_ema = (
                                model.outer_model.ema_decay * model.outer_model.loss_ema
                                + (1 - model.outer_model.ema_decay) * batch_loss
                            )
                        else:
                            # Default: consolidation_step with last hidden
                            hidden_last = out["hidden"][:, -1, :].detach()
                            model.outer_model.consolidation_step(
                                hidden_last,
                                current_loss=batch_loss,
                                bucket_id=None,
                            )

                        # Latent reactivation on high surprise
                        if warmup_latent and hasattr(model.outer_model, "try_reactivate"):
                            running_avg = model.outer_model.loss_ema.item()
                            surprise_ratio = batch_loss / max(running_avg, 1e-6)
                            if surprise_ratio > 1.0:
                                model.outer_model.try_reactivate(
                                    bucket_id=None, surprise=surprise_ratio,
                                )

                    # Gate-aware eval (if metabolic gate is active)
                    if metabolic_gate:
                        if metabolic_mode == "mcts":
                            from chaoscontrol.metabolic import micro_mcts
                            gated_out = micro_mcts(
                                model, inputs,
                                n_rollouts=metabolic_k, horizon=8,
                            )
                        elif metabolic_mode == "monte_carlo":
                            from chaoscontrol.metabolic import metabolic_monte_carlo
                            gated_out = metabolic_monte_carlo(
                                model, inputs,
                                k=metabolic_k,
                                noise_std=metabolic_noise_std,
                                generation_mode=generation_mode,
                                structured_proj=structured_proj,
                            )
                        else:
                            gated_out = metabolic_fork(
                                model, inputs,
                                k=metabolic_k,
                                noise_std=metabolic_noise_std,
                                score_mode=metabolic_score,
                                generation_mode=generation_mode,
                                structured_proj=structured_proj,
                            )
                        gated_logits = gated_out["logits"]
                        total_loss_gated += float(
                            F.cross_entropy(
                                gated_logits.float().reshape(-1, vocab_size),
                                targets.reshape(-1), reduction="sum",
                            ).item()
                        )

                total_loss += float(
                    F.cross_entropy(logits.float().reshape(-1, vocab_size), targets.reshape(-1), reduction="sum").item()
                )
                total_tokens += int(targets.numel())
    finally:
        if saved_outer_state is not None:
            om = model.outer_model
            om.loss_ema = saved_outer_state["loss_ema"]
            # Trigger state
            om._spike_seen = saved_outer_state["_spike_seen"]
            om._steps_since_spike = saved_outer_state["_steps_since_spike"]
            om._pre_spike_loss = saved_outer_state["_pre_spike_loss"]
            # MultiSlotOuterModel fields
            if "slots" in saved_outer_state:
                om._slots = saved_outer_state["slots"]
            if "survival" in saved_outer_state:
                om._survival = saved_outer_state["survival"]
            if "slot_buckets" in saved_outer_state:
                om._slot_buckets = saved_outer_state["slot_buckets"]
            if "_retrieval_weights" in saved_outer_state:
                om._retrieval_weights = saved_outer_state["_retrieval_weights"]
            if "_compression_consequences" in saved_outer_state:
                om._compression_consequences = saved_outer_state["_compression_consequences"]
            if "latent_traces" in saved_outer_state:
                om._latent_traces = saved_outer_state["latent_traces"]
            if "_compress_rng_state" in saved_outer_state:
                om._compress_rng.setstate(saved_outer_state["_compress_rng_state"])
            # Single-slot OuterModel state
            if "state" in saved_outer_state:
                om.state = saved_outer_state["state"]
            if "consolidation_w" in saved_outer_state:
                om.consolidation_w = saved_outer_state["consolidation_w"]
            if "_last_signal_was_pain" in saved_outer_state:
                om._last_signal_was_pain = saved_outer_state["_last_signal_was_pain"]
                om._last_loss = saved_outer_state["_last_loss"]
                om._last_wrote = saved_outer_state["_last_wrote"]
        if was_training:
            model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    result = {
        "loss": float(mean_loss),
        "bpb": float(mean_loss / math.log(2.0)),  # per-token bpb
        "tokens": float(total_tokens),
    }
    # When raw byte count is provided, add the proper tokenizer-agnostic bpb
    if total_raw_bytes is not None:
        result["bpb_raw"] = compute_bpb(total_loss, total_raw_bytes)
    if metabolic_gate:
        mean_loss_gated = total_loss_gated / max(total_tokens, 1)
        result["loss_gated"] = float(mean_loss_gated)
        result["bpb_gated"] = float(mean_loss_gated / math.log(2.0))
    return result
