"""Evaluation utilities for ChaosControl models."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from chaoscontrol.data import batch_from_starts, maybe_autocast
from chaoscontrol.metabolic import metabolic_fork, StructuredProjections


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
    generation_mode: str = "noise",
    structured_proj: Any = None,
    warmup: bool = False,
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
                        hidden_last = out["hidden"][:, -1, :].detach()
                        batch_loss = F.cross_entropy(
                            logits.float().reshape(-1, vocab_size),
                            targets.reshape(-1),
                        ).item()
                        model.outer_model.consolidation_step(
                            hidden_last,
                            current_loss=batch_loss,
                            bucket_id=None,
                        )

                    # Gate-aware eval (if metabolic gate is active)
                    if metabolic_gate:
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
        if was_training:
            model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    result = {
        "loss": float(mean_loss),
        "bpb": float(mean_loss / math.log(2.0)),
        "tokens": float(total_tokens),
    }
    if metabolic_gate:
        mean_loss_gated = total_loss_gated / max(total_tokens, 1)
        result["loss_gated"] = float(mean_loss_gated)
        result["bpb_gated"] = float(mean_loss_gated / math.log(2.0))
    return result
