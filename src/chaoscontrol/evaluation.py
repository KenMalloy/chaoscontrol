"""Evaluation utilities for ChaosControl models."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from chaoscontrol.data import batch_from_starts, maybe_autocast


def evaluate_chaoscontrol_bpb(
    model: Any,
    *,
    tokens: torch.Tensor,
    eval_starts: list[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate ChaosStudentLM, returning loss and bits-per-byte."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    vocab_size = model.vocab_size
    try:
        with torch.no_grad():
            for idx in range(0, len(eval_starts), batch_size):
                batch_starts = eval_starts[idx : idx + batch_size]
                inputs, targets = batch_from_starts(tokens, batch_starts, seq_len, device)
                autocast_dtype = next(model.parameters()).dtype if device.type == "cuda" else torch.float32
                with maybe_autocast(device, autocast_dtype):
                    logits = model(inputs)["logits"]
                total_loss += float(
                    F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="sum").item()
                )
                total_tokens += int(targets.numel())
    finally:
        if was_training:
            model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": float(mean_loss),
        "bpb": float(mean_loss / math.log(2.0)),
        "tokens": float(total_tokens),
    }
