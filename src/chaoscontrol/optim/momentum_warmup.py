"""Linear momentum warmup for Muon param groups.

Mirrors the schedule used in records/track_10min_16mb/2026-03-22_11L_EMA_*
(train_gpt.py:1239-1240) — linearly interpolates from `start` to `target` over
`steps` steps, then clamps at `target`.
"""
from __future__ import annotations

import torch


def apply_momentum_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    target: float,
    start: float,
    steps: int,
) -> float:
    """Set ``momentum`` on every param group of ``optimizer`` for the current step.

    Args:
        optimizer: Any optimizer that reads ``param_groups[i]["momentum"]`` per
            step (Muon does this in ``step()``).
        step: Current global training step (0-indexed).
        target: Final momentum value (e.g. 0.99 for SOTA submissions).
        start: Initial momentum value at step 0 (e.g. 0.92).
        steps: Number of warmup steps. ``steps=0`` disables warmup and pins
            momentum at ``target`` immediately.

    Returns:
        The momentum value applied (useful for telemetry).
    """
    if steps <= 0:
        frac = 1.0
    else:
        frac = min(step / steps, 1.0)
    momentum = (1.0 - frac) * start + frac * target
    for group in optimizer.param_groups:
        group["momentum"] = momentum
    return momentum
