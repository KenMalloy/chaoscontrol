"""Small helpers for the `experiments/23_fast_path/runner_fast_path.py`
CD wiring. Kept out of the runner file so they can be unit-tested in
isolation without pulling the heavy runner module."""
from __future__ import annotations

import torch


def compute_ce_minus_entropy_pressure_from_fused(
    per_token_ce: torch.Tensor,
    per_token_entropy: torch.Tensor,
) -> torch.Tensor:
    """Model-native surprise pressure — observed CE minus predictive entropy.

    Args:
        per_token_ce: `[B, T]` (or any matching shape) cross-entropy.
        per_token_entropy: same shape as `per_token_ce`.

    Returns:
        `relu(per_token_ce - per_token_entropy)` — non-negative pressure
        that fires on confident-wrong tokens and stays silent on
        confused-wrong tokens (where high entropy cancels high CE).
    """
    return torch.relu(per_token_ce - per_token_entropy)
