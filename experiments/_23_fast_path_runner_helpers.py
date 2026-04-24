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


def _alloc_pinned_evidence_buffers(
    *,
    num_layers: int,
    dim: int,
    use_pinned: bool,
) -> dict:
    """Allocate host-side A/B slots matching ``ingest_gpu``'s return dict.

    Each slot holds the five tensors ``ingest_gpu`` produces. When
    ``use_pinned=True`` (CUDA available), buffers are pinned so D2H
    copies can be asynchronous; on CPU-only hosts pinning is skipped
    (``pin_memory=True`` is CUDA-only and silently mis-behaves elsewhere).
    """
    def _slot() -> dict:
        slot = {
            "aggregated_excess_per_layer": torch.zeros(
                num_layers, dim, dtype=torch.float32,
                pin_memory=bool(use_pinned),
            ),
            "non_event_mean_future_energy_per_layer": torch.zeros(
                num_layers, dim, dtype=torch.float32,
                pin_memory=bool(use_pinned),
            ),
            "event_count_per_layer": torch.zeros(
                num_layers, dtype=torch.float32,
                pin_memory=bool(use_pinned),
            ),
            "n_events_scalar": torch.zeros(
                (), dtype=torch.int64, pin_memory=bool(use_pinned),
            ),
            "n_non_events_scalar": torch.zeros(
                (), dtype=torch.int64, pin_memory=bool(use_pinned),
            ),
        }
        return slot

    return {"A": _slot(), "B": _slot()}
