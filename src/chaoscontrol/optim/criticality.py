"""Criticality Distillation mechanism (design: docs/plans/2026-04-24-criticality-distillation.md).

ScOpt produces per-token pressure; CriticalityDistillation consumes pressure
and recurrence-state traces, scores channels by post-event trace survival,
allocates a budgeted set of near-critical seats, and emits a seat-masked MSE
loss on `log_a`. Diag-backend only in stage 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CriticalityDistillation(nn.Module):
    """Criticality distillation loss generator + trace bank.

    All mechanism state (bank, baseline EMA, current seats) is stored as
    registered buffers so checkpointing is automatic via `state_dict`.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        dim: int,
        trace_ttl_steps: int = 1024,
        trace_half_life_steps: float = 256.0,
        seat_refresh_interval: int = 64,
        criticality_budget_frac: float = 0.15,
        critical_value: float = 0.95,
        min_weighted_events_per_layer: float = 256.0,
        criticality_distill_weight: float = 1e-3,
        baseline_ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        if not 0.0 < criticality_budget_frac < 1.0:
            raise ValueError(
                f"criticality_budget_frac must be in (0, 1); got {criticality_budget_frac}"
            )
        if critical_value <= 0.0 or critical_value >= 1.0:
            raise ValueError(
                f"critical_value must be in (0, 1); got {critical_value}"
            )
        self.num_layers = int(num_layers)
        self.dim = int(dim)
        self.trace_ttl_steps = int(trace_ttl_steps)
        self.trace_half_life_steps = float(trace_half_life_steps)
        self.seat_refresh_interval = int(seat_refresh_interval)
        self.criticality_budget_frac = float(criticality_budget_frac)
        self.critical_value = float(critical_value)
        self.min_weighted_events_per_layer = float(min_weighted_events_per_layer)
        self.criticality_distill_weight = float(criticality_distill_weight)
        self.baseline_ema_decay = float(baseline_ema_decay)

        # Per-layer ring buffer keyed by step index (one evidence vector per
        # (layer, step) that had at least one event).
        self.register_buffer(
            "bank_evidence",
            torch.zeros(self.num_layers, self.trace_ttl_steps, self.dim, dtype=torch.float32),
        )
        # Slot's originating step; -1 means "empty".
        self.register_buffer(
            "bank_step",
            torch.full((self.num_layers, self.trace_ttl_steps), -1, dtype=torch.int64),
        )
        # Number of events contributing to this slot's evidence.
        self.register_buffer(
            "bank_event_count",
            torch.zeros(self.num_layers, self.trace_ttl_steps, dtype=torch.float32),
        )
        # Per-layer per-channel EMA of non-event future energy.
        self.register_buffer(
            "baseline_future_energy",
            torch.zeros(self.num_layers, self.dim, dtype=torch.float32),
        )
        # Current seat assignment per layer (top-k channels that feel the loss).
        self.register_buffer(
            "seat_mask",
            torch.zeros(self.num_layers, self.dim, dtype=torch.bool),
        )

    def add_step_evidence(
        self,
        *,
        layer: int,
        step: int,
        evidence: torch.Tensor,
        event_count: float,
    ) -> None:
        """Write one (layer, step) evidence vector into the bank.

        Slot selection rule:
          * If there is any empty slot (bank_step == -1), fill the smallest
            empty index.
          * Else evict the slot with the oldest bank_step value (lowest step).
        This gives us a TTL-wrapped ring without tracking a separate write
        pointer — the aging math naturally demotes the oldest evidence.
        """
        if not 0 <= layer < self.num_layers:
            raise IndexError(
                f"layer={layer} out of range for num_layers={self.num_layers}"
            )
        if evidence.shape != (self.dim,):
            raise ValueError(
                f"evidence must have shape ({self.dim},); got {tuple(evidence.shape)}"
            )

        slots = self.bank_step[layer]  # [trace_ttl_steps]
        empty = (slots == -1).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            slot = int(empty[0].item())
        else:
            # Evict oldest
            slot = int(slots.argmin().item())

        self.bank_evidence[layer, slot] = evidence.to(
            dtype=self.bank_evidence.dtype, device=self.bank_evidence.device
        )
        self.bank_step[layer, slot] = int(step)
        self.bank_event_count[layer, slot] = float(event_count)
