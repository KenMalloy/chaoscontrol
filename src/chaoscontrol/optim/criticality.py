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

    def score(self, current_step: int) -> torch.Tensor:
        """Age-weighted average of evidence across the bank.

        Returns `[num_layers, dim]` fp32 score. Empty bank (no valid slots)
        produces zeros.

        Age weight is `2 ** (-age / trace_half_life_steps)`, so `age ==
        trace_half_life_steps` carries weight 0.5, and `age == 0` carries
        weight 1.0.
        """
        valid = self.bank_step >= 0  # [L, T]
        age = (int(current_step) - self.bank_step).clamp_min(0).to(dtype=torch.float32)
        weight = torch.exp2(-age / self.trace_half_life_steps)
        weight = weight * valid.to(dtype=torch.float32)  # zero-out empty slots
        weight_sum = weight.sum(dim=1, keepdim=True)  # [L, 1]
        weighted_evidence = (weight.unsqueeze(-1) * self.bank_evidence).sum(dim=1)  # [L, D]
        safe_denom = weight_sum.clamp_min(1e-12)
        score = weighted_evidence / safe_denom
        # Layers with zero total weight -> zeros (not NaN).
        score = torch.where(
            weight_sum > 0,
            score,
            torch.zeros_like(score),
        )
        return score

    @torch.no_grad()
    def update_baseline_ema(
        self,
        *,
        layer: int,
        future_energy: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> None:
        """Update per-channel baseline EMA using only non-event positions."""
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer={layer}")
        if future_energy.shape[-1] != self.dim:
            raise ValueError(
                f"future_energy last dim must be {self.dim}; got {tuple(future_energy.shape)}"
            )
        non_event = ~event_mask  # [B, T]
        if not non_event.any():
            return  # no new information; leave EMA alone
        flat_fe = future_energy.reshape(-1, self.dim)  # [B*T, D]
        flat_m = non_event.reshape(-1)  # [B*T]
        obs = flat_fe[flat_m].mean(dim=0)  # [D]
        decay = self.baseline_ema_decay
        self.baseline_future_energy[layer].mul_(decay).add_(obs.to(self.baseline_future_energy.dtype), alpha=(1.0 - decay))

    @torch.no_grad()
    def ingest_step(
        self,
        *,
        step: int,
        pressure: torch.Tensor,
        states_per_layer: list,
        horizon_H: int,
        event_frac: float,
    ) -> None:
        """Full per-step evidence ingestion.

        Args:
            step: current training step index.
            pressure: `[B, T]` pressure field (any real-valued tensor).
            states_per_layer: list of length `num_layers`, each entry a
                `[B, T, D]` captured states tensor.
            horizon_H: trailing window for post-event energy.
            event_frac: fraction of positions to mark as events.
        """
        if len(states_per_layer) != self.num_layers:
            raise ValueError(
                f"states_per_layer must have {self.num_layers} entries; got {len(states_per_layer)}"
            )
        event_mask = compute_event_mask(pressure, event_frac=event_frac)  # [B, T]
        n_events = int(event_mask.sum().item())
        if n_events == 0:
            return
        for layer, states in enumerate(states_per_layer):
            if states.shape[-1] != self.dim:
                raise ValueError(
                    f"layer {layer}: states last dim {states.shape[-1]} != self.dim {self.dim}"
                )
            future_energy = compute_future_energy(states, horizon_H=horizon_H)  # [B, T, D]
            self.update_baseline_ema(
                layer=layer, future_energy=future_energy, event_mask=event_mask
            )
            baseline = self.baseline_future_energy[layer]  # [D]
            excess = (future_energy - baseline).clamp_min(0.0)  # [B, T, D]
            # Aggregate: mean over event positions.
            flat_excess = excess.reshape(-1, self.dim)  # [B*T, D]
            flat_mask = event_mask.reshape(-1)  # [B*T]
            aggregate = flat_excess[flat_mask].mean(dim=0)  # [D]
            self.add_step_evidence(
                layer=layer,
                step=step,
                evidence=aggregate,
                event_count=float(flat_mask.sum().item()),
            )

    @torch.no_grad()
    def allocate_seats(self, *, current_step: int) -> None:
        """Recompute per-layer seat assignment from current age-weighted score.

        Gate: a layer's total age-weighted event count must exceed
        `min_weighted_events_per_layer`; otherwise its seats are cleared.
        """
        valid = self.bank_step >= 0
        age = (int(current_step) - self.bank_step).clamp_min(0).to(dtype=torch.float32)
        weight = torch.exp2(-age / self.trace_half_life_steps)
        weight = weight * valid.to(dtype=torch.float32)  # [L, T]
        weighted_events_per_layer = (weight * self.bank_event_count).sum(dim=1)  # [L]

        k = max(1, int(round(self.dim * self.criticality_budget_frac)))
        scores = self.score(current_step=current_step)  # [L, D]

        for layer in range(self.num_layers):
            if weighted_events_per_layer[layer].item() < self.min_weighted_events_per_layer:
                self.seat_mask[layer].fill_(False)
                continue
            topk = torch.topk(scores[layer], k=k, largest=True)
            mask = torch.zeros(self.dim, dtype=torch.bool, device=self.seat_mask.device)
            mask[topk.indices] = True
            self.seat_mask[layer] = mask

    def criticality_loss(self, log_a_per_layer: list) -> torch.Tensor:
        """Seat-masked MSE loss pulling `1 - sigmoid(log_a[seat])` toward
        `critical_value`.

        Non-seat channels contribute exactly zero to the loss (and therefore
        exactly zero gradient to their log_a).

        Returns:
            Scalar tensor. Weight of this term in the total loss is applied
            externally (`criticality_distill_weight` is not multiplied here).
        """
        if len(log_a_per_layer) != self.num_layers:
            raise ValueError(
                f"log_a_per_layer must have {self.num_layers} entries"
            )
        total = torch.zeros((), dtype=torch.float32, device=self.seat_mask.device)
        any_seats = False
        for layer, log_a in enumerate(log_a_per_layer):
            mask = self.seat_mask[layer]
            if not mask.any():
                continue
            any_seats = True
            criticality = 1.0 - torch.sigmoid(log_a.to(dtype=torch.float32))
            err = (criticality - self.critical_value) ** 2
            # Select seat entries explicitly so non-seats contribute no op that
            # could produce grad through masking arithmetic.
            seat_err = err[mask]
            total = total + seat_err.mean()
        if not any_seats:
            return torch.zeros((), dtype=torch.float32, device=self.seat_mask.device)
        return total


def compute_event_mask(pressure: torch.Tensor, event_frac: float) -> torch.Tensor:
    """Top-`event_frac` positions of pressure become True.

    Args:
        pressure: any shape; absolute magnitude determines rank.
        event_frac: fraction in [0, 1].

    Returns:
        Boolean tensor, same shape as `pressure`.
    """
    if not 0.0 <= event_frac <= 1.0:
        raise ValueError(f"event_frac must be in [0, 1]; got {event_frac}")
    total = pressure.numel()
    k = int(round(event_frac * total))
    if k == 0:
        return torch.zeros_like(pressure, dtype=torch.bool)
    if k >= total:
        return torch.ones_like(pressure, dtype=torch.bool)
    flat = pressure.reshape(-1)
    _, idx = torch.topk(flat, k=k, largest=True)
    mask = torch.zeros(total, dtype=torch.bool, device=pressure.device)
    mask[idx] = True
    return mask.reshape(pressure.shape)


def compute_future_energy(states: torch.Tensor, horizon_H: int) -> torch.Tensor:
    """Per-position mean-square energy over the trailing window `[t+1, t+H]`.

    Args:
        states: `[B, T, D]` recurrence states.
        horizon_H: window length (strictly positive).

    Returns:
        `[B, T, D]` — empty windows (tail where `t+1 >= T`) produce zeros.
    """
    if horizon_H < 1:
        raise ValueError(f"horizon_H must be >= 1; got {horizon_H}")
    B, T, D = states.shape
    sq = states.pow(2)  # [B, T, D]
    out = torch.zeros_like(sq)
    for t in range(T):
        start = t + 1
        stop = min(t + 1 + horizon_H, T)
        if start >= stop:
            continue  # empty window -> zeros
        out[:, t, :] = sq[:, start:stop, :].mean(dim=1)
    return out
