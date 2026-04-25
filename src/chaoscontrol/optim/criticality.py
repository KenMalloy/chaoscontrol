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
        score_permute_before_topk: bool = False,
        fixed_random_seats: bool = False,
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
        self.score_permute_before_topk = bool(score_permute_before_topk)
        self.fixed_random_seats = bool(fixed_random_seats)

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
        # Flag: has the baseline been seeded for this layer? First
        # observation replaces the zero init rather than EMA-damping it.
        self.register_buffer(
            "baseline_initialized",
            torch.zeros(self.num_layers, dtype=torch.bool),
        )
        # Current seat assignment per layer (top-k channels that feel the loss).
        self.register_buffer(
            "seat_mask",
            torch.zeros(self.num_layers, self.dim, dtype=torch.bool),
        )
        # Running decayed accumulators for O(L·D) per-step scoring; replace
        # the full ring-bank rescan in the hot path.
        self.register_buffer(
            "score_num", torch.zeros(self.num_layers, self.dim, dtype=torch.float32)
        )
        self.register_buffer(
            "score_den", torch.zeros(self.num_layers, dtype=torch.float32)
        )
        self.register_buffer(
            "event_mass", torch.zeros(self.num_layers, dtype=torch.float32)
        )
        self.register_buffer(
            "last_decay_step", torch.tensor(-1, dtype=torch.int64)
        )
        if self.fixed_random_seats:
            # Falsifier: bind seats ONCE at construction using torch.randperm.
            # allocate_seats becomes a no-op; ingest still runs so cost and
            # evidence-bank behavior match the treatment cell.
            k = max(1, int(round(self.dim * self.criticality_budget_frac)))
            for layer in range(self.num_layers):
                perm = torch.randperm(self.dim)
                self.seat_mask[layer, perm[:k]] = True
        # Previous seat mask used by `diagnostics_snapshot` to compute
        # per-layer churn. Not a registered buffer — diagnostics don't
        # need to checkpoint, and it must not ride the state_dict load
        # path.
        self._previous_seat_mask_snapshot: torch.Tensor | None = None

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
        """Age-weighted, count-weighted average of evidence across the bank.

        Returns `[num_layers, dim]` fp32 score. Empty bank (no valid slots)
        produces zeros.

        Per-slot weight is `age_factor * event_count`, where
        `age_factor = 2 ** (-age / trace_half_life_steps)` (so `age ==
        trace_half_life_steps` carries age_factor 0.5, and `age == 0`
        carries 1.0). Matches the running-accumulator scorer, which
        adds each step's contribution as `evidence * event_count` and
        then ages it uniformly.
        """
        valid = self.bank_step >= 0  # [L, T]
        age = (int(current_step) - self.bank_step).clamp_min(0).to(dtype=torch.float32)
        age_weight = torch.exp2(-age / self.trace_half_life_steps)
        age_weight = age_weight * valid.to(dtype=torch.float32)  # zero-out empty slots
        count_weight = age_weight * self.bank_event_count  # [L, T]
        weight_sum = count_weight.sum(dim=1, keepdim=True)  # [L, 1]
        weighted_evidence = (count_weight.unsqueeze(-1) * self.bank_evidence).sum(dim=1)  # [L, D]
        safe_denom = weight_sum.clamp_min(1e-12)
        score = weighted_evidence / safe_denom
        # Layers with zero total weight -> zeros (not NaN).
        score = torch.where(
            weight_sum > 0,
            score,
            torch.zeros_like(score),
        )
        return score

    def score_from_accumulators(self) -> torch.Tensor:
        """Age-weighted count-weighted mean of evidence, read from
        running accumulators. Hot-path scorer.

        Returns `[num_layers, dim]` fp32. Layers with zero event_mass
        return zero (not NaN).
        """
        denom = self.score_den.clamp_min(1e-12).unsqueeze(-1)
        raw = self.score_num / denom  # [L, D]
        valid = self.event_mass > 0
        return torch.where(valid.unsqueeze(-1), raw, torch.zeros_like(raw))

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
        obs = flat_fe[flat_m].mean(dim=0).to(self.baseline_future_energy.dtype)  # [D]
        if not bool(self.baseline_initialized[layer].item()):
            self.baseline_future_energy[layer].copy_(obs)
            self.baseline_initialized[layer] = True
        else:
            decay = self.baseline_ema_decay
            self.baseline_future_energy[layer].mul_(decay).add_(obs, alpha=(1.0 - decay))

    @torch.no_grad()
    def _step_decay_accumulators(self, current_step: int) -> None:
        """Apply age decay to running accumulators between
        last_decay_step and current_step. Idempotent when called with
        the same step."""
        if int(current_step) <= int(self.last_decay_step.item()):
            return
        dt = int(current_step) - int(self.last_decay_step.item())
        if int(self.last_decay_step.item()) < 0:
            # First time — accumulators are all zero, no decay needed.
            self.last_decay_step.fill_(int(current_step))
            return
        factor = 2.0 ** (-float(dt) / self.trace_half_life_steps)
        self.score_num.mul_(factor)
        self.score_den.mul_(factor)
        self.event_mass.mul_(factor)
        self.last_decay_step.fill_(int(current_step))

    @torch.no_grad()
    def _add_contribution(
        self,
        *,
        layer: int,
        evidence: torch.Tensor,
        event_count: float,
    ) -> None:
        """Incremental additive update of accumulators. Weight for this
        step's contribution is always 1.0 (the age is zero). Evidence
        tensor must match `self.dim`."""
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer={layer}")
        if evidence.shape != (self.dim,):
            raise ValueError(
                f"evidence must have shape ({self.dim},); got {tuple(evidence.shape)}"
            )
        ec = float(event_count)
        ev = evidence.to(dtype=self.score_num.dtype, device=self.score_num.device)
        self.score_num[layer].add_(ev, alpha=ec)
        self.score_den[layer].add_(ec)
        self.event_mass[layer].add_(ec)

    @torch.no_grad()
    def _subtract_expired_contribution(
        self,
        *,
        layer: int,
        evicted_step: int,
        current_step: int,
        evicted_evidence: torch.Tensor,
        evicted_event_count: float,
    ) -> None:
        """Remove an expired/evicted slot's remaining contribution at
        its current decay weight. Call BEFORE writing a new entry into
        an occupied slot."""
        age = max(0, int(current_step) - int(evicted_step))
        factor = 2.0 ** (-float(age) / self.trace_half_life_steps)
        remaining_ec = float(evicted_event_count) * factor
        ev = evicted_evidence.to(
            dtype=self.score_num.dtype, device=self.score_num.device
        )
        self.score_num[layer].sub_(ev, alpha=remaining_ec)
        self.score_den[layer].sub_(remaining_ec)
        self.event_mass[layer].sub_(remaining_ec)

    @torch.no_grad()
    def _write_ring_slot(
        self,
        *,
        layer: int,
        step: int,
        evidence: torch.Tensor,
        event_count: float,
        current_step: int,
    ) -> None:
        """Write one evidence slot into the ring bank. If the chosen
        slot is non-empty, first subtract its remaining contribution
        from the running accumulators."""
        slots = self.bank_step[layer]
        empty = (slots == -1).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            slot = int(empty[0].item())
        else:
            # Evicting oldest — subtract its remaining contribution first.
            slot = int(slots.argmin().item())
            evicted_step = int(slots[slot].item())
            evicted_ev = self.bank_evidence[layer, slot].clone()
            evicted_cnt = float(self.bank_event_count[layer, slot].item())
            self._subtract_expired_contribution(
                layer=layer,
                evicted_step=evicted_step,
                current_step=current_step,
                evicted_evidence=evicted_ev,
                evicted_event_count=evicted_cnt,
            )
        self.bank_evidence[layer, slot] = evidence.to(
            dtype=self.bank_evidence.dtype, device=self.bank_evidence.device,
        )
        self.bank_step[layer, slot] = int(step)
        self.bank_event_count[layer, slot] = float(event_count)

    @torch.no_grad()
    def ingest_gpu(
        self,
        *,
        pressure: torch.Tensor,
        states_per_layer: list,
        horizon_H: int,
        event_frac: float,
    ) -> dict:
        """Phase 1 of two-phase ingest. Runs on the pressure/states
        device. Returns only small aggregates + scalar counts — no
        `[B, T]` masks in the payload, so the D2H copy stays tiny."""
        if len(states_per_layer) != self.num_layers:
            raise ValueError(
                f"states_per_layer must have {self.num_layers} entries; got {len(states_per_layer)}"
            )
        event_mask = compute_event_mask(pressure, event_frac=event_frac)  # [B, T]
        flat_mask = event_mask.reshape(-1)
        flat_non_event = ~flat_mask
        n_events = flat_mask.sum().to(torch.int64)
        n_non_events = flat_non_event.sum().to(torch.int64)
        aggregated_excess: list[torch.Tensor] = []
        non_event_mean_future_energy: list[torch.Tensor] = []
        event_count: list[torch.Tensor] = []
        for layer, states in enumerate(states_per_layer):
            if states.shape[-1] != self.dim:
                raise ValueError(
                    f"layer {layer}: states last dim {states.shape[-1]} != self.dim {self.dim}"
                )
            future_energy = compute_future_energy(states, horizon_H=horizon_H)
            flat_fe = future_energy.reshape(-1, self.dim)
            if flat_non_event.any():
                nonevt_mean = flat_fe[flat_non_event].mean(dim=0)
            else:
                nonevt_mean = torch.zeros(
                    self.dim, dtype=flat_fe.dtype, device=flat_fe.device
                )
            non_event_mean_future_energy.append(nonevt_mean)
            baseline = self.baseline_future_energy[layer].to(flat_fe.device)
            excess = (future_energy - baseline).clamp_min(0.0)
            flat_excess = excess.reshape(-1, self.dim)
            if flat_mask.any():
                agg = flat_excess[flat_mask].mean(dim=0)
                cnt = flat_mask.sum().to(torch.float32)
            else:
                agg = torch.zeros(
                    self.dim, dtype=flat_excess.dtype, device=flat_excess.device
                )
                cnt = torch.zeros((), dtype=torch.float32, device=flat_excess.device)
            aggregated_excess.append(agg)
            event_count.append(cnt)
        return {
            "aggregated_excess_per_layer": torch.stack(aggregated_excess, dim=0),
            "non_event_mean_future_energy_per_layer": torch.stack(
                non_event_mean_future_energy, dim=0
            ),
            "event_count_per_layer": torch.stack(event_count, dim=0),
            "n_events_scalar": n_events,
            "n_non_events_scalar": n_non_events,
        }

    @torch.no_grad()
    def ingest_cpu_from_prepared(self, *, step: int, prepared: dict) -> None:
        """CPU-side ingest from a pre-aggregated payload. Drives the
        incremental accumulators and writes to the ring bank for
        TTL/checkpoint state."""
        agg = prepared["aggregated_excess_per_layer"].to(
            device=self.bank_evidence.device, dtype=self.bank_evidence.dtype,
        )
        nonevt = prepared["non_event_mean_future_energy_per_layer"].to(
            device=self.baseline_future_energy.device,
            dtype=self.baseline_future_energy.dtype,
        )
        counts = prepared["event_count_per_layer"].to(
            device=self.bank_event_count.device,
            dtype=self.bank_event_count.dtype,
        )
        had_non_events = int(prepared["n_non_events_scalar"].item()) > 0
        decay = self.baseline_ema_decay
        # Advance accumulator decay to this step.
        self._step_decay_accumulators(current_step=step)
        for layer in range(self.num_layers):
            if had_non_events:
                obs = nonevt[layer]
                if not bool(self.baseline_initialized[layer].item()):
                    self.baseline_future_energy[layer].copy_(obs)
                    self.baseline_initialized[layer] = True
                else:
                    self.baseline_future_energy[layer].mul_(decay).add_(obs, alpha=(1.0 - decay))
            cnt = float(counts[layer].item())
            if cnt <= 0:
                continue
            # Incremental accumulator update.
            self._add_contribution(
                layer=layer,
                evidence=agg[layer],
                event_count=cnt,
            )
            # Ring-bank write with TTL-exact correction.
            self._write_ring_slot(
                layer=layer,
                step=step,
                evidence=agg[layer],
                event_count=cnt,
                current_step=step,
            )

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

        Thin wrapper: runs `ingest_gpu` (GPU-side aggregation) then
        `ingest_cpu_from_prepared` (CPU-side accumulator + ring-bank
        update). Keeping a single math path guarantees parity between
        the single-call and two-phase ingest routes.

        Args:
            step: current training step index.
            pressure: `[B, T]` pressure field (any real-valued tensor).
            states_per_layer: list of length `num_layers`, each entry a
                `[B, T, D]` captured states tensor.
            horizon_H: trailing window for post-event energy.
            event_frac: fraction of positions to mark as events.
        """
        prepared = self.ingest_gpu(
            pressure=pressure,
            states_per_layer=states_per_layer,
            horizon_H=horizon_H,
            event_frac=event_frac,
        )
        self.ingest_cpu_from_prepared(step=step, prepared=prepared)

    @torch.no_grad()
    def allocate_seats(self, *, current_step: int) -> None:
        """Recompute per-layer seat assignment from current age-weighted score.

        Gate: a layer's total age-weighted event count must exceed
        `min_weighted_events_per_layer`; otherwise its seats are cleared.
        """
        if self.fixed_random_seats:
            # Falsifier: seats were bound once at construction; never refresh.
            return
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
            if self.score_permute_before_topk:
                # Falsifier path: bypass score entirely and pick k channels
                # uniformly at random. NOTE: implementing this as "permute
                # score then top-k" would un-shuffle through the permutation
                # and still return the peak-score indices — don't do that.
                mask = torch.zeros(self.dim, dtype=torch.bool, device=self.seat_mask.device)
                perm = torch.randperm(self.dim, device=mask.device)
                mask[perm[:k]] = True
                self.seat_mask[layer] = mask
                continue
            topk = torch.topk(scores[layer], k=k, largest=True)
            mask = torch.zeros(self.dim, dtype=torch.bool, device=self.seat_mask.device)
            mask[topk.indices] = True
            self.seat_mask[layer] = mask

    @torch.no_grad()
    def allocate_seats_from_accumulators(self, *, current_step: int) -> None:
        """Hot-path seat allocator. Respects fixed_random_seats flag.
        Gates on event_mass. Honors score_permute_before_topk for
        falsifier cells."""
        if self.fixed_random_seats:
            return
        self._step_decay_accumulators(current_step=current_step)
        k = max(1, int(round(self.dim * self.criticality_budget_frac)))
        scores = self.score_from_accumulators()
        for layer in range(self.num_layers):
            if float(self.event_mass[layer].item()) < float(self.min_weighted_events_per_layer):
                self.seat_mask[layer].fill_(False)
                continue
            mask = torch.zeros(self.dim, dtype=torch.bool, device=self.seat_mask.device)
            if self.score_permute_before_topk:
                perm = torch.randperm(self.dim, device=self.seat_mask.device)
                mask[perm[:k]] = True
            else:
                topk = torch.topk(scores[layer], k=k, largest=True)
                mask[topk.indices] = True
            self.seat_mask[layer] = mask

    @torch.no_grad()
    def diagnostics_snapshot(
        self, *, log_a_per_layer: list, current_step: int,
    ) -> dict:
        """Per-layer diagnostic snapshot of CD's current state.

        Called by the runner at each seat refresh. Returns a dict with:
          - step: int
          - seat_churn_per_layer: fraction of seats changed since last snapshot
          - budget_occupancy_per_layer: fraction of SEATED channels with
            sigmoid-criticality >= 0.9 * critical_value
          - score_criticality_corr_per_layer: Spearman rank correlation
            of per-channel score vs criticality (0.0 if degenerate)
          - event_rate_per_layer: event_mass / max(1, populated_slots)
          - seat_mask_fraction_per_layer: fraction of D that is seated

        Churn is 0.0 per layer on the first snapshot; subsequent
        snapshots diff against the seat_mask cached from the previous
        call.
        """
        if len(log_a_per_layer) != self.num_layers:
            raise ValueError(
                f"log_a_per_layer must have {self.num_layers} entries; got {len(log_a_per_layer)}"
            )
        seat_churn: list[float] = []
        budget_occ: list[float] = []
        score_corr: list[float] = []
        event_rate: list[float] = []
        seat_frac: list[float] = []
        seat_indices: list[list[int]] = []
        score_p10: list[float] = []
        score_p50: list[float] = []
        score_p90: list[float] = []
        score_max: list[float] = []
        scores = self.score_from_accumulators()  # [L, D]
        current_mask = self.seat_mask
        for layer in range(self.num_layers):
            la = log_a_per_layer[layer].detach().to(dtype=torch.float32)
            crit = 1.0 - torch.sigmoid(la)  # [D]
            cur_lmask = current_mask[layer]  # [D] bool
            # Seat churn — fraction of SEATS that moved since last snapshot.
            # Normalize by the current seat count (k), not D, so it reads as
            # "of the k seats, what fraction are in new positions?" — every
            # seat that moved counts as 1 change; a single move flips two bits
            # in the mask (one seat leaving, one seat arriving), so we halve
            # the bit-diff count to recover the seat-count denominator.
            if self._previous_seat_mask_snapshot is None:
                churn = 0.0
            else:
                prev = self._previous_seat_mask_snapshot[layer]
                k = max(1, int(cur_lmask.sum().item()))
                bit_diffs = int((prev != cur_lmask).sum().item())
                churn = float(bit_diffs) / (2.0 * float(k))
            seat_churn.append(churn)
            # Budget occupancy: fraction of seated channels with criticality >= 0.9 * critical_value.
            k = int(cur_lmask.sum().item())
            if k == 0:
                budget_occ.append(0.0)
            else:
                threshold = 0.9 * self.critical_value
                seated_crit = crit[cur_lmask]
                budget_occ.append(
                    float((seated_crit >= threshold).float().mean().item())
                )
            # Rank correlation (Spearman) score vs criticality over all D.
            s = scores[layer].detach().to(dtype=torch.float32)
            if float(s.std().item()) < 1e-12 or float(crit.std().item()) < 1e-12:
                score_corr.append(0.0)
            else:
                rs = torch.argsort(torch.argsort(s)).to(torch.float32)
                rc = torch.argsort(torch.argsort(crit)).to(torch.float32)
                rs = (rs - rs.mean()) / rs.std().clamp_min(1e-12)
                rc = (rc - rc.mean()) / rc.std().clamp_min(1e-12)
                score_corr.append(float((rs * rc).mean().item()))
            # Event rate = event_mass / populated_slots.
            populated = int((self.bank_step[layer] != -1).sum().item())
            event_rate.append(
                float(self.event_mass[layer].item()) / max(1, populated)
            )
            # Seat mask fraction.
            seat_frac.append(float(cur_lmask.float().mean().item()))
            # Seat indices (sorted ascending) so post-hoc analysis can
            # compute seat overlap across seeds/arms.
            seat_indices.append(
                sorted(int(i) for i in cur_lmask.nonzero(as_tuple=True)[0].tolist())
            )
            # Score percentiles over all D channels for this layer. Tracks
            # whether score signal-to-noise is sharpening over training.
            s_layer = scores[layer].detach().to(dtype=torch.float32)
            q = torch.tensor([0.10, 0.50, 0.90], dtype=torch.float32)
            p10, p50, p90 = torch.quantile(s_layer, q).tolist()
            score_p10.append(float(p10))
            score_p50.append(float(p50))
            score_p90.append(float(p90))
            score_max.append(float(s_layer.max().item()))
        # Cache the current mask for next snapshot's churn calc.
        self._previous_seat_mask_snapshot = current_mask.clone()
        return {
            "step": int(current_step),
            "seat_churn_per_layer": seat_churn,
            "budget_occupancy_per_layer": budget_occ,
            "score_criticality_corr_per_layer": score_corr,
            "event_rate_per_layer": event_rate,
            "seat_mask_fraction_per_layer": seat_frac,
            "seat_indices_per_layer": seat_indices,
            "score_p10_per_layer": score_p10,
            "score_p50_per_layer": score_p50,
            "score_p90_per_layer": score_p90,
            "score_max_per_layer": score_max,
        }

    def criticality_loss(self, log_a_per_layer: list) -> torch.Tensor:
        """Seat-masked MSE loss pulling `1 - sigmoid(log_a[seat])` toward
        `critical_value`.

        Non-seat channels contribute exactly zero to the loss (and therefore
        exactly zero gradient to their log_a).

        Returns:
            Scalar tensor, already multiplied by
            `criticality_distill_weight` — the runner just adds it to the
            CE loss (no external multiply).
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
        return total * self.criticality_distill_weight


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
    positive = pressure > 0.0
    n_positive = int(positive.sum().item())
    if n_positive == 0:
        return torch.zeros_like(pressure, dtype=torch.bool)
    k = min(k, n_positive)
    flat = pressure.reshape(-1)
    _, idx = torch.topk(flat, k=k, largest=True)
    mask = torch.zeros(total, dtype=torch.bool, device=pressure.device)
    mask[idx] = True
    return mask.reshape(pressure.shape) & positive


def compute_future_energy(states: torch.Tensor, horizon_H: int) -> torch.Tensor:
    """Per-position mean-square energy over the trailing window `[t+1, t+H]`.

    Vectorized — no Python loop over T.

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
    zero_pad = torch.zeros(B, 1, D, dtype=sq.dtype, device=sq.device)
    csum = torch.cat([zero_pad, sq.cumsum(dim=1)], dim=1)  # [B, T+1, D]
    t = torch.arange(T, device=sq.device)
    t_start = t + 1
    t_end_excl = torch.clamp(t + 1 + horizon_H, max=T)
    valid = t_start < t_end_excl
    safe_start = torch.where(valid, t_start, torch.zeros_like(t_start))
    safe_end = torch.where(valid, t_end_excl, torch.zeros_like(t_end_excl))
    sum_energy = csum[:, safe_end, :] - csum[:, safe_start, :]
    count = torch.where(
        valid,
        (t_end_excl - t_start).to(torch.float32),
        torch.zeros_like(t_start, dtype=torch.float32),
    )
    safe_count = count.clamp_min(1.0).view(1, T, 1)
    out = sum_energy / safe_count.to(sum_energy.dtype)
    return torch.where(valid.view(1, T, 1), out, torch.zeros_like(out))
