"""Adaptive Residual Memory control plane for CRCT rank-3.

The episodic sidecar is a living residual layer over the SSM, not a
passive cache. Each memory slot is continuously scored relative to
the current model via counterfactual probes and classified into one
of six actions: PRESERVE, DECAY, EVICT, REFRESH, QUARANTINE, DISTILL.

The probe engine consumes a rolling stream of probe frames and bounded
slot-work microbatches. Each microbatch runs 1 SSM encode + vectorized
masked NLL over the selected hide-one variants, then a smaller oracle
confirmation pass measures active mutations under real SSM dynamics.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .cache_utility import chunked_nll_from_hidden
from .slot_table import SlotTable, SlotRecord, SlotId
from .slot_table import SLOT_WARMING, SLOT_ACTIVE, SLOT_SHARP
from .slot_table import SLOT_DECAYING, SLOT_QUARANTINED, SLOT_RETIRED

__all__ = [
    "ReplayEvictionLoop",
    "TickResult",
    "counterfactual_probe",
    "oracle_confirm_slots",
    "replay_score_slots",
    "MaintenancePolicy",
    "ProbeFrame",
    "_evict_slots",
]

SLOT_PRESERVE = "PRESERVE"
SLOT_DECAY = "DECAY"
SLOT_EVICT = "EVICT"
SLOT_REFRESH = "REFRESH"
SLOT_QUARANTINE = "QUARANTINE"
SLOT_DISTILL = "DISTILL"


@dataclass
class TickResult:
    evicted: list[int] = field(default_factory=list)
    refreshed: list[int] = field(default_factory=list)
    quarantined: list[int] = field(default_factory=list)
    released: list[int] = field(default_factory=list)
    distilled: list[int] = field(default_factory=list)
    decayed: list[int] = field(default_factory=list)

    @property
    def evicted_indices(self) -> list[int]:
        return sorted(set(self.evicted) | set(self.distilled))


@dataclass
class CounterfactualResult:
    marginal_gains: torch.Tensor
    sidecar_value: torch.Tensor
    nll_baseline: torch.Tensor
    nll_no_sidecar: torch.Tensor
    weights_baseline: torch.Tensor
    mask: torch.Tensor
    slot_indices: list[int]


@dataclass
class OracleConfirmationResult:
    """Real-physics batched confirmation for a small candidate set."""

    slot_indices: list[int]
    oracle_deltas: torch.Tensor
    nll_baseline: torch.Tensor
    nll_no_sidecar: torch.Tensor
    mask: torch.Tensor


@dataclass
class DistillReceipt:
    slot_id: int
    step: int
    marginal_gain_before: float
    marginal_gain_peak: float
    marginal_gain_current: float
    target: str
    accepted: bool = False


@dataclass
class MemoryEvent:
    step: int
    tick: int
    slot_id: int
    action: str
    marginal_gain: float = 0.0
    sharpness: float = 0.0
    activation_drift: float = 0.0
    representation_drift: float = 0.0
    semantic_drift: float = 0.0
    contradiction: float = 0.0
    retrieval_mass: float = 0.0
    peak_utility: float = 0.0
    peak_sharpness: float = 0.0
    score_count: int = 0
    accepted: bool = True
    reason: str = ""
    refresh_candidate: str = ""


@dataclass
class ProbeFrame:
    """One replay-maintenance frame in the rank-3 stream."""

    frame_id: int
    step: int
    input_ids: torch.Tensor
    valid_mask: torch.Tensor
    cue: torch.Tensor | None
    cache_read_cutoff: int | None
    stream_id: int
    ingested_at: float
    processed_slots: set[int] = field(default_factory=set)


@torch.inference_mode()
def counterfactual_probe(
    *,
    model: Any,
    outer: Any,
    probe_input_ids: torch.Tensor,
    probe_valid_mask: torch.Tensor,
    probe_cue: torch.Tensor | None = None,
    cache_read_cutoff: int | None = None,
    chunk_size: int = 16,
    score_slot_indices: list[int] | None = None,
) -> CounterfactualResult:
    """Score selected slots via vectorized counterfactual masking.

    1 SSM encode + batched masked NLL. ``score_slot_indices`` bounds
    maintenance work without touching the trunk path; omitted means score
    every visible slot for compatibility with older tests/helpers.
    """
    x = probe_input_ids[:, :-1]
    y = probe_input_ids[:, 1:]
    mask = probe_valid_mask[:, 1:].bool()
    B, T = x.shape

    dev = x.device
    dt = torch.bfloat16
    ac = torch.autocast(dev.type, dtype=dt) if dev.type == "cuda" else torch.autocast("cpu", dtype=dt)

    # Phase 1: SSM encode once (the only expensive op)
    with ac:
        h_base = model.encode(x, memory_mode="off", cache_read_cutoff=cache_read_cutoff)

    D = h_base.shape[-1]

    # Check for slots
    table = getattr(outer, "table", None)
    has_table = table is not None and len(table) > 0

    if not has_table:
        slots_list = getattr(outer, "_slots", [])
        if not slots_list:
            return CounterfactualResult(
                marginal_gains=torch.zeros(0, B, T, device=dev),
                sidecar_value=torch.zeros(B, T, device=dev),
                nll_baseline=torch.zeros(B, T, device=dev),
                nll_no_sidecar=torch.zeros(B, T, device=dev),
                weights_baseline=torch.zeros(B, 0, device=dev),
                mask=mask,
                slot_indices=[],
            )

    # Phase 2: Gather slot data
    if has_table:
        vis = table.visible_indices(read_cutoff=cache_read_cutoff)
        if not vis:
            return CounterfactualResult(
                marginal_gains=torch.zeros(0, B, T, device=dev),
                sidecar_value=torch.zeros(B, T, device=dev),
                nll_baseline=torch.zeros(B, T, device=dev),
                nll_no_sidecar=torch.zeros(B, T, device=dev),
                weights_baseline=torch.zeros(B, 0, device=dev),
                mask=mask,
                slot_indices=[],
            )
        slot_mat = table.slot_matrix(vis).to(dev)
    else:
        vis = list(range(len(slots_list)))
        slot_mat = torch.cat(slots_list, dim=0).to(dev)

    N = slot_mat.shape[0]
    if score_slot_indices is None:
        score_local = list(range(N))
        score_indices = list(vis)
    else:
        visible_pos = {int(phys): local for local, phys in enumerate(vis)}
        score_pairs = [
            (visible_pos[int(phys)], int(phys))
            for phys in dict.fromkeys(score_slot_indices)
            if int(phys) in visible_pos
        ]
        if not score_pairs:
            return CounterfactualResult(
                marginal_gains=torch.zeros(0, B, T, device=dev),
                sidecar_value=torch.zeros(B, T, device=dev),
                nll_baseline=torch.zeros(B, T, device=dev),
                nll_no_sidecar=torch.zeros(B, T, device=dev),
                weights_baseline=torch.zeros(B, 0, device=dev),
                mask=mask,
                slot_indices=[],
            )
        score_local = [local for local, _phys in score_pairs]
        score_indices = [phys for _local, phys in score_pairs]
    score_local_t = torch.tensor(score_local, device=dev, dtype=torch.long)

    # Phase 3: Cue + similarity
    cue_proj = getattr(outer, "cue_proj", None)
    decoder = getattr(outer, "decoder", None)
    if cue_proj is None or decoder is None:
        return CounterfactualResult(
            marginal_gains=torch.zeros(0, B, T, device=dev),
            sidecar_value=torch.zeros(B, T, device=dev),
            nll_baseline=torch.zeros(B, T, device=dev),
            nll_no_sidecar=torch.zeros(B, T, device=dev),
            weights_baseline=torch.zeros(B, len(score_indices), device=dev),
            mask=mask,
            slot_indices=score_indices,
        )

    with ac:
        cue = probe_cue.to(device=dev) if probe_cue is not None else h_base.detach().mean(dim=1)
        cue_outer = cue_proj(cue.to(dtype=cue_proj.weight.dtype))
        slot_mat_work = slot_mat.to(dtype=cue_outer.dtype)
        sim = cue_outer @ slot_mat_work.T
        weights_baseline = F.softmax(sim, dim=-1)
        retrieved_baseline = weights_baseline @ slot_mat_work

    # Phase 4: baseline + sidecar_off + selected hide-one variants.
    # Hide-one softmax can be computed exactly by renormalizing the
    # baseline weights after removing a slot:
    #   r_hide_i = (r_base - w_i * slot_i) / (1 - w_i)
    # This avoids a masked softmax+bmm over all slots for every variant.
    N_var = len(score_local) + 2

    # Phase 5: Vectorized probe in chunks
    nll_all = torch.empty(N_var, B, T, device=dev, dtype=torch.float32)

    cs = chunk_size if chunk_size > 0 else N_var
    for start in range(0, N_var, cs):
        end = min(start + cs, N_var)
        C = end - start

        with ac:
            retrieved = torch.zeros(
                C, B, slot_mat_work.shape[-1],
                device=dev,
                dtype=slot_mat_work.dtype,
            )
            if start == 0:
                retrieved[0] = retrieved_baseline

            hide_start = max(start, 2)
            if hide_start < end:
                local_start = hide_start - start
                hide_offsets = torch.arange(hide_start - 2, end - 2, device=dev)
                hide_slots = score_local_t[hide_offsets]
                w_i = weights_baseline[:, hide_slots].T.unsqueeze(-1)
                slot_i = slot_mat_work[hide_slots].unsqueeze(1)
                denom = 1.0 - w_i
                hide_retrieved = torch.where(
                    denom > 1e-7,
                    (retrieved_baseline.unsqueeze(0) - w_i * slot_i)
                    / denom.clamp_min(1e-7),
                    torch.zeros_like(slot_i).expand(-1, B, -1),
                )
                retrieved[local_start:] = hide_retrieved

            biases = F.linear(retrieved, decoder.weight.to(retrieved.dtype))
            h_variants = h_base.unsqueeze(0).expand(C, -1, -1, -1) + biases.unsqueeze(2)

        h_flat = h_variants.reshape(C * B, T, D).float()
        y_flat = y.unsqueeze(0).expand(C, -1, -1).reshape(C * B, -1)
        nll_chunk = chunked_nll_from_hidden(model, h_flat, y_flat)
        nll_shaped = nll_chunk.reshape(C, B, T)
        nll_all[start:end] = nll_shaped

    nll_baseline = nll_all[0]
    nll_no_sidecar = nll_all[1]
    marginal_gains = nll_all[2:] - nll_baseline.unsqueeze(0)
    sidecar_value = nll_no_sidecar - nll_baseline

    return CounterfactualResult(
        marginal_gains=marginal_gains.cpu(),
        sidecar_value=sidecar_value.cpu(),
        nll_baseline=nll_baseline.cpu(),
        nll_no_sidecar=nll_no_sidecar.cpu(),
        weights_baseline=weights_baseline[:, score_local_t].cpu(),
        mask=mask.cpu(),
        slot_indices=score_indices,
    )


@torch.inference_mode()
def oracle_confirm_slots(
    *,
    model: Any,
    outer: Any,
    probe_input_ids: torch.Tensor,
    probe_valid_mask: torch.Tensor,
    slot_indices: list[int],
    cache_read_cutoff: int | None = None,
) -> OracleConfirmationResult:
    """Confirm candidate slots with real memory-injected SSM dynamics.

    This is the "ripple" pass: one expanded batch evaluates baseline,
    no-sidecar, and hide-one variants using ``model.encode(...,
    memory_mode='force_on')``.  The cheap counterfactual map proposes
    candidate slots; this function measures their effect under the real
    pre-SSM injection path.
    """
    x = probe_input_ids[:, :-1]
    y = probe_input_ids[:, 1:]
    mask = probe_valid_mask[:, 1:].bool()
    B, T = x.shape
    dev = x.device

    table = getattr(outer, "table", None)
    n_slots = len(table) if table is not None else len(getattr(outer, "_slots", []))
    candidates = [
        int(i)
        for i in dict.fromkeys(slot_indices)
        if 0 <= int(i) < int(n_slots)
    ]
    if n_slots <= 0 or not candidates:
        empty = torch.zeros(0, B, T, device=dev)
        return OracleConfirmationResult(
            slot_indices=[],
            oracle_deltas=empty,
            nll_baseline=torch.zeros(B, T, device=dev),
            nll_no_sidecar=torch.zeros(B, T, device=dev),
            mask=mask,
        )

    variants = len(candidates) + 2
    slot_masks = torch.ones(variants, n_slots, device=dev, dtype=torch.bool)
    slot_masks[1, :] = False
    rows = torch.arange(2, variants, device=dev)
    cols = torch.tensor(candidates, device=dev, dtype=torch.long)
    slot_masks[rows, cols] = False

    x_exp = x.unsqueeze(0).expand(variants, -1, -1).reshape(variants * B, T)
    y_exp = y.unsqueeze(0).expand(variants, -1, -1).reshape(variants * B, T)
    slot_masks_exp = (
        slot_masks[:, None, :]
        .expand(variants, B, n_slots)
        .reshape(variants * B, n_slots)
    )

    with torch.autocast(dev.type, dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = model.encode(
            x_exp,
            memory_mode="force_on",
            cache_read_cutoff=cache_read_cutoff,
            memory_slot_mask=slot_masks_exp,
        )
    nll = chunked_nll_from_hidden(model, hidden, y_exp).reshape(variants, B, T)
    nll_baseline = nll[0]
    nll_no_sidecar = nll[1]
    oracle_deltas = nll[2:] - nll_baseline.unsqueeze(0)
    return OracleConfirmationResult(
        slot_indices=candidates,
        oracle_deltas=oracle_deltas.cpu(),
        nll_baseline=nll_baseline.cpu(),
        nll_no_sidecar=nll_no_sidecar.cpu(),
        mask=mask.cpu(),
    )


@torch.inference_mode()
def replay_score_slots(
    *,
    model: Any,
    probe_input_ids: torch.Tensor,
    probe_valid_mask: torch.Tensor,
    cache_read_cutoff: int | None = None,
) -> dict[str, Any]:
    """Backward-compatible scoring via counterfactual probe."""
    outer = getattr(model, "outer_model", None)
    if outer is None:
        return {"slot_utilities": torch.tensor([]), "slot_indices": [], "mean_utility": 0.0}

    result = counterfactual_probe(
        model=model,
        outer=outer,
        probe_input_ids=probe_input_ids,
        probe_valid_mask=probe_valid_mask,
        cache_read_cutoff=cache_read_cutoff,
    )

    if len(result.slot_indices) == 0:
        return {"slot_utilities": torch.tensor([]), "slot_indices": [], "mean_utility": 0.0}

    slot_utils = result.marginal_gains[:, :, :].float()
    m = result.mask.float()
    per_slot = (slot_utils * m.unsqueeze(0)).sum(dim=(1, 2)) / m.sum().clamp(min=1)

    mean_util = float(result.sidecar_value[result.mask].mean().item()) if result.mask.any() else 0.0

    return {
        "slot_utilities": per_slot,
        "slot_indices": list(result.slot_indices),
        "mean_utility": mean_util,
        "n_tokens_scored": int(result.mask.sum().item()),
        "retrieval_weights": result.weights_baseline,
        "per_token_utility": result.sidecar_value,
        "mask": result.mask,
    }


def _compute_per_slot_sharpness(
    marginal_gains: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-slot sharpness: variance of token-level marginal gain."""
    if marginal_gains.shape[0] == 0:
        return torch.zeros(0)
    N = marginal_gains.shape[0]
    m = mask.float()
    masked = marginal_gains * m.unsqueeze(0)
    n_valid = m.sum().clamp(min=1)

    means = masked.sum(dim=(1, 2)) / n_valid
    mean_sq = ((marginal_gains ** 2) * m.unsqueeze(0)).sum(dim=(1, 2)) / n_valid
    variance = (mean_sq - means ** 2).clamp_min(0)
    return variance


def _compute_representation_drift(
    outer: Any, slot_tensor: torch.Tensor
) -> float:
    """How cleanly a slot round-trips through decoder→encoder."""
    encoder = getattr(outer, "encoder", None)
    decoder = getattr(outer, "decoder", None)
    if encoder is None or decoder is None:
        return 0.0
    with torch.inference_mode():
        decoded = decoder(slot_tensor.to(dtype=decoder.weight.dtype))
        re_encoded = torch.tanh(encoder(decoded.to(dtype=encoder.weight.dtype)))
    cos = F.cosine_similarity(
        slot_tensor.float().flatten(),
        re_encoded.float().flatten(),
        dim=0,
    )
    return float(1.0 - cos.item())


class MaintenancePolicy:
    """Policy-shaped action selector. Deterministic rules now, swappable later."""

    def action_values(
        self,
        rec: SlotRecord,
        *,
        eviction_threshold: float,
        useful_threshold: float,
        drift_threshold: float,
        quarantine_threshold: float,
        distill_peak_threshold: float,
        peak_preserve_utility_threshold: float,
        peak_preserve_sharpness_threshold: float,
        min_age: int,
        min_score_count: int,
        capacity_pressure: bool = False,
    ) -> dict[str, float]:
        enough = rec.score_count >= min_score_count
        old = (
            (rec.last_scored_step - rec.created_step) >= min_age
            if rec.last_scored_step > 0
            else False
        )
        converged = enough and old
        peak_preserve = (
            rec.peak_utility >= peak_preserve_utility_threshold
            or rec.peak_sharpness >= peak_preserve_sharpness_threshold
        )

        quarantine_v = 0.0
        if converged and rec.contradiction_ema > abs(quarantine_threshold):
            quarantine_v = rec.contradiction_ema + 0.5 * rec.negative_streak

        distill_v = 0.0
        if (
            converged
            and not peak_preserve
            and rec.peak_utility > distill_peak_threshold
        ):
            internalization = 1.0 - rec.marginal_gain_ema / max(rec.peak_utility, 1e-6)
            if internalization > 0.5 and rec.marginal_gain_ema < 2 * eviction_threshold:
                distill_v = internalization * rec.peak_utility

        evict_v = 0.0
        if (
            capacity_pressure
            and converged
            and not peak_preserve
            and rec.utility_ema < eviction_threshold
        ):
            evict_v = (
                eviction_threshold - rec.utility_ema
                + max(0.0, useful_threshold - rec.marginal_gain_ema)
            )

        refresh_v = 0.0
        drift_max = max(rec.activation_drift_ema, rec.representation_drift_ema, rec.semantic_drift_ema)
        if drift_max > drift_threshold and rec.marginal_gain_ema > useful_threshold:
            refresh_v = 0.8 * drift_max + 0.6 * rec.marginal_gain_ema

        preserve_v = (
            rec.marginal_gain_ema
            + 0.7 * rec.sharpness_ema
            + 0.25 * rec.peak_utility
            + 0.25 * rec.peak_sharpness
        )

        decay_v = 0.0
        if not capacity_pressure and rec.utility_ema < 2 * eviction_threshold:
            decay_v = 2 * eviction_threshold - rec.utility_ema

        return {
            SLOT_QUARANTINE: quarantine_v,
            SLOT_DISTILL: distill_v,
            SLOT_EVICT: evict_v,
            SLOT_REFRESH: refresh_v,
            SLOT_PRESERVE: preserve_v,
            SLOT_DECAY: decay_v,
        }

    def choose(self, rec: SlotRecord, **kwargs: Any) -> str:
        vals = self.action_values(rec, **kwargs)
        return max(vals, key=lambda k: vals[k])


class ReplayEvictionLoop:
    """Adaptive Residual Memory control plane.

    Runs on rank 3 in idle time. Each tick runs a counterfactual probe,
    decomposes signals, classifies slots via policy, executes actions.
    """

    def __init__(
        self,
        *,
        action_mode: str = "active",
        memory_streams: int = 8,
        eviction_threshold: float = 0.01,
        eviction_ema_beta: float = 0.9,
        min_slot_age_steps: int = 128,
        max_seconds_per_tick: float = 0.5,
        trace_path: str | None = None,
        trace_max_rows: int = 0,
        probe_chunk_size: int = 16,
        oracle_confirm_top_k: int = 32,
        drift_threshold: float = 0.3,
        repr_drift_threshold: float = 0.2,
        refresh_lr: float = 0.1,
        refresh_margin: float = 0.001,
        quarantine_threshold: float = -0.01,
        max_quarantined: int = 8,
        quarantine_release_streak: int = 2,
        distill_peak_threshold: float = 0.04,
        peak_preserve_utility_threshold: float = 0.20,
        peak_preserve_sharpness_threshold: float = 0.20,
        useful_threshold: float = 0.005,
        min_score_count: int = 2,
        probe_buffer_size: int = 32,
        frame_ttl_steps: int = 256,
        slot_work_chunk_size: int = 64,
        action_agreement_count: int = 2,
    ) -> None:
        if action_mode not in {"active", "shadow"}:
            raise ValueError("action_mode must be 'active' or 'shadow'")
        self._action_mode = str(action_mode)
        self._memory_streams = max(1, int(memory_streams))
        self._threshold = float(eviction_threshold)
        self._ema_beta = float(eviction_ema_beta)
        self._min_age = int(min_slot_age_steps)
        self._max_seconds = float(max_seconds_per_tick)
        self._probe_chunk_size = int(probe_chunk_size)
        self._oracle_confirm_top_k = max(0, int(oracle_confirm_top_k))
        self._drift_threshold = float(drift_threshold)
        self._repr_drift_threshold = float(repr_drift_threshold)
        self._refresh_lr = float(refresh_lr)
        self._refresh_margin = float(refresh_margin)
        self._quarantine_threshold = float(quarantine_threshold)
        self._max_quarantined = int(max_quarantined)
        self._quarantine_release_streak = int(quarantine_release_streak)
        self._distill_peak_threshold = float(distill_peak_threshold)
        self._peak_preserve_utility_threshold = float(peak_preserve_utility_threshold)
        self._peak_preserve_sharpness_threshold = float(peak_preserve_sharpness_threshold)
        self._useful_threshold = float(useful_threshold)
        self._min_score_count = int(min_score_count)
        self._probe_buffer_size = max(1, int(probe_buffer_size))
        self._frame_ttl_steps = max(0, int(frame_ttl_steps))
        self._slot_work_chunk_size = max(1, int(slot_work_chunk_size))
        self._action_agreement_count = max(1, int(action_agreement_count))

        # Legacy per-index tracking (kept for backward compat in tests)
        self._slot_utility_ema: dict[int, float] = {}
        self._slot_first_seen_step: dict[int, int] = {}
        self._slot_score_count: dict[int, int] = {}

        # Probe cache
        self._probe_input_ids: torch.Tensor | None = None
        self._probe_valid_mask: torch.Tensor | None = None
        self._probe_cue: torch.Tensor | None = None
        self._probe_cache_cutoff: int | None = None
        self._probe_step: int = 0
        self._probe_stream_id: int = 0
        self._last_ingested_probe_step: int = -1
        self._probe_frames: deque[ProbeFrame] = deque(maxlen=self._probe_buffer_size)
        self._next_frame_id: int = 0
        self._slot_rr_cursor: int = 0
        self._active_frame_id: int = -1
        self._active_frame_step: int = 0
        self._started_at: float = time.monotonic()

        # Counters
        self._tick_count: int = 0
        self._evictions_total: int = 0
        self._refreshes_total: int = 0
        self._distills_total: int = 0
        self._decays_total: int = 0
        self._releases_total: int = 0
        self._replays_total: int = 0
        self._slots_scored_total: int = 0
        self._last_slots_tracked: int = 0
        self._shadow_actions_total: int = 0
        self._shadow_action_counts: dict[str, int] = {}
        self._oracle_confirmations_total: int = 0
        self._oracle_confirmed_actions_total: int = 0
        self._oracle_rejected_actions_total: int = 0
        self._proxy_oracle_sign_matches: int = 0
        self._proxy_oracle_pairs_total: int = 0
        self._proxy_oracle_abs_error_sum: float = 0.0
        self._last_proxy_oracle_sign_match_rate: float = 0.0
        self._last_proxy_oracle_abs_error: float = 0.0
        self._last_oracle_candidates: int = 0
        self._probe_seconds_total: float = 0.0
        self._last_probe_seconds: float = 0.0
        self._probe_over_budget_total: int = 0
        self._probe_frames_ingested: int = 0
        self._probe_frames_dropped_overflow: int = 0
        self._probe_frames_dropped_stale: int = 0
        self._probe_frames_completed: int = 0
        self._probe_ticks_skipped_no_frame: int = 0
        self._probe_ticks_skipped_no_slot_work: int = 0
        self._slot_work_items_total: int = 0
        self._last_slot_work_items: int = 0
        self._action_agreements_total: int = 0
        self._action_agreements_reset_total: int = 0
        self._last_queue_depth: int = 0
        self._queue_depth_sum: int = 0
        self._queue_depth_samples: int = 0
        self._queue_depth_max: int = 0
        self._last_frame_age_steps: int = 0
        self._frame_age_steps_sum: int = 0
        self._frame_age_steps_max: int = 0
        self._last_frame_age_seconds: float = 0.0
        self._frame_age_seconds_sum: float = 0.0
        self._frame_age_seconds_max: float = 0.0
        self._stream_ticks: dict[int, int] = {i: 0 for i in range(self._memory_streams)}
        self._stream_work_items: dict[int, int] = {i: 0 for i in range(self._memory_streams)}
        self._stream_probe_seconds: dict[int, float] = {i: 0.0 for i in range(self._memory_streams)}
        self._last_stage_seconds: dict[str, float] = {
            "select": 0.0,
            "probe": 0.0,
            "ema": 0.0,
            "oracle": 0.0,
            "action": 0.0,
        }
        self._stage_seconds_total: dict[str, float] = dict(self._last_stage_seconds)
        self._oracle_seconds_total: float = 0.0
        self._last_oracle_seconds: float = 0.0
        self._slot_last_scored_step: dict[int, int] = {}
        self._last_visible_slots: int = 0
        self._last_untouched_slots: int = 0
        self._last_max_untouched_steps: int = 0

        # Quarantine tracking (by slot_id when using SlotTable, by index otherwise)
        self._quarantined: set[int] = set()
        self._quarantine_positive_streak: dict[int, int] = {}
        self._action_evidence: dict[int, tuple[str, int, int, int]] = {}

        # Trace
        self._trace_path = None if trace_path in (None, "") else Path(str(trace_path))
        self._trace_max_rows = max(0, int(trace_max_rows))
        self._trace_rows_written = 0
        self._trace_buffer: list[str] = []

        # Policy
        self._policy = MaintenancePolicy()

    def cache_probe(
        self,
        *,
        input_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        cue: torch.Tensor | None = None,
        cache_read_cutoff: int | None,
        step: int,
        stream_id: int = 0,
    ) -> None:
        input_ids_d = input_ids.detach()
        valid_mask_d = valid_mask.detach()
        cue_d = None if cue is None else cue.detach()
        if len(self._probe_frames) == self._probe_frames.maxlen:
            self._probe_frames_dropped_overflow += 1
        frame = ProbeFrame(
            frame_id=self._next_frame_id,
            step=int(step),
            input_ids=input_ids_d,
            valid_mask=valid_mask_d,
            cue=cue_d,
            cache_read_cutoff=cache_read_cutoff,
            stream_id=int(stream_id) % self._memory_streams,
            ingested_at=time.monotonic(),
        )
        self._next_frame_id += 1
        self._probe_frames.append(frame)
        self._probe_frames_ingested += 1
        self._last_queue_depth = len(self._probe_frames)
        self._queue_depth_sum += self._last_queue_depth
        self._queue_depth_samples += 1
        self._queue_depth_max = max(self._queue_depth_max, self._last_queue_depth)

        self._probe_input_ids = input_ids_d
        self._probe_valid_mask = valid_mask_d
        self._probe_cue = cue_d
        self._probe_cache_cutoff = cache_read_cutoff
        self._probe_step = int(step)
        self._last_ingested_probe_step = int(step)
        self._probe_stream_id = frame.stream_id
        self._trace_frame_event("frame_ingest", frame, queue_depth=len(self._probe_frames))

    def has_probe(self) -> bool:
        return bool(self._probe_frames)

    def _drop_stale_frames(self, step: int) -> None:
        if self._frame_ttl_steps <= 0:
            return
        while self._probe_frames and (int(step) - self._probe_frames[0].step) > self._frame_ttl_steps:
            self._probe_frames.popleft()
            self._probe_frames_dropped_stale += 1

    def _visible_slot_indices(self, outer: Any) -> list[int]:
        table = getattr(outer, "table", None)
        if table is not None:
            return table.visible_indices(read_cutoff=None)
        return list(range(len(getattr(outer, "_slots", []))))

    def _select_frame_and_slot_work(
        self,
        *,
        outer: Any,
        step: int,
    ) -> tuple[ProbeFrame | None, list[int]]:
        self._drop_stale_frames(step)
        visible = self._visible_slot_indices(outer)
        if not visible:
            return None, []
        for frame in list(self._probe_frames):
            remaining = [idx for idx in visible if idx not in frame.processed_slots]
            if not remaining:
                continue
            start = self._slot_rr_cursor % len(remaining)
            ordered = remaining[start:] + remaining[:start]
            work = ordered[: self._slot_work_chunk_size]
            self._slot_rr_cursor = (self._slot_rr_cursor + len(work)) % max(1, len(remaining))
            return frame, work
        return None, []

    def _mark_frame_slots_processed(
        self,
        *,
        frame: ProbeFrame,
        outer: Any,
        slot_indices: list[int],
    ) -> None:
        visible = self._visible_slot_indices(outer)
        visible_set = set(int(i) for i in visible)
        frame.processed_slots.update(int(i) for i in slot_indices)
        frame.processed_slots.intersection_update(visible_set)
        if visible_set and len(frame.processed_slots) >= len(visible_set):
            self._probe_frames_completed += 1
            try:
                self._probe_frames.remove(frame)
            except ValueError:
                pass

    def _capacity_pressure(self, outer: Any) -> bool:
        table = getattr(outer, "table", None)
        max_slots = int(getattr(outer, "max_slots", 0) or 0)
        if table is None or max_slots <= 0:
            return False
        return len(table) >= max_slots

    def tick(self, *, model: Any, step: int) -> TickResult:
        """One maintenance tick. Probe, classify, act."""
        t0 = time.monotonic()
        self._tick_count += 1

        outer = getattr(model, "outer_model", None)
        if outer is None:
            return TickResult()
        select_t0 = time.monotonic()
        frame, slot_work = self._select_frame_and_slot_work(outer=outer, step=step)
        self._last_stage_seconds["select"] = time.monotonic() - select_t0
        self._stage_seconds_total["select"] += self._last_stage_seconds["select"]
        self._last_queue_depth = len(self._probe_frames)
        self._queue_depth_sum += self._last_queue_depth
        self._queue_depth_samples += 1
        self._queue_depth_max = max(self._queue_depth_max, self._last_queue_depth)
        if frame is None:
            self._probe_ticks_skipped_no_frame += 1
            return TickResult()
        if not slot_work:
            self._probe_ticks_skipped_no_slot_work += 1
            return TickResult()
        frame_age_steps = max(0, int(step) - int(frame.step))
        frame_age_seconds = max(0.0, time.monotonic() - frame.ingested_at)
        self._last_frame_age_steps = frame_age_steps
        self._frame_age_steps_sum += frame_age_steps
        self._frame_age_steps_max = max(self._frame_age_steps_max, frame_age_steps)
        self._last_frame_age_seconds = frame_age_seconds
        self._frame_age_seconds_sum += frame_age_seconds
        self._frame_age_seconds_max = max(self._frame_age_seconds_max, frame_age_seconds)
        self._active_frame_id = frame.frame_id
        self._active_frame_step = frame.step
        self._probe_input_ids = frame.input_ids
        self._probe_valid_mask = frame.valid_mask
        self._probe_cue = frame.cue
        self._probe_cache_cutoff = frame.cache_read_cutoff
        self._probe_step = frame.step
        self._probe_stream_id = frame.stream_id
        self._last_slot_work_items = len(slot_work)
        self._slot_work_items_total += len(slot_work)
        self._stream_ticks[frame.stream_id] = self._stream_ticks.get(frame.stream_id, 0) + 1
        self._stream_work_items[frame.stream_id] = (
            self._stream_work_items.get(frame.stream_id, 0) + len(slot_work)
        )
        self._trace_frame_event(
            "frame_dispatch",
            frame,
            queue_depth=len(self._probe_frames),
            slot_work_items=len(slot_work),
            frame_age_steps=frame_age_steps,
            frame_age_seconds=frame_age_seconds,
        )

        # Run counterfactual probe
        probe_t0 = time.monotonic()
        cf = counterfactual_probe(
            model=model,
            outer=outer,
            probe_input_ids=frame.input_ids,
            probe_valid_mask=frame.valid_mask,
            probe_cue=frame.cue,
            cache_read_cutoff=frame.cache_read_cutoff,
            chunk_size=self._probe_chunk_size,
            score_slot_indices=slot_work,
        )
        self._last_probe_seconds = time.monotonic() - probe_t0
        self._probe_seconds_total += self._last_probe_seconds
        self._last_stage_seconds["probe"] = self._last_probe_seconds
        self._stage_seconds_total["probe"] += self._last_probe_seconds
        self._stream_probe_seconds[frame.stream_id] = (
            self._stream_probe_seconds.get(frame.stream_id, 0.0)
            + self._last_probe_seconds
        )
        if self._last_probe_seconds > self._max_seconds:
            self._probe_over_budget_total += 1

        if len(cf.slot_indices) == 0:
            return TickResult()
        self._mark_frame_slots_processed(frame=frame, outer=outer, slot_indices=cf.slot_indices)
        for phys_idx in cf.slot_indices:
            self._slot_last_scored_step[int(phys_idx)] = int(step)
        self._last_visible_slots = len(self._visible_slot_indices(outer))
        untouched_ages = [
            max(0, int(step) - self._slot_last_scored_step.get(int(phys_idx), int(step)))
            for phys_idx in self._visible_slot_indices(outer)
        ]
        self._last_untouched_slots = sum(
            1 for age in untouched_ages if age > self._frame_ttl_steps
        )
        self._last_max_untouched_steps = max(untouched_ages, default=0)

        self._replays_total += 1
        self._slots_scored_total += len(cf.slot_indices)
        self._last_slots_tracked = len(cf.slot_indices)

        # Compute signals
        ema_t0 = time.monotonic()
        sharpness_per_slot = _compute_per_slot_sharpness(cf.marginal_gains, cf.mask)
        mask_f = cf.mask.float()
        n_valid = mask_f.sum().clamp(min=1)
        n_scored = len(cf.slot_indices)

        # Per-slot marginal gain (mean over tokens and batch)
        if cf.marginal_gains.shape[0] > 0:
            slot_marginals_t = (
                (cf.marginal_gains * mask_f.unsqueeze(0)).sum(dim=(1, 2)) / n_valid
            ).float()
        else:
            slot_marginals_t = torch.zeros(0, dtype=torch.float32)
        if slot_marginals_t.numel() < n_scored:
            slot_marginals_t = torch.cat(
                [
                    slot_marginals_t,
                    torch.zeros(
                        n_scored - slot_marginals_t.numel(),
                        dtype=slot_marginals_t.dtype,
                        device=slot_marginals_t.device,
                    ),
                ]
            )
        slot_marginals = slot_marginals_t.tolist()

        # Per-slot retrieval mass
        if cf.weights_baseline.numel() > 0:
            slot_retrieval_mass_t = cf.weights_baseline.mean(dim=0).float()
        else:
            slot_retrieval_mass_t = torch.zeros(0, dtype=torch.float32)
        if slot_retrieval_mass_t.numel() < n_scored:
            slot_retrieval_mass_t = torch.cat(
                [
                    slot_retrieval_mass_t,
                    torch.zeros(
                        n_scored - slot_retrieval_mass_t.numel(),
                        dtype=slot_retrieval_mass_t.dtype,
                        device=slot_retrieval_mass_t.device,
                    ),
                ]
            )
        slot_retrieval_mass = slot_retrieval_mass_t.tolist()

        sharpness_t = sharpness_per_slot.float()
        if sharpness_t.numel() < n_scored:
            sharpness_t = torch.cat(
                [
                    sharpness_t,
                    torch.zeros(
                        n_scored - sharpness_t.numel(),
                        dtype=sharpness_t.dtype,
                        device=sharpness_t.device,
                    ),
                ]
            )

        # Update SlotRecords (or legacy dicts)
        table = getattr(outer, "table", None)
        has_table = table is not None

        if has_table:
            update_rows: list[tuple[int, int, SlotRecord]] = []
            repr_drifts: list[float] = []
            for i, phys_idx in enumerate(cf.slot_indices):
                sid = table.physical_to_slot_id(phys_idx)
                if sid is None:
                    continue
                rec = table.record(sid)
                if rec is None:
                    continue
                slot_tensor = table.get_tensor(sid)
                repr_drifts.append(
                    _compute_representation_drift(outer, slot_tensor)
                    if slot_tensor is not None
                    else 0.0
                )
                update_rows.append((i, phys_idx, rec))

            if update_rows:
                beta = self._ema_beta
                one_minus = 1.0 - beta
                row_idx = torch.tensor(
                    [row[0] for row in update_rows],
                    dtype=torch.long,
                    device=slot_marginals_t.device,
                )
                mg_t = slot_marginals_t[row_idx]
                sharp_t = sharpness_t.to(device=slot_marginals_t.device)[row_idx]
                rm_t = slot_retrieval_mass_t.to(device=slot_marginals_t.device)[row_idx]
                repr_t = torch.tensor(
                    repr_drifts,
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )

                records = [row[2] for row in update_rows]
                old_utility = torch.tensor(
                    [rec.utility_ema for rec in records],
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )
                old_marginal = torch.tensor(
                    [rec.marginal_gain_ema for rec in records],
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )
                old_sharpness = torch.tensor(
                    [rec.sharpness_ema for rec in records],
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )
                old_activation = torch.tensor(
                    [rec.activation_drift_ema for rec in records],
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )
                old_repr = torch.tensor(
                    [rec.representation_drift_ema for rec in records],
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )
                old_semantic = torch.tensor(
                    [rec.semantic_drift_ema for rec in records],
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )
                old_retrieval = torch.tensor(
                    [rec.retrieval_mass_ema for rec in records],
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )
                old_contradiction = torch.tensor(
                    [rec.contradiction_ema for rec in records],
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )
                old_peak_utility = torch.tensor(
                    [rec.peak_utility for rec in records],
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )
                old_peak_sharpness = torch.tensor(
                    [rec.peak_sharpness for rec in records],
                    dtype=torch.float32,
                    device=slot_marginals_t.device,
                )

                new_utility = beta * old_utility + one_minus * mg_t
                new_marginal = beta * old_marginal + one_minus * mg_t
                new_sharpness = beta * old_sharpness + one_minus * sharp_t
                new_activation = (
                    beta * old_activation
                    + one_minus * torch.abs(rm_t - old_retrieval)
                )
                new_repr = beta * old_repr + one_minus * repr_t
                new_semantic = (
                    beta * old_semantic
                    + one_minus * torch.abs(mg_t - new_marginal)
                )
                new_retrieval = beta * old_retrieval + one_minus * rm_t
                new_contradiction = (
                    beta * old_contradiction + one_minus * torch.clamp(-mg_t, min=0.0)
                )
                new_peak_utility = torch.maximum(old_peak_utility, new_utility)
                new_peak_sharpness = torch.maximum(old_peak_sharpness, new_sharpness)

                utility_l = new_utility.tolist()
                marginal_l = new_marginal.tolist()
                sharpness_l = new_sharpness.tolist()
                activation_l = new_activation.tolist()
                repr_l = new_repr.tolist()
                semantic_l = new_semantic.tolist()
                retrieval_l = new_retrieval.tolist()
                contradiction_l = new_contradiction.tolist()
                peak_utility_l = new_peak_utility.tolist()
                peak_sharpness_l = new_peak_sharpness.tolist()
                mg_l = mg_t.tolist()

                for j, rec in enumerate(records):
                    mg = mg_l[j]
                    rec.utility_ema = utility_l[j]
                    rec.marginal_gain_ema = marginal_l[j]
                    rec.sharpness_ema = sharpness_l[j]
                    rec.activation_drift_ema = activation_l[j]
                    rec.representation_drift_ema = repr_l[j]
                    rec.semantic_drift_ema = semantic_l[j]
                    rec.retrieval_mass_ema = retrieval_l[j]
                    rec.contradiction_ema = contradiction_l[j]
                    rec.peak_utility = peak_utility_l[j]
                    rec.peak_sharpness = peak_sharpness_l[j]
                    rec.score_count += 1
                    rec.last_scored_step = step

                    if mg < 0:
                        rec.negative_streak += 1
                        rec.positive_streak = 0
                    else:
                        rec.positive_streak += 1
                        rec.negative_streak = 0

                    if rec.state == SLOT_WARMING and rec.score_count >= self._min_score_count:
                        rec.state = SLOT_ACTIVE
        else:
            for i, phys_idx in enumerate(cf.slot_indices):
                mg = slot_marginals[i]
                # Legacy path
                if phys_idx not in self._slot_first_seen_step:
                    self._slot_first_seen_step[phys_idx] = step
                    self._slot_utility_ema[phys_idx] = mg
                    self._slot_score_count[phys_idx] = 1
                else:
                    old = self._slot_utility_ema.get(phys_idx, mg)
                    self._slot_utility_ema[phys_idx] = self._ema_beta * old + (1 - self._ema_beta) * mg
                    self._slot_score_count[phys_idx] = self._slot_score_count.get(phys_idx, 0) + 1
        self._last_stage_seconds["ema"] = time.monotonic() - ema_t0
        self._stage_seconds_total["ema"] += self._last_stage_seconds["ema"]

        elapsed = time.monotonic() - t0
        if elapsed > self._max_seconds:
            return TickResult()

        # Classify and act.  Shadow mode is the experiment-safe lane: it
        # computes the same policy decisions and emits telemetry, but it never
        # mutates the table.  This lets CRCT vs CRCT+maintenance be measured
        # without conflating the first run with a new controller.
        if self._action_mode == "shadow":
            action_t0 = time.monotonic()
            self._classify_shadow(model=model, outer=outer, step=step, cf=cf, t0=t0)
            self._last_stage_seconds["action"] = time.monotonic() - action_t0
            self._stage_seconds_total["action"] += self._last_stage_seconds["action"]
            return TickResult()

        action_t0 = time.monotonic()
        result = self._classify_and_act(
            model=model, outer=outer, step=step, cf=cf,
            slot_marginals=slot_marginals, sharpness_per_slot=sharpness_per_slot,
            t0=t0,
        )
        self._last_stage_seconds["action"] = time.monotonic() - action_t0
        self._stage_seconds_total["action"] += self._last_stage_seconds["action"]
        return result

    def _classify_shadow(
        self,
        *,
        model: Any,
        outer: Any,
        step: int,
        cf: CounterfactualResult,
        t0: float,
    ) -> None:
        table = getattr(outer, "table", None)
        if table is None:
            return
        policy_kwargs = dict(
            eviction_threshold=self._threshold,
            useful_threshold=self._useful_threshold,
            drift_threshold=self._drift_threshold,
            quarantine_threshold=self._quarantine_threshold,
            distill_peak_threshold=self._distill_peak_threshold,
            peak_preserve_utility_threshold=self._peak_preserve_utility_threshold,
            peak_preserve_sharpness_threshold=self._peak_preserve_sharpness_threshold,
            min_age=self._min_age,
            min_score_count=self._min_score_count,
            capacity_pressure=self._capacity_pressure(outer),
        )
        actions: dict[int, str] = {}
        for phys_idx in cf.slot_indices:
            sid = table.physical_to_slot_id(phys_idx)
            if sid is None:
                continue
            rec = table.record(sid)
            if rec is None:
                continue
            action = self._policy.choose(rec, **policy_kwargs)
            actions[sid] = action
        confirmations = self._confirm_actions_with_oracle(
            model=model, outer=outer, actions=actions, cf=cf, t0=t0
        )
        for sid, action in actions.items():
            rec = table.record(sid)
            if rec is None:
                continue
            confirmed = confirmations.get(sid, action == SLOT_PRESERVE)
            self._shadow_actions_total += 1
            self._shadow_action_counts[action] = (
                self._shadow_action_counts.get(action, 0) + 1
            )
            self._trace_event(
                step,
                sid,
                action,
                rec,
                accepted=confirmed,
                reason="shadow_confirmed" if confirmed else "shadow_oracle_rejected",
                extra=self._decision_trace_extra(rec, policy_kwargs),
            )

    def _budget_exhausted(self, t0: float) -> bool:
        return (time.monotonic() - t0) > self._max_seconds

    def _confirm_actions_with_oracle(
        self,
        *,
        model: Any,
        outer: Any,
        actions: dict[int, str],
        cf: CounterfactualResult,
        t0: float,
    ) -> dict[int, bool]:
        """Use one batched real-encode oracle pass to confirm actions.

        The proxy saliency map can rank all slots cheaply.  Active mutations
        still need real SSM dynamics, so this confirms the top physical slots
        with ``oracle_confirm_slots`` and records proxy/oracle disagreement.
        """
        table = getattr(outer, "table", None)
        if (
            table is None
            or self._oracle_confirm_top_k <= 0
            or self._probe_input_ids is None
            or self._probe_valid_mask is None
            or self._budget_exhausted(t0)
        ):
            return {}

        needs_oracle = {
            SLOT_DECAY,
            SLOT_EVICT,
            SLOT_REFRESH,
            SLOT_QUARANTINE,
            SLOT_DISTILL,
        }
        candidate_pairs: list[tuple[int, int, str, float]] = []
        for sid, action in actions.items():
            if action not in needs_oracle:
                continue
            phys = table.slot_id_to_physical(sid)
            if phys is None:
                continue
            proxy_score = self._slot_mean_marginal(phys, cf)
            candidate_pairs.append((sid, phys, action, proxy_score))
        if not candidate_pairs:
            return {}

        candidate_pairs.sort(key=lambda row: abs(row[3]), reverse=True)
        candidate_pairs = candidate_pairs[: self._oracle_confirm_top_k]
        oracle_t0 = time.monotonic()
        oracle = oracle_confirm_slots(
            model=model,
            outer=outer,
            probe_input_ids=self._probe_input_ids,
            probe_valid_mask=self._probe_valid_mask,
            slot_indices=[phys for _sid, phys, _action, _proxy in candidate_pairs],
            cache_read_cutoff=self._probe_cache_cutoff,
        )
        self._last_oracle_seconds = time.monotonic() - oracle_t0
        self._oracle_seconds_total += self._last_oracle_seconds
        self._last_stage_seconds["oracle"] = self._last_oracle_seconds
        self._stage_seconds_total["oracle"] += self._last_oracle_seconds
        self._last_oracle_candidates = len(oracle.slot_indices)
        if not oracle.slot_indices:
            return {}

        mask_f = oracle.mask.float()
        denom = mask_f.sum().clamp(min=1.0)
        oracle_by_phys: dict[int, float] = {}
        for local_idx, phys in enumerate(oracle.slot_indices):
            if local_idx >= oracle.oracle_deltas.shape[0]:
                continue
            delta = (oracle.oracle_deltas[local_idx] * mask_f).sum() / denom
            oracle_by_phys[int(phys)] = float(delta.item())

        confirmations: dict[int, bool] = {}
        sign_matches = 0
        abs_error_sum = 0.0
        n_pairs = 0
        for sid, phys, action, proxy_score in candidate_pairs:
            if phys not in oracle_by_phys:
                continue
            oracle_score = oracle_by_phys[phys]
            confirmed = self._action_confirmed(
                action=action,
                proxy_score=proxy_score,
                oracle_score=oracle_score,
            )
            confirmations[sid] = confirmed
            n_pairs += 1
            if (proxy_score >= 0.0) == (oracle_score >= 0.0):
                sign_matches += 1
            abs_error_sum += abs(proxy_score - oracle_score)
            if confirmed:
                self._oracle_confirmed_actions_total += 1
            else:
                self._oracle_rejected_actions_total += 1

            rec = table.record(sid)
            self._trace_oracle_event(
                step=self._probe_step,
                slot_id=sid,
                action=action,
                rec=rec,
                proxy_score=proxy_score,
                oracle_score=oracle_score,
                confirmed=confirmed,
            )

        if n_pairs:
            self._oracle_confirmations_total += n_pairs
            self._proxy_oracle_sign_matches += sign_matches
            self._proxy_oracle_pairs_total += n_pairs
            self._proxy_oracle_abs_error_sum += abs_error_sum
            self._last_proxy_oracle_sign_match_rate = sign_matches / n_pairs
            self._last_proxy_oracle_abs_error = abs_error_sum / n_pairs
        return confirmations

    def _action_confirmed(
        self,
        *,
        action: str,
        proxy_score: float,
        oracle_score: float,
    ) -> bool:
        """Decide whether the oracle supports a proxy-proposed action."""
        del proxy_score
        low_value = oracle_score <= max(self._threshold, self._useful_threshold)
        if action in {SLOT_EVICT, SLOT_QUARANTINE}:
            return low_value
        if action == SLOT_DECAY:
            return oracle_score <= max(2.0 * self._threshold, self._useful_threshold)
        if action == SLOT_DISTILL:
            return low_value and oracle_score >= self._quarantine_threshold
        if action == SLOT_REFRESH:
            return oracle_score > self._useful_threshold
        return True

    def _has_action_agreement(
        self,
        slot_id: int,
        action: str,
        rec: SlotRecord | None = None,
    ) -> bool:
        """Require repeated fresh-frame agreement before structural mutation."""
        if self._action_agreement_count <= 1:
            return True
        generation = int(rec.write_generation) if rec is not None else 0
        last_action, count, last_frame_id, last_generation = self._action_evidence.get(
            int(slot_id),
            ("", 0, -1, generation),
        )
        if (
            last_action == action
            and last_generation == generation
            and last_frame_id != self._active_frame_id
        ):
            count += 1
        elif last_action == action and last_generation == generation:
            count = max(1, count)
        else:
            count = 1
            self._action_agreements_reset_total += 1
        self._action_evidence[int(slot_id)] = (
            action,
            count,
            self._active_frame_id,
            generation,
        )
        if count >= self._action_agreement_count:
            self._action_agreements_total += 1
            return True
        return False

    def _classify_and_act(
        self,
        *,
        model: Any,
        outer: Any,
        step: int,
        cf: CounterfactualResult,
        slot_marginals: list[float],
        sharpness_per_slot: torch.Tensor,
        t0: float,
    ) -> TickResult:
        table = getattr(outer, "table", None)
        has_table = table is not None
        result = TickResult()
        policy_kwargs = dict(
            eviction_threshold=self._threshold,
            useful_threshold=self._useful_threshold,
            drift_threshold=self._drift_threshold,
            quarantine_threshold=self._quarantine_threshold,
            distill_peak_threshold=self._distill_peak_threshold,
            peak_preserve_utility_threshold=self._peak_preserve_utility_threshold,
            peak_preserve_sharpness_threshold=self._peak_preserve_sharpness_threshold,
            min_age=self._min_age,
            min_score_count=self._min_score_count,
            capacity_pressure=self._capacity_pressure(outer),
        )

        # Collect actions per slot
        actions: dict[int, str] = {}  # slot_id_or_idx -> action

        if has_table:
            # Check quarantine releases first
            for sid in list(self._quarantined):
                rec = table.record(sid)
                if rec is None:
                    self._quarantined.discard(sid)
                    continue
                if rec.positive_streak >= self._quarantine_release_streak:
                    table.release(sid)
                    self._quarantined.discard(sid)
                    self._quarantine_positive_streak.pop(sid, None)
                    self._releases_total += 1
                    result.released.append(sid)
                    rec.state = SLOT_ACTIVE
                    self._trace_event(step, sid, "RELEASE", rec)

            # Classify each active slot
            for i, phys_idx in enumerate(cf.slot_indices):
                sid = table.physical_to_slot_id(phys_idx)
                if sid is None or sid in self._quarantined:
                    continue
                rec = table.record(sid)
                if rec is None:
                    continue
                action = self._policy.choose(rec, **policy_kwargs)
                actions[sid] = action

            confirmations = self._confirm_actions_with_oracle(
                model=model, outer=outer, actions=actions, cf=cf, t0=t0
            )

            # Execute non-structural actions first (refresh, quarantine, decay)
            for sid, action in actions.items():
                if self._budget_exhausted(t0):
                    break
                rec = table.record(sid)
                if rec is None:
                    continue
                extra = self._decision_trace_extra(rec, policy_kwargs)
                confirmed = confirmations.get(sid, action == SLOT_PRESERVE)
                if action == SLOT_REFRESH:
                    accepted = False
                    agreed = confirmed and self._has_action_agreement(sid, action, rec)
                    if agreed:
                        accepted = self._execute_refresh(model, outer, sid, cf, t0=t0)
                    if accepted:
                        result.refreshed.append(sid)
                        self._refreshes_total += 1
                        rec.refresh_count += 1
                        rec.state = SLOT_ACTIVE
                    rec.last_action = action
                    reason = (
                        ""
                        if agreed
                        else "oracle_rejected" if not confirmed else "awaiting_agreement"
                    )
                    self._trace_event(
                        step, sid, action, rec,
                        accepted=accepted, reason=reason, extra=extra,
                    )
                elif action == SLOT_QUARANTINE:
                    if not confirmed:
                        rec.last_action = action
                        self._trace_event(
                            step, sid, action, rec,
                            accepted=False, reason="oracle_rejected", extra=extra,
                        )
                        continue
                    if not self._has_action_agreement(sid, action, rec):
                        rec.last_action = action
                        self._trace_event(
                            step, sid, action, rec,
                            accepted=False, reason="awaiting_agreement", extra=extra,
                        )
                        continue
                    self._execute_quarantine(outer, sid)
                    result.quarantined.append(sid)
                    rec.last_action = action
                    self._trace_event(step, sid, action, rec, extra=extra)
                elif action == SLOT_DECAY:
                    if not confirmed:
                        rec.last_action = action
                        self._trace_event(
                            step, sid, action, rec,
                            accepted=False, reason="oracle_rejected", extra=extra,
                        )
                        continue
                    self._execute_decay(outer, sid)
                    result.decayed.append(sid)
                    self._decays_total += 1
                    rec.state = SLOT_DECAYING
                    rec.last_action = action
                    self._trace_event(step, sid, action, rec, extra=extra)
                elif action == SLOT_PRESERVE:
                    if rec.sharpness_ema > self._useful_threshold:
                        rec.state = SLOT_SHARP
                    elif rec.state == SLOT_WARMING and rec.score_count >= self._min_score_count:
                        rec.state = SLOT_ACTIVE
                    rec.last_action = action
                    self._trace_event(step, sid, action, rec, extra=extra)

            # Execute structural actions (evict, distill) — descending physical order
            retire_ids = []
            for sid, action in actions.items():
                if action in (SLOT_EVICT, SLOT_DISTILL):
                    retire_ids.append((sid, action))

            for sid, action in retire_ids:
                if self._budget_exhausted(t0):
                    break
                rec = table.record(sid)
                extra = self._decision_trace_extra(rec, policy_kwargs)
                if not confirmations.get(sid, False):
                    self._trace_event(
                        step, sid, action, rec,
                        accepted=False, reason="oracle_rejected", extra=extra,
                    )
                    if rec:
                        rec.last_action = action
                    continue
                if not self._has_action_agreement(sid, action, rec):
                    self._trace_event(
                        step, sid, action, rec,
                        accepted=False, reason="awaiting_agreement", extra=extra,
                    )
                    if rec:
                        rec.last_action = action
                    continue
                if action == SLOT_DISTILL:
                    receipt = self._execute_distill(outer, sid, step)
                    if receipt.accepted:
                        result.distilled.append(sid)
                        self._distills_total += 1
                        self._trace_event(
                            step, sid, action, rec, reason="distilled", extra=extra
                        )
                    else:
                        self._trace_event(
                            step, sid, action, rec,
                            accepted=False,
                            reason=f"distill_{receipt.target}_unavailable",
                            extra=extra,
                        )
                else:
                    table.retire(sid, reason="evicted")
                    result.evicted.append(sid)
                    self._evictions_total += 1
                    self._trace_event(
                        step, sid, action, rec, reason="evicted", extra=extra
                    )
                if rec:
                    rec.last_action = action

        else:
            # Legacy path (no SlotTable)
            result = self._apply_evictions_legacy(model=model, step=step)

        return result

    def _execute_refresh(
        self,
        model: Any,
        outer: Any,
        slot_id: int,
        cf: CounterfactualResult,
        *,
        t0: float,
    ) -> bool:
        table = getattr(outer, "table", None)
        if table is None:
            return False
        slot = table.get_tensor(slot_id)
        if slot is None:
            return False

        encoder = getattr(outer, "encoder", None)
        decoder = getattr(outer, "decoder", None)
        cue_proj = getattr(outer, "cue_proj", None)
        if encoder is None or decoder is None:
            return False

        # Generate candidates
        candidates: list[tuple[str, torch.Tensor]] = [("identity", slot)]

        # Roundtrip: decode → re-encode
        with torch.inference_mode():
            decoded = decoder(slot.to(dtype=decoder.weight.dtype))
            rt = torch.tanh(encoder(decoded.to(dtype=encoder.weight.dtype)))
            if torch.isfinite(rt).all():
                candidates.append(("roundtrip", rt.detach()))

            # Cue-aligned: lerp toward mean cue direction
            if cue_proj is not None and cf.weights_baseline is not None:
                cue_mean = cf.weights_baseline.mean(dim=0)  # not a true cue, but shape (N,)
                # Use the probe hidden states to get an actual cue
                if self._probe_input_ids is not None:
                    # We already have h_base from the probe, but we don't store it
                    # Use roundtrip as base, nudge toward better alignment
                    nudged = slot + self._refresh_lr * (rt - slot)
                    if torch.isfinite(nudged).all():
                        candidates.append(("cue_aligned", nudged.detach()))

            # Sharp-preserving: keep high-magnitude components
            slot_abs = slot.abs()
            threshold = slot_abs.mean() + slot_abs.std()
            sharp_mask = (slot_abs > threshold).float()
            preserved = slot * sharp_mask + rt * (1 - sharp_mask)
            if torch.isfinite(preserved).all():
                candidates.append(("sharp_preserving", preserved.detach()))

        # Score candidates: use identity as real-physics oracle baseline.
        if len(candidates) <= 1:
            return False

        identity_tensor = candidates[0][1].detach().clone()
        best_name = "identity"
        best_tensor = identity_tensor
        best_improvement = 0.0

        phys = table.slot_id_to_physical(slot_id)
        if phys is None:
            return False
        baseline_score = self._oracle_slot_score(model=model, outer=outer, phys=phys, t0=t0)
        if baseline_score is None:
            return False

        accepted = False
        try:
            for name, tensor in candidates[1:]:
                if self._budget_exhausted(t0):
                    break
                table.replace_tensor(slot_id, tensor, bump_generation=False)
                candidate_score = self._oracle_slot_score(
                    model=model, outer=outer, phys=phys, t0=t0
                )
                if candidate_score is None:
                    break
                improvement = candidate_score - baseline_score
                if improvement > best_improvement + self._refresh_margin:
                    best_name = name
                    best_tensor = tensor.detach().clone()
                    best_improvement = improvement

            # Apply best or revert to identity.  Candidate probes are not
            # writes; only the accepted refresh bumps write_generation.
            if best_name != "identity":
                table.replace_tensor(slot_id, best_tensor)
                accepted = True
                return True
            return False
        finally:
            if not accepted:
                table.replace_tensor(slot_id, identity_tensor, bump_generation=False)

    def _oracle_slot_score(
        self,
        *,
        model: Any,
        outer: Any,
        phys: int,
        t0: float,
    ) -> float | None:
        if (
            self._probe_input_ids is None
            or self._probe_valid_mask is None
            or self._budget_exhausted(t0)
        ):
            return None
        oracle_t0 = time.monotonic()
        oracle = oracle_confirm_slots(
            model=model,
            outer=outer,
            probe_input_ids=self._probe_input_ids,
            probe_valid_mask=self._probe_valid_mask,
            slot_indices=[phys],
            cache_read_cutoff=self._probe_cache_cutoff,
        )
        elapsed = time.monotonic() - oracle_t0
        self._last_oracle_seconds = elapsed
        self._oracle_seconds_total += elapsed
        self._last_stage_seconds["oracle"] = elapsed
        self._stage_seconds_total["oracle"] += elapsed
        self._last_oracle_candidates = len(oracle.slot_indices)
        if not oracle.slot_indices or oracle.oracle_deltas.shape[0] == 0:
            return None
        mask_f = oracle.mask.float()
        denom = mask_f.sum().clamp(min=1.0)
        return float((oracle.oracle_deltas[0] * mask_f).sum() / denom)

    def _slot_mean_marginal(
        self, phys_idx: int, cf: CounterfactualResult
    ) -> float:
        """Return mean marginal gain for a physical slot in a probe result."""
        try:
            local_idx = cf.slot_indices.index(int(phys_idx))
        except ValueError:
            return 0.0
        if local_idx >= cf.marginal_gains.shape[0]:
            return 0.0
        mg = cf.marginal_gains[local_idx]
        m = cf.mask.float()
        return float((mg * m).sum() / m.sum().clamp(min=1))

    def _quick_refresh_score(
        self, outer: Any, phys_idx: int, cf: CounterfactualResult
    ) -> float:
        """Backward-compatible mean marginal helper for older tests."""
        return self._slot_mean_marginal(phys_idx, cf)

    def _execute_quarantine(self, outer: Any, slot_id: int) -> None:
        table = getattr(outer, "table", None)
        if table is not None:
            table.quarantine(slot_id)
        self._quarantined.add(slot_id)
        self._quarantine_positive_streak[slot_id] = 0

        # Overflow: evict worst quarantined if over limit
        if len(self._quarantined) > self._max_quarantined and table is not None:
            worst_id = None
            worst_util = float("inf")
            for qid in self._quarantined:
                rec = table.record(qid)
                if rec and rec.utility_ema < worst_util:
                    worst_util = rec.utility_ema
                    worst_id = qid
            if worst_id is not None and worst_id != slot_id:
                table.retire(worst_id, reason="quarantine_overflow")
                self._quarantined.discard(worst_id)
                self._evictions_total += 1

    def _execute_decay(self, outer: Any, slot_id: int) -> None:
        table = getattr(outer, "table", None)
        if table is not None:
            table.scale_survival(slot_id, 0.9)

    def _execute_distill(
        self, outer: Any, slot_id: int, step: int
    ) -> DistillReceipt:
        table = getattr(outer, "table", None)
        rec = table.record(slot_id) if table else None
        receipt = DistillReceipt(
            slot_id=slot_id,
            step=step,
            marginal_gain_before=rec.marginal_gain_ema if rec else 0.0,
            marginal_gain_peak=rec.peak_utility if rec else 0.0,
            marginal_gain_current=rec.marginal_gain_ema if rec else 0.0,
            target="latent_trace",
        )

        if table is None:
            receipt.target = "missing_table"
            return receipt
        slot_tensor = table.get_tensor(slot_id)
        if slot_tensor is None:
            receipt.target = "missing_slot"
            return receipt
        latent_traces = getattr(outer, "_latent_traces", None)
        if latent_traces is None:
            receipt.target = "missing_latent_trace"
            return receipt

        bucket_id = rec.bucket_id if rec else -1
        latent_traces.append({
            "bucket_id": bucket_id,
            "centroid_contrib": slot_tensor.detach().clone(),
        })
        max_latent = int(getattr(outer, "max_slots", 0) or 0)
        while max_latent > 0 and len(latent_traces) > max_latent:
            latent_traces.pop(0)
        receipt.target = "latent_trace"
        receipt.accepted = True
        table.retire(slot_id, reason="distilled")

        return receipt

    def _decision_trace_extra(
        self,
        rec: SlotRecord | None,
        policy_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if rec is None:
            return {}
        values = self._policy.action_values(rec, **policy_kwargs)
        extra: dict[str, Any] = {
            f"action_value_{action.lower()}": float(value)
            for action, value in values.items()
        }
        for factor in (0.5, 2.0):
            kwargs = dict(policy_kwargs)
            kwargs["eviction_threshold"] = float(kwargs["eviction_threshold"]) * factor
            kwargs["useful_threshold"] = float(kwargs["useful_threshold"]) * factor
            kwargs["quarantine_threshold"] = (
                float(kwargs["quarantine_threshold"]) * factor
            )
            label = str(factor).replace(".", "p")
            extra[f"counterfactual_action_threshold_x{label}"] = self._policy.choose(
                rec, **kwargs
            )
        return extra

    def _apply_evictions_legacy(self, *, model: Any, step: int) -> TickResult:
        """Legacy eviction path for models without SlotTable."""
        outer = getattr(model, "outer_model", None)
        if outer is None:
            return TickResult()

        evict_indices: list[int] = []
        for idx, ema_util in self._slot_utility_ema.items():
            first_seen = self._slot_first_seen_step.get(idx, step)
            age = step - first_seen
            sc = self._slot_score_count.get(idx, 0)
            if age < self._min_age:
                continue
            if sc < 2:
                continue
            if ema_util < self._threshold:
                evict_indices.append(idx)

        if not evict_indices:
            return TickResult()

        actual = _evict_slots(outer, evict_indices)
        for idx in actual:
            self._slot_utility_ema.pop(idx, None)
            self._slot_first_seen_step.pop(idx, None)
            self._slot_score_count.pop(idx, None)
        self._evictions_total += len(actual)

        if actual:
            self._reindex_after_eviction(actual)

        return TickResult(evicted=actual)

    def _reindex_after_eviction(self, evicted: list[int]) -> None:
        evicted_set = set(evicted)
        sorted_evicted = sorted(evicted_set, reverse=True)

        def reindex(d: dict) -> dict:
            new = {}
            for old_idx in sorted(d.keys()):
                if old_idx in evicted_set:
                    continue
                shift = sum(1 for e in sorted_evicted if e < old_idx)
                new[old_idx - shift] = d[old_idx]
            return new

        self._slot_utility_ema = reindex(self._slot_utility_ema)
        self._slot_first_seen_step = reindex(self._slot_first_seen_step)
        self._slot_score_count = reindex(self._slot_score_count)

    def _trace_event(
        self,
        step: int,
        slot_id: int,
        action: str,
        rec: SlotRecord | None,
        *,
        accepted: bool = True,
        reason: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._trace_path is None:
            return
        if self._trace_max_rows > 0 and self._trace_rows_written >= self._trace_max_rows:
            return
        event = {
            "row_type": f"replay_{action.lower()}",
            "step": step,
            "tick": self._tick_count,
            "slot_id": slot_id,
            "action": action,
            "accepted": accepted,
            "reason": reason,
            "frame_id": self._active_frame_id,
            "frame_step": self._active_frame_step,
            "frame_age_steps": self._last_frame_age_steps,
            "frame_age_seconds": self._last_frame_age_seconds,
            "stream_id": self._probe_stream_id,
            "queue_depth": self._last_queue_depth,
            "slot_work_items": self._last_slot_work_items,
            "probe_seconds": self._last_probe_seconds,
            "probe_budget_seconds": self._max_seconds,
            "probe_budget_utilization": (
                self._last_probe_seconds / self._max_seconds
                if self._max_seconds > 0.0
                else 0.0
            ),
        }
        if rec is not None:
            event.update({
                "slot_write_generation": rec.write_generation,
                "slot_created_step": rec.created_step,
                "slot_age_steps": step - rec.created_step,
                "slot_bucket_id": rec.bucket_id,
                "slot_event_id": rec.event_id,
                "marginal_gain": rec.marginal_gain_ema,
                "sharpness": rec.sharpness_ema,
                "activation_drift": rec.activation_drift_ema,
                "representation_drift": rec.representation_drift_ema,
                "semantic_drift": rec.semantic_drift_ema,
                "contradiction": rec.contradiction_ema,
                "retrieval_mass": rec.retrieval_mass_ema,
                "peak_utility": rec.peak_utility,
                "peak_sharpness": rec.peak_sharpness,
                "score_count": rec.score_count,
                "state": rec.state,
            })
        if extra:
            event.update(extra)
        self._trace_buffer.append(json.dumps(event, separators=(",", ":")) + "\n")
        self._trace_rows_written += 1

    def _trace_oracle_event(
        self,
        *,
        step: int,
        slot_id: int,
        action: str,
        rec: SlotRecord | None,
        proxy_score: float,
        oracle_score: float,
        confirmed: bool,
    ) -> None:
        if self._trace_path is None:
            return
        if self._trace_max_rows > 0 and self._trace_rows_written >= self._trace_max_rows:
            return
        event = {
            "row_type": "replay_oracle_confirm",
            "step": int(step),
            "tick": self._tick_count,
            "slot_id": int(slot_id),
            "action": action,
            "frame_id": self._active_frame_id,
            "frame_step": self._active_frame_step,
            "frame_age_steps": self._last_frame_age_steps,
            "frame_age_seconds": self._last_frame_age_seconds,
            "stream_id": self._probe_stream_id,
            "queue_depth": self._last_queue_depth,
            "slot_work_items": self._last_slot_work_items,
            "proxy_score": float(proxy_score),
            "oracle_score": float(oracle_score),
            "proxy_oracle_abs_error": float(abs(proxy_score - oracle_score)),
            "proxy_oracle_sign_match": bool(
                (proxy_score >= 0.0) == (oracle_score >= 0.0)
            ),
            "oracle_confirmed": bool(confirmed),
            "oracle_seconds": self._last_oracle_seconds,
        }
        if rec is not None:
            event.update(
                {
                    "slot_write_generation": rec.write_generation,
                    "slot_created_step": rec.created_step,
                    "slot_age_steps": int(step) - rec.created_step,
                    "slot_bucket_id": rec.bucket_id,
                    "slot_event_id": rec.event_id,
                    "marginal_gain": rec.marginal_gain_ema,
                    "sharpness": rec.sharpness_ema,
                    "activation_drift": rec.activation_drift_ema,
                    "representation_drift": rec.representation_drift_ema,
                    "semantic_drift": rec.semantic_drift_ema,
                    "contradiction": rec.contradiction_ema,
                    "retrieval_mass": rec.retrieval_mass_ema,
                    "peak_utility": rec.peak_utility,
                    "peak_sharpness": rec.peak_sharpness,
                    "state": rec.state,
                }
            )
        self._trace_buffer.append(json.dumps(event, separators=(",", ":")) + "\n")
        self._trace_rows_written += 1

    def _trace_frame_event(
        self,
        row_type: str,
        frame: ProbeFrame,
        *,
        queue_depth: int,
        slot_work_items: int = 0,
        frame_age_steps: int = 0,
        frame_age_seconds: float = 0.0,
    ) -> None:
        if self._trace_path is None:
            return
        if self._trace_max_rows > 0 and self._trace_rows_written >= self._trace_max_rows:
            return
        event = {
            "row_type": row_type,
            "step": int(frame.step),
            "tick": self._tick_count,
            "frame_id": int(frame.frame_id),
            "frame_step": int(frame.step),
            "stream_id": int(frame.stream_id),
            "queue_depth": int(queue_depth),
            "slot_work_items": int(slot_work_items),
            "frame_age_steps": int(frame_age_steps),
            "frame_age_seconds": float(frame_age_seconds),
            "processed_slots": len(frame.processed_slots),
            "probe_buffer_size": self._probe_buffer_size,
        }
        self._trace_buffer.append(json.dumps(event, separators=(",", ":")) + "\n")
        self._trace_rows_written += 1

    def flush_trace(self) -> None:
        if self._trace_path is None or not self._trace_buffer:
            return
        try:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as fh:
                fh.write("".join(self._trace_buffer))
            self._trace_buffer.clear()
        except Exception:
            pass

    def diagnostics(self) -> dict[str, Any]:
        self.flush_trace()
        elapsed_wall = max(1e-9, time.monotonic() - self._started_at)
        queue_depth_mean = (
            self._queue_depth_sum / self._queue_depth_samples
            if self._queue_depth_samples
            else 0.0
        )
        frame_age_steps_mean = (
            self._frame_age_steps_sum / self._replays_total
            if self._replays_total
            else 0.0
        )
        frame_age_seconds_mean = (
            self._frame_age_seconds_sum / self._replays_total
            if self._replays_total
            else 0.0
        )
        stream_duty = {
            str(i): self._stream_probe_seconds.get(i, 0.0) / elapsed_wall
            for i in range(self._memory_streams)
        }
        return {
            "enabled": True,
            "action_mode": self._action_mode,
            "memory_streams": self._memory_streams,
            "memory_streams_requested": self._memory_streams,
            "memory_streams_active": True,
            "memory_streams_note": "probe-frame stream ids actively partition replay-maintenance work on rank 3",
            "tick_count": self._tick_count,
            "replays_total": self._replays_total,
            "evictions_total": self._evictions_total,
            "refreshes_total": self._refreshes_total,
            "distills_total": self._distills_total,
            "decays_total": self._decays_total,
            "releases_total": self._releases_total,
            "slots_scored_total": self._slots_scored_total,
            "slots_tracked": self._last_slots_tracked
            if self._last_slots_tracked
            else len(self._slot_utility_ema),
            "shadow_actions_total": self._shadow_actions_total,
            "shadow_action_counts": dict(self._shadow_action_counts),
            "oracle_confirm_top_k": self._oracle_confirm_top_k,
            "oracle_confirmations_total": self._oracle_confirmations_total,
            "oracle_confirmed_actions_total": self._oracle_confirmed_actions_total,
            "oracle_rejected_actions_total": self._oracle_rejected_actions_total,
            "proxy_oracle_pairs_total": self._proxy_oracle_pairs_total,
            "proxy_oracle_sign_match_rate_total": (
                self._proxy_oracle_sign_matches / self._proxy_oracle_pairs_total
                if self._proxy_oracle_pairs_total
                else 0.0
            ),
            "proxy_oracle_abs_error_mean": (
                self._proxy_oracle_abs_error_sum / self._proxy_oracle_pairs_total
                if self._proxy_oracle_pairs_total
                else 0.0
            ),
            "probe_seconds_total": self._probe_seconds_total,
            "probe_seconds_mean": (
                self._probe_seconds_total / self._tick_count
                if self._tick_count
                else 0.0
            ),
            "last_probe_seconds": self._last_probe_seconds,
            "probe_over_budget_total": self._probe_over_budget_total,
            "probe_buffer_size": self._probe_buffer_size,
            "probe_frames_buffered": len(self._probe_frames),
            "probe_frames_ingested": self._probe_frames_ingested,
            "probe_frames_dropped_overflow": self._probe_frames_dropped_overflow,
            "probe_frames_dropped_stale": self._probe_frames_dropped_stale,
            "probe_frames_completed": self._probe_frames_completed,
            "probe_ticks_skipped_no_frame": self._probe_ticks_skipped_no_frame,
            "probe_ticks_skipped_no_slot_work": self._probe_ticks_skipped_no_slot_work,
            "queue_depth_last": self._last_queue_depth,
            "queue_depth_mean": queue_depth_mean,
            "queue_depth_max": self._queue_depth_max,
            "frame_ttl_steps": self._frame_ttl_steps,
            "frame_age_steps_last": self._last_frame_age_steps,
            "frame_age_steps_mean": frame_age_steps_mean,
            "frame_age_steps_max": self._frame_age_steps_max,
            "frame_age_seconds_last": self._last_frame_age_seconds,
            "frame_age_seconds_mean": frame_age_seconds_mean,
            "frame_age_seconds_max": self._frame_age_seconds_max,
            "slot_work_chunk_size": self._slot_work_chunk_size,
            "slot_work_items_total": self._slot_work_items_total,
            "last_slot_work_items": self._last_slot_work_items,
            "slot_work_items_mean": (
                self._slot_work_items_total / self._replays_total
                if self._replays_total
                else 0.0
            ),
            "last_visible_slots": self._last_visible_slots,
            "slots_untouched_past_ttl": self._last_untouched_slots,
            "max_untouched_slot_steps": self._last_max_untouched_steps,
            "slot_coverage_per_minute": (
                self._slots_scored_total / (elapsed_wall / 60.0)
                if elapsed_wall > 0.0
                else 0.0
            ),
            "action_agreement_count": self._action_agreement_count,
            "action_agreements_total": self._action_agreements_total,
            "action_agreements_reset_total": self._action_agreements_reset_total,
            "active_frame_id": self._active_frame_id,
            "active_frame_step": self._active_frame_step,
            "stage_seconds_last": dict(self._last_stage_seconds),
            "stage_seconds_total": dict(self._stage_seconds_total),
            "stream_ticks": {str(k): v for k, v in self._stream_ticks.items()},
            "stream_work_items": {str(k): v for k, v in self._stream_work_items.items()},
            "stream_probe_seconds": {
                str(k): v for k, v in self._stream_probe_seconds.items()
            },
            "stream_probe_duty_cycle": stream_duty,
            "oracle_seconds_total": self._oracle_seconds_total,
            "last_oracle_seconds": self._last_oracle_seconds,
            "last_proxy_oracle_sign_match_rate": self._last_proxy_oracle_sign_match_rate,
            "last_proxy_oracle_abs_error": self._last_proxy_oracle_abs_error,
            "last_oracle_candidates": self._last_oracle_candidates,
            "quarantined_count": len(self._quarantined),
            "eviction_threshold": self._threshold,
            "peak_preserve_utility_threshold": self._peak_preserve_utility_threshold,
            "peak_preserve_sharpness_threshold": self._peak_preserve_sharpness_threshold,
            "has_probe": self.has_probe(),
            "probe_step": self._probe_step,
            "probe_stream_id": self._probe_stream_id,
        }


def _evict_slots(outer: Any, indices: list[int]) -> list[int]:
    """Remove slots from a MultiSlotOuterModel by index (backward compat)."""
    if not hasattr(outer, "_slots"):
        return []
    n_slots = len(outer._slots)
    valid = sorted([i for i in indices if 0 <= i < n_slots], reverse=True)
    for i in valid:
        del outer._slots[i]
        if i < len(outer._survival):
            del outer._survival[i]
        if i < len(outer._slot_buckets):
            del outer._slot_buckets[i]
        if i < len(outer._slot_event_ids):
            del outer._slot_event_ids[i]
    return sorted(valid)
