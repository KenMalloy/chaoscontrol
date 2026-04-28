"""Adaptive Residual Memory control plane for CRCT rank-3.

The episodic sidecar is a living residual layer over the SSM, not a
passive cache. Each memory slot is continuously scored relative to
the current model via counterfactual probes and classified into one
of six actions: PRESERVE, DECAY, EVICT, REFRESH, QUARANTINE, DISTILL.

The probe engine runs 1 SSM encode + vectorized masked NLL to score
all slots every tick. Working set ~700KB, fully L2-resident on H100.
"""
from __future__ import annotations

import json
import time
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
    "replay_score_slots",
    "MaintenancePolicy",
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
    score_count: int = 0
    accepted: bool = True
    reason: str = ""
    refresh_candidate: str = ""


@torch.inference_mode()
def counterfactual_probe(
    *,
    model: Any,
    outer: Any,
    probe_input_ids: torch.Tensor,
    probe_valid_mask: torch.Tensor,
    cache_read_cutoff: int | None = None,
    chunk_size: int = 16,
) -> CounterfactualResult:
    """Score all slots via vectorized counterfactual masking.

    1 SSM encode + batched masked NLL. All slots scored every tick.
    Working set ~700KB — fully L2-resident on H100 SXM.
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

    # Phase 3: Cue + similarity
    cue_proj = getattr(outer, "cue_proj", None)
    decoder = getattr(outer, "decoder", None)
    if cue_proj is None or decoder is None:
        return CounterfactualResult(
            marginal_gains=torch.zeros(0, B, T, device=dev),
            sidecar_value=torch.zeros(B, T, device=dev),
            nll_baseline=torch.zeros(B, T, device=dev),
            nll_no_sidecar=torch.zeros(B, T, device=dev),
            weights_baseline=torch.zeros(B, 0, device=dev),
            mask=mask,
            slot_indices=vis,
        )

    with ac:
        cue = h_base.detach().mean(dim=1)
        cue_outer = cue_proj(cue.to(dtype=cue_proj.weight.dtype))
        sim = cue_outer @ slot_mat.to(dtype=cue_outer.dtype).T

    # Phase 4: Build masks — baseline + sidecar_off + N hide-one
    N_var = N + 2
    masks = torch.ones(N_var, N, device=dev)
    masks[1, :] = 0
    for i in range(N):
        masks[i + 2, i] = 0

    # Phase 5: Vectorized probe in chunks
    nll_baseline = None
    nll_no_sidecar = None
    nll_hide = []
    weights_baseline = None

    cs = chunk_size if chunk_size > 0 else N_var
    for start in range(0, N_var, cs):
        end = min(start + cs, N_var)
        chunk_masks = masks[start:end]
        C = chunk_masks.shape[0]

        with ac:
            sim_exp = sim.unsqueeze(0).expand(C, -1, -1)
            sim_masked = sim_exp.masked_fill(chunk_masks.unsqueeze(1) == 0, float("-inf"))

            all_neg_inf = (sim_masked == float("-inf")).all(dim=-1, keepdim=True)
            w = torch.where(
                all_neg_inf.expand_as(sim_masked),
                torch.zeros_like(sim_masked),
                F.softmax(sim_masked, dim=-1),
            )

            retrieved = torch.bmm(w, slot_mat.unsqueeze(0).expand(C, -1, -1).to(w.dtype))
            biases = F.linear(retrieved, decoder.weight.to(w.dtype))
            h_variants = h_base.unsqueeze(0).expand(C, -1, -1, -1) + biases.unsqueeze(2)

        h_flat = h_variants.reshape(C * B, T, D).float()
        y_flat = y.unsqueeze(0).expand(C, -1, -1).reshape(C * B, -1)
        nll_chunk = chunked_nll_from_hidden(model, h_flat, y_flat)
        nll_shaped = nll_chunk.reshape(C, B, T)

        for ci in range(C):
            global_idx = start + ci
            if global_idx == 0:
                nll_baseline = nll_shaped[ci]
                weights_baseline = w[ci]
            elif global_idx == 1:
                nll_no_sidecar = nll_shaped[ci]
            else:
                nll_hide.append(nll_shaped[ci])

    if nll_baseline is None:
        nll_baseline = torch.zeros(B, T, device=dev)
    if nll_no_sidecar is None:
        nll_no_sidecar = torch.zeros(B, T, device=dev)
    if weights_baseline is None:
        weights_baseline = torch.zeros(B, N, device=dev)

    marginal_gains = torch.stack(nll_hide, dim=0) - nll_baseline.unsqueeze(0) if nll_hide else torch.zeros(0, B, T, device=dev)
    sidecar_value = nll_no_sidecar - nll_baseline

    return CounterfactualResult(
        marginal_gains=marginal_gains.cpu(),
        sidecar_value=sidecar_value.cpu(),
        nll_baseline=nll_baseline.cpu(),
        nll_no_sidecar=nll_no_sidecar.cpu(),
        weights_baseline=weights_baseline.cpu(),
        mask=mask.cpu(),
        slot_indices=vis,
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
    diffs = (masked - means.view(N, 1, 1) * m.unsqueeze(0))
    variance = (diffs ** 2 * m.unsqueeze(0)).sum(dim=(1, 2)) / n_valid
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
        min_age: int,
        min_score_count: int,
    ) -> dict[str, float]:
        enough = rec.score_count >= min_score_count
        old = (rec.last_scored_step - rec.created_step) >= min_age if rec.last_scored_step > 0 else False

        quarantine_v = 0.0
        if enough and rec.contradiction_ema > abs(quarantine_threshold):
            quarantine_v = rec.contradiction_ema + 0.5 * rec.negative_streak

        distill_v = 0.0
        if old and enough and rec.peak_utility > distill_peak_threshold:
            internalization = 1.0 - rec.marginal_gain_ema / max(rec.peak_utility, 1e-6)
            if internalization > 0.5 and rec.marginal_gain_ema < 2 * eviction_threshold:
                distill_v = internalization * rec.peak_utility

        evict_v = 0.0
        if old and enough and rec.utility_ema < eviction_threshold:
            evict_v = eviction_threshold - rec.utility_ema

        refresh_v = 0.0
        drift_max = max(rec.activation_drift_ema, rec.representation_drift_ema, rec.semantic_drift_ema)
        if drift_max > drift_threshold and rec.marginal_gain_ema > useful_threshold:
            refresh_v = 0.8 * drift_max + 0.6 * rec.marginal_gain_ema

        preserve_v = rec.marginal_gain_ema + 0.7 * rec.sharpness_ema

        decay_v = 0.0
        if not old and rec.utility_ema < 2 * eviction_threshold:
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
        eviction_threshold: float = 0.01,
        eviction_ema_beta: float = 0.9,
        min_slot_age_steps: int = 128,
        max_seconds_per_tick: float = 0.5,
        trace_path: str | None = None,
        trace_max_rows: int = 0,
        probe_chunk_size: int = 16,
        drift_threshold: float = 0.3,
        repr_drift_threshold: float = 0.2,
        refresh_lr: float = 0.1,
        refresh_margin: float = 0.001,
        quarantine_threshold: float = -0.01,
        max_quarantined: int = 8,
        quarantine_release_streak: int = 2,
        distill_peak_threshold: float = 0.04,
        useful_threshold: float = 0.005,
        min_score_count: int = 2,
    ) -> None:
        self._threshold = float(eviction_threshold)
        self._ema_beta = float(eviction_ema_beta)
        self._min_age = int(min_slot_age_steps)
        self._max_seconds = float(max_seconds_per_tick)
        self._probe_chunk_size = int(probe_chunk_size)
        self._drift_threshold = float(drift_threshold)
        self._repr_drift_threshold = float(repr_drift_threshold)
        self._refresh_lr = float(refresh_lr)
        self._refresh_margin = float(refresh_margin)
        self._quarantine_threshold = float(quarantine_threshold)
        self._max_quarantined = int(max_quarantined)
        self._quarantine_release_streak = int(quarantine_release_streak)
        self._distill_peak_threshold = float(distill_peak_threshold)
        self._useful_threshold = float(useful_threshold)
        self._min_score_count = int(min_score_count)

        # Legacy per-index tracking (kept for backward compat in tests)
        self._slot_utility_ema: dict[int, float] = {}
        self._slot_first_seen_step: dict[int, int] = {}
        self._slot_score_count: dict[int, int] = {}

        # Probe cache
        self._probe_input_ids: torch.Tensor | None = None
        self._probe_valid_mask: torch.Tensor | None = None
        self._probe_cache_cutoff: int | None = None
        self._probe_step: int = 0

        # Counters
        self._tick_count: int = 0
        self._evictions_total: int = 0
        self._refreshes_total: int = 0
        self._distills_total: int = 0
        self._decays_total: int = 0
        self._releases_total: int = 0
        self._replays_total: int = 0
        self._slots_scored_total: int = 0

        # Quarantine tracking (by slot_id when using SlotTable, by index otherwise)
        self._quarantined: set[int] = set()
        self._quarantine_positive_streak: dict[int, int] = {}

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
        cache_read_cutoff: int | None,
        step: int,
    ) -> None:
        self._probe_input_ids = input_ids.detach()
        self._probe_valid_mask = valid_mask.detach()
        self._probe_cache_cutoff = cache_read_cutoff
        self._probe_step = step

    def has_probe(self) -> bool:
        return self._probe_input_ids is not None

    def tick(self, *, model: Any, step: int) -> TickResult:
        """One maintenance tick. Probe, classify, act."""
        if self._probe_input_ids is None:
            return TickResult()

        t0 = time.monotonic()
        self._tick_count += 1

        outer = getattr(model, "outer_model", None)
        if outer is None:
            return TickResult()

        # Run counterfactual probe
        cf = counterfactual_probe(
            model=model,
            outer=outer,
            probe_input_ids=self._probe_input_ids,
            probe_valid_mask=self._probe_valid_mask,
            cache_read_cutoff=self._probe_cache_cutoff,
            chunk_size=self._probe_chunk_size,
        )

        if len(cf.slot_indices) == 0:
            return TickResult()

        self._replays_total += 1
        self._slots_scored_total += len(cf.slot_indices)

        # Compute signals
        sharpness_per_slot = _compute_per_slot_sharpness(cf.marginal_gains, cf.mask)
        mask_f = cf.mask.float()
        n_valid = mask_f.sum().clamp(min=1)

        # Per-slot marginal gain (mean over tokens and batch)
        slot_marginals = []
        for i in range(len(cf.slot_indices)):
            if i < cf.marginal_gains.shape[0]:
                mg = (cf.marginal_gains[i] * mask_f).sum() / n_valid
                slot_marginals.append(float(mg.item()))
            else:
                slot_marginals.append(0.0)

        # Per-slot retrieval mass
        slot_retrieval_mass = []
        for i in range(len(cf.slot_indices)):
            if i < cf.weights_baseline.shape[-1]:
                rm = cf.weights_baseline[:, i].mean()
                slot_retrieval_mass.append(float(rm.item()))
            else:
                slot_retrieval_mass.append(0.0)

        # Update SlotRecords (or legacy dicts)
        table = getattr(outer, "table", None)
        has_table = table is not None

        for i, phys_idx in enumerate(cf.slot_indices):
            mg = slot_marginals[i]
            sharp = float(sharpness_per_slot[i].item()) if i < len(sharpness_per_slot) else 0.0
            rm = slot_retrieval_mass[i]

            if has_table:
                sid = table.physical_to_slot_id(phys_idx)
                if sid is None:
                    continue
                rec = table.record(sid)
                if rec is None:
                    continue
                slot_tensor = table.get_tensor(sid)
                repr_drift = _compute_representation_drift(outer, slot_tensor) if slot_tensor is not None else 0.0

                beta = self._ema_beta
                rec.utility_ema = beta * rec.utility_ema + (1 - beta) * mg
                rec.marginal_gain_ema = beta * rec.marginal_gain_ema + (1 - beta) * mg
                rec.sharpness_ema = beta * rec.sharpness_ema + (1 - beta) * sharp
                rec.activation_drift_ema = beta * rec.activation_drift_ema + (1 - beta) * abs(rm - rec.retrieval_mass_ema)
                rec.representation_drift_ema = beta * rec.representation_drift_ema + (1 - beta) * repr_drift
                rec.semantic_drift_ema = beta * rec.semantic_drift_ema + (1 - beta) * abs(mg - rec.marginal_gain_ema)
                rec.retrieval_mass_ema = beta * rec.retrieval_mass_ema + (1 - beta) * rm
                rec.contradiction_ema = beta * rec.contradiction_ema + (1 - beta) * max(0.0, -mg)
                rec.peak_utility = max(rec.peak_utility, rec.utility_ema)
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
                # Legacy path
                if phys_idx not in self._slot_first_seen_step:
                    self._slot_first_seen_step[phys_idx] = step
                    self._slot_utility_ema[phys_idx] = mg
                    self._slot_score_count[phys_idx] = 1
                else:
                    old = self._slot_utility_ema.get(phys_idx, mg)
                    self._slot_utility_ema[phys_idx] = self._ema_beta * old + (1 - self._ema_beta) * mg
                    self._slot_score_count[phys_idx] = self._slot_score_count.get(phys_idx, 0) + 1

        elapsed = time.monotonic() - t0
        if elapsed > self._max_seconds:
            return TickResult()

        # Classify and act
        result = self._classify_and_act(
            model=model, outer=outer, step=step, cf=cf,
            slot_marginals=slot_marginals, sharpness_per_slot=sharpness_per_slot,
        )
        return result

    def _classify_and_act(
        self,
        *,
        model: Any,
        outer: Any,
        step: int,
        cf: CounterfactualResult,
        slot_marginals: list[float],
        sharpness_per_slot: torch.Tensor,
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
            min_age=self._min_age,
            min_score_count=self._min_score_count,
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

            # Execute non-structural actions first (refresh, quarantine, decay)
            for sid, action in actions.items():
                rec = table.record(sid)
                if rec is None:
                    continue
                if action == SLOT_REFRESH:
                    accepted = self._execute_refresh(outer, sid, cf)
                    if accepted:
                        result.refreshed.append(sid)
                        self._refreshes_total += 1
                        rec.refresh_count += 1
                        rec.state = SLOT_ACTIVE
                    rec.last_action = action
                    self._trace_event(step, sid, action, rec, accepted=accepted)
                elif action == SLOT_QUARANTINE:
                    self._execute_quarantine(outer, sid)
                    result.quarantined.append(sid)
                    rec.last_action = action
                    self._trace_event(step, sid, action, rec)
                elif action == SLOT_DECAY:
                    self._execute_decay(outer, sid)
                    result.decayed.append(sid)
                    self._decays_total += 1
                    rec.state = SLOT_DECAYING
                    rec.last_action = action
                    self._trace_event(step, sid, action, rec)
                elif action == SLOT_PRESERVE:
                    if rec.sharpness_ema > self._useful_threshold:
                        rec.state = SLOT_SHARP
                    elif rec.state == SLOT_WARMING and rec.score_count >= self._min_score_count:
                        rec.state = SLOT_ACTIVE
                    rec.last_action = action

            # Execute structural actions (evict, distill) — descending physical order
            retire_ids = []
            for sid, action in actions.items():
                if action in (SLOT_EVICT, SLOT_DISTILL):
                    retire_ids.append((sid, action))

            for sid, action in retire_ids:
                rec = table.record(sid)
                if action == SLOT_DISTILL:
                    receipt = self._execute_distill(outer, sid, step)
                    result.distilled.append(sid)
                    self._distills_total += 1
                    self._trace_event(step, sid, action, rec, reason="distilled")
                else:
                    table.retire(sid, reason="evicted")
                    result.evicted.append(sid)
                    self._evictions_total += 1
                    self._trace_event(step, sid, action, rec, reason="evicted")
                if rec:
                    rec.last_action = action

        else:
            # Legacy path (no SlotTable)
            result = self._apply_evictions_legacy(model=model, step=step)

        return result

    def _execute_refresh(
        self, outer: Any, slot_id: int, cf: CounterfactualResult
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

        # Score candidates: use identity as baseline
        if len(candidates) <= 1:
            return False

        identity_tensor = candidates[0][1]
        best_name = "identity"
        best_tensor = identity_tensor
        best_improvement = 0.0

        phys = table.slot_id_to_physical(slot_id)
        if phys is None:
            return False

        for name, tensor in candidates[1:]:
            table.replace_tensor(slot_id, tensor)
            # Quick marginal check via the cached probe data
            improvement = self._quick_refresh_score(outer, phys, cf)
            if improvement > best_improvement + self._refresh_margin:
                best_name = name
                best_tensor = tensor
                best_improvement = improvement

        # Apply best or revert to identity
        if best_name != "identity":
            table.replace_tensor(slot_id, best_tensor)
            return True
        else:
            table.replace_tensor(slot_id, identity_tensor)
            return False

    def _quick_refresh_score(
        self, outer: Any, phys_idx: int, cf: CounterfactualResult
    ) -> float:
        """Estimate refresh improvement from cached probe data."""
        if phys_idx >= cf.marginal_gains.shape[0]:
            return 0.0
        mg = cf.marginal_gains[phys_idx]
        m = cf.mask.float()
        return float((mg * m).sum() / m.sum().clamp(min=1))

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
            phys = table.slot_id_to_physical(slot_id)
            if phys is not None and phys < len(table._survival):
                table._survival[phys] *= 0.9

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

        # Save to latent traces
        if table is not None:
            slot_tensor = table.get_tensor(slot_id)
            if slot_tensor is not None:
                latent_traces = getattr(outer, "_latent_traces", None)
                if latent_traces is not None:
                    bucket_id = rec.bucket_id if rec else -1
                    latent_traces.append({
                        "bucket_id": bucket_id,
                        "centroid_contrib": slot_tensor.detach().clone(),
                    })
                    receipt.target = "latent_trace"
                    receipt.accepted = True

            table.retire(slot_id, reason="distilled")

        return receipt

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
        }
        if rec is not None:
            event.update({
                "marginal_gain": rec.marginal_gain_ema,
                "sharpness": rec.sharpness_ema,
                "activation_drift": rec.activation_drift_ema,
                "representation_drift": rec.representation_drift_ema,
                "semantic_drift": rec.semantic_drift_ema,
                "contradiction": rec.contradiction_ema,
                "retrieval_mass": rec.retrieval_mass_ema,
                "peak_utility": rec.peak_utility,
                "score_count": rec.score_count,
                "state": rec.state,
            })
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
        return {
            "tick_count": self._tick_count,
            "replays_total": self._replays_total,
            "evictions_total": self._evictions_total,
            "refreshes_total": self._refreshes_total,
            "distills_total": self._distills_total,
            "decays_total": self._decays_total,
            "releases_total": self._releases_total,
            "slots_scored_total": self._slots_scored_total,
            "slots_tracked": len(self._slot_utility_ema),
            "quarantined_count": len(self._quarantined),
            "eviction_threshold": self._threshold,
            "has_probe": self.has_probe(),
            "probe_step": self._probe_step,
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
