"""Adaptive Residual Memory control plane for CRCT rank-3.

The episodic sidecar is a living residual layer over the SSM, not a
passive cache. Each memory slot is continuously scored relative to
the current model via counterfactual probes and classified into one
of seven actions: PRESERVE, DECAY, EVICT, REFRESH, QUARANTINE, DISTILL,
RELEASE.

The probe engine consumes a rolling stream of probe frames and bounded
slot-work microbatches. Each microbatch runs 1 SSM encode + vectorized
masked NLL over the selected hide-one variants, then a smaller oracle
confirmation pass measures active mutations under real SSM dynamics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .cache_utility import chunked_nll_from_hidden
from .kernels import _cpu_ssm_controller as _ext
from .slot_table import SlotTable, SlotRecord, SlotId
from .slot_table import SLOT_WARMING, SLOT_ACTIVE
from .slot_table import SLOT_DECAYING, SLOT_QUARANTINED, SLOT_RETIRED

__all__ = [
    "ReplayEvictionLoop",
    "TickResult",
    "counterfactual_probe",
    "oracle_confirm_slots",
    "oracle_confirm_refresh_candidates",
    "replay_score_slots",
    "MaintenancePolicy",
    "FullAControllerState",
    "LearnedCommitPolicy",
    "CpuRefreshProposalModel",
    "ProbeFrame",
    "_evict_slots",
]

SLOT_PRESERVE = "PRESERVE"
SLOT_DECAY = "DECAY"
SLOT_EVICT = "EVICT"
SLOT_REFRESH = "REFRESH"
SLOT_QUARANTINE = "QUARANTINE"
SLOT_DISTILL = "DISTILL"
SLOT_RELEASE = "RELEASE"
SLOT_ACTION_ORDER = (
    SLOT_PRESERVE,
    SLOT_DECAY,
    SLOT_EVICT,
    SLOT_REFRESH,
    SLOT_QUARANTINE,
    SLOT_DISTILL,
    SLOT_RELEASE,
)


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
    scoring_mode: str = "proxy"


@dataclass
class OracleConfirmationResult:
    """Real-physics batched confirmation for a small candidate set."""

    slot_indices: list[int]
    oracle_deltas: torch.Tensor
    nll_baseline: torch.Tensor
    nll_no_sidecar: torch.Tensor
    mask: torch.Tensor


@dataclass
class RefreshCandidateOracleResult:
    """Real-physics verification for CPU-proposed refresh candidates."""

    slot_index: int
    candidate_names: list[str]
    candidate_scores: torch.Tensor
    candidate_improvements: torch.Tensor
    nll_no_sidecar: torch.Tensor
    nll_candidates: torch.Tensor
    mask: torch.Tensor
    chunk_count: int = 0
    variants_total: int = 0


@dataclass
class DistillReceipt:
    slot_id: int
    step: int
    marginal_gain_before: float
    marginal_gain_peak: float
    marginal_gain_current: float
    target: str
    accepted: bool = False
    prototype_updated: bool = False
    prototype_reason: str = ""


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
    variant_chunk_size: int = 1,
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

    ac = (
        torch.autocast(dev.type, dtype=torch.bfloat16)
        if dev.type == "cuda"
        else torch.autocast("cpu", dtype=torch.bfloat16)
    )

    def _score_masks(slot_masks: torch.Tensor) -> torch.Tensor:
        variants = int(slot_masks.shape[0])
        x_exp = x.unsqueeze(0).expand(variants, -1, -1).reshape(variants * B, T)
        y_exp = y.unsqueeze(0).expand(variants, -1, -1).reshape(variants * B, T)
        slot_masks_exp = (
            slot_masks[:, None, :]
            .expand(variants, B, n_slots)
            .reshape(variants * B, n_slots)
        )
        with ac:
            hidden = model.encode(
                x_exp,
                memory_mode="force_on",
                cache_read_cutoff=cache_read_cutoff,
                memory_slot_mask=slot_masks_exp,
            )
        return chunked_nll_from_hidden(model, hidden, y_exp).reshape(variants, B, T)

    base_masks = torch.ones(2, n_slots, device=dev, dtype=torch.bool)
    base_masks[1, :] = False
    base_nll = _score_masks(base_masks)
    nll_baseline = base_nll[0]
    nll_no_sidecar = base_nll[1]
    score_device = base_nll.device

    candidate_chunk = max(1, int(variant_chunk_size))
    oracle_deltas = torch.empty(
        len(candidates), B, T, device=score_device, dtype=torch.float32
    )
    for start in range(0, len(candidates), candidate_chunk):
        chunk = candidates[start : start + candidate_chunk]
        slot_masks = torch.ones(len(chunk), n_slots, device=dev, dtype=torch.bool)
        rows = torch.arange(len(chunk), device=dev)
        cols = torch.tensor(chunk, device=dev, dtype=torch.long)
        slot_masks[rows, cols] = False
        nll_hide = _score_masks(slot_masks)
        oracle_deltas[start : start + len(chunk)] = (
            nll_hide - nll_baseline.unsqueeze(0)
        )
    return OracleConfirmationResult(
        slot_indices=candidates,
        oracle_deltas=oracle_deltas.cpu(),
        nll_baseline=nll_baseline.cpu(),
        nll_no_sidecar=nll_no_sidecar.cpu(),
        mask=mask.cpu(),
    )


@torch.inference_mode()
def oracle_confirm_refresh_candidates(
    *,
    model: Any,
    outer: Any,
    probe_input_ids: torch.Tensor,
    probe_valid_mask: torch.Tensor,
    slot_index: int,
    candidate_tensors: list[tuple[str, torch.Tensor]],
    cache_read_cutoff: int | None = None,
    variant_chunk_size: int = 16,
) -> RefreshCandidateOracleResult:
    """Verify candidate slot tensors under the real force-on read path.

    CPU proposes tensors; this function is the GPU3 physics gate.  It scores
    all candidates as one expanded batch using a sparse per-sample SlotTable
    override, so refresh acceptance never serially mutates the table and never
    falls back to the cheap proxy.
    """
    x = probe_input_ids[:, :-1]
    y = probe_input_ids[:, 1:]
    mask = probe_valid_mask[:, 1:].bool()
    B, T = x.shape
    dev = x.device

    table = getattr(outer, "table", None)
    n_slots = len(table) if table is not None else len(getattr(outer, "_slots", []))
    phys = int(slot_index)
    if n_slots <= 0 or phys < 0 or phys >= int(n_slots) or not candidate_tensors:
        empty = torch.zeros(0, B, T, device=dev)
        return RefreshCandidateOracleResult(
            slot_index=phys,
            candidate_names=[],
            candidate_scores=torch.zeros(0, device=dev),
            candidate_improvements=torch.zeros(0, device=dev),
            nll_no_sidecar=torch.zeros(B, T, device=dev),
            nll_candidates=empty,
            mask=mask,
            chunk_count=0,
            variants_total=0,
        )

    names: list[str] = []
    values: list[torch.Tensor] = []
    for name, tensor in candidate_tensors:
        t = tensor.detach()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() == 3 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.dim() != 2 or t.shape[0] != 1:
            raise ValueError("refresh candidate tensors must have shape (1, outer_dim)")
        names.append(str(name))
        values.append(t)
    candidate_stack = torch.cat(values, dim=0).to(device=dev)
    K = int(candidate_stack.shape[0])

    ac = (
        torch.autocast(dev.type, dtype=torch.bfloat16)
        if dev.type == "cuda"
        else torch.autocast("cpu", dtype=torch.bfloat16)
    )

    def _score_hidden(hidden: torch.Tensor, targets: torch.Tensor, variants: int) -> torch.Tensor:
        return chunked_nll_from_hidden(model, hidden, targets).reshape(variants, B, T)

    off_mask = torch.zeros(1, n_slots, device=dev, dtype=torch.bool)
    off_mask_exp = off_mask.expand(B, n_slots)
    with ac:
        hidden_off = model.encode(
            x,
            memory_mode="force_on",
            cache_read_cutoff=cache_read_cutoff,
            memory_slot_mask=off_mask_exp,
        )
    nll_no_sidecar = _score_hidden(hidden_off, y, 1)[0]

    chunk = max(1, int(variant_chunk_size))
    chunk_count = 0
    nll_candidates = torch.empty(K, B, T, device=nll_no_sidecar.device, dtype=torch.float32)
    for start in range(0, K, chunk):
        chunk_count += 1
        end = min(K, start + chunk)
        k = end - start
        x_exp = x.unsqueeze(0).expand(k, -1, -1).reshape(k * B, T)
        y_exp = y.unsqueeze(0).expand(k, -1, -1).reshape(k * B, T)
        values_exp = (
            candidate_stack[start:end]
            .to(dtype=candidate_stack.dtype)
            .unsqueeze(1)
            .expand(k, B, candidate_stack.shape[-1])
            .reshape(k * B, candidate_stack.shape[-1])
        )
        with ac:
            hidden = model.encode(
                x_exp,
                memory_mode="force_on",
                cache_read_cutoff=cache_read_cutoff,
                memory_slot_override_index=phys,
                memory_slot_override_values=values_exp,
            )
        nll_candidates[start:end] = _score_hidden(hidden, y_exp, k)

    mask_f = mask.to(device=nll_candidates.device, dtype=torch.float32)
    denom = mask_f.sum().clamp(min=1.0)
    candidate_scores = (
        (nll_no_sidecar.to(device=nll_candidates.device).unsqueeze(0) - nll_candidates)
        * mask_f.unsqueeze(0)
    ).sum(dim=(1, 2)) / denom
    candidate_improvements = candidate_scores - candidate_scores[0].detach()
    return RefreshCandidateOracleResult(
        slot_index=phys,
        candidate_names=names,
        candidate_scores=candidate_scores.cpu(),
        candidate_improvements=candidate_improvements.cpu(),
        nll_no_sidecar=nll_no_sidecar.cpu(),
        nll_candidates=nll_candidates.cpu(),
        mask=mask.cpu(),
        chunk_count=chunk_count,
        variants_total=K,
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


class FullAControllerState:
    """Small full-A recurrent state for CPU-owned memory maintenance.

    This is the controller's temporal substrate, not the AMX scorer and not the
    trunk SSM. It tracks maintenance-cycle evidence with coupled, rotational
    dynamics and emits a slot-space proposal direction. The dynamics are
    hard-projected to a near-critical stable envelope so a bad proposal stream
    cannot turn the controller into an unbounded oscillator.
    """

    input_dim = 16

    def __init__(
        self,
        *,
        state_dim: int = 32,
        rank: int = 8,
        dt: float = 1.0,
        gamma: float = 0.08,
        max_top_log_sv: float = -0.05,
        max_state_norm: float = 8.0,
        perturbation_scale: float = 0.25,
        feedback_lr: float = 0.05,
        seed: int = 1729,
    ) -> None:
        self.state_dim = max(2, int(state_dim))
        self.rank = max(1, min(int(rank), self.state_dim))
        self.dt = max(1.0e-4, float(dt))
        self.gamma = max(1.0e-4, float(gamma))
        self.max_top_log_sv = float(max_top_log_sv)
        self.max_state_norm = max(1.0e-4, float(max_state_norm))
        self.perturbation_scale = max(0.0, float(perturbation_scale))
        self.feedback_lr = max(0.0, float(feedback_lr))
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))
        self._state = torch.zeros(1, self.state_dim, dtype=torch.float32)
        self._slot_dim = 0
        self._readout: torch.Tensor | None = None
        self.steps = 0
        self.feedback_updates = 0
        self.projection_count = 0
        self.state_clamp_count = 0
        self.finite_state = True
        self.top_log_sv = 0.0
        self.sv_log_var = 0.0
        self.state_norm_last = 0.0
        self.state_norm_max = 0.0
        self._input_proj = self._make_input_projection()
        self._A_d = self._make_projected_dynamics()

    def _make_input_projection(self) -> torch.Tensor:
        proj = torch.randn(
            self.input_dim, self.state_dim, generator=self._generator
        )
        return F.normalize(proj, dim=0, eps=1e-8) * 0.25

    def _make_projected_dynamics(self) -> torch.Tensor:
        S = torch.zeros(self.state_dim, self.state_dim, dtype=torch.float32)
        n_pairs = self.state_dim // 2
        if n_pairs:
            freqs = torch.linspace(0.08, 0.55, n_pairs)
            for pair_idx, omega in enumerate(freqs.tolist()):
                a = 2 * pair_idx
                b = a + 1
                S[a, b] = -omega
                S[b, a] = omega
        U = torch.randn(self.state_dim, self.rank, generator=self._generator)
        V = torch.randn(self.state_dim, self.rank, generator=self._generator)
        U = F.normalize(U, dim=0, eps=1e-8)
        V = F.normalize(V, dim=0, eps=1e-8)
        low_rank = U @ V.T
        low_rank_norm = torch.linalg.matrix_norm(low_rank, ord=2).clamp_min(1e-8)
        low_rank = low_rank * (self.perturbation_scale / low_rank_norm)
        A_c = S - self.gamma * torch.eye(self.state_dim) + low_rank
        A_d = torch.matrix_exp(self.dt * A_c)
        return self._project_discrete_dynamics(A_d)

    def _project_discrete_dynamics(self, A_d: torch.Tensor) -> torch.Tensor:
        svs = torch.linalg.svdvals(A_d.float()).clamp_min(1e-8)
        logs = torch.log(svs)
        top = float(logs.max().item())
        if top > self.max_top_log_sv:
            A_d = A_d * float(torch.exp(torch.tensor(self.max_top_log_sv - top)))
            self.projection_count += 1
            svs = torch.linalg.svdvals(A_d.float()).clamp_min(1e-8)
            logs = torch.log(svs)
        self.top_log_sv = float(logs.max().item())
        self.sv_log_var = float(logs.var(unbiased=False).item())
        return A_d.contiguous()

    def _ensure_slot_dim(self, slot_dim: int) -> None:
        slot_dim = int(slot_dim)
        if self._readout is not None and self._slot_dim == slot_dim:
            return
        self._slot_dim = slot_dim
        readout = torch.randn(
            self.state_dim, slot_dim, generator=self._generator, dtype=torch.float32
        )
        self._readout = F.normalize(readout, dim=0, eps=1e-8) * 0.05

    def step(self, features: torch.Tensor, *, slot_dim: int) -> torch.Tensor:
        self._ensure_slot_dim(slot_dim)
        x = features.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
        if x.shape[-1] != self.input_dim:
            padded = torch.zeros(1, self.input_dim, dtype=torch.float32)
            n = min(self.input_dim, x.shape[-1])
            padded[:, :n] = x[:, :n]
            x = padded
        drive = torch.tanh(x @ self._input_proj)
        self._state = self._state @ self._A_d.T + drive
        self.finite_state = bool(torch.isfinite(self._state).all().item())
        if not self.finite_state:
            self._state.zero_()
        norm = float(self._state.norm().item())
        if norm > self.max_state_norm:
            self._state.mul_(self.max_state_norm / max(norm, 1e-8))
            self.state_clamp_count += 1
            norm = self.max_state_norm
        self.state_norm_last = norm
        self.state_norm_max = max(self.state_norm_max, norm)
        self.steps += 1
        assert self._readout is not None
        proposal = self._state @ self._readout
        if not torch.isfinite(proposal).all() or proposal.norm().item() <= 1e-8:
            return torch.zeros(1, int(slot_dim), dtype=torch.float32)
        return F.normalize(proposal, dim=-1, eps=1e-8)

    def hidden_state(self) -> torch.Tensor:
        return self._state.detach().cpu().float().clone()

    def update_feedback(
        self,
        *,
        identity: torch.Tensor,
        best: torch.Tensor,
        improvement: float,
        structural_accepted: bool,
    ) -> None:
        if self._readout is None or self.feedback_lr <= 0.0:
            return
        delta = best.detach().to(device="cpu", dtype=torch.float32) - identity.detach().to(
            device="cpu", dtype=torch.float32
        )
        if not torch.isfinite(delta).all() or delta.norm().item() <= 1e-8:
            return
        trust = 1.0 if structural_accepted else 0.25
        gain = max(0.0, min(1.0, float(improvement)))
        if gain <= 0.0:
            return
        h = F.normalize(self._state.detach(), dim=-1, eps=1e-8)
        d = F.normalize(delta.reshape(1, -1), dim=-1, eps=1e-8)
        self._readout = self._readout + self.feedback_lr * trust * gain * (h.T @ d)
        col_norm = self._readout.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self._readout = self._readout * torch.clamp(1.0 / col_norm, max=1.0)
        self.feedback_updates += 1

    def diagnostics(self) -> dict[str, Any]:
        return {
            "recurrence_mode": "full_a",
            "state_dim": self.state_dim,
            "rank": self.rank,
            "dt": self.dt,
            "gamma": self.gamma,
            "target_log_sv": self.max_top_log_sv,
            "top_log_sv": self.top_log_sv,
            "sv_log_var": self.sv_log_var,
            "state_norm_last": self.state_norm_last,
            "state_norm_max": self.state_norm_max,
            "finite_state": self.finite_state,
            "projection_count": self.projection_count,
            "state_clamp_count": self.state_clamp_count,
            "steps": self.steps,
            "feedback_updates": self.feedback_updates,
        }

    def state_dict(self) -> dict[str, Any]:
        """Persist the learned full-A proposal dynamics.

        This is eval-time state, not trace telemetry.  The state is small
        enough to keep in fp32 so reload does not perturb the controller.
        """
        return {
            "state_dim": self.state_dim,
            "rank": self.rank,
            "dt": self.dt,
            "gamma": self.gamma,
            "max_top_log_sv": self.max_top_log_sv,
            "max_state_norm": self.max_state_norm,
            "perturbation_scale": self.perturbation_scale,
            "feedback_lr": self.feedback_lr,
            "state": self._state.detach().cpu(),
            "slot_dim": self._slot_dim,
            "readout": None if self._readout is None else self._readout.detach().cpu(),
            "input_proj": self._input_proj.detach().cpu(),
            "A_d": self._A_d.detach().cpu(),
            "generator_state": self._generator.get_state(),
            "steps": self.steps,
            "feedback_updates": self.feedback_updates,
            "projection_count": self.projection_count,
            "state_clamp_count": self.state_clamp_count,
            "finite_state": self.finite_state,
            "top_log_sv": self.top_log_sv,
            "sv_log_var": self.sv_log_var,
            "state_norm_last": self.state_norm_last,
            "state_norm_max": self.state_norm_max,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.state_dim = int(state.get("state_dim", self.state_dim))
        self.rank = int(state.get("rank", self.rank))
        self.dt = float(state.get("dt", self.dt))
        self.gamma = float(state.get("gamma", self.gamma))
        self.max_top_log_sv = float(state.get("max_top_log_sv", self.max_top_log_sv))
        self.max_state_norm = float(state.get("max_state_norm", self.max_state_norm))
        self.perturbation_scale = float(
            state.get("perturbation_scale", self.perturbation_scale)
        )
        self.feedback_lr = float(state.get("feedback_lr", self.feedback_lr))
        self._state = state.get("state", self._state).detach().cpu().float()
        self._slot_dim = int(state.get("slot_dim", self._slot_dim))
        readout = state.get("readout")
        self._readout = None if readout is None else readout.detach().cpu().float()
        self._input_proj = state.get("input_proj", self._input_proj).detach().cpu().float()
        self._A_d = state.get("A_d", self._A_d).detach().cpu().float().contiguous()
        generator_state = state.get("generator_state")
        if isinstance(generator_state, torch.Tensor):
            self._generator.set_state(generator_state.cpu())
        self.steps = int(state.get("steps", self.steps))
        self.feedback_updates = int(state.get("feedback_updates", self.feedback_updates))
        self.projection_count = int(state.get("projection_count", self.projection_count))
        self.state_clamp_count = int(
            state.get("state_clamp_count", self.state_clamp_count)
        )
        self.finite_state = bool(state.get("finite_state", self.finite_state))
        self.top_log_sv = float(state.get("top_log_sv", self.top_log_sv))
        self.sv_log_var = float(state.get("sv_log_var", self.sv_log_var))
        self.state_norm_last = float(state.get("state_norm_last", self.state_norm_last))
        self.state_norm_max = float(state.get("state_norm_max", self.state_norm_max))


class CpuRefreshProposalModel:
    """CPU-side proposal model for refresh candidates.

    It is deliberately lightweight but stateful: the CPU controller owns
    proposal generation, learns a residual direction from GPU3 oracle feedback,
    and emits a vectorized candidate set for the GPU3 verifier.  The SlotTable
    is only mutated after oracle approval.
    """

    def __init__(
        self,
        *,
        k: int = 16,
        rank: int = 8,
        lr: float = 0.1,
        noise_scale: float = 0.04,
        momentum: float = 0.9,
        weight_sync_interval_steps: int = 64,
        seed: int = 1729,
        controller_state_dim: int = 32,
        controller_rank: int = 8,
        controller_dt: float = 1.0,
        controller_gamma: float = 0.08,
        controller_target_log_sv: float = -0.05,
        controller_max_state_norm: float = 8.0,
        controller_perturbation_scale: float = 0.25,
        controller_feedback_lr: float = 0.05,
        controller_seed: int | None = None,
    ) -> None:
        self.k = max(2, int(k))
        self.rank = max(1, int(rank))
        self.lr = float(lr)
        self.noise_scale = float(noise_scale)
        self.momentum = float(momentum)
        self.weight_sync_interval_steps = max(1, int(weight_sync_interval_steps))
        self._seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self._seed)
        self._basis: torch.Tensor | None = None
        self._learned_direction: torch.Tensor | None = None
        self._weight_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor | None]] = {}
        self._active_step = 0
        self._dim: int = 0
        self.samples_total = 0
        self.updates_total = 0
        self.positive_updates_total = 0
        self.structural_accepts_total = 0
        self.structural_rejects_total = 0
        self.rejected_positive_updates_total = 0
        self.weight_syncs_total = 0
        self.improvement_sum = 0.0
        self.improvement_abs_ema = 0.0
        self.improvement_scale_updates = 0
        self.last_best_index = 0
        self.last_best_name = "identity"
        self.last_best_improvement = 0.0
        self._controller = FullAControllerState(
            state_dim=int(controller_state_dim),
            rank=int(controller_rank),
            dt=float(controller_dt),
            gamma=float(controller_gamma),
            max_top_log_sv=float(controller_target_log_sv),
            max_state_norm=float(controller_max_state_norm),
            perturbation_scale=float(controller_perturbation_scale),
            feedback_lr=float(controller_feedback_lr),
            seed=int(self._seed + 97 if controller_seed is None else controller_seed),
        )

    def _ensure_dim(self, dim: int) -> None:
        if self._basis is not None and self._dim == int(dim):
            return
        self._dim = int(dim)
        basis = torch.randn(self.rank, self._dim, generator=self._generator)
        basis = F.normalize(basis, dim=-1, eps=1e-8)
        self._basis = basis
        self._learned_direction = torch.zeros(1, self._dim)

    def _cpu_linear(self, module: Any, x: torch.Tensor) -> torch.Tensor | None:
        weight = getattr(module, "weight", None)
        if weight is None:
            return None
        bias = getattr(module, "bias", None)
        bucket = int(self._active_step) // self.weight_sync_interval_steps
        key = (id(module), bucket)
        cached = self._weight_cache.get(key)
        if cached is None:
            stale_keys = [old_key for old_key in self._weight_cache if old_key[0] == id(module)]
            for old_key in stale_keys:
                self._weight_cache.pop(old_key, None)
            weight_cpu = weight.detach().to(device="cpu", dtype=torch.float32)
            bias_cpu = (
                None
                if bias is None
                else bias.detach().to(device="cpu", dtype=torch.float32)
            )
            self._weight_cache[key] = (weight_cpu, bias_cpu)
            self.weight_syncs_total += 1
        else:
            weight_cpu, bias_cpu = cached
        return F.linear(x, weight_cpu, bias_cpu)

    def _roundtrip(self, outer: Any, slot_cpu: torch.Tensor) -> torch.Tensor | None:
        encoder = getattr(outer, "encoder", None)
        decoder = getattr(outer, "decoder", None)
        if encoder is None or decoder is None:
            return None
        decoded = self._cpu_linear(decoder, slot_cpu)
        if decoded is None:
            return None
        rt_raw = self._cpu_linear(encoder, decoded)
        if rt_raw is None:
            return None
        rt = torch.tanh(rt_raw)
        return rt if torch.isfinite(rt).all() else None

    def _cue_direction(
        self,
        outer: Any,
        slot_cpu: torch.Tensor,
        context: dict[str, Any] | None,
    ) -> torch.Tensor | None:
        if not context:
            return None
        cue = context.get("probe_cue")
        if cue is None:
            return None
        cue_cpu = cue.detach().to(device="cpu", dtype=torch.float32)
        if cue_cpu.dim() == 3:
            cue_cpu = cue_cpu.mean(dim=1)
        if cue_cpu.dim() == 1:
            cue_cpu = cue_cpu.unsqueeze(0)
        if cue_cpu.dim() != 2:
            return None
        if cue_cpu.shape[-1] != slot_cpu.shape[-1]:
            cue_proj = getattr(outer, "cue_proj", None)
            if cue_proj is None:
                return None
            projected = self._cpu_linear(cue_proj, cue_cpu)
            if projected is None:
                return None
            cue_cpu = projected
        target = cue_cpu.mean(dim=0, keepdim=True)
        delta = target - slot_cpu
        if not torch.isfinite(delta).all() or delta.norm().item() <= 1e-8:
            return None
        return F.normalize(delta, dim=-1, eps=1e-8)

    def _controller_features(
        self,
        slot_cpu: torch.Tensor,
        context: dict[str, Any] | None,
    ) -> torch.Tensor:
        context = context or {}
        slot_norm = slot_cpu.norm().item() / max(1.0, float(slot_cpu.numel()) ** 0.5)
        def f(name: str, default: float = 0.0) -> float:
            try:
                return float(context.get(name, default))
            except Exception:
                return default
        features = [
            f("marginal_gain"),
            f("utility_ema"),
            f("sharpness"),
            f("activation_drift"),
            f("representation_drift"),
            f("semantic_drift"),
            f("contradiction"),
            f("retrieval_mass"),
            f("peak_utility"),
            f("peak_sharpness"),
            min(8.0, max(0.0, f("frame_age_steps")) / 256.0),
            min(8.0, max(0.0, f("frame_age_seconds")) / 60.0),
            min(8.0, max(0.0, f("stream_id")) / 8.0),
            min(8.0, max(0.0, f("score_count")) / 32.0),
            self.last_best_improvement,
            slot_norm,
        ]
        return torch.tensor(features, dtype=torch.float32).reshape(1, -1)

    def sample_k(
        self,
        *,
        outer: Any,
        slot: torch.Tensor,
        context: dict[str, Any] | None = None,
    ) -> list[tuple[str, torch.Tensor]]:
        if context and "step" in context:
            self._active_step = int(context["step"])
        slot_cpu = slot.detach().to(device="cpu", dtype=torch.float32).reshape(1, -1).clone()
        self._ensure_dim(slot_cpu.shape[-1])
        assert self._basis is not None
        assert self._learned_direction is not None

        candidates: list[tuple[str, torch.Tensor]] = [("identity", slot_cpu)]
        rt = self._roundtrip(outer, slot_cpu)
        directions: list[tuple[str, torch.Tensor]] = []
        controller_dir = self._controller.step(
            self._controller_features(slot_cpu, context),
            slot_dim=slot_cpu.shape[-1],
        )
        if controller_dir.norm().item() > 1e-8:
            directions.append(("controller_full_a", controller_dir))
            directions.append(("controller_full_a_anti", -controller_dir))
        if rt is not None:
            rt_delta = rt - slot_cpu
            if rt_delta.norm().item() > 1e-8:
                directions.append(("roundtrip", F.normalize(rt_delta, dim=-1, eps=1e-8)))
                slot_abs = slot_cpu.abs()
                threshold = slot_abs.mean() + slot_abs.std()
                sharp_mask = (slot_abs > threshold).float()
                sharp = (slot_cpu * sharp_mask + rt * (1.0 - sharp_mask)) - slot_cpu
                if sharp.norm().item() > 1e-8:
                    directions.append(("sharp_preserve", F.normalize(sharp, dim=-1, eps=1e-8)))

        cue_dir = self._cue_direction(outer, slot_cpu, context)
        if cue_dir is not None:
            directions.append(("cue_align", cue_dir))
            directions.append(("cue_anti", -cue_dir))

        if self._learned_direction.norm().item() > 1e-8:
            learned = F.normalize(self._learned_direction, dim=-1, eps=1e-8)
            directions.append(("learned", learned))
            directions.append(("learned_anti", -learned))

        norm = slot_cpu.norm().item() / max(1.0, float(slot_cpu.numel()) ** 0.5)
        amp_base = max(1e-4, self.lr * (norm + 1e-3))
        scales = (0.25, 0.5, 1.0, 1.5)

        for name, direction in directions:
            for scale in scales:
                if len(candidates) >= self.k:
                    break
                cand = slot_cpu + (amp_base * scale) * direction
                if torch.isfinite(cand).all():
                    candidates.append((f"{name}_{scale:g}", cand.detach().clone()))
            if len(candidates) >= self.k:
                break

        while len(candidates) < self.k:
            coeff = torch.randn(1, self.rank, generator=self._generator)
            direction = coeff @ self._basis
            if self._learned_direction.norm().item() > 1e-8:
                direction = direction + 0.35 * F.normalize(
                    self._learned_direction, dim=-1, eps=1e-8
                )
            direction = F.normalize(direction, dim=-1, eps=1e-8)
            jitter = torch.randn(slot_cpu.shape, generator=self._generator) * self.noise_scale
            cand = slot_cpu + amp_base * direction + amp_base * 0.1 * jitter
            if torch.isfinite(cand).all():
                candidates.append((f"proposal_{len(candidates)}", cand.detach().clone()))

        self.samples_total += len(candidates)
        return candidates[: self.k]

    def update(
        self,
        *,
        candidates: list[tuple[str, torch.Tensor]],
        scores: torch.Tensor,
        accepted_index: int,
        structural_accepted: bool,
    ) -> None:
        if not candidates:
            return
        idx = int(accepted_index)
        scores_cpu = scores.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        improvement = (
            float(scores_cpu[idx].item() - scores_cpu[0].item())
            if 0 <= idx < int(scores_cpu.numel())
            else 0.0
        )
        self.updates_total += 1
        self.last_best_index = idx
        self.last_best_name = candidates[idx][0] if 0 <= idx < len(candidates) else ""
        self.last_best_improvement = improvement
        self.improvement_sum += improvement
        abs_improvement = abs(improvement)
        if self.improvement_scale_updates == 0:
            self.improvement_abs_ema = abs_improvement
        else:
            self.improvement_abs_ema = (
                0.95 * self.improvement_abs_ema + 0.05 * abs_improvement
            )
        self.improvement_scale_updates += 1
        scale = max(self.improvement_abs_ema, 1.0e-8)
        improvement_credit = max(-1.0, min(1.0, improvement / scale))
        if structural_accepted:
            self.structural_accepts_total += 1
        else:
            self.structural_rejects_total += 1
        if idx > 0 and improvement > 0.0:
            identity = candidates[0][1].detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
            best = candidates[idx][1].detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
            self._ensure_dim(identity.shape[-1])
            assert self._learned_direction is not None
            delta = best - identity
            self._learned_direction = (
                self.momentum * self._learned_direction
                + (1.0 - self.momentum) * delta
            )
            self._controller.update_feedback(
                identity=identity,
                best=best,
                improvement=max(0.0, improvement_credit),
                structural_accepted=structural_accepted,
            )
            self.positive_updates_total += 1
            if not structural_accepted:
                self.rejected_positive_updates_total += 1
        else:
            if self._learned_direction is not None:
                self._learned_direction = self.momentum * self._learned_direction

    def diagnostics(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "rank": self.rank,
            "lr": self.lr,
            "noise_scale": self.noise_scale,
            "momentum": self.momentum,
            "weight_sync_interval_steps": self.weight_sync_interval_steps,
            "samples_total": self.samples_total,
            "updates_total": self.updates_total,
            "positive_updates_total": self.positive_updates_total,
            "structural_accepts_total": self.structural_accepts_total,
            "structural_rejects_total": self.structural_rejects_total,
            "rejected_positive_updates_total": self.rejected_positive_updates_total,
            "weight_syncs_total": self.weight_syncs_total,
            "improvement_sum": self.improvement_sum,
            "improvement_abs_ema": self.improvement_abs_ema,
            "improvement_scale_updates": self.improvement_scale_updates,
            "last_best_index": self.last_best_index,
            "last_best_name": self.last_best_name,
            "last_best_improvement": self.last_best_improvement,
            "controller": self._controller.diagnostics(),
        }

    def state_dict(self) -> dict[str, Any]:
        """Persist learned CPU proposal state for online eval."""
        return {
            "k": self.k,
            "rank": self.rank,
            "lr": self.lr,
            "noise_scale": self.noise_scale,
            "momentum": self.momentum,
            "weight_sync_interval_steps": self.weight_sync_interval_steps,
            "seed": self._seed,
            "generator_state": self._generator.get_state(),
            "basis": None if self._basis is None else self._basis.detach().cpu(),
            "learned_direction": (
                None
                if self._learned_direction is None
                else self._learned_direction.detach().cpu()
            ),
            "active_step": self._active_step,
            "dim": self._dim,
            "samples_total": self.samples_total,
            "updates_total": self.updates_total,
            "positive_updates_total": self.positive_updates_total,
            "structural_accepts_total": self.structural_accepts_total,
            "structural_rejects_total": self.structural_rejects_total,
            "rejected_positive_updates_total": self.rejected_positive_updates_total,
            "weight_syncs_total": self.weight_syncs_total,
            "improvement_sum": self.improvement_sum,
            "improvement_abs_ema": self.improvement_abs_ema,
            "improvement_scale_updates": self.improvement_scale_updates,
            "last_best_index": self.last_best_index,
            "last_best_name": self.last_best_name,
            "last_best_improvement": self.last_best_improvement,
            "controller": self._controller.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.k = int(state.get("k", self.k))
        self.rank = int(state.get("rank", self.rank))
        self.lr = float(state.get("lr", self.lr))
        self.noise_scale = float(state.get("noise_scale", self.noise_scale))
        self.momentum = float(state.get("momentum", self.momentum))
        self.weight_sync_interval_steps = int(
            state.get("weight_sync_interval_steps", self.weight_sync_interval_steps)
        )
        self._seed = int(state.get("seed", self._seed))
        generator_state = state.get("generator_state")
        if isinstance(generator_state, torch.Tensor):
            self._generator.set_state(generator_state.cpu())
        basis = state.get("basis")
        self._basis = None if basis is None else basis.detach().cpu().float()
        learned = state.get("learned_direction")
        self._learned_direction = None if learned is None else learned.detach().cpu().float()
        self._active_step = int(state.get("active_step", self._active_step))
        self._dim = int(state.get("dim", self._dim))
        self._weight_cache.clear()
        self.samples_total = int(state.get("samples_total", self.samples_total))
        self.updates_total = int(state.get("updates_total", self.updates_total))
        self.positive_updates_total = int(
            state.get("positive_updates_total", self.positive_updates_total)
        )
        self.structural_accepts_total = int(
            state.get("structural_accepts_total", self.structural_accepts_total)
        )
        self.structural_rejects_total = int(
            state.get("structural_rejects_total", self.structural_rejects_total)
        )
        self.rejected_positive_updates_total = int(
            state.get("rejected_positive_updates_total", self.rejected_positive_updates_total)
        )
        self.weight_syncs_total = int(
            state.get("weight_syncs_total", self.weight_syncs_total)
        )
        self.improvement_sum = float(state.get("improvement_sum", self.improvement_sum))
        self.improvement_abs_ema = float(
            state.get("improvement_abs_ema", self.improvement_abs_ema)
        )
        self.improvement_scale_updates = int(
            state.get("improvement_scale_updates", self.improvement_scale_updates)
        )
        self.last_best_index = int(state.get("last_best_index", self.last_best_index))
        self.last_best_name = str(state.get("last_best_name", self.last_best_name))
        self.last_best_improvement = float(
            state.get("last_best_improvement", self.last_best_improvement)
        )
        controller_state = state.get("controller")
        if isinstance(controller_state, dict):
            self._controller.load_state_dict(controller_state)


class MaintenancePolicy:
    """Legacy rule policy and counterfactual trace baseline.

    This is no longer the active owner of maintenance mutations in Exp26. The
    learned Full-A action simplex chooses from learned logits; trace rows keep
    the rule winner as a counterfactual shadow, and tests can still exercise
    the old rule mode explicitly.
    """

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
        capacity_pressure: bool = False,
    ) -> dict[str, float]:
        enough = rec.score_count > 0
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

        release_v = 0.0
        if rec.state == SLOT_QUARANTINED and enough:
            release_v = max(0.0, rec.marginal_gain_ema) + 0.25 * rec.positive_streak

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
            SLOT_RELEASE: release_v,
        }

    def choose(self, rec: SlotRecord, **kwargs: Any) -> str:
        vals = self.action_values(rec, **kwargs)
        return max(vals, key=lambda k: vals[k])


@dataclass
class CommitDecision:
    action: str
    rule_action: str
    confidence: float
    probability: float
    entropy: float
    logits: dict[str, float]
    probabilities: dict[str, float]
    rule_values: dict[str, float]
    vetoed_action: str = ""
    veto_reason: str = ""


class LearnedCommitPolicy:
    """Full-A recurrent simplex over the six maintenance commit actions."""

    input_dim = FullAControllerState.input_dim

    def __init__(
        self,
        *,
        controller_state_dim: int = 32,
        controller_rank: int = 8,
        controller_dt: float = 1.0,
        controller_gamma: float = 0.08,
        controller_target_log_sv: float = -0.05,
        controller_max_state_norm: float = 8.0,
        controller_perturbation_scale: float = 0.25,
        controller_feedback_lr: float = 0.05,
        online_lr: float = 0.05,
        temperature: float = 0.75,
        seed: int = 2718,
    ) -> None:
        self.actions = tuple(SLOT_ACTION_ORDER)
        self.online_lr = max(0.0, float(online_lr))
        self.temperature = max(1.0e-4, float(temperature))
        self._controller = FullAControllerState(
            state_dim=int(controller_state_dim),
            rank=int(controller_rank),
            dt=float(controller_dt),
            gamma=float(controller_gamma),
            max_top_log_sv=float(controller_target_log_sv),
            max_state_norm=float(controller_max_state_norm),
            perturbation_scale=float(controller_perturbation_scale),
            feedback_lr=float(controller_feedback_lr),
            seed=int(seed),
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 211)
        self._bias = torch.zeros(len(self.actions), dtype=torch.float32)
        self._readout_lr = torch.randn(
            self._controller.state_dim,
            len(self.actions),
            generator=gen,
            dtype=torch.float32,
        ) * 0.01
        self.decisions_total = 0
        self.feedback_updates = 0
        self.oracle_accepts = 0
        self.oracle_rejects = 0
        self.safety_vetoes = 0
        self.rule_disagreements = 0
        self.last_action = SLOT_PRESERVE
        self.last_rule_action = SLOT_PRESERVE
        self.last_confidence = 0.0
        self.last_entropy = 0.0
        self.oracle_score_abs_ema = 0.0
        self.oracle_score_scale_updates = 0

    def _features(
        self,
        rec: SlotRecord,
        *,
        capacity_pressure: bool,
        frame_age_steps: int,
        queue_depth: int,
    ) -> torch.Tensor:
        age = max(0, int(rec.last_scored_step) - int(rec.created_step))
        drift = max(
            float(rec.activation_drift_ema),
            float(rec.representation_drift_ema),
            float(rec.semantic_drift_ema),
        )
        values = torch.tensor(
            [
                float(rec.marginal_gain_ema),
                float(rec.utility_ema),
                float(rec.sharpness_ema),
                drift,
                float(rec.contradiction_ema),
                float(rec.retrieval_mass_ema),
                float(rec.peak_utility),
                float(rec.peak_sharpness),
                math.tanh(float(rec.score_count) / 8.0),
                math.tanh(float(age) / 512.0),
                math.tanh(float(rec.negative_streak) / 4.0),
                math.tanh(float(rec.positive_streak) / 4.0),
                1.0 if capacity_pressure else 0.0,
                math.tanh(float(frame_age_steps) / 256.0),
                math.tanh(float(queue_depth) / 32.0),
                1.0 if rec.state == SLOT_QUARANTINED else 0.0,
            ],
            dtype=torch.float32,
        )
        return torch.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

    def choose(
        self,
        rec: SlotRecord,
        *,
        rule_values: dict[str, float],
        capacity_pressure: bool,
        frame_age_steps: int,
        queue_depth: int,
    ) -> CommitDecision:
        features = self._features(
            rec,
            capacity_pressure=capacity_pressure,
            frame_age_steps=frame_age_steps,
            queue_depth=queue_depth,
        )
        controller_logits = self._controller.step(
            features,
            slot_dim=len(self.actions),
        ).flatten()
        hidden = self._controller.hidden_state()
        learned_logits = (
            controller_logits.reshape(-1)
            + (hidden @ self._readout_lr).reshape(-1)
            + self._bias
        )
        logits = learned_logits
        action, vetoed, reason = self._select_legal_action(
            rec=rec,
            logits=logits,
            capacity_pressure=capacity_pressure,
        )
        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=0)
        entropy = float(-(probs * probs.clamp_min(1.0e-9).log()).sum().item())
        idx = self.actions.index(action)
        top2 = torch.topk(probs, k=min(2, probs.numel())).values
        confidence = (
            float((top2[0] - top2[1]).item())
            if top2.numel() >= 2
            else float(top2[0].item())
        )
        rule_action = max(rule_values, key=lambda k: rule_values[k])
        self.decisions_total += 1
        self.last_action = action
        self.last_rule_action = rule_action
        self.last_confidence = confidence
        self.last_entropy = entropy
        if action != rule_action:
            self.rule_disagreements += 1
        if vetoed:
            self.safety_vetoes += 1
        return CommitDecision(
            action=action,
            rule_action=rule_action,
            confidence=confidence,
            probability=float(probs[idx].item()),
            entropy=entropy,
            logits={a: float(logits[i].item()) for i, a in enumerate(self.actions)},
            probabilities={a: float(probs[i].item()) for i, a in enumerate(self.actions)},
            rule_values={a: float(rule_values.get(a, 0.0)) for a in self.actions},
            vetoed_action=vetoed,
            veto_reason=reason,
        )

    def _select_legal_action(
        self,
        *,
        rec: SlotRecord,
        logits: torch.Tensor,
        capacity_pressure: bool,
    ) -> tuple[str, str, str]:
        order = torch.argsort(logits, descending=True).tolist()
        vetoed = ""
        reason = ""
        for idx in order:
            action = self.actions[int(idx)]
            ok, why = self._legal_action(rec, action, capacity_pressure)
            if ok:
                return action, vetoed, reason
            if not vetoed:
                vetoed = action
                reason = why
        return SLOT_PRESERVE, vetoed or SLOT_PRESERVE, reason or "no_legal_action"

    @staticmethod
    def _legal_action(
        rec: SlotRecord,
        action: str,
        capacity_pressure: bool,
    ) -> tuple[bool, str]:
        if rec.score_count <= 0 and action != SLOT_PRESERVE:
            return False, "unscored_slot"
        if action == SLOT_RELEASE and rec.state != SLOT_QUARANTINED:
            return False, "not_quarantined"
        if rec.state == SLOT_QUARANTINED and action == SLOT_QUARANTINE:
            return False, "already_quarantined"
        if action == SLOT_EVICT and not capacity_pressure:
            return False, "below_capacity"
        return True, ""

    def apply_feedback(
        self,
        decision: CommitDecision,
        *,
        oracle_score: float | None,
        accepted: bool,
        structural: bool,
    ) -> bool:
        if self.online_lr <= 0.0:
            return False
        if oracle_score is None:
            return False
        action_idx = self.actions.index(decision.action)
        p = torch.tensor(
            [decision.probabilities.get(a, 0.0) for a in self.actions],
            dtype=torch.float32,
        )
        one_hot = torch.zeros_like(p)
        one_hot[action_idx] = 1.0
        abs_score = abs(float(oracle_score))
        if self.oracle_score_scale_updates == 0:
            self.oracle_score_abs_ema = abs_score
        else:
            self.oracle_score_abs_ema = (
                0.95 * self.oracle_score_abs_ema + 0.05 * abs_score
            )
        self.oracle_score_scale_updates += 1
        scale = max(self.oracle_score_abs_ema, 1.0e-8)
        score_mag = min(1.0, abs_score / scale)
        advantage = score_mag if accepted else -score_mag
        grad = (one_hot - p) * float(advantage) * self.online_lr
        self._bias = torch.clamp(self._bias + grad, -4.0, 4.0)
        hidden = self._controller.hidden_state().flatten().unsqueeze(1)
        self._readout_lr = torch.clamp(
            self._readout_lr + hidden @ grad.unsqueeze(0),
            -1.0,
            1.0,
        )
        identity = p.reshape(1, -1)
        best = one_hot.reshape(1, -1) if accepted else torch.zeros_like(identity)
        self._controller.update_feedback(
            identity=identity,
            best=best,
            improvement=max(0.0, float(advantage)),
            structural_accepted=bool(structural and accepted),
        )
        self.feedback_updates += 1
        if accepted:
            self.oracle_accepts += 1
        else:
            self.oracle_rejects += 1
        return True

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": "full_a_action_simplex",
            "actions": list(self.actions),
            "online_lr": self.online_lr,
            "temperature": self.temperature,
            "decisions_total": self.decisions_total,
            "feedback_updates": self.feedback_updates,
            "oracle_accepts": self.oracle_accepts,
            "oracle_rejects": self.oracle_rejects,
            "safety_vetoes": self.safety_vetoes,
            "rule_disagreements": self.rule_disagreements,
            "last_action": self.last_action,
            "last_rule_action": self.last_rule_action,
            "last_confidence": self.last_confidence,
            "last_entropy": self.last_entropy,
            "oracle_score_abs_ema": self.oracle_score_abs_ema,
            "oracle_score_scale_updates": self.oracle_score_scale_updates,
            "controller": self._controller.diagnostics(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "online_lr": self.online_lr,
            "temperature": self.temperature,
            "bias": self._bias.detach().cpu(),
            "readout_lr": self._readout_lr.detach().cpu(),
            "controller": self._controller.state_dict(),
            "decisions_total": self.decisions_total,
            "feedback_updates": self.feedback_updates,
            "oracle_accepts": self.oracle_accepts,
            "oracle_rejects": self.oracle_rejects,
            "safety_vetoes": self.safety_vetoes,
            "rule_disagreements": self.rule_disagreements,
            "last_action": self.last_action,
            "last_rule_action": self.last_rule_action,
            "last_confidence": self.last_confidence,
            "last_entropy": self.last_entropy,
            "oracle_score_abs_ema": self.oracle_score_abs_ema,
            "oracle_score_scale_updates": self.oracle_score_scale_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online_lr = float(state.get("online_lr", self.online_lr))
        self.temperature = float(state.get("temperature", self.temperature))
        bias = state.get("bias")
        if torch.is_tensor(bias) and bias.numel() == len(self.actions):
            self._bias = bias.detach().cpu().float().reshape(-1)
        readout = state.get("readout_lr")
        if torch.is_tensor(readout) and tuple(readout.shape) == tuple(self._readout_lr.shape):
            self._readout_lr = readout.detach().cpu().float()
        controller_state = state.get("controller")
        if isinstance(controller_state, dict):
            self._controller.load_state_dict(controller_state)
        self.decisions_total = int(state.get("decisions_total", self.decisions_total))
        self.feedback_updates = int(
            state.get("feedback_updates", self.feedback_updates)
        )
        self.oracle_accepts = int(state.get("oracle_accepts", self.oracle_accepts))
        self.oracle_rejects = int(state.get("oracle_rejects", self.oracle_rejects))
        self.safety_vetoes = int(state.get("safety_vetoes", self.safety_vetoes))
        self.rule_disagreements = int(
            state.get("rule_disagreements", self.rule_disagreements)
        )
        self.last_action = str(state.get("last_action", self.last_action))
        self.last_rule_action = str(
            state.get("last_rule_action", self.last_rule_action)
        )
        self.last_confidence = float(state.get("last_confidence", self.last_confidence))
        self.last_entropy = float(state.get("last_entropy", self.last_entropy))
        self.oracle_score_abs_ema = float(
            state.get("oracle_score_abs_ema", self.oracle_score_abs_ema)
        )
        self.oracle_score_scale_updates = int(
            state.get("oracle_score_scale_updates", self.oracle_score_scale_updates)
        )


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
        trace_flush_rows: int = 256,
        probe_chunk_size: int = 16,
        scoring_mode: str = "proxy",
        oracle_confirm_top_k: int = 32,
        oracle_variant_chunk_size: int = 1,
        drift_threshold: float = 0.3,
        repr_drift_threshold: float = 0.2,
        refresh_lr: float = 0.1,
        refresh_candidate_count: int = 16,
        refresh_proposal_rank: int = 8,
        refresh_proposal_noise_scale: float = 0.04,
        refresh_proposal_momentum: float = 0.9,
        refresh_proposal_weight_sync_interval_steps: int = 64,
        refresh_candidate_variant_chunk_size: int = 16,
        refresh_proposal_seed: int = 1729,
        controller_state_dim: int = 32,
        controller_rank: int = 8,
        controller_dt: float = 1.0,
        controller_gamma: float = 0.08,
        controller_target_log_sv: float = -0.05,
        controller_max_state_norm: float = 8.0,
        controller_perturbation_scale: float = 0.25,
        controller_feedback_lr: float = 0.05,
        quarantine_threshold: float = -0.01,
        max_quarantined: int = 8,
        distill_peak_threshold: float = 0.04,
        peak_preserve_utility_threshold: float = 0.20,
        peak_preserve_sharpness_threshold: float = 0.20,
        useful_threshold: float = 0.005,
        probe_buffer_size: int = 32,
        frame_ttl_steps: int = 256,
        slot_work_chunk_size: int = 64,
        action_agreement_count: int = 2,
        commit_policy: str = "learned",
        commit_online_lr: float = 0.05,
        commit_temperature: float = 0.75,
        arm_runtime_enabled: bool = False,
        arm_runtime_namespace: str | None = None,
        evidence_engine_enabled: bool = False,
        evidence_engine_d_model: int = 384,
        evidence_engine_lanes: int = 8,
    ) -> None:
        if action_mode not in {"active", "shadow"}:
            raise ValueError("action_mode must be 'active' or 'shadow'")
        if scoring_mode not in {"proxy", "oracle"}:
            raise ValueError("scoring_mode must be 'proxy' or 'oracle'")
        if commit_policy not in {"learned", "rule"}:
            raise ValueError("commit_policy must be 'learned' or 'rule'")
        self._action_mode = str(action_mode)
        self._scoring_mode = str(scoring_mode)
        self._commit_policy_mode = str(commit_policy)
        self._memory_streams = max(1, int(memory_streams))
        self._threshold = float(eviction_threshold)
        self._ema_beta = float(eviction_ema_beta)
        self._min_age = int(min_slot_age_steps)
        self._max_seconds = float(max_seconds_per_tick)
        self._probe_chunk_size = int(probe_chunk_size)
        self._oracle_confirm_top_k = max(0, int(oracle_confirm_top_k))
        self._oracle_variant_chunk_size = max(1, int(oracle_variant_chunk_size))
        self._drift_threshold = float(drift_threshold)
        self._repr_drift_threshold = float(repr_drift_threshold)
        self._refresh_lr = float(refresh_lr)
        self._refresh_candidate_count = max(2, int(refresh_candidate_count))
        self._refresh_candidate_variant_chunk_size = max(
            1, int(refresh_candidate_variant_chunk_size)
        )
        self._quarantine_threshold = float(quarantine_threshold)
        self._max_quarantined = int(max_quarantined)
        self._distill_peak_threshold = float(distill_peak_threshold)
        self._peak_preserve_utility_threshold = float(peak_preserve_utility_threshold)
        self._peak_preserve_sharpness_threshold = float(peak_preserve_sharpness_threshold)
        self._useful_threshold = float(useful_threshold)
        self._probe_buffer_size = max(1, int(probe_buffer_size))
        self._frame_ttl_steps = max(0, int(frame_ttl_steps))
        self._slot_work_chunk_size = max(1, int(slot_work_chunk_size))
        self._action_agreement_count = max(1, int(action_agreement_count))
        self._commit_online_lr = max(0.0, float(commit_online_lr))
        self._commit_temperature = max(1.0e-4, float(commit_temperature))
        self._refresh_proposal_model = CpuRefreshProposalModel(
            k=self._refresh_candidate_count,
            rank=max(1, int(refresh_proposal_rank)),
            lr=self._refresh_lr,
            noise_scale=float(refresh_proposal_noise_scale),
            momentum=float(refresh_proposal_momentum),
            weight_sync_interval_steps=int(refresh_proposal_weight_sync_interval_steps),
            seed=int(refresh_proposal_seed),
            controller_state_dim=int(controller_state_dim),
            controller_rank=int(controller_rank),
            controller_dt=float(controller_dt),
            controller_gamma=float(controller_gamma),
            controller_target_log_sv=float(controller_target_log_sv),
            controller_max_state_norm=float(controller_max_state_norm),
            controller_perturbation_scale=float(controller_perturbation_scale),
            controller_feedback_lr=float(controller_feedback_lr),
            controller_seed=int(refresh_proposal_seed) + 97,
        )
        self._arm_runtime_enabled_requested = bool(arm_runtime_enabled)
        self._arm_runtime_active = False
        self._arm_runtime_error = ""
        self._arm_scheduler: Any | None = None
        self._arm_job_ring: Any | None = None
        self._arm_job_worker_ring: Any | None = None
        self._arm_result_ring: Any | None = None
        self._arm_result_worker_ring: Any | None = None
        self._arm_job_ring_name = ""
        self._arm_result_ring_name = ""
        self._arm_jobs_pushed = 0
        self._arm_jobs_popped = 0
        self._arm_job_ring_drops = 0
        self._arm_result_ring_pushes = 0
        self._arm_result_ring_pops = 0
        self._arm_result_ring_drops = 0
        self._active_arm_job: dict[str, Any] | None = None
        if self._arm_runtime_enabled_requested:
            self._init_arm_runtime(arm_runtime_namespace)

        # CPU evidence engine (typed firewall around CPU-side cue ingest)
        self._evidence_engine_enabled_requested = bool(evidence_engine_enabled)
        self._evidence_engine_active = False
        self._evidence_engine_error = ""
        self._evidence_engine_d_model = max(1, int(evidence_engine_d_model))
        self._evidence_engine_lanes = max(1, int(evidence_engine_lanes))
        self._evidence_engine: Any | None = None
        self._evidence_cue_buffer: torch.Tensor | None = None
        self._evidence_targets_buffer: torch.Tensor | None = None
        self._evidence_frames_ingested = 0
        self._evidence_ingest_errors_total = 0
        self._evidence_lm_head_refreshes_total = 0
        self._evidence_lm_head_refresh_errors_total = 0
        self._evidence_last_lm_head_refresh_step: int = -1
        self._evidence_last_lm_head_refresh_error = ""
        self._evidence_cue_populated_total = 0
        self._evidence_cue_skipped_no_cue_total = 0
        self._evidence_cue_skipped_shape_total = 0
        self._evidence_cue_populate_errors_total = 0
        if self._evidence_engine_enabled_requested:
            self._init_evidence_engine()

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
        self._refresh_candidate_proposals_total: int = 0
        self._refresh_candidate_proposal_seconds_total: float = 0.0
        self._refresh_candidate_oracle_batches_total: int = 0
        self._refresh_candidate_oracle_chunks_total: int = 0
        self._refresh_candidate_oracle_variants_total: int = 0
        self._refresh_candidate_oracle_seconds_total: float = 0.0
        self._refresh_candidate_accepts_total: int = 0
        self._refresh_candidate_rejects_total: int = 0
        self._last_refresh_candidate_count: int = 0
        self._last_refresh_candidate_device: str = ""
        self._last_refresh_candidate_best_name: str = ""
        self._last_refresh_candidate_best_index: int = 0
        self._last_refresh_candidate_best_improvement: float = 0.0
        self._last_refresh_candidate_oracle_chunks: int = 0
        self._last_refresh_candidate_oracle_variants: int = 0
        self._distills_total: int = 0
        self._prototype_distills_total: int = 0
        self._prototype_distill_skips_total: int = 0
        self._last_prototype_distill_bucket: int = -1
        self._last_prototype_distill_reason: str = ""
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
        self._last_oracle_score_by_slot: dict[int, float] = {}
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
            "oracle_score": 0.0,
            "action": 0.0,
        }
        self._stage_seconds_total: dict[str, float] = dict(self._last_stage_seconds)
        self._oracle_seconds_total: float = 0.0
        self._last_oracle_seconds: float = 0.0
        self._oracle_scored_slots_total: int = 0
        self._oracle_direct_confirmations_total: int = 0
        self._oracle_scoring_seconds_total: float = 0.0
        self._last_oracle_scoring_seconds: float = 0.0
        self._slot_last_scored_step: dict[int, int] = {}
        self._last_visible_slots: int = 0
        self._last_untouched_slots: int = 0
        self._last_max_untouched_steps: int = 0

        # Quarantine tracking (by slot_id when using SlotTable, by index otherwise)
        self._quarantined: set[int] = set()
        self._action_evidence: dict[int, tuple[str, int, int, int]] = {}

        # Trace
        self._trace_path = None if trace_path in (None, "") else Path(str(trace_path))
        self._trace_max_rows = max(0, int(trace_max_rows))
        self._trace_flush_rows = max(0, int(trace_flush_rows))
        self._trace_rows_written = 0
        self._trace_buffer: list[str] = []

        # Policy
        self._policy = MaintenancePolicy()
        self._commit_policy = LearnedCommitPolicy(
            controller_state_dim=int(controller_state_dim),
            controller_rank=int(controller_rank),
            controller_dt=float(controller_dt),
            controller_gamma=float(controller_gamma),
            controller_target_log_sv=float(controller_target_log_sv),
            controller_max_state_norm=float(controller_max_state_norm),
            controller_perturbation_scale=float(controller_perturbation_scale),
            controller_feedback_lr=float(controller_feedback_lr),
            online_lr=self._commit_online_lr,
            temperature=self._commit_temperature,
            seed=int(refresh_proposal_seed) + 313,
        )
        self._commit_decisions_total = 0
        self._commit_rule_disagreements_total = 0
        self._commit_feedback_updates_total = 0

    def _policy_kwargs(self, outer: Any) -> dict[str, Any]:
        return dict(
            eviction_threshold=self._threshold,
            useful_threshold=self._useful_threshold,
            drift_threshold=self._drift_threshold,
            quarantine_threshold=self._quarantine_threshold,
            distill_peak_threshold=self._distill_peak_threshold,
            peak_preserve_utility_threshold=self._peak_preserve_utility_threshold,
            peak_preserve_sharpness_threshold=self._peak_preserve_sharpness_threshold,
            min_age=self._min_age,
            capacity_pressure=self._capacity_pressure(outer),
        )

    def _choose_commit_decision(
        self,
        rec: SlotRecord,
        policy_kwargs: dict[str, Any],
    ) -> CommitDecision:
        values = self._policy.action_values(rec, **policy_kwargs)
        rule_action = max(values, key=lambda k: values[k])
        if self._commit_policy_mode == "rule":
            return CommitDecision(
                action=rule_action,
                rule_action=rule_action,
                confidence=1.0,
                probability=1.0,
                entropy=0.0,
                logits={a: float(values.get(a, 0.0)) for a in SLOT_ACTION_ORDER},
                probabilities={
                    a: 1.0 if a == rule_action else 0.0 for a in SLOT_ACTION_ORDER
                },
                rule_values={a: float(values.get(a, 0.0)) for a in SLOT_ACTION_ORDER},
            )
        decision = self._commit_policy.choose(
            rec,
            rule_values=values,
            capacity_pressure=bool(policy_kwargs.get("capacity_pressure", False)),
            frame_age_steps=self._last_frame_age_steps,
            queue_depth=self._last_queue_depth,
        )
        self._commit_decisions_total += 1
        if decision.action != decision.rule_action:
            self._commit_rule_disagreements_total += 1
        return decision

    def _commit_action_ready(
        self,
        slot_id: int,
        action: str,
        rec: SlotRecord | None,
    ) -> bool:
        if self._commit_policy_mode == "learned":
            return True
        return self._has_action_agreement(slot_id, action, rec)

    def _record_commit_feedback(
        self,
        decision: CommitDecision,
        *,
        accepted: bool,
        structural: bool,
        slot_id: int,
    ) -> None:
        if self._commit_policy_mode != "learned":
            return
        updated = self._commit_policy.apply_feedback(
            decision,
            oracle_score=self._last_oracle_score_by_slot.get(int(slot_id)),
            accepted=bool(accepted),
            structural=bool(structural),
        )
        if updated:
            self._commit_feedback_updates_total += 1

    def _init_arm_runtime(self, namespace: str | None) -> None:
        """Create the native CPU conductor and its local shm ring boundary."""
        try:
            ns_raw = namespace or f"{os.getpid()}_{id(self) & 0xFFFF:x}"
            ns_text = str(ns_raw)
            ns_clean = "".join(ch for ch in ns_text if ch.isalnum())[:16] or "arm"
            ns_digest = hashlib.blake2s(
                ns_text.encode("utf-8"), digest_size=4
            ).hexdigest()
            ns = f"{ns_clean}{ns_digest}"
            self._arm_job_ring_name = f"/ccarmj_{ns}"
            self._arm_result_ring_name = f"/ccarmr_{ns}"
            try:
                _ext.ShmRingArmMaintenanceJob.unlink(self._arm_job_ring_name)
            except Exception:
                pass
            try:
                _ext.ShmRingArmMaintenanceResult.unlink(self._arm_result_ring_name)
            except Exception:
                pass
            self._arm_scheduler = _ext.ArmMaintenanceScheduler(
                int(self._memory_streams),
                int(self._slot_work_chunk_size),
                int(self._frame_ttl_steps),
                int(self._probe_buffer_size),
            )
            self._arm_job_ring = _ext.ShmRingArmMaintenanceJob.create(
                self._arm_job_ring_name
            )
            self._arm_job_worker_ring = _ext.ShmRingArmMaintenanceJob.attach(
                self._arm_job_ring_name
            )
            self._arm_result_ring = _ext.ShmRingArmMaintenanceResult.create(
                self._arm_result_ring_name
            )
            self._arm_result_worker_ring = _ext.ShmRingArmMaintenanceResult.attach(
                self._arm_result_ring_name
            )
            self._arm_runtime_active = True
        except Exception as exc:
            self._arm_runtime_active = False
            self._arm_runtime_error = f"{exc.__class__.__name__}: {exc}"
            self._arm_scheduler = None
            self._arm_job_ring = None
            self._arm_job_worker_ring = None
            self._arm_result_ring = None
            self._arm_result_worker_ring = None
            raise RuntimeError(
                f"ARM runtime requested but failed to initialize: {self._arm_runtime_error}"
            ) from exc

    def _init_evidence_engine(self) -> None:
        """Construct the CPU evidence engine and its pinned cue buffers.

        Mirrors the ARM-runtime init pattern: opt-in via the
        ``evidence_engine_enabled`` flag, fail-loud if requested but
        unavailable. The engine is a typed firewall — the only Python
        surface accepts a fixed-shape pinned bf16 cue digest, by design.
        """
        try:
            engine = _ext.CpuEvidenceEngine(
                lanes=int(self._evidence_engine_lanes),
                d_model=int(self._evidence_engine_d_model),
            )
            cue_buffer = torch.zeros(
                (32, int(self._evidence_engine_d_model)),
                dtype=torch.bfloat16,
            ).pin_memory()
            targets_buffer = torch.zeros(32, dtype=torch.int32).pin_memory()
            self._evidence_engine = engine
            self._evidence_cue_buffer = cue_buffer
            self._evidence_targets_buffer = targets_buffer
            self._evidence_engine_active = True
        except Exception as exc:
            self._evidence_engine_active = False
            self._evidence_engine_error = f"{exc.__class__.__name__}: {exc}"
            self._evidence_engine = None
            self._evidence_cue_buffer = None
            self._evidence_targets_buffer = None
            raise RuntimeError(
                "CpuEvidenceEngine requested but failed to initialize: "
                f"{self._evidence_engine_error}"
            ) from exc

    def _populate_evidence_cue(
        self,
        *,
        cue: torch.Tensor | None,
        input_ids: torch.Tensor,
    ) -> None:
        """Copy the trunk's cue activations + token ids into pinned buffers.

        The engine's binding contract is bf16[T_PROBE_MAX, D] cue + i32
        [T_PROBE_MAX] targets. The probe frame's ``cue`` may be ``None``
        (no encoded cue available), 2-D ``[T, D]``, or 3-D ``[B, T, D]``
        with B == 1. Anything else is a shape we cannot honestly project
        onto the engine's contract, so we fall back to the zero buffer
        and bump a counter rather than block the maintenance loop.
        """
        cue_buffer = self._evidence_cue_buffer
        targets_buffer = self._evidence_targets_buffer
        if cue_buffer is None or targets_buffer is None:
            return
        t_max, d_model = int(cue_buffer.shape[0]), int(cue_buffer.shape[1])
        if cue is None:
            cue_buffer.zero_()
            targets_buffer.zero_()
            self._evidence_cue_skipped_no_cue_total += 1
            return
        try:
            cue_cpu = cue.detach().to(device="cpu", dtype=torch.bfloat16)
            if cue_cpu.dim() == 3 and cue_cpu.shape[0] >= 1:
                cue_cpu = cue_cpu[0]
            if cue_cpu.dim() != 2 or int(cue_cpu.shape[1]) != d_model:
                cue_buffer.zero_()
                targets_buffer.zero_()
                self._evidence_cue_skipped_shape_total += 1
                return
            ids_cpu = input_ids.detach().to(device="cpu", dtype=torch.int32)
            if ids_cpu.dim() == 2 and ids_cpu.shape[0] >= 1:
                ids_cpu = ids_cpu[0]
            t_used = min(int(cue_cpu.shape[0]), int(ids_cpu.shape[0]), t_max)
            cue_buffer.zero_()
            targets_buffer.zero_()
            if t_used > 0:
                cue_buffer[:t_used].copy_(cue_cpu[:t_used])
                targets_buffer[:t_used].copy_(ids_cpu[:t_used])
            self._evidence_cue_populated_total += 1
        except Exception:
            cue_buffer.zero_()
            targets_buffer.zero_()
            self._evidence_cue_populate_errors_total += 1

    def refresh_evidence_weights(
        self,
        *,
        norm_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        step: int = -1,
    ) -> bool:
        """Push the trunk's RMSNorm + LM head into the CPU evidence engine.

        Caller owns cadence — pair this with the GPU3 teacher-snapshot
        refresh so the cue NLL the engine computes matches the trunk
        weights GPU3 just adopted. ``norm_weight`` is the RMSNorm scale
        ([d_model] f32) and ``lm_head_weight`` is the raw lm_head matrix
        ([V, D] in any dtype/device). The engine wants the VNNI-packed
        bf16 form, so we convert here.

        Returns True if the refresh actually landed; False if the engine
        is disabled or AMX is unavailable.
        """
        if self._evidence_engine is None:
            return False
        try:
            norm_cpu = (
                norm_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
            )
            head_cpu = lm_head_weight.detach().to(device="cpu", dtype=torch.bfloat16)
            head_t = head_cpu.t().contiguous()
            head_vnni = _ext.amx_pack_b_vnni(head_t).contiguous()
            self._evidence_engine.set_lm_head(norm_cpu, head_vnni)
            self._evidence_lm_head_refreshes_total += 1
            self._evidence_last_lm_head_refresh_step = int(step)
            self._evidence_last_lm_head_refresh_error = ""
            return True
        except Exception as exc:
            self._evidence_lm_head_refresh_errors_total += 1
            self._evidence_last_lm_head_refresh_error = (
                f"{exc.__class__.__name__}: {exc}"
            )
            raise

    def close(self) -> None:
        """Best-effort cleanup for ARM runtime shm names."""
        if self._evidence_engine is not None:
            try:
                self._evidence_engine.shutdown()
            except Exception:
                pass
            self._evidence_engine = None
            self._evidence_cue_buffer = None
            self._evidence_targets_buffer = None
            self._evidence_engine_active = False
        for cls_name, name in (
            ("ShmRingArmMaintenanceJob", self._arm_job_ring_name),
            ("ShmRingArmMaintenanceResult", self._arm_result_ring_name),
        ):
            if not name:
                continue
            try:
                getattr(_ext, cls_name).unlink(name)
            except Exception:
                pass
        self._arm_job_ring = None
        self._arm_job_worker_ring = None
        self._arm_result_ring = None
        self._arm_result_worker_ring = None

    def __del__(self) -> None:  # pragma: no cover - destructor best effort
        try:
            self.close()
        except Exception:
            pass

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
        if self._arm_runtime_active and self._arm_scheduler is not None:
            try:
                self._arm_scheduler.ingest_frame(
                    int(frame.frame_id),
                    int(frame.step),
                    int(frame.stream_id),
                    (
                        (1 << 64) - 1
                        if frame.cache_read_cutoff is None
                        else int(frame.cache_read_cutoff)
                    ),
                )
            except Exception as exc:
                self._arm_runtime_active = False
                self._arm_runtime_error = f"{exc.__class__.__name__}: {exc}"
                raise RuntimeError(
                    f"ARM runtime requested but frame ingest failed: {self._arm_runtime_error}"
                ) from exc
        self._last_queue_depth = len(self._probe_frames)
        self._queue_depth_sum += self._last_queue_depth
        self._queue_depth_samples += 1
        self._queue_depth_max = max(self._queue_depth_max, self._last_queue_depth)

        if (
            self._evidence_engine is not None
            and self._evidence_cue_buffer is not None
            and self._evidence_targets_buffer is not None
        ):
            self._populate_evidence_cue(cue=cue_d, input_ids=input_ids_d)
            try:
                self._evidence_engine.ingest_frame(
                    self._evidence_cue_buffer,
                    self._evidence_targets_buffer,
                    frame_id=int(frame.frame_id),
                    step=int(frame.step),
                    stream_id=int(frame.stream_id),
                )
                self._evidence_frames_ingested += 1
            except Exception as exc:
                self._evidence_ingest_errors_total += 1
                self._evidence_engine_error = f"{exc.__class__.__name__}: {exc}"
                raise

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

    def latest_probe_step(self) -> int | None:
        """Return the newest cached probe frame's train-step timestamp."""
        if not self._probe_frames:
            return None
        return int(self._probe_step)

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

    def _frame_by_id(self, frame_id: int) -> ProbeFrame | None:
        for frame in self._probe_frames:
            if int(frame.frame_id) == int(frame_id):
                return frame
        return None

    def _select_native_job(
        self,
        *,
        outer: Any,
        step: int,
    ) -> tuple[ProbeFrame | None, list[int]]:
        if (
            not self._arm_runtime_active
            or self._arm_scheduler is None
            or self._arm_job_ring is None
            or self._arm_job_worker_ring is None
        ):
            return None, []
        visible = [int(x) for x in self._visible_slot_indices(outer)]
        job = self._arm_scheduler.next_job(int(step), visible)
        if job is None:
            return None, []
        job_d = dict(job)
        if not bool(self._arm_job_ring.push(job_d)):
            self._arm_job_ring_drops += 1
            self._record_arm_runtime_result(
                job=job_d,
                step=step,
                slots_scored=0,
                confirmed_delta=0,
                rejected_delta=0,
                frame_age_seconds=0.0,
                status=1,
                through_ring=False,
            )
            return None, []
        self._arm_jobs_pushed += 1
        popped = self._arm_job_worker_ring.pop()
        if popped is None:
            self._record_arm_runtime_result(
                job=job_d,
                step=step,
                slots_scored=0,
                confirmed_delta=0,
                rejected_delta=0,
                frame_age_seconds=0.0,
                status=2,
                through_ring=False,
            )
            return None, []
        self._arm_jobs_popped += 1
        job_d = dict(popped)
        frame = self._frame_by_id(int(job_d["frame_id"]))
        if frame is None:
            self._record_arm_runtime_result(
                job=job_d,
                step=step,
                slots_scored=0,
                confirmed_delta=0,
                rejected_delta=0,
                frame_age_seconds=0.0,
                status=3,
                through_ring=False,
            )
            return None, []
        slot_count = min(
            int(job_d["slot_count"]),
            int(_ext.wire_event_constants()["ARM_MAINTENANCE_SLOT_CAPACITY"]),
        )
        slot_work = [int(x) for x in list(job_d["slot_ids"])[:slot_count]]
        slot_work = [x for x in slot_work if x != (1 << 64) - 1]
        self._active_arm_job = job_d
        return frame, slot_work

    def _select_frame_and_slot_work(
        self,
        *,
        outer: Any,
        step: int,
    ) -> tuple[ProbeFrame | None, list[int]]:
        if self._arm_runtime_active:
            frame, work = self._select_native_job(outer=outer, step=step)
            if frame is not None or work:
                return frame, work
            # If native scheduling is enabled and simply has no job ready, do
            # not fall through to the legacy Python scheduler — that would hide
            # a conductor/worker bug in telemetry.
            return None, []
        if self._arm_runtime_enabled_requested:
            raise RuntimeError("ARM runtime was requested but is not active")
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

        # Run GPU3 maintenance scoring.  In ``proxy`` mode this is the cheap
        # saliency map plus later oracle confirmation.  In ``oracle`` mode
        # GPU3 does the real force_on/no-sidecar/hide-slot physics for the
        # scheduled slot work immediately; the CPU-side loop remains the
        # scheduler/telemetry/action-evidence plane.
        probe_t0 = time.monotonic()
        if self._scoring_mode == "oracle":
            cf = self._oracle_counterfactual_probe(
                model=model,
                outer=outer,
                frame=frame,
                slot_work=slot_work,
            )
        else:
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
            self._complete_arm_runtime_job(
                step=step,
                cf=cf,
                frame_age_seconds=frame_age_seconds,
            )
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

                    if rec.state == SLOT_WARMING and rec.score_count > 0:
                        rec.state = SLOT_ACTIVE
        self._last_stage_seconds["ema"] = time.monotonic() - ema_t0
        self._stage_seconds_total["ema"] += self._last_stage_seconds["ema"]

        elapsed = time.monotonic() - t0
        if elapsed > self._max_seconds:
            self._complete_arm_runtime_job(
                step=step,
                cf=cf,
                frame_age_seconds=frame_age_seconds,
                status=1,
            )
            return TickResult()

        # Classify and act.  Shadow mode is the experiment-safe lane: it
        # computes the same policy decisions and emits telemetry, but it never
        # mutates the table.  This lets CRCT vs CRCT+maintenance be measured
        # without conflating the first run with a new controller.
        if self._action_mode == "shadow":
            action_t0 = time.monotonic()
            confirmed_before = self._oracle_confirmed_actions_total
            rejected_before = self._oracle_rejected_actions_total
            self._classify_shadow(model=model, outer=outer, step=step, cf=cf, t0=t0)
            self._last_stage_seconds["action"] = time.monotonic() - action_t0
            self._stage_seconds_total["action"] += self._last_stage_seconds["action"]
            self._complete_arm_runtime_job(
                step=step,
                cf=cf,
                frame_age_seconds=frame_age_seconds,
                confirmed_delta=(
                    self._oracle_confirmed_actions_total - confirmed_before
                ),
                rejected_delta=(
                    self._oracle_rejected_actions_total - rejected_before
                ),
            )
            return TickResult()

        action_t0 = time.monotonic()
        confirmed_before = self._oracle_confirmed_actions_total
        rejected_before = self._oracle_rejected_actions_total
        result = self._classify_and_act(
            model=model, outer=outer, step=step, cf=cf,
            slot_marginals=slot_marginals, sharpness_per_slot=sharpness_per_slot,
            t0=t0,
        )
        self._last_stage_seconds["action"] = time.monotonic() - action_t0
        self._stage_seconds_total["action"] += self._last_stage_seconds["action"]
        self._complete_arm_runtime_job(
            step=step,
            cf=cf,
            frame_age_seconds=frame_age_seconds,
            confirmed_delta=self._oracle_confirmed_actions_total - confirmed_before,
            rejected_delta=self._oracle_rejected_actions_total - rejected_before,
        )
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
        policy_kwargs = self._policy_kwargs(outer)
        actions: dict[int, str] = {}
        decisions: dict[int, CommitDecision] = {}
        for phys_idx in cf.slot_indices:
            sid = table.physical_to_slot_id(phys_idx)
            if sid is None:
                continue
            rec = table.record(sid)
            if rec is None:
                continue
            decision = self._choose_commit_decision(rec, policy_kwargs)
            actions[sid] = decision.action
            decisions[sid] = decision
        confirmations = self._confirm_actions_with_oracle(
            model=model, outer=outer, actions=actions, cf=cf, t0=t0
        )
        for sid, action in actions.items():
            rec = table.record(sid)
            if rec is None:
                continue
            confirmed = confirmations.get(sid, action == SLOT_PRESERVE)
            decision = decisions.get(sid)
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
                extra=self._decision_trace_extra(rec, policy_kwargs, decision),
            )

    def _budget_exhausted(self, t0: float) -> bool:
        return (time.monotonic() - t0) > self._max_seconds

    def _record_arm_runtime_result(
        self,
        *,
        job: dict[str, Any],
        step: int,
        slots_scored: int,
        confirmed_delta: int,
        rejected_delta: int,
        frame_age_seconds: float,
        status: int,
        through_ring: bool,
    ) -> None:
        if not self._arm_runtime_active or self._arm_scheduler is None:
            return
        slot_ids = [int(x) for x in list(job["slot_ids"])]
        cap = int(_ext.wire_event_constants()["ARM_MAINTENANCE_SLOT_CAPACITY"])
        if len(slot_ids) < cap:
            slot_ids.extend([(1 << 64) - 1] * (cap - len(slot_ids)))
        result = {
            "event_type": 5,
            "job_type": int(job["job_type"]),
            "stream_id": int(job["stream_id"]),
            "status": int(status),
            "slot_count": int(job["slot_count"]),
            "job_id": int(job["job_id"]),
            "frame_id": int(job["frame_id"]),
            "step": int(step),
            "slots_scored": int(max(0, slots_scored)),
            "actions_confirmed": int(max(0, confirmed_delta)),
            "actions_rejected": int(max(0, rejected_delta)),
            "probe_seconds": float(self._last_probe_seconds),
            "oracle_seconds": float(self._last_oracle_scoring_seconds),
            "cpu_seconds": float(
                self._last_stage_seconds.get("select", 0.0)
                + self._last_stage_seconds.get("ema", 0.0)
                + self._last_stage_seconds.get("action", 0.0)
            ),
            "frame_age_seconds": float(frame_age_seconds),
            "slot_ids": slot_ids[:cap],
        }
        if not through_ring:
            self._arm_scheduler.record_result(result)
            return
        if self._arm_result_worker_ring is None or self._arm_result_ring is None:
            self._arm_scheduler.record_result(result)
            return
        if not bool(self._arm_result_worker_ring.push(result)):
            self._arm_result_ring_drops += 1
            self._arm_scheduler.record_result({**result, "status": max(1, int(status))})
            return
        self._arm_result_ring_pushes += 1
        popped = self._arm_result_ring.pop()
        if popped is None:
            self._arm_scheduler.record_result({**result, "status": max(1, int(status))})
            return
        self._arm_result_ring_pops += 1
        self._arm_scheduler.record_result(dict(popped))

    def _complete_arm_runtime_job(
        self,
        *,
        step: int,
        cf: CounterfactualResult,
        frame_age_seconds: float,
        confirmed_delta: int = 0,
        rejected_delta: int = 0,
        status: int = 0,
    ) -> None:
        if (
            not self._arm_runtime_active
            or self._arm_scheduler is None
            or self._arm_result_ring is None
            or self._active_arm_job is None
        ):
            self._active_arm_job = None
            return
        job = self._active_arm_job
        self._active_arm_job = None
        self._record_arm_runtime_result(
            job=job,
            step=step,
            slots_scored=len(cf.slot_indices),
            confirmed_delta=confirmed_delta,
            rejected_delta=rejected_delta,
            frame_age_seconds=frame_age_seconds,
            status=status,
            through_ring=True,
        )

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
            SLOT_RELEASE,
        }
        self._last_oracle_score_by_slot.clear()
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
        if cf.scoring_mode == "oracle":
            confirmations: dict[int, bool] = {}
            n_pairs = 0
            for sid, phys, action, oracle_score in candidate_pairs:
                self._last_oracle_score_by_slot[int(sid)] = float(oracle_score)
                confirmed = self._action_confirmed(
                    action=action,
                    proxy_score=oracle_score,
                    oracle_score=oracle_score,
                )
                confirmations[sid] = confirmed
                n_pairs += 1
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
                    proxy_score=oracle_score,
                    oracle_score=oracle_score,
                    confirmed=confirmed,
                )
            self._oracle_direct_confirmations_total += n_pairs
            self._oracle_confirmations_total += n_pairs
            self._last_oracle_candidates = n_pairs
            self._last_proxy_oracle_sign_match_rate = 1.0 if n_pairs else 0.0
            self._last_proxy_oracle_abs_error = 0.0
            return confirmations

        candidate_pairs = candidate_pairs[: self._oracle_confirm_top_k]
        oracle_t0 = time.monotonic()
        oracle = oracle_confirm_slots(
            model=model,
            outer=outer,
            probe_input_ids=self._probe_input_ids,
            probe_valid_mask=self._probe_valid_mask,
            slot_indices=[phys for _sid, phys, _action, _proxy in candidate_pairs],
            cache_read_cutoff=self._probe_cache_cutoff,
            variant_chunk_size=self._oracle_variant_chunk_size,
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
            self._last_oracle_score_by_slot[int(sid)] = float(oracle_score)
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
        if self._commit_policy_mode == "learned":
            return self._learned_action_physics_confirmed(
                action=action,
                oracle_score=oracle_score,
            )
        low_value = oracle_score <= max(self._threshold, self._useful_threshold)
        if action in {SLOT_EVICT, SLOT_QUARANTINE}:
            return low_value
        if action == SLOT_DECAY:
            return oracle_score <= max(2.0 * self._threshold, self._useful_threshold)
        if action == SLOT_DISTILL:
            return low_value and oracle_score >= self._quarantine_threshold
        if action == SLOT_REFRESH:
            return oracle_score > self._useful_threshold
        if action == SLOT_RELEASE:
            return oracle_score > self._useful_threshold
        return True

    @staticmethod
    def _learned_action_physics_confirmed(
        *,
        action: str,
        oracle_score: float,
    ) -> bool:
        """Threshold-free GPU3 physics veto for learned commit actions.

        ``oracle_score`` is mean ``NLL_hide_slot - NLL_force_on``. Positive
        means hiding the slot hurts, so the slot is currently useful. Negative
        means hiding the slot helps, so the slot is harmful. The learned
        controller owns the action choice; this only prevents physically
        incoherent mutations.
        """
        score = float(oracle_score)
        if not math.isfinite(score):
            return False
        if action in {SLOT_EVICT, SLOT_QUARANTINE, SLOT_DECAY, SLOT_DISTILL}:
            return score <= 0.0
        if action in {SLOT_REFRESH, SLOT_RELEASE}:
            return score > 0.0
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
        policy_kwargs = self._policy_kwargs(outer)

        # Collect actions per slot
        actions: dict[int, str] = {}  # slot_id_or_idx -> action
        decisions: dict[int, CommitDecision] = {}

        if has_table:
            # Classify each scheduled slot, including quarantined slots. Release
            # is a learned action now, not a streak-triggered side effect.
            for i, phys_idx in enumerate(cf.slot_indices):
                sid = table.physical_to_slot_id(phys_idx)
                if sid is None:
                    continue
                rec = table.record(sid)
                if rec is None:
                    continue
                decision = self._choose_commit_decision(rec, policy_kwargs)
                actions[sid] = decision.action
                decisions[sid] = decision

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
                decision = decisions.get(sid)
                extra = self._decision_trace_extra(rec, policy_kwargs, decision)
                confirmed = confirmations.get(sid, action == SLOT_PRESERVE)
                if action == SLOT_REFRESH:
                    accepted = False
                    agreed = confirmed and self._commit_action_ready(sid, action, rec)
                    if agreed:
                        accepted = self._execute_refresh(model, outer, sid, cf, t0=t0)
                    if decision is not None:
                        self._record_commit_feedback(
                            decision,
                            accepted=accepted,
                            structural=True,
                            slot_id=sid,
                        )
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
                        if decision is not None:
                            self._record_commit_feedback(
                                decision,
                                accepted=False,
                                structural=True,
                                slot_id=sid,
                            )
                        self._trace_event(
                            step, sid, action, rec,
                            accepted=False, reason="oracle_rejected", extra=extra,
                        )
                        continue
                    if not self._commit_action_ready(sid, action, rec):
                        rec.last_action = action
                        self._trace_event(
                            step, sid, action, rec,
                            accepted=False, reason="awaiting_agreement", extra=extra,
                        )
                        continue
                    self._execute_quarantine(outer, sid)
                    result.quarantined.append(sid)
                    rec.last_action = action
                    if decision is not None:
                        self._record_commit_feedback(
                            decision,
                            accepted=True,
                            structural=True,
                            slot_id=sid,
                        )
                    self._trace_event(step, sid, action, rec, extra=extra)
                elif action == SLOT_DECAY:
                    if not confirmed:
                        rec.last_action = action
                        if decision is not None:
                            self._record_commit_feedback(
                                decision,
                                accepted=False,
                                structural=False,
                                slot_id=sid,
                            )
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
                    if decision is not None:
                        self._record_commit_feedback(
                            decision,
                            accepted=True,
                            structural=False,
                            slot_id=sid,
                        )
                    self._trace_event(step, sid, action, rec, extra=extra)
                elif action == SLOT_RELEASE:
                    if not confirmed:
                        rec.last_action = action
                        if decision is not None:
                            self._record_commit_feedback(
                                decision,
                                accepted=False,
                                structural=True,
                                slot_id=sid,
                            )
                        self._trace_event(
                            step, sid, action, rec,
                            accepted=False, reason="oracle_rejected", extra=extra,
                        )
                        continue
                    if not self._commit_action_ready(sid, action, rec):
                        rec.last_action = action
                        self._trace_event(
                            step, sid, action, rec,
                            accepted=False, reason="awaiting_agreement", extra=extra,
                        )
                        continue
                    table.release(sid)
                    self._quarantined.discard(sid)
                    result.released.append(sid)
                    self._releases_total += 1
                    rec.state = SLOT_ACTIVE
                    rec.last_action = action
                    if decision is not None:
                        self._record_commit_feedback(
                            decision,
                            accepted=True,
                            structural=True,
                            slot_id=sid,
                        )
                    self._trace_event(step, sid, action, rec, extra=extra)
                elif action == SLOT_PRESERVE:
                    if rec.state == SLOT_WARMING and rec.score_count > 0:
                        rec.state = SLOT_ACTIVE
                    rec.last_action = action
                    if decision is not None:
                        self._record_commit_feedback(
                            decision,
                            accepted=True,
                            structural=False,
                            slot_id=sid,
                        )
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
                decision = decisions.get(sid)
                extra = self._decision_trace_extra(rec, policy_kwargs, decision)
                if not confirmations.get(sid, False):
                    self._trace_event(
                        step, sid, action, rec,
                        accepted=False, reason="oracle_rejected", extra=extra,
                    )
                    if rec:
                        rec.last_action = action
                    if decision is not None:
                        self._record_commit_feedback(
                            decision,
                            accepted=False,
                            structural=True,
                            slot_id=sid,
                        )
                    continue
                if not self._commit_action_ready(sid, action, rec):
                    self._trace_event(
                        step, sid, action, rec,
                        accepted=False, reason="awaiting_agreement", extra=extra,
                    )
                    if rec:
                        rec.last_action = action
                    continue
                if action == SLOT_DISTILL:
                    receipt = self._execute_distill(outer, sid, step, model=model)
                    distill_extra = dict(extra)
                    distill_extra.update({
                        "distill_target": receipt.target,
                        "distill_prototype_updated": receipt.prototype_updated,
                        "distill_prototype_reason": receipt.prototype_reason,
                    })
                    if receipt.accepted:
                        result.distilled.append(sid)
                        self._distills_total += 1
                        if decision is not None:
                            self._record_commit_feedback(
                                decision,
                                accepted=True,
                                structural=True,
                                slot_id=sid,
                            )
                        self._trace_event(
                            step,
                            sid,
                            action,
                            rec,
                            reason="distilled",
                            extra=distill_extra,
                        )
                    else:
                        if decision is not None:
                            self._record_commit_feedback(
                                decision,
                                accepted=False,
                                structural=True,
                                slot_id=sid,
                            )
                        self._trace_event(
                            step, sid, action, rec,
                            accepted=False,
                            reason=f"distill_{receipt.target}_unavailable",
                            extra=distill_extra,
                        )
                else:
                    table.retire(sid, reason="evicted")
                    result.evicted.append(sid)
                    self._evictions_total += 1
                    if decision is not None:
                        self._record_commit_feedback(
                            decision,
                            accepted=True,
                            structural=True,
                            slot_id=sid,
                        )
                    self._trace_event(
                        step, sid, action, rec, reason="evicted", extra=extra
                    )
                if rec:
                    rec.last_action = action

        else:
            result = TickResult()

        return result

    def _oracle_counterfactual_probe(
        self,
        *,
        model: Any,
        outer: Any,
        frame: ProbeFrame,
        slot_work: list[int],
    ) -> CounterfactualResult:
        """Score scheduled slots with exact GPU3 memory physics.

        This is the non-cheap path: one real no-sidecar/baseline pass plus
        hide-one variants through ``model.encode(memory_mode='force_on')``.
        The surrounding loop is still CPU-side scheduling and evidence
        bookkeeping; only the tensor physics lives on GPU3.
        """
        oracle_t0 = time.monotonic()
        oracle = oracle_confirm_slots(
            model=model,
            outer=outer,
            probe_input_ids=frame.input_ids,
            probe_valid_mask=frame.valid_mask,
            slot_indices=slot_work,
            cache_read_cutoff=frame.cache_read_cutoff,
            variant_chunk_size=self._oracle_variant_chunk_size,
        )
        elapsed = time.monotonic() - oracle_t0
        self._last_oracle_scoring_seconds = elapsed
        self._oracle_scoring_seconds_total += elapsed
        self._last_stage_seconds["oracle_score"] = elapsed
        self._stage_seconds_total["oracle_score"] += elapsed
        self._oracle_scored_slots_total += len(oracle.slot_indices)
        sidecar_value = oracle.nll_no_sidecar - oracle.nll_baseline
        weights = torch.zeros(
            sidecar_value.shape[0],
            len(oracle.slot_indices),
            dtype=torch.float32,
        )
        return CounterfactualResult(
            marginal_gains=oracle.oracle_deltas,
            sidecar_value=sidecar_value,
            nll_baseline=oracle.nll_baseline,
            nll_no_sidecar=oracle.nll_no_sidecar,
            weights_baseline=weights,
            mask=oracle.mask,
            slot_indices=oracle.slot_indices,
            scoring_mode="oracle",
        )

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

        rec = table.record(slot_id)
        candidates = self._propose_refresh_candidates_cpu(
            outer=outer,
            slot=slot,
            rec=rec,
        )
        if len(candidates) <= 1:
            return False

        phys = table.slot_id_to_physical(slot_id)
        if phys is None:
            return False
        if (
            self._probe_input_ids is None
            or self._probe_valid_mask is None
            or self._budget_exhausted(t0)
        ):
            return False

        oracle_t0 = time.monotonic()
        oracle = oracle_confirm_refresh_candidates(
            model=model,
            outer=outer,
            probe_input_ids=self._probe_input_ids,
            probe_valid_mask=self._probe_valid_mask,
            slot_index=phys,
            candidate_tensors=candidates,
            cache_read_cutoff=self._probe_cache_cutoff,
            variant_chunk_size=self._refresh_candidate_variant_chunk_size,
        )
        elapsed = time.monotonic() - oracle_t0
        self._last_oracle_seconds = elapsed
        self._oracle_seconds_total += elapsed
        self._refresh_candidate_oracle_batches_total += 1
        self._refresh_candidate_oracle_chunks_total += int(oracle.chunk_count)
        self._refresh_candidate_oracle_variants_total += int(oracle.variants_total)
        self._refresh_candidate_oracle_seconds_total += elapsed
        self._last_refresh_candidate_oracle_chunks = int(oracle.chunk_count)
        self._last_refresh_candidate_oracle_variants = int(oracle.variants_total)
        self._last_stage_seconds["oracle"] = elapsed
        self._stage_seconds_total["oracle"] += elapsed
        self._last_oracle_candidates = len(oracle.candidate_names)
        if not oracle.candidate_names or oracle.candidate_scores.numel() == 0:
            self._refresh_candidate_rejects_total += 1
            return False

        best_idx = int(torch.argmax(oracle.candidate_scores).item())
        best_improvement = float(oracle.candidate_improvements[best_idx].item())
        best_name = oracle.candidate_names[best_idx]
        self._last_refresh_candidate_best_index = best_idx
        self._last_refresh_candidate_best_name = best_name
        self._last_refresh_candidate_best_improvement = best_improvement
        self._last_oracle_score_by_slot[int(slot_id)] = best_improvement
        structural_accepted = best_idx > 0 and best_improvement > 0.0
        self._refresh_proposal_model.update(
            candidates=candidates,
            scores=oracle.candidate_scores,
            accepted_index=best_idx,
            structural_accepted=structural_accepted,
        )
        self._trace_refresh_candidates(
            slot_id=slot_id,
            oracle=oracle,
            accepted_index=best_idx,
            accepted=structural_accepted,
        )

        if structural_accepted:
            best_tensor = candidates[best_idx][1].detach().to(
                device=slot.device,
                dtype=slot.dtype,
            )
            table.replace_tensor(slot_id, best_tensor)
            self._refresh_candidate_accepts_total += 1
            return True

        self._refresh_candidate_rejects_total += 1
        return False

    def _propose_refresh_candidates_cpu(
        self,
        *,
        outer: Any,
        slot: torch.Tensor,
        rec: SlotRecord | None = None,
    ) -> list[tuple[str, torch.Tensor]]:
        """Generate learned refresh candidates on CPU; GPU3 verifies physics."""
        proposal_t0 = time.monotonic()
        self._last_refresh_candidate_device = "cpu"
        with torch.inference_mode():
            context = {
                "stream_id": self._probe_stream_id,
                "frame_id": self._active_frame_id,
                "step": self._probe_step,
                "frame_age_steps": self._last_frame_age_steps,
                "frame_age_seconds": self._last_frame_age_seconds,
                "probe_cue": self._probe_cue,
            }
            if rec is not None:
                context.update({
                    "marginal_gain": rec.marginal_gain_ema,
                    "utility_ema": rec.utility_ema,
                    "sharpness": rec.sharpness_ema,
                    "activation_drift": rec.activation_drift_ema,
                    "representation_drift": rec.representation_drift_ema,
                    "semantic_drift": rec.semantic_drift_ema,
                    "contradiction": rec.contradiction_ema,
                    "retrieval_mass": rec.retrieval_mass_ema,
                    "peak_utility": rec.peak_utility,
                    "peak_sharpness": rec.peak_sharpness,
                    "score_count": rec.score_count,
                })
            candidates = self._refresh_proposal_model.sample_k(
                outer=outer,
                slot=slot,
                context=context,
            )
        self._last_refresh_candidate_count = len(candidates)
        self._refresh_candidate_proposals_total += max(0, len(candidates) - 1)
        self._refresh_candidate_proposal_seconds_total += (
            time.monotonic() - proposal_t0
        )
        return candidates

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
        self, outer: Any, slot_id: int, step: int, *, model: Any | None = None
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
        trace = {
            "bucket_id": bucket_id,
            "centroid_contrib": slot_tensor.detach().clone(),
        }
        latent_traces.append(trace)
        max_latent = int(getattr(outer, "max_slots", 0) or 0)
        while max_latent > 0 and len(latent_traces) > max_latent:
            latent_traces.pop(0)
        receipt.target = "latent_trace"
        receipt.accepted = True
        if model is not None:
            (
                receipt.prototype_updated,
                receipt.prototype_reason,
            ) = self._distill_trace_into_bucket_prototype(
                model=model,
                trace=trace,
            )
            self._last_prototype_distill_bucket = int(bucket_id)
            self._last_prototype_distill_reason = receipt.prototype_reason
            if receipt.prototype_updated:
                receipt.target = "latent_trace+bucket_prototype"
                self._prototype_distills_total += 1
            else:
                self._prototype_distill_skips_total += 1
        table.retire(slot_id, reason="distilled")

        return receipt

    def _distill_trace_into_bucket_prototype(
        self,
        *,
        model: Any,
        trace: dict[str, Any],
    ) -> tuple[bool, str]:
        prototypes = getattr(model, "bucket_prototypes_module", None)
        if prototypes is None:
            return False, "missing_bucket_prototypes"
        proto_buf = getattr(prototypes, "prototypes", None)
        if proto_buf is None:
            return False, "missing_prototype_buffer"
        if "bucket_id" not in trace or "centroid_contrib" not in trace:
            return False, "malformed_latent_trace"
        bucket_id = int(trace["bucket_id"])
        if bucket_id < 0 or bucket_id >= int(getattr(prototypes, "k_max", 0)):
            return False, "bucket_out_of_range"
        slot_tensor = trace["centroid_contrib"]
        if not torch.is_tensor(slot_tensor):
            return False, "latent_trace_not_tensor"
        proto_dim = int(getattr(prototypes, "prototype_dim", proto_buf.shape[-1]))
        value = slot_tensor.detach().reshape(-1, slot_tensor.shape[-1])
        if int(value.shape[-1]) != proto_dim:
            return False, "prototype_dim_mismatch"
        value = value.to(device=proto_buf.device, dtype=proto_buf.dtype)
        with torch.no_grad():
            prototypes.update(int(bucket_id), value)
        return True, "updated"

    def _decision_trace_extra(
        self,
        rec: SlotRecord | None,
        policy_kwargs: dict[str, Any],
        decision: CommitDecision | None = None,
    ) -> dict[str, Any]:
        if rec is None:
            return {}
        values = self._policy.action_values(rec, **policy_kwargs)
        extra: dict[str, Any] = {
            f"action_value_{action.lower()}": float(value)
            for action, value in values.items()
        }
        extra["commit_policy"] = self._commit_policy_mode
        if decision is not None:
            extra.update({
                "learned_commit_action": decision.action,
                "rule_shadow_action": decision.rule_action,
                "learned_commit_confidence": decision.confidence,
                "learned_commit_probability": decision.probability,
                "learned_commit_entropy": decision.entropy,
                "learned_commit_vetoed_action": decision.vetoed_action,
                "learned_commit_veto_reason": decision.veto_reason,
            })
            for action in SLOT_ACTION_ORDER:
                key = action.lower()
                extra[f"learned_commit_logit_{key}"] = decision.logits.get(
                    action, 0.0
                )
                extra[f"learned_commit_p_{key}"] = decision.probabilities.get(
                    action, 0.0
                )
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

    def _trace_refresh_candidates(
        self,
        *,
        slot_id: int,
        oracle: RefreshCandidateOracleResult,
        accepted_index: int,
        accepted: bool,
    ) -> None:
        if self._trace_path is None:
            return
        if self._trace_max_rows > 0 and self._trace_rows_written >= self._trace_max_rows:
            return
        scores = oracle.candidate_scores.detach().cpu().tolist()
        improvements = oracle.candidate_improvements.detach().cpu().tolist()
        event = {
            "row_type": "replay_refresh_candidates",
            "step": int(self._active_frame_step),
            "tick": self._tick_count,
            "slot_id": int(slot_id),
            "frame_id": self._active_frame_id,
            "frame_step": self._active_frame_step,
            "frame_age_steps": self._last_frame_age_steps,
            "frame_age_seconds": self._last_frame_age_seconds,
            "stream_id": self._probe_stream_id,
            "candidate_count": len(oracle.candidate_names),
            "candidate_names": oracle.candidate_names,
            "candidate_scores": [float(x) for x in scores],
            "candidate_improvements": [float(x) for x in improvements],
            "oracle_chunk_count": int(oracle.chunk_count),
            "oracle_variants_total": int(oracle.variants_total),
            "accepted_index": int(accepted_index),
            "accepted_name": (
                oracle.candidate_names[int(accepted_index)]
                if 0 <= int(accepted_index) < len(oracle.candidate_names)
                else ""
            ),
            "accepted": bool(accepted),
            "best_improvement": (
                float(improvements[int(accepted_index)])
                if 0 <= int(accepted_index) < len(improvements)
                else 0.0
            ),
            "oracle_seconds": self._last_oracle_seconds,
            "proposal_model": self._refresh_proposal_model.diagnostics(),
        }
        self._trace_buffer.append(json.dumps(event, separators=(",", ":")) + "\n")
        self._trace_rows_written += 1
        self._flush_trace_if_needed()

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
        self._flush_trace_if_needed()

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
        self._flush_trace_if_needed()

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
        self._flush_trace_if_needed()

    def _flush_trace_if_needed(self) -> None:
        if self._trace_flush_rows <= 0:
            return
        if len(self._trace_buffer) >= self._trace_flush_rows:
            self.flush_trace()

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

    def state_dict(self) -> dict[str, Any]:
        """Persist learned online-maintenance state.

        Probe frames, shm ring names, trace buffers, and active runtime jobs are
        intentionally omitted.  They are transport/runtime state.  The learned
        proposal model, action evidence, quarantines, EMAs, and coverage clocks
        are kept so eval can continue the same maintenance policy instead of
        cold-starting it.
        """
        return {
            "schema_version": 1,
            "action_mode": self._action_mode,
            "scoring_mode": self._scoring_mode,
            "memory_streams": self._memory_streams,
            "threshold": self._threshold,
            "ema_beta": self._ema_beta,
            "min_age": self._min_age,
            "max_seconds": self._max_seconds,
            "probe_chunk_size": self._probe_chunk_size,
            "oracle_confirm_top_k": self._oracle_confirm_top_k,
            "oracle_variant_chunk_size": self._oracle_variant_chunk_size,
            "drift_threshold": self._drift_threshold,
            "repr_drift_threshold": self._repr_drift_threshold,
            "refresh_lr": self._refresh_lr,
            "refresh_candidate_count": self._refresh_candidate_count,
            "refresh_candidate_variant_chunk_size": (
                self._refresh_candidate_variant_chunk_size
            ),
            "quarantine_threshold": self._quarantine_threshold,
            "max_quarantined": self._max_quarantined,
            "distill_peak_threshold": self._distill_peak_threshold,
            "peak_preserve_utility_threshold": (
                self._peak_preserve_utility_threshold
            ),
            "peak_preserve_sharpness_threshold": (
                self._peak_preserve_sharpness_threshold
            ),
            "useful_threshold": self._useful_threshold,
            "probe_buffer_size": self._probe_buffer_size,
            "frame_ttl_steps": self._frame_ttl_steps,
            "slot_work_chunk_size": self._slot_work_chunk_size,
            "action_agreement_count": self._action_agreement_count,
            "commit_policy": self._commit_policy_mode,
            "commit_online_lr": self._commit_online_lr,
            "commit_temperature": self._commit_temperature,
            "commit_policy_state": self._commit_policy.state_dict(),
            "refresh_proposal_model": self._refresh_proposal_model.state_dict(),
            "slot_last_scored_step": dict(self._slot_last_scored_step),
            "quarantined": sorted(self._quarantined),
            "action_evidence": {
                int(k): (str(v[0]), int(v[1]), int(v[2]), int(v[3]))
                for k, v in self._action_evidence.items()
            },
            "tick_count": self._tick_count,
            "evictions_total": self._evictions_total,
            "refreshes_total": self._refreshes_total,
            "refresh_candidate_proposals_total": (
                self._refresh_candidate_proposals_total
            ),
            "refresh_candidate_proposal_seconds_total": (
                self._refresh_candidate_proposal_seconds_total
            ),
            "refresh_candidate_oracle_batches_total": (
                self._refresh_candidate_oracle_batches_total
            ),
            "refresh_candidate_oracle_chunks_total": (
                self._refresh_candidate_oracle_chunks_total
            ),
            "refresh_candidate_oracle_variants_total": (
                self._refresh_candidate_oracle_variants_total
            ),
            "refresh_candidate_oracle_seconds_total": (
                self._refresh_candidate_oracle_seconds_total
            ),
            "refresh_candidate_accepts_total": (
                self._refresh_candidate_accepts_total
            ),
            "refresh_candidate_rejects_total": (
                self._refresh_candidate_rejects_total
            ),
            "distills_total": self._distills_total,
            "prototype_distills_total": self._prototype_distills_total,
            "prototype_distill_skips_total": self._prototype_distill_skips_total,
            "decays_total": self._decays_total,
            "releases_total": self._releases_total,
            "replays_total": self._replays_total,
            "slots_scored_total": self._slots_scored_total,
            "shadow_actions_total": self._shadow_actions_total,
            "shadow_action_counts": dict(self._shadow_action_counts),
            "oracle_confirmations_total": self._oracle_confirmations_total,
            "oracle_confirmed_actions_total": self._oracle_confirmed_actions_total,
            "oracle_rejected_actions_total": self._oracle_rejected_actions_total,
            "proxy_oracle_sign_matches": self._proxy_oracle_sign_matches,
            "proxy_oracle_pairs_total": self._proxy_oracle_pairs_total,
            "proxy_oracle_abs_error_sum": self._proxy_oracle_abs_error_sum,
            "probe_seconds_total": self._probe_seconds_total,
            "probe_over_budget_total": self._probe_over_budget_total,
            "probe_frames_ingested": self._probe_frames_ingested,
            "probe_frames_dropped_overflow": self._probe_frames_dropped_overflow,
            "probe_frames_dropped_stale": self._probe_frames_dropped_stale,
            "probe_frames_completed": self._probe_frames_completed,
            "probe_ticks_skipped_no_frame": self._probe_ticks_skipped_no_frame,
            "probe_ticks_skipped_no_slot_work": (
                self._probe_ticks_skipped_no_slot_work
            ),
            "slot_work_items_total": self._slot_work_items_total,
            "action_agreements_total": self._action_agreements_total,
            "action_agreements_reset_total": self._action_agreements_reset_total,
            "commit_decisions_total": self._commit_decisions_total,
            "commit_rule_disagreements_total": self._commit_rule_disagreements_total,
            "commit_feedback_updates_total": self._commit_feedback_updates_total,
            "queue_depth_sum": self._queue_depth_sum,
            "queue_depth_samples": self._queue_depth_samples,
            "queue_depth_max": self._queue_depth_max,
            "frame_age_steps_sum": self._frame_age_steps_sum,
            "frame_age_steps_max": self._frame_age_steps_max,
            "frame_age_seconds_sum": self._frame_age_seconds_sum,
            "frame_age_seconds_max": self._frame_age_seconds_max,
            "stream_ticks": dict(self._stream_ticks),
            "stream_work_items": dict(self._stream_work_items),
            "stream_probe_seconds": dict(self._stream_probe_seconds),
            "stage_seconds_total": dict(self._stage_seconds_total),
            "oracle_seconds_total": self._oracle_seconds_total,
            "oracle_scored_slots_total": self._oracle_scored_slots_total,
            "oracle_direct_confirmations_total": (
                self._oracle_direct_confirmations_total
            ),
            "oracle_scoring_seconds_total": self._oracle_scoring_seconds_total,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        proposal_state = state.get("refresh_proposal_model")
        if isinstance(proposal_state, dict):
            self._refresh_proposal_model.load_state_dict(proposal_state)
        commit_state = state.get("commit_policy_state")
        if isinstance(commit_state, dict):
            self._commit_policy.load_state_dict(commit_state)
        self._commit_policy_mode = str(
            state.get("commit_policy", self._commit_policy_mode)
        )
        self._commit_online_lr = float(
            state.get("commit_online_lr", self._commit_online_lr)
        )
        self._commit_temperature = float(
            state.get("commit_temperature", self._commit_temperature)
        )
        self._slot_last_scored_step = {
            int(k): int(v)
            for k, v in state.get("slot_last_scored_step", {}).items()
        }
        self._quarantined = {int(x) for x in state.get("quarantined", [])}
        self._action_evidence = {
            int(k): (str(v[0]), int(v[1]), int(v[2]), int(v[3]))
            for k, v in state.get("action_evidence", {}).items()
        }

        int_fields = {
            "_tick_count": "tick_count",
            "_evictions_total": "evictions_total",
            "_refreshes_total": "refreshes_total",
            "_refresh_candidate_proposals_total": "refresh_candidate_proposals_total",
            "_refresh_candidate_oracle_batches_total": (
                "refresh_candidate_oracle_batches_total"
            ),
            "_refresh_candidate_oracle_chunks_total": (
                "refresh_candidate_oracle_chunks_total"
            ),
            "_refresh_candidate_oracle_variants_total": (
                "refresh_candidate_oracle_variants_total"
            ),
            "_refresh_candidate_accepts_total": "refresh_candidate_accepts_total",
            "_refresh_candidate_rejects_total": "refresh_candidate_rejects_total",
            "_distills_total": "distills_total",
            "_prototype_distills_total": "prototype_distills_total",
            "_prototype_distill_skips_total": "prototype_distill_skips_total",
            "_decays_total": "decays_total",
            "_releases_total": "releases_total",
            "_replays_total": "replays_total",
            "_slots_scored_total": "slots_scored_total",
            "_shadow_actions_total": "shadow_actions_total",
            "_oracle_confirmations_total": "oracle_confirmations_total",
            "_oracle_confirmed_actions_total": "oracle_confirmed_actions_total",
            "_oracle_rejected_actions_total": "oracle_rejected_actions_total",
            "_proxy_oracle_sign_matches": "proxy_oracle_sign_matches",
            "_proxy_oracle_pairs_total": "proxy_oracle_pairs_total",
            "_probe_over_budget_total": "probe_over_budget_total",
            "_probe_frames_ingested": "probe_frames_ingested",
            "_probe_frames_dropped_overflow": "probe_frames_dropped_overflow",
            "_probe_frames_dropped_stale": "probe_frames_dropped_stale",
            "_probe_frames_completed": "probe_frames_completed",
            "_probe_ticks_skipped_no_frame": "probe_ticks_skipped_no_frame",
            "_probe_ticks_skipped_no_slot_work": "probe_ticks_skipped_no_slot_work",
            "_slot_work_items_total": "slot_work_items_total",
            "_action_agreements_total": "action_agreements_total",
            "_action_agreements_reset_total": "action_agreements_reset_total",
            "_commit_decisions_total": "commit_decisions_total",
            "_commit_rule_disagreements_total": "commit_rule_disagreements_total",
            "_commit_feedback_updates_total": "commit_feedback_updates_total",
            "_queue_depth_sum": "queue_depth_sum",
            "_queue_depth_samples": "queue_depth_samples",
            "_queue_depth_max": "queue_depth_max",
            "_frame_age_steps_sum": "frame_age_steps_sum",
            "_frame_age_steps_max": "frame_age_steps_max",
            "_oracle_scored_slots_total": "oracle_scored_slots_total",
            "_oracle_direct_confirmations_total": "oracle_direct_confirmations_total",
        }
        for attr, key in int_fields.items():
            if key in state:
                setattr(self, attr, int(state[key]))

        float_fields = {
            "_refresh_candidate_proposal_seconds_total": (
                "refresh_candidate_proposal_seconds_total"
            ),
            "_refresh_candidate_oracle_seconds_total": (
                "refresh_candidate_oracle_seconds_total"
            ),
            "_proxy_oracle_abs_error_sum": "proxy_oracle_abs_error_sum",
            "_probe_seconds_total": "probe_seconds_total",
            "_frame_age_seconds_sum": "frame_age_seconds_sum",
            "_frame_age_seconds_max": "frame_age_seconds_max",
            "_oracle_seconds_total": "oracle_seconds_total",
            "_oracle_scoring_seconds_total": "oracle_scoring_seconds_total",
        }
        for attr, key in float_fields.items():
            if key in state:
                setattr(self, attr, float(state[key]))

        self._shadow_action_counts = {
            str(k): int(v) for k, v in state.get("shadow_action_counts", {}).items()
        }
        self._stream_ticks = {
            int(k): int(v) for k, v in state.get("stream_ticks", {}).items()
        } or {i: 0 for i in range(self._memory_streams)}
        self._stream_work_items = {
            int(k): int(v)
            for k, v in state.get("stream_work_items", {}).items()
        } or {i: 0 for i in range(self._memory_streams)}
        self._stream_probe_seconds = {
            int(k): float(v)
            for k, v in state.get("stream_probe_seconds", {}).items()
        } or {i: 0.0 for i in range(self._memory_streams)}
        stage_totals = state.get("stage_seconds_total")
        if isinstance(stage_totals, dict):
            for key, value in stage_totals.items():
                self._stage_seconds_total[str(key)] = float(value)
        self._started_at = time.monotonic()

    def diagnostics(self) -> dict[str, Any]:
        self.flush_trace()
        elapsed_wall = max(1e-9, time.monotonic() - self._started_at)
        unique_slots_scored = len(self._slot_last_scored_step)
        slot_coverage_ratio = (
            min(1.0, unique_slots_scored / self._last_visible_slots)
            if self._last_visible_slots
            else 0.0
        )
        slot_scored_sweeps = (
            self._slots_scored_total / self._last_visible_slots
            if self._last_visible_slots
            else 0.0
        )
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
        arm_native_diag = (
            self._arm_scheduler.diagnostics()
            if self._arm_scheduler is not None
            else None
        )
        arm_runtime_diag = {
            "enabled_requested": self._arm_runtime_enabled_requested,
            "active": self._arm_runtime_active,
            "error": self._arm_runtime_error,
            "job_ring_name": self._arm_job_ring_name,
            "result_ring_name": self._arm_result_ring_name,
            "jobs_pushed": self._arm_jobs_pushed,
            "jobs_popped": self._arm_jobs_popped,
            "job_ring_drops": self._arm_job_ring_drops,
            "result_ring_pushes": self._arm_result_ring_pushes,
            "result_ring_pops": self._arm_result_ring_pops,
            "result_ring_drops": self._arm_result_ring_drops,
            "native_scheduler": arm_native_diag,
            "transport_mode": (
                "inline_worker_attached_shm_endpoints"
                if self._arm_runtime_active
                else "inactive"
            ),
        }
        evidence_engine_native_diag: dict[str, Any] | None = None
        if self._evidence_engine is not None:
            try:
                evidence_engine_native_diag = self._evidence_engine.diagnostics()
            except Exception as exc:
                evidence_engine_native_diag = {
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
        evidence_engine_diag = {
            "enabled_requested": self._evidence_engine_enabled_requested,
            "active": self._evidence_engine_active,
            "error": self._evidence_engine_error,
            "d_model": self._evidence_engine_d_model,
            "lanes": self._evidence_engine_lanes,
            "frames_ingested": self._evidence_frames_ingested,
            "ingest_errors_total": self._evidence_ingest_errors_total,
            "lm_head_refreshes_total": self._evidence_lm_head_refreshes_total,
            "lm_head_refresh_errors_total": self._evidence_lm_head_refresh_errors_total,
            "last_lm_head_refresh_step": self._evidence_last_lm_head_refresh_step,
            "last_lm_head_refresh_error": self._evidence_last_lm_head_refresh_error,
            "cue_populated_total": self._evidence_cue_populated_total,
            "cue_skipped_no_cue_total": self._evidence_cue_skipped_no_cue_total,
            "cue_skipped_shape_total": self._evidence_cue_skipped_shape_total,
            "cue_populate_errors_total": self._evidence_cue_populate_errors_total,
            "native": evidence_engine_native_diag,
        }
        starvation_reasons = (
            "ok",
            "no_slots",
            "confidence_gate",
            "frame_stale",
            "scheduler_behind",
            "job_ring_empty",
            "result_ring_full",
            "oracle_saturated",
        )
        gpu3_starvation_reason = "ok"
        if self._arm_result_ring_drops > 0:
            gpu3_starvation_reason = "result_ring_full"
        elif self._probe_over_budget_total > 0:
            gpu3_starvation_reason = "oracle_saturated"
        elif self._probe_frames_dropped_stale > 0:
            gpu3_starvation_reason = "frame_stale"
        elif self._probe_ticks_skipped_no_slot_work > 0 and self._replays_total == 0:
            gpu3_starvation_reason = "no_slots"
        elif self._arm_runtime_enabled_requested and not self._arm_runtime_active:
            gpu3_starvation_reason = "scheduler_behind"
        elif self._arm_runtime_active and self._arm_jobs_pushed == self._arm_jobs_popped:
            gpu3_starvation_reason = "job_ring_empty"
        gpu3_idle_seconds_by_reason = {reason: 0.0 for reason in starvation_reasons}
        gpu3_idle_seconds_by_reason[gpu3_starvation_reason] = (
            max(0.0, elapsed_wall - self._oracle_seconds_total)
            if gpu3_starvation_reason != "ok"
            else 0.0
        )
        return {
            "enabled": True,
            "action_mode": self._action_mode,
            "scoring_mode": self._scoring_mode,
            "exact_oracle_backend": "gpu3_torch",
            "cpu_exact_scorer_enabled": False,
            "gpu3_starvation_reason": gpu3_starvation_reason,
            "gpu3_idle_seconds_by_reason": gpu3_idle_seconds_by_reason,
            "memory_streams": self._memory_streams,
            "memory_streams_requested": self._memory_streams,
            "memory_streams_active": self._arm_runtime_active,
            "memory_stream_execution_mode": (
                "native_cpu_scheduler_inline_gpu3_worker"
                if self._arm_runtime_active
                else "single_threaded_time_sliced"
            ),
            "memory_streams_note": (
                "native CPU conductor owns frame TTL, slot coverage, stream "
                "assignment, and attached shm job/result endpoints; the rank-3 "
                "GPU3 worker consumes those jobs inline and executes Torch oracle "
                "physics"
                if self._arm_runtime_active
                else "stream_id partitions replay-maintenance telemetry and work "
                "ownership, but this Python control plane executes one probe "
                "chunk at a time"
            ),
            "arm_runtime": arm_runtime_diag,
            "cpu_evidence_engine": evidence_engine_diag,
            "tick_count": self._tick_count,
            "replays_total": self._replays_total,
            "evictions_total": self._evictions_total,
            "refreshes_total": self._refreshes_total,
            "refresh_candidate_count_config": self._refresh_candidate_count,
            "refresh_candidate_variant_chunk_size": (
                self._refresh_candidate_variant_chunk_size
            ),
            "refresh_candidate_proposal_device": self._last_refresh_candidate_device,
            "refresh_candidate_proposals_total": self._refresh_candidate_proposals_total,
            "refresh_candidate_proposal_seconds_total": (
                self._refresh_candidate_proposal_seconds_total
            ),
            "refresh_candidate_oracle_batches_total": (
                self._refresh_candidate_oracle_batches_total
            ),
            "refresh_candidate_oracle_chunks_total": (
                self._refresh_candidate_oracle_chunks_total
            ),
            "refresh_candidate_oracle_variants_total": (
                self._refresh_candidate_oracle_variants_total
            ),
            "refresh_candidate_oracle_seconds_total": (
                self._refresh_candidate_oracle_seconds_total
            ),
            "refresh_candidate_accepts_total": self._refresh_candidate_accepts_total,
            "refresh_candidate_rejects_total": self._refresh_candidate_rejects_total,
            "last_refresh_candidate_count": self._last_refresh_candidate_count,
            "last_refresh_candidate_best_name": self._last_refresh_candidate_best_name,
            "last_refresh_candidate_best_index": self._last_refresh_candidate_best_index,
            "last_refresh_candidate_best_improvement": (
                self._last_refresh_candidate_best_improvement
            ),
            "last_refresh_candidate_oracle_chunks": (
                self._last_refresh_candidate_oracle_chunks
            ),
            "last_refresh_candidate_oracle_variants": (
                self._last_refresh_candidate_oracle_variants
            ),
            "refresh_proposal_model": self._refresh_proposal_model.diagnostics(),
            "distills_total": self._distills_total,
            "prototype_distills_total": self._prototype_distills_total,
            "prototype_distill_skips_total": self._prototype_distill_skips_total,
            "last_prototype_distill_bucket": self._last_prototype_distill_bucket,
            "last_prototype_distill_reason": self._last_prototype_distill_reason,
            "decays_total": self._decays_total,
            "releases_total": self._releases_total,
            "slots_scored_total": self._slots_scored_total,
            "slots_tracked": self._last_slots_tracked,
            "shadow_actions_total": self._shadow_actions_total,
            "shadow_action_counts": dict(self._shadow_action_counts),
            "trace_flush_rows": self._trace_flush_rows,
            "oracle_confirm_top_k": self._oracle_confirm_top_k,
            "oracle_variant_chunk_size": self._oracle_variant_chunk_size,
            "oracle_confirmations_total": self._oracle_confirmations_total,
            "oracle_confirmed_actions_total": self._oracle_confirmed_actions_total,
            "oracle_rejected_actions_total": self._oracle_rejected_actions_total,
            "oracle_direct_confirmations_total": self._oracle_direct_confirmations_total,
            "oracle_scored_slots_total": self._oracle_scored_slots_total,
            "oracle_scoring_seconds_total": self._oracle_scoring_seconds_total,
            "last_oracle_scoring_seconds": self._last_oracle_scoring_seconds,
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
            "unique_slots_scored": unique_slots_scored,
            "slot_coverage_ratio": slot_coverage_ratio,
            "slot_scored_sweeps": slot_scored_sweeps,
            "slots_untouched_past_ttl": self._last_untouched_slots,
            "max_untouched_slot_steps": self._last_max_untouched_steps,
            "slot_coverage_per_minute": (
                self._slots_scored_total / (elapsed_wall / 60.0)
                if elapsed_wall > 0.0
                else 0.0
            ),
            "commit_policy": self._commit_policy_mode,
            "learned_commit_policy": self._commit_policy.diagnostics(),
            "commit_decisions_total": self._commit_decisions_total,
            "commit_rule_disagreements_total": self._commit_rule_disagreements_total,
            "commit_feedback_updates_total": self._commit_feedback_updates_total,
            "commit_online_lr": self._commit_online_lr,
            "commit_temperature": self._commit_temperature,
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
