"""Tests for the Adaptive Residual Memory control plane."""
from __future__ import annotations

import time

import torch
import torch.nn as nn

import pytest

from chaoscontrol.replay_eviction import (
    ReplayEvictionLoop, TickResult, MaintenancePolicy,
    counterfactual_probe, oracle_confirm_slots, replay_score_slots, _evict_slots,
    _compute_per_slot_sharpness, _compute_representation_drift,
    CounterfactualResult, DistillReceipt, MemoryEvent,
    SLOT_PRESERVE, SLOT_DECAY, SLOT_EVICT, SLOT_REFRESH,
    SLOT_QUARANTINE, SLOT_DISTILL,
)
from chaoscontrol.slot_table import (
    SlotTable, SlotRecord,
    SLOT_WARMING, SLOT_ACTIVE, SLOT_SHARP, SLOT_DECAYING,
    SLOT_QUARANTINED, SLOT_RETIRED,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubOuterModel:
    def __init__(self, outer_dim: int = 8, n_slots: int = 10) -> None:
        self.outer_dim = outer_dim
        self.encoder = nn.Linear(16, outer_dim, bias=False)
        self.encoder.weight.requires_grad_(False)
        self.decoder = nn.Linear(outer_dim, 16, bias=False)
        self.cue_proj = nn.Linear(16, outer_dim, bias=False)

        self._slots = [torch.randn(1, outer_dim) for _ in range(n_slots)]
        self._survival = [1.0] * n_slots
        self._slot_buckets = [0] * n_slots
        self._slot_event_ids = [i for i in range(n_slots)]
        self._retrieval_weights: torch.Tensor | None = None
        self._retrieval_indices: list[int] | None = None
        self._latent_traces: list[dict] = []
        self._quarantined_indices: set[int] = set()

        self.table: SlotTable | None = None


class _StubOuterWithTable:
    def __init__(self, outer_dim: int = 8, n_slots: int = 10) -> None:
        self.outer_dim = outer_dim
        self.encoder = nn.Linear(16, outer_dim, bias=False)
        self.encoder.weight.requires_grad_(False)
        self.decoder = nn.Linear(outer_dim, 16, bias=False)
        self.cue_proj = nn.Linear(16, outer_dim, bias=False)
        self._latent_traces: list[dict] = []

        self.table = SlotTable()
        for i in range(n_slots):
            self.table.append(torch.randn(1, outer_dim), bucket_id=0, step=0)


class _StubModel(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        vocab: int = 32,
        outer_dim: int = 8,
        n_slots: int = 10,
        memory_benefit: float = 0.5,
        use_table: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        if use_table:
            self.outer_model = _StubOuterWithTable(outer_dim, n_slots)
        else:
            self.outer_model = _StubOuterModel(outer_dim, n_slots)
        self._memory_benefit = memory_benefit

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        memory_mode: str = "off",
        cache_read_cutoff=None,
        memory_slot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del memory_slot_mask
        batch, seq = input_ids.shape
        base = torch.randn(batch, seq, self.dim) * 0.1
        if memory_mode == "force_on" and self.outer_model is not None:
            base = base + self._memory_benefit
            outer = self.outer_model
            if hasattr(outer, 'table') and outer.table is not None:
                n_slots = len(outer.table)
            elif hasattr(outer, '_slots'):
                n_slots = len(outer._slots)
            else:
                n_slots = 0
            if n_slots > 0 and hasattr(outer, '_retrieval_weights'):
                weights = torch.ones(batch, n_slots) / n_slots
                outer._retrieval_weights = weights
                outer._retrieval_indices = list(range(n_slots))
        elif self.outer_model is not None and hasattr(self.outer_model, '_retrieval_weights'):
            self.outer_model._retrieval_weights = None
            self.outer_model._retrieval_indices = None
        return base


# ---------------------------------------------------------------------------
# Tests for _evict_slots (backward compat)
# ---------------------------------------------------------------------------


class TestEvictSlots:
    def test_evict_single(self):
        outer = _StubOuterModel(n_slots=5)
        assert len(outer._slots) == 5
        evicted = _evict_slots(outer, [2])
        assert evicted == [2]
        assert len(outer._slots) == 4

    def test_evict_multiple_descending_order(self):
        outer = _StubOuterModel(n_slots=6)
        evicted = _evict_slots(outer, [1, 4, 2])
        assert sorted(evicted) == [1, 2, 4]
        assert len(outer._slots) == 3

    def test_evict_out_of_range_ignored(self):
        outer = _StubOuterModel(n_slots=3)
        evicted = _evict_slots(outer, [0, 5, 10])
        assert evicted == [0]
        assert len(outer._slots) == 2

    def test_evict_empty(self):
        outer = _StubOuterModel(n_slots=3)
        evicted = _evict_slots(outer, [])
        assert evicted == []
        assert len(outer._slots) == 3

    def test_evict_all(self):
        outer = _StubOuterModel(n_slots=4)
        evicted = _evict_slots(outer, [0, 1, 2, 3])
        assert sorted(evicted) == [0, 1, 2, 3]
        assert len(outer._slots) == 0


# ---------------------------------------------------------------------------
# Tests for replay_score_slots (backward compat)
# ---------------------------------------------------------------------------


class TestReplayScoreSlots:
    def test_basic_scoring(self):
        model = _StubModel(n_slots=5, memory_benefit=1.0)
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)
        result = replay_score_slots(
            model=model,
            probe_input_ids=input_ids,
            probe_valid_mask=valid_mask,
        )
        assert "slot_utilities" in result
        assert "slot_indices" in result

    def test_no_outer_model(self):
        model = _StubModel(n_slots=0)
        model.outer_model = None
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)
        result = replay_score_slots(
            model=model,
            probe_input_ids=input_ids,
            probe_valid_mask=valid_mask,
        )
        assert len(result["slot_indices"]) == 0

    def test_empty_slots(self):
        model = _StubModel(n_slots=0)
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)
        result = replay_score_slots(
            model=model,
            probe_input_ids=input_ids,
            probe_valid_mask=valid_mask,
        )
        assert len(result["slot_indices"]) == 0


# ---------------------------------------------------------------------------
# Tests for TickResult
# ---------------------------------------------------------------------------


class TestTickResult:
    def test_empty(self):
        r = TickResult()
        assert r.evicted == []
        assert r.refreshed == []
        assert r.quarantined == []
        assert r.released == []
        assert r.distilled == []
        assert r.decayed == []
        assert r.evicted_indices == []

    def test_evicted_indices_union(self):
        r = TickResult(evicted=[1, 3], distilled=[5, 7])
        assert sorted(r.evicted_indices) == [1, 3, 5, 7]


# ---------------------------------------------------------------------------
# Tests for ReplayEvictionLoop (legacy path)
# ---------------------------------------------------------------------------


class TestReplayEvictionLoop:
    def test_no_probe_returns_empty(self):
        loop = ReplayEvictionLoop()
        model = _StubModel()
        result = loop.tick(model=model, step=100)
        assert result.evicted_indices == []

    def test_cache_probe_enables_tick(self):
        loop = ReplayEvictionLoop(min_slot_age_steps=0, eviction_threshold=100.0)
        model = _StubModel(n_slots=5)
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)

        assert not loop.has_probe()
        loop.cache_probe(
            input_ids=input_ids, valid_mask=valid_mask,
            cache_read_cutoff=None, step=0,
        )
        assert loop.has_probe()
        result = loop.tick(model=model, step=200)
        assert loop._tick_count == 1

    def test_cache_probe_replaces_previous_batch(self):
        loop = ReplayEvictionLoop()
        first = torch.zeros(2, 17, dtype=torch.long)
        second = torch.ones(2, 17, dtype=torch.long)
        mask = torch.ones(2, 17)
        loop.cache_probe(
            input_ids=first,
            valid_mask=mask,
            cache_read_cutoff=None,
            step=10,
        )
        loop.cache_probe(
            input_ids=second,
            valid_mask=mask,
            cache_read_cutoff=None,
            step=20,
            stream_id=9,
        )
        assert loop._probe_step == 20
        assert loop._probe_stream_id == 1
        assert torch.equal(loop._probe_input_ids, second)

    def test_min_age_respected(self):
        loop = ReplayEvictionLoop(min_slot_age_steps=1000, eviction_threshold=100.0)
        model = _StubModel(n_slots=5, memory_benefit=0.0)
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)
        loop.cache_probe(input_ids=input_ids, valid_mask=valid_mask,
                         cache_read_cutoff=None, step=0)
        loop.tick(model=model, step=10)
        result = loop.tick(model=model, step=20)
        assert result.evicted_indices == []

    def test_score_count_gate(self):
        loop = ReplayEvictionLoop(min_slot_age_steps=0, eviction_threshold=100.0)
        model = _StubModel(n_slots=5, memory_benefit=0.0)
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)
        loop.cache_probe(input_ids=input_ids, valid_mask=valid_mask,
                         cache_read_cutoff=None, step=0)
        result = loop.tick(model=model, step=200)
        assert result.evicted_indices == []

    def test_eviction_after_multiple_ticks(self):
        loop = ReplayEvictionLoop(min_slot_age_steps=0, eviction_threshold=100.0)
        model = _StubModel(n_slots=5, memory_benefit=0.0)
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)
        loop.cache_probe(input_ids=input_ids, valid_mask=valid_mask,
                         cache_read_cutoff=None, step=0)
        loop.tick(model=model, step=200)
        result = loop.tick(model=model, step=300)
        assert len(result.evicted_indices) > 0

    def test_diagnostics(self):
        loop = ReplayEvictionLoop(memory_streams=8)
        diag = loop.diagnostics()
        assert "tick_count" in diag
        assert "evictions_total" in diag
        assert "slots_tracked" in diag
        assert "quarantined_count" in diag
        assert diag["tick_count"] == 0
        assert diag["memory_streams"] == 8
        assert diag["memory_streams_requested"] == 8
        assert diag["memory_streams_active"] is False
        assert "oracle_confirmations_total" in diag
        assert "proxy_oracle_abs_error_mean" in diag
        assert "last_probe_seconds" in diag
        assert "probe_over_budget_total" in diag

    def test_ema_smoothing(self):
        loop = ReplayEvictionLoop(eviction_ema_beta=0.9)
        slot_idx = 42
        loop._slot_first_seen_step[slot_idx] = 0
        loop._slot_utility_ema[slot_idx] = 1.0
        loop._slot_score_count[slot_idx] = 1
        old = loop._slot_utility_ema[slot_idx]
        loop._slot_utility_ema[slot_idx] = 0.9 * old + 0.1 * 0.0
        loop._slot_score_count[slot_idx] = 2
        assert loop._slot_utility_ema[slot_idx] == pytest.approx(0.9, abs=1e-6)

    def test_reindex_after_eviction(self):
        loop = ReplayEvictionLoop()
        loop._slot_utility_ema = {0: 0.5, 1: 0.1, 2: 0.8, 3: 0.2, 4: 0.9}
        loop._slot_first_seen_step = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        loop._slot_score_count = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}
        loop._reindex_after_eviction([1, 3])
        assert set(loop._slot_utility_ema.keys()) == {0, 1, 2}
        assert loop._slot_utility_ema[0] == pytest.approx(0.5)
        assert loop._slot_utility_ema[1] == pytest.approx(0.8)
        assert loop._slot_utility_ema[2] == pytest.approx(0.9)

    def test_refresh_score_maps_physical_index_to_probe_local_index(self):
        loop = ReplayEvictionLoop()
        cf = CounterfactualResult(
            marginal_gains=torch.tensor([
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
            ]),
            sidecar_value=torch.zeros(1, 2),
            nll_baseline=torch.zeros(1, 2),
            nll_no_sidecar=torch.zeros(1, 2),
            weights_baseline=torch.zeros(1, 3),
            mask=torch.ones(1, 2, dtype=torch.bool),
            slot_indices=[1, 3, 5],
        )
        assert loop._quick_refresh_score(None, 3, cf) == pytest.approx(2.0)
        assert loop._quick_refresh_score(None, 4, cf) == pytest.approx(0.0)

    def test_cache_probe_stores_real_read_cue(self):
        loop = ReplayEvictionLoop()
        ids = torch.zeros(2, 17, dtype=torch.long)
        mask = torch.ones(2, 17)
        cue = torch.randn(2, 16)
        loop.cache_probe(
            input_ids=ids,
            valid_mask=mask,
            cue=cue,
            cache_read_cutoff=None,
            step=7,
        )
        assert torch.equal(loop._probe_cue, cue)


# ---------------------------------------------------------------------------
# Tests for counterfactual probe
# ---------------------------------------------------------------------------


class TestCounterfactualProbe:
    def test_returns_correct_shapes(self):
        model = _StubModel(n_slots=3, memory_benefit=1.0)
        outer = model.outer_model
        input_ids = torch.randint(0, 32, (2, 17))
        valid_mask = torch.ones(2, 17)
        result = counterfactual_probe(
            model=model, outer=outer,
            probe_input_ids=input_ids, probe_valid_mask=valid_mask,
        )
        assert result.marginal_gains.shape[0] == 3  # one per slot
        assert result.nll_baseline.shape == (2, 16)
        assert result.nll_no_sidecar.shape == (2, 16)

    def test_no_slots(self):
        model = _StubModel(n_slots=0)
        outer = model.outer_model
        input_ids = torch.randint(0, 32, (2, 17))
        valid_mask = torch.ones(2, 17)
        result = counterfactual_probe(
            model=model, outer=outer,
            probe_input_ids=input_ids, probe_valid_mask=valid_mask,
        )
        assert len(result.slot_indices) == 0

    def test_chunking_does_not_change_probe_values(self):
        torch.manual_seed(123)
        model = _StubModel(n_slots=4, memory_benefit=1.0)
        outer = model.outer_model
        input_ids = torch.randint(0, 32, (2, 17))
        valid_mask = torch.ones(2, 17)

        torch.manual_seed(999)
        one_at_a_time = counterfactual_probe(
            model=model,
            outer=outer,
            probe_input_ids=input_ids,
            probe_valid_mask=valid_mask,
            chunk_size=1,
        )
        torch.manual_seed(999)
        all_at_once = counterfactual_probe(
            model=model,
            outer=outer,
            probe_input_ids=input_ids,
            probe_valid_mask=valid_mask,
            chunk_size=99,
        )

        assert one_at_a_time.slot_indices == all_at_once.slot_indices
        assert torch.allclose(one_at_a_time.marginal_gains, all_at_once.marginal_gains)
        assert torch.allclose(one_at_a_time.sidecar_value, all_at_once.sidecar_value)
        assert torch.allclose(one_at_a_time.weights_baseline, all_at_once.weights_baseline)

    def test_subset_probe_matches_full_probe_for_selected_slot(self):
        torch.manual_seed(123)
        model = _StubModel(n_slots=4, memory_benefit=1.0)
        outer = model.outer_model
        input_ids = torch.randint(0, 32, (2, 17))
        valid_mask = torch.ones(2, 17)

        torch.manual_seed(999)
        full = counterfactual_probe(
            model=model,
            outer=outer,
            probe_input_ids=input_ids,
            probe_valid_mask=valid_mask,
        )
        torch.manual_seed(999)
        subset = counterfactual_probe(
            model=model,
            outer=outer,
            probe_input_ids=input_ids,
            probe_valid_mask=valid_mask,
            score_slot_indices=[2],
        )

        assert subset.slot_indices == [2]
        assert torch.allclose(subset.marginal_gains[0], full.marginal_gains[2])
        assert torch.allclose(subset.sidecar_value, full.sidecar_value)
        assert torch.allclose(subset.weights_baseline[:, 0], full.weights_baseline[:, 2])


class TestOracleConfirmation:
    def test_batched_oracle_uses_one_encode_with_slot_masks(self):
        model = _StubModel(n_slots=4, memory_benefit=0.25, use_table=True)
        original_encode = model.encode
        seen_masks: list[torch.Tensor] = []

        def counted_encode(input_ids: torch.Tensor, **kwargs):
            mask = kwargs.get("memory_slot_mask")
            assert mask is not None
            seen_masks.append(mask.detach().clone())
            return original_encode(input_ids, **kwargs)

        model.encode = counted_encode  # type: ignore[method-assign]
        input_ids = torch.randint(0, 32, (2, 17))
        valid_mask = torch.ones(2, 17)
        result = oracle_confirm_slots(
            model=model,
            outer=model.outer_model,
            probe_input_ids=input_ids,
            probe_valid_mask=valid_mask,
            slot_indices=[0, 2],
        )
        assert result.slot_indices == [0, 2]
        assert result.oracle_deltas.shape[0] == 2
        assert len(seen_masks) == 1
        mask = seen_masks[0]
        assert mask.shape == (8, 4)  # (baseline, off, hide0, hide2) x batch
        assert mask[0:2].all()
        assert not mask[2:4].any()
        assert not mask[4:6, 0].any()
        assert mask[4:6, 1:].all()
        assert not mask[6:8, 2].any()


# ---------------------------------------------------------------------------
# Tests for signal decomposition
# ---------------------------------------------------------------------------


class TestSharpness:
    def test_high_variance_is_sharp(self):
        mg = torch.tensor([[[0.0, 0.0, 5.0, 0.0]]])  # one slot, spike on one token
        mask = torch.ones(1, 4).bool()
        sharp = _compute_per_slot_sharpness(mg, mask)
        assert sharp[0].item() > 0

    def test_uniform_is_not_sharp(self):
        mg = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        mask = torch.ones(1, 4).bool()
        sharp = _compute_per_slot_sharpness(mg, mask)
        assert sharp[0].item() == pytest.approx(0.0, abs=1e-6)

    def test_empty(self):
        mg = torch.zeros(0, 2, 4)
        mask = torch.ones(2, 4).bool()
        sharp = _compute_per_slot_sharpness(mg, mask)
        assert len(sharp) == 0


class TestRepresentationDrift:
    def test_identical_roundtrip_zero_drift(self):
        outer = _StubOuterModel(outer_dim=8)
        # Make encoder and decoder inverses (approximately)
        with torch.no_grad():
            outer.encoder.weight.copy_(torch.eye(8, 16)[:8, :])
            outer.decoder.weight.copy_(torch.eye(16, 8)[:, :])
        # A slot that round-trips cleanly through tanh(encoder(decoder(slot)))
        slot = torch.tanh(torch.randn(1, 8) * 0.1)
        drift = _compute_representation_drift(outer, slot)
        # Should be low (near-identity with tanh compression)
        assert drift < 0.5

    def test_no_encoder(self):
        outer = _StubOuterModel()
        outer.encoder = None
        drift = _compute_representation_drift(outer, torch.randn(1, 8))
        assert drift == 0.0


# ---------------------------------------------------------------------------
# Tests for MaintenancePolicy
# ---------------------------------------------------------------------------


class TestPolicy:
    def _make_rec(self, **kwargs):
        defaults = dict(
            slot_id=0, state=SLOT_ACTIVE, created_step=0,
            last_scored_step=500, score_count=5,
        )
        defaults.update(kwargs)
        return SlotRecord(**defaults)

    def _policy_kwargs(self):
        return dict(
            eviction_threshold=0.01,
            useful_threshold=0.005,
            drift_threshold=0.3,
            quarantine_threshold=-0.01,
            distill_peak_threshold=0.04,
            peak_preserve_utility_threshold=0.04,
            peak_preserve_sharpness_threshold=0.04,
            min_age=128,
            min_score_count=2,
            capacity_pressure=False,
        )

    def test_preserve_high_utility(self):
        p = MaintenancePolicy()
        rec = self._make_rec(marginal_gain_ema=0.5, utility_ema=0.5)
        action = p.choose(rec, **self._policy_kwargs())
        assert action == SLOT_PRESERVE

    def test_low_utility_below_capacity_does_not_evict(self):
        p = MaintenancePolicy()
        rec = self._make_rec(utility_ema=0.001, marginal_gain_ema=0.001)
        action = p.choose(rec, **self._policy_kwargs())
        assert action != SLOT_EVICT

    def test_capacity_pressure_evicts_low_utility(self):
        p = MaintenancePolicy()
        rec = self._make_rec(utility_ema=0.001, marginal_gain_ema=0.001)
        kwargs = self._policy_kwargs()
        kwargs["capacity_pressure"] = True
        action = p.choose(rec, **kwargs)
        assert action == SLOT_EVICT

    def test_quarantine_contradictory(self):
        p = MaintenancePolicy()
        rec = self._make_rec(
            contradiction_ema=0.5, negative_streak=3,
            utility_ema=-0.1, marginal_gain_ema=-0.1,
        )
        action = p.choose(rec, **self._policy_kwargs())
        assert action == SLOT_QUARANTINE

    def test_distill_internalized(self):
        p = MaintenancePolicy()
        rec = self._make_rec(
            peak_utility=0.5, utility_ema=0.015,
            marginal_gain_ema=0.015,
        )
        kwargs = self._policy_kwargs()
        kwargs["peak_preserve_utility_threshold"] = 1.0
        action = p.choose(rec, **kwargs)
        assert action == SLOT_DISTILL

    def test_peak_utility_blocks_distill_and_evict(self):
        p = MaintenancePolicy()
        rec = self._make_rec(
            peak_utility=0.5,
            utility_ema=0.001,
            marginal_gain_ema=0.001,
        )
        kwargs = self._policy_kwargs()
        kwargs["capacity_pressure"] = True
        action = p.choose(rec, **kwargs)
        assert action not in {SLOT_DISTILL, SLOT_EVICT}

    def test_peak_sharpness_blocks_distill_and_evict(self):
        p = MaintenancePolicy()
        rec = self._make_rec(
            peak_utility=0.5,
            peak_sharpness=0.5,
            utility_ema=0.001,
            marginal_gain_ema=0.001,
        )
        kwargs = self._policy_kwargs()
        kwargs["capacity_pressure"] = True
        kwargs["peak_preserve_utility_threshold"] = 1.0
        action = p.choose(rec, **kwargs)
        assert action not in {SLOT_DISTILL, SLOT_EVICT}

    def test_incidental_high_peak_trace_is_preserved_under_capacity_pressure(self):
        p = MaintenancePolicy()
        rec = self._make_rec(
            peak_utility=0.5,
            utility_ema=0.001,
            marginal_gain_ema=0.001,
        )
        kwargs = self._policy_kwargs()
        kwargs["capacity_pressure"] = True
        action = p.choose(rec, **kwargs)
        assert action == SLOT_PRESERVE

    def test_refresh_drifted(self):
        p = MaintenancePolicy()
        rec = self._make_rec(
            activation_drift_ema=0.5, marginal_gain_ema=0.1,
            utility_ema=0.1,
        )
        action = p.choose(rec, **self._policy_kwargs())
        assert action == SLOT_REFRESH

    def test_decay_young_weak(self):
        p = MaintenancePolicy()
        rec = self._make_rec(
            last_scored_step=10, utility_ema=0.005,
            marginal_gain_ema=0.001, sharpness_ema=0.0,
        )
        action = p.choose(rec, **self._policy_kwargs())
        assert action == SLOT_DECAY

    def test_action_values_dict(self):
        p = MaintenancePolicy()
        rec = self._make_rec(marginal_gain_ema=0.1, utility_ema=0.1)
        vals = p.action_values(rec, **self._policy_kwargs())
        assert set(vals.keys()) == {
            SLOT_PRESERVE, SLOT_DECAY, SLOT_EVICT,
            SLOT_REFRESH, SLOT_QUARANTINE, SLOT_DISTILL,
        }


# ---------------------------------------------------------------------------
# Tests for SlotTable-based tick
# ---------------------------------------------------------------------------


class TestSlotTableTick:
    def test_tick_with_table(self):
        loop = ReplayEvictionLoop(min_slot_age_steps=0, eviction_threshold=100.0)
        model = _StubModel(n_slots=3, memory_benefit=0.0, use_table=True)
        input_ids = torch.randint(0, 32, (2, 17))
        valid_mask = torch.ones(2, 17)
        loop.cache_probe(input_ids=input_ids, valid_mask=valid_mask,
                         cache_read_cutoff=None, step=0)
        result = loop.tick(model=model, step=100)
        assert isinstance(result, TickResult)
        assert loop._tick_count == 1

    def test_shadow_mode_classifies_without_mutating_table(self):
        loop = ReplayEvictionLoop(
            action_mode="shadow",
            min_slot_age_steps=0,
            eviction_threshold=100.0,
        )
        model = _StubModel(n_slots=3, memory_benefit=0.0, use_table=True)
        before = list(model.outer_model.table.active_slot_ids())
        input_ids = torch.randint(0, 32, (2, 17))
        valid_mask = torch.ones(2, 17)
        loop.cache_probe(
            input_ids=input_ids,
            valid_mask=valid_mask,
            cache_read_cutoff=None,
            step=0,
        )
        result = loop.tick(model=model, step=100)
        assert result.evicted_indices == []
        assert model.outer_model.table.active_slot_ids() == before
        diag = loop.diagnostics()
        assert diag["action_mode"] == "shadow"
        assert diag["shadow_actions_total"] > 0

    def test_oracle_rejection_blocks_active_mutation(self):
        loop = ReplayEvictionLoop(action_mode="active", min_slot_age_steps=0)
        model = _StubModel(n_slots=1, memory_benefit=0.0, use_table=True)
        table = model.outer_model.table
        sid = table.active_slot_ids()[0]
        rec = table.record(sid)
        rec.score_count = 3
        rec.utility_ema = -1.0
        rec.marginal_gain_ema = -1.0
        rec.contradiction_ema = 1.0
        rec.negative_streak = 3
        cf = CounterfactualResult(
            marginal_gains=torch.zeros(1, 1, 2),
            sidecar_value=torch.zeros(1, 2),
            nll_baseline=torch.zeros(1, 2),
            nll_no_sidecar=torch.zeros(1, 2),
            weights_baseline=torch.ones(1, 1),
            mask=torch.ones(1, 2, dtype=torch.bool),
            slot_indices=[0],
        )
        loop._confirm_actions_with_oracle = lambda **_kwargs: {sid: False}  # type: ignore[method-assign]
        result = loop._classify_and_act(
            model=model,
            outer=model.outer_model,
            step=200,
            cf=cf,
            slot_marginals=[-1.0],
            sharpness_per_slot=torch.zeros(1),
            t0=time.monotonic(),
        )
        assert result.evicted_indices == []
        assert table.active_slot_ids() == [sid]
        assert sid not in loop._quarantined


class TestDistillReceipt:
    def test_fields(self):
        r = DistillReceipt(
            slot_id=1, step=100,
            marginal_gain_before=0.5, marginal_gain_peak=1.0,
            marginal_gain_current=0.01, target="latent_trace",
        )
        assert r.slot_id == 1
        assert r.accepted is False


class TestMemoryEventStruct:
    def test_fields(self):
        e = MemoryEvent(step=10, tick=1, slot_id=0, action="PRESERVE")
        assert e.step == 10
        assert e.accepted is True
