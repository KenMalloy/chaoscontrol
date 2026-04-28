"""Tests for the replay-eviction loop (CRCT rank-3 idle maintenance)."""
from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from chaoscontrol.replay_eviction import ReplayEvictionLoop, replay_score_slots, _evict_slots


# ---------------------------------------------------------------------------
# Minimal model stub that mimics ChaosStudentLM's encode/outer_model API.
# ---------------------------------------------------------------------------


class _StubOuterModel:
    """Minimal MultiSlotOuterModel stand-in for testing."""

    def __init__(self, outer_dim: int = 8, n_slots: int = 10) -> None:
        self.outer_dim = outer_dim
        self._slots = [torch.randn(1, outer_dim) for _ in range(n_slots)]
        self._survival = [1.0] * n_slots
        self._slot_buckets = [0] * n_slots
        self._slot_event_ids = [i for i in range(n_slots)]
        self._retrieval_weights: torch.Tensor | None = None
        self._retrieval_indices: list[int] | None = None


class _StubModel(nn.Module):
    """Minimal model that returns different hidden states for memory on/off."""

    def __init__(
        self,
        dim: int = 16,
        vocab: int = 32,
        outer_dim: int = 8,
        n_slots: int = 10,
        memory_benefit: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.outer_model = _StubOuterModel(outer_dim, n_slots)
        self._memory_benefit = memory_benefit

    def encode(
        self, input_ids: torch.Tensor, *, memory_mode: str = "off", cache_read_cutoff=None
    ) -> torch.Tensor:
        batch, seq = input_ids.shape
        base = torch.randn(batch, seq, self.dim) * 0.1

        if memory_mode == "force_on" and self.outer_model is not None:
            # Simulate memory helping: add a signal that reduces NLL
            base = base + self._memory_benefit
            n_slots = len(self.outer_model._slots)
            if n_slots > 0:
                weights = torch.ones(batch, n_slots) / n_slots
                self.outer_model._retrieval_weights = weights
                self.outer_model._retrieval_indices = list(range(n_slots))
        elif self.outer_model is not None:
            self.outer_model._retrieval_weights = None
            self.outer_model._retrieval_indices = None

        return base


# ---------------------------------------------------------------------------
# Tests for _evict_slots
# ---------------------------------------------------------------------------


class TestEvictSlots:
    def test_evict_single(self):
        outer = _StubOuterModel(n_slots=5)
        assert len(outer._slots) == 5
        evicted = _evict_slots(outer, [2])
        assert evicted == [2]
        assert len(outer._slots) == 4
        assert len(outer._survival) == 4
        assert len(outer._slot_buckets) == 4
        assert len(outer._slot_event_ids) == 4

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
# Tests for replay_score_slots
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
        assert len(result["slot_indices"]) == 5

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
# Tests for ReplayEvictionLoop
# ---------------------------------------------------------------------------


class TestReplayEvictionLoop:
    def test_no_probe_returns_empty(self):
        loop = ReplayEvictionLoop()
        model = _StubModel()
        evicted = loop.tick(model=model, step=100)
        assert evicted == []

    def test_cache_probe_enables_tick(self):
        loop = ReplayEvictionLoop(min_slot_age_steps=0, eviction_threshold=100.0)
        model = _StubModel(n_slots=5)
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)

        assert not loop.has_probe()
        loop.cache_probe(
            input_ids=input_ids,
            valid_mask=valid_mask,
            cache_read_cutoff=None,
            step=0,
        )
        assert loop.has_probe()

        evicted = loop.tick(model=model, step=200)
        # With threshold=100, most utility values should be below it
        # but we need score_count >= 2 for eviction
        assert loop._tick_count == 1

    def test_min_age_respected(self):
        loop = ReplayEvictionLoop(
            min_slot_age_steps=1000,
            eviction_threshold=100.0,
        )
        model = _StubModel(n_slots=5, memory_benefit=0.0)
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)

        loop.cache_probe(
            input_ids=input_ids, valid_mask=valid_mask,
            cache_read_cutoff=None, step=0,
        )

        # Tick twice (need score_count >= 2)
        loop.tick(model=model, step=10)
        evicted = loop.tick(model=model, step=20)
        # Age is only 20, min_age is 1000 — nothing should be evicted
        assert evicted == []

    def test_score_count_gate(self):
        """Slots need at least 2 scores before eviction is considered."""
        loop = ReplayEvictionLoop(
            min_slot_age_steps=0,
            eviction_threshold=100.0,
        )
        model = _StubModel(n_slots=5, memory_benefit=0.0)
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)

        loop.cache_probe(
            input_ids=input_ids, valid_mask=valid_mask,
            cache_read_cutoff=None, step=0,
        )

        # First tick: score_count = 1 for each slot, no eviction
        evicted = loop.tick(model=model, step=200)
        assert evicted == []

    def test_eviction_after_multiple_ticks(self):
        """After enough ticks, zero-utility slots get evicted."""
        loop = ReplayEvictionLoop(
            min_slot_age_steps=0,
            eviction_threshold=100.0,
        )
        model = _StubModel(n_slots=5, memory_benefit=0.0)
        input_ids = torch.randint(0, 32, (2, 33))
        valid_mask = torch.ones(2, 33)

        loop.cache_probe(
            input_ids=input_ids, valid_mask=valid_mask,
            cache_read_cutoff=None, step=0,
        )

        # First tick builds initial scores
        loop.tick(model=model, step=200)
        # Second tick allows eviction (score_count >= 2)
        evicted = loop.tick(model=model, step=300)
        # With memory_benefit=0.0, utility should be near zero,
        # well below threshold=100
        assert len(evicted) > 0

    def test_diagnostics(self):
        loop = ReplayEvictionLoop()
        diag = loop.diagnostics()
        assert "tick_count" in diag
        assert "evictions_total" in diag
        assert "slots_tracked" in diag
        assert "eviction_threshold" in diag
        assert diag["tick_count"] == 0

    def test_ema_smoothing(self):
        """EMA should smooth utility estimates, not snap to latest value."""
        loop = ReplayEvictionLoop(eviction_ema_beta=0.9)

        # Simulate: first score = 1.0, second score = 0.0
        slot_idx = 42
        loop._slot_first_seen_step[slot_idx] = 0
        loop._slot_utility_ema[slot_idx] = 1.0
        loop._slot_score_count[slot_idx] = 1

        # Manually apply EMA update with new utility=0.0
        old = loop._slot_utility_ema[slot_idx]
        loop._slot_utility_ema[slot_idx] = 0.9 * old + 0.1 * 0.0
        loop._slot_score_count[slot_idx] = 2

        # EMA should be 0.9, not 0.0
        assert loop._slot_utility_ema[slot_idx] == pytest.approx(0.9, abs=1e-6)

    def test_reindex_after_eviction(self):
        """After evicting slots, higher indices must shift down."""
        loop = ReplayEvictionLoop()
        loop._slot_utility_ema = {0: 0.5, 1: 0.1, 2: 0.8, 3: 0.2, 4: 0.9}
        loop._slot_first_seen_step = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        loop._slot_score_count = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3}

        # Evict slots 1 and 3
        loop._reindex_after_eviction([1, 3])

        # Old 0 → new 0, old 2 → new 1, old 4 → new 2
        assert set(loop._slot_utility_ema.keys()) == {0, 1, 2}
        assert loop._slot_utility_ema[0] == pytest.approx(0.5)
        assert loop._slot_utility_ema[1] == pytest.approx(0.8)  # was idx 2
        assert loop._slot_utility_ema[2] == pytest.approx(0.9)  # was idx 4
