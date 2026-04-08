"""Tests for partition-scoped SleepCycle — stages operate on owned slots only."""
from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock

from chaoscontrol.sleep import SleepCycle, SleepConfig
from chaoscontrol.partition import SemanticPartition
from chaoscontrol.wake_cache import WakeCache
from chaoscontrol.memory import MultiSlotOuterModel


def _make_model(n_slots=8, dim=16, outer_dim=8):
    """Create a mock model with real MultiSlotOuterModel."""
    om = MultiSlotOuterModel(model_dim=dim, outer_dim=outer_dim, max_slots=32)
    for i in range(n_slots):
        om.write(torch.randn(1, 1, dim), bucket_id=i % 4)

    model = MagicMock()
    model.outer_model = om
    model.semantic_tier = None
    model.typed_storage = True
    return model


def test_partition_scoped_n3_isolation():
    """N3 prune should only affect partition's slots."""
    model = _make_model(8)
    om = model.outer_model
    cache = WakeCache(max_moments=8, max_hidden_buffer=8)

    # Partition 0 owns buckets {0, 1} -> slots at indices 0,1,4,5
    partition = SemanticPartition(partition_id=0, bucket_ids={0, 1})
    partition.mode = "sleeping"

    # Set all survivals low so N3 would want to prune
    for i in range(len(om._survival)):
        om._survival[i] = 0.01

    # Record survivals for non-partition slots
    non_partition_survivals = {i: om._survival[i] for i in [2, 3, 6, 7]}

    cfg = SleepConfig(stages="n3_only", budget=128, survival_floor=0.5)
    cycle = SleepCycle(cfg)
    diag = cycle.run(model, cache, device="cpu", partition=partition)

    # Non-partition slots (buckets 2,3) must be unchanged
    for idx, surv in non_partition_survivals.items():
        if idx < len(om._survival):
            assert om._survival[idx] == surv, (
                f"Slot {idx} (bucket {om._slot_buckets[idx]}) was modified by wrong partition"
            )


def test_no_partition_backward_compatible():
    """Without partition param, behavior is the same as before."""
    model = _make_model(4)
    cache = WakeCache(max_moments=8, max_hidden_buffer=8)

    cfg = SleepConfig(stages="n3_only", budget=128, survival_floor=0.5)
    cycle = SleepCycle(cfg)

    # Should not raise
    diag = cycle.run(model, cache, device="cpu")
    assert "n3" in diag


def test_run_accepts_partition_param():
    """run() signature accepts partition keyword."""
    model = _make_model(4)
    cache = WakeCache(max_moments=8, max_hidden_buffer=8)
    partition = SemanticPartition(partition_id=0, bucket_ids={0, 1}, mode="sleeping")

    cfg = SleepConfig(stages="n3_only", budget=64)
    cycle = SleepCycle(cfg)
    diag = cycle.run(model, cache, device="cpu", partition=partition)
    assert isinstance(diag, dict)


def test_partition_scoped_n1_only_flags_partition_slots():
    """N1 should only report unstable indices within the partition."""
    model = _make_model(8)
    om = model.outer_model

    # Mark all slots as unstable (survival=0)
    for i in range(len(om._survival)):
        om._survival[i] = 0.0

    partition = SemanticPartition(partition_id=0, bucket_ids={0, 1})
    partition.mode = "sleeping"

    cfg = SleepConfig(stages="full_cycle", n2_budget=0, rem_budget=0,
                      rem_dreams=0, survival_floor=0.0)
    cycle = SleepCycle(cfg)
    diag = cycle.run(model, cache=WakeCache(max_moments=8, max_hidden_buffer=8),
                     device="cpu", partition=partition)

    # N1 unstable indices should only include partition's slots
    partition_indices = set(om.get_partition_slot_indices(partition))
    n1_unstable = set(diag["n1"].get("unstable_indices", []))
    assert n1_unstable <= partition_indices, (
        f"N1 flagged non-partition slots: {n1_unstable - partition_indices}"
    )


def test_partition_scoped_n3_prunes_only_partition_slots():
    """N3 should only prune slots belonging to the partition."""
    model = _make_model(8)
    om = model.outer_model

    # Partition owns buckets {2, 3} -> slots at indices 2,3,6,7
    partition = SemanticPartition(partition_id=1, bucket_ids={2, 3})
    partition.mode = "sleeping"

    # Set partition slots to low survival, non-partition slots to high
    for i in range(len(om._survival)):
        bucket = om._slot_buckets[i]
        if bucket in {2, 3}:
            om._survival[i] = 0.01  # below floor -> prune
        else:
            om._survival[i] = 1.0  # safe

    initial_count = len(om._slots)
    non_partition_count = sum(
        1 for b in om._slot_buckets if b not in {2, 3}
    )

    cfg = SleepConfig(stages="n3_only", budget=128, survival_floor=0.5)
    cycle = SleepCycle(cfg)
    diag = cycle.run(model, cache=WakeCache(max_moments=8, max_hidden_buffer=8),
                     device="cpu", partition=partition)

    # Non-partition slots should still be present
    remaining_non_partition = sum(
        1 for b in om._slot_buckets if b not in {2, 3}
    )
    assert remaining_non_partition == non_partition_count, (
        "Non-partition slots were pruned"
    )

    # Some partition slots should have been pruned
    assert diag["n3"]["pruned"] > 0


def test_partition_scoped_n2_only_scores_partition_slots():
    """N2 should only score slots belonging to the partition."""
    model = _make_model(8)
    om = model.outer_model
    cache = WakeCache(max_moments=8, max_hidden_buffer=8)

    # Add moments so N2 has data to score
    vocab_size = 256
    for i in range(4):
        cache.record_moment(
            surprise=float(i + 1),
            inputs=torch.randint(0, vocab_size, (1, 16)),
            targets=torch.randint(0, vocab_size, (1, 16)),
            hidden=torch.randn(1, 16, 16),
        )

    # Make model() return a dict with logits tensor so _compute_mean_ce works
    def fake_forward(inputs):
        batch, seq = inputs.shape
        return {"logits": torch.randn(batch, seq, vocab_size)}
    model.side_effect = fake_forward

    partition = SemanticPartition(partition_id=0, bucket_ids={0, 1})
    partition.mode = "sleeping"

    # Give all slots nonzero survival so N1 doesn't skip them
    for i in range(len(om._survival)):
        om._survival[i] = 0.5

    # Record non-partition survivals before
    non_partition_indices = [
        i for i in range(len(om._slots))
        if om._slot_buckets[i] not in {0, 1}
    ]
    non_partition_survivals = {i: om._survival[i] for i in non_partition_indices}

    cfg = SleepConfig(stages="n2_n3", n2_budget=64, survival_floor=0.0)
    cycle = SleepCycle(cfg)
    diag = cycle.run(model, cache, device="cpu", partition=partition)

    # N2 should only score partition slots
    scored_indices = [idx for idx, _ in diag["n2"].get("utilities", [])]
    partition_slot_indices = set(om.get_partition_slot_indices(partition))

    for scored_idx in scored_indices:
        assert scored_idx in partition_slot_indices, (
            f"N2 scored non-partition slot index {scored_idx}"
        )
