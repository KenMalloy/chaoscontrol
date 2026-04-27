"""Tests for the EpisodicCache substrate (component 1 of the memory subsystem)."""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.optim.episodic_cache import CacheEntry, EpisodicCache


def _entry_kwargs(
    *,
    key_fp: int,
    span_length: int = 4,
    key_rep_dim: int = 8,
    value_anchor_id: int = 0,
    current_step: int = 0,
    embedding_version: int = 0,
    span_start: int = 0,
) -> dict:
    return {
        "key_fp": key_fp,
        "key_rep": torch.full((key_rep_dim,), float(key_fp % 17)),
        "value_tok_ids": torch.arange(
            span_start, span_start + span_length, dtype=torch.int64,
        ),
        "value_anchor_id": value_anchor_id,
        "current_step": current_step,
        "embedding_version": embedding_version,
    }


def test_constructs_with_zeroed_storage():
    cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
    assert len(cache) == 0
    assert not cache.is_full
    # All slots unoccupied; tensors initialized.
    assert cache.occupied.dtype == torch.bool
    assert not cache.occupied.any()
    assert cache.key_fp.shape == (4,)
    assert cache.value_tok_ids.shape == (4, 4)
    assert cache.key_rep.shape == (4, 8)


def test_rejects_invalid_construction_args():
    with pytest.raises(ValueError):
        EpisodicCache(capacity=0)
    with pytest.raises(ValueError):
        EpisodicCache(capacity=4, span_length=0)
    with pytest.raises(ValueError):
        EpisodicCache(capacity=4, key_rep_dim=0)
    with pytest.raises(ValueError):
        EpisodicCache(capacity=4, utility_ema_decay=0.0)
    with pytest.raises(ValueError):
        EpisodicCache(capacity=4, utility_ema_decay=1.0)


def test_append_then_query_returns_entry():
    cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
    slot = cache.append(**_entry_kwargs(key_fp=12345, current_step=5))
    assert slot == 0
    assert len(cache) == 1
    entry = cache.query(12345)
    assert entry is not None
    assert isinstance(entry, CacheEntry)
    assert entry.slot == 0
    assert entry.key_fp == 12345
    assert entry.write_step == 5
    assert torch.equal(entry.value_tok_ids, torch.tensor([0, 1, 2, 3]))


def test_query_returns_none_on_miss():
    cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
    assert cache.query(99999) is None
    cache.append(**_entry_kwargs(key_fp=42))
    assert cache.query(99999) is None
    assert cache.query(42) is not None


def test_append_rejects_wrong_shape():
    cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
    with pytest.raises(ValueError, match="key_rep must have shape"):
        cache.append(
            key_fp=1,
            key_rep=torch.zeros(7),
            value_tok_ids=torch.zeros(4, dtype=torch.int64),
            value_anchor_id=0,
            current_step=0,
            embedding_version=0,
        )
    with pytest.raises(ValueError, match="value_tok_ids must have shape"):
        cache.append(
            key_fp=1,
            key_rep=torch.zeros(8),
            value_tok_ids=torch.zeros(3, dtype=torch.int64),
            value_anchor_id=0,
            current_step=0,
            embedding_version=0,
        )


def test_capacity_bound_triggers_eviction():
    cache = EpisodicCache(
        capacity=3, span_length=2, key_rep_dim=4, grace_steps=10,
    )
    for i in range(3):
        cache.append(**_entry_kwargs(
            key_fp=100 + i, span_length=2, key_rep_dim=4,
            current_step=i,
        ))
    assert cache.is_full
    # All 3 entries are within grace; the next append must evict the
    # oldest (write_step=0 -> key_fp=100).
    cache.append(**_entry_kwargs(
        key_fp=999, span_length=2, key_rep_dim=4, current_step=5,
    ))
    assert len(cache) == 3
    assert cache.query(100) is None
    assert cache.query(101) is not None
    assert cache.query(102) is not None
    assert cache.query(999) is not None


def test_eviction_prefers_lowest_utility_past_grace():
    cache = EpisodicCache(
        capacity=3, span_length=2, key_rep_dim=4, grace_steps=10,
    )
    for i in range(3):
        cache.append(**_entry_kwargs(
            key_fp=100 + i, span_length=2, key_rep_dim=4,
            current_step=i,
        ))
    # Push EMAs apart: middle entry is high-utility, others low.
    e0 = cache.query(100)
    e1 = cache.query(101)
    e2 = cache.query(102)
    assert e0 is not None and e1 is not None and e2 is not None
    cache.update_utility(e0.slot, ce_delta=0.0)
    cache.update_utility(e1.slot, ce_delta=10.0)
    cache.update_utility(e2.slot, ce_delta=0.5)
    # Force all entries past their grace period.
    current_step = 100  # >> 10 grace
    cache.append(**_entry_kwargs(
        key_fp=999, span_length=2, key_rep_dim=4,
        current_step=current_step,
    ))
    # e1 (high utility) and e2 (medium) survive; e0 (lowest) evicted.
    assert cache.query(100) is None, "lowest-utility entry must be evicted"
    assert cache.query(101) is not None
    assert cache.query(102) is not None
    assert cache.query(999) is not None


def test_eviction_protection_score_resists_low_utility_eviction():
    """Learned eviction head contributes a bounded protection residual.

    The base utility still matters, but a protected entry should not be the
    first past-grace victim when another entry has the lower combined score.
    """
    cache = EpisodicCache(
        capacity=2, span_length=2, key_rep_dim=4, grace_steps=0,
    )
    slot_a = cache.append(**_entry_kwargs(
        key_fp=10, span_length=2, key_rep_dim=4, current_step=0,
    ))
    slot_b = cache.append(**_entry_kwargs(
        key_fp=11, span_length=2, key_rep_dim=4, current_step=1,
    ))
    cache.utility_u[slot_a] = 0.1
    cache.utility_u[slot_b] = 0.4
    cache.protection_score[slot_a] = 1.0
    cache.protection_score[slot_b] = 0.0

    cache.append(**_entry_kwargs(
        key_fp=12, span_length=2, key_rep_dim=4, current_step=10,
    ))

    assert cache.query(10) is not None
    assert cache.query(11) is None
    assert cache.query(12) is not None


def test_eviction_within_grace_falls_back_to_oldest_write_step():
    """When no entries are past their grace period, the cache must still
    rotate to make room — by oldest write_step (FIFO during warm-up)."""
    cache = EpisodicCache(
        capacity=2, span_length=2, key_rep_dim=4, grace_steps=1000,
    )
    cache.append(**_entry_kwargs(
        key_fp=1, span_length=2, key_rep_dim=4, current_step=0,
    ))
    cache.append(**_entry_kwargs(
        key_fp=2, span_length=2, key_rep_dim=4, current_step=10,
    ))
    # All entries within grace at step=20.
    cache.append(**_entry_kwargs(
        key_fp=3, span_length=2, key_rep_dim=4, current_step=20,
    ))
    assert cache.query(1) is None, "oldest write_step entry should be evicted"
    assert cache.query(2) is not None
    assert cache.query(3) is not None


def test_utility_ema_update_pulls_toward_signal():
    cache = EpisodicCache(
        capacity=2, span_length=2, key_rep_dim=4,
        utility_ema_decay=0.5,
    )
    cache.append(**_entry_kwargs(
        key_fp=1, span_length=2, key_rep_dim=4, current_step=0,
    ))
    e = cache.query(1)
    assert e is not None
    # New entries enter at utility_u=1.0 so retrieval-time scoring
    # `score = cosine_sim × utility_u` doesn't degenerate to zero.
    assert e.utility_u == pytest.approx(1.0)
    cache.update_utility(e.slot, ce_delta=2.0)
    e = cache.query(1)
    assert e is not None
    # With decay 0.5: new = 0.5 * 1.0 + 0.5 * 2.0 = 1.5
    assert e.utility_u == pytest.approx(1.5)
    cache.update_utility(e.slot, ce_delta=2.0)
    e = cache.query(1)
    assert e is not None
    # new = 0.5 * 1.5 + 0.5 * 2.0 = 1.75
    assert e.utility_u == pytest.approx(1.75)


def test_update_utility_on_unoccupied_slot_is_silent():
    """Feedback packets racing eviction must not crash the controller."""
    cache = EpisodicCache(capacity=2, span_length=2, key_rep_dim=4)
    cache.update_utility(0, ce_delta=5.0)
    cache.update_utility(1, ce_delta=-3.0)
    assert len(cache) == 0


def test_mark_fired_updates_last_fired_step_only():
    cache = EpisodicCache(capacity=2, span_length=2, key_rep_dim=4)
    cache.append(**_entry_kwargs(
        key_fp=1, span_length=2, key_rep_dim=4, current_step=0,
    ))
    e = cache.query(1)
    assert e is not None
    assert e.last_fired_step == -1
    cache.mark_fired(e.slot, current_step=42)
    e2 = cache.query(1)
    assert e2 is not None
    assert e2.last_fired_step == 42
    # Utility was not changed by mark_fired (still at init=1.0).
    assert e2.utility_u == pytest.approx(1.0)


def test_evict_clears_slot_and_removes_index_entry():
    cache = EpisodicCache(capacity=2, span_length=2, key_rep_dim=4)
    cache.append(**_entry_kwargs(
        key_fp=1, span_length=2, key_rep_dim=4, current_step=0,
    ))
    e = cache.query(1)
    assert e is not None
    cache.evict(e.slot)
    assert cache.query(1) is None
    assert len(cache) == 0
    # Idempotent.
    cache.evict(e.slot)
    assert len(cache) == 0


def test_duplicate_fingerprint_writes_into_a_new_slot():
    """If the same fingerprint is written twice, the most recent write wins
    on lookup. The older slot remains occupied but unreachable by hash;
    eviction will reclaim it on subsequent appends."""
    cache = EpisodicCache(capacity=4, span_length=2, key_rep_dim=4)
    cache.append(**_entry_kwargs(
        key_fp=42, span_length=2, key_rep_dim=4,
        current_step=0, span_start=0,
    ))
    cache.append(**_entry_kwargs(
        key_fp=42, span_length=2, key_rep_dim=4,
        current_step=1, span_start=100,
    ))
    e = cache.query(42)
    assert e is not None
    # The later write's tokens are returned.
    assert torch.equal(e.value_tok_ids, torch.tensor([100, 101]))
    assert e.write_step == 1


def test_snapshot_to_returns_tensor_dict_on_requested_device():
    cache = EpisodicCache(capacity=4, span_length=2, key_rep_dim=4)
    cache.append(**_entry_kwargs(
        key_fp=7, span_length=2, key_rep_dim=4, current_step=3,
    ))
    snap = cache.snapshot_to(torch.device("cpu"))
    expected_keys = {
        "key_fp", "key_rep", "value_tok_ids", "value_anchor_id",
        "utility_u", "last_fired_step", "write_step",
        "birth_embedding_version", "occupied", "pressure_at_write",
        "source_write_id", "write_bucket", "protection_score", "slot_state",
        "simplex_edge_slot", "simplex_edge_weight",
    }
    assert set(snap.keys()) == expected_keys
    for name, t in snap.items():
        assert t.device.type == "cpu", f"{name} on wrong device: {t.device}"
    # Snapshot reflects the current state of the cache.
    assert snap["occupied"][0].item() is True
    assert int(snap["key_fp"][0].item()) == 7
    assert int(snap["write_step"][0].item()) == 3


def test_evict_invalid_slot_raises():
    cache = EpisodicCache(capacity=2, span_length=2, key_rep_dim=4)
    with pytest.raises(IndexError):
        cache.evict(5)
    with pytest.raises(IndexError):
        cache.evict(-1)


def test_reset_returns_to_post_construction_state():
    """reset() must put the cache in a state indistinguishable from a freshly
    constructed one of the same shape — needed for per-doc cache resets at
    eval time. Pin the same invariants that test_constructs_with_zeroed_storage
    pins on a brand-new cache.
    """
    cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
    # Fill it with state.
    cache.append(**_entry_kwargs(
        key_fp=42, span_length=4, key_rep_dim=8, current_step=5,
        embedding_version=7, value_anchor_id=99,
    ))
    cache.append(**_entry_kwargs(
        key_fp=99, span_length=4, key_rep_dim=8, current_step=10,
        embedding_version=7,
    ))
    e = cache.query(42)
    assert e is not None
    cache.mark_fired(e.slot, current_step=20)
    cache.update_utility(e.slot, ce_delta=2.5)
    assert len(cache) == 2

    cache.reset()

    # Mirror of test_constructs_with_zeroed_storage assertions:
    assert len(cache) == 0
    assert not cache.is_full
    assert cache.occupied.dtype == torch.bool
    assert not cache.occupied.any()
    assert cache.key_fp.shape == (4,)
    assert cache.value_tok_ids.shape == (4, 4)
    assert cache.key_rep.shape == (4, 8)
    # All field tensors zeroed (or -1 for the step trackers, matching __init__).
    assert torch.all(cache.key_fp == 0)
    assert torch.all(cache.key_rep == 0)
    assert torch.all(cache.value_tok_ids == 0)
    assert torch.all(cache.value_anchor_id == 0)
    assert torch.all(cache.utility_u == 0)
    assert torch.all(cache.last_fired_step == -1)
    assert torch.all(cache.write_step == -1)
    assert torch.all(cache.birth_embedding_version == 0)
    # Hash index cleared so the same fingerprint can be reinserted from a
    # blank slate.
    assert cache._fp_index == {}
    assert cache.query(42) is None
    assert cache.query(99) is None

    # Post-reset, the cache is fully usable again — re-append should land in
    # slot 0 and be queryable.
    slot = cache.append(**_entry_kwargs(
        key_fp=42, span_length=4, key_rep_dim=8, current_step=0,
    ))
    assert slot == 0
    e2 = cache.query(42)
    assert e2 is not None
    assert e2.slot == 0
    assert e2.write_step == 0


def test_reset_preserves_capacity_and_config():
    """reset() must not change capacity, span_length, key_rep_dim, grace_steps,
    or utility_ema_decay — those are construction-time choices, not state.
    """
    cache = EpisodicCache(
        capacity=8, span_length=6, key_rep_dim=12,
        grace_steps=128, utility_ema_decay=0.95,
        fingerprint_window=10,
    )
    cache.append(**_entry_kwargs(
        key_fp=1, span_length=6, key_rep_dim=12, current_step=0,
    ))
    cache.reset()
    assert cache.capacity == 8
    assert cache.span_length == 6
    assert cache.key_rep_dim == 12
    assert cache.grace_steps == 128
    assert cache.utility_ema_decay == pytest.approx(0.95)
    assert cache.fingerprint_window == 10


def test_to_dict_from_dict_round_trip_preserves_state():
    """The save/load envelope must round-trip every field byte-equal — any
    silent default in from_dict is the failure mode that lets Arm B's
    cache shape silently diverge from the trainer's.
    """
    cache = EpisodicCache(
        capacity=4, span_length=4, key_rep_dim=8,
        grace_steps=50, utility_ema_decay=0.97,
        fingerprint_window=6,
    )
    # Populate two entries so the hash index, occupancy, and per-slot
    # tensors all carry non-default values.
    cache.append(**_entry_kwargs(
        key_fp=42, span_length=4, key_rep_dim=8,
        current_step=3, embedding_version=7, value_anchor_id=11,
    ))
    cache.append(**_entry_kwargs(
        key_fp=99, span_length=4, key_rep_dim=8,
        current_step=8, embedding_version=7, value_anchor_id=22,
        span_start=100,
    ))
    e = cache.query(42)
    assert e is not None
    cache.mark_fired(e.slot, current_step=15)
    cache.update_utility(e.slot, ce_delta=2.5)

    blob = cache.to_dict()
    restored = EpisodicCache.from_dict(blob)

    # Config fields equal.
    assert restored.capacity == cache.capacity
    assert restored.span_length == cache.span_length
    assert restored.key_rep_dim == cache.key_rep_dim
    assert restored.grace_steps == cache.grace_steps
    assert restored.utility_ema_decay == pytest.approx(cache.utility_ema_decay)
    assert restored.fingerprint_window == cache.fingerprint_window

    # Per-slot tensors equal element-wise.
    for name in EpisodicCache._TENSOR_FIELDS:
        original = getattr(cache, name)
        loaded = getattr(restored, name)
        assert torch.equal(original, loaded), f"{name} diverged after round-trip"

    # Hash index entries equal — must reconstruct content-addressable lookup.
    assert restored._fp_index == cache._fp_index
    # Functional check: queries on the restored cache return the same entries.
    e_orig = cache.query(42)
    e_load = restored.query(42)
    assert e_orig is not None and e_load is not None
    assert e_load.slot == e_orig.slot
    assert e_load.write_step == e_orig.write_step
    assert e_load.last_fired_step == e_orig.last_fired_step
    assert e_load.utility_u == pytest.approx(e_orig.utility_u)
    assert torch.equal(e_load.value_tok_ids, e_orig.value_tok_ids)


def test_to_dict_returns_clones_so_mutating_blob_does_not_corrupt_cache():
    """to_dict must hand back tensor clones — otherwise a downstream save
    pipeline mutating the blob would silently corrupt live cache state."""
    cache = EpisodicCache(capacity=2, span_length=2, key_rep_dim=4)
    cache.append(**_entry_kwargs(
        key_fp=1, span_length=2, key_rep_dim=4, current_step=0,
    ))
    blob = cache.to_dict()
    blob["key_fp"][0] = 99999
    # Live cache fingerprint must be unchanged.
    assert int(cache.key_fp[0].item()) == 1
    # And the index lookup still works.
    assert cache.query(1) is not None


def test_from_dict_raises_keyerror_on_missing_required_field():
    """Silent defaults here are the falsifier-failure mode: if a checkpoint
    payload is missing a required field, the load MUST raise KeyError so
    the run aborts early instead of silently scoring noise."""
    # Build a valid blob, then drop one required field at a time.
    cache = EpisodicCache(capacity=2, span_length=2, key_rep_dim=4)
    cache.append(**_entry_kwargs(
        key_fp=1, span_length=2, key_rep_dim=4, current_step=0,
    ))
    full = cache.to_dict()

    # Empty dict is missing every required field.
    with pytest.raises(KeyError, match="capacity"):
        EpisodicCache.from_dict({})

    required_fields = (
        *EpisodicCache._CONFIG_FIELDS,
        *EpisodicCache._TENSOR_FIELDS,
        "fp_index",
    )
    for field_name in required_fields:
        partial = dict(full)
        partial.pop(field_name)
        with pytest.raises(KeyError, match=field_name):
            EpisodicCache.from_dict(partial)


def test_cpu_ssm_extended_fields_round_trip_with_cache_payload():
    """CPU-controller V1 needs write pressure, durable source IDs, per-slot
    SSM state, and simplex edges to live on the cache substrate.

    These fields must round-trip through to_dict/from_dict because the
    trainer-side cache checkpoint is the handoff to cache-aware eval.
    """
    cache = EpisodicCache(
        capacity=3,
        span_length=2,
        key_rep_dim=4,
        slot_state_dim=3,
        simplex_k_max=2,
    )
    slot = cache.append(
        key_fp=123,
        key_rep=torch.ones(4),
        value_tok_ids=torch.tensor([7, 8], dtype=torch.int64),
        value_anchor_id=7,
        current_step=5,
        embedding_version=0,
        pressure_at_write=4.25,
        source_write_id=99,
        write_bucket=2,
        protection_score=0.75,
    )
    cache.slot_state[slot] = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float16)
    cache.simplex_edge_slot[slot] = torch.tensor([1, 2], dtype=torch.int64)
    cache.simplex_edge_weight[slot] = torch.tensor([0.25, 0.75], dtype=torch.float16)

    entry = cache.query(123)
    assert entry is not None
    assert entry.pressure_at_write == pytest.approx(4.25)
    assert entry.source_write_id == 99
    assert entry.write_bucket == 2
    assert entry.protection_score == pytest.approx(0.75)

    snap = cache.snapshot_to(torch.device("cpu"))
    for name in (
        "pressure_at_write",
        "source_write_id",
        "write_bucket",
        "protection_score",
        "slot_state",
        "simplex_edge_slot",
        "simplex_edge_weight",
    ):
        assert name in snap

    restored = EpisodicCache.from_dict(cache.to_dict())
    assert restored.slot_state_dim == 3
    assert restored.simplex_k_max == 2
    torch.testing.assert_close(restored.pressure_at_write, cache.pressure_at_write)
    assert torch.equal(restored.source_write_id, cache.source_write_id)
    assert torch.equal(restored.write_bucket, cache.write_bucket)
    torch.testing.assert_close(restored.protection_score, cache.protection_score)
    torch.testing.assert_close(restored.slot_state, cache.slot_state)
    assert torch.equal(restored.simplex_edge_slot, cache.simplex_edge_slot)
    torch.testing.assert_close(restored.simplex_edge_weight, cache.simplex_edge_weight)
