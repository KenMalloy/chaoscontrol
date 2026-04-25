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
        "birth_embedding_version", "occupied",
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
