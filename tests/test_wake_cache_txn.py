"""Transactional WakeCache causality tests for CRCT."""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.wake_cache_txn import (
    CausalEventClock,
    TransactionalWakeCache,
)


def _moment(cache: TransactionalWakeCache, surprise: float, *, txn=None) -> None:
    x = torch.tensor([[1, 2, 3]])
    h = torch.randn(1, 3, 4)
    cache.record_moment(
        surprise=surprise,
        inputs=x,
        targets=x,
        hidden=h,
        txn=txn,
    )


def test_begin_batch_snapshots_read_cutoff_and_hides_same_batch_writes():
    cache = TransactionalWakeCache(clock=CausalEventClock(current=3))
    txn = cache.begin_batch()

    _moment(cache, 1.0, txn=txn)
    assert cache.visible_moments(txn.read_cutoff) == []

    cache.commit(txn)
    assert len(cache.moments) == 1
    assert cache.moments[0]["_event_id"] == 4
    assert cache.visible_moments(txn.read_cutoff) == []
    assert cache.visible_moments(cache.clock.current) == cache.moments


def test_commit_is_single_use():
    cache = TransactionalWakeCache()
    txn = cache.begin_batch()
    cache.commit(txn)

    with pytest.raises(RuntimeError, match="already been committed"):
        cache.commit(txn)


def test_immediate_write_gets_new_event_id_and_respects_wakecache_eviction():
    cache = TransactionalWakeCache(max_moments=2)
    _moment(cache, 0.1)
    _moment(cache, 0.2)
    _moment(cache, 0.05)
    _moment(cache, 1.0)

    surprises = sorted(float(m["surprise"]) for m in cache.moments)
    assert surprises == [0.2, 1.0]
    by_surprise = {float(m["surprise"]): int(m["_event_id"]) for m in cache.moments}
    assert by_surprise == {0.2: 2, 1.0: 4}


def test_visible_hidden_uses_event_cutoff():
    cache = TransactionalWakeCache()
    h0 = torch.zeros(2, 3)
    cache.push_hidden(h0)
    txn = cache.begin_batch()
    cache.push_hidden(torch.ones(2, 3), txn=txn)
    cache.commit(txn)

    visible_old = cache.visible_hidden(txn.read_cutoff)
    visible_now = cache.visible_hidden(cache.clock.current)
    assert len(visible_old) == 1
    assert len(visible_now) == 2
