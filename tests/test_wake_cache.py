#!/usr/bin/env python3
"""Tests for WakeCache — high-signal moment buffer for sleep consolidation."""
from __future__ import annotations

import unittest

import torch

from chaoscontrol.wake_cache import WakeCache


class TestWakeCacheRecordAndRetrieve(unittest.TestCase):
    """record_moment stores moments that can be read back."""

    def test_single_moment_stored(self) -> None:
        wc = WakeCache(max_moments=4)
        wc.record_moment(
            surprise=3.0,
            inputs=torch.randn(2, 8),
            targets=torch.randn(2, 8),
            hidden=torch.randn(2, 16),
        )
        assert len(wc.moments) == 1
        assert wc.moments[0]["surprise"] == 3.0

    def test_optional_fields_stored(self) -> None:
        wc = WakeCache(max_moments=4)
        wc.record_moment(
            surprise=1.0,
            inputs=torch.randn(2, 8),
            targets=torch.randn(2, 8),
            hidden=torch.randn(2, 16),
            bucket_ids=torch.tensor([0, 1, 2]),
            slot_cues=torch.tensor([0.5, 0.5]),
        )
        assert "bucket_ids" in wc.moments[0]
        assert "slot_cues" in wc.moments[0]

    def test_tensors_on_cpu(self) -> None:
        wc = WakeCache(max_moments=4)
        wc.record_moment(
            surprise=2.0,
            inputs=torch.randn(2, 8),
            targets=torch.randn(2, 8),
            hidden=torch.randn(2, 16),
        )
        for key in ("inputs", "targets", "hidden"):
            assert wc.moments[0][key].device == torch.device("cpu")


class TestWakeCacheEviction(unittest.TestCase):
    """When at capacity, the lowest abs(surprise) moment is evicted."""

    def test_eviction_keeps_highest_signal(self) -> None:
        wc = WakeCache(max_moments=3)
        for s in [1.0, 5.0, 3.0]:
            wc.record_moment(
                surprise=s,
                inputs=torch.zeros(1),
                targets=torch.zeros(1),
                hidden=torch.zeros(1),
            )
        assert len(wc.moments) == 3

        # Adding surprise=4.0 should evict the moment with surprise=1.0.
        wc.record_moment(
            surprise=4.0,
            inputs=torch.zeros(1),
            targets=torch.zeros(1),
            hidden=torch.zeros(1),
        )
        assert len(wc.moments) == 3
        surprises = sorted(m["surprise"] for m in wc.moments)
        assert surprises == [3.0, 4.0, 5.0]

    def test_negative_surprise_uses_abs(self) -> None:
        wc = WakeCache(max_moments=2)
        wc.record_moment(
            surprise=-10.0,
            inputs=torch.zeros(1),
            targets=torch.zeros(1),
            hidden=torch.zeros(1),
        )
        wc.record_moment(
            surprise=0.1,
            inputs=torch.zeros(1),
            targets=torch.zeros(1),
            hidden=torch.zeros(1),
        )
        # Cache is full. Adding surprise=0.5 should evict surprise=0.1
        # (abs 0.1 < abs 0.5 < abs -10.0).
        wc.record_moment(
            surprise=0.5,
            inputs=torch.zeros(1),
            targets=torch.zeros(1),
            hidden=torch.zeros(1),
        )
        assert len(wc.moments) == 2
        surprises = sorted(abs(m["surprise"]) for m in wc.moments)
        assert surprises == [0.5, 10.0]

    def test_weak_moment_not_inserted(self) -> None:
        wc = WakeCache(max_moments=2)
        wc.record_moment(
            surprise=5.0,
            inputs=torch.zeros(1),
            targets=torch.zeros(1),
            hidden=torch.zeros(1),
        )
        wc.record_moment(
            surprise=3.0,
            inputs=torch.zeros(1),
            targets=torch.zeros(1),
            hidden=torch.zeros(1),
        )
        # Inserting surprise=2.0 should not evict anything (weaker than min=3.0).
        wc.record_moment(
            surprise=2.0,
            inputs=torch.zeros(1),
            targets=torch.zeros(1),
            hidden=torch.zeros(1),
        )
        surprises = sorted(m["surprise"] for m in wc.moments)
        assert surprises == [3.0, 5.0]


class TestBucketDistribution(unittest.TestCase):
    """bucket_distribution returns normalised frequencies."""

    def test_uniform_when_no_data(self) -> None:
        wc = WakeCache()
        dist = wc.bucket_distribution(4)
        assert dist.shape == (4,)
        assert torch.allclose(dist, torch.tensor([0.25, 0.25, 0.25, 0.25]))

    def test_normalised_after_update(self) -> None:
        wc = WakeCache()
        wc.update_bucket_counts(torch.tensor([0, 0, 1, 1, 1, 2]))
        dist = wc.bucket_distribution(4)
        assert dist.shape == (4,)
        assert torch.isclose(dist.sum(), torch.tensor(1.0))
        # bucket 0: 2/6, bucket 1: 3/6, bucket 2: 1/6, bucket 3: 0/6
        expected = torch.tensor([2 / 6, 3 / 6, 1 / 6, 0.0])
        assert torch.allclose(dist, expected)

    def test_accumulates_across_calls(self) -> None:
        wc = WakeCache()
        wc.update_bucket_counts(torch.tensor([0, 0]))
        wc.update_bucket_counts(torch.tensor([1, 1, 1]))
        dist = wc.bucket_distribution(2)
        expected = torch.tensor([2 / 5, 3 / 5])
        assert torch.allclose(dist, expected)


class TestHiddenBuffer(unittest.TestCase):
    """push_hidden respects the rolling deque capacity."""

    def test_rolling_cap(self) -> None:
        wc = WakeCache(max_hidden_buffer=3)
        for i in range(5):
            wc.push_hidden(torch.tensor([float(i)]))
        assert len(wc.hidden_buffer) == 3
        # Oldest entries (0, 1) should have been evicted.
        values = [h.item() for h in wc.hidden_buffer]
        assert values == [2.0, 3.0, 4.0]

    def test_tensors_on_cpu(self) -> None:
        wc = WakeCache(max_hidden_buffer=4)
        wc.push_hidden(torch.randn(2, 8))
        assert wc.hidden_buffer[0].device == torch.device("cpu")


class TestClear(unittest.TestCase):
    """clear resets all internal state."""

    def test_clear_resets_everything(self) -> None:
        wc = WakeCache(max_moments=4, max_hidden_buffer=4)
        wc.record_moment(
            surprise=1.0,
            inputs=torch.zeros(1),
            targets=torch.zeros(1),
            hidden=torch.zeros(1),
        )
        wc.push_hidden(torch.zeros(1))
        wc.update_bucket_counts(torch.tensor([0, 1]))
        wc.clear()

        assert len(wc.moments) == 0
        assert len(wc.hidden_buffer) == 0
        assert wc._bucket_counts is None
        # After clear, bucket_distribution should return uniform again.
        dist = wc.bucket_distribution(3)
        assert torch.allclose(dist, torch.ones(3) / 3)


if __name__ == "__main__":
    unittest.main()
