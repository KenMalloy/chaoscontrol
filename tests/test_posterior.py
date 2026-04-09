"""Tests for Phase D posterior-state modules."""
from __future__ import annotations

import unittest

import torch

from chaoscontrol.posterior import GlobalDelta, BucketDelta, ResidualCache


class TestGlobalDelta(unittest.TestCase):
    def test_init_zeros(self) -> None:
        gd = GlobalDelta(model_dim=16)
        read = gd.read(batch_size=2)
        assert read.shape == (2, 16)
        assert torch.allclose(read, torch.zeros_like(read))

    def test_update_changes_delta(self) -> None:
        gd = GlobalDelta(model_dim=16, lr=1.0)
        grad = torch.ones(16)
        gd.update(grad)
        read = gd.read(batch_size=1)
        assert read.abs().sum() > 0

    def test_update_accumulates(self) -> None:
        gd = GlobalDelta(model_dim=16, lr=1.0)
        grad = torch.ones(16)
        gd.update(grad)
        read1 = gd.read(batch_size=1).clone()
        gd.update(grad)
        read2 = gd.read(batch_size=1)
        # Two updates should give 2x the first
        assert torch.allclose(read2, 2 * read1)

    def test_update_with_2d_grad(self) -> None:
        gd = GlobalDelta(model_dim=16, lr=1.0)
        grad = torch.ones(1, 16)
        gd.update(grad)
        read = gd.read(batch_size=1)
        assert read.abs().sum() > 0

    def test_lr_scaling(self) -> None:
        gd = GlobalDelta(model_dim=16, lr=0.1)
        grad = torch.ones(16) * 10.0
        gd.update(grad)
        read = gd.read(batch_size=1)
        # lr=0.1 * grad=10 = delta of 1.0 per dim
        assert torch.allclose(read, torch.ones(1, 16))

    def test_reset_clears_state(self) -> None:
        gd = GlobalDelta(model_dim=16, lr=1.0)
        gd.update(torch.randn(16))
        assert gd.read(batch_size=1).abs().sum() > 0
        gd.reset()
        read = gd.read(batch_size=1)
        assert torch.allclose(read, torch.zeros_like(read))

    def test_batch_expansion(self) -> None:
        gd = GlobalDelta(model_dim=8, lr=1.0)
        gd.update(torch.ones(8))
        read = gd.read(batch_size=4)
        assert read.shape == (4, 8)
        # All batch entries should be identical
        assert torch.allclose(read[0], read[1])
        assert torch.allclose(read[0], read[3])


class TestBucketDelta(unittest.TestCase):
    def test_init_zeros(self) -> None:
        bd = BucketDelta(k_max=8, model_dim=16)
        for b in range(8):
            read = bd.read(bucket_id=b, batch_size=2)
            assert read.shape == (2, 16)
            assert torch.allclose(read, torch.zeros_like(read))

    def test_update_affects_only_target_bucket(self) -> None:
        bd = BucketDelta(k_max=8, model_dim=16, lr=1.0)
        grad = torch.ones(16)
        bd.update(bucket_id=3, prediction_error_grad=grad)
        # Bucket 3 should be updated
        assert bd.read(bucket_id=3, batch_size=1).abs().sum() > 0
        # Other buckets should remain zero
        assert torch.allclose(bd.read(bucket_id=0, batch_size=1), torch.zeros(1, 16))
        assert torch.allclose(bd.read(bucket_id=7, batch_size=1), torch.zeros(1, 16))

    def test_update_accumulates(self) -> None:
        bd = BucketDelta(k_max=4, model_dim=8, lr=1.0)
        grad = torch.ones(8)
        bd.update(bucket_id=1, prediction_error_grad=grad)
        read1 = bd.read(bucket_id=1, batch_size=1).clone()
        bd.update(bucket_id=1, prediction_error_grad=grad)
        read2 = bd.read(bucket_id=1, batch_size=1)
        assert torch.allclose(read2, 2 * read1)

    def test_update_with_2d_grad(self) -> None:
        bd = BucketDelta(k_max=4, model_dim=8, lr=1.0)
        grad = torch.ones(1, 8)
        bd.update(bucket_id=0, prediction_error_grad=grad)
        assert bd.read(bucket_id=0, batch_size=1).abs().sum() > 0

    def test_reset_clears_all_buckets(self) -> None:
        bd = BucketDelta(k_max=4, model_dim=8, lr=1.0)
        for b in range(4):
            bd.update(bucket_id=b, prediction_error_grad=torch.randn(8))
        bd.reset()
        for b in range(4):
            assert torch.allclose(bd.read(bucket_id=b, batch_size=1), torch.zeros(1, 8))

    def test_batch_expansion(self) -> None:
        bd = BucketDelta(k_max=4, model_dim=8, lr=1.0)
        bd.update(bucket_id=2, prediction_error_grad=torch.ones(8))
        read = bd.read(bucket_id=2, batch_size=3)
        assert read.shape == (3, 8)
        assert torch.allclose(read[0], read[2])


class TestResidualCache(unittest.TestCase):
    def test_read_empty_returns_zeros(self) -> None:
        rc = ResidualCache(model_dim=16, k=4)
        query = torch.randn(2, 16)
        read = rc.read(query)
        assert read.shape == (2, 16)
        assert torch.allclose(read, torch.zeros_like(read))

    def test_store_and_read(self) -> None:
        rc = ResidualCache(model_dim=16, k=4)
        key = torch.randn(16)
        val = torch.ones(16) * 5.0
        rc.store(key, val)
        # Query with the same key should retrieve a non-zero result
        read = rc.read(key.unsqueeze(0))
        assert read.shape == (1, 16)
        assert read.abs().sum() > 0

    def test_similar_query_retrieves_matching_correction(self) -> None:
        rc = ResidualCache(model_dim=16, k=2)
        # Store two contrasting corrections
        key_a = torch.randn(16)
        val_a = torch.ones(16) * 10.0
        key_b = -key_a  # opposite direction
        val_b = torch.ones(16) * -10.0
        rc.store(key_a, val_a)
        rc.store(key_b, val_b)
        # Query similar to key_a should lean toward val_a
        read_a = rc.read(key_a.unsqueeze(0))
        read_b = rc.read(key_b.unsqueeze(0))
        assert not torch.allclose(read_a, read_b, atol=1e-3)

    def test_store_with_2d_input(self) -> None:
        rc = ResidualCache(model_dim=8, k=2)
        rc.store(torch.randn(1, 8), torch.randn(1, 8))
        assert len(rc._keys) == 1

    def test_max_entries_eviction(self) -> None:
        rc = ResidualCache(model_dim=8, k=2, max_entries=5)
        for _ in range(10):
            rc.store(torch.randn(8), torch.randn(8))
        assert len(rc._keys) == 5
        assert len(rc._values) == 5

    def test_reset_clears_cache(self) -> None:
        rc = ResidualCache(model_dim=8, k=2)
        rc.store(torch.randn(8), torch.randn(8))
        rc.store(torch.randn(8), torch.randn(8))
        assert len(rc._keys) == 2
        rc.reset()
        assert len(rc._keys) == 0
        assert len(rc._values) == 0
        # Read after reset should return zeros
        read = rc.read(torch.randn(1, 8))
        assert torch.allclose(read, torch.zeros_like(read))

    def test_topk_smaller_than_cache(self) -> None:
        rc = ResidualCache(model_dim=8, k=2, max_entries=100)
        # Store many entries
        for _ in range(20):
            rc.store(torch.randn(8), torch.randn(8))
        # Should still work with k=2
        read = rc.read(torch.randn(3, 8))
        assert read.shape == (3, 8)

    def test_topk_larger_than_cache(self) -> None:
        rc = ResidualCache(model_dim=8, k=10)
        # Only 2 entries but k=10
        rc.store(torch.randn(8), torch.randn(8))
        rc.store(torch.randn(8), torch.randn(8))
        read = rc.read(torch.randn(1, 8))
        assert read.shape == (1, 8)

    def test_batch_query(self) -> None:
        rc = ResidualCache(model_dim=8, k=2)
        rc.store(torch.randn(8), torch.ones(8))
        read = rc.read(torch.randn(4, 8))
        assert read.shape == (4, 8)


if __name__ == "__main__":
    unittest.main()
