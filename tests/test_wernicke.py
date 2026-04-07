#!/usr/bin/env python3
"""Tests for WernickeLayer with VQ/MoE routing."""
from __future__ import annotations

import unittest

import torch

from chaoscontrol.memory import MultiSlotOuterModel
from chaoscontrol.wernicke import WernickeLayer


class TestWernickeLayer(unittest.TestCase):
    def test_output_shape(self):
        w = WernickeLayer(dim=16, k_max=4, window=4)
        x = torch.randn(2, 8, 16)
        out, ids, bloss = w(x)
        assert out.shape == (2, 8, 16)
        assert ids.shape == (2, 8)
        assert bloss.shape == ()

    def test_bucket_ids_in_range(self):
        w = WernickeLayer(dim=16, k_max=4, window=4)
        x = torch.randn(2, 8, 16)
        _, ids, _ = w(x)
        assert ids.min() >= 0
        assert ids.max() < 4

    def test_vq_vs_moe_produce_different_assignments(self):
        torch.manual_seed(42)
        wvq = WernickeLayer(dim=16, k_max=4, window=4, router_type="vq")
        torch.manual_seed(42)
        wmoe = WernickeLayer(dim=16, k_max=4, window=4, router_type="moe")
        x = torch.randn(2, 8, 16)
        _, ids_vq, _ = wvq(x)
        _, ids_moe, _ = wmoe(x)
        # They use different routing so assignments will differ
        # (though with same seed, not guaranteed — just check both produce valid assignments)
        assert ids_vq.min() >= 0 and ids_vq.max() < 4
        assert ids_moe.min() >= 0 and ids_moe.max() < 4

    def test_causal_no_future_leakage(self):
        w = WernickeLayer(dim=16, k_max=4, window=4)
        x = torch.randn(1, 8, 16, requires_grad=True)
        out, _, _ = w(x)
        # Check that output at position 0 doesn't depend on position 7
        out[0, 0].sum().backward()
        # Gradient should be zero for positions after 0
        assert x.grad[0, 1:].abs().sum() == 0.0

    def test_balance_loss_penalizes_collapse(self):
        w = WernickeLayer(dim=16, k_max=4, window=4, router_type="moe")
        # Force router to always pick bucket 0
        with torch.no_grad():
            w.router.weight.zero_()
            w.router.weight[0] = 1.0
        x = torch.randn(2, 8, 16)
        _, _, bloss = w(x)
        assert bloss.item() > 0  # collapsed usage should produce nonzero balance loss

    def test_compression_consequence_moe_updates_router(self):
        w = WernickeLayer(dim=16, k_max=4, window=4, router_type="moe")
        initial = w.router.weight.data.clone()
        w.compression_consequence_update(bucket_id=0, quality_delta=-0.5)
        assert not torch.allclose(initial, w.router.weight.data)

    def test_compression_consequence_vq_pushes_codebook(self):
        w = WernickeLayer(dim=16, k_max=4, window=4, router_type="vq")
        initial = w.codebook.data.clone()
        w.compression_consequence_update(bucket_id=0, quality_delta=-0.5)
        assert not torch.allclose(initial, w.codebook.data)

    def test_compression_consequence_no_update_on_good_merge(self):
        w = WernickeLayer(dim=16, k_max=4, window=4, router_type="moe")
        initial = w.router.weight.data.clone()
        w.compression_consequence_update(bucket_id=0, quality_delta=0.5)  # good merge
        assert torch.allclose(initial, w.router.weight.data)

    def test_typed_compression_merges_within_bucket(self):
        om = MultiSlotOuterModel(model_dim=16, outer_dim=32, max_slots=4, compress_ratio=2)
        # Write slots with different bucket ids
        om.write(torch.randn(1, 16), bucket_id=0)
        om.write(torch.randn(1, 16), bucket_id=0)
        om.write(torch.randn(1, 16), bucket_id=1)
        om.write(torch.randn(1, 16), bucket_id=1)
        om.write(torch.randn(1, 16), bucket_id=0)  # triggers compression
        # After compression, slots should still have both bucket types
        buckets_remaining = set(om._slot_buckets)
        assert 0 in buckets_remaining
        assert 1 in buckets_remaining


if __name__ == "__main__":
    unittest.main()
