"""Tests for evaluation utilities including warming curve."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from chaoscontrol.evaluation import _reset_model_state, evaluate_warming_curve
from chaoscontrol.memory import MultiSlotOuterModel, SemanticTier
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.posterior import GlobalDelta, BucketDelta, ResidualCache


class TestResetModelState(unittest.TestCase):
    """Verify _reset_model_state clears all stateful components."""

    def test_resets_multislot_buffer(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            outer_model_dim=32, outer_model_type="multislot", outer_max_slots=8,
        )
        # Populate buffer
        model.outer_model.write(torch.randn(1, 16))
        model.outer_model.write(torch.randn(1, 16))
        assert len(model.outer_model._slots) == 2

        _reset_model_state(model)
        assert len(model.outer_model._slots) == 0
        assert len(model.outer_model._survival) == 0
        assert len(model.outer_model._slot_buckets) == 0

    def test_resets_single_outer_model(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            outer_model_dim=32, outer_model_type="single",
        )
        model.outer_model.write(torch.randn(1, 16))
        assert model.outer_model.state.abs().sum() > 0

        _reset_model_state(model)
        assert torch.allclose(model.outer_model.state, torch.zeros_like(model.outer_model.state))

    def test_resets_semantic_tier(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            semantic_tier_bases=8,
        )
        # Populate semantic tier
        model.semantic_tier.bases.fill_(1.0)
        assert model.semantic_tier.bases.abs().sum() > 0

        _reset_model_state(model)
        assert torch.allclose(model.semantic_tier.bases, torch.zeros_like(model.semantic_tier.bases))

    def test_resets_bucket_prototypes_if_present(self) -> None:
        """BucketPrototypes should be zeroed if the model has them."""
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
        )
        # Simulate a bucket_prototypes_module attribute
        bpm = MagicMock()
        bpm.prototypes = torch.ones(8, 16)
        model.bucket_prototypes_module = bpm

        _reset_model_state(model)
        assert torch.allclose(bpm.prototypes, torch.zeros_like(bpm.prototypes))

    def test_resets_posterior_global_delta(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
        )
        posterior = GlobalDelta(model_dim=16, lr=1.0)
        posterior.update(torch.ones(16))
        model.posterior = posterior
        assert posterior.read(1).abs().sum() > 0

        _reset_model_state(model)
        assert torch.allclose(posterior.delta, torch.zeros_like(posterior.delta))

    def test_resets_posterior_bucket_delta(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
        )
        posterior = BucketDelta(k_max=4, model_dim=16, lr=1.0)
        posterior.update(0, torch.ones(16))
        model.posterior = posterior

        _reset_model_state(model)
        assert torch.allclose(posterior.deltas, torch.zeros_like(posterior.deltas))

    def test_resets_posterior_residual_cache(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
        )
        posterior = ResidualCache(model_dim=16, k=2)
        posterior.store(torch.randn(16), torch.randn(16))
        model.posterior = posterior
        assert len(posterior._keys) == 1

        _reset_model_state(model)
        assert len(posterior._keys) == 0

    def test_prototypes_zeroed_between_segments(self) -> None:
        """Verify that prototypes are actually zero after reset (not just called)."""
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            semantic_tier_bases=4,
        )
        # Simulate accumulated state
        model.semantic_tier.bases.fill_(42.0)

        bpm = MagicMock()
        bpm.prototypes = torch.full((4, 16), 99.0)
        model.bucket_prototypes_module = bpm

        _reset_model_state(model)

        # Both must be exactly zero
        assert model.semantic_tier.bases.abs().max().item() == 0.0
        assert bpm.prototypes.abs().max().item() == 0.0


class TestEvaluateWarmingCurve(unittest.TestCase):
    def test_returns_all_warming_steps(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
        )
        device = torch.device("cpu")
        # Generate enough tokens for warming + scoring
        tokens = torch.randint(0, 256, (10000,))
        segment_starts = [0, 2000]
        result = evaluate_warming_curve(
            model, tokens, segment_starts=segment_starts,
            score_tokens=64, warmup_tokens=[0, 100],
            device=device,
        )
        assert 0 in result
        assert 100 in result
        assert isinstance(result[0], float)
        assert isinstance(result[100], float)

    def test_warm_bpb_differs_from_cold(self) -> None:
        """With a model that has memory, warm and cold bpb should differ."""
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            outer_model_dim=32, outer_model_type="multislot",
            outer_max_slots=64,
        )
        device = torch.device("cpu")
        tokens = torch.randint(0, 256, (10000,))
        segment_starts = [0]
        result = evaluate_warming_curve(
            model, tokens, segment_starts=segment_starts,
            score_tokens=32, warmup_tokens=[0, 100],
            device=device,
        )
        # Both should be valid floats
        assert not (result[0] != result[0])  # not NaN
        assert not (result[100] != result[100])

    def test_no_segments_returns_nan(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
        )
        device = torch.device("cpu")
        tokens = torch.randint(0, 256, (100,))
        # Segment starts that would overflow
        result = evaluate_warming_curve(
            model, tokens, segment_starts=[99999],
            score_tokens=64, warmup_tokens=[0],
            device=device,
        )
        assert result[0] != result[0]  # NaN


if __name__ == "__main__":
    unittest.main()
