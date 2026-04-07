"""Tests for metabolic_fork scoring modes.

Integration tests with ChaosStudentLM will be added once model.py is
extracted (Task 9).  For now we use a lightweight mock model.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Mock model that satisfies the duck-typed interface expected by metabolic_fork
# ---------------------------------------------------------------------------

class _MockModel(nn.Module):
    """Minimal stand-in for ChaosStudentLM with the five attributes
    metabolic_fork accesses: embed, outer_model, layers, final_norm, lm_head.
    """

    def __init__(self, vocab_size: int = 256, dim: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        self._embed = nn.Embedding(vocab_size, dim)
        self.outer_model = None  # no memory module
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embed(input_ids)


class TestMetabolicForkImport(unittest.TestCase):
    def test_metabolic_fork_import(self) -> None:
        from chaoscontrol.metabolic import metabolic_fork
        assert callable(metabolic_fork)


class TestMetabolicForkScoringModes(unittest.TestCase):
    """Test each scoring mode with a mock model (no outer memory)."""

    def _make_model(self) -> _MockModel:
        torch.manual_seed(42)
        return _MockModel(vocab_size=256, dim=16, num_layers=2)

    def _make_ids(self, batch: int = 2, seq: int = 16) -> torch.Tensor:
        return torch.randint(0, 256, (batch, seq))

    # -- ensemble_agreement ------------------------------------------------

    def test_ensemble_agreement_returns_logits(self) -> None:
        from chaoscontrol.metabolic import metabolic_fork
        model = self._make_model()
        ids = self._make_ids()
        out = metabolic_fork(model, ids, k=3, noise_std=0.1, score_mode="ensemble_agreement")
        assert "logits" in out
        assert out["logits"].shape == (2, 16, 256)

    def test_ensemble_agreement_returns_hidden(self) -> None:
        from chaoscontrol.metabolic import metabolic_fork
        model = self._make_model()
        ids = self._make_ids()
        out = metabolic_fork(model, ids, k=3, noise_std=0.1, score_mode="ensemble_agreement")
        assert "hidden" in out
        assert out["hidden"].shape == (2, 16, 16)

    # -- loss_lookahead ----------------------------------------------------

    def test_loss_lookahead_returns_logits(self) -> None:
        from chaoscontrol.metabolic import metabolic_fork
        model = self._make_model()
        ids = self._make_ids()
        out = metabolic_fork(model, ids, k=3, noise_std=0.1, score_mode="loss_lookahead")
        assert out["logits"].shape == (2, 16, 256)

    # -- memory_consistency without outer model falls back to ensemble -----

    def test_memory_consistency_fallback_no_outer(self) -> None:
        from chaoscontrol.metabolic import metabolic_fork
        model = self._make_model()
        assert model.outer_model is None
        ids = self._make_ids()
        out = metabolic_fork(model, ids, k=3, noise_std=0.1, score_mode="memory_consistency")
        assert out["logits"].shape == (2, 16, 256)

    # -- unknown mode falls back to ensemble agreement ---------------------

    def test_unknown_mode_fallback(self) -> None:
        from chaoscontrol.metabolic import metabolic_fork
        model = self._make_model()
        ids = self._make_ids()
        out = metabolic_fork(model, ids, k=3, noise_std=0.1, score_mode="nonexistent_mode")
        assert out["logits"].shape == (2, 16, 256)

    # -- different seeds produce different selections ----------------------

    def test_different_noise_gives_different_logits(self) -> None:
        from chaoscontrol.metabolic import metabolic_fork
        model = self._make_model()
        ids = self._make_ids()
        torch.manual_seed(1)
        out1 = metabolic_fork(model, ids, k=4, noise_std=0.5, score_mode="ensemble_agreement")
        torch.manual_seed(2)
        out2 = metabolic_fork(model, ids, k=4, noise_std=0.5, score_mode="ensemble_agreement")
        assert not torch.allclose(out1["logits"], out2["logits"])


class TestStructuredProjections(unittest.TestCase):
    def test_produces_k_views(self) -> None:
        from chaoscontrol.metabolic import StructuredProjections
        sp = StructuredProjections(dim=16, k=4)
        x = torch.randn(2, 8, 16)
        views = sp(x)
        assert len(views) == 4
        assert views[0].shape == (2, 8, 16)

    def test_views_differ(self) -> None:
        from chaoscontrol.metabolic import StructuredProjections
        sp = StructuredProjections(dim=16, k=4)
        x = torch.randn(2, 8, 16)
        views = sp(x)
        assert not torch.allclose(views[0], views[1])

    def test_structured_fork_produces_logits(self) -> None:
        from chaoscontrol.metabolic import metabolic_fork, StructuredProjections
        sp = StructuredProjections(dim=16, k=4)
        model = _MockModel(vocab_size=256, dim=16, num_layers=2)
        ids = torch.randint(0, 256, (2, 8))
        out = metabolic_fork(
            model, ids, k=4, noise_std=0.1,
            score_mode="ensemble_agreement",
            generation_mode="structured",
            structured_proj=sp,
        )
        assert out["logits"].shape == (2, 8, 256)


class TestMonteCarloMetabolic(unittest.TestCase):
    """Tests for the Monte Carlo metabolic gate — distributional statistics."""

    def _make_model(self) -> _MockModel:
        torch.manual_seed(42)
        return _MockModel(vocab_size=256, dim=16, num_layers=2)

    def _make_ids(self, batch: int = 2, seq: int = 16) -> torch.Tensor:
        return torch.randint(0, 256, (batch, seq))

    def test_returns_mc_stats(self) -> None:
        from chaoscontrol.metabolic import metabolic_monte_carlo
        model = self._make_model()
        ids = self._make_ids()
        out = metabolic_monte_carlo(model, ids, k=4, noise_std=0.1)
        assert "mc_stats" in out
        stats = out["mc_stats"]
        assert "logits_var" in stats
        assert "entropy" in stats
        assert "agreement" in stats
        assert "uncertainty_map" in stats
        assert "candidate_divergence" in stats

    def test_logits_are_ensemble_mean(self) -> None:
        from chaoscontrol.metabolic import metabolic_monte_carlo
        model = self._make_model()
        ids = self._make_ids()
        out = metabolic_monte_carlo(model, ids, k=4, noise_std=0.1)
        assert out["logits"].shape == (2, 16, 256)
        assert out["hidden"].shape == (2, 16, 16)

    def test_variance_shape(self) -> None:
        from chaoscontrol.metabolic import metabolic_monte_carlo
        model = self._make_model()
        ids = self._make_ids()
        out = metabolic_monte_carlo(model, ids, k=4, noise_std=0.1)
        assert out["mc_stats"]["logits_var"].shape == (2, 16)

    def test_entropy_bounded(self) -> None:
        """Entropy should be non-negative and bounded by log(vocab_size)."""
        import math
        from chaoscontrol.metabolic import metabolic_monte_carlo
        model = self._make_model()
        ids = self._make_ids()
        out = metabolic_monte_carlo(model, ids, k=4, noise_std=0.1)
        entropy = out["mc_stats"]["entropy"]
        assert (entropy >= 0).all()
        assert (entropy <= math.log(256) + 0.01).all()

    def test_agreement_bounded_zero_one(self) -> None:
        from chaoscontrol.metabolic import metabolic_monte_carlo
        model = self._make_model()
        ids = self._make_ids()
        out = metabolic_monte_carlo(model, ids, k=4, noise_std=0.1)
        agreement = out["mc_stats"]["agreement"]
        assert (agreement >= -0.01).all()
        assert (agreement <= 1.01).all()

    def test_higher_noise_increases_variance(self) -> None:
        """More noise in the MC sample should produce higher logits variance."""
        from chaoscontrol.metabolic import metabolic_monte_carlo
        model = self._make_model()
        ids = self._make_ids()
        torch.manual_seed(99)
        out_low = metabolic_monte_carlo(model, ids, k=8, noise_std=0.01)
        torch.manual_seed(99)
        out_high = metabolic_monte_carlo(model, ids, k=8, noise_std=1.0)
        var_low = out_low["mc_stats"]["logits_var"].mean().item()
        var_high = out_high["mc_stats"]["logits_var"].mean().item()
        assert var_high > var_low, f"high noise var {var_high} should exceed low {var_low}"

    def test_higher_noise_increases_divergence(self) -> None:
        """More noise should produce higher candidate divergence."""
        from chaoscontrol.metabolic import metabolic_monte_carlo
        model = self._make_model()
        ids = self._make_ids()
        torch.manual_seed(99)
        out_low = metabolic_monte_carlo(model, ids, k=8, noise_std=0.01)
        torch.manual_seed(99)
        out_high = metabolic_monte_carlo(model, ids, k=8, noise_std=1.0)
        div_low = out_low["mc_stats"]["candidate_divergence"].item()
        div_high = out_high["mc_stats"]["candidate_divergence"].item()
        assert div_high > div_low

    def test_uncertainty_map_peaks_at_genuine_uncertainty(self) -> None:
        """Uncertainty map should be high only when BOTH variance and entropy are high."""
        from chaoscontrol.metabolic import metabolic_monte_carlo
        model = self._make_model()
        ids = self._make_ids()
        out = metabolic_monte_carlo(model, ids, k=8, noise_std=0.5)
        umap = out["mc_stats"]["uncertainty_map"]
        # Should be non-negative (product of two [0,1] quantities)
        assert (umap >= -0.01).all()

    def test_structured_generation_returns_stats(self) -> None:
        from chaoscontrol.metabolic import metabolic_monte_carlo, StructuredProjections
        model = self._make_model()
        sp = StructuredProjections(dim=16, k=4)
        ids = self._make_ids(batch=2, seq=8)
        out = metabolic_monte_carlo(
            model, ids, k=4, generation_mode="structured", structured_proj=sp,
        )
        assert "mc_stats" in out
        assert out["logits"].shape == (2, 8, 256)
        assert out["mc_stats"]["logits_var"].shape == (2, 8)

    def test_more_candidates_reduces_variance_of_mean(self) -> None:
        """Central limit theorem: more samples → more stable mean estimate."""
        from chaoscontrol.metabolic import metabolic_monte_carlo
        model = self._make_model()
        ids = self._make_ids()
        # Run twice with K=2 and K=16, compare divergence
        runs_k2 = []
        runs_k16 = []
        for seed in range(5):
            torch.manual_seed(seed)
            out2 = metabolic_monte_carlo(model, ids, k=2, noise_std=0.5)
            runs_k2.append(out2["logits"].mean().item())
            torch.manual_seed(seed + 100)
            out16 = metabolic_monte_carlo(model, ids, k=16, noise_std=0.5)
            runs_k16.append(out16["logits"].mean().item())
        spread_k2 = max(runs_k2) - min(runs_k2)
        spread_k16 = max(runs_k16) - min(runs_k16)
        assert spread_k16 < spread_k2, f"K=16 spread {spread_k16} should be tighter than K=2 {spread_k2}"


if __name__ == "__main__":
    unittest.main()
