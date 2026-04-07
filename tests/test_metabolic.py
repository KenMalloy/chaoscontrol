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


if __name__ == "__main__":
    unittest.main()
