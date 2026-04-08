"""Tests for ChaosStudentLM.dream_step() — full-tier single-token forward."""
from __future__ import annotations

import unittest

import torch

from chaoscontrol.model import ChaosStudentLM


class TestDreamStepBasic(unittest.TestCase):
    """dream_step exists and returns correct shapes."""

    def test_returns_correct_shapes(self):
        torch.manual_seed(42)
        model = ChaosStudentLM(vocab_size=256, dim=16, num_layers=2)
        token_ids = torch.tensor([[42], [7]])  # (batch=2, seq=1)
        state = model.init_state(batch_size=2)
        logits, hidden, new_state = model.dream_step(token_ids, state)
        assert logits.shape == (2, 256), f"logits shape {logits.shape}"
        assert hidden.shape == (2, 16), f"hidden shape {hidden.shape}"
        assert len(new_state) == 2, "one state per layer"
        for s in new_state:
            assert s.shape == (2, 16), f"state shape {s.shape}"

    def test_matches_step_on_bare_model(self):
        """Without Wernicke/memory/semantic, dream_step equals step."""
        torch.manual_seed(42)
        model = ChaosStudentLM(vocab_size=256, dim=16, num_layers=2)
        token_ids = torch.tensor([[42], [7]])
        state = model.init_state(batch_size=2)
        logits_step, hidden_step, state_step = model.step(token_ids, state)

        state2 = model.init_state(batch_size=2)
        logits_dream, hidden_dream, state_dream = model.dream_step(token_ids, state2)

        assert torch.allclose(logits_step, logits_dream, atol=1e-5), \
            f"max diff: {(logits_step - logits_dream).abs().max()}"
        assert torch.allclose(hidden_step, hidden_dream, atol=1e-5)

    def test_diverges_from_step_with_features(self):
        """dream_step SHOULD differ from step when features are active."""
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2,
            wernicke_enabled=True, wernicke_k_max=4, wernicke_window=4,
            outer_model_dim=8, outer_model_type="multislot",
            semantic_tier_bases=4,
        )
        # Write a slot so memory read is non-trivial (write takes model_dim)
        h = torch.randn(1, 16)
        model.outer_model.write(h)

        # Consolidate semantic tier so it's non-zero
        model.semantic_tier.consolidate_from_episodes(torch.randn(4, 16))

        token_ids = torch.tensor([[42], [7]])
        state = model.init_state(batch_size=2)
        logits_step, _, _ = model.step(token_ids, state)

        state2 = model.init_state(batch_size=2)
        logits_dream, _, _ = model.dream_step(token_ids, state2)

        assert not torch.allclose(logits_step, logits_dream, atol=0.01), \
            "dream_step should diverge from step when features are active"


class TestDreamStepWithOuterModel(unittest.TestCase):
    """dream_step works with outer_model (write a slot first)."""

    def test_with_single_outer_model(self):
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2,
            outer_model_dim=8, outer_model_type="single",
        )
        # Write to the outer model so the read is non-zero
        h = torch.randn(2, 16)
        model.outer_model.write(h)

        token_ids = torch.tensor([[42], [7]])
        state = model.init_state(batch_size=2)
        logits, hidden, new_state = model.dream_step(token_ids, state)
        assert logits.shape == (2, 256)
        assert hidden.shape == (2, 16)

    def test_with_multislot_outer_model(self):
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2,
            outer_model_dim=8, outer_model_type="multislot",
        )
        # Write a slot (write takes model_dim, not outer_dim)
        h = torch.randn(1, 16)
        model.outer_model.write(h)

        token_ids = torch.tensor([[42], [7]])
        state = model.init_state(batch_size=2)
        logits, hidden, new_state = model.dream_step(token_ids, state)
        assert logits.shape == (2, 256)
        assert hidden.shape == (2, 16)

    def test_multislot_cue_dependent_read(self):
        """Verify dream_step uses cue-dependent retrieval on multislot."""
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2,
            outer_model_dim=8, outer_model_type="multislot",
        )
        # Write two distinct slots (write takes model_dim, not outer_dim)
        model.outer_model.write(torch.randn(1, 16))
        model.outer_model.write(torch.randn(1, 16))

        # Different tokens should produce different logits (cue differs)
        state = model.init_state(batch_size=1)
        logits_a, _, _ = model.dream_step(torch.tensor([[10]]), state)

        state = model.init_state(batch_size=1)
        logits_b, _, _ = model.dream_step(torch.tensor([[200]]), state)

        # Logits should differ (different cue -> different memory read)
        assert not torch.allclose(logits_a, logits_b, atol=1e-3), \
            "different tokens should produce different dream_step logits"


class TestDreamStepWithWernicke(unittest.TestCase):
    """dream_step works with Wernicke enabled."""

    def test_wernicke_processes_input(self):
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2,
            wernicke_enabled=True, wernicke_k_max=4, wernicke_window=4,
        )
        token_ids = torch.tensor([[42], [7]])
        state = model.init_state(batch_size=2)
        logits, hidden, new_state = model.dream_step(token_ids, state)
        assert logits.shape == (2, 256)
        assert hidden.shape == (2, 16)

    def test_wernicke_with_all_tiers(self):
        """Full config: Wernicke + memory + semantic."""
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2,
            wernicke_enabled=True, wernicke_k_max=4, wernicke_window=4,
            outer_model_dim=8, outer_model_type="multislot",
            semantic_tier_bases=4,
        )
        token_ids = torch.tensor([[42], [7]])
        state = model.init_state(batch_size=2)
        logits, hidden, new_state = model.dream_step(token_ids, state)
        assert logits.shape == (2, 256)
        assert hidden.shape == (2, 16)
        assert len(new_state) == 2

    def test_sequential_dream_steps(self):
        """Multiple sequential dream_steps accumulate state correctly."""
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2,
            wernicke_enabled=True, wernicke_k_max=4, wernicke_window=4,
            outer_model_dim=8, outer_model_type="single",
        )
        model.outer_model.write(torch.randn(1, 16))

        state = model.init_state(batch_size=1)
        tokens = [10, 20, 30, 40]
        for t in tokens:
            logits, hidden, state = model.dream_step(torch.tensor([[t]]), state)

        # After several steps, state should be non-zero
        assert logits.shape == (1, 256)
        for s in state:
            assert s.abs().sum() > 0, "state should be non-zero after several steps"


if __name__ == "__main__":
    unittest.main()
