"""Tests for ChaosSSMBlock and ChaosStudentLM."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from chaoscontrol.model import ChaosSSMBlock, ChaosStudentLM


class TestChaosSSMBlock(unittest.TestCase):
    def test_basic_forward(self) -> None:
        block = ChaosSSMBlock(16, ff_mult=2, a_mode="diag", rich_b_mode="none")
        x = torch.randn(2, 8, 16)
        out = block(x)
        assert out.shape == (2, 8, 16)

    def test_with_rich_b_nn(self) -> None:
        block = ChaosSSMBlock(16, ff_mult=2, a_mode="diag", rich_b_mode="nn", rich_b_bottleneck=8)
        x = torch.randn(2, 8, 16)
        out = block(x)
        assert out.shape == (2, 8, 16)

    def test_jacobian_stats(self) -> None:
        block = ChaosSSMBlock(16, ff_mult=2, a_mode="full", a_full_rank=4, rich_b_mode="none")
        x = torch.randn(2, 8, 16)
        out, stats = block(x, return_jacobian_stats=True)
        assert out.shape == (2, 8, 16)
        assert "lambda_max" in stats


class TestChaosStudentLM(unittest.TestCase):
    def test_base_forward_produces_logits(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)

    def test_full_config_forward(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="full", a_full_rank=4,
            rich_b_mode="assembly", rich_b_bottleneck=8, rich_b_num_subnets=4,
            outer_model_dim=32,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)

    def test_gradients_flow(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="paired", rich_b_mode="nn", rich_b_bottleneck=8,
            outer_model_dim=0,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        loss = F.cross_entropy(out["logits"].reshape(-1, 256), ids.reshape(-1))
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_artifact_bytes_under_budget(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=128, num_layers=4, ff_mult=2,
            a_mode="full", a_full_rank=8,
            rich_b_mode="hybrid", rich_b_bottleneck=32, rich_b_num_subnets=4,
            outer_model_dim=64,
        )
        assert model.artifact_bytes() < 16 * 1024 * 1024

    def test_with_outer_model_reads(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=32,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)

    def test_jacobian_stats_with_full(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="full", a_full_rank=4, rich_b_mode="none", outer_model_dim=0,
        )
        ids = torch.randint(0, 256, (2, 8))
        out = model(ids, return_jacobian_stats=True)
        assert "jacobian_stats" in out


class TestChaosStudentLMStep(unittest.TestCase):
    """Test single-token stepping on the full model."""

    def test_step_returns_logits_and_state(self):
        torch.manual_seed(42)
        model = ChaosStudentLM(vocab_size=256, dim=16, num_layers=2)
        token_ids = torch.tensor([[42], [7]])  # (batch=2, seq=1)
        state = model.init_state(batch_size=2)
        logits, hidden, new_state = model.step(token_ids, state)
        assert logits.shape == (2, 256)  # (batch, vocab)
        assert hidden.shape == (2, 16)  # (batch, dim)
        assert len(new_state) == 2  # one state per layer

    def test_step_sequence_matches_forward(self):
        """Stepping token-by-token should match forward() logits."""
        torch.manual_seed(42)
        model = ChaosStudentLM(vocab_size=256, dim=16, num_layers=2)
        ids = torch.randint(0, 256, (2, 8))

        # Full forward
        out_full = model(ids)
        logits_full = out_full["logits"]

        # Step-by-step
        state = model.init_state(batch_size=2)
        step_logits = []
        for t in range(8):
            logits_t, hidden_t, state = model.step(ids[:, t:t+1], state)
            step_logits.append(logits_t)
        logits_step = torch.stack(step_logits, dim=1)

        assert torch.allclose(logits_full, logits_step, atol=1e-4), f"max diff: {(logits_full - logits_step).abs().max()}"


if __name__ == "__main__":
    unittest.main()
