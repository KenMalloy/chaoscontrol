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


    def test_posterior_global_delta_forward(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
            posterior_mode="global_delta", posterior_lr=0.01,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)
        assert model.posterior is not None

    def test_posterior_bucket_delta_forward(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
            wernicke_enabled=True, wernicke_k_max=8, wernicke_router="moe",
            posterior_mode="bucket_delta", posterior_lr=0.01,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)
        assert model.posterior is not None

    def test_posterior_residual_cache_forward(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
            posterior_mode="residual_cache", residual_cache_k=2,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)
        assert model.posterior is not None

    def test_posterior_none_default(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
        )
        assert model.posterior is None


class TestChaosSSMHybridBlock(unittest.TestCase):
    def test_hybrid_block_forward_shape(self) -> None:
        from chaoscontrol.model import ChaosSSMHybridBlock
        block = ChaosSSMHybridBlock(
            dim=32, ff_mult=2, a_mode="diag",
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        x = torch.randn(2, 12, 32)
        y = block(x)
        assert y.shape == (2, 12, 32)

    def test_hybrid_block_step_shape(self) -> None:
        from chaoscontrol.model import ChaosSSMHybridBlock
        block = ChaosSSMHybridBlock(
            dim=32, ff_mult=2, a_mode="diag",
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        state = torch.zeros(2, 32)
        x = torch.randn(2, 32)
        out, new_state = block.step(x, state)
        assert out.shape == (2, 32)
        assert new_state.shape == (2, 32)

    def test_hybrid_block_gate_starts_near_zero(self) -> None:
        from chaoscontrol.model import ChaosSSMHybridBlock
        block = ChaosSSMHybridBlock(
            dim=32, ff_mult=2, a_mode="diag",
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        # gate_bias initialized to -4, sigmoid(-4) ~ 0.018
        assert block.gate_bias.item() < -3.0


if __name__ == "__main__":
    unittest.main()
