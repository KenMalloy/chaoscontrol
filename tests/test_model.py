"""Tests for ChaosSSMBlock and ChaosStudentLM."""
from __future__ import annotations

import unittest
from unittest import mock

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

    def test_encode_memory_mode_off_disables_outer_memory_read(self) -> None:
        torch.manual_seed(0)
        model = ChaosStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )
        assert model.outer_model is not None
        model.outer_model.state.fill_(3.0)
        ids = torch.randint(0, 64, (2, 6))

        h_off = model.encode(ids, memory_mode="off")
        h_on = model.encode(ids, memory_mode="force_on")

        assert not torch.allclose(h_off, h_on)

    def test_encode_default_controller_mode_is_force_on_without_controller(self) -> None:
        torch.manual_seed(1)
        model = ChaosStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )
        ids = torch.randint(0, 64, (2, 6))

        h_default = model.encode(ids)
        h_force = model.encode(ids, memory_mode="force_on")

        assert torch.allclose(h_default, h_force, atol=0.0, rtol=0.0)

    def test_encode_teacher_gate_zero_matches_memory_off(self) -> None:
        torch.manual_seed(2)
        model = ChaosStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )
        assert model.outer_model is not None
        model.outer_model.state.fill_(2.0)
        ids = torch.randint(0, 64, (2, 6))
        zero_gate = torch.zeros(2, 6)

        h_off = model.encode(ids, memory_mode="off")
        h_gate0 = model.encode(
            ids, memory_mode="teacher_gate", teacher_gate=zero_gate
        )

        assert torch.allclose(h_off, h_gate0, atol=0.0, rtol=0.0)

    def test_encode_controller_mode_can_return_logits_and_meta(self) -> None:
        torch.manual_seed(3)
        model = ChaosStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
            enable_controller=True,
        )
        ids = torch.randint(0, 64, (2, 6))

        out = model.encode(
            ids,
            memory_mode="controller",
            cache_read_cutoff=123,
            return_controller_logits=True,
            return_memory_meta=True,
        )

        assert isinstance(out, dict)
        assert out["hidden"].shape == (2, 6, 8)
        assert out["controller_logits"].shape == (2, 6)
        assert out["memory_meta"]["cache_read_cutoff"] == 123
        assert out["memory_meta"]["memory_gate"].shape == (2, 6)

    def test_encode_cache_read_cutoff_filters_append_only_multislot_reads(self) -> None:
        torch.manual_seed(4)
        model = ChaosStudentLM(
            vocab_size=64,
            dim=8,
            num_layers=1,
            ff_mult=2,
            a_mode="diag",
            rich_b_mode="none",
            outer_model_dim=8,
            outer_model_type="multislot",
            outer_max_slots=8,
            buffer_mode="append_only",
            retrieval_mode="bucket_mean",
        )
        assert model.outer_model is not None
        with torch.no_grad():
            model.outer_model.decoder.weight.copy_(torch.eye(8))
        model.outer_model._append_kv_batch_committed(
            torch.ones(1, 8),
            torch.tensor([0]),
            event_ids=torch.tensor([2]),
        )
        ids = torch.randint(0, 64, (2, 6))

        h_off = model.encode(ids, memory_mode="off")
        h_cutoff_before_write = model.encode(
            ids,
            memory_mode="force_on",
            cache_read_cutoff=1,
        )
        h_cutoff_at_write = model.encode(
            ids,
            memory_mode="force_on",
            cache_read_cutoff=2,
        )

        torch.testing.assert_close(h_cutoff_before_write, h_off, rtol=0, atol=0)
        assert not torch.allclose(h_cutoff_at_write, h_off)


class TestChaosStudentLMHybrid(unittest.TestCase):
    def test_student_lm_with_hybrid_top_block(self) -> None:
        model = ChaosStudentLM(
            vocab_size=64, dim=32, num_layers=4, ff_mult=2,
            a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        x = torch.randint(0, 64, (2, 10))
        out = model(x)
        assert out["logits"].shape == (2, 10, 64)

    def test_student_lm_hybrid_step(self) -> None:
        model = ChaosStudentLM(
            vocab_size=64, dim=32, num_layers=4, ff_mult=2,
            a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        states = model.init_state(2)
        kv_caches = model.init_kv_caches()
        token = torch.randint(0, 64, (2, 1))
        logits, hidden, new_states = model.step(token, states, kv_caches=kv_caches)
        assert logits.shape == (2, 64)
        assert len(new_states) == 4

    def test_student_lm_hybrid_step_raises_without_kv_caches(self) -> None:
        model = ChaosStudentLM(
            vocab_size=64, dim=32, num_layers=4, ff_mult=2,
            a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        states = model.init_state(2)
        token = torch.randint(0, 64, (2, 1))
        with self.assertRaises(RuntimeError):
            model.step(token, states)

    def test_student_lm_hybrid_dream_step(self) -> None:
        model = ChaosStudentLM(
            vocab_size=64, dim=32, num_layers=4, ff_mult=2,
            a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        states = model.init_state(2)
        kv_caches = model.init_kv_caches()
        token = torch.randint(0, 64, (2, 1))
        logits, hidden, new_states = model.dream_step(token, states, kv_caches=kv_caches)
        assert logits.shape == (2, 64)
        assert len(new_states) == 4

    def test_hybrid_block_jacobian_stats_returns_zeros(self) -> None:
        from chaoscontrol.model import ChaosSSMHybridBlock
        block = ChaosSSMHybridBlock(
            dim=32, ff_mult=2, a_mode="diag",
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        x = torch.randn(2, 6, 32)
        y, stats = block(x, return_jacobian_stats=True)
        assert y.shape == (2, 6, 32)
        assert "lambda_max" in stats
        assert "sv_log_var" in stats
        # Hybrid block returns zeros (documented limitation)
        assert stats["lambda_max"].item() == 0.0

    def test_student_lm_no_hybrid_by_default(self) -> None:
        model = ChaosStudentLM(
            vocab_size=64, dim=32, num_layers=4, ff_mult=2,
            a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
        )
        # All layers should be plain SSM blocks
        for layer in model.layers:
            assert isinstance(layer, ChaosSSMBlock)


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

    def test_hybrid_block_first_step_is_causal(self) -> None:
        from chaoscontrol.model import ChaosSSMHybridBlock
        block = ChaosSSMHybridBlock(
            dim=32, ff_mult=2, a_mode="diag",
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        cache = block._init_kv_cache()
        state = torch.zeros(2, 32)
        x = torch.randn(2, 32)
        with mock.patch.object(block.local_attn, "forward", wraps=block.local_attn.forward) as attn_forward:
            out, new_state = block.step(x, state, kv_cache=cache)
        assert out.shape == (2, 32)
        assert new_state.shape == (2, 32)
        assert attn_forward.call_count == 0

    def test_hybrid_parallel_forward_matches_sequential_step(self) -> None:
        """Regression: parallel forward() must match sequential step() loop."""
        from chaoscontrol.model import ChaosSSMHybridBlock
        for window in (8, 16, 32):
            torch.manual_seed(42)
            block = ChaosSSMHybridBlock(
                dim=32, ff_mult=2, a_mode="diag",
                local_attn_window=window, local_attn_heads=1, local_attn_dim=16,
            )
            block.eval()
            x = torch.randn(2, 24, 32)
            with torch.no_grad():
                y_parallel = block(x)
                kv_cache = block._init_kv_cache()
                state = x.new_zeros(2, 32)
                outputs = []
                for t in range(24):
                    out, state = block.step(x[:, t], state, kv_cache=kv_cache)
                    outputs.append(out)
                y_sequential = torch.stack(outputs, dim=1)
            max_diff = (y_parallel - y_sequential).abs().max().item()
            assert max_diff < 1e-4, (
                f"window={window}: parallel vs sequential max diff {max_diff:.2e} exceeds 1e-4"
            )

    def test_topk_is_causal_future_perturbation(self) -> None:
        """Regression: top-k by score must be strictly causal.

        Perturbing future tokens must not change past outputs. Probes the
        subtle -1e9 invariant in the score-topk branch where threshold can
        be -1e9 on early rows but non-causal positions still carry -1e9
        scores so they don't affect softmax.
        """
        from chaoscontrol.model import ChaosSSMHybridBlock
        for topk in (8, 16):
            torch.manual_seed(42)
            block = ChaosSSMHybridBlock(
                dim=32, ff_mult=2, a_mode="diag",
                local_attn_window=64, local_attn_heads=1, local_attn_dim=16,
                local_attn_topk=topk, local_attn_topk_random=False,
            )
            block.eval()
            x1 = torch.randn(2, 24, 32)
            x2 = x1.clone()
            # Perturb the future (positions >= split)
            split = 12
            x2[:, split:] = torch.randn(2, 24 - split, 32)
            with torch.no_grad():
                y1 = block(x1)
                y2 = block(x2)
            past_diff = (y1[:, :split] - y2[:, :split]).abs().max().item()
            assert past_diff < 1e-5, (
                f"topk={topk}: past outputs changed after future perturbation "
                f"(max diff {past_diff:.2e}) — causality violated"
            )

    def test_topk_random_is_causal_future_perturbation(self) -> None:
        """Regression: top-k random must also be strictly causal.

        Same property test for the random selection branch, which picks
        k random causal positions per forward call.
        """
        from chaoscontrol.model import ChaosSSMHybridBlock
        for topk in (8, 16):
            torch.manual_seed(42)
            block = ChaosSSMHybridBlock(
                dim=32, ff_mult=2, a_mode="diag",
                local_attn_window=64, local_attn_heads=1, local_attn_dim=16,
                local_attn_topk=topk, local_attn_topk_random=True,
            )
            block.eval()
            x1 = torch.randn(2, 24, 32)
            x2 = x1.clone()
            split = 12
            x2[:, split:] = torch.randn(2, 24 - split, 32)
            # Use same random seed for both calls so the selection mask is identical,
            # isolating causality from random variation.
            with torch.no_grad():
                torch.manual_seed(777)
                y1 = block(x1)
                torch.manual_seed(777)
                y2 = block(x2)
            past_diff = (y1[:, :split] - y2[:, :split]).abs().max().item()
            assert past_diff < 1e-5, (
                f"topk_random={topk}: past outputs changed after future perturbation "
                f"(max diff {past_diff:.2e}) — causality violated"
            )


if __name__ == "__main__":
    unittest.main()
