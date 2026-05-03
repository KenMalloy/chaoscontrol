"""Tests for CareSSMBlock and CareStudentLM."""
from __future__ import annotations

import unittest
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from chaoscontrol.model import CareSSMBlock, CareStudentLM


class TestCareSSMBlock(unittest.TestCase):
    def test_basic_forward(self) -> None:
        block = CareSSMBlock(16, ff_mult=2, a_mode="diag", rich_b_mode="none")
        x = torch.randn(2, 8, 16)
        out = block(x)
        assert out.shape == (2, 8, 16)

    def test_with_rich_b_nn(self) -> None:
        block = CareSSMBlock(16, ff_mult=2, a_mode="diag", rich_b_mode="nn", rich_b_bottleneck=8)
        x = torch.randn(2, 8, 16)
        out = block(x)
        assert out.shape == (2, 8, 16)

    def test_jacobian_stats(self) -> None:
        block = CareSSMBlock(16, ff_mult=2, a_mode="full", a_full_rank=4, rich_b_mode="none")
        x = torch.randn(2, 8, 16)
        out, stats = block(x, return_jacobian_stats=True)
        assert out.shape == (2, 8, 16)
        assert "lambda_max" in stats


class TestCareStudentLM(unittest.TestCase):
    def test_base_forward_produces_logits(self) -> None:
        model = CareStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)

    def test_full_config_forward(self) -> None:
        model = CareStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="full", a_full_rank=4,
            rich_b_mode="assembly", rich_b_bottleneck=8, rich_b_num_subnets=4,
            outer_model_dim=32,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)

    def test_gradients_flow(self) -> None:
        model = CareStudentLM(
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
        model = CareStudentLM(
            vocab_size=256, dim=128, num_layers=4, ff_mult=2,
            a_mode="full", a_full_rank=8,
            rich_b_mode="hybrid", rich_b_bottleneck=32, rich_b_num_subnets=4,
            outer_model_dim=64,
        )
        assert model.artifact_bytes() < 16 * 1024 * 1024

    def test_with_outer_model_reads(self) -> None:
        model = CareStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=32,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)

    def test_jacobian_stats_with_full(self) -> None:
        model = CareStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="full", a_full_rank=4, rich_b_mode="none", outer_model_dim=0,
        )
        ids = torch.randint(0, 256, (2, 8))
        out = model(ids, return_jacobian_stats=True)
        assert "jacobian_stats" in out


    def test_posterior_global_delta_forward(self) -> None:
        model = CareStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
            posterior_mode="global_delta", posterior_lr=0.01,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)
        assert model.posterior is not None

    def test_posterior_bucket_delta_forward(self) -> None:
        model = CareStudentLM(
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
        model = CareStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
            posterior_mode="residual_cache", residual_cache_k=2,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, 256)
        assert model.posterior is not None

    def test_posterior_none_default(self) -> None:
        model = CareStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
        )
        assert model.posterior is None

    def test_encode_memory_mode_off_disables_outer_memory_read(self) -> None:
        torch.manual_seed(0)
        model = CareStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )
        assert model.outer_model is not None
        model.outer_model.state.fill_(3.0)
        ids = torch.randint(0, 64, (2, 6))

        h_off = model.encode(ids, memory_mode="off")
        h_on = model.encode(ids, memory_mode="force_on")

        assert not torch.allclose(h_off, h_on)

    def test_encode_default_packet_mode_matches_memory_off(self) -> None:
        torch.manual_seed(1)
        model = CareStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )
        ids = torch.randint(0, 64, (2, 6))

        h_default = model.encode(ids)
        h_off = model.encode(ids, memory_mode="off")

        assert torch.allclose(h_default, h_off, atol=0.0, rtol=0.0)

    def test_encode_rejects_removed_controller_modes(self) -> None:
        torch.manual_seed(2)
        model = CareStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )
        ids = torch.randint(0, 64, (2, 6))

        with pytest.raises(ValueError, match="memory_mode"):
            model.encode(ids, memory_mode="controller")

        with pytest.raises(ValueError, match="memory_mode"):
            model.encode(ids, memory_mode="teacher_gate")

    def test_model_has_no_trunk_local_memory_controller(self) -> None:
        torch.manual_seed(3)
        model = CareStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )

        assert not hasattr(model, "memory_controller")

    def test_encode_packet_mode_zero_payload_matches_memory_off(self) -> None:
        torch.manual_seed(33)
        model = CareStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )
        assert model.outer_model is not None
        model.outer_model.state.fill_(3.0)
        ids = torch.randint(0, 64, (2, 6))

        h_off = model.encode(ids, memory_mode="off")
        h_packet = model.encode(ids, memory_mode="packet")

        assert torch.allclose(h_off, h_packet, atol=0.0, rtol=0.0)

    def test_encode_packet_mode_injects_residual_without_controller_head(self) -> None:
        torch.manual_seed(34)
        model = CareStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )
        ids = torch.randint(0, 64, (2, 6))
        residual = torch.randn(2, 1, 8)
        gate = torch.ones(2, 6)

        h_off = model.encode(ids, memory_mode="off")
        out = model.encode(
            ids,
            memory_mode="packet",
            episodic_residual=residual,
            episodic_gate=gate,
            return_memory_meta=True,
        )

        assert isinstance(out, dict)
        assert out["hidden"].shape == h_off.shape
        assert not torch.allclose(h_off, out["hidden"])
        assert out["memory_meta"]["memory_gate"].shape == (2, 6)
        assert out["memory_meta"]["memory_residual"].shape == (2, 1, 8)

    def test_encode_packet_mode_does_not_compute_sidecar_cue(self) -> None:
        torch.manual_seed(36)
        model = CareStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
            outer_model_type="multislot",
            buffer_mode="append_only",
            retrieval_mode="softmax_all",
        )
        ids = torch.randint(0, 64, (2, 6))
        residual = torch.randn(2, 1, 8)
        gate = torch.ones(2, 6)

        model.encode(
            ids,
            memory_mode="packet",
            episodic_residual=residual,
            episodic_gate=gate,
        )
        assert model._last_outer_cue is None

        model.encode(ids, memory_mode="force_on")
        assert model._last_outer_cue is not None

    def test_encode_packet_mode_rejects_sequence_residual_packets(self) -> None:
        model = CareStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )
        ids = torch.randint(0, 64, (2, 6))
        residual = torch.randn(2, 6, 8)
        gate = torch.ones(2, 6)

        with pytest.raises(ValueError, match="compact"):
            model.encode(
                ids,
                memory_mode="packet",
                episodic_residual=residual,
                episodic_gate=gate,
            )

    def test_force_on_memory_residual_can_replay_as_packet(self) -> None:
        torch.manual_seed(35)
        model = CareStudentLM(
            vocab_size=64, dim=8, num_layers=1, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=8,
        )
        assert model.outer_model is not None
        model.outer_model.state.fill_(2.0)
        ids = torch.randint(0, 64, (2, 6))

        force = model.encode(
            ids,
            memory_mode="force_on",
            return_memory_meta=True,
        )
        assert isinstance(force, dict)
        residual = force["memory_meta"]["memory_residual"]
        gate = force["memory_meta"]["memory_gate"]
        assert residual is not None
        assert gate is not None
        assert residual.abs().sum() > 0

        packet = model.encode(
            ids,
            memory_mode="packet",
            episodic_residual=residual,
            episodic_gate=gate,
        )
        off = model.encode(ids, memory_mode="off")
        zero_gate = torch.zeros_like(gate)
        packet_zero = model.encode(
            ids,
            memory_mode="packet",
            episodic_residual=residual,
            episodic_gate=zero_gate,
        )

        assert torch.allclose(packet, force["hidden"], atol=0.0, rtol=0.0)
        assert torch.allclose(packet_zero, off, atol=0.0, rtol=0.0)

    def test_encode_paired_for_score_matches_separate_off_and_force_on(self) -> None:
        torch.manual_seed(36)
        model = CareStudentLM(
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
        ids = torch.randint(0, 64, (3, 7))

        h_off = model.encode(ids, memory_mode="off", cache_read_cutoff=2)
        force = model.encode(
            ids,
            memory_mode="force_on",
            cache_read_cutoff=2,
            return_memory_meta=True,
        )
        assert isinstance(force, dict)
        paired_off, paired_mem, meta = model.encode_paired_for_score(
            ids,
            cache_read_cutoff=2,
        )

        torch.testing.assert_close(paired_off, h_off, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(paired_mem, force["hidden"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            meta["memory_residual"],
            force["memory_meta"]["memory_residual"],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            meta["memory_gate"],
            force["memory_meta"]["memory_gate"],
            rtol=0,
            atol=0,
        )

    def test_append_memory_accepts_selected_event_ids(self) -> None:
        model = CareStudentLM(
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
        )
        hidden = torch.randn(2, 5, 8)
        score = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        event_ids = torch.tensor([100, 101, 102], dtype=torch.long)

        write_records = model.append_memory_from_hidden(
            hidden,
            score=score,
            max_tokens=3,
            event_ids=event_ids,
        )

        assert len(write_records) == 3
        assert model.outer_model is not None
        assert model.outer_model.table._slot_event_ids == [100, 101, 102]
        assert len(model.outer_model.table) == 3

    def test_encode_cache_read_cutoff_filters_append_only_multislot_reads(self) -> None:
        torch.manual_seed(4)
        model = CareStudentLM(
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

    def test_encode_slot_override_matches_temporary_replacement_in_bucket_path(self) -> None:
        torch.manual_seed(5)
        model = CareStudentLM(
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
            retrieval_mode="softmax_all",
        )
        assert model.outer_model is not None
        with torch.no_grad():
            model.outer_model.decoder.weight.copy_(torch.eye(8))
            model.outer_model.cue_proj.weight.copy_(torch.eye(8))
        model.outer_model._append_kv_batch_committed(
            torch.stack([torch.randn(8), torch.randn(8)]),
            torch.tensor([0, 0]),
            event_ids=torch.tensor([1, 1]),
        )
        ids = torch.randint(0, 64, (3, 6))
        candidate = torch.randn(1, 8)

        h_override = model.encode(
            ids,
            memory_mode="force_on",
            memory_slot_override_index=0,
            memory_slot_override_values=candidate.expand(ids.shape[0], -1),
        )
        original = model.outer_model.table.get_tensor(0).detach().clone()
        model.outer_model.table.replace_tensor(0, candidate, bump_generation=False)
        h_replaced = model.encode(ids, memory_mode="force_on")
        model.outer_model.table.replace_tensor(0, original, bump_generation=False)

        torch.testing.assert_close(h_override, h_replaced, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(model.outer_model.table.get_tensor(0), original)


class TestCareStudentLMHybrid(unittest.TestCase):
    def test_student_lm_with_hybrid_top_block(self) -> None:
        model = CareStudentLM(
            vocab_size=64, dim=32, num_layers=4, ff_mult=2,
            a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        x = torch.randint(0, 64, (2, 10))
        out = model(x)
        assert out["logits"].shape == (2, 10, 64)

    def test_student_lm_hybrid_step(self) -> None:
        model = CareStudentLM(
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
        model = CareStudentLM(
            vocab_size=64, dim=32, num_layers=4, ff_mult=2,
            a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        states = model.init_state(2)
        token = torch.randint(0, 64, (2, 1))
        with self.assertRaises(RuntimeError):
            model.step(token, states)

    def test_student_lm_hybrid_dream_step(self) -> None:
        model = CareStudentLM(
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
        from chaoscontrol.model import CareSSMHybridBlock
        block = CareSSMHybridBlock(
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
        model = CareStudentLM(
            vocab_size=64, dim=32, num_layers=4, ff_mult=2,
            a_mode="diag", outer_model_dim=0, wernicke_enabled=False,
        )
        # All layers should be plain SSM blocks
        for layer in model.layers:
            assert isinstance(layer, CareSSMBlock)


class TestCareSSMHybridBlock(unittest.TestCase):
    def test_hybrid_block_forward_shape(self) -> None:
        from chaoscontrol.model import CareSSMHybridBlock
        block = CareSSMHybridBlock(
            dim=32, ff_mult=2, a_mode="diag",
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        x = torch.randn(2, 12, 32)
        y = block(x)
        assert y.shape == (2, 12, 32)

    def test_hybrid_block_step_shape(self) -> None:
        from chaoscontrol.model import CareSSMHybridBlock
        block = CareSSMHybridBlock(
            dim=32, ff_mult=2, a_mode="diag",
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        state = torch.zeros(2, 32)
        x = torch.randn(2, 32)
        out, new_state = block.step(x, state)
        assert out.shape == (2, 32)
        assert new_state.shape == (2, 32)

    def test_hybrid_block_gate_starts_near_zero(self) -> None:
        from chaoscontrol.model import CareSSMHybridBlock
        block = CareSSMHybridBlock(
            dim=32, ff_mult=2, a_mode="diag",
            local_attn_window=8, local_attn_heads=1, local_attn_dim=16,
        )
        # gate_bias initialized to -4, sigmoid(-4) ~ 0.018
        assert block.gate_bias.item() < -3.0

    def test_hybrid_block_first_step_is_causal(self) -> None:
        from chaoscontrol.model import CareSSMHybridBlock
        block = CareSSMHybridBlock(
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
        from chaoscontrol.model import CareSSMHybridBlock
        for window in (8, 16, 32):
            torch.manual_seed(42)
            block = CareSSMHybridBlock(
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
        from chaoscontrol.model import CareSSMHybridBlock
        for topk in (8, 16):
            torch.manual_seed(42)
            block = CareSSMHybridBlock(
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
        from chaoscontrol.model import CareSSMHybridBlock
        for topk in (8, 16):
            torch.manual_seed(42)
            block = CareSSMHybridBlock(
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


def test_packet_forward_state_keys_cover_packet_forward() -> None:
    """Mutating non-packet weights must not change ``build_episodic_packet``.

    The packet-serving rank consumes a snapshot subset advertised by
    ``CareStudentLM.packet_forward_state_keys``. This test fails loud if a
    future change to ``build_episodic_packet`` starts reading a parameter
    outside that subset — drop a NaN into every non-listed weight, re-run
    the packet path, and require a bit-identical output.
    """
    torch.manual_seed(0)
    model = CareStudentLM(
        vocab_size=32,
        dim=8,
        num_layers=2,
        ff_mult=2,
        a_mode="diag",
        outer_model_dim=4,
        outer_model_type="multislot",
        outer_max_slots=8,
        buffer_mode="append_only",
        wernicke_enabled=True,
        wernicke_k_max=4,
        cue_projection=True,
        semantic_tier_bases=2,
        bucket_prototypes=True,
        prototype_dim=4,
    )
    model.eval()

    packet_keys = model.packet_forward_state_keys()
    state = model.state_dict()
    non_packet_tensor_keys = [
        name
        for name, value in state.items()
        if torch.is_tensor(value) and name not in packet_keys
    ]
    assert non_packet_tensor_keys, (
        "test fixture should expose at least one non-packet tensor; "
        "if the model now ships only packet tensors, this guard is obsolete"
    )

    input_ids = torch.randint(0, 32, (2, 5), dtype=torch.long)
    with torch.no_grad():
        baseline = model.build_episodic_packet(input_ids)
    baseline_residual = baseline["memory_residual"].detach().clone()
    baseline_gate = baseline["memory_gate"].detach().clone()

    with torch.no_grad():
        for name in non_packet_tensor_keys:
            param = dict(model.named_parameters()).get(name)
            buffer = dict(model.named_buffers()).get(name)
            target = param if param is not None else buffer
            if target is None:
                continue
            target.fill_(float("nan"))

    with torch.no_grad():
        perturbed = model.build_episodic_packet(input_ids)
    torch.testing.assert_close(perturbed["memory_residual"], baseline_residual)
    torch.testing.assert_close(perturbed["memory_gate"], baseline_gate)


if __name__ == "__main__":
    unittest.main()
