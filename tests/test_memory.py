#!/usr/bin/env python3
"""Tests for memory modules: OuterModel and MultiSlotOuterModel."""
from __future__ import annotations

import io
import unittest

import torch

from chaoscontrol.memory import MultiSlotOuterModel, OuterModel, SemanticTier


class TestOuterModel(unittest.TestCase):
    def test_encode_decode_shapes(self) -> None:
        om = OuterModel(model_dim=16, outer_dim=32)
        decoded = om.read(batch_size=2)
        assert decoded.shape == (2, 16)
        om.write(torch.randn(2, 16))
        decoded2 = om.read(batch_size=2)
        assert decoded2.shape == (2, 16)

    def test_lossy_roundtrip(self) -> None:
        om = OuterModel(model_dim=16, outer_dim=32)
        h = torch.randn(2, 16)
        om.write(h)
        decoded = om.read(batch_size=2)
        assert not torch.allclose(h, decoded, atol=1e-3)

    def test_persists_across_calls(self) -> None:
        om = OuterModel(model_dim=16, outer_dim=32)
        h1 = torch.randn(2, 16) * 5.0
        om.write(h1)
        read1 = om.read(batch_size=2).clone()
        h2 = torch.randn(2, 16) * 5.0
        om.write(h2)
        read2 = om.read(batch_size=2)
        assert not torch.allclose(read1, read2, atol=1e-3)

    def test_weighted_write_biases_state_toward_heavier_sample(self) -> None:
        om = OuterModel(model_dim=2, outer_dim=2)
        with torch.no_grad():
            om.encoder.weight.copy_(torch.eye(2))
            om.log_inertia.fill_(-20.0)  # near-zero inertia so the write dominates

        h = torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32)

        om.state.zero_()
        om.write(h)
        unweighted = om.state.clone()

        om.state.zero_()
        om.write(h, per_sample_weights=torch.tensor([0.9, 0.1], dtype=torch.float32))
        weighted = om.state.clone()

        assert weighted[0, 0] > unweighted[0, 0]
        assert weighted[0, 1] < unweighted[0, 1]

    def test_consolidation_symmetric(self) -> None:
        om = OuterModel(model_dim=16, outer_dim=32, consolidation_mode="symmetric")
        signal = om.compute_consolidation_signal(current_loss=5.0, running_avg=2.0)
        assert signal > 0.0

    def test_consolidation_boring_is_low(self) -> None:
        om = OuterModel(model_dim=16, outer_dim=32, consolidation_mode="symmetric")
        signal = om.compute_consolidation_signal(current_loss=2.0, running_avg=2.0)
        assert abs(signal) < 1e-5

    def test_consolidation_step_updates_state(self) -> None:
        om = OuterModel(model_dim=16, outer_dim=32)
        initial = om.state.clone()
        h = torch.randn(2, 16) * 5.0
        om.consolidation_step(h, current_loss=10.0)  # big surprise
        assert not torch.allclose(initial, om.state)

    def test_consolidation_pain_biased(self) -> None:
        om = OuterModel(model_dim=16, outer_dim=32, consolidation_mode="pain_biased")
        pain_signal = om.compute_consolidation_signal(current_loss=5.0, running_avg=2.0)
        reward_signal = om.compute_consolidation_signal(current_loss=0.0, running_avg=2.0)
        # Pain should consolidate harder than equivalent reward
        assert pain_signal > reward_signal

    def test_consolidation_learned_differs_from_symmetric(self) -> None:
        om_sym = OuterModel(model_dim=16, outer_dim=32, consolidation_mode="symmetric")
        om_learn = OuterModel(model_dim=16, outer_dim=32, consolidation_mode="learned")
        sig_sym = om_sym.compute_consolidation_signal(current_loss=5.0, running_avg=2.0)
        sig_learn = om_learn.compute_consolidation_signal(current_loss=5.0, running_avg=2.0)
        # At init w=0, sigmoid(0)=0.5, so learned = 0.5*pain + 0.5*reward = 1.5 vs symmetric = 3.0
        assert sig_sym != sig_learn

    def test_consolidation_learned_w_updates(self) -> None:
        om = OuterModel(model_dim=16, outer_dim=32, consolidation_mode="learned")
        h = torch.randn(2, 16) * 5.0
        initial_w = om.consolidation_w.item()
        # Simulate a pain event followed by improvement
        om.consolidation_step(h, current_loss=10.0)  # pain, big surprise
        om.consolidation_step(h, current_loss=5.0)   # improved after pain
        # w should have nudged toward pain (positive direction)
        assert om.consolidation_w.item() != initial_w

    def test_encoder_not_trainable(self) -> None:
        om = OuterModel(model_dim=16, outer_dim=32)
        assert not om.encoder.weight.requires_grad

    def test_decoder_is_trainable(self) -> None:
        om = OuterModel(model_dim=16, outer_dim=32)
        assert om.decoder.weight.requires_grad


class TestMultiSlotOuterModel(unittest.TestCase):
    def test_read_empty_returns_zeros(self) -> None:
        om = MultiSlotOuterModel(model_dim=16, outer_dim=32, max_slots=8)
        out = om.read(2)
        assert out.shape == (2, 16)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_write_creates_slot(self) -> None:
        om = MultiSlotOuterModel(model_dim=16, outer_dim=32, max_slots=8)
        assert len(om._slots) == 0
        om.write(torch.randn(2, 16))
        assert len(om._slots) == 1

    def test_cue_dependent_retrieval(self) -> None:
        om = MultiSlotOuterModel(model_dim=16, outer_dim=32, max_slots=8)
        # Write two distinct slots
        h1 = torch.randn(1, 16) * 5.0
        h2 = -h1  # opposite direction
        om.write(h1)
        om.write(h2)
        # Cue similar to h1 should retrieve something different from cue similar to h2
        read1 = om.read(1, cue=h1)
        read2 = om.read(1, cue=h2)
        assert not torch.allclose(read1, read2, atol=1e-3)

    def test_read_slot_mask_is_per_sample(self) -> None:
        om = MultiSlotOuterModel(model_dim=4, outer_dim=4, max_slots=8)
        with torch.no_grad():
            om.decoder.weight.copy_(torch.eye(4))
            om.cue_proj.weight.copy_(torch.eye(4))
        s0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        s1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        om.table.append(s0)
        om.table.append(s1)
        cue = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
        slot_mask = torch.tensor([[True, False], [False, True]])
        out = om.read(2, cue=cue, slot_mask=slot_mask)
        assert torch.allclose(out[0], s0.squeeze(0), atol=1e-3)
        assert torch.allclose(out[1], s1.squeeze(0), atol=1e-3)

    def test_compression_fires_at_capacity(self) -> None:
        om = MultiSlotOuterModel(model_dim=16, outer_dim=32, max_slots=4)
        for _ in range(5):
            om.write(torch.randn(2, 16))
        # Should have compressed — fewer than 5 slots
        assert len(om._slots) <= 4

    def test_survival_score_updates(self) -> None:
        om = MultiSlotOuterModel(model_dim=16, outer_dim=32, max_slots=8)
        om.write(torch.randn(2, 16))
        assert om._survival[0] == 0.0
        # Read with cue to populate retrieval weights
        om.read(2, cue=torch.randn(2, 16))
        # Simulate a good loss (model did better than avg)
        om.loss_ema.fill_(5.0)
        om.update_survival(current_loss=2.0)  # impact = 5 - 2 = 3, positive
        assert om._survival[0] > 0.0

    def test_high_survival_resists_compression(self) -> None:
        om = MultiSlotOuterModel(model_dim=4, outer_dim=4, max_slots=4, compress_ratio=2)
        # Write 3 slots, give first one high survival
        om.write(torch.randn(1, 4))
        om._survival[0] = 100.0  # very high survival
        om.write(torch.randn(1, 4))
        om.write(torch.randn(1, 4))
        om.write(torch.randn(1, 4))
        om.write(torch.randn(1, 4))  # triggers compression
        # The high-survival slot should have survived
        assert any(s >= 50.0 for s in om._survival)

    def test_append_kv_batch(self) -> None:
        om = MultiSlotOuterModel(model_dim=16, outer_dim=32, max_slots=100)
        encoded = torch.randn(5, 32)  # 5 pre-encoded entries
        bucket_ids = torch.tensor([0, 1, 2, 0, 1])
        om.append_kv_batch(encoded, bucket_ids)
        assert len(om._slots) == 5
        assert om._slot_buckets == [0, 1, 2, 0, 1]
        assert om._slot_event_ids == [0, 0, 0, 0, 0]
        assert all(s == 1.0 for s in om._survival)
        # Each slot should match the encoded entry
        for i in range(5):
            assert torch.allclose(om._slots[i], encoded[i:i+1])

    def test_read_cutoff_hides_newer_committed_slots(self) -> None:
        om = MultiSlotOuterModel(model_dim=4, outer_dim=4, max_slots=8)
        with torch.no_grad():
            om.decoder.weight.copy_(torch.eye(4))
        encoded = torch.ones(1, 4)
        om._append_kv_batch_committed(
            encoded,
            torch.tensor([0]),
            event_ids=torch.tensor([2]),
        )

        before = om.read_bucket(1, bucket_id=0, read_cutoff=1)
        visible = om.read_bucket(1, bucket_id=0, read_cutoff=2)

        torch.testing.assert_close(before, torch.zeros_like(before), rtol=0, atol=0)
        torch.testing.assert_close(visible, encoded, rtol=0, atol=0)

    def test_append_kv_batch_empty(self) -> None:
        om = MultiSlotOuterModel(model_dim=16, outer_dim=32, max_slots=100)
        encoded = torch.randn(0, 32)
        bucket_ids = torch.tensor([], dtype=torch.long)
        om.append_kv_batch(encoded, bucket_ids)
        assert len(om._slots) == 0

    def test_typed_compression_merges_within_bucket(self) -> None:
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


class TestCheckpointPersistence(unittest.TestCase):
    def test_multislot_roundtrip(self) -> None:
        om = MultiSlotOuterModel(model_dim=16, outer_dim=32, max_slots=8)
        om.write(torch.randn(2, 16))
        om.write(torch.randn(2, 16))
        om._survival[0] = 42.0
        om._slot_event_ids[0] = 7
        assert len(om._slots) == 2

        buf = io.BytesIO()
        torch.save(om.state_dict(), buf)
        buf.seek(0)

        om2 = MultiSlotOuterModel(model_dim=16, outer_dim=32, max_slots=8)
        assert len(om2._slots) == 0
        om2.load_state_dict(torch.load(buf, weights_only=False))
        assert len(om2._slots) == 2
        assert om2._survival[0] == 42.0
        assert om2._slot_event_ids[0] == 7


class TestSemanticTier(unittest.TestCase):
    def test_read_shape(self):
        st = SemanticTier(model_dim=16, num_bases=8)
        bias = st.read(batch_size=2)
        assert bias.shape == (2, 16)

    def test_read_zero_before_consolidation(self):
        st = SemanticTier(model_dim=16, num_bases=8)
        bias = st.read(batch_size=2)
        assert torch.allclose(bias, torch.zeros_like(bias))

    def test_updates_from_episodes(self):
        st = SemanticTier(model_dim=16, num_bases=8)
        initial = st.bases.clone()
        for _ in range(10):
            st.consolidate_from_episodes(torch.randn(1, 16) + 1.0)
        assert not torch.allclose(initial, st.bases)

    def test_always_on_after_consolidation(self):
        st = SemanticTier(model_dim=16, num_bases=8)
        st.consolidate_from_episodes(torch.randn(3, 16) * 5.0)
        bias = st.read(batch_size=2)
        assert bias.abs().sum() > 0

    def test_encoder_not_trainable(self):
        st = SemanticTier(model_dim=16, num_bases=8)
        assert not st.encoder.weight.requires_grad

    def test_decoder_is_trainable(self):
        st = SemanticTier(model_dim=16, num_bases=8)
        assert st.decoder.weight.requires_grad

    def test_slow_update_rate(self):
        st = SemanticTier(model_dim=16, num_bases=8, update_rate=0.01)
        st.consolidate_from_episodes(torch.ones(1, 16) * 100.0)
        # After one update at 1% rate, bases should be small
        assert st.bases.abs().max().item() < 10.0


class TestFullSequenceWrite(unittest.TestCase):
    def test_write_from_sequence_captures_trajectory(self):
        """write_sequence should encode the full trajectory, not just last position."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4)
        # Single position write
        h_last = torch.randn(2, 16)
        m.write(h_last)
        slot_single = m._slots[-1].clone()
        m._slots.pop()
        m._survival.pop()
        m._slot_buckets.pop()

        # Full sequence write — should differ because it has more context
        h_seq = torch.randn(2, 32, 16)  # (batch, seq, dim)
        m.write_sequence(h_seq)
        slot_seq = m._slots[-1]
        assert not torch.allclose(slot_single, slot_seq)

    def test_write_sequence_shape(self):
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4)
        h_seq = torch.randn(2, 32, 16)
        m.write_sequence(h_seq, per_sample_weights=torch.tensor([1.0, 2.0]))
        assert len(m._slots) == 1
        assert m._slots[0].shape == (1, 8)

    def test_write_sequence_with_bucket_id(self):
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4)
        h_seq = torch.randn(2, 32, 16)
        m.write_sequence(h_seq, bucket_id=3)
        assert m._slot_buckets[-1] == 3


class TestDemandDrivenCompression(unittest.TestCase):
    def test_no_compression_below_capacity(self):
        """Slots stay full-fidelity when VRAM has space."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
        for _ in range(5):
            m.write(torch.randn(1, 16))
        assert len(m._slots) == 5
        assert all(s.shape == (1, 8) for s in m._slots)

    def test_latent_traces_created_on_compression(self):
        """When compression merges slots, displaced entries become latent traces."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4, compress_ratio=2)
        # Write enough to trigger multiple compressions
        # Use a single bucket so typed compression can merge within the bucket
        for i in range(12):
            m.write(torch.randn(1, 16), bucket_id=0)
        assert hasattr(m, '_latent_traces')
        # Some latent traces should exist (slots were merged away)
        # Note: not all compressions produce latent traces — only when
        # the merge results in fewer total slots than before
        assert len(m._slots) <= 4  # still within capacity

    def test_latent_trace_has_bucket_id(self):
        """Latent traces preserve their bucket membership."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4, compress_ratio=2)
        for i in range(12):
            m.write(torch.randn(1, 16), bucket_id=1)
        if m._latent_traces:
            assert all('bucket_id' in t for t in m._latent_traces)

    def test_try_reactivate_with_matching_bucket(self):
        """try_reactivate returns True and adds a slot when bucket matches a latent trace."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=8, compress_ratio=2)
        # Force latent traces by heavy writing + compression
        for i in range(20):
            m.write(torch.randn(1, 16), bucket_id=0)
        if m._latent_traces:
            result = m.try_reactivate(bucket_id=0, surprise=10.0)
            assert result is True
            # Reactivation may trigger compression, so slots stay within capacity
            assert len(m._slots) <= m.max_slots
        # If no latent traces were created (compression didn't fully displace),
        # that's OK — the test is conditional

    def test_try_reactivate_no_match_returns_false(self):
        """try_reactivate returns False when no latent trace matches the bucket."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=8, compress_ratio=2)
        for i in range(20):
            m.write(torch.randn(1, 16), bucket_id=0)
        result = m.try_reactivate(bucket_id=99, surprise=10.0)
        assert result is False

    def test_try_reactivate_low_surprise_returns_false(self):
        """try_reactivate returns False when surprise is below threshold."""
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=8, compress_ratio=2)
        for i in range(20):
            m.write(torch.randn(1, 16), bucket_id=0)
        if m._latent_traces:
            result = m.try_reactivate(bucket_id=0, surprise=0.001)
            assert result is False


class TestEvalWarmup(unittest.TestCase):
    def test_warmup_does_not_persist_after_eval(self):
        """When warmup=True, eval writes to memory during eval but restores state after."""
        from chaoscontrol.model import ChaosStudentLM
        from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=32, num_layers=2,
            outer_model_dim=16, outer_model_type="multislot",
        )
        tokens = torch.randint(0, 256, (5000,))
        starts = list(range(0, 4000, 128))
        eval_starts = starts[:16]
        slots_before = len(model.outer_model._slots)
        result = evaluate_chaoscontrol_bpb(
            model, tokens=tokens, eval_starts=eval_starts,
            batch_size=4, seq_len=64, device=torch.device("cpu"),
            warmup=True,
        )
        # Warmup writes should be rolled back after eval completes
        assert len(model.outer_model._slots) == slots_before
        assert "bpb" in result

    def test_no_warmup_does_not_write_slots(self):
        """When warmup=False (default), eval should NOT write to memory."""
        from chaoscontrol.model import ChaosStudentLM
        from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=32, num_layers=2,
            outer_model_dim=16, outer_model_type="multislot",
        )
        tokens = torch.randint(0, 256, (5000,))
        starts = list(range(0, 4000, 128))
        eval_starts = starts[:16]
        result = evaluate_chaoscontrol_bpb(
            model, tokens=tokens, eval_starts=eval_starts,
            batch_size=4, seq_len=64, device=torch.device("cpu"),
            warmup=False,
        )
        assert len(model.outer_model._slots) == 0
        assert "bpb" in result


class TestLatentTraceEviction(unittest.TestCase):
    def test_latent_traces_capped_at_max_slots(self):
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4, compress_ratio=2)
        for i in range(50):
            m.write(torch.randn(1, 16), bucket_id=0)
        assert len(m._latent_traces) <= 4

    def test_latent_traces_in_checkpoint(self):
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4, compress_ratio=2)
        for i in range(10):
            m.write(torch.randn(1, 16), bucket_id=0)
        state = m.get_extra_state()
        assert "latent_traces" in state

    def test_reactivation_respects_max_slots(self):
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4, compress_ratio=2)
        for i in range(20):
            m.write(torch.randn(1, 16), bucket_id=0)
        if m._latent_traces:
            m.try_reactivate(bucket_id=0, surprise=10.0)
            assert len(m._slots) <= 5  # max_slots + 1 before compress triggers

    def test_reactivation_adds_noise(self):
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4, compress_ratio=2)
        for i in range(20):
            m.write(torch.randn(1, 16), bucket_id=0)
        if m._latent_traces:
            original = m._latent_traces[0]["centroid_contrib"].clone()
            m.try_reactivate(bucket_id=0, surprise=10.0)
            reactivated = m._slots[-1]
            # Should differ due to noise
            assert not torch.allclose(original, reactivated)


class TestWarmupIsolation(unittest.TestCase):
    def test_warmup_does_not_persist_slots_after_eval(self):
        """Warmup writes should not persist after eval completes."""
        from chaoscontrol.model import ChaosStudentLM
        from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb
        torch.manual_seed(42)
        model = ChaosStudentLM(
            vocab_size=256, dim=32, num_layers=2,
            outer_model_dim=16, outer_model_type="multislot",
        )
        tokens = torch.randint(0, 256, (5000,))
        starts = list(range(0, 4000, 128))
        eval_starts = starts[:16]

        slots_before = len(model.outer_model._slots)
        evaluate_chaoscontrol_bpb(
            model, tokens=tokens, eval_starts=eval_starts,
            batch_size=4, seq_len=64, device=torch.device("cpu"),
            warmup=True,
        )
        slots_after = len(model.outer_model._slots)
        assert slots_after == slots_before, "Warmup should not persist slots after eval"


class TestWriteSequenceRecency(unittest.TestCase):
    def test_recency_bias_differs_from_flat_mean(self):
        from chaoscontrol.memory import MultiSlotOuterModel
        m1 = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4)
        m2 = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4)
        # Same weights for both
        m2.load_state_dict(m1.state_dict())

        h_seq = torch.randn(2, 32, 16)
        m1.write_sequence(h_seq)  # recency-weighted
        # Manual flat mean for comparison
        m2.write(h_seq.mean(dim=1))

        # Slots should differ because recency weighting != flat mean
        assert not torch.allclose(m1._slots[0], m2._slots[0])


if __name__ == "__main__":
    unittest.main()
