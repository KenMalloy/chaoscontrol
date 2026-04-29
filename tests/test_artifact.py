"""Tests for the 16MB artifact pipeline (serialize, load, eval)."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from chaoscontrol.config import ChaosControlConfig
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.memory import MultiSlotOuterModel
from chaoscontrol.artifact import serialize_artifact, load_artifact
from chaoscontrol.replay_eviction import ReplayEvictionLoop


def _make_tiny_config(**overrides) -> ChaosControlConfig:
    """Build a minimal config for testing (tiny model, fast)."""
    defaults = dict(
        data_path="/tmp/fake",
        vocab_size=256,
        model_dim=16,
        num_layers=2,
        ff_mult=2,
        seq_len=32,
        batch_size=4,
        a_mode="diag",
        rich_b_mode="none",
        outer_model_dim=0,
    )
    defaults.update(overrides)
    return ChaosControlConfig(**defaults)


def _build_tiny_model(config: ChaosControlConfig) -> ChaosStudentLM:
    """Instantiate a small model from config on CPU."""
    return ChaosStudentLM(
        vocab_size=config.vocab_size,
        dim=config.model_dim,
        num_layers=config.num_layers,
        ff_mult=config.ff_mult,
        a_mode=config.a_mode,
        rich_b_mode=config.rich_b_mode,
        outer_model_dim=config.outer_model_dim,
        outer_model_type=config.outer_model_type,
        outer_max_slots=config.outer_max_slots,
        outer_compress_ratio=config.outer_compress_ratio,
        semantic_tier_bases=config.semantic_tier_bases,
    )


def _train_tiny(model, steps=2):
    """Run a few gradient steps so weights are non-trivial."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        ids = torch.randint(0, model.vocab_size, (2, 32))
        out = model(ids)
        loss = F.cross_entropy(out["logits"].reshape(-1, model.vocab_size), ids.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()


class TestSerializeAndLoadRoundtrip(unittest.TestCase):
    """Serialize a tiny model, load it back, verify outputs are close."""

    def test_roundtrip_produces_similar_output(self):
        torch.manual_seed(42)
        config = _make_tiny_config()
        model = _build_tiny_model(config)
        _train_tiny(model, steps=2)

        # Get reference output
        model.eval()
        ids = torch.randint(0, 256, (2, 32))
        with torch.no_grad():
            ref_logits = model(ids)["logits"]

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test.artifact"
            meta = serialize_artifact(model, None, config, artifact_path)
            self.assertTrue(Path(meta["path"]).exists())

            loaded_model, loaded_tok, loaded_config = load_artifact(artifact_path, "cpu")
            self.assertIsNone(loaded_tok)

            loaded_model.eval()
            with torch.no_grad():
                loaded_logits = loaded_model(ids)["logits"]

            # Outputs should be close (int8 quantization introduces small error)
            max_diff = (ref_logits - loaded_logits).abs().max().item()
            self.assertLess(max_diff, 5.0, f"Max logit difference {max_diff} too large for int8 roundtrip")


class TestArtifactSizeUnderBudget(unittest.TestCase):
    """Verify serialized artifact fits under 16MB for a small model."""

    def test_small_model_under_16mb(self):
        torch.manual_seed(42)
        config = _make_tiny_config()
        model = _build_tiny_model(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test.artifact"
            meta = serialize_artifact(model, None, config, artifact_path)
            self.assertLessEqual(meta["size_bytes"], 16_777_216)

    def test_medium_model_under_16mb(self):
        """A slightly larger model should still fit."""
        torch.manual_seed(42)
        config = _make_tiny_config(model_dim=128, num_layers=4)
        model = _build_tiny_model(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test.artifact"
            meta = serialize_artifact(model, None, config, artifact_path)
            self.assertLessEqual(meta["size_bytes"], 16_777_216)


class TestQuantizationDegradesGracefully(unittest.TestCase):
    """Verify loaded model's loss is within 20% of original."""

    def test_loss_within_tolerance(self):
        torch.manual_seed(42)
        config = _make_tiny_config()
        model = _build_tiny_model(config)
        _train_tiny(model, steps=5)

        # Compute reference loss
        model.eval()
        ids = torch.randint(0, 256, (4, 32))
        with torch.no_grad():
            ref_out = model(ids)
            ref_loss = F.cross_entropy(
                ref_out["logits"].reshape(-1, 256), ids.reshape(-1)
            ).item()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test.artifact"
            serialize_artifact(model, None, config, artifact_path)

            loaded_model, _, _ = load_artifact(artifact_path, "cpu")
            loaded_model.eval()
            with torch.no_grad():
                loaded_out = loaded_model(ids)
                loaded_loss = F.cross_entropy(
                    loaded_out["logits"].reshape(-1, 256), ids.reshape(-1)
                ).item()

        # Allow 20% degradation
        self.assertLess(
            loaded_loss,
            ref_loss * 1.20 + 0.1,  # +0.1 absolute margin for very small losses
            f"Loaded loss {loaded_loss:.4f} exceeds 20% of ref {ref_loss:.4f}",
        )


class TestEpisodicSlotsSurviveRoundtrip(unittest.TestCase):
    """Verify episodic slots persist through serialize/load."""

    def test_slots_present_after_roundtrip(self):
        torch.manual_seed(42)
        config = _make_tiny_config(
            outer_model_dim=8,
            outer_model_type="multislot",
            outer_max_slots=16,
        )
        model = _build_tiny_model(config)

        # Write some episodic slots
        om = model.outer_model
        self.assertIsInstance(om, MultiSlotOuterModel)
        for i in range(5):
            h = torch.randn(1, config.model_dim)
            om.write(h)

        self.assertEqual(len(om._slots), 5)
        self.assertEqual(len(om._survival), 5)

        # Save reference slot values
        ref_slots = [s.clone() for s in om._slots]

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test.artifact"
            serialize_artifact(model, None, config, artifact_path)

            loaded_model, _, _ = load_artifact(artifact_path, "cpu")
            loaded_om = loaded_model.outer_model
            self.assertIsInstance(loaded_om, MultiSlotOuterModel)
            self.assertEqual(len(loaded_om._slots), 5)
            self.assertEqual(len(loaded_om._survival), 5)
            self.assertEqual(len(loaded_om._slot_buckets), 5)

            # Values should be close (quantization noise)
            for orig, loaded in zip(ref_slots, loaded_om._slots):
                max_diff = (orig - loaded).abs().max().item()
                self.assertLess(max_diff, 0.5, f"Slot difference {max_diff} too large")

    def test_slot_table_records_survive_roundtrip(self):
        torch.manual_seed(42)
        config = _make_tiny_config(
            outer_model_dim=8,
            outer_model_type="multislot",
            outer_max_slots=16,
        )
        model = _build_tiny_model(config)
        om = model.outer_model
        self.assertIsInstance(om, MultiSlotOuterModel)

        slot_id = om.table.append(
            torch.randn(1, config.outer_model_dim),
            bucket_id=3,
            event_id=17,
            step=11,
            survival=0.75,
        )
        rec = om.table.record(slot_id)
        self.assertIsNotNone(rec)
        rec.utility_ema = 0.125
        rec.marginal_gain_ema = 0.25
        rec.sharpness_ema = 0.5
        rec.score_count = 7
        rec.last_action = "REFRESH"

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test.artifact"
            serialize_artifact(model, None, config, artifact_path)
            loaded_model, _, _ = load_artifact(artifact_path, "cpu")

        loaded_om = loaded_model.outer_model
        self.assertIsInstance(loaded_om, MultiSlotOuterModel)
        loaded_rec = loaded_om.table.record(slot_id)
        self.assertIsNotNone(loaded_rec)
        self.assertEqual(loaded_rec.bucket_id, 3)
        self.assertEqual(loaded_rec.event_id, 17)
        self.assertEqual(loaded_rec.created_step, 11)
        self.assertEqual(loaded_rec.score_count, 7)
        self.assertEqual(loaded_rec.last_action, "REFRESH")
        self.assertAlmostEqual(loaded_rec.utility_ema, 0.125)
        self.assertAlmostEqual(loaded_rec.marginal_gain_ema, 0.25)
        self.assertAlmostEqual(loaded_rec.sharpness_ema, 0.5)

    def test_online_replay_eviction_state_survives_artifact_roundtrip(self):
        torch.manual_seed(42)
        config = _make_tiny_config(
            outer_model_dim=8,
            outer_model_type="multislot",
            outer_max_slots=16,
            replay_eviction_enabled=True,
        )
        model = _build_tiny_model(config)
        loop = ReplayEvictionLoop(
            refresh_candidate_count=4,
            refresh_proposal_rank=2,
            controller_state_dim=8,
            controller_rank=2,
            arm_runtime_enabled=False,
        )
        slot = torch.randn(1, config.outer_model_dim)
        candidates = loop._refresh_proposal_model.sample_k(
            outer=model.outer_model,
            slot=slot,
            context={"marginal_gain": 0.1, "utility_ema": 0.1, "step": 5},
        )
        scores = torch.tensor([0.0, 0.4, 0.1, -0.1])
        loop._refresh_proposal_model.update(
            candidates=candidates,
            scores=scores,
            accepted_index=1,
            structural_accepted=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test.artifact"
            serialize_artifact(
                model,
                None,
                config,
                artifact_path,
                replay_eviction_loop=loop,
            )
            loaded_model, _, _ = load_artifact(artifact_path, "cpu")

        online_state = loaded_model._online_eval_state
        self.assertIn("replay_eviction", online_state)
        restored = ReplayEvictionLoop(
            refresh_candidate_count=4,
            refresh_proposal_rank=2,
            controller_state_dim=8,
            controller_rank=2,
            arm_runtime_enabled=False,
        )
        restored.load_state_dict(online_state["replay_eviction"])
        diag = restored.diagnostics()["refresh_proposal_model"]
        self.assertEqual(diag["updates_total"], 1)
        self.assertEqual(diag["positive_updates_total"], 1)
        self.assertGreater(diag["controller"]["feedback_updates"], 0)

        learned = online_state["replay_eviction"]["refresh_proposal_model"][
            "learned_direction"
        ]
        self.assertIsInstance(learned, torch.Tensor)
        self.assertGreater(float(learned.norm().item()), 0.0)

    def test_semantic_tier_survives_roundtrip(self):
        torch.manual_seed(42)
        config = _make_tiny_config(semantic_tier_bases=4)
        model = _build_tiny_model(config)

        # Set some non-zero bases
        model.semantic_tier.bases = torch.randn(1, 4)
        ref_bases = model.semantic_tier.bases.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test.artifact"
            serialize_artifact(model, None, config, artifact_path)

            loaded_model, _, _ = load_artifact(artifact_path, "cpu")
            self.assertIsNotNone(loaded_model.semantic_tier)
            loaded_bases = loaded_model.semantic_tier.bases
            max_diff = (ref_bases - loaded_bases).abs().max().item()
            # fp16 roundtrip tolerance
            self.assertLess(max_diff, 0.01, f"Semantic bases diff {max_diff} too large")


if __name__ == "__main__":
    unittest.main()
