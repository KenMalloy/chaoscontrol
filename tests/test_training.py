"""Tests for training loop, matrix runner, evaluation, and CLI."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from chaoscontrol.config import ChaosControlConfig
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.metabolic import metabolic_fork
from chaoscontrol.training import (
    build_chaoscontrol_matrix,
    parse_chaoscontrol_args,
    train_chaoscontrol_for_budget,
)


class TestTraining(unittest.TestCase):
    def test_train_runs(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
        )
        tokens = torch.randint(0, 256, (512,))
        starts = list(range(0, 400, 32))
        result = train_chaoscontrol_for_budget(
            model, train_tokens=tokens, train_starts=starts,
            seq_len=16, batch_size=2, device=torch.device("cpu"),
            param_dtype=torch.float32, budget_seconds=2.0,
            base_lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0,
            seed=42, crit_reg_alpha=0.0, crit_reg_beta=0.0,
        )
        assert result["steps"] > 0
        assert "history" in result

    def test_train_with_outer_model(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=32,
        )
        tokens = torch.randint(0, 256, (512,))
        starts = list(range(0, 400, 32))
        initial_state = model.outer_model.state.clone()
        result = train_chaoscontrol_for_budget(
            model, train_tokens=tokens, train_starts=starts,
            seq_len=16, batch_size=2, device=torch.device("cpu"),
            param_dtype=torch.float32, budget_seconds=2.0,
            base_lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0,
            seed=42, crit_reg_alpha=0.0, crit_reg_beta=0.0,
        )
        assert not torch.allclose(initial_state, model.outer_model.state)


class TestOptimizerSelection(unittest.TestCase):
    """CPU smoke tests that the optimizer kwarg routes to the right class.

    Each test trains a tiny model for a few steps and asserts: (a) no NaN
    in the loss trajectory, (b) ``train_result['optimizer_type']`` matches
    the expected class name so the branch taken is observable from the
    return value (not just inferred from side effects).
    """

    def _build_model(self) -> ChaosStudentLM:
        torch.manual_seed(0)
        return ChaosStudentLM(
            vocab_size=64, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
        )

    def _train(self, optimizer: str) -> dict:
        model = self._build_model()
        tokens = torch.randint(0, 64, (256,))
        starts = list(range(0, 200, 8))
        return train_chaoscontrol_for_budget(
            model, train_tokens=tokens, train_starts=starts,
            seq_len=8, batch_size=2, device=torch.device("cpu"),
            param_dtype=torch.float32, budget_seconds=1.5,
            base_lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0,
            seed=42, crit_reg_alpha=0.0, crit_reg_beta=0.0,
            optimizer=optimizer,
        )

    def _assert_trajectory_finite(self, history: list[dict]) -> None:
        losses = [float(h["loss"]) for h in history]
        self.assertTrue(len(losses) >= 1)
        for i, loss in enumerate(losses):
            self.assertFalse(
                loss != loss,  # NaN check
                msg=f"step {i}: NaN loss in optimizer trajectory",
            )
            self.assertFalse(
                loss == float("inf") or loss == float("-inf"),
                msg=f"step {i}: inf loss in optimizer trajectory",
            )

    def test_adamw_branch(self) -> None:
        result = self._train("adamw")
        self.assertEqual(result["optimizer_type"], "AdamW")
        self.assertEqual(result["optimizer_name"], "adamw")
        self._assert_trajectory_finite(result["history"])

    def test_muon_branch(self) -> None:
        result = self._train("muon")
        self.assertEqual(result["optimizer_type"], "Muon")
        self.assertEqual(result["optimizer_name"], "muon")
        self._assert_trajectory_finite(result["history"])

    def test_lamb_branch(self) -> None:
        result = self._train("lamb")
        self.assertEqual(result["optimizer_type"], "LAMB")
        self.assertEqual(result["optimizer_name"], "lamb")
        self._assert_trajectory_finite(result["history"])

    def test_unknown_optimizer_raises(self) -> None:
        model = self._build_model()
        with self.assertRaises(ValueError):
            train_chaoscontrol_for_budget(
                model, train_tokens=torch.zeros(64, dtype=torch.long),
                train_starts=[0, 8, 16, 24],
                seq_len=4, batch_size=2, device=torch.device("cpu"),
                param_dtype=torch.float32, budget_seconds=0.1,
                base_lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0,
                seed=1, optimizer="sgd",
            )


class TestMatrixRunner(unittest.TestCase):
    def test_matrix_generates_all_cells(self) -> None:
        cells = build_chaoscontrol_matrix()
        assert len(cells) == 24
        a_modes = set(c["a_mode"] for c in cells)
        assert a_modes == {"diag", "paired", "full"}
        b_modes = set(c["rich_b_mode"] for c in cells)
        assert b_modes == {"none", "nn", "hub", "assembly", "hybrid"}


class TestMetabolicGate(unittest.TestCase):
    def test_fork_produces_logits(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
        )
        ids = torch.randint(0, 256, (2, 16))
        out = metabolic_fork(model, ids, k=3, noise_std=0.1, score_mode="ensemble_agreement")
        assert out["logits"].shape == (2, 16, 256)

    def test_training_with_gate_tracks_forks(self) -> None:
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
        )
        tokens = torch.randint(0, 256, (512,))
        starts = list(range(0, 400, 32))
        result = train_chaoscontrol_for_budget(
            model, train_tokens=tokens, train_starts=starts,
            seq_len=16, batch_size=2, device=torch.device("cpu"),
            param_dtype=torch.float32, budget_seconds=2.0,
            base_lr=1e-3, weight_decay=0.0, grad_clip_norm=1.0,
            seed=42, crit_reg_alpha=0.0, crit_reg_beta=0.0,
            metabolic_gate=True, metabolic_k=2, metabolic_threshold=0.0,
            metabolic_noise_std=0.1, metabolic_score="ensemble_agreement",
        )
        assert "fork_count" in result
        # With threshold=0.0, most steps should fork
        assert result["fork_count"] > 0


class TestCLI(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = parse_chaoscontrol_args(["--data-path", "/tmp/data"])
        cfg = ChaosControlConfig(**{k.replace("-", "_"): v for k, v in vars(args).items() if k != "run_matrix"})
        assert cfg.a_mode == "diag"
        assert cfg.consolidation_trigger == "immediate"
        assert cfg.a_full_rank == 8
        assert cfg.rich_b_bottleneck == 32
        assert cfg.crit_target_coupling == 0.88

    def test_parse_args_all_knobs_reachable(self) -> None:
        args = parse_chaoscontrol_args([
            "--data-path", "/tmp/data",
            "--a-full-rank", "16",
            "--rich-b-settling-steps", "3",
            "--consolidation-trigger", "resolution",
            "--consolidation-ema-decay", "0.95",
            "--crit-target-coupling", "0.92",
        ])
        cfg = ChaosControlConfig(**{k.replace("-", "_"): v for k, v in vars(args).items() if k != "run_matrix"})
        assert cfg.a_full_rank == 16
        assert cfg.rich_b_settling_steps == 3
        assert cfg.consolidation_trigger == "resolution"
        assert cfg.consolidation_ema_decay == 0.95
        assert cfg.crit_target_coupling == 0.92


class TestWarmupTriggerStateIsolated(unittest.TestCase):
    def test_warmup_trigger_state_isolated(self) -> None:
        """Eval warmup must not leak trigger state back to the model."""
        from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb
        from chaoscontrol.memory import MultiSlotOuterModel

        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none",
            outer_model_dim=16, outer_model_type="multislot",
        )
        om = model.outer_model

        # Seed some trigger state before eval
        om._spike_seen = True
        om._steps_since_spike = 5
        om._pre_spike_loss = 1.23
        om._retrieval_weights = torch.tensor([[0.5, 0.5]])
        # Write a slot so memory is non-empty
        om.write(torch.randn(1, 16), bucket_id=0)
        initial_n_slots = len(om._slots)

        tokens = torch.randint(0, 256, (512,))
        eval_starts = list(range(0, 400, 32))

        evaluate_chaoscontrol_bpb(
            model,
            tokens=tokens,
            eval_starts=eval_starts,
            batch_size=2,
            seq_len=16,
            device=torch.device("cpu"),
            warmup=True,
            warmup_write_mode="full_sequence",
            warmup_latent=True,
            warmup_cold_start=False,
        )

        # Trigger state must be restored exactly
        assert om._spike_seen is True, "_spike_seen not restored"
        assert om._steps_since_spike == 5, "_steps_since_spike not restored"
        assert om._pre_spike_loss == 1.23, "_pre_spike_loss not restored"
        assert om._retrieval_weights is not None, "_retrieval_weights not restored"
        assert torch.allclose(om._retrieval_weights, torch.tensor([[0.5, 0.5]])), \
            "_retrieval_weights value not restored"
        # Slot count must be restored (warmup may have added slots, but they should be rolled back)
        assert len(om._slots) == initial_n_slots, \
            f"slot count changed: {initial_n_slots} -> {len(om._slots)}"

    def test_warmup_cold_start_wipes_then_restores(self) -> None:
        """Cold start should wipe memory during eval but restore it after."""
        from chaoscontrol.evaluation import evaluate_chaoscontrol_bpb

        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none",
            outer_model_dim=16, outer_model_type="multislot",
        )
        om = model.outer_model
        # Seed some slots
        om.write(torch.randn(1, 16), bucket_id=0)
        om.write(torch.randn(1, 16), bucket_id=1)
        initial_n_slots = len(om._slots)
        assert initial_n_slots == 2

        tokens = torch.randint(0, 256, (512,))
        eval_starts = list(range(0, 400, 32))

        evaluate_chaoscontrol_bpb(
            model,
            tokens=tokens,
            eval_starts=eval_starts,
            batch_size=2,
            seq_len=16,
            device=torch.device("cpu"),
            warmup=True,
            warmup_cold_start=True,
        )

        # After eval, slots must be restored to pre-eval state
        assert len(om._slots) == initial_n_slots, \
            f"cold start did not restore slots: {initial_n_slots} -> {len(om._slots)}"


class TestLatentReactivationWithoutBucket(unittest.TestCase):
    def test_latent_reactivation_without_bucket(self) -> None:
        from chaoscontrol.memory import MultiSlotOuterModel
        m = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=4, compress_ratio=2)
        for _ in range(20):
            m.write(torch.randn(1, 16), bucket_id=0)
        if m._latent_traces:
            result = m.try_reactivate(bucket_id=None, surprise=10.0)
            assert result is True  # Should match any trace


if __name__ == "__main__":
    unittest.main()
