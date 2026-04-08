"""Tests for sleep cycle integration in the training loop."""
from __future__ import annotations

import unittest

import torch

from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.training import train_chaoscontrol_for_budget


class TestTrainingWithSleep(unittest.TestCase):
    def test_training_with_sleep_runs(self) -> None:
        """Training with sleep enabled completes without error."""
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none",
            outer_model_dim=16, outer_model_type="multislot",
            outer_max_slots=64,
        )
        tokens = torch.randint(0, 256, (512,))
        starts = list(range(0, 400, 32))
        result = train_chaoscontrol_for_budget(
            model,
            train_tokens=tokens,
            train_starts=starts,
            seq_len=16,
            batch_size=2,
            device=torch.device("cpu"),
            param_dtype=torch.float32,
            budget_seconds=10.0,
            base_lr=1e-3,
            weight_decay=0.0,
            grad_clip_norm=1.0,
            seed=42,
            crit_reg_alpha=0.0,
            crit_reg_beta=0.0,
            sleep_enabled=True,
            sleep_stages="full_cycle",
            sleep_interval=5,
            sleep_budget=16,
            sleep_n2_budget=4,
            sleep_rem_budget=8,
            sleep_n2_batches=2,
            sleep_rem_dreams=1,
            sleep_rem_length=8,
            sleep_merge_sim_threshold=0.85,
            sleep_survival_floor=0.1,
        )
        assert result["steps"] > 0, "Training should complete at least one step"
        assert "sleep_cycles" in result, "Result must contain sleep_cycles key"

    def test_training_without_sleep_has_zero_cycles(self) -> None:
        """Training without sleep enabled reports zero sleep cycles."""
        model = ChaosStudentLM(
            vocab_size=256, dim=16, num_layers=2, ff_mult=2,
            a_mode="diag", rich_b_mode="none", outer_model_dim=0,
        )
        tokens = torch.randint(0, 256, (512,))
        starts = list(range(0, 400, 32))
        result = train_chaoscontrol_for_budget(
            model,
            train_tokens=tokens,
            train_starts=starts,
            seq_len=16,
            batch_size=2,
            device=torch.device("cpu"),
            param_dtype=torch.float32,
            budget_seconds=2.0,
            base_lr=1e-3,
            weight_decay=0.0,
            grad_clip_norm=1.0,
            seed=42,
            crit_reg_alpha=0.0,
            crit_reg_beta=0.0,
            sleep_enabled=False,
        )
        assert result["sleep_cycles"] == 0


if __name__ == "__main__":
    unittest.main()
