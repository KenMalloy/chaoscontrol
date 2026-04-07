"""Tests for ChaosSSMCore and criticality_loss from chaoscontrol.core."""
from __future__ import annotations

import math
import unittest

import torch

from chaoscontrol.core import ChaosSSMCore, criticality_loss


class TestChaosSSMCore(unittest.TestCase):
    def test_diag_output_shape(self) -> None:
        core = ChaosSSMCore(dim=16, a_mode="diag")
        x = torch.randn(2, 8, 16)
        y = core(x)
        assert y.shape == (2, 8, 16)

    def test_diag_deterministic(self) -> None:
        core = ChaosSSMCore(dim=16, a_mode="diag")
        x = torch.randn(2, 8, 16)
        y1 = core(x)
        y2 = core(x)
        assert torch.allclose(y1, y2)

    def test_paired_output_shape(self) -> None:
        core = ChaosSSMCore(dim=16, a_mode="paired")
        x = torch.randn(2, 8, 16)
        y = core(x)
        assert y.shape == (2, 8, 16)

    def test_full_output_shape(self) -> None:
        core = ChaosSSMCore(dim=16, a_mode="full", a_full_rank=4)
        x = torch.randn(2, 8, 16)
        y = core(x)
        assert y.shape == (2, 8, 16)

    def test_full_jacobian_stats_returned(self) -> None:
        core = ChaosSSMCore(dim=16, a_mode="full", a_full_rank=4)
        x = torch.randn(2, 8, 16)
        y, stats = core(x, return_jacobian_stats=True)
        assert "lambda_max" in stats
        assert "sv_log_var" in stats

    def test_full_skew_symmetric_is_antisymmetric(self) -> None:
        core = ChaosSSMCore(dim=16, a_mode="full", a_full_rank=4)
        S = core._build_skew_symmetric()
        assert torch.allclose(S, -S.T, atol=1e-6)

    def test_paired_oscillates(self) -> None:
        """A paired-mode core with theta=pi/4 should oscillate, not just decay."""
        core = ChaosSSMCore(dim=16, a_mode="paired")
        with torch.no_grad():
            core.theta.fill_(math.pi / 4)
            # log_r = -10 => softplus(-10) ~ 0 => r = exp(0) ~ 1.0 (minimal decay)
            # This keeps the rotation near-unit modulus so oscillation dominates
            core.log_r.fill_(-10.0)
        # Single kick token followed by 31 zero tokens
        kick = torch.randn(1, 1, 16)
        zeros = torch.zeros(1, 31, 16)
        x = torch.cat([kick, zeros], dim=1)  # (1, 32, 16)
        y = core(x)
        # Compute output norms per timestep
        norms = y[0].norm(dim=-1)  # (32,)
        # Check non-monotonic: at least one later timestep has larger norm
        # than the previous one (proves oscillation, not just decay)
        found_increase = False
        for t in range(1, len(norms)):
            if norms[t].item() > norms[t - 1].item():
                found_increase = True
                break
        assert found_increase, (
            f"Expected non-monotonic norms (oscillation), but norms were monotonically "
            f"non-increasing: {norms.tolist()}"
        )


class TestCriticalityRegularizer(unittest.TestCase):
    def test_crit_loss_is_scalar(self) -> None:
        core = ChaosSSMCore(dim=16, a_mode="full", a_full_rank=4)
        x = torch.randn(2, 8, 16)
        _, stats = core(x, return_jacobian_stats=True)
        loss = criticality_loss(stats, alpha=0.01, beta=0.001)
        assert loss.shape == ()

    def test_crit_loss_minimal_at_target(self) -> None:
        """Loss should be minimal when lambda_max hits the subcritical target."""
        target = math.log(0.88)  # ~-0.13
        stats = {"lambda_max": torch.tensor(target), "sv_log_var": torch.tensor(0.0)}
        loss = criticality_loss(stats, alpha=0.01, beta=0.001)
        assert abs(loss.item()) < 1e-5

    def test_crit_loss_penalizes_supercritical(self) -> None:
        """Lambda_max > target should produce nonzero loss."""
        stats = {"lambda_max": torch.tensor(0.0), "sv_log_var": torch.tensor(0.0)}
        loss = criticality_loss(stats, alpha=0.01, beta=0.001)
        assert loss.item() > 1e-5  # 0.0 is above the -0.13 target


if __name__ == "__main__":
    unittest.main()
