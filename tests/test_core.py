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

    def test_diag_scan_matches_closed_loop_reference(self) -> None:
        """Vectorized diag path should match an explicit recurrence rollout."""
        torch.manual_seed(7)
        core = ChaosSSMCore(dim=16, a_mode="diag")
        x = torch.randn(3, 11, 16)

        y_scan = core(x)

        a_base = torch.sigmoid(core.log_a)[None, :]
        state = torch.zeros(3, 16)
        outputs = []
        for idx in range(x.shape[1]):
            inp = x[:, idx, :]
            delta = torch.nn.functional.softplus(core.delta_proj(inp)).clamp_min(1e-4)
            decay = torch.exp(-delta * a_base)
            select = torch.sigmoid(core.select_proj(inp))
            candidate = torch.tanh(core.in_proj(inp))
            update = select * candidate
            state = decay * state + update
            out = torch.sigmoid(core.gate_proj(inp)) * state
            outputs.append(core.out_proj(out))
        y_ref = torch.stack(outputs, dim=1)

        assert torch.allclose(y_scan, y_ref, atol=1e-5), f"max diff: {(y_scan - y_ref).abs().max()}"

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


class TestChaosSSMCoreStep(unittest.TestCase):
    """Test single-token stepping matches sequential forward pass."""

    def test_diag_step_matches_forward(self):
        """Stepping token-by-token should produce the same output as forward()."""
        torch.manual_seed(42)
        core = ChaosSSMCore(dim=16, a_mode="diag")
        x = torch.randn(2, 8, 16)

        # Full forward
        y_full = core(x)

        # Step-by-step
        state = torch.zeros(2, 16)
        outputs = []
        for t in range(8):
            out, state = core.step(x[:, t, :], state)
            outputs.append(out)
        y_step = torch.stack(outputs, dim=1)

        assert torch.allclose(y_full, y_step, atol=1e-5), f"max diff: {(y_full - y_step).abs().max()}"

    def test_paired_step_matches_forward(self):
        torch.manual_seed(42)
        core = ChaosSSMCore(dim=16, a_mode="paired")
        x = torch.randn(2, 8, 16)
        y_full = core(x)
        state = torch.zeros(2, 16)
        outputs = []
        for t in range(8):
            out, state = core.step(x[:, t, :], state)
            outputs.append(out)
        y_step = torch.stack(outputs, dim=1)
        assert torch.allclose(y_full, y_step, atol=1e-5)

    def test_full_step_matches_forward(self):
        torch.manual_seed(42)
        core = ChaosSSMCore(dim=16, a_mode="full", a_full_rank=4)
        x = torch.randn(2, 8, 16)
        y_full = core(x)
        state = torch.zeros(2, 16)
        outputs = []
        for t in range(8):
            out, state = core.step(x[:, t, :], state)
            outputs.append(out)
        y_step = torch.stack(outputs, dim=1)
        assert torch.allclose(y_full, y_step, atol=1e-4)

    def test_step_returns_correct_shapes(self):
        core = ChaosSSMCore(dim=16, a_mode="diag")
        inp = torch.randn(2, 16)  # (batch, dim) — single token
        state = torch.zeros(2, 16)
        out, new_state = core.step(inp, state)
        assert out.shape == (2, 16)
        assert new_state.shape == (2, 16)

    def test_diag_scan_gradient_matches_step_reference(self):
        """Diag scan backend should preserve gradients for the common bare path."""
        torch.manual_seed(123)
        core_scan = ChaosSSMCore(dim=8, a_mode="diag")
        core_ref = ChaosSSMCore(dim=8, a_mode="diag")
        core_ref.load_state_dict(core_scan.state_dict())

        x_scan = torch.randn(2, 6, 8, requires_grad=True)
        x_ref = x_scan.detach().clone().requires_grad_(True)

        y_scan = core_scan(x_scan)
        loss_scan = y_scan.pow(2).mean()
        loss_scan.backward()

        state = torch.zeros(2, 8)
        outputs = []
        for t in range(x_ref.shape[1]):
            out, state = core_ref.step(x_ref[:, t, :], state)
            outputs.append(out)
        y_ref = torch.stack(outputs, dim=1)
        loss_ref = y_ref.pow(2).mean()
        loss_ref.backward()

        assert torch.allclose(x_scan.grad, x_ref.grad, atol=1e-5), \
            f"input grad max diff: {(x_scan.grad - x_ref.grad).abs().max()}"
        for name, p_scan in core_scan.named_parameters():
            p_ref = dict(core_ref.named_parameters())[name]
            assert p_scan.grad is not None
            assert p_ref.grad is not None
            assert torch.allclose(p_scan.grad, p_ref.grad, atol=1e-5), \
                f"{name} grad max diff: {(p_scan.grad - p_ref.grad).abs().max()}"


if __name__ == "__main__":
    unittest.main()
