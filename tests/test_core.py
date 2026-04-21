"""Tests for ChaosSSMCore and criticality_loss from chaoscontrol.core."""
from __future__ import annotations

import math
import unittest

import torch

from chaoscontrol.core import ChaosSSMCore, RMSNorm, criticality_loss


def test_rms_norm_module_keeps_legacy_pytorch_math() -> None:
    layer = RMSNorm(4, eps=1e-5)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    out = layer(x)
    expected = (
        torch.nn.functional.rms_norm(x.float(), (x.size(-1),), eps=layer.eps)
        .to(x.dtype)
        * layer.weight
    )

    assert out.dtype == expected.dtype
    assert torch.equal(out, expected)


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


class TestChunkedDiagScan(unittest.TestCase):
    """Exp 18 Test 0: parity test for the chunked vectorized diag scan backend.

    The chunked backend must produce bit-identical-to-float32-noise output
    compared to the sequential Python loop across a range of sequence lengths,
    decay magnitudes, and chunk sizes. This is the prerequisite gate for
    Exp 18 Test 1 (throughput bench) — if parity fails, we can't use the
    chunked backend regardless of how fast it is.
    """

    def test_chunked_matches_loop_at_realistic_decay(self) -> None:
        from chaoscontrol.core import _diag_recurrence_chunked, _diag_recurrence_inner

        torch.manual_seed(42)
        # Realistic training regime: decay in [0.65, 0.95]
        decay = torch.rand(2, 512, 64) * 0.3 + 0.65
        update = torch.randn(2, 512, 64) * 0.1
        y_loop = _diag_recurrence_inner(decay, update)
        y_chunked = _diag_recurrence_chunked(decay, update, chunk_size=32)
        max_diff = (y_loop - y_chunked).abs().max().item()
        assert max_diff < 1e-4, f"chunked backend diverges from loop: {max_diff:.2e}"

    def test_chunked_handles_extreme_decay(self) -> None:
        """Even at decay=0.01 (where naive cumprod underflows), chunking stays stable."""
        from chaoscontrol.core import _diag_recurrence_chunked, _diag_recurrence_inner

        for decay_val in (0.01, 0.1, 0.5, 0.95):
            decay = torch.full((2, 512, 32), decay_val)
            update = torch.randn(2, 512, 32) * 0.1
            y_loop = _diag_recurrence_inner(decay, update)
            y_chunked = _diag_recurrence_chunked(decay, update, chunk_size=32)
            max_diff = (y_loop - y_chunked).abs().max().item()
            assert max_diff < 1e-4, (
                f"chunked diverges at decay={decay_val}: {max_diff:.2e}"
            )

    def test_chunked_handles_non_multiple_seq_lengths(self) -> None:
        """Chunked scan must pad correctly for T not divisible by chunk_size."""
        from chaoscontrol.core import _diag_recurrence_chunked, _diag_recurrence_inner

        for T in (7, 100, 257, 513, 1023):
            torch.manual_seed(T)
            decay = torch.rand(2, T, 16) * 0.3 + 0.65
            update = torch.randn(2, T, 16) * 0.1
            y_loop = _diag_recurrence_inner(decay, update)
            y_chunked = _diag_recurrence_chunked(decay, update, chunk_size=32)
            assert y_chunked.shape == y_loop.shape
            max_diff = (y_loop - y_chunked).abs().max().item()
            assert max_diff < 1e-4, f"chunked diverges at T={T}: {max_diff:.2e}"

    def test_chunked_matches_across_chunk_sizes(self) -> None:
        """Different chunk sizes must produce identical results (within noise)."""
        from chaoscontrol.core import _diag_recurrence_chunked

        torch.manual_seed(1)
        decay = torch.rand(2, 256, 32) * 0.3 + 0.65
        update = torch.randn(2, 256, 32) * 0.1
        results = [
            _diag_recurrence_chunked(decay, update, chunk_size=K)
            for K in (8, 16, 32, 64, 128)
        ]
        baseline = results[0]
        for K, result in zip((8, 16, 32, 64, 128), results):
            max_diff = (baseline - result).abs().max().item()
            assert max_diff < 1e-5, f"chunk_size={K} diverges from chunk_size=8: {max_diff:.2e}"

    def test_chunked_backend_selectable_via_env(self) -> None:
        """CHAOSCONTROL_DIAG_SCAN_BACKEND=chunked should route _diag_recurrence to chunked."""
        import importlib
        import os
        import sys

        old_env = os.environ.get("CHAOSCONTROL_DIAG_SCAN_BACKEND")
        os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] = "chunked"
        try:
            # Reload to reset cached backend resolution
            if "chaoscontrol.core" in sys.modules:
                importlib.reload(sys.modules["chaoscontrol.core"])
            import chaoscontrol.core as core

            info = core.get_diag_recurrence_backend()
            assert info["backend"] == "chunked", f"expected chunked backend, got {info}"

            torch.manual_seed(5)
            decay = torch.rand(2, 128, 16) * 0.3 + 0.65
            update = torch.randn(2, 128, 16) * 0.1
            y_env = core._diag_recurrence(decay, update)
            y_loop = core._diag_recurrence_inner(decay, update)
            max_diff = (y_env - y_loop).abs().max().item()
            assert max_diff < 1e-4
        finally:
            if old_env is None:
                os.environ.pop("CHAOSCONTROL_DIAG_SCAN_BACKEND", None)
            else:
                os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] = old_env
            # Reload again so other tests see the default backend
            if "chaoscontrol.core" in sys.modules:
                importlib.reload(sys.modules["chaoscontrol.core"])

    def test_chunked_backend_gradients_match_loop(self) -> None:
        """Exp 18 Test 1 follow-up: backward gradients must match the Python loop.

        The chunked forward pass uses cumprod / cumsum and float64 intermediates,
        which trace a different computational graph than the 512-step sequential
        muladd loop. Autograd through those different graphs could in principle
        produce different gradients. This test verifies that backward gradients
        agree to within float32 numerical noise at a realistic sequence length.

        Without this, a "faster scan" could silently train a different model
        and Test 1's throughput gain would be meaningless or harmful.
        """
        from chaoscontrol.core import _diag_recurrence_inner, _diag_recurrence_chunked

        torch.manual_seed(1)
        # Realistic scale: B=2, T=512, D=64, decay in typical training range
        decay_base = torch.rand(2, 512, 64) * 0.3 + 0.65
        update_base = torch.randn(2, 512, 64) * 0.1

        d_loop = decay_base.clone().detach().requires_grad_(True)
        u_loop = update_base.clone().detach().requires_grad_(True)
        d_chunked = decay_base.clone().detach().requires_grad_(True)
        u_chunked = update_base.clone().detach().requires_grad_(True)

        # Scalar loss depending on all positions / channels, exercising the
        # full backward graph.
        (_diag_recurrence_inner(d_loop, u_loop) ** 2).sum().backward()
        (_diag_recurrence_chunked(d_chunked, u_chunked, chunk_size=32) ** 2).sum().backward()

        decay_grad_diff = (d_loop.grad - d_chunked.grad).abs().max().item()
        update_grad_diff = (u_loop.grad - u_chunked.grad).abs().max().item()

        # Relative tolerance: diff normalized by gradient magnitude
        decay_rel = decay_grad_diff / (d_loop.grad.abs().max().item() + 1e-12)
        update_rel = update_grad_diff / (u_loop.grad.abs().max().item() + 1e-12)

        assert decay_rel < 1e-5, (
            f"decay gradient drift: abs={decay_grad_diff:.2e}, rel={decay_rel:.2e}"
        )
        assert update_rel < 1e-5, (
            f"update gradient drift: abs={update_grad_diff:.2e}, rel={update_rel:.2e}"
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
