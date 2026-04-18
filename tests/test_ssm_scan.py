"""Tests for the hand-written CUDA diag SSM scan kernel.

Scope (Phase 1B-4):
  * Forward correctness against ``_diag_recurrence_inner`` on fp32
    inputs (tight tolerance).
  * Forward correctness on bf16 inputs compared to an fp32-reference
    (the kernel accumulates in fp32; bf16-in/bf16-out matches the
    fp32 reference cast back to bf16, NOT the bf16-in/bf16-out loop).
  * Edge cases: B=1, T=1, D=1, T<chunk_size, T>submission regime.
  * Backend wiring: CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan routes the
    core dispatcher through the kernel when available, falls back to
    chunked when the extension isn't built.
  * Autograd wrapper: ``ssm_scan`` produces gradients consistent with
    the Python loop (via the autograd.Function fallback in
    ``_SSMScanForwardFn.backward``).

Skipped when:
  * No CUDA available (dev mac), or
  * Extension not built (partial pod setup).

These tests are correctness-only. Throughput bench lives alongside in
``benchmarks/`` / is exercised from the pod; we don't gate unit tests
on wall-clock.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

if not torch.cuda.is_available():  # pragma: no cover — dev-mac branch
    pytest.skip("ssm_scan tests require CUDA", allow_module_level=True)

try:
    from chaoscontrol.kernels._ssm_scan import _C as _ssm_scan_C  # noqa: F401
    from chaoscontrol.kernels._ssm_scan import ssm_scan, ssm_scan_forward
except ImportError as e:  # pragma: no cover
    pytest.skip(
        f"ssm_scan extension not built: {e!r}. "
        "Re-run `pip install -e .` on a pod with CUDA.",
        allow_module_level=True,
    )

from chaoscontrol.core import _diag_recurrence_inner


def _fp32_reference(decay: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Run the sequential recurrence in fp32 and cast back to ``update``'s dtype.

    This is the right baseline for bf16 inputs: our kernel promotes the
    state accumulator to fp32 regardless of input dtype. A bf16-in/bf16-
    out loop accumulates drift bf16 cannot express, so comparing the
    kernel to a bf16-native loop would fail for reasons unrelated to
    kernel correctness.
    """
    d32 = decay.to(torch.float32)
    u32 = update.to(torch.float32)
    y32 = _diag_recurrence_inner(d32, u32)
    return y32.to(update.dtype)


# ---------------------------------------------------------------------------
# Forward parity.
# ---------------------------------------------------------------------------


class TestForwardParity:
    def test_forward_fp32_matches_inner(self):
        torch.manual_seed(0)
        B, T, D = 3, 13, 8
        decay = (torch.rand(B, T, D, device="cuda") * 0.3 + 0.65)
        update = torch.randn(B, T, D, device="cuda") * 0.1
        y_ref = _diag_recurrence_inner(decay, update)
        y_ker = ssm_scan_forward(decay, update)
        max_diff = (y_ref - y_ker).abs().max().item()
        assert max_diff < 1e-5, f"fp32 forward diff: {max_diff:.2e}"

    def test_forward_fp32_matches_inner_submission_shape(self):
        """Cross-check at the shape we actually submit with (downscaled B for test speed)."""
        torch.manual_seed(1)
        # B=16 not 1024 — the kernel's correctness doesn't depend on B,
        # and 1024 would balloon test memory.
        B, T, D = 16, 512, 256
        decay = (torch.rand(B, T, D, device="cuda") * 0.3 + 0.65)
        update = torch.randn(B, T, D, device="cuda") * 0.1
        y_ref = _diag_recurrence_inner(decay, update)
        y_ker = ssm_scan_forward(decay, update)
        max_diff = (y_ref - y_ker).abs().max().item()
        assert max_diff < 1e-4, f"fp32 forward diff at T=512: {max_diff:.2e}"

    def test_forward_bf16_matches_fp32_reference(self):
        """Kernel's bf16 output should match an fp32-reference cast back to bf16.

        See ``_fp32_reference`` docstring for why we don't compare to a
        bf16-native loop.
        """
        torch.manual_seed(2)
        B, T, D = 4, 64, 32
        decay = (torch.rand(B, T, D, device="cuda") * 0.3 + 0.65).to(torch.bfloat16)
        update = (torch.randn(B, T, D, device="cuda") * 0.1).to(torch.bfloat16)
        y_ref = _fp32_reference(decay, update)
        y_ker = ssm_scan_forward(decay, update)
        assert y_ker.dtype == torch.bfloat16
        max_diff = (y_ref.float() - y_ker.float()).abs().max().item()
        # bf16 rounding noise at the last cast: each element has ~7-bit
        # mantissa precision, so unit-magnitude outputs can differ by ~1e-2.
        assert max_diff < 5e-3, f"bf16 forward diff vs fp32 ref: {max_diff:.2e}"

    def test_forward_fp16_matches_fp32_reference(self):
        torch.manual_seed(3)
        B, T, D = 4, 64, 32
        decay = (torch.rand(B, T, D, device="cuda") * 0.3 + 0.65).to(torch.float16)
        update = (torch.randn(B, T, D, device="cuda") * 0.1).to(torch.float16)
        y_ref = _fp32_reference(decay, update)
        y_ker = ssm_scan_forward(decay, update)
        assert y_ker.dtype == torch.float16
        max_diff = (y_ref.float() - y_ker.float()).abs().max().item()
        assert max_diff < 1e-3, f"fp16 forward diff vs fp32 ref: {max_diff:.2e}"


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


class TestForwardEdgeCases:
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 1),
            (1, 1, 8),
            (1, 8, 1),
            (1, 512, 256),   # submission shape, B=1
            (8, 1, 256),     # T=1
            (8, 16, 1),      # D=1
            (2, 64, 32),     # T smaller than chunk_size (32)
            (2, 1024, 32),   # T larger than submission regime
            (2, 300, 256),   # T not power of 2 and not multiple of 32
        ],
    )
    def test_forward_fp32_shapes(self, shape):
        torch.manual_seed(sum(shape))
        B, T, D = shape
        decay = (torch.rand(B, T, D, device="cuda") * 0.3 + 0.65)
        update = torch.randn(B, T, D, device="cuda") * 0.1
        y_ref = _diag_recurrence_inner(decay, update)
        y_ker = ssm_scan_forward(decay, update)
        assert y_ker.shape == y_ref.shape
        max_diff = (y_ref - y_ker).abs().max().item()
        assert max_diff < 1e-4, f"shape {shape} diff: {max_diff:.2e}"

    def test_rejects_non_contiguous(self):
        """Stride check should raise on a non-contiguous input."""
        B, T, D = 2, 16, 32
        decay = torch.rand(B, T, D, device="cuda")
        update = torch.randn(B, T, D, device="cuda")
        # Induce non-contiguity: transpose the T and D axes.
        decay_bad = decay.transpose(1, 2).transpose(1, 2)  # still contiguous
        # Real non-contiguity:
        decay_bad = decay[:, :, ::2]
        # Make update match shape so the first check (shape mismatch) doesn't
        # preempt the contiguity one.
        update_bad = update[:, :, ::2]
        with pytest.raises((RuntimeError, ValueError)):
            ssm_scan_forward(decay_bad, update_bad)

    def test_rejects_shape_mismatch(self):
        decay = torch.rand(2, 8, 4, device="cuda")
        update = torch.randn(2, 8, 8, device="cuda")
        with pytest.raises((RuntimeError, ValueError)):
            ssm_scan_forward(decay, update)

    def test_rejects_cpu(self):
        decay = torch.rand(2, 8, 4)
        update = torch.randn(2, 8, 4)
        with pytest.raises((RuntimeError, ValueError)):
            ssm_scan_forward(decay, update)


# ---------------------------------------------------------------------------
# Autograd wrapper.
# ---------------------------------------------------------------------------


class TestAutograd:
    def test_backward_matches_python_loop_fp32(self):
        """``ssm_scan`` backward should match the Python loop's gradients.

        The backward path in ``_SSMScanForwardFn`` re-runs the Python
        reference with grad enabled, so by construction the gradients
        agree. This is a regression pin in case we later swap in a
        kernel-level backward: the test must still pass against the
        fallback path.
        """
        torch.manual_seed(7)
        B, T, D = 2, 64, 16
        decay_base = (torch.rand(B, T, D, device="cuda") * 0.3 + 0.65)
        update_base = torch.randn(B, T, D, device="cuda") * 0.1

        d_loop = decay_base.clone().detach().requires_grad_(True)
        u_loop = update_base.clone().detach().requires_grad_(True)
        d_ker = decay_base.clone().detach().requires_grad_(True)
        u_ker = update_base.clone().detach().requires_grad_(True)

        y_loop = _diag_recurrence_inner(d_loop, u_loop)
        y_loop.pow(2).sum().backward()

        y_ker = ssm_scan(d_ker, u_ker)
        y_ker.pow(2).sum().backward()

        max_decay = (d_loop.grad - d_ker.grad).abs().max().item()
        max_update = (u_loop.grad - u_ker.grad).abs().max().item()
        assert max_decay < 1e-4, f"decay grad diff: {max_decay:.2e}"
        assert max_update < 1e-4, f"update grad diff: {max_update:.2e}"

    def test_backward_handles_one_input_requires_grad(self):
        """Backward must honor requires_grad=False on one input."""
        torch.manual_seed(8)
        B, T, D = 2, 32, 8
        d_ker = (torch.rand(B, T, D, device="cuda") * 0.3 + 0.65).requires_grad_(False)
        u_ker = (torch.randn(B, T, D, device="cuda") * 0.1).requires_grad_(True)
        y = ssm_scan(d_ker, u_ker)
        y.pow(2).sum().backward()
        assert d_ker.grad is None
        assert u_ker.grad is not None
        # Sanity: match against the Python loop with only update requiring grad.
        d_ref = d_ker.detach().clone().requires_grad_(False)
        u_ref = u_ker.detach().clone().requires_grad_(True)
        _diag_recurrence_inner(d_ref, u_ref).pow(2).sum().backward()
        assert torch.allclose(u_ker.grad, u_ref.grad, atol=1e-4)


# ---------------------------------------------------------------------------
# Backend wiring in core.py.
# ---------------------------------------------------------------------------


class TestDynamoTraceable:
    """``ssm_scan_forward`` must trace through ``torch.compile`` as an
    opaque primitive. The ``torch.library.custom_op`` + ``register_fake``
    wiring in ``__init__.py`` is the mechanism; this test is the contract
    pin in case that wiring regresses silently. Matches the cuBLASLt
    precedent which has the same dynamo-traceability requirement for
    ``cublaslt_fp8_linear_fwd``.
    """

    def test_torch_compile_traces_forward(self):
        @torch.compile(dynamic=False)
        def traced(decay, update):
            return ssm_scan_forward(decay, update)

        torch.manual_seed(11)
        B, T, D = 2, 16, 8
        decay = (torch.rand(B, T, D, device="cuda") * 0.3 + 0.65).to(torch.bfloat16)
        update = (torch.randn(B, T, D, device="cuda") * 0.1).to(torch.bfloat16)
        y_traced = traced(decay, update)
        y_eager = ssm_scan_forward(decay, update)
        # Same kernel, same inputs — output should be bit-identical.
        assert torch.equal(y_traced, y_eager), (
            f"compile-traced kernel diverges from eager call at shape "
            f"{tuple(y_traced.shape)}"
        )


class TestBackendWiring:
    def test_ssm_scan_backend_selectable_via_env(self):
        """CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan routes through the kernel."""
        old_env = os.environ.get("CHAOSCONTROL_DIAG_SCAN_BACKEND")
        os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] = "ssm_scan"
        try:
            if "chaoscontrol.core" in sys.modules:
                importlib.reload(sys.modules["chaoscontrol.core"])
            import chaoscontrol.core as core

            info = core.get_diag_recurrence_backend()
            assert info["backend"] == "ssm_scan", (
                f"expected ssm_scan backend, got {info}"
            )

            torch.manual_seed(10)
            decay = (torch.rand(2, 64, 16, device="cuda") * 0.3 + 0.65)
            update = torch.randn(2, 64, 16, device="cuda") * 0.1
            y_env = core._diag_recurrence(decay, update)
            y_loop = core._diag_recurrence_inner(decay, update)
            max_diff = (y_env - y_loop).abs().max().item()
            assert max_diff < 1e-4
        finally:
            if old_env is None:
                os.environ.pop("CHAOSCONTROL_DIAG_SCAN_BACKEND", None)
            else:
                os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] = old_env
            if "chaoscontrol.core" in sys.modules:
                importlib.reload(sys.modules["chaoscontrol.core"])

    def test_ssm_scan_backend_falls_back_when_missing(self, monkeypatch):
        """If the extension import fails, backend should fall back to chunked.

        We simulate the missing-extension case by pointing the import at a
        nonexistent module and reloading core.py.
        """
        import chaoscontrol.kernels._ssm_scan as ext_pkg

        # Stash the real ssm_scan fn; replace with a module that has
        # no ssm_scan attribute (import succeeds, ImportError at
        # ``from ... import ssm_scan``).
        real_ssm_scan = getattr(ext_pkg, "ssm_scan", None)
        monkeypatch.delattr(ext_pkg, "ssm_scan", raising=False)

        old_env = os.environ.get("CHAOSCONTROL_DIAG_SCAN_BACKEND")
        os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] = "ssm_scan"
        try:
            if "chaoscontrol.core" in sys.modules:
                importlib.reload(sys.modules["chaoscontrol.core"])
            import chaoscontrol.core as core

            with pytest.warns(RuntimeWarning, match="ssm_scan"):
                info = core.get_diag_recurrence_backend()
            assert info["backend"] == "chunked", (
                f"expected chunked fallback, got {info}"
            )
        finally:
            if real_ssm_scan is not None:
                setattr(ext_pkg, "ssm_scan", real_ssm_scan)
            if old_env is None:
                os.environ.pop("CHAOSCONTROL_DIAG_SCAN_BACKEND", None)
            else:
                os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] = old_env
            if "chaoscontrol.core" in sys.modules:
                importlib.reload(sys.modules["chaoscontrol.core"])
