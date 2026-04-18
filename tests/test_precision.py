"""Unit tests for ``chaoscontrol.precision``.

Must pass on CPU-only, TE-less environments — the module was designed
so the import itself doesn't touch transformer_engine. The fp8-enabled
path is gated behind a runtime marker that skips when TE is missing,
so pod runs exercise it automatically while local CI stays portable.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from chaoscontrol.precision import (
    _check_fused_fp8_available,
    _check_te_available,
    autocast_context,
    maybe_promote_linears_to_fused_fp8,
    maybe_promote_linears_to_te,
)


class TestAutocastContext:
    """Context-manager behavior for each supported precision dtype."""

    def test_bf16_context_is_valid_context_manager(self) -> None:
        # On CPU, autocast_context("bf16", device_type="cpu") returns a
        # nullcontext by project convention (matches
        # chaoscontrol.data.maybe_autocast: bf16 autocast is CUDA-only
        # in this codebase). The test only verifies the context is a
        # valid context manager that yields control to the caller.
        # The actual bf16 autocast behavior is only exercised on CUDA
        # pods — covered by end-to-end runner tests, not unit tests.
        with autocast_context("bf16", device_type="cpu"):
            result = torch.randn(2, 2) @ torch.randn(2, 2)
        assert result.shape == (2, 2)

    def test_fp32_context_preserves_fp32_matmul(self) -> None:
        # nullcontext: inside the block, explicit fp32 tensors stay fp32.
        with autocast_context("fp32"):
            a = torch.randn(2, 2, dtype=torch.float32)
            b = torch.randn(2, 2, dtype=torch.float32)
            result = a @ b
        assert result.dtype == torch.float32

    def test_invalid_dtype_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported precision dtype"):
            with autocast_context("int8"):
                pass

    def test_fp8_without_te_raises_runtime_error(self) -> None:
        # On TE-less environments, fp8 must fail loudly at the context
        # entry, not silently fall through to bf16 or fp32.
        if _check_te_available():
            pytest.skip("TE is available in this env — covered by fp8_with_te_works")
        with pytest.raises(RuntimeError, match="transformer_engine"):
            with autocast_context("fp8"):
                pass

    @pytest.mark.skipif(not _check_te_available(), reason="requires transformer_engine")
    def test_fp8_with_te_returns_working_context(self) -> None:
        # Only exercised on pods; lock in the contract that TE's
        # fp8_autocast is invoked and yields back to the caller.
        with autocast_context("fp8"):
            _ = torch.zeros(1)

    def test_fp8_fused_cpu_is_nullcontext(self) -> None:
        # CPU path for fp8_fused mirrors bf16's CPU behavior: fall
        # through to nullcontext so CPU-only tests don't crash. The
        # fp8-fused path is CUDA-only — the policy here is about ambient
        # activation dtype, not the fp8 cast itself.
        with autocast_context("fp8_fused", device_type="cpu"):
            result = torch.randn(2, 2) @ torch.randn(2, 2)
        assert result.shape == (2, 2)

    def test_fp8_fused_does_not_require_te(self) -> None:
        # fp8_fused must NOT raise on TE-less environments — our bespoke
        # path does not depend on transformer_engine. The runtime check
        # that gates promotion is separate (below).
        with autocast_context("fp8_fused", device_type="cpu"):
            pass


class TestMaybePromoteLinears:
    """nn.Linear -> te.Linear promotion helper contract."""

    def test_promote_disabled_is_noop(self) -> None:
        # enabled=False: model must be untouched and return 0. This is
        # the default bf16/fp32 path — promotion only happens when the
        # caller explicitly opts in for fp8.
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        n = maybe_promote_linears_to_te(model, enabled=False)
        assert n == 0
        # All original nn.Linear children still present in the walk
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 2
        # And they're still plain nn.Linear — no sneaky te.Linear subclass
        for m in linears:
            assert type(m) is nn.Linear

    def test_check_te_available_returns_bool(self) -> None:
        # Lazy probe must return a concrete bool without crashing, on
        # any machine. Caching is an internal implementation detail —
        # the contract is just "returns True or False, never raises".
        result = _check_te_available()
        assert isinstance(result, bool)
        # Repeat call should be free (cached) and idempotent.
        assert _check_te_available() is result


class TestMaybePromoteLinearsToFusedFp8:
    """nn.Linear -> FusedFP8Linear promotion helper contract.

    Same discipline as the TE version: must behave cleanly on machines
    without the extension built, and must preserve weights + bias on
    machines where it IS built.
    """

    def test_promote_disabled_is_noop(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        n = maybe_promote_linears_to_fused_fp8(model, enabled=False)
        assert n == 0
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 2
        for m in linears:
            assert type(m) is nn.Linear

    def test_promote_enabled_without_extension_warns_and_returns_zero(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # On machines without the compiled _C.so (dev mac / partial pod),
        # promotion must degrade gracefully: return 0, print a warning,
        # leave the model untouched. Same policy as
        # maybe_promote_linears_to_te when TE is unavailable.
        if _check_fused_fp8_available():
            pytest.skip(
                "extension is built in this env — covered by the "
                "enabled-with-extension test",
            )
        model = nn.Sequential(nn.Linear(4, 4))
        n = maybe_promote_linears_to_fused_fp8(model, enabled=True)
        assert n == 0
        captured = capsys.readouterr()
        assert "chaoscontrol.kernels._cublaslt._C is not built" in captured.out
        assert type(model[0]) is nn.Linear

    @pytest.mark.skipif(
        not _check_fused_fp8_available(),
        reason="requires compiled chaoscontrol.kernels._cublaslt._C",
    )
    def test_promote_enabled_with_extension_swaps_and_preserves_weights(
        self,
    ) -> None:
        # Only exercised on pods with the extension built. Locks in:
        #   * every nn.Linear child got swapped
        #   * return value matches the number swapped
        #   * weights + bias were copied over (by sampling weight norm)
        from chaoscontrol.kernels.fp8_linear import FusedFP8Linear

        # Use bf16 since FusedFP8Linear's master-weight dtype defaults
        # to bf16 and from_nn_linear preserves the source dtype.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Sequential(
            nn.Linear(4, 4, bias=True).to(device=device, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(4, 2, bias=False).to(device=device, dtype=torch.bfloat16),
        )
        pre_weights = [model[0].weight.clone(), model[2].weight.clone()]
        n = maybe_promote_linears_to_fused_fp8(model, enabled=True)
        assert n == 2
        assert isinstance(model[0], FusedFP8Linear)
        assert isinstance(model[2], FusedFP8Linear)
        assert torch.allclose(model[0].weight, pre_weights[0])
        assert torch.allclose(model[2].weight, pre_weights[1])
        # bias=False on the second linear must be preserved.
        assert model[2].bias is None

    def test_check_fused_fp8_available_returns_bool(self) -> None:
        result = _check_fused_fp8_available()
        assert isinstance(result, bool)
        assert _check_fused_fp8_available() is result
