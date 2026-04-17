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
    _check_te_available,
    autocast_context,
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
