"""AMX BF16 matmul surface for the CPU SSM controller extension."""

import re

import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_amx_bf16_matmul_surface_imports():
    assert hasattr(_ext, "amx_bf16_kernel_available")
    assert hasattr(_ext, "amx_bf16_matmul")

    assert isinstance(_ext.amx_bf16_kernel_available(), bool)


def test_amx_bf16_matmul_unavailable_raises_clear_error():
    if _ext.amx_bf16_kernel_available():
        pytest.skip("AMX BF16 kernel is compiled in this extension build")

    a = torch.randn(16, 32, dtype=torch.bfloat16)
    b = torch.randn(32, 16, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=re.compile("AMX.*kernel.*unavailable", re.I)):
        _ext.amx_bf16_matmul(a, b)


def test_amx_bf16_matmul_matches_reference_when_available():
    if not (_ext.has_amx_bf16() and _ext.amx_bf16_kernel_available()):
        pytest.skip("AMX BF16 hardware/OS state and compiled kernel are required")

    torch.manual_seed(0)
    a = torch.randn(16, 32, dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn(32, 16, dtype=torch.float32).to(torch.bfloat16)

    actual = _ext.amx_bf16_matmul(a, b)
    expected = a.float().matmul(b.float())

    assert actual.dtype is torch.float32
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)


def test_amx_bf16_matmul_shape_mismatch_raises_when_available():
    if not _ext.amx_bf16_kernel_available():
        pytest.skip("shape validation belongs to the compiled AMX kernel path")

    a = torch.randn(16, 31, dtype=torch.bfloat16)
    b = torch.randn(32, 16, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=re.compile("shape|K|dimension", re.I)):
        _ext.amx_bf16_matmul(a, b)
