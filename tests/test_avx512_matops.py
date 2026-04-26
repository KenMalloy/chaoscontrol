"""AVX-512 matops surface for the CPU SSM controller extension."""

import re

import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_avx512_matops_surface_imports():
    assert hasattr(_ext, "avx512_matops_kernel_available")
    assert hasattr(_ext, "avx512_matvec_fma_with_decay")
    assert hasattr(_ext, "avx512_axpy_fma")

    assert isinstance(_ext.avx512_matops_kernel_available(), bool)


def test_avx512_matops_unavailable_raises_clear_error():
    if _ext.avx512_matops_kernel_available():
        pytest.skip("AVX-512 matops kernel is compiled in this extension build")

    w = torch.zeros((4, 8), dtype=torch.float32)
    decay = torch.zeros(4, dtype=torch.float32)
    state = torch.zeros(4, dtype=torch.float32)
    x = torch.zeros(8, dtype=torch.float32)
    out = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=re.compile("AVX.*kernel.*unavailable", re.I)):
        _ext.avx512_matvec_fma_with_decay(w, decay, state, x, out)

    y = torch.zeros(8, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=re.compile("AVX.*kernel.*unavailable", re.I)):
        _ext.avx512_axpy_fma(0.5, x, y)


def test_avx512_matvec_fma_with_decay_matches_reference_when_available():
    if not (_ext.has_avx512f() and _ext.avx512_matops_kernel_available()):
        pytest.skip("AVX-512F hardware/OS state and compiled kernel are required")

    torch.manual_seed(0)
    n, k = 32, 64
    w = torch.randn(n, k, dtype=torch.float32)
    decay = torch.rand(n, dtype=torch.float32)
    state = torch.randn(n, dtype=torch.float32)
    x = torch.randn(k, dtype=torch.float32)
    out = torch.zeros(n, dtype=torch.float32)

    expected = decay * state + (w * x.unsqueeze(0)).sum(dim=1)

    _ext.avx512_matvec_fma_with_decay(w, decay, state, x, out)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_avx512_matvec_fma_with_decay_handles_tail_when_available():
    if not (_ext.has_avx512f() and _ext.avx512_matops_kernel_available()):
        pytest.skip("AVX-512F hardware/OS state and compiled kernel are required")

    torch.manual_seed(1)
    n, k = 7, 25  # K = 25 forces a 9-lane scalar tail (16 + 9)
    w = torch.randn(n, k, dtype=torch.float32)
    decay = torch.rand(n, dtype=torch.float32)
    state = torch.randn(n, dtype=torch.float32)
    x = torch.randn(k, dtype=torch.float32)
    out = torch.zeros(n, dtype=torch.float32)

    expected = decay * state + (w * x.unsqueeze(0)).sum(dim=1)

    _ext.avx512_matvec_fma_with_decay(w, decay, state, x, out)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_avx512_axpy_fma_matches_reference_when_available():
    if not (_ext.has_avx512f() and _ext.avx512_matops_kernel_available()):
        pytest.skip("AVX-512F hardware/OS state and compiled kernel are required")

    torch.manual_seed(2)
    k = 64
    alpha = 0.375
    x = torch.randn(k, dtype=torch.float32)
    y = torch.randn(k, dtype=torch.float32)
    expected = y + alpha * x

    _ext.avx512_axpy_fma(alpha, x, y)

    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)


def test_avx512_axpy_fma_handles_tail_when_available():
    if not (_ext.has_avx512f() and _ext.avx512_matops_kernel_available()):
        pytest.skip("AVX-512F hardware/OS state and compiled kernel are required")

    k = 25  # 16 vector lanes + 9-element scalar tail
    alpha = -0.75
    x = torch.linspace(-1.0, 1.0, k, dtype=torch.float32)
    y = torch.linspace(2.0, 3.0, k, dtype=torch.float32)
    expected = y + alpha * x

    _ext.avx512_axpy_fma(alpha, x, y)

    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)
