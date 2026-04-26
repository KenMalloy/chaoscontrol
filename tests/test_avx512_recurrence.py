"""AVX-512 diagonal recurrence surface for the CPU SSM controller extension."""

import re

import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_avx512_diagonal_recurrence_surface_imports():
    assert hasattr(_ext, "avx512_recurrence_kernel_available")
    assert hasattr(_ext, "avx512_diagonal_recurrence")

    assert isinstance(_ext.avx512_recurrence_kernel_available(), bool)


def test_avx512_diagonal_recurrence_unavailable_raises_clear_error():
    if _ext.avx512_recurrence_kernel_available():
        pytest.skip("AVX-512 recurrence kernel is compiled in this extension build")

    decay = torch.full((4,), 0.5, dtype=torch.float32)
    x = torch.arange(4, dtype=torch.float32)
    h = torch.ones(4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=re.compile("AVX.*kernel.*unavailable", re.I)):
        _ext.avx512_diagonal_recurrence(decay, x, h)


def test_avx512_diagonal_recurrence_matches_reference_when_available():
    if not (_ext.has_avx512f() and _ext.avx512_recurrence_kernel_available()):
        pytest.skip("AVX-512F hardware/OS state and compiled kernel are required")

    torch.manual_seed(0)
    decay = torch.rand(64, dtype=torch.float32)
    x = torch.randn(64, dtype=torch.float32)
    h = torch.randn(64, dtype=torch.float32)
    expected = decay * h + x

    out = _ext.avx512_diagonal_recurrence(decay, x, h)

    assert out is None
    torch.testing.assert_close(h, expected, rtol=1e-5, atol=1e-5)


def test_avx512_diagonal_recurrence_tail_matches_reference_when_available():
    if not (_ext.has_avx512f() and _ext.avx512_recurrence_kernel_available()):
        pytest.skip("AVX-512F hardware/OS state and compiled kernel are required")

    decay = torch.linspace(0.0, 1.0, 18, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, 18, dtype=torch.float32)
    h = torch.linspace(2.0, 3.0, 18, dtype=torch.float32)
    expected = decay * h + x

    _ext.avx512_diagonal_recurrence(decay, x, h)

    torch.testing.assert_close(h, expected, rtol=1e-5, atol=1e-5)


def test_avx512_diagonal_recurrence_shape_mismatch_raises_when_available():
    if not _ext.avx512_recurrence_kernel_available():
        pytest.skip("shape validation belongs to the compiled AVX-512 kernel path")

    decay = torch.ones(4, dtype=torch.float32)
    x = torch.ones(5, dtype=torch.float32)
    h = torch.ones(4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=re.compile("shape|same", re.I)):
        _ext.avx512_diagonal_recurrence(decay, x, h)
