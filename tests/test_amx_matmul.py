"""AMX BF16 matmul surface for the CPU SSM controller extension."""

import re

import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_amx_bf16_matmul_surface_imports():
    assert hasattr(_ext, "amx_bf16_kernel_available")
    assert hasattr(_ext, "amx_bf16_matmul")

    assert isinstance(_ext.amx_bf16_kernel_available(), bool)


@pytest.mark.scalar_path_only
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


@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 32, 64),     # per-event matvec
        (16, 32, 64),    # 16-event batch
        (64, 64, 64),    # pretrain mid layer
        (128, 32, 64),   # pretrain in_proj
        (13, 17, 30),    # tail-sensitive: M tail, N tail, K tail all fire
        (16, 16, 32),    # canonical single-tile sanity
        (16, 16, 48),    # K > K_TILE with K-tail (exercises padding plumbing)
        (16, 16, 66),    # K > 2*K_TILE with K-tail (multi-iter K + tail)
    ],
)
def test_amx_bf16_matmul_tiled_matches_reference_when_available(
    M: int, N: int, K: int,
):
    """Hardware-gated parity for the tiled E2 kernel across real shapes.

    Skips on hosts without AMX BF16 (i.e., everywhere outside Sapphire
    Rapids / GNR pods); the local guarantee for these shapes lives in
    ``test_amx_matmul_vnni_packing.py`` against the SDM-derived
    simulator. This test is the on-pod tripwire that the C++ tiling +
    zero-pad K plumbing matches the bf16-cast matmul reference.
    """
    if not (_ext.has_amx_bf16() and _ext.amx_bf16_kernel_available()):
        pytest.skip("AMX BF16 hardware/OS state and compiled kernel are required")

    torch.manual_seed(0xC0FFEE ^ (M * 1009 + N * 31 + K))
    a = torch.randn(M, K, dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.float32).to(torch.bfloat16)

    actual = _ext.amx_bf16_matmul(a, b)
    expected = a.float().matmul(b.float())

    assert actual.shape == (M, N)
    assert actual.dtype is torch.float32
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_amx_bf16_matmul_shape_mismatch_raises_when_available():
    if not _ext.amx_bf16_kernel_available():
        pytest.skip("shape validation belongs to the compiled AMX kernel path")

    a = torch.randn(16, 31, dtype=torch.bfloat16)
    b = torch.randn(32, 16, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=re.compile("shape|K|dimension", re.I)):
        _ext.amx_bf16_matmul(a, b)
