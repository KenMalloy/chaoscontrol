"""Local CPU-only proof that ``_tile_dpbf16ps`` requires VNNI packing of B.

The hardware-gated test in ``test_amx_matmul.py`` only fires on a real
Sapphire Rapids host. This test simulates ``_tile_dpbf16ps`` per Intel
SDM semantics on any CPU so the layout contract is pinned without
booking pod time.

Per the Intel ISE Reference (TDPBF16PS):

    dst = M x N fp32 tile
    src1 = M x K bf16 tile, conventional row-major (M rows of K bf16)
    src2 = K/2 x N tile, **VNNI-packed**: row r contains N
           pairs (b[2r, n], b[2r+1, n]) for n in [0, N)
    dst[m, n] += SUM_{k=0..K-1} src1[m, k] * src2_logical[k, n]

The kernel today (commit ``e01ab01`` ``amx_matmul.cpp:111``) packs B as
``b.transpose(0, 1).contiguous()`` which is N rows of K bf16 — the
*transpose* layout, not VNNI. This test proves that against the SDM
semantics regardless of whether the host can run AMX.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# Use ml_dtypes if available for round-trip-clean bf16 simulation; else
# fall back to torch's bf16 cast which gives the same numerical truncation.
def _to_bf16_then_fp32(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x).to(torch.bfloat16).to(torch.float32)
    return t.numpy()


def _simulate_tdpbf16ps_with_logical_b(
    a_bf16: np.ndarray,
    b_logical_bf16: np.ndarray,
) -> np.ndarray:
    # Reference path: compute C = A @ B in fp32 with a single bf16 cast on
    # each operand, matching TDPBF16PS's internal accumulation precision
    # (bf16 inputs, fp32 accumulator).
    a_fp32 = _to_bf16_then_fp32(a_bf16.astype(np.float32))
    b_fp32 = _to_bf16_then_fp32(b_logical_bf16.astype(np.float32))
    return a_fp32 @ b_fp32


def _simulate_tdpbf16ps_with_vnni_packed(
    a_bf16: np.ndarray,
    b_vnni_bf16: np.ndarray,
    *,
    M: int,
    N: int,
    K: int,
) -> np.ndarray:
    # Mirrors what the hardware does given a tile2 that's been configured
    # as K/2 rows x N pairs (each pair = 2 bf16). For each (m, n), sum
    # over the K-axis by pulling pair entries out of the VNNI buffer:
    #   src2_logical[2*r + 0, n] = vnni[r, 2*n + 0]
    #   src2_logical[2*r + 1, n] = vnni[r, 2*n + 1]
    assert b_vnni_bf16.shape == (K // 2, 2 * N), (
        f"vnni shape must be (K/2, 2N) = ({K // 2}, {2 * N}), got {b_vnni_bf16.shape}"
    )
    a_fp32 = _to_bf16_then_fp32(a_bf16.astype(np.float32))
    vnni_fp32 = _to_bf16_then_fp32(b_vnni_bf16.astype(np.float32))
    out = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            acc = 0.0
            for r in range(K // 2):
                acc += float(a_fp32[m, 2 * r + 0]) * float(vnni_fp32[r, 2 * n + 0])
                acc += float(a_fp32[m, 2 * r + 1]) * float(vnni_fp32[r, 2 * n + 1])
            out[m, n] = acc
    return out


def _simulate_tdpbf16ps_with_transposed_contiguous(
    a_bf16: np.ndarray,
    b_transposed_bf16: np.ndarray,
    *,
    M: int,
    N: int,
    K: int,
) -> np.ndarray:
    # The hardware reads tile2 byte-for-byte. With the kernel's current
    # tile config (rows=N, colsb=K*sizeof(bf16)), the buffer is N rows of
    # K bf16. But TDPBF16PS *interprets* its src2 tile as K/2 rows of 2N
    # bf16 pairs — so we reinterpret the same byte buffer under VNNI rules
    # and simulate the dot product.
    assert b_transposed_bf16.shape == (N, K), (
        f"transposed shape must be (N, K) = ({N}, {K}), got {b_transposed_bf16.shape}"
    )
    # Reinterpret the (N, K) bf16 buffer as (K/2, 2*N) — the byte count
    # matches (N*K = (K/2)*(2*N)), so the hardware would silently re-row
    # the buffer when the tile config says K/2 rows of 2N bf16.
    bytes_view = b_transposed_bf16.reshape(-1)
    # Both layouts are bf16; reshape to the VNNI shape that TDPBF16PS expects.
    vnni_view = bytes_view.reshape(K // 2, 2 * N)
    return _simulate_tdpbf16ps_with_vnni_packed(
        a_bf16, vnni_view, M=M, N=N, K=K,
    )


@pytest.mark.parametrize(
    "M, N, K",
    [
        (16, 16, 32),
        (8, 12, 16),
        (4, 8, 8),
    ],
)
def test_vnni_packed_b_matches_at_matmul(M: int, N: int, K: int):
    # A in (M, K) bf16, B in (K, N) bf16; reference is A @ B in fp32.
    rng = np.random.default_rng(2024)
    a_np = rng.standard_normal((M, K)).astype(np.float32)
    b_np = rng.standard_normal((K, N)).astype(np.float32)
    a_bf16 = _to_bf16_then_fp32(a_np)
    b_bf16 = _to_bf16_then_fp32(b_np)
    expected = _simulate_tdpbf16ps_with_logical_b(a_bf16, b_bf16)

    # Pack B in VNNI: K/2 rows, each carrying N pairs of (b[2r, n], b[2r+1, n]).
    vnni = np.empty((K // 2, 2 * N), dtype=b_bf16.dtype)
    for r in range(K // 2):
        for n in range(N):
            vnni[r, 2 * n + 0] = b_bf16[2 * r + 0, n]
            vnni[r, 2 * n + 1] = b_bf16[2 * r + 1, n]

    actual = _simulate_tdpbf16ps_with_vnni_packed(
        a_bf16, vnni, M=M, N=N, K=K,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "M, N, K",
    [
        (16, 16, 32),
        (8, 12, 16),
    ],
)
def test_transposed_contiguous_b_diverges_from_at_matmul(
    M: int, N: int, K: int,
):
    # The kernel's current packing (b.transpose(0,1).contiguous()) is
    # NOT VNNI. Feeding that buffer to TDPBF16PS produces something that
    # disagrees with A @ B by a meaningful amount on non-trivial inputs.
    # This test pins the divergence so the rewrite has a regression
    # gate to point at.
    rng = np.random.default_rng(7919)
    a_np = rng.standard_normal((M, K)).astype(np.float32)
    b_np = rng.standard_normal((K, N)).astype(np.float32)
    a_bf16 = _to_bf16_then_fp32(a_np)
    b_bf16 = _to_bf16_then_fp32(b_np)
    expected = _simulate_tdpbf16ps_with_logical_b(a_bf16, b_bf16)

    # b_transposed: (N, K) bf16 — the layout the kernel currently feeds
    # into tile2 via b.transpose(0, 1).contiguous().
    b_transposed = np.ascontiguousarray(b_bf16.T)
    assert b_transposed.shape == (N, K)

    actual = _simulate_tdpbf16ps_with_transposed_contiguous(
        a_bf16, b_transposed, M=M, N=N, K=K,
    )

    # Quantify the divergence: at non-trivial M/N/K with iid Gaussian
    # inputs, the relative L2 error is O(1) — i.e., the result is
    # essentially uncorrelated with A @ B.
    rel_l2_err = float(
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
    )
    assert rel_l2_err > 0.5, (
        f"transposed-contiguous packing produced a result within 50% of A @ B "
        f"(rel_l2_err={rel_l2_err}); the divergence is supposed to be O(1). "
        f"Either the SDM semantics changed or this simulator drifted."
    )
