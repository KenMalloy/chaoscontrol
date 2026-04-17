"""cuBLASLt fp8 matmul — bespoke replacement for torch._scaled_mm.

Phase 1 scope: forward-only fp8 matmul with per-tensor fp32 scales and an
optional fused bias. Semantics match ``torch._scaled_mm`` exactly so this
is a like-for-like drop-in for measuring the pure dispatch-overhead win.

Attribution: the kernel design was informed by reading NVIDIA
TransformerEngine v2.13 (Apache 2.0); no upstream source files were
copied. See ``NOTICE`` and ``LICENSE_UPSTREAM`` in this directory.
"""
from __future__ import annotations

from typing import Any

import torch

_C: Any
try:
    from . import _C  # type: ignore[attr-defined]
except ImportError as e:  # pragma: no cover — path used on dev macs only
    _C = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def cublaslt_fp8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused cuBLASLt fp8 matmul. See module docstring for semantics.

    Args:
        a: fp8 E4M3 (or E5M2), row-major, shape ``[M, K]``.
        b: fp8 E4M3 (or E5M2), column-major, shape ``[K, N]``.
        scale_a: fp32 scalar device tensor — dequant multiplier for ``a``.
        scale_b: fp32 scalar device tensor — dequant multiplier for ``b``.
        bias: optional bf16/fp16 ``[N]`` bias, fused in the cuBLASLt
            epilogue (no separate kernel launch).
        out_dtype: ``torch.bfloat16`` or ``torch.float16``.

    Returns:
        Tensor of shape ``[M, N]`` in ``out_dtype``, row-major.

    Raises:
        ImportError: the cuBLASLt extension wasn't compiled (usually because
            the current host has no CUDA toolchain — dev mac, etc.).
    """
    if _C is None:  # pragma: no cover — import-time failure path
        raise ImportError(
            "chaoscontrol.kernels._cublaslt._C is not built; rerun "
            "`pip install -e .` on a pod with the CUDA 13 toolchain. "
            f"Original import error: {_IMPORT_ERROR!r}"
        )
    return _C.cublaslt_fp8_matmul(a, b, scale_a, scale_b, bias, out_dtype)


__all__ = ["cublaslt_fp8_matmul"]
