"""Native helpers for the Exp23 LM-head/loss hot path.

Current scope is deliberately narrow: ``fused_rms_norm`` replaces the
Python-visible ``F.rms_norm(...).to(dtype) * weight`` fragment immediately
before the LM head. The large projection itself still uses PyTorch/cuBLAS.

If the CUDA extension is unavailable, or if tensors are on CPU / unsupported
dtypes, the public function falls back to the exact PyTorch expression.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

_C: Any
try:
    from . import _C  # type: ignore[attr-defined]
except ImportError as e:  # pragma: no cover - dev macs and partial pod setups
    _C = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _require_ext() -> None:
    if _C is None:  # pragma: no cover - import-time failure path
        raise ImportError(
            "chaoscontrol.kernels._lm_head_loss._C is not built; rerun "
            "`pip install -e . --no-build-isolation` on a CUDA pod. "
            f"Original import error: {_IMPORT_ERROR!r}"
        )


_KERNEL_DTYPES: frozenset[torch.dtype] = frozenset({
    torch.bfloat16,
    torch.float16,
    torch.float32,
})


def _fallback_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.rms_norm(x.float(), (x.size(-1),), eps=float(eps)).to(x.dtype) * weight


def _is_kernel_eligible(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if _C is None:
        return False
    if not x.is_cuda or not weight.is_cuda:
        return False
    if not x.is_contiguous() or not weight.is_contiguous():
        return False
    if x.dtype not in _KERNEL_DTYPES:
        return False
    # Production Exp23 has model params in bf16/fp16/fp32 together.
    # Mixed x/weight dtypes stay on the PyTorch fallback until we have
    # profiler evidence that widening the kernel matters.
    if weight.dtype != x.dtype:
        return False
    return True


class _FusedRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        out, inv_rms = _C.rms_norm_forward(x, weight, float(eps))
        ctx.save_for_backward(x, weight, inv_rms)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        x, weight, inv_rms = ctx.saved_tensors
        grad_x, grad_weight = _C.rms_norm_backward(
            grad_out.contiguous(), x, weight, inv_rms
        )
        return grad_x, grad_weight, None


def fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm with native CUDA fast path and exact PyTorch fallback.

    Args:
        x: ``(..., D)`` activation tensor.
        weight: ``(D,)`` RMSNorm scale tensor.
        eps: epsilon added to the row mean-square before rsqrt.

    Returns:
        Tensor with the same shape and dtype promotion semantics as
        ``F.rms_norm(x.float(), (D,), eps).to(x.dtype) * weight``.
    """
    if _is_kernel_eligible(x, weight):
        return _FusedRMSNormFn.apply(x.contiguous(), weight.contiguous(), float(eps))
    return _fallback_rms_norm(x, weight, float(eps))


__all__ = [
    "fused_rms_norm",
]
