"""Native helpers for the Exp23 LM-head/loss hot path.

``fused_rms_norm`` replaces the Python-visible
``F.rms_norm(...).to(dtype) * weight`` fragment immediately before the LM
head. ``fused_linear_cross_entropy`` is the projection+CE hook; until the
native CUDA extension is built and eligible it falls back to the exact PyTorch
``linear -> cross_entropy`` expression.

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


def _fallback_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str,
) -> torch.Tensor:
    flat_x = x.reshape(-1, x.size(-1))
    logits = F.linear(flat_x, weight)
    return F.cross_entropy(
        logits.float(),
        targets.reshape(-1),
        reduction=reduction,
    )


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


def _is_linear_ce_kernel_eligible(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    backend: str = "v1",
) -> bool:
    if _C is None:
        return False
    if not x.is_cuda or not weight.is_cuda:
        return False
    if not x.is_contiguous() or not weight.is_contiguous():
        return False
    if x.dtype not in _KERNEL_DTYPES or weight.dtype not in _KERNEL_DTYPES:
        return False
    if weight.dtype != x.dtype and not (
        weight.dtype == torch.float32 and x.dtype in {torch.bfloat16, torch.float16}
    ):
        return False
    if backend == "streaming":
        forward_name = "linear_ce_streaming_forward"
        backward_name = "linear_ce_streaming_backward"
    elif backend == "streaming_v2":
        forward_name = "linear_ce_streaming_v2_forward"
        backward_name = "linear_ce_streaming_v2_backward"
    else:
        forward_name = "linear_ce_forward"
        backward_name = "linear_ce_backward"
    if not hasattr(_C, forward_name) or not hasattr(_C, backward_name):
        return False
    if not targets.is_cuda:
        return False
    if targets.dtype != torch.long:
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


class _FusedLinearCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        reduction: str,
        tile_size: int,
    ) -> torch.Tensor:
        flat_x = x.reshape(-1, x.size(-1)).contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        compute_weight = weight.contiguous()
        if compute_weight.dtype != flat_x.dtype:
            compute_weight = compute_weight.to(dtype=flat_x.dtype)
        reduction_id = 0 if reduction == "mean" else 1
        loss, lse = _C.linear_ce_forward(
            flat_x,
            compute_weight,
            flat_targets,
            reduction_id,
            int(tile_size),
        )
        ctx.save_for_backward(flat_x, compute_weight, flat_targets, lse)
        ctx.x_shape = tuple(x.shape)
        ctx.weight_dtype = weight.dtype
        ctx.reduction_id = reduction_id
        ctx.tile_size = int(tile_size)
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):  # type: ignore[override]
        flat_x, weight, flat_targets, lse = ctx.saved_tensors
        grad_x, grad_weight = _C.linear_ce_backward(
            grad_loss.contiguous(),
            flat_x,
            weight,
            flat_targets,
            lse.contiguous(),
            ctx.reduction_id,
            ctx.tile_size,
        )
        if grad_weight.dtype != ctx.weight_dtype:
            grad_weight = grad_weight.to(dtype=ctx.weight_dtype)
        return grad_x.reshape(ctx.x_shape), grad_weight, None, None, None


class _StreamingLinearCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        reduction: str,
        tile_size: int,
    ) -> torch.Tensor:
        flat_x = x.reshape(-1, x.size(-1)).contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        compute_weight = weight.contiguous()
        if compute_weight.dtype != flat_x.dtype:
            compute_weight = compute_weight.to(dtype=flat_x.dtype)
        reduction_id = 0 if reduction == "mean" else 1
        loss, lse = _C.linear_ce_streaming_forward(
            flat_x,
            compute_weight,
            flat_targets,
            reduction_id,
            int(tile_size),
        )
        ctx.save_for_backward(flat_x, compute_weight, flat_targets, lse)
        ctx.x_shape = tuple(x.shape)
        ctx.weight_dtype = weight.dtype
        ctx.reduction_id = reduction_id
        ctx.tile_size = int(tile_size)
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):  # type: ignore[override]
        flat_x, weight, flat_targets, lse = ctx.saved_tensors
        grad_x, grad_weight = _C.linear_ce_streaming_backward(
            grad_loss.contiguous(),
            flat_x,
            weight,
            flat_targets,
            lse.contiguous(),
            ctx.reduction_id,
            ctx.tile_size,
        )
        if grad_weight.dtype != ctx.weight_dtype:
            grad_weight = grad_weight.to(dtype=ctx.weight_dtype)
        return grad_x.reshape(ctx.x_shape), grad_weight, None, None, None


class _StreamingV2LinearCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        reduction: str,
        tile_size: int,
    ) -> torch.Tensor:
        flat_x = x.reshape(-1, x.size(-1)).contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        compute_weight = weight.contiguous()
        if compute_weight.dtype != flat_x.dtype:
            compute_weight = compute_weight.to(dtype=flat_x.dtype)
        reduction_id = 0 if reduction == "mean" else 1
        loss, lse = _C.linear_ce_streaming_v2_forward(
            flat_x,
            compute_weight,
            flat_targets,
            reduction_id,
            int(tile_size),
        )
        ctx.save_for_backward(flat_x, compute_weight, flat_targets, lse)
        ctx.x_shape = tuple(x.shape)
        ctx.weight_dtype = weight.dtype
        ctx.reduction_id = reduction_id
        ctx.tile_size = int(tile_size)
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):  # type: ignore[override]
        flat_x, weight, flat_targets, lse = ctx.saved_tensors
        grad_x, grad_weight = _C.linear_ce_streaming_v2_backward(
            grad_loss.contiguous(),
            flat_x,
            weight,
            flat_targets,
            lse.contiguous(),
            ctx.reduction_id,
            ctx.tile_size,
        )
        if grad_weight.dtype != ctx.weight_dtype:
            grad_weight = grad_weight.to(dtype=ctx.weight_dtype)
        return grad_x.reshape(ctx.x_shape), grad_weight, None, None, None


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


def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
    tile_size: int = 1024,
    backend: str = "auto",
) -> torch.Tensor:
    """Linear projection plus cross entropy without changing math.

    The fallback path is intentionally exact and may materialize full logits.
    The native CUDA path, when available, owns the memory-safe tiled
    implementation.
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError(
            "fused_linear_cross_entropy: reduction must be 'mean' or 'sum', "
            f"got {reduction!r}"
        )
    backend_name = str(backend).strip().lower()
    if backend_name == "auto":
        backend_name = "v1"
    if backend_name not in {"v1", "streaming", "streaming_v2"}:
        raise ValueError(
            "fused_linear_cross_entropy: backend must be 'auto', 'v1', "
            f"'streaming', or 'streaming_v2', got {backend!r}"
        )
    if x.size(-1) != weight.size(-1):
        raise ValueError(
            "fused_linear_cross_entropy: x last dimension must match weight "
            f"input dimension, got {x.size(-1)} and {weight.size(-1)}"
        )
    if targets.numel() != x.numel() // x.size(-1):
        raise ValueError(
            "fused_linear_cross_entropy: targets must contain one class index "
            f"per input row, got {targets.numel()} targets for "
            f"{x.numel() // x.size(-1)} rows"
        )
    if int(tile_size) <= 0:
        raise ValueError(
            f"fused_linear_cross_entropy: tile_size must be positive, got {tile_size}"
        )
    if backend_name == "streaming" and _is_linear_ce_kernel_eligible(
        x, weight, targets, backend="streaming"
    ):
        return _StreamingLinearCrossEntropyFn.apply(
            x,
            weight,
            targets,
            reduction,
            int(tile_size),
        )
    if backend_name == "streaming_v2" and _is_linear_ce_kernel_eligible(
        x, weight, targets, backend="streaming_v2"
    ):
        return _StreamingV2LinearCrossEntropyFn.apply(
            x,
            weight,
            targets,
            reduction,
            int(tile_size),
        )
    if backend_name == "v1" and _is_linear_ce_kernel_eligible(
        x, weight, targets, backend="v1"
    ):
        return _FusedLinearCrossEntropyFn.apply(
            x,
            weight,
            targets,
            reduction,
            int(tile_size),
        )
    return _fallback_linear_cross_entropy(
        x,
        weight,
        targets,
        reduction=reduction,
    )


__all__ = [
    "fused_linear_cross_entropy",
    "fused_rms_norm",
]
