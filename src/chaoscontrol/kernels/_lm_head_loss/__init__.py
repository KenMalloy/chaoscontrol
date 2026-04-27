"""Native helpers for the Exp23 LM-head/loss hot path.

``fused_rms_norm`` replaces the Python-visible
``F.rms_norm(...).to(dtype) * weight`` fragment immediately before the LM
head and silently falls back to the PyTorch reference on CPU / unsupported
dtypes (no large intermediate; safe to leave silent).

``fused_linear_cross_entropy`` and the three sibling fused-CE dispatchers
DO NOT fall back silently. The fp32 reference materializes a full
``(rows, vocab)`` logits tensor and OOMs at submission regime, so the
public dispatchers ``_require_ext()`` upfront and raise ``RuntimeError`` if
no backend predicate matches the inputs. Call ``_fallback_*`` helpers
directly if you genuinely want the slow path (e.g. CPU numerical-reference
tests).
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(loss, per_token_ce)``.

    The per-token CE is returned as a detached fp32 tensor of shape
    ``(rows,)`` so the fallback path matches the CUDA kernels' contract.
    The reduced scalar uses the same ``F.cross_entropy(..., reduction=...)``
    call the old fallback used, so bitwise equality against the previous
    API is preserved. Per-token CE is computed separately via a
    ``reduction='none'`` call — this is an intentional second pass on the
    CPU fallback because the two reduction modes can differ at the
    last-ULP level and some tests pin exact equality.
    """
    flat_x = x.reshape(-1, x.size(-1))
    logits = F.linear(flat_x, weight)
    logits_f = logits.float()
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(logits_f, flat_targets, reduction=reduction)
    per_token_ce = F.cross_entropy(logits_f, flat_targets, reduction="none")
    return loss, per_token_ce.detach()


def _flat_token_weight(
    token_weight: torch.Tensor,
    *,
    rows: int,
    device: torch.device,
    op_name: str,
) -> torch.Tensor:
    if token_weight.numel() != rows:
        raise ValueError(
            f"{op_name}: token_weight must contain one value per input row, "
            f"got {token_weight.numel()} weights for {rows} rows"
        )
    if token_weight.device != device:
        raise ValueError(
            f"{op_name}: token_weight must be on the same device as x, got "
            f"{token_weight.device} and {device}"
        )
    return token_weight.reshape(-1).detach().to(dtype=torch.float32).contiguous()


def _weighted_loss_from_per_token_ce(
    per_token_ce: torch.Tensor,
    flat_token_weight: torch.Tensor,
) -> torch.Tensor:
    normalizer = flat_token_weight.sum().clamp_min(1.0)
    return (per_token_ce * flat_token_weight).sum() / normalizer


def _fallback_linear_cross_entropy_weighted(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    token_weight: torch.Tensor,
    *,
    op_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.numel() // x.size(-1)
    flat_weight = _flat_token_weight(
        token_weight,
        rows=rows,
        device=x.device,
        op_name=op_name,
    )
    flat_x = x.reshape(-1, x.size(-1))
    logits = flat_x @ weight.t()
    per_token_ce = F.cross_entropy(
        logits.float(),
        targets.reshape(-1),
        reduction="none",
    )
    loss = _weighted_loss_from_per_token_ce(per_token_ce, flat_weight)
    return loss, per_token_ce.detach()


def _fallback_rms_linear_cross_entropy(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    normed = _fallback_rms_norm(x, norm_weight, float(eps))
    return _fallback_linear_cross_entropy(
        normed,
        linear_weight,
        targets,
        reduction=reduction,
    )


def _fallback_rms_linear_cross_entropy_weighted(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    targets: torch.Tensor,
    token_weight: torch.Tensor,
    *,
    eps: float,
    op_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    normed = _fallback_rms_norm(x, norm_weight, float(eps))
    return _fallback_linear_cross_entropy_weighted(
        normed,
        linear_weight,
        targets,
        token_weight,
        op_name=op_name,
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
    elif backend == "streaming_cached":
        forward_name = "linear_ce_streaming_cached_forward"
        backward_name = "linear_ce_streaming_cached_backward"
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


def _linear_ce_weighted_backward_name(backend: str) -> str:
    if backend == "streaming":
        return "linear_ce_streaming_weighted_backward"
    if backend == "streaming_v2":
        return "linear_ce_streaming_v2_weighted_backward"
    if backend == "streaming_cached":
        return "linear_ce_streaming_cached_weighted_backward"
    return "linear_ce_weighted_backward"


def _is_linear_ce_weighted_kernel_eligible(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    token_weight: torch.Tensor,
    *,
    backend: str,
) -> bool:
    if not _is_linear_ce_kernel_eligible(x, weight, targets, backend=backend):
        return False
    if token_weight.device != x.device:
        return False
    if token_weight.dtype != torch.float32:
        return False
    if not token_weight.is_contiguous():
        return False
    if not hasattr(_C, _linear_ce_weighted_backward_name(backend)):
        return False
    return True


def _is_rms_linear_ce_kernel_eligible(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    backend: str,
) -> bool:
    if not _is_kernel_eligible(x, norm_weight):
        return False
    # Eligibility for linear CE is checked against the post-RMS tensor shape and
    # dtype; it matches x except that it is definitely contiguous.
    return _is_linear_ce_kernel_eligible(
        x,
        linear_weight,
        targets,
        backend=backend,
    )


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_x = x.reshape(-1, x.size(-1)).contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        compute_weight = weight.contiguous()
        if compute_weight.dtype != flat_x.dtype:
            compute_weight = compute_weight.to(dtype=flat_x.dtype)
        reduction_id = 0 if reduction == "mean" else 1
        loss, lse, per_token_ce = _C.linear_ce_forward(
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
        ctx.mark_non_differentiable(per_token_ce)
        return loss, per_token_ce

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_per_token_ce: torch.Tensor):  # type: ignore[override]
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_x = x.reshape(-1, x.size(-1)).contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        compute_weight = weight.contiguous()
        if compute_weight.dtype != flat_x.dtype:
            compute_weight = compute_weight.to(dtype=flat_x.dtype)
        reduction_id = 0 if reduction == "mean" else 1
        loss, lse, per_token_ce = _C.linear_ce_streaming_forward(
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
        ctx.mark_non_differentiable(per_token_ce)
        return loss, per_token_ce

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_per_token_ce: torch.Tensor):  # type: ignore[override]
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_x = x.reshape(-1, x.size(-1)).contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        compute_weight = weight.contiguous()
        if compute_weight.dtype != flat_x.dtype:
            compute_weight = compute_weight.to(dtype=flat_x.dtype)
        reduction_id = 0 if reduction == "mean" else 1
        loss, lse, per_token_ce = _C.linear_ce_streaming_v2_forward(
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
        ctx.mark_non_differentiable(per_token_ce)
        return loss, per_token_ce

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_per_token_ce: torch.Tensor):  # type: ignore[override]
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


class _StreamingCachedLinearCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        reduction: str,
        tile_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_x = x.reshape(-1, x.size(-1)).contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        compute_weight = weight.contiguous()
        if compute_weight.dtype != flat_x.dtype:
            compute_weight = compute_weight.to(dtype=flat_x.dtype)
        reduction_id = 0 if reduction == "mean" else 1
        loss, lse, per_token_ce, logits_cache = _C.linear_ce_streaming_cached_forward(
            flat_x,
            compute_weight,
            flat_targets,
            reduction_id,
            int(tile_size),
        )
        ctx.save_for_backward(
            flat_x,
            compute_weight,
            flat_targets,
            lse,
            logits_cache,
        )
        ctx.x_shape = tuple(x.shape)
        ctx.weight_dtype = weight.dtype
        ctx.reduction_id = reduction_id
        ctx.tile_size = int(tile_size)
        ctx.mark_non_differentiable(per_token_ce)
        return loss, per_token_ce

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_per_token_ce: torch.Tensor):  # type: ignore[override]
        flat_x, weight, flat_targets, lse, logits_cache = ctx.saved_tensors
        grad_x, grad_weight = _C.linear_ce_streaming_cached_backward(
            grad_loss.contiguous(),
            flat_x,
            weight,
            flat_targets,
            lse.contiguous(),
            logits_cache,
            ctx.reduction_id,
            ctx.tile_size,
        )
        if grad_weight.dtype != ctx.weight_dtype:
            grad_weight = grad_weight.to(dtype=ctx.weight_dtype)
        return grad_x.reshape(ctx.x_shape), grad_weight, None, None, None


class _StreamingCachedLinearCrossEntropyWithEntropyFn(torch.autograd.Function):
    """Streaming_cached forward that additionally emits per-token entropy.

    Forward returns ``(loss, lse, per_token_ce, per_token_entropy)``.
    ``per_token_entropy`` is fp32, shape ``[rows]``, and non-differentiable —
    it is a diagnostic from the same online-softmax accumulator that builds
    ``lse``. Backward wiring is identical to the plain streaming_cached
    Function: only ``grad_loss`` propagates through to ``grad_x`` / ``grad_weight``.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        reduction: str,
        tile_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_x = x.reshape(-1, x.size(-1)).contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        compute_weight = weight.contiguous()
        if compute_weight.dtype != flat_x.dtype:
            compute_weight = compute_weight.to(dtype=flat_x.dtype)
        reduction_id = 0 if reduction == "mean" else 1
        (
            loss,
            lse,
            per_token_ce,
            logits_cache,
            per_token_entropy,
        ) = _C.linear_ce_streaming_cached_forward_with_entropy(
            flat_x,
            compute_weight,
            flat_targets,
            reduction_id,
            int(tile_size),
        )
        ctx.save_for_backward(
            flat_x,
            compute_weight,
            flat_targets,
            lse,
            logits_cache,
        )
        ctx.x_shape = tuple(x.shape)
        ctx.weight_dtype = weight.dtype
        ctx.reduction_id = reduction_id
        ctx.tile_size = int(tile_size)
        ctx.mark_non_differentiable(lse, per_token_ce, per_token_entropy)
        return loss, lse, per_token_ce, per_token_entropy

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_loss: torch.Tensor,
        grad_lse: torch.Tensor,
        grad_per_token_ce: torch.Tensor,
        grad_per_token_entropy: torch.Tensor,
    ):
        flat_x, weight, flat_targets, lse, logits_cache = ctx.saved_tensors
        grad_x, grad_weight = _C.linear_ce_streaming_cached_backward(
            grad_loss.contiguous(),
            flat_x,
            weight,
            flat_targets,
            lse.contiguous(),
            logits_cache,
            ctx.reduction_id,
            ctx.tile_size,
        )
        if grad_weight.dtype != ctx.weight_dtype:
            grad_weight = grad_weight.to(dtype=ctx.weight_dtype)
        return grad_x.reshape(ctx.x_shape), grad_weight, None, None, None


class _WeightedLinearCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        token_weight: torch.Tensor,
        tile_size: int,
        backend: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_x = x.reshape(-1, x.size(-1)).contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        flat_token_weight = token_weight.contiguous()
        compute_weight = weight.contiguous()
        if compute_weight.dtype != flat_x.dtype:
            compute_weight = compute_weight.to(dtype=flat_x.dtype)
        reduction_id = 1

        if backend == "streaming_v2":
            _, lse, per_token_ce = _C.linear_ce_streaming_v2_forward(
                flat_x,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )
            saved = (flat_x, compute_weight, flat_targets, lse, flat_token_weight)
        elif backend == "streaming":
            _, lse, per_token_ce = _C.linear_ce_streaming_forward(
                flat_x,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )
            saved = (flat_x, compute_weight, flat_targets, lse, flat_token_weight)
        elif backend == "streaming_cached":
            _, lse, per_token_ce, logits_cache = _C.linear_ce_streaming_cached_forward(
                flat_x,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )
            saved = (
                flat_x,
                compute_weight,
                flat_targets,
                lse,
                flat_token_weight,
                logits_cache,
            )
        else:
            _, lse, per_token_ce = _C.linear_ce_forward(
                flat_x,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )
            saved = (flat_x, compute_weight, flat_targets, lse, flat_token_weight)

        normalizer = flat_token_weight.sum().clamp_min(1.0).contiguous()
        loss = (per_token_ce * flat_token_weight).sum() / normalizer
        ctx.save_for_backward(*saved, normalizer)
        ctx.x_shape = tuple(x.shape)
        ctx.weight_dtype = weight.dtype
        ctx.tile_size = int(tile_size)
        ctx.backend = backend
        ctx.has_logits_cache = backend == "streaming_cached"
        ctx.mark_non_differentiable(per_token_ce)
        return loss, per_token_ce

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_per_token_ce: torch.Tensor):  # type: ignore[override]
        saved = ctx.saved_tensors
        if ctx.has_logits_cache:
            (
                flat_x,
                weight,
                flat_targets,
                lse,
                flat_token_weight,
                logits_cache,
                normalizer,
            ) = saved
            grad_x, grad_weight = _C.linear_ce_streaming_cached_weighted_backward(
                grad_loss.contiguous(),
                flat_x,
                weight,
                flat_targets,
                lse.contiguous(),
                logits_cache,
                flat_token_weight,
                normalizer,
                ctx.tile_size,
            )
        else:
            flat_x, weight, flat_targets, lse, flat_token_weight, normalizer = saved
            backward_name = _linear_ce_weighted_backward_name(ctx.backend)
            grad_x, grad_weight = getattr(_C, backward_name)(
                grad_loss.contiguous(),
                flat_x,
                weight,
                flat_targets,
                lse.contiguous(),
                flat_token_weight,
                normalizer,
                ctx.tile_size,
            )
        if grad_weight.dtype != ctx.weight_dtype:
            grad_weight = grad_weight.to(dtype=ctx.weight_dtype)
        return grad_x.reshape(ctx.x_shape), grad_weight, None, None, None, None


class _FusedRMSLinearCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        linear_weight: torch.Tensor,
        targets: torch.Tensor,
        eps: float,
        reduction: str,
        tile_size: int,
        backend: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_contig = x.contiguous()
        norm_weight_contig = norm_weight.contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        reduction_id = 0 if reduction == "mean" else 1

        normed, inv_rms = _C.rms_norm_forward(
            x_contig,
            norm_weight_contig,
            float(eps),
        )
        flat_normed = normed.reshape(-1, normed.size(-1)).contiguous()
        compute_weight = linear_weight.contiguous()
        if compute_weight.dtype != flat_normed.dtype:
            compute_weight = compute_weight.to(dtype=flat_normed.dtype)

        if backend == "streaming_v2":
            loss, lse, per_token_ce = _C.linear_ce_streaming_v2_forward(
                flat_normed,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )
        elif backend == "streaming":
            loss, lse, per_token_ce = _C.linear_ce_streaming_forward(
                flat_normed,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )
        else:
            loss, lse, per_token_ce = _C.linear_ce_forward(
                flat_normed,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )

        ctx.save_for_backward(
            x_contig,
            norm_weight_contig,
            inv_rms,
            flat_normed,
            compute_weight,
            flat_targets,
            lse,
        )
        ctx.x_shape = tuple(x.shape)
        ctx.linear_weight_dtype = linear_weight.dtype
        ctx.reduction_id = reduction_id
        ctx.tile_size = int(tile_size)
        ctx.backend = backend
        ctx.mark_non_differentiable(per_token_ce)
        return loss, per_token_ce

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_per_token_ce: torch.Tensor):  # type: ignore[override]
        (
            x,
            norm_weight,
            inv_rms,
            flat_normed,
            linear_weight,
            flat_targets,
            lse,
        ) = ctx.saved_tensors
        grad_loss_contig = grad_loss.contiguous()

        if ctx.backend == "streaming_v2":
            grad_normed, grad_linear_weight = _C.linear_ce_streaming_v2_backward(
                grad_loss_contig,
                flat_normed,
                linear_weight,
                flat_targets,
                lse.contiguous(),
                ctx.reduction_id,
                ctx.tile_size,
            )
        elif ctx.backend == "streaming":
            grad_normed, grad_linear_weight = _C.linear_ce_streaming_backward(
                grad_loss_contig,
                flat_normed,
                linear_weight,
                flat_targets,
                lse.contiguous(),
                ctx.reduction_id,
                ctx.tile_size,
            )
        else:
            grad_normed, grad_linear_weight = _C.linear_ce_backward(
                grad_loss_contig,
                flat_normed,
                linear_weight,
                flat_targets,
                lse.contiguous(),
                ctx.reduction_id,
                ctx.tile_size,
            )

        grad_hidden, grad_norm_weight = _C.rms_norm_backward(
            grad_normed.reshape(ctx.x_shape).contiguous(),
            x,
            norm_weight,
            inv_rms,
        )
        if grad_linear_weight.dtype != ctx.linear_weight_dtype:
            grad_linear_weight = grad_linear_weight.to(
                dtype=ctx.linear_weight_dtype
            )
        return (
            grad_hidden,
            grad_norm_weight,
            grad_linear_weight,
            None,
            None,
            None,
            None,
            None,
        )


class _FusedRMSWeightedLinearCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        linear_weight: torch.Tensor,
        targets: torch.Tensor,
        token_weight: torch.Tensor,
        eps: float,
        tile_size: int,
        backend: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_contig = x.contiguous()
        norm_weight_contig = norm_weight.contiguous()
        flat_targets = targets.reshape(-1).contiguous()
        flat_token_weight = token_weight.contiguous()

        normed, inv_rms = _C.rms_norm_forward(
            x_contig,
            norm_weight_contig,
            float(eps),
        )
        flat_normed = normed.reshape(-1, normed.size(-1)).contiguous()
        compute_weight = linear_weight.contiguous()
        if compute_weight.dtype != flat_normed.dtype:
            compute_weight = compute_weight.to(dtype=flat_normed.dtype)

        reduction_id = 1
        if backend == "streaming_v2":
            _, lse, per_token_ce = _C.linear_ce_streaming_v2_forward(
                flat_normed,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )
            saved = (
                x_contig,
                norm_weight_contig,
                inv_rms,
                flat_normed,
                compute_weight,
                flat_targets,
                lse,
                flat_token_weight,
            )
        elif backend == "streaming":
            _, lse, per_token_ce = _C.linear_ce_streaming_forward(
                flat_normed,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )
            saved = (
                x_contig,
                norm_weight_contig,
                inv_rms,
                flat_normed,
                compute_weight,
                flat_targets,
                lse,
                flat_token_weight,
            )
        elif backend == "streaming_cached":
            _, lse, per_token_ce, logits_cache = _C.linear_ce_streaming_cached_forward(
                flat_normed,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )
            saved = (
                x_contig,
                norm_weight_contig,
                inv_rms,
                flat_normed,
                compute_weight,
                flat_targets,
                lse,
                flat_token_weight,
                logits_cache,
            )
        else:
            _, lse, per_token_ce = _C.linear_ce_forward(
                flat_normed,
                compute_weight,
                flat_targets,
                reduction_id,
                int(tile_size),
            )
            saved = (
                x_contig,
                norm_weight_contig,
                inv_rms,
                flat_normed,
                compute_weight,
                flat_targets,
                lse,
                flat_token_weight,
            )

        normalizer = flat_token_weight.sum().clamp_min(1.0).contiguous()
        loss = (per_token_ce * flat_token_weight).sum() / normalizer
        ctx.save_for_backward(*saved, normalizer)
        ctx.x_shape = tuple(x.shape)
        ctx.linear_weight_dtype = linear_weight.dtype
        ctx.tile_size = int(tile_size)
        ctx.backend = backend
        ctx.has_logits_cache = backend == "streaming_cached"
        ctx.mark_non_differentiable(per_token_ce)
        return loss, per_token_ce

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_per_token_ce: torch.Tensor):  # type: ignore[override]
        saved = ctx.saved_tensors
        if ctx.has_logits_cache:
            (
                x,
                norm_weight,
                inv_rms,
                flat_normed,
                linear_weight,
                flat_targets,
                lse,
                flat_token_weight,
                logits_cache,
                normalizer,
            ) = saved
            grad_normed, grad_linear_weight = _C.linear_ce_streaming_cached_weighted_backward(
                grad_loss.contiguous(),
                flat_normed,
                linear_weight,
                flat_targets,
                lse.contiguous(),
                logits_cache,
                flat_token_weight,
                normalizer,
                ctx.tile_size,
            )
        else:
            (
                x,
                norm_weight,
                inv_rms,
                flat_normed,
                linear_weight,
                flat_targets,
                lse,
                flat_token_weight,
                normalizer,
            ) = saved
            backward_name = _linear_ce_weighted_backward_name(ctx.backend)
            grad_normed, grad_linear_weight = getattr(_C, backward_name)(
                grad_loss.contiguous(),
                flat_normed,
                linear_weight,
                flat_targets,
                lse.contiguous(),
                flat_token_weight,
                normalizer,
                ctx.tile_size,
            )

        grad_hidden, grad_norm_weight = _C.rms_norm_backward(
            grad_normed.reshape(ctx.x_shape).contiguous(),
            x,
            norm_weight,
            inv_rms,
        )
        if grad_linear_weight.dtype != ctx.linear_weight_dtype:
            grad_linear_weight = grad_linear_weight.to(
                dtype=ctx.linear_weight_dtype
            )
        return (
            grad_hidden,
            grad_norm_weight,
            grad_linear_weight,
            None,
            None,
            None,
            None,
            None,
        )


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


def _fused_linear_cross_entropy_dispatch(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str,
    tile_size: int,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the right backend and always return ``(loss, per_token_ce)``.

    The CUDA autograd Functions return ``(loss, per_token_ce)`` directly with
    ``per_token_ce`` marked non-differentiable inside ``forward``. The fallback
    produces the same shape on CPU/unsupported configurations. Public wrappers
    pick which outputs to expose.
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError(
            "fused_linear_cross_entropy: reduction must be 'mean' or 'sum', "
            f"got {reduction!r}"
        )
    backend_name = str(backend).strip().lower()
    if backend_name == "auto":
        backend_name = "v1"
    if backend_name not in {"v1", "streaming", "streaming_v2", "streaming_cached"}:
        raise ValueError(
            "fused_linear_cross_entropy: backend must be 'auto', 'v1', "
            f"'streaming', 'streaming_v2', or 'streaming_cached', got {backend!r}"
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
    # Cheap argument validation runs first so callers on dev macs still see
    # ValueError for bad reduction/shapes; from here on we are committing to
    # a real CUDA backend, so a missing extension is a hard failure.
    _require_ext()
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
    if (
        backend_name == "streaming_cached"
        and weight.size(0) % int(tile_size) == 0
        and _is_linear_ce_kernel_eligible(
            x, weight, targets, backend="streaming_cached"
        )
    ):
        return _StreamingCachedLinearCrossEntropyFn.apply(
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
    raise RuntimeError(
        "fused_linear_cross_entropy: no eligible kernel matched and silent "
        "fallback is disabled. backend="
        f"{backend_name!r}, x.shape={tuple(x.shape)}, x.dtype={x.dtype}, "
        f"x.device={x.device}, weight.shape={tuple(weight.shape)}, "
        f"weight.dtype={weight.dtype}, targets.dtype={targets.dtype}, "
        "tile_size="
        f"{int(tile_size)}. See _is_linear_ce_kernel_eligible (and the "
        "dispatcher in chaoscontrol/kernels/_lm_head_loss/__init__.py) "
        "for the predicates that gate each backend; if you genuinely want "
        "the fp32 PyTorch reference, call _fallback_linear_cross_entropy "
        "directly instead of routing through the public dispatcher."
    )


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
    loss, _ = _fused_linear_cross_entropy_dispatch(
        x,
        weight,
        targets,
        reduction=reduction,
        tile_size=int(tile_size),
        backend=backend,
    )
    return loss


def fused_linear_cross_entropy_with_ce(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
    tile_size: int = 1024,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as :func:`fused_linear_cross_entropy`, additionally returning the
    per-token CE as a detached ``(rows,)`` fp32 tensor.

    Per-token CE is always computed inside the kernel as ``lse -
    target_logit`` on the way to the reduced scalar; returning it to the
    caller is free. ScOpt's
    :func:`chaoscontrol.optim.scopt.scarcity_pressure_from_ce` consumes it
    shape-``(batch, seq)`` — reshape at the call site.
    """
    return _fused_linear_cross_entropy_dispatch(
        x,
        weight,
        targets,
        reduction=reduction,
        tile_size=int(tile_size),
        backend=backend,
    )


def fused_lm_head_forward_with_ce_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    tile_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused LM-head forward emitting (loss, lse, per_token_ce, per_token_entropy).

    ``per_token_entropy`` is a detached (non-differentiable) fp32 tensor of
    shape ``[B*T]`` giving ``H[softmax(logits)]`` per token — computed in the
    same tile loop as the cross-entropy with an online-softmax accumulator
    trick, so emission is near-free.

    The forward is functionally identical to
    :func:`fused_linear_cross_entropy_with_ce` for
    ``(loss, lse, per_token_ce)`` under ``backend="streaming_cached"`` and
    ``reduction="mean"``; the backward path is unchanged —
    ``per_token_entropy`` carries no gradient.

    Args:
        x: ``(B, T, D)`` or ``(rows, D)`` activation tensor on CUDA.
        weight: ``(vocab, D)`` LM-head weight on CUDA; ``vocab`` must be a
            multiple of ``tile_size``.
        target: ``(B, T)`` or ``(rows,)`` int64 class indices on CUDA.
        tile_size: vocab tile width. Default 8192 matches Exp23 submission.

    Returns:
        ``(loss, lse, per_token_ce, per_token_entropy)`` — loss is a scalar
        mean-reduced CE; the remaining three are fp32 tensors of shape
        ``[rows]``.
    """
    _require_ext()
    if not hasattr(_C, "linear_ce_streaming_cached_forward_with_entropy"):
        raise ImportError(
            "chaoscontrol.kernels._lm_head_loss._C is built but missing the "
            "`linear_ce_streaming_cached_forward_with_entropy` entrypoint. "
            "Rebuild with `pip install -e . --no-build-isolation`."
        )
    if x.size(-1) != weight.size(-1):
        raise ValueError(
            "fused_lm_head_forward_with_ce_entropy: x last dimension must "
            f"match weight input dimension, got {x.size(-1)} and "
            f"{weight.size(-1)}"
        )
    rows = x.numel() // x.size(-1)
    if target.numel() != rows:
        raise ValueError(
            "fused_lm_head_forward_with_ce_entropy: target must contain one "
            f"class index per input row, got {target.numel()} targets for "
            f"{rows} rows"
        )
    if int(tile_size) <= 0:
        raise ValueError(
            "fused_lm_head_forward_with_ce_entropy: tile_size must be "
            f"positive, got {tile_size}"
        )
    if weight.size(0) % int(tile_size) != 0:
        raise ValueError(
            "fused_lm_head_forward_with_ce_entropy: vocab must be an exact "
            f"multiple of tile_size; got vocab={weight.size(0)}, "
            f"tile_size={tile_size}"
        )
    return _StreamingCachedLinearCrossEntropyWithEntropyFn.apply(
        x,
        weight,
        target,
        "mean",
        int(tile_size),
    )


def _fused_linear_cross_entropy_weighted_dispatch(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    token_weight: torch.Tensor,
    *,
    tile_size: int,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    op_name = "fused_linear_cross_entropy_weighted"
    backend_name = str(backend).strip().lower()
    if backend_name == "auto":
        backend_name = "v1"
    if backend_name not in {"v1", "streaming", "streaming_v2", "streaming_cached"}:
        raise ValueError(
            f"{op_name}: backend must be 'auto', 'v1', 'streaming', "
            f"'streaming_v2', or 'streaming_cached', got {backend!r}"
        )
    if x.size(-1) != weight.size(-1):
        raise ValueError(
            f"{op_name}: x last dimension must match weight input dimension, "
            f"got {x.size(-1)} and {weight.size(-1)}"
        )
    rows = x.numel() // x.size(-1)
    if targets.numel() != rows:
        raise ValueError(
            f"{op_name}: targets must contain one class index per input row, "
            f"got {targets.numel()} targets for {rows} rows"
        )
    if int(tile_size) <= 0:
        raise ValueError(f"{op_name}: tile_size must be positive, got {tile_size}")
    # See unweighted dispatcher: arg-validation runs on every device; from here
    # on we need a real CUDA backend, and a missing extension is a hard fail.
    _require_ext()
    flat_token_weight = _flat_token_weight(
        token_weight,
        rows=rows,
        device=x.device,
        op_name=op_name,
    )
    if backend_name == "streaming" and _is_linear_ce_weighted_kernel_eligible(
        x, weight, targets, flat_token_weight, backend="streaming"
    ):
        return _WeightedLinearCrossEntropyFn.apply(
            x,
            weight,
            targets,
            flat_token_weight,
            int(tile_size),
            "streaming",
        )
    if backend_name == "streaming_v2" and _is_linear_ce_weighted_kernel_eligible(
        x, weight, targets, flat_token_weight, backend="streaming_v2"
    ):
        return _WeightedLinearCrossEntropyFn.apply(
            x,
            weight,
            targets,
            flat_token_weight,
            int(tile_size),
            "streaming_v2",
        )
    if (
        backend_name == "streaming_cached"
        and weight.size(0) % int(tile_size) == 0
        and _is_linear_ce_weighted_kernel_eligible(
            x, weight, targets, flat_token_weight, backend="streaming_cached"
        )
    ):
        return _WeightedLinearCrossEntropyFn.apply(
            x,
            weight,
            targets,
            flat_token_weight,
            int(tile_size),
            "streaming_cached",
        )
    if backend_name == "v1" and _is_linear_ce_weighted_kernel_eligible(
        x, weight, targets, flat_token_weight, backend="v1"
    ):
        return _WeightedLinearCrossEntropyFn.apply(
            x,
            weight,
            targets,
            flat_token_weight,
            int(tile_size),
            "v1",
        )
    raise RuntimeError(
        f"{op_name}: no eligible kernel matched and silent fallback is "
        f"disabled. backend={backend_name!r}, x.shape={tuple(x.shape)}, "
        f"x.dtype={x.dtype}, x.device={x.device}, "
        f"weight.shape={tuple(weight.shape)}, weight.dtype={weight.dtype}, "
        f"targets.dtype={targets.dtype}, "
        f"token_weight.dtype={flat_token_weight.dtype}, "
        f"tile_size={int(tile_size)}. See "
        "_is_linear_ce_weighted_kernel_eligible (and the weighted dispatcher "
        "in chaoscontrol/kernels/_lm_head_loss/__init__.py) for the gating "
        "predicates; call _fallback_linear_cross_entropy_weighted directly "
        "if you genuinely want the fp32 PyTorch reference."
    )


def fused_linear_cross_entropy_weighted(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    token_weight: torch.Tensor,
    tile_size: int = 1024,
    backend: str = "auto",
) -> torch.Tensor:
    """Linear projection plus weighted cross entropy.

    ``token_weight`` is detached, flattened with ``targets``, and used only as
    row geometry: ``sum(CE_i * w_i) / max(sum(w_i), 1)``. This is the ScOpt
    rare-event actuator; the unweighted CE APIs remain unchanged.
    """
    loss, _ = _fused_linear_cross_entropy_weighted_dispatch(
        x,
        weight,
        targets,
        token_weight,
        tile_size=int(tile_size),
        backend=backend,
    )
    return loss


def fused_linear_cross_entropy_weighted_with_ce(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    token_weight: torch.Tensor,
    tile_size: int = 1024,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted CE plus detached per-token CE, shape ``(rows,)``."""
    return _fused_linear_cross_entropy_weighted_dispatch(
        x,
        weight,
        targets,
        token_weight,
        tile_size=int(tile_size),
        backend=backend,
    )


def _fused_rms_linear_cross_entropy_dispatch(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float,
    reduction: str,
    tile_size: int,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reduction not in {"mean", "sum"}:
        raise ValueError(
            "fused_rms_linear_cross_entropy: reduction must be 'mean' or 'sum', "
            f"got {reduction!r}"
        )
    backend_name = str(backend).strip().lower()
    if backend_name == "auto":
        backend_name = "v1"
    if backend_name not in {"v1", "streaming", "streaming_v2"}:
        raise ValueError(
            "fused_rms_linear_cross_entropy: backend must be 'auto', 'v1', "
            f"'streaming', or 'streaming_v2', got {backend!r}"
        )
    if x.size(-1) != norm_weight.numel():
        raise ValueError(
            "fused_rms_linear_cross_entropy: x last dimension must match "
            f"norm_weight length, got {x.size(-1)} and {norm_weight.numel()}"
        )
    if x.size(-1) != linear_weight.size(-1):
        raise ValueError(
            "fused_rms_linear_cross_entropy: x last dimension must match "
            f"linear_weight input dimension, got {x.size(-1)} and "
            f"{linear_weight.size(-1)}"
        )
    if targets.numel() != x.numel() // x.size(-1):
        raise ValueError(
            "fused_rms_linear_cross_entropy: targets must contain one class "
            f"index per input row, got {targets.numel()} targets for "
            f"{x.numel() // x.size(-1)} rows"
        )
    if int(tile_size) <= 0:
        raise ValueError(
            "fused_rms_linear_cross_entropy: tile_size must be positive, "
            f"got {tile_size}"
        )
    # See unweighted dispatcher: arg-validation runs on every device; from here
    # on we need a real CUDA backend, and a missing extension is a hard fail.
    _require_ext()
    if _is_rms_linear_ce_kernel_eligible(
        x,
        norm_weight,
        linear_weight,
        targets,
        backend=backend_name,
    ):
        return _FusedRMSLinearCrossEntropyFn.apply(
            x,
            norm_weight,
            linear_weight,
            targets,
            float(eps),
            reduction,
            int(tile_size),
            backend_name,
        )
    raise RuntimeError(
        "fused_rms_linear_cross_entropy: no eligible kernel matched and "
        f"silent fallback is disabled. backend={backend_name!r}, "
        f"x.shape={tuple(x.shape)}, x.dtype={x.dtype}, x.device={x.device}, "
        f"norm_weight.shape={tuple(norm_weight.shape)}, "
        f"norm_weight.dtype={norm_weight.dtype}, "
        f"linear_weight.shape={tuple(linear_weight.shape)}, "
        f"linear_weight.dtype={linear_weight.dtype}, "
        f"targets.dtype={targets.dtype}, tile_size={int(tile_size)}. See "
        "_is_rms_linear_ce_kernel_eligible (and the dispatcher in "
        "chaoscontrol/kernels/_lm_head_loss/__init__.py) for the gating "
        "predicates; call _fallback_rms_linear_cross_entropy directly if "
        "you genuinely want the fp32 PyTorch reference."
    )


def fused_rms_linear_cross_entropy(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float = 1e-6,
    reduction: str = "mean",
    tile_size: int = 1024,
    backend: str = "auto",
) -> torch.Tensor:
    """RMSNorm + linear projection + cross entropy as one exact op.

    The CUDA fast path reuses the same native RMSNorm and tiled CE kernels as
    the separate calls, but presents them as one autograd node. That trims the
    Python/autograd orchestration around the Exp23 final head path and gives
    the deeper CUDA fusion work a single stable public entry point.
    """
    loss, _ = _fused_rms_linear_cross_entropy_dispatch(
        x,
        norm_weight,
        linear_weight,
        targets,
        eps=float(eps),
        reduction=reduction,
        tile_size=int(tile_size),
        backend=backend,
    )
    return loss


def fused_rms_linear_cross_entropy_with_ce(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float = 1e-6,
    reduction: str = "mean",
    tile_size: int = 1024,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as :func:`fused_rms_linear_cross_entropy`, additionally returning
    the per-token CE as a detached ``(rows,)`` fp32 tensor. See
    :func:`fused_linear_cross_entropy_with_ce` for the contract.
    """
    return _fused_rms_linear_cross_entropy_dispatch(
        x,
        norm_weight,
        linear_weight,
        targets,
        eps=float(eps),
        reduction=reduction,
        tile_size=int(tile_size),
        backend=backend,
    )


def _fused_rms_linear_cross_entropy_weighted_dispatch(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    targets: torch.Tensor,
    token_weight: torch.Tensor,
    *,
    eps: float,
    tile_size: int,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    op_name = "fused_rms_linear_cross_entropy_weighted"
    backend_name = str(backend).strip().lower()
    if backend_name == "auto":
        backend_name = "v1"
    if backend_name not in {"v1", "streaming", "streaming_v2", "streaming_cached"}:
        raise ValueError(
            f"{op_name}: backend must be 'auto', 'v1', 'streaming', "
            f"'streaming_v2', or 'streaming_cached', got {backend!r}"
        )
    if x.size(-1) != norm_weight.numel():
        raise ValueError(
            f"{op_name}: x last dimension must match norm_weight length, "
            f"got {x.size(-1)} and {norm_weight.numel()}"
        )
    if x.size(-1) != linear_weight.size(-1):
        raise ValueError(
            f"{op_name}: x last dimension must match linear_weight input "
            f"dimension, got {x.size(-1)} and {linear_weight.size(-1)}"
        )
    rows = x.numel() // x.size(-1)
    if targets.numel() != rows:
        raise ValueError(
            f"{op_name}: targets must contain one class index per input row, "
            f"got {targets.numel()} targets for {rows} rows"
        )
    if int(tile_size) <= 0:
        raise ValueError(f"{op_name}: tile_size must be positive, got {tile_size}")
    # See unweighted dispatcher: arg-validation runs on every device; from here
    # on we need a real CUDA backend, and a missing extension is a hard fail.
    _require_ext()
    flat_token_weight = _flat_token_weight(
        token_weight,
        rows=rows,
        device=x.device,
        op_name=op_name,
    )
    if (
        _is_kernel_eligible(x, norm_weight)
        and backend_name == "streaming"
        and _is_linear_ce_weighted_kernel_eligible(
            x, linear_weight, targets, flat_token_weight, backend="streaming"
        )
    ):
        return _FusedRMSWeightedLinearCrossEntropyFn.apply(
            x,
            norm_weight,
            linear_weight,
            targets,
            flat_token_weight,
            float(eps),
            int(tile_size),
            "streaming",
        )
    if (
        _is_kernel_eligible(x, norm_weight)
        and backend_name == "streaming_v2"
        and _is_linear_ce_weighted_kernel_eligible(
            x, linear_weight, targets, flat_token_weight, backend="streaming_v2"
        )
    ):
        return _FusedRMSWeightedLinearCrossEntropyFn.apply(
            x,
            norm_weight,
            linear_weight,
            targets,
            flat_token_weight,
            float(eps),
            int(tile_size),
            "streaming_v2",
        )
    if (
        _is_kernel_eligible(x, norm_weight)
        and backend_name == "streaming_cached"
        and linear_weight.size(0) % int(tile_size) == 0
        and _is_linear_ce_weighted_kernel_eligible(
            x,
            linear_weight,
            targets,
            flat_token_weight,
            backend="streaming_cached",
        )
    ):
        return _FusedRMSWeightedLinearCrossEntropyFn.apply(
            x,
            norm_weight,
            linear_weight,
            targets,
            flat_token_weight,
            float(eps),
            int(tile_size),
            "streaming_cached",
        )
    if (
        _is_kernel_eligible(x, norm_weight)
        and backend_name == "v1"
        and _is_linear_ce_weighted_kernel_eligible(
            x, linear_weight, targets, flat_token_weight, backend="v1"
        )
    ):
        return _FusedRMSWeightedLinearCrossEntropyFn.apply(
            x,
            norm_weight,
            linear_weight,
            targets,
            flat_token_weight,
            float(eps),
            int(tile_size),
            "v1",
        )
    raise RuntimeError(
        f"{op_name}: no eligible kernel matched and silent fallback is "
        f"disabled. backend={backend_name!r}, x.shape={tuple(x.shape)}, "
        f"x.dtype={x.dtype}, x.device={x.device}, "
        f"norm_weight.shape={tuple(norm_weight.shape)}, "
        f"norm_weight.dtype={norm_weight.dtype}, "
        f"linear_weight.shape={tuple(linear_weight.shape)}, "
        f"linear_weight.dtype={linear_weight.dtype}, "
        f"targets.dtype={targets.dtype}, "
        f"token_weight.dtype={flat_token_weight.dtype}, "
        f"tile_size={int(tile_size)}. See _is_kernel_eligible / "
        "_is_linear_ce_weighted_kernel_eligible (and the dispatcher in "
        "chaoscontrol/kernels/_lm_head_loss/__init__.py) for the gating "
        "predicates; call _fallback_rms_linear_cross_entropy_weighted "
        "directly if you genuinely want the fp32 PyTorch reference."
    )


def fused_rms_linear_cross_entropy_weighted(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    token_weight: torch.Tensor,
    eps: float = 1e-6,
    tile_size: int = 1024,
    backend: str = "auto",
) -> torch.Tensor:
    """RMSNorm + linear projection + weighted CE as one public op."""
    loss, _ = _fused_rms_linear_cross_entropy_weighted_dispatch(
        x,
        norm_weight,
        linear_weight,
        targets,
        token_weight,
        eps=float(eps),
        tile_size=int(tile_size),
        backend=backend,
    )
    return loss


def fused_rms_linear_cross_entropy_weighted_with_ce(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    token_weight: torch.Tensor,
    eps: float = 1e-6,
    tile_size: int = 1024,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted RMSNorm+linear CE plus detached per-token CE."""
    return _fused_rms_linear_cross_entropy_weighted_dispatch(
        x,
        norm_weight,
        linear_weight,
        targets,
        token_weight,
        eps=float(eps),
        tile_size=int(tile_size),
        backend=backend,
    )


__all__ = [
    "fused_linear_cross_entropy",
    "fused_linear_cross_entropy_with_ce",
    "fused_linear_cross_entropy_weighted",
    "fused_linear_cross_entropy_weighted_with_ce",
    "fused_lm_head_forward_with_ce_entropy",
    "fused_rms_linear_cross_entropy",
    "fused_rms_linear_cross_entropy_with_ce",
    "fused_rms_linear_cross_entropy_weighted",
    "fused_rms_linear_cross_entropy_weighted_with_ce",
    "fused_rms_norm",
]
