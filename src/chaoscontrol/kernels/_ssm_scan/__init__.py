"""Diag SSM scan — hand-written CUDA kernels for the diagonal recurrence
``state[t] = decay[t] * state[t-1] + update[t]``.

Replaces the torch.compile path that produced a -63% regression at
submission regime. Forward is ``src/ssm_scan_fwd.cu`` and backward is
``src/ssm_scan_bwd.cu`` — both per-thread serial scans on (b, d) lanes
with fp32 accumulator. Save-for-backward stores (decay, state) where
state is the forward output; the backward kernel needs state[t-1] for
grad_decay.

On dev macs (``_C is None``), the autograd.Function falls back to
re-running the fp32 Python reference with grad enabled and routing
through ``torch.autograd.grad``.

Both kernels are registered with ``torch.library.custom_op`` so dynamo
can trace them as opaque primitives when the caller is compiled.

Attribution: informed by FLA (MIT) and Mamba's selective_scan (Apache 2.0)
— see ``NOTICE`` in this directory. No source files copied.
"""
from __future__ import annotations

from typing import Any

import torch

_C: Any
try:
    from . import _C  # type: ignore[attr-defined]
except ImportError as e:  # pragma: no cover — dev macs and partial pod setups
    _C = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _require_ext() -> None:
    if _C is None:  # pragma: no cover — import-time failure path
        raise ImportError(
            "chaoscontrol.kernels._ssm_scan._C is not built; rerun "
            "`pip install -e .` on a pod with the CUDA toolchain. "
            f"Original import error: {_IMPORT_ERROR!r}"
        )


# Register forward and backward as torch custom ops so dynamo treats
# them as opaque primitives. Mirrors ``cublaslt_fp8_linear_fwd_op`` in
# the sibling _cublaslt extension.
_ssm_scan_forward_op: Any = None
_ssm_scan_backward_op: Any = None

if _C is not None:  # pragma: no cover — only on pods with the extension
    @torch.library.custom_op(
        "chaoscontrol_ssm_scan::forward",
        mutates_args=(),
    )
    def _ssm_scan_forward_op(  # type: ignore[no-redef]
        decay: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        return _C.ssm_scan_forward(decay, update)

    @_ssm_scan_forward_op.register_fake
    def _ssm_scan_forward_fake(
        decay: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        # Shape + dtype contract: output matches `update`.
        return torch.empty_like(update)

    @torch.library.custom_op(
        "chaoscontrol_ssm_scan::backward",
        mutates_args=(),
    )
    def _ssm_scan_backward_op(  # type: ignore[no-redef]
        grad_state: torch.Tensor,
        decay: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _C.ssm_scan_backward(grad_state, decay, state)

    @_ssm_scan_backward_op.register_fake
    def _ssm_scan_backward_fake(
        grad_state: torch.Tensor,
        decay: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # grad_decay has decay dtype; grad_update has state (=update) dtype.
        return torch.empty_like(decay), torch.empty_like(state)


class _SSMScanForwardFn(torch.autograd.Function):
    """Autograd wrapper around the forward+backward CUDA kernels.

    Save-for-backward stores ``(decay, state)`` where ``state`` is the
    forward output. The backward kernel consumes ``state[t-1]`` for
    ``grad_decay``, so saving the output is sufficient — no need to
    keep ``update`` around.

    When the C extension isn't built (``_C is None`` — dev macs), the
    backward path falls back to re-running the fp32 Python reference
    with autograd enabled. That path still saves ``(decay, update)``
    because it has to retrace the forward to produce gradients.
    """

    @staticmethod
    def forward(ctx, decay: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        state = _ssm_scan_forward_op(decay, update)
        if _C is not None:
            # Kernel backward only needs decay + state.
            ctx.save_for_backward(decay, state)
            ctx._fallback = False
        else:  # pragma: no cover — dev-mac branch
            # Python-loop fallback retraces forward, needs (decay, update).
            ctx.save_for_backward(decay, update)
            ctx._fallback = True
        ctx.decay_dtype = decay.dtype
        ctx.update_dtype = update.dtype
        ctx.decay_requires_grad = decay.requires_grad
        ctx.update_requires_grad = update.requires_grad
        return state

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if ctx._fallback:  # pragma: no cover — dev-mac branch
            return _SSMScanForwardFn._backward_python(ctx, grad_out)

        decay, state = ctx.saved_tensors
        # Kernel dispatch requires grad_state.dtype == state.dtype
        # (== update_dtype). Upstream autograd occasionally hands back
        # an fp32 grad_out even when the forward produced bf16 — the
        # .to() is a no-op when dtypes match and a safe cast otherwise.
        grad_out_cast = grad_out.to(ctx.update_dtype)
        grad_out_cast = grad_out_cast.contiguous()

        grad_decay, grad_update = _ssm_scan_backward_op(grad_out_cast, decay, state)

        out: list[torch.Tensor | None] = []
        out.append(grad_decay if ctx.decay_requires_grad else None)
        out.append(grad_update if ctx.update_requires_grad else None)
        return tuple(out)

    @staticmethod
    def _backward_python(ctx, grad_out: torch.Tensor):  # pragma: no cover — dev-mac branch
        """Dev-mac fallback: autograd through the fp32 Python reference."""
        from chaoscontrol.core import _diag_recurrence_inner

        decay, update = ctx.saved_tensors

        d_ref = decay.detach().to(torch.float32).requires_grad_(ctx.decay_requires_grad)
        u_ref = update.detach().to(torch.float32).requires_grad_(ctx.update_requires_grad)
        with torch.enable_grad():
            y_ref = _diag_recurrence_inner(d_ref, u_ref)
        grad_out_f32 = grad_out.to(torch.float32)

        grads = torch.autograd.grad(
            outputs=y_ref,
            inputs=[t for t, req in [(d_ref, ctx.decay_requires_grad),
                                      (u_ref, ctx.update_requires_grad)] if req],
            grad_outputs=grad_out_f32,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        out: list[torch.Tensor | None] = []
        g_iter = iter(grads)
        if ctx.decay_requires_grad:
            out.append(next(g_iter).to(ctx.decay_dtype))
        else:
            out.append(None)
        if ctx.update_requires_grad:
            out.append(next(g_iter).to(ctx.update_dtype))
        else:
            out.append(None)
        return tuple(out)


def ssm_scan_forward(
    decay: torch.Tensor,
    update: torch.Tensor,
) -> torch.Tensor:
    """Forward-only diag SSM scan.

    See module docstring. Runs the custom kernel; does NOT go through
    autograd. For the autograd-traceable version call
    ``ssm_scan(decay, update)``.
    """
    _require_ext()
    return _ssm_scan_forward_op(decay, update)


def ssm_scan_backward(
    grad_state: torch.Tensor,
    decay: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw backward-kernel call.

    Given ``grad_state`` (upstream dL/ds), the saved ``decay``, and the
    forward output ``state``, compute ``(grad_decay, grad_update)``.
    Does NOT go through autograd; use ``ssm_scan`` for the standard
    autograd-traceable path.
    """
    _require_ext()
    return _ssm_scan_backward_op(grad_state, decay, state)


def ssm_scan(
    decay: torch.Tensor,
    update: torch.Tensor,
) -> torch.Tensor:
    """Autograd-enabled diag SSM scan.

    Forward: custom CUDA kernel.
    Backward: custom CUDA kernel (per-lane reverse recurrence).

    On dev macs where the C extension isn't built, both paths fall back
    to pure-Python references (forward: the sequential loop; backward:
    autograd through the fp32 reference).

    Args:
        decay: (B, T, D) bf16/fp16/fp32, contiguous, row-major.
        update: (B, T, D) matching dtype.

    Returns:
        (B, T, D) state tensor; dtype matches ``update``.
    """
    _require_ext()
    return _SSMScanForwardFn.apply(decay, update)


__all__ = [
    "ssm_scan_forward",
    "ssm_scan_backward",
    "ssm_scan",
]
