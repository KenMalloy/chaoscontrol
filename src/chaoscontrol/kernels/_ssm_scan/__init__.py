"""Diag SSM scan — hand-written CUDA kernel for the diagonal recurrence
``state[t] = decay[t] * state[t-1] + update[t]``.

Replaces the torch.compile path that produced a -63% regression at
submission regime. Forward is a custom CUDA kernel (see
``src/ssm_scan_fwd.cu``); backward falls back to autograd through
``_diag_recurrence_inner`` as a Phase 1 scope contract.

The kernel is registered with ``torch.library.custom_op`` so dynamo can
trace it as an opaque primitive when the caller is compiled.

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


# Register the forward as a torch custom op so dynamo treats it as an
# opaque primitive. Mirrors ``cublaslt_fp8_linear_fwd_op`` in the sibling
# _cublaslt extension.
_ssm_scan_forward_op: Any = None

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


class _SSMScanForwardFn(torch.autograd.Function):
    """Autograd wrapper for the forward kernel.

    Backward falls back to autograd on the pure-Python reference — we
    re-run the reference with grad enabled on detached copies of the
    saved inputs, and pull the gradients out via ``torch.autograd.grad``.
    This is intentionally slow; kernel-level backward is Phase 2.

    Why not dispatch through ``_diag_recurrence_chunked``: the chunked
    backend has its own cumulative-sum accumulation graph and we want
    the backward semantics of the *kernel forward* — which matches the
    sequential fp32 recurrence — not the chunked one. Running the Python
    reference in fp32 and letting autograd trace the recurrence gives
    exact sequential backward semantics.
    """

    @staticmethod
    def forward(ctx, decay: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(decay, update)
        ctx.decay_dtype = decay.dtype
        ctx.update_dtype = update.dtype
        return _ssm_scan_forward_op(decay, update)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        from chaoscontrol.core import _diag_recurrence_inner

        decay, update = ctx.saved_tensors

        # Re-run the reference with grad enabled. Fp32 for stability;
        # we cast the gradients back to the original input dtypes at
        # the return.
        d_ref = decay.detach().to(torch.float32).requires_grad_(decay.requires_grad)
        u_ref = update.detach().to(torch.float32).requires_grad_(update.requires_grad)
        with torch.enable_grad():
            y_ref = _diag_recurrence_inner(d_ref, u_ref)
        # `grad_out` arrives in the kernel output dtype (matches update);
        # promote to fp32 so grad dtypes line up with the fp32 reference.
        grad_out_f32 = grad_out.to(torch.float32)

        grads = torch.autograd.grad(
            outputs=y_ref,
            inputs=[t for t, req in [(d_ref, decay.requires_grad),
                                      (u_ref, update.requires_grad)] if req],
            grad_outputs=grad_out_f32,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        # Re-map the returned grads back to the original input slots,
        # honoring requires_grad for each.
        out: list[torch.Tensor | None] = []
        g_iter = iter(grads)
        if decay.requires_grad:
            out.append(next(g_iter).to(ctx.decay_dtype))
        else:
            out.append(None)
        if update.requires_grad:
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


def ssm_scan(
    decay: torch.Tensor,
    update: torch.Tensor,
) -> torch.Tensor:
    """Autograd-enabled diag SSM scan.

    Forward: custom CUDA kernel.
    Backward: autograd through the Python reference (fp32-promoted).

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
    "ssm_scan",
]
