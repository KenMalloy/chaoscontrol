"""Diag SSM scan — hand-written CUDA kernels for the diagonal recurrence
``state[t] = decay[t] * state[t-1] + update[t]``.

Replaces the torch.compile path that produced a -63% regression at
submission regime. Forward is ``src/ssm_scan_fwd.cu`` and backward is
``src/ssm_scan_bwd.cu`` — both per-thread serial scans on (b, d) lanes
with fp32 accumulator.

The forward kernel writes TWO tensors: a dtype-matching ``out`` for
downstream ops and a separate fp32 ``state_fp32`` tensor that the
backward kernel consumes. Backward differentiates through the exact
fp32 recurrence this way, not a bf16/fp16-quantized surrogate.

Save-for-backward stores ``(decay, state_fp32)`` — the fp32 state is
all backward needs for ``grad_decay = G * state[t-1]``. Memory cost vs
the previous (correctness-broken) design: one extra (B, T, D) fp32
tensor, ~500MB at submission regime.

On dev macs (``_C is None``) AND on CPU tensors (pod pre-warm probes),
the autograd.Function falls back to the fp32 Python reference — no
kernel dispatch, no CUDA requirement. ``ssm_scan(decay, update)`` is
safe to call on any device without ``_require_ext()``.

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _C.ssm_scan_forward(decay, update)

    @_ssm_scan_forward_op.register_fake
    def _ssm_scan_forward_fake(
        decay: torch.Tensor,
        update: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shape + dtype contract: out matches `update`, state_fp32 is
        # (B, T, D) fp32 regardless of input dtype.
        out = torch.empty_like(update)
        state_fp32 = torch.empty(
            decay.shape, dtype=torch.float32, device=decay.device
        )
        return out, state_fp32

    @torch.library.custom_op(
        "chaoscontrol_ssm_scan::backward",
        mutates_args=(),
    )
    def _ssm_scan_backward_op(  # type: ignore[no-redef]
        grad_state: torch.Tensor,
        decay: torch.Tensor,
        state_fp32: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _C.ssm_scan_backward(grad_state, decay, state_fp32)

    @_ssm_scan_backward_op.register_fake
    def _ssm_scan_backward_fake(
        grad_state: torch.Tensor,
        decay: torch.Tensor,
        state_fp32: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # grad_decay has decay dtype; grad_update has grad_state dtype.
        return torch.empty_like(decay), torch.empty_like(grad_state)


_KERNEL_DTYPE_COMBOS: frozenset[tuple[torch.dtype, torch.dtype]] = frozenset({
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.float16),
    (torch.float16, torch.float16),
    (torch.float32, torch.float32),
    # Autocast bf16/fp16 production path: torch.exp(-delta * a_base)
    # upcasts decay to fp32 while update stays in the autocast dtype.
    (torch.float32, torch.bfloat16),
    (torch.float32, torch.float16),
})


def _is_kernel_eligible(decay: torch.Tensor, update: torch.Tensor) -> bool:
    """Return True when the CUDA kernel can actually run for these tensors.

    False on dev macs (no ``_C`` built), on CPU tensors (pre-warm
    probes, DDP test fixtures), and on any (decay, update) dtype tuple
    outside the kernel's dispatch table. The autograd.Function routes
    ineligible inputs through the fp32 Python fallback instead of
    calling the kernel and raising a TORCH_CHECK — we don't want a
    pod-probe path or an unusual dtype combo to tear down the global
    backend cache with a bogus "ssm_scan broke" error.

    The dtype whitelist mirrors ``_KERNEL_DTYPE_COMBOS`` / the dispatcher
    in ``ssm_scan_fwd.cu`` / the binding whitelist — keep in lockstep.
    """
    if _C is None:
        return False
    if not decay.is_cuda or not update.is_cuda:
        return False
    if (decay.dtype, update.dtype) not in _KERNEL_DTYPE_COMBOS:
        return False
    return True


def _fp32_python_reference(
    decay: torch.Tensor, update: torch.Tensor
) -> torch.Tensor:
    """Sequential diag scan in fp32, cast back to ``update`` dtype.

    Mirrors ``chaoscontrol.core._diag_recurrence_inner`` but promotes
    to fp32 internally for stability. Used by the non-CUDA fallback
    branch of ``ssm_scan`` — dev macs and CPU pre-warm probes.
    """
    out_dtype = update.dtype
    d32 = decay.to(torch.float32)
    u32 = update.to(torch.float32)
    batch, seq, dim = d32.shape
    state = torch.zeros(batch, dim, dtype=torch.float32, device=d32.device)
    outputs = []
    for t in range(seq):
        state = d32[:, t] * state + u32[:, t]
        outputs.append(state)
    return torch.stack(outputs, dim=1).to(out_dtype)


class _SSMScanForwardFn(torch.autograd.Function):
    """Autograd wrapper around the forward+backward CUDA kernels.

    Kernel path: save-for-backward stores ``(decay, state_fp32)``. The
    backward kernel consumes ``state_fp32[t-1]`` for ``grad_decay``;
    fp32-exact state means backward differentiates through the true
    recurrence rather than a bf16/fp16-quantized surrogate.

    Python-reference fallback: stores ``(decay, update)`` and retraces
    the fp32 reference under ``torch.enable_grad`` to produce gradients.
    Used on dev macs (``_C is None``) and on CPU tensors (pre-warm
    probes). Gated by ``_is_kernel_eligible`` so the kernel is only
    invoked when it can actually run.
    """

    @staticmethod
    def forward(
        ctx, decay: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        if _is_kernel_eligible(decay, update):
            out, state_fp32 = _ssm_scan_forward_op(decay, update)
            ctx.save_for_backward(decay, state_fp32)
            ctx._fallback = False
        else:
            # Python fallback: retrace forward in backward.
            out = _fp32_python_reference(decay, update)
            ctx.save_for_backward(decay, update)
            ctx._fallback = True
        ctx.decay_dtype = decay.dtype
        ctx.update_dtype = update.dtype
        ctx.decay_requires_grad = decay.requires_grad
        ctx.update_requires_grad = update.requires_grad
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if ctx._fallback:
            return _SSMScanForwardFn._backward_python(ctx, grad_out)

        decay, state_fp32 = ctx.saved_tensors
        # Kernel dispatch requires grad_state.dtype == update_dtype.
        # Upstream autograd occasionally hands back an fp32 grad_out
        # even when the forward produced bf16 — the .to() is a no-op
        # when dtypes match and a safe cast otherwise.
        grad_out_cast = grad_out.to(ctx.update_dtype)
        grad_out_cast = grad_out_cast.contiguous()

        grad_decay, grad_update = _ssm_scan_backward_op(
            grad_out_cast, decay, state_fp32
        )

        out: list[torch.Tensor | None] = []
        out.append(grad_decay if ctx.decay_requires_grad else None)
        out.append(grad_update if ctx.update_requires_grad else None)
        return tuple(out)

    @staticmethod
    def _backward_python(ctx, grad_out: torch.Tensor):
        """Python-reference fallback: autograd through the fp32 recurrence."""
        decay, update = ctx.saved_tensors

        d_ref = decay.detach().to(torch.float32).requires_grad_(
            ctx.decay_requires_grad
        )
        u_ref = update.detach().to(torch.float32).requires_grad_(
            ctx.update_requires_grad
        )
        with torch.enable_grad():
            y_ref = _fp32_python_reference(d_ref, u_ref).to(torch.float32)
        grad_out_f32 = grad_out.to(torch.float32)

        grads = torch.autograd.grad(
            outputs=y_ref,
            inputs=[
                t for t, req in [
                    (d_ref, ctx.decay_requires_grad),
                    (u_ref, ctx.update_requires_grad),
                ] if req
            ],
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
    """Forward-only diag SSM scan (no autograd).

    Returns only the storage-dtype ``out`` tensor. The internal fp32
    state snapshot is discarded since it's only useful for backward.
    For the autograd-traceable version call ``ssm_scan(decay, update)``.

    On non-CUDA inputs this falls back to the fp32 Python reference —
    matches ``ssm_scan``'s dispatch rule.
    """
    if _is_kernel_eligible(decay, update):
        out, _state_fp32 = _ssm_scan_forward_op(decay, update)
        return out
    return _fp32_python_reference(decay, update)


def ssm_scan_forward_with_state(
    decay: torch.Tensor,
    update: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diag SSM scan returning ``(out, state_fp32)``.

    Exposes the fp32 state snapshot — mainly for unit tests that want
    to call ``ssm_scan_backward`` directly with the saved state. Raises
    via ``_require_ext`` if the extension isn't built; this helper has
    no fallback (callers that want the fallback should use
    ``ssm_scan_forward`` or the autograd wrapper instead).
    """
    _require_ext()
    return _ssm_scan_forward_op(decay, update)


def ssm_scan_backward(
    grad_state: torch.Tensor,
    decay: torch.Tensor,
    state_fp32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw backward-kernel call.

    Given ``grad_state`` (upstream dL/ds), the saved ``decay``, and the
    saved fp32 state snapshot ``state_fp32`` from forward, compute
    ``(grad_decay, grad_update)``. Does NOT go through autograd; use
    ``ssm_scan`` for the standard autograd-traceable path.
    """
    _require_ext()
    return _ssm_scan_backward_op(grad_state, decay, state_fp32)


def ssm_scan(
    decay: torch.Tensor,
    update: torch.Tensor,
) -> torch.Tensor:
    """Autograd-enabled diag SSM scan.

    Forward: custom CUDA kernel on CUDA inputs when the extension is
    built; fp32 Python reference otherwise (dev mac or CPU tensor).
    Backward: custom CUDA kernel on CUDA inputs; autograd-through-fp32
    reference otherwise.

    No ``_require_ext()`` guard — this function is safe to call on any
    device. Non-CUDA inputs route through the Python reference.

    Args:
        decay: (B, T, D) bf16/fp16/fp32, contiguous, row-major.
        update: (B, T, D) matching dtype.

    Returns:
        (B, T, D) state tensor; dtype matches ``update``.
    """
    return _SSMScanForwardFn.apply(decay, update)


__all__ = [
    "ssm_scan_forward",
    "ssm_scan_forward_with_state",
    "ssm_scan_backward",
    "ssm_scan",
]
