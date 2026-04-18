"""cuBLASLt fp8 matmul — bespoke replacement for torch._scaled_mm.

Scope:
  * ``cublaslt_fp8_matmul`` — forward fp8 matmul, fast-accum ON, optional
    CUBLASLT_EPILOGUE_BIAS.
  * ``cublaslt_fp8_matmul_grad_x`` — backward grad_x = grad_y @ W,
    fast-accum OFF (split-accumulator convention, matches TE).
  * ``cublaslt_fp8_matmul_grad_w`` — backward grad_w = grad_y.t() @ x,
    fast-accum OFF, with optional CUBLASLT_EPILOGUE_BGRADB for fused
    bias-gradient reduction.

All three share per-tensor fp32 scales and match ``torch._scaled_mm``'s
semantic convention (scales are dequant multipliers applied on the
accumulator side). The wrapper raises ``ImportError`` at call time if
the extension wasn't compiled (dev mac / partial pod setup).

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


def _require_ext() -> None:
    if _C is None:  # pragma: no cover — import-time failure path
        raise ImportError(
            "chaoscontrol.kernels._cublaslt._C is not built; rerun "
            "`pip install -e .` on a pod with the CUDA 13 toolchain. "
            f"Original import error: {_IMPORT_ERROR!r}"
        )


def cublaslt_fp8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused cuBLASLt fp8 matmul (forward). See module docstring.

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
    """
    _require_ext()
    return _C.cublaslt_fp8_matmul(a, b, scale_a, scale_b, bias, out_dtype)


def cublaslt_fp8_matmul_grad_x(
    grad_y_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_gy: torch.Tensor,
    scale_w: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Backward fp8 matmul: ``grad_x = grad_y @ W``.

    Args:
        grad_y_fp8: fp8 E5M2 row-major ``[M, N]``.
        weight_fp8: fp8 E4M3 column-major ``[N, K]``.
        scale_gy: fp32 scalar device tensor.
        scale_w: fp32 scalar device tensor.
        out_dtype: ``torch.bfloat16`` (default) or ``torch.float16``.

    Returns:
        Tensor of shape ``[M, K]`` in ``out_dtype``, row-major.
    """
    _require_ext()
    return _C.cublaslt_fp8_matmul_grad_x(
        grad_y_fp8, weight_fp8, scale_gy, scale_w, out_dtype,
    )


def cublaslt_fp8_matmul_grad_w(
    grad_y_fp8_t: torch.Tensor,
    x_fp8: torch.Tensor,
    scale_gy: torch.Tensor,
    scale_x: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    compute_bias_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Backward fp8 matmul: ``grad_w = grad_y.t() @ x`` + optional fused dbias.

    Args:
        grad_y_fp8_t: fp8 E5M2 row-major ``[N, M]`` (transpose already
            materialized by the caller).
        x_fp8: fp8 E4M3 column-major ``[M, K]``.
        scale_gy: fp32 scalar.
        scale_x: fp32 scalar.
        out_dtype: ``torch.bfloat16`` (default) or ``torch.float16``.
        compute_bias_grad: if True, enable the CUBLASLT_EPILOGUE_BGRADB
            epilogue and return ``grad_bias`` alongside ``grad_w``.

    Returns:
        Tuple ``(grad_w, grad_bias)`` where ``grad_w`` has shape ``[N, K]``
        in ``out_dtype``; ``grad_bias`` has shape ``[N]`` in ``out_dtype``
        if ``compute_bias_grad`` else ``None``.
    """
    _require_ext()
    return _C.cublaslt_fp8_matmul_grad_w(
        grad_y_fp8_t, x_fp8, scale_gy, scale_x, out_dtype, compute_bias_grad,
    )


def cublaslt_fp8_linear_fwd(
    a_bf16: torch.Tensor,
    b_bf16: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    x_pending: torch.Tensor,
    w_pending: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused amax-update + bf16→E4M3 cast + forward cuBLASLt matmul.

    One C++ frame replaces the per-call Python orchestration of
    ``torch.maximum + (x / scale).to(e4m3) + matmul``. See phase 3
    task spec for rationale.

    Args:
        a_bf16: bf16 ``[M, K]`` row-major activations.
        b_bf16: bf16 ``[N, K]`` row-major weight (transposed inside to
            a column-major ``[K, N]`` fp8 tensor — no Python-side ``.t()``).
        x_scale, w_scale: fp32 scalar dequant multipliers; read stale,
            refreshed by ``flush_amax_history()``.
        x_pending, w_pending: fp32 scalar pending-amax buffers; updated
            atomically with ``max(|a|)``, ``max(|b|)``.
        bias: optional ``[N]`` bias, fused via CUBLASLT_EPILOGUE_BIAS.
        out_dtype: ``torch.bfloat16`` or ``torch.float16``.

    Returns:
        ``[M, N]`` tensor in ``out_dtype``.
    """
    _require_ext()
    return _C.cublaslt_fp8_linear_fwd(
        a_bf16, b_bf16, x_scale, w_scale, x_pending, w_pending, bias, out_dtype,
    )


def cublaslt_fp8_linear_bwd_x(
    grad_y_bf16: torch.Tensor,
    weight_bf16: torch.Tensor,
    gy_scale: torch.Tensor,
    w_scale: torch.Tensor,
    gy_pending: torch.Tensor,
    gx_pending: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused amax-update + fp8 cast + backward matmul for ``grad_x``.

    Updates ``gy_pending`` with ``amax(grad_y)``. ``gx_pending`` is
    reserved for a diagnostic grad_x amax fold; currently the C++
    implementation does not write to it (the Python wrapper folds it
    post-return).

    Args:
        grad_y_bf16: bf16 ``[M, N]`` row-major.
        weight_bf16: bf16 ``[N, K]`` row-major (same layout as
            ``nn.Linear.weight``).
        gy_scale, w_scale: fp32 scalar dequant multipliers.
        gy_pending, gx_pending: fp32 scalar pending buffers.
        out_dtype: ``torch.bfloat16`` (default).

    Returns:
        ``[M, K]`` grad_x tensor.
    """
    _require_ext()
    return _C.cublaslt_fp8_linear_bwd_x(
        grad_y_bf16, weight_bf16, gy_scale, w_scale, gy_pending, gx_pending,
        out_dtype,
    )


def cublaslt_fp8_linear_bwd_w(
    grad_y_bf16: torch.Tensor,
    x_bf16: torch.Tensor,
    gy_scale: torch.Tensor,
    x_scale: torch.Tensor,
    gy_pending: torch.Tensor,
    x_pending: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    compute_bias_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused amax-update + fp8 cast + backward matmul for ``grad_w``.

    Handles the grad_y and x transposes internally — no Python-side
    ``.t().contiguous()``. ``x_pending`` is currently ignored inside
    (the forward already updated the same tensor's pending this step);
    we still accept it for API uniformity.

    Args:
        grad_y_bf16: bf16 ``[M, N]`` row-major.
        x_bf16: bf16 ``[M, K]`` row-major activations (the same tensor
            used in the forward).
        gy_scale, x_scale: fp32 scalar dequant multipliers.
        gy_pending, x_pending: fp32 scalar pending buffers.
        out_dtype: ``torch.bfloat16`` (default).
        compute_bias_grad: if True, fuse CUBLASLT_EPILOGUE_BGRADB.

    Returns:
        ``(grad_w [N, K], grad_bias [N] or None)``.
    """
    _require_ext()
    return _C.cublaslt_fp8_linear_bwd_w(
        grad_y_bf16, x_bf16, gy_scale, x_scale, gy_pending, x_pending,
        out_dtype, compute_bias_grad,
    )


def cublaslt_fp8_flush_amax(
    history: torch.Tensor,
    pending: torch.Tensor,
    scale: torch.Tensor,
    max_rep: float,
) -> None:
    """One-shot C++ amax flush.

    Single kernel launch replaces the Python chain
    ``history = torch.roll(history, -1); history[-1] = pending; scale =
    max(history) / max_rep; pending.zero_()``. All updates in place.
    """
    _require_ext()
    _C.cublaslt_fp8_flush_amax(history, pending, scale, float(max_rep))


def cublaslt_fp8_flush_amax_diagnostic(
    history: torch.Tensor,
    pending: torch.Tensor,
) -> None:
    """Diagnostic flush variant: roll history, zero pending, no scale."""
    _require_ext()
    _C.cublaslt_fp8_flush_amax_diagnostic(history, pending)


__all__ = [
    "cublaslt_fp8_matmul",
    "cublaslt_fp8_matmul_grad_x",
    "cublaslt_fp8_matmul_grad_w",
    "cublaslt_fp8_linear_fwd",
    "cublaslt_fp8_linear_bwd_x",
    "cublaslt_fp8_linear_bwd_w",
    "cublaslt_fp8_flush_amax",
    "cublaslt_fp8_flush_amax_diagnostic",
]
