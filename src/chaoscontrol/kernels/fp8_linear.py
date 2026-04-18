"""Bespoke fp8 Linear — drop-in replacement for ``te.Linear``.

Task 1B-3 state: ``FusedFP8Linear.forward/backward`` calls through our
forked cuBLASLt extension (``cublaslt_fp8_matmul`` +
``cublaslt_fp8_matmul_grad_x`` + ``cublaslt_fp8_matmul_grad_w``). The
``torch._scaled_mm`` path is no longer exercised in the hot loop; it
remains in-tree as ``_scaled_mm_forward_reference`` so future tasks have
a numerical oracle.

Deferred amax (Task 1B-3):
    Forward no longer touches the ring-buffer history. Instead, each
    forward folds its new per-tensor amax into a ``*_pending`` device
    buffer via ``torch.maximum`` — one reduction kernel, no ring-buffer
    copy, no host sync. Scales are READ from separate ``x_scale`` /
    ``w_scale`` / ``gy_scale`` device buffers that the training loop
    refreshes by calling ``flush_amax_history()`` once per optimizer
    step (typically after ``optimizer.step()``). A convenience
    ``fused_fp8_flush_all(model)`` walks the module tree and flushes
    every ``FusedFP8Linear``.

    The first forward before any flush uses scale = 1.0 (initialized in
    ``__init__``). This matches TE's cold-start behavior: the first fp8
    cast clamps to the representable range; subsequent steps pick up
    real scales once the first flush lands.

Scale convention:
    Our cuBLASLt kernel (like ``torch._scaled_mm``) computes
    ``(scale_a * a) @ (scale_b * b)`` on the accumulator side, where
    ``a``/``b`` are fp8 and ``scale_a``/``scale_b`` are fp32. We quantize
    by DIVIDING by the per-tensor scale (``a_fp8 = (x / x_scale).to(e4m3)``
    with ``x_scale = amax / 448``) and pass the same scale back to the
    kernel so it dequantizes on the accumulator side. This keeps fp8
    operands well inside the E4M3 range (|max| = 448) and is the TE
    convention.

Design choices worth flagging:

* ``weight`` (and optional ``bias``) live on the ``FusedFP8Linear``
  module itself. No delegate; autograd tracks these leaves directly.
* Amax history + pending + scale buffers are registered as
  ``persistent=True`` so checkpoints capture the full scale-factor
  trajectory. Reloading a checkpoint preserves in-flight scaling state.
* ``transformer_engine`` is NOT imported at module scope anywhere in
  this file. The fused path is pure torch + our extension. TE remains a
  test-only dependency for the numerical reference.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from chaoscontrol.kernels._cublaslt import (
    cublaslt_fp8_flush_amax,
    cublaslt_fp8_flush_amax_diagnostic,
    cublaslt_fp8_linear_bwd_w,
    cublaslt_fp8_linear_bwd_x,
    cublaslt_fp8_linear_fwd,
    cublaslt_fp8_matmul,
    cublaslt_fp8_matmul_grad_w,
    cublaslt_fp8_matmul_grad_x,
)


# E4M3 absolute-value ceiling. ``torch.float8_e4m3fn`` representable
# max is 448.0; we scale so amax → 448 and operands stay in-range.
_E4M3_MAX: float = 448.0

# E5M2 absolute-value ceiling. ``torch.float8_e5m2`` max is 57344.0
# (wider dynamic range, less precision — the standard gradient dtype).
_E5M2_MAX: float = 57344.0

# Default amax history length. Matches TE's default ``fp8_format`` amax
# history window, and is short enough that a single outlier step decays
# out of the scale within a few iterations.
_DEFAULT_AMAX_HISTORY_LEN: int = 16


# ---------------------------------------------------------------------------
# _scaled_mm reference (unused in hot path; kept for numerical oracle).
# ---------------------------------------------------------------------------


def _scaled_mm_forward_reference(
    a: torch.Tensor, a_scale: torch.Tensor,
    b: torch.Tensor, b_scale: torch.Tensor,
    a_dtype: torch.dtype, b_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference forward path via ``torch._scaled_mm``.

    Not called in the hot path as of Task 1B-3 (the cuBLASLt fork is
    faster), but kept in-tree as a numerical oracle for tests and as
    a fallback on torch builds without the extension compiled.
    """
    a_fp8 = (a / a_scale).to(a_dtype)
    b_fp8 = (b / b_scale).to(b_dtype)
    return torch._scaled_mm(
        a_fp8, b_fp8,
        scale_a=a_scale, scale_b=b_scale,
        bias=bias, out_dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------------
# Ring-buffer helpers (driven by flush_amax_history; NOT per-forward).
# ---------------------------------------------------------------------------


def _push_amax_and_rescale(
    history: torch.Tensor,
    pending: torch.Tensor,
    scale_buf: torch.Tensor,
    max_rep: float,
) -> None:
    """Roll history, append ``pending``, recompute ``scale_buf``, zero pending.

    All in-place on device. No host sync — the max over the history is
    still a GPU-side reduction, but it runs ONCE per optimizer step (per
    tensor per layer), not once per forward.

    Phase 3: the hot path now calls ``cublaslt_fp8_flush_amax`` which
    collapses this whole block into a single kernel launch. This Python
    implementation is kept as a numerical reference for tests and as a
    fallback on hosts where the extension isn't compiled — same policy
    as ``_scaled_mm_forward_reference``.
    """
    with torch.no_grad():
        history.copy_(torch.roll(history, -1))
        history[-1] = pending.squeeze()
        recent_max = history.max()
        new_scale = torch.where(
            recent_max > 0,
            recent_max / max_rep,
            torch.ones_like(recent_max),
        )
        scale_buf.copy_(new_scale)
        pending.zero_()


# ---------------------------------------------------------------------------
# Autograd Function — fp8 forward + fp8 backward via the cuBLASLt fork.
# ---------------------------------------------------------------------------


class _FusedFP8LinearFn(torch.autograd.Function):
    """Full fp8 forward + fp8 backward via our cuBLASLt extension.

    Dtype convention (matches TE's default fp8 recipe):
      - Forward operands (activations x, weights W): E4M3.
      - Backward operands (gradients grad_y): E5M2 — wider dynamic
        range handles the larger amax variance that grads exhibit
        across training.

    Linear math:
        forward:     y      = x @ W.t()  + b
        backward:    dL/dx  = dL/dy @ W              (E5M2 × E4M3, grad_x GEMM)
                     dL/dW  = dL/dy.t() @ x          (E5M2 × E4M3, grad_w GEMM)
                     dL/db  = dL/dy.sum(0)           (fused into grad_w via BGRADB)

    where x is [M, K], W is [N, K], grad_y is [M, N].

    Scale SOURCE: scales are read from the module's ``x_scale``,
    ``w_scale``, ``gy_scale`` buffers — recomputed once per optimizer
    step by ``flush_amax_history()``. The per-forward amax observation
    is folded into the matching ``*_pending`` buffer.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x_flat: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        gy_scale: torch.Tensor,
        x_pending: torch.Tensor,
        w_pending: torch.Tensor,
        gy_pending: torch.Tensor,
        gx_pending: torch.Tensor,
    ) -> torch.Tensor:
        # Phase 3: single C++ frame folds
        #   * max-reduce of |x_flat|, atomic into x_pending,
        #   * max-reduce of |weight|, atomic into w_pending,
        #   * bf16→E4M3 cast of both at the CURRENT scales,
        #   * fp8 cuBLASLt matmul + optional CUBLASLT_EPILOGUE_BIAS.
        # Replaces ~6 Python-launched kernels with a single call.
        y_flat = cublaslt_fp8_linear_fwd(
            x_flat, weight,
            x_scale=x_scale, w_scale=w_scale,
            x_pending=x_pending, w_pending=w_pending,
            bias=bias, out_dtype=torch.bfloat16,
        )

        # Save the bf16 operands (not the fp8 staging tensors — those
        # were released at the C++ frame boundary). The backward
        # re-casts x and grad_y via the same fused kernel. This costs
        # one extra fp8 cast vs caching the staging fp8 forward-side,
        # but avoids two Python-side transpose allocations, which
        # dominated Phase 2's backward time.
        ctx.save_for_backward(
            x_flat, weight, x_scale, w_scale, gy_scale,
            gy_pending, x_pending, gx_pending,
        )
        ctx.has_bias = bias is not None
        return y_flat

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_y: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        (x_flat, weight, x_scale, w_scale, gy_scale,
         gy_pending, x_pending, gx_pending) = ctx.saved_tensors

        # Autograd can hand back a non-contiguous grad_y when upstream
        # ops fused views. The fused kernel requires row-major bf16.
        grad_y_c = grad_y.contiguous()

        # grad_x = grad_y @ W — fused cast of grad_y (E5M2) + cast of
        # weight to col-major E4M3 + GEMM, all in one C++ frame.
        grad_x_flat = cublaslt_fp8_linear_bwd_x(
            grad_y_c, weight,
            gy_scale=gy_scale, w_scale=w_scale,
            gy_pending=gy_pending, gx_pending=gx_pending,
            out_dtype=torch.bfloat16,
        )

        # grad_w = grad_y.t() @ x + optional dbias — fused transpose-casts
        # of grad_y (→[N, M] E5M2) and x (→[M, K] col-major E4M3) + GEMM
        # with optional CUBLASLT_EPILOGUE_BGRADB.
        grad_w, grad_b = cublaslt_fp8_linear_bwd_w(
            grad_y_c, x_flat,
            gy_scale=gy_scale, x_scale=x_scale,
            gy_pending=gy_pending, x_pending=x_pending,
            out_dtype=torch.bfloat16,
            compute_bias_grad=ctx.has_bias,
        )
        # Bias gradient is produced inside the bwd_w cast kernel via a
        # fused column-sum reduction (task 25, 2026-04-18) — piggybacks
        # on the bf16 read bandwidth we're already paying for the fp8
        # cast of grad_y. Replaces the former Python grad_y.sum(0) path,
        # which existed because cuBLASLt's BGRADB epilogue is rejected
        # for fp8 E5M2×E4M3 on every cuBLAS version we've probed
        # (12.8.4, 13.4.0.1). Our path is independent of NVIDIA shipping
        # that epilogue.

        # Phase 3: the gx_pending update was Python-orchestrated (two
        # kernel launches: .abs().amax() + .maximum()) and was eating
        # ~30-40 us per backward. The gx_amax signal is purely
        # diagnostic — no consumer scale reads it — so we drop it from
        # the hot path. The gx_amax_history ring will read zero in
        # downstream diagnostics; any future consumer should fold the
        # amax inside the C++ bwd_x entry point.
        _ = gx_pending   # suppress unused-arg lint

        # Return order mirrors forward args.
        return (
            grad_x_flat,   # x_flat
            grad_w,        # weight
            grad_b,        # bias
            None, None, None,   # x_scale, w_scale, gy_scale (not inputs)
            None, None, None, None,  # pending buffers (not inputs)
        )


class FusedFP8Linear(nn.Module):
    """fp8-path ``nn.Linear`` backed by our forked cuBLASLt extension.

    Public API is identical to the Task 1B-1 scaffold. Args:
        in_features: inner dim of the matmul.
        out_features: outer dim of the matmul.
        bias: whether to allocate a bias parameter.
        device: device for ``weight`` / ``bias``. ``None`` means default.
        dtype: master-weight dtype. bf16 by default to match the rest
            of the training loop; the fp8 cast happens inside forward.
        amax_history_len: ring-buffer length for per-tensor amax
            lookback. Defaults to 16, matching TE. Checkpoint-persistent.

    Training-loop contract:
        Call ``flush_amax_history()`` (or ``fused_fp8_flush_all(model)``)
        once per optimizer step AFTER ``optimizer.step()``. Skipping the
        flush is permitted — scales stay stale but correctness holds; the
        only cost is missing the recent-amax trajectory.
    """

    in_features: int
    out_features: int
    weight: nn.Parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        amax_history_len: int = _DEFAULT_AMAX_HISTORY_LEN,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((out_features,), device=device, dtype=dtype),
            )
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Amax ring-buffer history — fp32, persistent. Captures the
        # lookback window that drives the scale.
        history_kwargs: dict[str, Any] = dict(dtype=torch.float32, device=device)
        for name in ("x_amax_history", "w_amax_history",
                     "gy_amax_history", "gx_amax_history"):
            self.register_buffer(
                name, torch.zeros(amax_history_len, **history_kwargs),
                persistent=True,
            )

        # Pending amax — folded each forward/backward via torch.maximum,
        # flushed into the history (and cleared) by flush_amax_history().
        # Length-[1] fp32 scalar tensors; persistent so checkpoints
        # capture mid-step pending amax (rarely relevant but cheap).
        for name in ("x_amax_pending", "w_amax_pending",
                     "gy_amax_pending", "gx_amax_pending"):
            self.register_buffer(
                name, torch.zeros(1, **history_kwargs),
                persistent=True,
            )

        # Live scales — recomputed by flush_amax_history(). Initialized
        # to 1.0 so the very first forward (before any flush) produces
        # unit scaling; the fp8 cast then clamps, same as TE's cold start.
        for name in ("x_scale", "w_scale", "gy_scale"):
            self.register_buffer(
                name, torch.ones(1, **history_kwargs),
                persistent=True,
            )

    @classmethod
    def from_nn_linear(cls, m: nn.Linear) -> "FusedFP8Linear":
        """Adopt an existing ``nn.Linear`` — copy its weight + bias in."""
        has_bias = m.bias is not None
        out = cls(
            m.in_features, m.out_features, bias=has_bias,
            device=m.weight.device, dtype=m.weight.dtype,
        )
        with torch.no_grad():
            out.weight.data.copy_(m.weight.data)
            if has_bias:
                assert out.bias is not None
                assert m.bias is not None
                out.bias.data.copy_(m.bias.data)
        return out

    def flush_amax_history(self) -> None:
        """Roll pending amax into history + recompute scales, once per step.

        Phase 3: each (history, pending, scale) triple flushes via a
        single C++ kernel launch — replaces ~6 Python-orchestrated ops
        per tensor (``torch.roll + indexing + .max() + torch.where +
        .copy_() + .zero_()``). Four tensors total → four kernel
        launches instead of ~21. No host sync.
        """
        cublaslt_fp8_flush_amax(
            self.x_amax_history, self.x_amax_pending, self.x_scale, _E4M3_MAX,
        )
        cublaslt_fp8_flush_amax(
            self.w_amax_history, self.w_amax_pending, self.w_scale, _E4M3_MAX,
        )
        cublaslt_fp8_flush_amax(
            self.gy_amax_history, self.gy_amax_pending, self.gy_scale, _E5M2_MAX,
        )
        # grad_x amax is diagnostic — roll + zero, no consumer scale.
        cublaslt_fp8_flush_amax_diagnostic(
            self.gx_amax_history, self.gx_amax_pending,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fused fp8 Linear via the cuBLASLt fork.

        Folds leading batch dims, quantizes to fp8 using the scales
        READ from the ``x_scale`` / ``w_scale`` buffers (refreshed by
        ``flush_amax_history()``), dispatches the fused kernel, and
        folds this forward's amax into the ``*_pending`` buffers without
        a host sync. Scales remain stable between flushes; the first
        forward before any flush uses unit scale (cold start).
        """
        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        y_flat = _FusedFP8LinearFn.apply(
            x_flat, self.weight, self.bias,
            self.x_scale, self.w_scale, self.gy_scale,
            self.x_amax_pending, self.w_amax_pending,
            self.gy_amax_pending, self.gx_amax_pending,
        )
        return y_flat.reshape(*leading_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def fused_fp8_flush_all(model: nn.Module) -> int:
    """Walk ``model`` and flush every ``FusedFP8Linear`` submodule.

    Returns the number of modules flushed so the caller can sanity-check
    that the model actually contains fp8 Linears. Training loops should
    call this once per optimizer step AFTER ``optimizer.step()``.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, FusedFP8Linear):
            m.flush_amax_history()
            n += 1
    return n
