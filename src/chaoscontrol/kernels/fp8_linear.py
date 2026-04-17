"""Bespoke fp8 Linear — drop-in replacement for ``te.Linear``.

Task 1B-2 stage: ``FusedFP8Linear.forward`` calls ``torch._scaled_mm``
directly — a single fused ``cast(x) + cast(w) + mm + cast(y)`` cuBLASLt
op on H100. The TE delegate from Task 1B-1 is gone; the raw
``weight`` / ``bias`` Parameters on this module are now the sole state.

Amax tracking (this task):
    Per-call. Each ``forward()`` updates on-device ring buffers
    (``x_amax_history`` / ``w_amax_history``) with this step's
    per-tensor abs-max AFTER the matmul, and computes the scale for
    the NEXT call from the lookback window. Task 1B-3 will defer the
    amax update to an explicit ``flush_amax_history`` call to drop the
    per-forward GPU→GPU reduction sync; until then, the buffers at
    least avoid the Python-side host sync TE performs on every call.

Scale convention:
    ``_scaled_mm`` computes ``scale_result * ((scale_a * a) @ (scale_b
    * b))``, where ``a``/``b`` are fp8 and ``scale_a``/``scale_b`` are
    fp32. We quantize by DIVIDING by the per-tensor scale
    (``a_fp8 = (x / x_scale).to(e4m3)`` with ``x_scale = amax / 448``)
    and pass the same scale back to ``_scaled_mm`` so the kernel
    dequantizes on the accumulator side. This keeps the fp8 operands
    well inside the E4M3 range (|max| = 448) and is the TE convention.

Design choices worth flagging:

* ``weight`` (and optional ``bias``) live on the ``FusedFP8Linear``
  module itself. No delegate; autograd tracks these leaves directly.
  ``_scaled_mm`` has native autograd registration in torch 2.3+, so
  backward works through plain autograd (no custom
  ``torch.autograd.Function`` needed).
* Amax history buffers are registered as ``persistent=True`` so
  checkpoints capture the scale-factor trajectory. Reloading a
  checkpoint preserves the in-flight scaling state.
* ``transformer_engine`` is NOT imported at module scope anywhere in
  this file. The fused path is pure torch. TE remains a test-only
  dependency for the numerical reference.

TODO(1B-3): replace the in-forward amax update with a ring-buffer
push that does NOT force a sync, plus a module method
``flush_amax_history()`` the training loop calls once per optimizer
step.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# E4M3 absolute-value ceiling. ``torch.float8_e4m3fn`` representable
# max is 448.0; we scale so amax → 448 and operands stay in-range.
_E4M3_MAX: float = 448.0

# Default amax history length. Matches TE's default ``fp8_format`` amax
# history window, and is short enough that a single outlier step decays
# out of the scale within a few iterations.
_DEFAULT_AMAX_HISTORY_LEN: int = 16


def _scale_from_amax(history: torch.Tensor) -> torch.Tensor:
    """Return fp32 scalar scale ``= max(history) / 448`` or 1.0 on cold start.

    The scale is a DIVISOR when quantizing (``x_fp8 = x / scale``) and a
    MULTIPLIER when dequantizing inside ``_scaled_mm`` (``scale_a``).
    Returns 1.0 when the history is all zeros (first forward before any
    amax has been observed), so the very first step uses unit scaling
    and the fp8 cast just clamps; subsequent steps get a real scale.
    """
    recent_max = history.max()
    scale = recent_max / _E4M3_MAX
    # torch.where preserves the scalar fp32 dtype and avoids a host sync
    # that ``if recent_max.item() > 0`` would trigger.
    return torch.where(
        recent_max > 0,
        scale,
        torch.ones_like(scale),
    )


# E5M2 absolute-value ceiling. ``torch.float8_e5m2`` representable max
# is 57344.0 — E5M2 has wider dynamic range than E4M3 (precision trade
# for range), which is the right choice for gradient tensors whose
# amax spans much more than activation amax.
_E5M2_MAX: float = 57344.0


def _fp8_matmul(
    a: torch.Tensor, a_scale: torch.Tensor,
    b: torch.Tensor, b_scale: torch.Tensor,
    a_dtype: torch.dtype, b_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused fp8 GEMM via ``torch._scaled_mm``. Returns bf16.

    Assumes ``a`` is M×K row-major and ``b`` is K×N column-major (the
    cuBLASLt fp8 contract). Quantizes each operand by DIVIDING by its
    scale (so values fit in the target dtype's representable range),
    then passes the same scales to ``_scaled_mm`` as dequant multipliers.
    """
    a_fp8 = (a / a_scale).to(a_dtype)
    b_fp8 = (b / b_scale).to(b_dtype)
    return torch._scaled_mm(
        a_fp8, b_fp8,
        scale_a=a_scale, scale_b=b_scale,
        bias=bias, out_dtype=torch.bfloat16,
    )


class _FusedFP8LinearFn(torch.autograd.Function):
    """Full fp8 forward + fp8 backward custom autograd Function.

    ``torch._scaled_mm`` on torch 2.11+cu130 has NO registered autograd
    derivative (neither the low-level ``torch._scaled_mm`` nor the
    higher-level ``torch.nn.functional.scaled_mm`` work — both dispatch
    to aten ops with no derivative). We implement the backward
    explicitly here so the whole Linear — forward + backward — runs
    through fused fp8 GEMMs. No bf16 fallback.

    Dtype convention (matches TE's default fp8 recipe):
      - Forward operands (activations x, weights W): E4M3.
      - Backward operands (gradients grad_y): E5M2 — wider dynamic
        range handles the larger amax variance that grads exhibit
        across training. Mixing E5M2 × E4M3 is a standard fp8 GEMM.

    Linear math:
        forward:     y      = x @ W.t()  + b
        backward:    dL/dx  = dL/dy @ W              (E5M2 × E4M3)
                     dL/dW  = dL/dy.t() @ x          (E5M2 × E4M3)
                     dL/db  = dL/dy.sum(0)           (bf16 reduction)

    where x is [M, K], W is [N, K], grad_y is [M, N].
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x_flat: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        gy_amax_history: torch.Tensor,
        gx_amax_history: torch.Tensor,
    ) -> torch.Tensor:
        # Forward fp8 GEMM — cuBLASLt needs a row-major, b column-major.
        # weight is N×K row-major; .t() view is K×N column-major.
        y_flat = _fp8_matmul(
            x_flat, x_scale,
            weight.t(), w_scale,
            torch.float8_e4m3fn, torch.float8_e4m3fn,
            bias=bias,
        )
        # Save bf16 master tensors for backward GEMMs (they'll be
        # requantized per backward direction; amax histories for the
        # grad tensors are tracked in the calling module).
        ctx.save_for_backward(x_flat, weight, x_scale, w_scale,
                              gy_amax_history, gx_amax_history)
        ctx.has_bias = bias is not None
        return y_flat

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_y: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        (x_flat, weight, x_scale, w_scale,
         gy_amax_history, gx_amax_history) = ctx.saved_tensors

        # Compute grad_y scale from its amax history (lookback, same
        # pattern as forward's activation/weight scales). gy_scale is
        # read BEFORE updating the history so the current step uses a
        # prior-step scale, and the update appends this step's amax.
        gy_scale = _scale_from_amax_e5m2(gy_amax_history)

        # dL/dx = grad_y @ W.
        # a = grad_y (M, N) row-major, b = weight (N, K) — need K-minor
        # for b's column-major layout. weight as-is is N×K row-major
        # → grad_x contract is [M,N] @ [N,K] = [M,K]; b must be
        # column-major K×N? No — the contraction is over N.
        # For _scaled_mm(a, b), a is M×K, b is K×N, output is M×N.
        # Here we want M×N × N×K = M×K, so we map:
        #   mapped M = actual M, mapped K = actual N (contraction),
        #   mapped N = actual K.
        # a = grad_y [M, N] — row-major, fp8. b must be [N, K]
        # column-major — which is ... exactly weight [N, K] with its
        # NATIVE layout interpreted as "K-stride-1, N-stride-K", i.e.
        # column-major if we pass weight.t().t() (a no-op that returns
        # the same view). Simpler: _scaled_mm requires a row-major,
        # b column-major — weight.t() gives K×N column-major. We want
        # b as N×K column-major. That's just weight.t().t() = weight,
        # but weight's native layout is row-major (N×K stride K,1).
        # To get N×K column-major we need weight.contiguous().t().t()
        # which is weight — so weight is N×K row-major, NOT column-
        # major. Solution: transpose grad_y instead, and compute the
        # result transposed.
        #
        # Easiest correct recipe for dL/dx = grad_y @ W:
        #   compute (W.t() @ grad_y.t()).t() via _scaled_mm, which maps
        #   to _scaled_mm(a=W.t() [K,N] row-major, b=grad_y.t() [N,M]
        #   column-major) -> [K, M], then transpose to [M, K].
        grad_x_flat_t = _fp8_matmul(
            weight.t().contiguous(), w_scale,   # [K, N] row-major fp8 E4M3
            grad_y.t(), gy_scale,                # [N, M] column-major view of [M,N] row-major
            torch.float8_e4m3fn, torch.float8_e5m2,
        )
        grad_x_flat = grad_x_flat_t.t()  # [M, K]

        # dL/dW = grad_y.t() @ x.
        # a = grad_y.t() [N, M] row-major → need .contiguous().
        # b = x_flat [M, K] — must be column-major. x_flat native
        # layout is [M, K] row-major; to get column-major [M, K]
        # take x_flat.t().t().contiguous().t() = x_flat.t().contiguous().t()
        # which materializes the transpose then views it back — so
        # b = x_flat.t() [K, M] column-major? No — that's the wrong
        # shape. _scaled_mm wants b as K×N. Here K=M (contraction),
        # N=K (output feature). So b should be [M, K] column-major.
        # Simplest: x_flat.t().contiguous() gives [K, M] row-major,
        # then .t() gives [M, K] column-major view.
        grad_w = _fp8_matmul(
            grad_y.t().contiguous(), gy_scale,   # [N, M] row-major fp8 E5M2
            x_flat.t().contiguous().t(), x_scale,  # [M, K] column-major fp8 E4M3
            torch.float8_e5m2, torch.float8_e4m3fn,
        )

        # Bias grad: simple bf16 reduction. TE does the same — a per-
        # element sum isn't worth routing through fp8.
        grad_b = grad_y.sum(dim=0) if ctx.has_bias else None

        # Update the grad-y amax history AFTER computing the grads.
        # (_update_amax_history handles no_grad + in-place shift.)
        _update_amax_history(gy_amax_history, grad_y.detach().abs().amax())
        # grad_x amax tracked for completeness / future inspection;
        # not currently consumed (only grad_y is used on the backward
        # GEMM operand side), but keeping it aligns with TE's per-
        # tensor history model.
        _update_amax_history(gx_amax_history, grad_x_flat.detach().abs().amax())

        # Return order matches forward args exactly.
        return grad_x_flat, grad_w, grad_b, None, None, None, None


def _scale_from_amax_e5m2(history: torch.Tensor) -> torch.Tensor:
    """E5M2 version of ``_scale_from_amax``. Max representable is 57344."""
    recent_max = history.max()
    scale = recent_max / _E5M2_MAX
    return torch.where(
        recent_max > 0,
        scale,
        torch.ones_like(scale),
    )


def _update_amax_history(history: torch.Tensor, new_amax: torch.Tensor) -> None:
    """In-place ring-buffer shift + write of a new per-tensor amax value.

    Shifts the buffer left by one (oldest value falls off the front)
    and writes ``new_amax`` into the last slot. Done under ``no_grad``;
    the caller must also be in an appropriate context if ``new_amax``
    is derived from an autograd-tracked tensor.
    """
    with torch.no_grad():
        history.copy_(torch.roll(history, -1))
        history[-1] = new_amax


class FusedFP8Linear(nn.Module):
    """fp8-path ``nn.Linear`` backed by fused ``torch._scaled_mm``.

    Public API is identical to the Task 1B-1 scaffold. The forward
    body now calls ``torch._scaled_mm`` directly rather than delegating
    to ``te.Linear``; output parity against stock TE is maintained
    within ``rtol=atol=3e-2`` (the bar set in the Phase 1 plan).

    Args:
        in_features: inner dim of the matmul.
        out_features: outer dim of the matmul.
        bias: whether to allocate a bias parameter.
        device: device for ``weight`` / ``bias``. ``None`` means default
            (usually CPU on import, moved later via ``.to(...)``).
        dtype: master-weight dtype. bf16 by default to match the rest
            of the training loop; the fp8 cast happens inside forward.
        amax_history_len: ring-buffer length for per-tensor amax
            lookback. Defaults to 16, matching TE. Checkpoint-persistent.

    Notes:
        This module does NOT require ``transformer_engine`` — the fused
        path is pure torch. TE remains a test-only dependency used
        purely as a numerical reference.
    """

    in_features: int
    out_features: int
    weight: nn.Parameter
    # bias is Optional[nn.Parameter]; declared via ``register_parameter``
    # below so static attribute access is ``self.bias`` either way.

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
            # ``register_parameter(name, None)`` is the idiomatic way to
            # declare an optional parameter as absent; ``self.bias`` then
            # returns None and is not included in ``.parameters()``.
            self.register_parameter("bias", None)

        # Default-initialize to match ``nn.Linear``'s kaiming-uniform
        # scheme so bare construction produces a usable layer.
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Amax ring buffers. fp32 for scale-factor precision (scales are
        # read back into fp32 scalars for _scaled_mm). Persistent so
        # checkpoints capture the scale-factor trajectory and reloading
        # preserves in-flight scaling state.
        self.register_buffer(
            "x_amax_history",
            torch.zeros(amax_history_len, dtype=torch.float32, device=device),
            persistent=True,
        )
        self.register_buffer(
            "w_amax_history",
            torch.zeros(amax_history_len, dtype=torch.float32, device=device),
            persistent=True,
        )
        # Backward-pass amax buffers: grad_y is the incoming gradient
        # (wrt output y), grad_x is the outgoing gradient (wrt input x).
        # Tracked separately from forward because grads have different
        # dynamic range (E5M2 dtype) and their own amax trajectory.
        self.register_buffer(
            "gy_amax_history",
            torch.zeros(amax_history_len, dtype=torch.float32, device=device),
            persistent=True,
        )
        self.register_buffer(
            "gx_amax_history",
            torch.zeros(amax_history_len, dtype=torch.float32, device=device),
            persistent=True,
        )

    @classmethod
    def from_nn_linear(cls, m: nn.Linear) -> "FusedFP8Linear":
        """Adopt an existing ``nn.Linear`` — copy its weight + bias in.

        The returned instance lives on the same device/dtype as ``m``.
        Weight and bias values are byte-copied from ``m``, so call sites
        that swap a pre-initialized ``nn.Linear`` for a fused fp8 one
        preserve training state exactly.
        """
        has_bias = m.bias is not None
        out = cls(
            m.in_features,
            m.out_features,
            bias=has_bias,
            device=m.weight.device,
            dtype=m.weight.dtype,
        )
        with torch.no_grad():
            out.weight.data.copy_(m.weight.data)
            if has_bias:
                assert out.bias is not None
                assert m.bias is not None
                out.bias.data.copy_(m.bias.data)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fused fp8 Linear via ``torch._scaled_mm``.

        Folds leading batch dims to satisfy ``_scaled_mm``'s 2-D input
        contract, computes per-tensor scales from the amax history
        (lookback, not lookahead — the first step uses unit scaling),
        quantizes to E4M3, dispatches the fused kernel, and updates the
        amax history with this step's max AFTER the matmul.

        The kernel call is:

            _scaled_mm(a_fp8_MxK, b_fp8_KxN,
                       scale_a=x_scale, scale_b=w_scale,
                       bias=..., out_dtype=bf16)

        where ``scale_a``/``scale_b`` are the DEQUANT multipliers (equal
        to ``amax/448``). We quantize by dividing by the same scales.
        """
        # Fold leading dims: _scaled_mm wants 2-D inputs.
        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        # Scales computed from the LOOKBACK history (prior step's amax)
        # — a lookahead would require a two-pass sync we're avoiding.
        x_scale = _scale_from_amax(self.x_amax_history)
        w_scale = _scale_from_amax(self.w_amax_history)

        # Dispatch through the custom autograd Function. It handles
        # fp8 forward via _scaled_mm, and implements backward with two
        # explicit fp8 matmuls (E5M2 grads × E4M3 activations/weights).
        # See _FusedFP8LinearFn docstring for the full dtype convention.
        y_flat = _FusedFP8LinearFn.apply(
            x_flat, self.weight, self.bias, x_scale, w_scale,
            self.gy_amax_history, self.gx_amax_history,
        )

        # Update amax history with this step's max AFTER the matmul.
        # This means the CURRENT step used last step's scale (lookback);
        # subsequent steps will use this step's amax. On the first call
        # the history is all zeros and _scale_from_amax returns 1.0.
        with torch.no_grad():
            _update_amax_history(self.x_amax_history, x_flat.detach().abs().amax())
            _update_amax_history(self.w_amax_history, self.weight.detach().abs().amax())

        return y_flat.reshape(*leading_shape, self.out_features)

    def extra_repr(self) -> str:
        # Match ``nn.Linear.extra_repr`` so model print trees stay readable.
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
