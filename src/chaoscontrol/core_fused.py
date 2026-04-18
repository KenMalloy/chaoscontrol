"""Post-scan kernel fusion for the ChaosControl SSM block.

Exp 18 Test 8: reduce kernel-launch count in the block's non-scan hot path.

The diag chunked scan produces `y` from the normalized input. The block then
runs:

    x = x + y                           # residual add              (1 kernel)
    x = x + self.ff(self.ff_norm(x))    # RMSNorm + FC + SiLU + Proj + residual

In eager mode that's roughly:
    residual add (1) + RMSNorm upcast/rstd/mul/downcast (~3) +
    FC matmul (1) + SiLU (1) + Proj matmul (1) + residual add (1)
= ~8 distinct kernel launches per block forward, all of them bandwidth-bound
(two matmuls are small at the per-token level for FF multiplier 2-4).

This module consolidates the full post-scan chain into a single
torch.compile region so Inductor can fuse the elementwise ops together.
Matmuls remain separate (Inductor does not fuse them with surrounding
elementwise ops today on most backends), but the ~5-6 surrounding kernels
collapse into 1-2 fused passes.

Design notes:

- The fused function is written in pure PyTorch with NO algorithmic
  changes relative to the unfused block. Line-for-line, it reproduces
  `x + y`, `RMSNorm.forward`, `FeedForward.forward`, and the second
  residual. torch.compile tracing should capture an identical FX graph
  to eager, differing only in Inductor's fusion passes.

- We follow the same lazy-resolve-with-runtime-fallback pattern as
  `chaoscontrol.core._diag_recurrence`: compile on first call, catch
  any InductorError at runtime, fall back to the uncompiled function,
  and set a warning. This keeps the package usable on stacks where
  Inductor codegen fails (local dev machines without a working C++
  toolchain, mismatched CUDA/toolchain, etc.) without making
  import-time behavior brittle.

- `FusedChaosSSMBlock` is a **drop-in replacement** for
  `ChaosSSMBlock` when `a_mode == "diag"`, `rich_b_mode == "none"`, and
  `return_jacobian_stats` is False. Any other configuration falls back
  to the unfused `ChaosSSMBlock.forward` path.

- `FusedChaosSSMBlock.from_unfused(block)` copies weights from a
  canonical `ChaosSSMBlock`, keeping the same parameter names and
  submodule layout so that state_dict round-trip is bit-exact. This
  makes the parity test clean.

- This module does NOT touch `ChaosSSMCore.forward` or the chunked
  scan backend — both are validated by their own tests
  (`tests/test_core.py`). The fused path wraps the scan as an opaque
  primitive, consuming only its output tensor.

Limitations / follow-ups:

- No `step()` method. The unfused `ChaosSSMBlock.step()` is used for
  single-token autoregressive inference (MCTS rollout, sampling).
  FusedChaosSSMBlock targets the training forward pass only, where the
  post-scan block cost is visible in tok/s. Single-token inference
  should continue to use `ChaosSSMBlock` or a dedicated fused step
  implementation — do NOT drop FusedChaosSSMBlock into an inference
  path until a step method is added.

- Maintenance coupling with `core.RMSNorm` and `core.FeedForward`.
  `_post_scan_fused_eager` inlines their operations as raw
  `F.rms_norm` / `F.linear` / `F.silu` calls so torch.compile can
  trace them in a single Python function. If RMSNorm or FeedForward
  is ever changed (e.g. adding a gate to FF, swapping activation),
  the fused function MUST be updated in lockstep. The parity test
  will detect drift immediately, but the coupling is not obvious
  from `core.py` — flag it in any PR that touches those modules.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chaoscontrol.core import ChaosSSMCore, FeedForward, RMSNorm


def _post_scan_fused_eager(
    x_pre_residual: torch.Tensor,
    scan_out: torch.Tensor,
    ff_norm_weight: torch.Tensor,
    ff_norm_eps: float,
    ff_fc_weight: torch.Tensor,
    ff_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Unfused reference implementation of the post-scan chain.

    Exactly matches:
        x = x_pre_residual + scan_out
        x = x + FeedForward(RMSNorm(x))

    Where RMSNorm is `F.rms_norm(x.float(), ..., eps) -> .to(x.dtype) * weight`
    and FeedForward is `Linear(SiLU(Linear(x)))` (no bias, no gate).

    This is the function that gets passed to `torch.compile`. Inductor
    traces the Python operations, fuses the elementwise sections, and
    emits one or two fused kernels instead of the 5-6 unfused launches.

    All operations are written to route exactly through the same PyTorch
    primitives as the eager `ChaosSSMBlock.forward` to guarantee autograd
    and numerical parity.

    Args:
        x_pre_residual: (B, T, D) — block input before residual add.
        scan_out: (B, T, D) — ChaosSSMCore diag-scan output.
        ff_norm_weight: (D,) — RMSNorm scale parameter for the FF norm.
        ff_norm_eps: RMSNorm epsilon (scalar).
        ff_fc_weight: (D * ff_mult, D) — FeedForward.fc.weight.
        ff_proj_weight: (D, D * ff_mult) — FeedForward.proj.weight.

    Returns:
        (B, T, D) — block output, same dtype as the inputs.
    """
    # First residual: x = x + y
    x_resid1 = x_pre_residual + scan_out

    # RMSNorm(x_resid1) — preserves the .float() upcast to guard bf16
    # precision during the reciprocal-sqrt statistic. Downcast back to
    # the input dtype before the weight multiply, matching eager
    # semantics exactly.
    dim = x_resid1.size(-1)
    normed = F.rms_norm(x_resid1.float(), (dim,), eps=ff_norm_eps)
    normed = normed.to(x_resid1.dtype) * ff_norm_weight

    # FeedForward(normed) — F.linear handles bias=None cleanly and is
    # autograd-identical to nn.Linear(bias=False).forward.
    hidden = F.linear(normed, ff_fc_weight)
    activated = F.silu(hidden)
    ff_out = F.linear(activated, ff_proj_weight)

    # Second residual: x = x + ff(ff_norm(x))
    return x_resid1 + ff_out


# Lazy-resolved compiled implementation. Mirrors the _diag_recurrence
# resolver in core.py: set on first call, catch runtime compile failures,
# fall back to eager.
_post_scan_impl: Any = None
_post_scan_backend: str = "eager"
_post_scan_note: str = "uninitialized"


def _resolve_post_scan_impl() -> Any:
    """Resolve the fastest available post-scan backend.

    Backends (selectable via CHAOSCONTROL_POST_SCAN_BACKEND env var):
        "eager"   — uncompiled pure-PyTorch function
        "compile" — torch.compile'd wrapper (default)

    Legacy flag CHAOSCONTROL_DISABLE_TORCH_COMPILE=1 forces eager.
    """
    global _post_scan_impl, _post_scan_backend, _post_scan_note
    if _post_scan_impl is not None:
        return _post_scan_impl

    requested = os.environ.get("CHAOSCONTROL_POST_SCAN_BACKEND", "").strip().lower()
    if os.environ.get("CHAOSCONTROL_DISABLE_TORCH_COMPILE", "").strip() == "1":
        requested = "eager"

    if requested == "eager":
        _post_scan_impl = _post_scan_fused_eager
        _post_scan_backend = "eager"
        _post_scan_note = "explicit eager backend"
        return _post_scan_impl

    # Default: torch.compile. We don't use mode="reduce-overhead" (CUDA
    # graphs) here because the surrounding ChaosSSMCore scan already
    # synchronizes and sizes the activation batches — cudagraphs tend to
    # conflict with dynamic batch shapes that training uses. Default mode
    # still gets the Inductor elementwise fusion.
    try:
        _post_scan_impl = torch.compile(_post_scan_fused_eager, dynamic=False)
        _post_scan_backend = "compile"
        _post_scan_note = "torch.compile(dynamic=False)"
    except Exception as exc:  # pragma: no cover — only triggers on broken stacks
        _post_scan_impl = _post_scan_fused_eager
        _post_scan_backend = "eager"
        _post_scan_note = f"compile unavailable: {exc.__class__.__name__}: {exc}"
        warnings.warn(
            "torch.compile unavailable for post-scan fusion; falling back to eager. "
            f"Reason: {_post_scan_note}",
            RuntimeWarning,
            stacklevel=2,
        )
    return _post_scan_impl


def get_post_scan_backend() -> dict[str, str]:
    """Report which post-scan backend is active."""
    _resolve_post_scan_impl()
    return {
        "backend": _post_scan_backend,
        "note": _post_scan_note,
    }


def post_scan_fused(
    x_pre_residual: torch.Tensor,
    scan_out: torch.Tensor,
    ff_norm_weight: torch.Tensor,
    ff_norm_eps: float,
    ff_fc_weight: torch.Tensor,
    ff_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Dispatcher with runtime fallback to eager on compile failure."""
    # Same rationale as ``core._diag_recurrence``: when an outer
    # ``torch.compile`` is tracing this call, route through the eager
    # implementation so the outer gets a single unified graph rather
    # than nesting into our cached ``torch.compile`` wrapper.
    if torch.compiler.is_compiling():
        return _post_scan_fused_eager(
            x_pre_residual,
            scan_out,
            ff_norm_weight,
            ff_norm_eps,
            ff_fc_weight,
            ff_proj_weight,
        )
    global _post_scan_impl, _post_scan_backend, _post_scan_note
    impl = _resolve_post_scan_impl()
    try:
        return impl(
            x_pre_residual,
            scan_out,
            ff_norm_weight,
            ff_norm_eps,
            ff_fc_weight,
            ff_proj_weight,
        )
    except Exception as exc:  # pragma: no cover
        if impl is _post_scan_fused_eager:
            raise
        _post_scan_impl = _post_scan_fused_eager
        _post_scan_backend = "eager"
        _post_scan_note = f"compile runtime failure: {exc.__class__.__name__}: {exc}"
        warnings.warn(
            "torch.compile failed during post-scan execution; falling back to eager. "
            f"Reason: {_post_scan_note}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _post_scan_fused_eager(
            x_pre_residual,
            scan_out,
            ff_norm_weight,
            ff_norm_eps,
            ff_fc_weight,
            ff_proj_weight,
        )


class FusedChaosSSMBlock(nn.Module):
    """Kernel-fused variant of ChaosSSMBlock.

    Structure and parameter naming are identical to `ChaosSSMBlock` so
    state_dicts round-trip. The only difference is the forward pass:
    the post-scan residual + RMSNorm + FF + residual chain runs through
    `post_scan_fused`, which torch.compile fuses into 1-2 kernels
    instead of ~5-6 eager launches.

    This fused path only engages when:
        - a_mode == "diag"
        - rich_b_mode == "none" (matching the scan fast path)
        - return_jacobian_stats is False

    In every other configuration `forward` dispatches to the same
    eager code path as `ChaosSSMBlock`, so the fused block is a
    safe drop-in for any config — correctness is preserved even in
    cases where fusion isn't available.
    """

    def __init__(
        self,
        dim: int,
        ff_mult: int = 2,
        *,
        a_mode: str = "diag",
        a_full_rank: int = 8,
        a_full_gamma: float = 0.05,
    ) -> None:
        super().__init__()
        # Names match ChaosSSMBlock so from_unfused() can copy weights
        # without any remapping.
        self.input_norm = RMSNorm(dim)
        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_mult)
        self.core = ChaosSSMCore(
            dim, a_mode=a_mode, a_full_rank=a_full_rank, a_full_gamma=a_full_gamma
        )
        self.rich_b: nn.Module | None = None  # Only "none" supported in fused path
        self._a_mode = a_mode

    @classmethod
    def from_unfused(cls, block: nn.Module) -> "FusedChaosSSMBlock":
        """Build a fused block by copying weights from an unfused ChaosSSMBlock.

        Matches the source block's parameter dtype exactly, so bf16 source
        blocks produce bf16 fused blocks. This is critical for bf16 parity
        tests: a float32 fused block loaded with bf16 state_dict values
        would upcast silently and produce different numerics downstream.

        Uses strict state_dict load after casting to the source dtype.
        The source block must have `a_mode == "diag"` and
        `rich_b is None`. Any other configuration raises.
        """
        core = block.core
        if core.a_mode != "diag":
            raise ValueError(
                f"FusedChaosSSMBlock.from_unfused requires a_mode='diag', got {core.a_mode!r}"
            )
        if block.rich_b is not None:
            raise ValueError(
                "FusedChaosSSMBlock.from_unfused requires rich_b=None"
            )
        dim = core.dim
        ff_mult = block.ff.fc.out_features // dim
        fused = cls(dim, ff_mult, a_mode="diag")
        # Match source block dtype before copying weights, otherwise
        # load_state_dict would stuff bf16 values into float32 params.
        src_dtype = block.ff.fc.weight.dtype
        fused = fused.to(src_dtype)
        fused.load_state_dict(block.state_dict(), strict=True)
        return fused

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_jacobian_stats: bool = False,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # Unsupported configurations fall back to the unfused path. This
        # keeps FusedChaosSSMBlock behaviorally identical to ChaosSSMBlock
        # for all inputs; fusion is strictly a fast path.
        # Non-zero initial_state forces the unfused path because the fused
        # kernel reuses the core fast scan which cannot seed a carry. Zero
        # initial state (or None) with return_final_state=True stays on the
        # fused path — we extract final_state from an extra core call that
        # returns the states tensor; still cheaper than the unfused loop.
        if return_jacobian_stats or self._a_mode != "diag" or initial_state is not None:
            return self._forward_unfused(
                x,
                return_jacobian_stats=return_jacobian_stats,
                initial_state=initial_state,
                return_final_state=return_final_state,
            )

        # 1) Input norm — identical to ChaosSSMBlock.
        normed = self.input_norm(x)

        # 2) ChaosSSMCore diag scan — untouched, delegates to core.forward.
        if return_final_state:
            scan_out, final_state = self.core(
                normed, rich_b=None, return_jacobian_stats=False,
                return_final_state=True,
            )
        else:
            scan_out = self.core(normed, rich_b=None, return_jacobian_stats=False)
            final_state = None

        # 3) Fused post-scan chain: residual + RMSNorm + FF + residual.
        y = post_scan_fused(
            x,
            scan_out,
            self.ff_norm.weight,
            self.ff_norm.eps,
            self.ff.fc.weight,
            self.ff.proj.weight,
        )
        if return_final_state:
            return y, final_state
        return y

    def _forward_unfused(
        self,
        x: torch.Tensor,
        *,
        return_jacobian_stats: bool = False,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Fallback path matching ChaosSSMBlock.forward byte-for-byte."""
        normed = self.input_norm(x)
        result = self.core(
            normed, rich_b=self.rich_b,
            return_jacobian_stats=return_jacobian_stats,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )
        if return_jacobian_stats and return_final_state:
            y, stats, final_state = result
        elif return_jacobian_stats:
            y, stats = result
            final_state = None
        elif return_final_state:
            y, final_state = result
            stats = None
        else:
            y = result
            stats = None
            final_state = None
        x = x + y
        x = x + self.ff(self.ff_norm(x))
        if return_jacobian_stats and return_final_state:
            return x, stats, final_state
        if return_jacobian_stats:
            return x, stats
        if return_final_state:
            return x, final_state
        return x
