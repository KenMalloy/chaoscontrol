"""Precision autocast policy for train_ssm training loop.

Supported dtypes:
    "fp32" — no autocast (vanilla float32)
    "bf16" — torch.amp.autocast(dtype=torch.bfloat16)
    "fp8"  — transformer_engine.pytorch.fp8_autocast(enabled=True)

Callers pass cfg["precision"] and get the right context back; no
branching elsewhere in the loop. TE availability is lazily probed
the first time fp8 is requested, so the module imports cleanly on
machines without TE installed.

Also exposes maybe_promote_linears_to_te(model, enabled) which does
in-place nn.Linear -> te.Linear swaps when fp8 is requested. te.Linear
implements fp8 matmul inside a te.fp8_autocast context; without the
swap, nn.Linear runs at its declared dtype regardless of the autocast
context. Promotion preserves weights, bias, and device/dtype.
"""
from __future__ import annotations

import contextlib
from typing import ContextManager

import torch
import torch.nn as nn


# Module-level cache for the lazy TE-availability probe. ``None`` = not yet
# checked, ``True`` = importable, ``False`` = not importable. Caching the
# result means repeat calls don't re-pay the import attempt cost, and we
# only print the "TE unavailable" warning once per process.
_TE_AVAILABLE: bool | None = None


def _check_te_available() -> bool:
    """Lazy probe for transformer_engine.pytorch availability.

    Returns True if the module is importable, False otherwise. Cached
    at module level so repeat calls after the first probe are free.
    Module import itself (``import chaoscontrol.precision``) does NOT
    touch TE — only this function does — so the module loads cleanly
    on machines without TE installed.
    """
    global _TE_AVAILABLE
    if _TE_AVAILABLE is not None:
        return _TE_AVAILABLE
    try:
        import transformer_engine.pytorch  # noqa: F401
        _TE_AVAILABLE = True
    except Exception:
        # Catch broadly — TE's import can fail in ways beyond ImportError
        # (CUDA mismatch, missing shared libs, etc.) on pod environments
        # where the package is half-installed.
        _TE_AVAILABLE = False
    return _TE_AVAILABLE


@contextlib.contextmanager
def autocast_context(dtype: str, device_type: str = "cuda") -> ContextManager:
    """Context manager that enables the requested mixed-precision policy.

    Args:
        dtype: one of {"fp32", "bf16", "fp8"}.
        device_type: passed through to ``torch.amp.autocast`` for the
            bf16 path. Ignored by the fp8 path (TE's ``fp8_autocast``
            doesn't take a device argument — it uses the current CUDA
            device implicitly).

    Raises:
        ValueError: for any dtype not in the supported set.
        RuntimeError: if fp8 is requested on a machine without TE.
    """
    if dtype == "fp32":
        # nullcontext is cheap — entering/exiting does nothing. Keeps
        # the caller's with-block uniform regardless of the dtype.
        with contextlib.nullcontext():
            yield
        return
    if dtype == "bf16":
        # bf16 autocast is CUDA-only by project convention — see
        # ``chaoscontrol.data.maybe_autocast``. On CPU it's a nullcontext
        # so CPU-only tests (no CUDA available) run at their declared
        # fp32 dtype, matching the pre-precision-abstraction behavior.
        if device_type != "cuda":
            with contextlib.nullcontext():
                yield
            return
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            yield
        return
    if dtype == "fp8":
        if not _check_te_available():
            raise RuntimeError(
                "fp8 autocast requires transformer_engine; "
                "pip install transformer-engine[pytorch]"
            )
        import transformer_engine.pytorch as te  # type: ignore[import-not-found]
        with te.fp8_autocast(enabled=True):
            yield
        return
    raise ValueError(
        f"Unsupported precision dtype: {dtype!r}. "
        f"Expected one of {{'fp32', 'bf16', 'fp8'}}."
    )


def _iter_linear_children(module: nn.Module) -> list[tuple[nn.Module, str, nn.Linear]]:
    """Walk ``module`` and return ``(parent, child_name, linear_module)``
    tuples for every direct-child ``nn.Linear``. The walk is recursive
    but only yields direct-child relationships so the caller can do
    ``setattr(parent, child_name, te_linear)`` to swap in place.

    Excluded: ``te.Linear`` instances (already-promoted linears — safe
    to re-run the promotion). We detect these without importing TE at
    module level by checking the class module-name.
    """
    pairs: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in module.modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            # Skip already-promoted te.Linear without touching TE imports
            # in the no-promotion path. te.Linear subclasses nn.Linear
            # in some TE versions and doesn't in others — class-module
            # check covers both.
            cls_module = type(child).__module__
            if cls_module.startswith("transformer_engine"):
                continue
            pairs.append((parent, child_name, child))
    return pairs


def maybe_promote_linears_to_te(model: nn.Module, enabled: bool) -> int:
    """In-place swap ``nn.Linear`` children with ``te.Linear`` so fp8
    autocast actually emits fp8 matmuls.

    Preserves weight data, bias (if present), and the original device
    and dtype. Returns the number of layers promoted; 0 means the walk
    found no plain ``nn.Linear`` children to swap, which is the
    expected return when ``enabled=False`` or TE is unavailable.

    No-op cases (return 0):
      - ``enabled=False``: caller explicitly opted out.
      - TE unavailable on this machine: prints a one-line warning to
        stdout (not a raise) so dev machines without TE can run the
        same code path without crashing.

    Design note: te.Linear uses fp8 matmul only inside an
    ``fp8_autocast`` context. Outside that context it behaves like
    ``nn.Linear`` at its declared dtype. So the promotion is safe for
    eval paths that don't enter the fp8 context.
    """
    if not enabled:
        return 0
    if not _check_te_available():
        # Print once per call — callers usually only invoke this per
        # training run, so duplicate warnings aren't a real risk. A
        # raise would force every dev machine to have TE installed just
        # to read the fp8 runner code path.
        print(
            "[precision] WARNING: maybe_promote_linears_to_te(enabled=True) "
            "was called but transformer_engine is not available; "
            "skipping promotion (returning 0).",
            flush=True,
        )
        return 0

    import transformer_engine.pytorch as te  # type: ignore[import-not-found]

    pairs = _iter_linear_children(model)
    promoted = 0
    for parent, child_name, old in pairs:
        # Preserve the originating device and dtype; te.Linear defaults
        # to fp32 params if unspecified, which would silently widen the
        # model and break any downstream dtype assumptions.
        weight_device = old.weight.device
        weight_dtype = old.weight.dtype
        has_bias = old.bias is not None

        # ``params_dtype`` is the storage dtype of te.Linear's master
        # weights; fp8 matmul happens at runtime inside the autocast
        # context, independent of storage dtype.
        new = te.Linear(
            old.in_features,
            old.out_features,
            bias=has_bias,
            params_dtype=weight_dtype,
            device=weight_device,
        )
        # Copy weight (and bias if present) in-place — avoids allocating
        # fresh random init for params that have already been initialized
        # by the model builder.
        with torch.no_grad():
            new.weight.data.copy_(old.weight.data)
            if has_bias:
                new.bias.data.copy_(old.bias.data)

        setattr(parent, child_name, new)
        promoted += 1
    return promoted
