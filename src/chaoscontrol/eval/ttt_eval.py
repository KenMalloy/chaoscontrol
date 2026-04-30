"""Multi-calc_type eval harness over ``ValCache``.

Each calc_type is a separate test-time-training (TTT) strategy. Per cell,
a single trained checkpoint is evaluated under N strategies in series;
each strategy runs its own pass over the validation set and emits its
own BPB. The orchestrator collects them into
``result["eval"]["calc_types"][<name>] = {"bpb", "loss", ...}``.

Contract for a calc_type:

    @register_calc_type("my_calc_type", ...)
    def my_calc_type(ctx: CalcTypeContext) -> CalcTypeResult: ...

The registry is module-level. Calc_types live in
``chaoscontrol.eval.calc_types`` and are imported eagerly by that
package's ``__init__`` so registration happens on package import.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from chaoscontrol.eval_stream.val_cache import ValCache


@dataclass
class CalcTypeContext:
    """Inputs handed to a calc_type's eval pass.

    The calc_type owns the per-doc loop, the state-passing decision, and
    any optimizer wiring. It must NOT mutate ``model`` permanently — any
    parameter snapshot/restore is the calc_type's responsibility.
    """

    model: nn.Module
    val_cache: ValCache
    device: torch.device
    base_bytes_lut: torch.Tensor
    has_leading_space_lut: torch.Tensor
    is_boundary_token_lut: torch.Tensor
    config: dict[str, Any]


@dataclass
class CalcTypeResult:
    """What a calc_type returns.

    ``bpb`` and ``loss`` are the headline numbers for the per-cell dict
    written to ``result["eval"]["calc_types"][<name>]``. ``hyperparams``
    records the actual values used so a downstream reader can reproduce.
    ``extra`` is a free dict for calc-type-specific telemetry.
    """

    bpb: float
    loss: float
    docs_scored: int
    tokens_scored: int
    raw_bytes: int
    hyperparams: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)


CalcTypeFn = Callable[[CalcTypeContext], CalcTypeResult]


CALC_TYPE_REGISTRY: dict[str, CalcTypeFn] = {}
CALC_TYPE_METADATA: dict[str, dict[str, Any]] = {}


def register_calc_type(
    name: str,
    *,
    requires_source_order: bool = False,
    requires_grad: bool = False,
    description: str = "",
) -> Callable[[CalcTypeFn], CalcTypeFn]:
    """Register a calc_type implementation under ``name``.

    ``requires_source_order``: True for calc_types that break the
    reset-commutativity assumption (e.g. ``carry_state``, continual
    ``dreamworld_eval``). The orchestrator loads ``ValCache`` with
    ``source_order`` ordering for those passes; reset-commutative
    calc_types may use the default ordering.

    ``requires_grad``: True for calc_types that need autograd at eval
    time (e.g. ``dreamworld_eval``). The dispatcher will not wrap such
    calc_types in ``torch.no_grad()``; they manage their own grad scope.
    """

    def deco(fn: CalcTypeFn) -> CalcTypeFn:
        if name in CALC_TYPE_REGISTRY:
            raise ValueError(f"calc_type {name!r} already registered")
        CALC_TYPE_REGISTRY[name] = fn
        CALC_TYPE_METADATA[name] = {
            "requires_source_order": bool(requires_source_order),
            "requires_grad": bool(requires_grad),
            "description": str(description),
        }
        return fn

    return deco


def calc_type_metadata(name: str) -> dict[str, Any]:
    """Return the metadata dict registered alongside ``name``."""
    if name not in CALC_TYPE_METADATA:
        raise ValueError(f"unknown calc_type: {name!r}")
    return dict(CALC_TYPE_METADATA[name])


def evaluate_with_calc_types(
    *,
    model: nn.Module,
    val_cache: ValCache,
    calc_types: list[str],
    calc_type_configs: dict[str, dict[str, Any]] | None = None,
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    source_order_preserved: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run each requested calc_type as a separate eval pass.

    Each calc_type owns its own per-doc loop and state-passing
    discipline. The dispatcher only routes ``ctx`` to the registered
    function and packs the result into the per-cell dict shape.

    Order of ``calc_types`` is preserved in the returned dict.

    The same ``val_cache`` instance is passed to every calc_type. If
    a calc_type requires a different doc ordering, the caller must
    provide a re-ordered ``val_cache`` (the registry's
    ``requires_source_order`` metadata is advisory; the orchestrator
    consumes it to pre-sort).

    ``source_order_preserved`` is a tripwire. ``ValCache`` produced by
    :func:`chaoscontrol.eval_stream.val_cache.write_val_cache` is always
    in source order (DocStreamer reads JSONL in order). Future code that
    shuffles the cache must pass ``source_order_preserved=False``; the
    dispatcher then refuses to run any calc_type with
    ``requires_source_order=True``.
    """
    if calc_type_configs is None:
        calc_type_configs = {}
    unknown = [n for n in calc_types if n not in CALC_TYPE_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown calc_type(s): {unknown!r}; "
            f"registered={sorted(CALC_TYPE_REGISTRY)}"
        )
    if not source_order_preserved:
        order_sensitive = [
            n for n in calc_types
            if CALC_TYPE_METADATA[n]["requires_source_order"]
        ]
        if order_sensitive:
            raise ValueError(
                f"calc_type(s) {order_sensitive!r} require source-ordered "
                f"docs but source_order_preserved=False"
            )

    out: dict[str, dict[str, Any]] = {}
    for name in calc_types:
        fn = CALC_TYPE_REGISTRY[name]
        ctx = CalcTypeContext(
            model=model,
            val_cache=val_cache,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            config=dict(calc_type_configs.get(name, {})),
        )
        result = fn(ctx)
        out[name] = {
            "bpb": float(result.bpb),
            "loss": float(result.loss),
            "docs_scored": int(result.docs_scored),
            "tokens_scored": int(result.tokens_scored),
            "raw_bytes": int(result.raw_bytes),
            "hyperparams": dict(result.hyperparams),
            **dict(result.extra),
        }
    return out


def list_registered_calc_types() -> list[str]:
    """Return the names of every registered calc_type, sorted."""
    return sorted(CALC_TYPE_REGISTRY)
