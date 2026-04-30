"""Bridge between the fast-path runner's eval call site and the
multi-calc_type dispatcher.

The runner picks one of two eval paths per cell:

  - Legacy: a single ``evaluate_bpb_sp`` pass over random-window starts.
    Used by every experiment before exp27. Returned as-is.
  - Calc_types: a ``ValCache``-driven multi-strategy pass. The headline
    BPB is set to ``score_only_reset`` for backward compatibility with
    downstream tooling that reads ``result["eval"]["bpb"]``; the full
    per-calc_type dict lives under ``result["eval"]["calc_types"]``.

The legacy eval is supplied as an injected callable so this module
stays in ``src/`` without reverse-importing from ``experiments/``.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch import nn

from chaoscontrol.eval.ttt_eval import evaluate_with_calc_types
from chaoscontrol.eval_stream.val_cache import ValCache


LegacyEvalFn = Callable[..., dict[str, Any]]


def dispatch_eval_for_config(
    config: dict[str, Any],
    *,
    model: nn.Module,
    val_cache: ValCache | None,
    val_tokens: torch.Tensor,
    eval_starts: list[int] | torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    legacy_evaluate_fn: LegacyEvalFn,
) -> dict[str, Any]:
    """Run the eval path implied by ``config``.

    If ``config["calc_types"]`` is a non-empty list, dispatch to
    :func:`evaluate_with_calc_types` over ``val_cache`` and return:

        {
          "calc_types": {<name>: {"bpb", "loss", ...}, ...},
          "bpb":   <score_only_reset bpb if present, else first calc_type's bpb>,
          "loss":  <matching loss>,
        }

    Otherwise fall back to ``legacy_evaluate_fn(model, tokens=val_tokens,
    eval_starts=eval_starts, batch_size=batch_size, seq_len=seq_len,
    device=device, base_bytes_lut=base_bytes_lut, ...)`` and return its
    dict unchanged.
    """
    calc_types = list(config.get("calc_types") or [])
    if not calc_types:
        return legacy_evaluate_fn(
            model,
            tokens=val_tokens,
            eval_starts=eval_starts,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )

    if val_cache is None:
        raise ValueError(
            "config sets calc_types but val_cache is None; "
            "pass --val-cache-dir to the runner"
        )

    calc_type_configs = dict(config.get("calc_type_configs") or {})
    ct_results = evaluate_with_calc_types(
        model=model,
        val_cache=val_cache,
        calc_types=calc_types,
        calc_type_configs=calc_type_configs,
        device=device,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        source_order_preserved=True,
    )

    headline_name = (
        "score_only_reset" if "score_only_reset" in ct_results
        else next(iter(ct_results))
    )
    headline = ct_results[headline_name]

    nonfinite = [
        n for n, entry in ct_results.items()
        if not math.isfinite(float(entry["bpb"]))
        or not math.isfinite(float(entry["loss"]))
    ]
    if nonfinite:
        raise RuntimeError(
            f"non-finite bpb/loss for calc_type(s) {nonfinite!r}: "
            f"{ {n: ct_results[n] for n in nonfinite} }"
        )

    return {
        "calc_types": ct_results,
        "bpb": float(headline["bpb"]),
        "loss": float(headline["loss"]),
        "headline_calc_type": headline_name,
    }


__all__ = ["dispatch_eval_for_config"]
