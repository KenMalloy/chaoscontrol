"""Wrap optimizer.step() to apply momentum warmup and update weight EMA.

Single injection point: call ``wrap_optimizer_step`` once after
constructing the optimizer in ``_build_optimizer``. Every subsequent call
to ``optimizer.step()`` from any site in the runner will (1) blend
momentum from ``warmup_start`` to ``target_momentum`` over
``warmup_steps`` calls, then (2) call the original step, then (3)
update the rank-0 EMA shadow if enabled.
"""
from __future__ import annotations

from typing import Iterable

import torch

from chaoscontrol.optim.momentum_warmup import apply_momentum_warmup
from chaoscontrol.optim.weight_ema import WeightEMA


def wrap_optimizer_step(
    optimizer: torch.optim.Optimizer,
    *,
    model: torch.nn.Module,
    target_momentum: float,
    warmup_start: float,
    warmup_steps: int,
    weight_ema_decay: float,
    is_rank_zero: bool,
    ema_exclude_prefixes: Iterable[str],
    weight_ema_fake_quant_bits: int = 0,
) -> None:
    """Replace ``optimizer.step`` with a closure that schedules momentum and
    updates a weight EMA. Mutates ``optimizer`` in place.

    On non-rank-0 ranks (``is_rank_zero=False``), the EMA is not constructed
    and ``optimizer._weight_ema`` is set to ``None``. Momentum warmup still
    runs on every rank (so DDP all-reduces remain consistent).
    """
    if is_rank_zero and weight_ema_decay > 0.0:
        ema: WeightEMA | None = WeightEMA(
            model,
            decay=weight_ema_decay,
            exclude_prefixes=tuple(ema_exclude_prefixes),
            fake_quant_bits=weight_ema_fake_quant_bits,
        )
    else:
        ema = None

    optimizer._weight_ema = ema  # type: ignore[attr-defined]
    optimizer._momentum_warmup_step = 0  # type: ignore[attr-defined]

    original_step = optimizer.step

    def wrapped_step(*args, **kwargs):
        apply_momentum_warmup(
            optimizer,
            step=optimizer._momentum_warmup_step,  # type: ignore[attr-defined]
            target=target_momentum,
            start=warmup_start,
            steps=warmup_steps,
        )
        result = original_step(*args, **kwargs)
        optimizer._momentum_warmup_step += 1  # type: ignore[attr-defined]
        if ema is not None:
            ema.update(model)
        return result

    optimizer.step = wrapped_step  # type: ignore[assignment]
