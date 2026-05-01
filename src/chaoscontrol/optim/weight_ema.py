"""Rank-0-only fp32 weight EMA shadow for SemanticEngine training.

Mirrors the pattern used in records/track_10min_16mb/2026-03-22_11L_EMA_*
(train_gpt.py:1185-1289):
    ema_state = {name: t.detach().float().clone() for name, t in state_dict().items()}
    ema_decay = 0.997
    # per step:
    for name, t in state_dict().items():
        ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)

This implementation differs in three ways:
  1. Excludes prefixes (memory components — outer_model, semantic_tier, etc.)
     so episodic state is not averaged.
  2. Lives in a single rank-0 instance; other ranks construct WeightEMA(model,
     decay=0.0, ...) which produces an inert no-op shadow.
  3. Provides an ``applied(model)`` context manager for eval-time swap.
"""
from __future__ import annotations

import contextlib
from typing import Iterable, Iterator

import torch


class WeightEMA:
    """fp32 EMA shadow of a model's gradient-trained parameters.

    Construct once, call ``update(model)`` after each ``optimizer.step()``,
    and use ``with ema.applied(model): ...`` to temporarily swap EMA weights
    into the model for evaluation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        decay: float,
        exclude_prefixes: Iterable[str] = (),
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.exclude_prefixes = tuple(exclude_prefixes)
        # Build the shadow from named_parameters() — NOT state_dict() — so we
        # average gradient-trained tensors only. Excludes registered buffers
        # (BN running stats, RoPE caches, SSM channel buffers, etc.) and any
        # frozen / non-trainable parameters.
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_excluded(name):
                continue
            self.shadow[name] = param.detach().to(torch.float32).clone()

    def _is_excluded(self, name: str) -> bool:
        return any(name.startswith(prefix) for prefix in self.exclude_prefixes)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update the shadow toward ``model``'s current state_dict in-place."""
        if self.decay == 0.0:
            return
        decay = self.decay
        alpha = 1.0 - decay
        sd = model.state_dict()
        for name, shadow_t in self.shadow.items():
            current = sd[name]
            shadow_t.mul_(decay).add_(current.detach().to(torch.float32), alpha=alpha)

    @contextlib.contextmanager
    def applied(self, model: torch.nn.Module) -> Iterator[None]:
        """Context manager: swap shadow into model, restore on exit.

        The restore step uses the saved ORIGINAL values, so any updates
        the model receives inside the context are discarded. Use only
        for eval — do not train inside ``with ema.applied(model):``.
        """
        sd = model.state_dict()
        saved: dict[str, torch.Tensor] = {}
        for name, shadow_t in self.shadow.items():
            saved[name] = sd[name].detach().clone()
            sd[name].copy_(shadow_t.to(sd[name].dtype))
        try:
            yield
        finally:
            for name, original in saved.items():
                sd[name].copy_(original)


from typing import Callable, TypeVar

T = TypeVar("T")


def eval_with_ema(
    model: torch.nn.Module,
    ema: WeightEMA | None,
    eval_fn: Callable[[], T],
) -> T:
    """Run ``eval_fn()`` with EMA weights swapped into ``model``.

    If ``ema`` is None (e.g. on non-rank-0 ranks), ``eval_fn`` runs against
    the model's current weights unchanged. Always restores the original
    weights before returning.
    """
    if ema is None:
        return eval_fn()
    with ema.applied(model):
        return eval_fn()
