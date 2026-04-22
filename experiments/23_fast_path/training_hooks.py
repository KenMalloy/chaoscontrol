"""Training-time mechanism hooks for Exp 23.

These helpers implement low-overhead training mechanics that support the
experimental "fast-slow" consolidation pattern and spectral regularization
diagnostics used by the experiment runner.
"""

import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class FastSlowConsolidator:
    """Track a slow EMA-style model copy and occasional consolidation steps."""

    enabled: bool
    interval: int
    alpha: float
    slow_state: dict[str, torch.Tensor]
    sync_count: int = 0

    @classmethod
    def from_config(cls, model: torch.nn.Module, config: dict[str, object]) -> "FastSlowConsolidator":
        fast_slow_enabled = bool(config.get("fast_slow_enabled", False))
        interval = int(config.get("fast_slow_interval", 0)) if fast_slow_enabled else 0
        alpha = float(config.get("fast_slow_alpha", 0.0)) if fast_slow_enabled else 0.0

        slow_state = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

        return cls(
            enabled=fast_slow_enabled,
            interval=interval,
            alpha=alpha,
            slow_state=slow_state,
        )

    def after_optimizer_step(self, model: torch.nn.Module, *, step: int) -> None:
        if not self.enabled or self.interval <= 0 or step % self.interval != 0:
            return

        model_params = dict(model.named_parameters())
        slow_list: list[torch.Tensor] = []
        fast_list: list[torch.Tensor] = []
        with torch.no_grad():
            for name, slow_param in self.slow_state.items():
                model_param = model_params.get(name)
                if model_param is None:
                    continue
                fast_param = model_param.detach()
                if (
                    fast_param.device != slow_param.device
                    or fast_param.dtype != slow_param.dtype
                ):
                    slow_param.lerp_(fast_param, self.alpha)
                    continue
                slow_list.append(slow_param)
                fast_list.append(fast_param)

            if slow_list:
                torch._foreach_lerp_(slow_list, fast_list, self.alpha)

        self.sync_count += 1

    def copy_slow_to_model(self, model: torch.nn.Module) -> None:
        model_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, slow_param in self.slow_state.items():
                model_param = model_params.get(name)
                if model_param is None:
                    continue
                model_param.copy_(slow_param.to(device=model_param.device, dtype=model_param.dtype))

    def diagnostics(self, model: torch.nn.Module) -> dict[str, object]:
        model_params = dict(model.named_parameters())
        diff_sq_sum = 0.0
        diff_count = 0
        for name, slow_param in self.slow_state.items():
            model_param = model_params.get(name)
            if model_param is None:
                continue
            aligned_slow = slow_param.to(device=model_param.device, dtype=model_param.dtype)
            diff = model_param.detach() - aligned_slow
            diff_sq_sum += float((diff.float() ** 2).sum())
            diff_count += diff.numel()
        fast_slow_l2 = math.sqrt(diff_sq_sum) if diff_count > 0 else 0.0

        return {
            "enabled": self.enabled,
            "interval": self.interval,
            "alpha": self.alpha,
            "sync_count": self.sync_count,
            "fast_slow_l2": fast_slow_l2,
        }


def iter_log_a_params(model: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield 1-D log_a parameters."""
    for name, param in model.named_parameters():
        if name.endswith(".log_a") and param.ndim == 1:
            yield name, param


def spectral_regularization_loss(
    model: torch.nn.Module,
    lambda_dead: float,
    lambda_sticky: float,
    min_a: float,
    max_a: float,
) -> torch.Tensor | None:
    penalties = []
    for _, log_a in iter_log_a_params(model):
        a = torch.sigmoid(log_a.float())
        penalty = (
            torch.relu(min_a - a).pow(2) * lambda_dead
            + torch.relu(a - max_a).pow(2) * lambda_sticky
        )
        penalties.append(penalty.mean())

    if not penalties:
        return None

    return torch.stack(penalties).mean()


def spectral_summary(model: torch.nn.Module) -> dict[str, float | int]:
    a_values = []
    log_a_param_count = 0
    for _, log_a in iter_log_a_params(model):
        a_values.append(torch.sigmoid(log_a.float()).reshape(-1))
        log_a_param_count += 1

    if not a_values:
        return {"log_a_param_count": 0}

    a = torch.cat(a_values).detach()
    return {
        "log_a_param_count": log_a_param_count,
        "a_min": float(a.min()),
        "a_max": float(a.max()),
        "a_mean": float(a.mean()),
        "a_p05": float(torch.quantile(a, 0.05)),
        "a_p50": float(torch.quantile(a, 0.50)),
        "a_p95": float(torch.quantile(a, 0.95)),
    }


def predictive_auxiliary_loss(
    hidden: torch.Tensor,
    *,
    projection: torch.nn.Module,
    horizon: int,
) -> torch.Tensor | None:
    horizon = int(horizon)
    if horizon <= 0 or hidden.size(1) <= horizon:
        return None
    pred = projection(hidden[:, :-horizon])
    target = hidden[:, horizon:].detach()
    return torch.nn.functional.mse_loss(pred.float(), target.float())


def zero_embedding_grad_until(model: torch.nn.Module, step: int, freeze_steps: int) -> None:
    if freeze_steps <= 0 or step >= freeze_steps:
        return

    embedding = getattr(model, "embed", None)
    if embedding is None or not hasattr(embedding, "weight"):
        return

    grad = embedding.weight.grad
    if grad is None:
        return
    grad.zero_()
