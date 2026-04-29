"""Training-time mechanism hooks for Exp 23.

These helpers implement low-overhead training mechanics that support the
experimental "fast-slow" consolidation pattern and spectral regularization
diagnostics used by the experiment runner.
"""

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class FastSlowDecision:
    """One slow-weight consolidation decision."""

    mode: str
    accepted: bool
    alpha: float
    gate: float
    effective_alpha: float
    step: int
    reason: str


@dataclass
class FastSlowConsolidator:
    """Track a slow EMA-style model copy and occasional consolidation steps."""

    enabled: bool
    interval: int
    alpha: float
    slow_state: dict[str, torch.Tensor]
    sync_count: int = 0
    decision_count: int = 0
    learned_decision_count: int = 0
    learned_sync_count: int = 0
    last_sync_step: int = -1
    last_decision: FastSlowDecision | None = None
    last_reward: float = 0.0
    reward_update_count: int = 0
    _last_loss_value: float | None = None
    _last_credit_key: int | None = None

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

    def after_optimizer_step(
        self,
        model: torch.nn.Module,
        *,
        step: int,
        action_space: Any | None = None,
        reward_context: dict[str, Any] | None = None,
        loss_value: float | None = None,
    ) -> FastSlowDecision:
        decision = self.decide(
            step=step,
            action_space=action_space,
            reward_context=reward_context,
            loss_value=loss_value,
        )
        self.apply_decision(model, decision)
        return decision

    def decide(
        self,
        *,
        step: int,
        action_space: Any | None = None,
        reward_context: dict[str, Any] | None = None,
        loss_value: float | None = None,
    ) -> FastSlowDecision:
        if not self.enabled:
            return self._record_decision(
                FastSlowDecision(
                    mode="disabled",
                    accepted=False,
                    alpha=0.0,
                    gate=0.0,
                    effective_alpha=0.0,
                    step=int(step),
                    reason="disabled",
                )
            )

        context = dict(reward_context or {})
        context.setdefault("steps_since_slow_sync", float(self.steps_since_sync(step)))
        context.setdefault("fast_slow_sync_count", float(self.sync_count))
        context.setdefault("fast_slow_last_reward", float(self.last_reward))
        self._apply_loss_reward(
            action_space=action_space,
            step=int(step),
            loss_value=loss_value,
            reward_context=context,
        )

        if action_space is not None and _readiness(action_space, "consolidation") > 0.0:
            gate = float(
                action_space.scalar_value(
                    head_name="consolidation",
                    gpu_step=int(step),
                    fallback=0.0,
                    reward_context=context,
                )
            )
            if _readiness(action_space, "ema_alpha") > 0.0:
                alpha = float(
                    action_space.scalar_value(
                        head_name="ema_alpha",
                        gpu_step=int(step),
                        fallback=float(self.alpha),
                        reward_context=context,
                    )
                )
            else:
                alpha = float(self.alpha)
            gate = _clamp(gate, 0.0, 1.0)
            alpha = _clamp(alpha, 0.0, 1.0)
            effective_alpha = _clamp(gate * alpha, 0.0, 1.0)
            accepted = effective_alpha > 0.0
            decision = FastSlowDecision(
                mode="learned",
                accepted=accepted,
                alpha=alpha,
                gate=gate,
                effective_alpha=effective_alpha,
                step=int(step),
                reason="controller_consolidation_head",
            )
            if accepted:
                self._last_credit_key = int(step)
                recorder = getattr(action_space, "record_credit_assignment", None)
                if callable(recorder):
                    heads = ["consolidation"]
                    if _readiness(action_space, "ema_alpha") > 0.0:
                        heads.append("ema_alpha")
                    recorder(
                        key=int(step),
                        head_names=heads,
                        gpu_step=int(step),
                        reward_context=context,
                    )
            return self._record_decision(decision)

        if self.interval > 0 and int(step) % int(self.interval) == 0:
            return self._record_decision(
                FastSlowDecision(
                    mode="interval",
                    accepted=True,
                    alpha=float(self.alpha),
                    gate=1.0,
                    effective_alpha=float(self.alpha),
                    step=int(step),
                    reason="fixed_interval_fallback",
                )
            )

        return self._record_decision(
            FastSlowDecision(
                mode="hold",
                accepted=False,
                alpha=float(self.alpha),
                gate=0.0,
                effective_alpha=0.0,
                step=int(step),
                reason="not_due",
            )
        )

    def apply_decision(
        self,
        model: torch.nn.Module,
        decision: FastSlowDecision,
    ) -> None:
        self.last_decision = decision
        if not decision.accepted or decision.effective_alpha <= 0.0:
            return

        model_params = dict(model.named_parameters())
        slow_list: list[torch.Tensor] = []
        fast_list: list[torch.Tensor] = []
        alpha = float(decision.effective_alpha)
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
                    slow_param.lerp_(fast_param, alpha)
                    continue
                slow_list.append(slow_param)
                fast_list.append(fast_param)

            if slow_list:
                torch._foreach_lerp_(slow_list, fast_list, alpha)

        self.sync_count += 1
        self.last_sync_step = int(decision.step)
        if decision.mode == "learned":
            self.learned_sync_count += 1

    def apply_external_reward(
        self,
        *,
        action_space: Any | None,
        key: int,
        reward: float,
        step: int,
        reward_context: dict[str, Any] | None = None,
    ) -> int:
        if action_space is None:
            return 0
        reward_f = float(reward)
        if not math.isfinite(reward_f):
            return 0
        applier = getattr(action_space, "apply_reward", None)
        if not callable(applier):
            return 0
        applied = int(
            applier(
                key=int(key),
                reward=reward_f,
                gpu_step=int(step),
                reward_context=dict(reward_context or {}),
            )
        )
        if applied > 0:
            self.reward_update_count += applied
            self.last_reward = reward_f
        return applied

    def copy_slow_to_model(self, model: torch.nn.Module) -> None:
        model_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, slow_param in self.slow_state.items():
                model_param = model_params.get(name)
                if model_param is None:
                    continue
                model_param.copy_(
                    slow_param.to(device=model_param.device, dtype=model_param.dtype)
                )

    def should_copy_slow_to_model_for_eval(self) -> bool:
        """Return true only after the slow copy has received real consolidation."""
        return bool(self.enabled and self.sync_count > 0)

    def diagnostics(self, model: torch.nn.Module) -> dict[str, object]:
        model_params = dict(model.named_parameters())
        diff_sq_sum = 0.0
        diff_count = 0
        for name, slow_param in self.slow_state.items():
            model_param = model_params.get(name)
            if model_param is None:
                continue
            aligned_slow = slow_param.to(
                device=model_param.device,
                dtype=model_param.dtype,
            )
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
            "decision_count": self.decision_count,
            "learned_decision_count": self.learned_decision_count,
            "learned_sync_count": self.learned_sync_count,
            "last_sync_step": self.last_sync_step,
            "steps_since_sync": self.steps_since_sync(None),
            "eval_copy_ready": self.should_copy_slow_to_model_for_eval(),
            "last_reward": self.last_reward,
            "reward_update_count": self.reward_update_count,
            "last_decision": (
                None
                if self.last_decision is None
                else {
                    "mode": self.last_decision.mode,
                    "accepted": self.last_decision.accepted,
                    "alpha": self.last_decision.alpha,
                    "gate": self.last_decision.gate,
                    "effective_alpha": self.last_decision.effective_alpha,
                    "step": self.last_decision.step,
                    "reason": self.last_decision.reason,
                }
            ),
        }

    def steps_since_sync(self, step: int | None) -> int:
        if self.last_sync_step < 0:
            return 0 if step is None else int(step)
        if step is None:
            if self.last_decision is None:
                return 0
            step = self.last_decision.step
        return max(0, int(step) - int(self.last_sync_step))

    def _record_decision(self, decision: FastSlowDecision) -> FastSlowDecision:
        self.decision_count += 1
        if decision.mode == "learned":
            self.learned_decision_count += 1
        self.last_decision = decision
        return decision

    def _apply_loss_reward(
        self,
        *,
        action_space: Any | None,
        step: int,
        loss_value: float | None,
        reward_context: dict[str, Any],
    ) -> None:
        if action_space is None or loss_value is None:
            return
        loss_f = float(loss_value)
        if not math.isfinite(loss_f):
            return
        reward_context["loss"] = loss_f
        if self._last_loss_value is not None and self._last_credit_key is not None:
            reward = float(self._last_loss_value) - loss_f
            applier = getattr(action_space, "apply_reward", None)
            if callable(applier):
                applied = int(
                    applier(
                        key=int(self._last_credit_key),
                        reward=reward,
                        gpu_step=int(step),
                        reward_context=reward_context,
                    )
                )
                if applied > 0:
                    self.reward_update_count += applied
                    self.last_reward = reward
        self._last_loss_value = loss_f
        self._last_credit_key = None


def _readiness(action_space: Any, head_name: str) -> float:
    readiness = getattr(action_space, "readiness", None)
    if not callable(readiness):
        return 0.0
    try:
        value = float(readiness(str(head_name)))
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    x = float(value)
    if not math.isfinite(x):
        return float(lo)
    return max(float(lo), min(float(hi), x))


def fast_slow_decision_to_dict(decision: FastSlowDecision | None) -> dict[str, object] | None:
    if decision is None:
        return None
    return {
        "mode": str(decision.mode),
        "accepted": bool(decision.accepted),
        "alpha": float(decision.alpha),
        "gate": float(decision.gate),
        "effective_alpha": float(decision.effective_alpha),
        "step": int(decision.step),
        "reason": str(decision.reason),
    }


def fast_slow_decision_from_dict(payload: object) -> FastSlowDecision | None:
    if not isinstance(payload, dict):
        return None
    try:
        return FastSlowDecision(
            mode=str(payload.get("mode", "")),
            accepted=bool(payload.get("accepted", False)),
            alpha=float(payload.get("alpha", 0.0)),
            gate=float(payload.get("gate", 0.0)),
            effective_alpha=float(payload.get("effective_alpha", 0.0)),
            step=int(payload.get("step", 0)),
            reason=str(payload.get("reason", "")),
        )
    except Exception:
        return None


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
