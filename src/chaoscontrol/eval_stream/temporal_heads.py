from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from chaoscontrol.eval_stream.delta_mod import DeltaModulator


def uniform_logprob_mixture(log_probs: list[torch.Tensor]) -> torch.Tensor:
    """Uniform probability-space mixture of per-head log-probabilities."""
    if not log_probs:
        raise ValueError("uniform_logprob_mixture requires at least one tensor")
    if len(log_probs) == 1:
        return log_probs[0]

    weight = torch.log(
        torch.tensor(
            1.0 / len(log_probs),
            device=log_probs[0].device,
            dtype=log_probs[0].dtype,
        )
    )
    stacked = torch.stack([logp + weight for logp in log_probs], dim=0)
    return torch.logsumexp(stacked, dim=0)


@dataclass(frozen=True)
class TemporalHeadConfig:
    horizon_shifts: tuple[float, ...] = (-0.5, 0.0, 0.5)
    horizon_knob: Literal["log_a_shift"] = "log_a_shift"
    mixer: Literal["uniform_logprob"] = "uniform_logprob"
    policy: Literal["always", "previous_chunk_priority"] = "always"
    threshold: float | None = None


@dataclass
class TemporalHeadChunkResult:
    loss_nats: float
    tokens_scored: int
    mixed_log_probs: torch.Tensor
    final_states_by_shift: dict[float, list[torch.Tensor]]
    per_head_loss_nats: dict[float, float]


@dataclass
class PreviousChunkPriorityGate:
    """Pre-registered primary gate for running temporal heads on the next chunk."""

    threshold: float
    entropy_weight: float = 1.0
    loss_spike_weight: float = 1.0
    state_delta_weight: float = 1.0
    use_disagreement_ema: bool = False
    disagreement_weight: float = 0.0
    disagreement_decay: float = 0.9

    def __post_init__(self) -> None:
        self._priority = 0.0
        self._disagreement_ema = 0.0

    def update_after_chunk(
        self,
        *,
        entropy: float,
        loss_spike: float,
        state_delta_norm: float,
        head_disagreement: float | None = None,
        temporal_heads_ran: bool = False,
    ) -> None:
        if self.use_disagreement_ema:
            if temporal_heads_ran and head_disagreement is not None:
                self._disagreement_ema = (
                    self.disagreement_decay * self._disagreement_ema
                    + (1.0 - self.disagreement_decay) * float(head_disagreement)
                )
            else:
                self._disagreement_ema *= self.disagreement_decay
        else:
            self._disagreement_ema = 0.0

        self._priority = (
            self.entropy_weight * float(entropy)
            + self.loss_spike_weight * float(loss_spike)
            + self.state_delta_weight * float(state_delta_norm)
            + self.disagreement_weight * self._disagreement_ema
        )

    def should_run(self, *, extra_cost_seconds: float, slack_remaining_seconds: float) -> bool:
        return (
            self._priority > self.threshold
            and float(slack_remaining_seconds) >= float(extra_cost_seconds)
        )


def _nll_from_log_probs(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    token_log_probs = log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -token_log_probs.sum()


def score_temporal_heads_chunk(
    model,
    chunk: torch.Tensor,
    *,
    states_by_shift: dict[float, list[torch.Tensor] | None],
    cfg: TemporalHeadConfig,
) -> TemporalHeadChunkResult:
    """Score one chunk under parallel horizon shifts without updating weights."""
    if chunk.size(1) < 2:
        raise ValueError(
            f"score_temporal_heads_chunk needs chunk length >= 2; got {tuple(chunk.shape)}"
        )
    if cfg.horizon_knob != "log_a_shift":
        raise ValueError(f"unsupported horizon_knob: {cfg.horizon_knob!r}")
    if cfg.mixer != "uniform_logprob":
        raise ValueError(f"unsupported mixer: {cfg.mixer!r}")

    model.eval()
    targets = chunk[:, 1:]
    log_probs: list[torch.Tensor] = []
    final_states_by_shift: dict[float, list[torch.Tensor]] = {}
    per_head_loss_nats: dict[float, float] = {}

    with torch.no_grad():
        for shift in cfg.horizon_shifts:
            kwargs = {}
            initial_states = states_by_shift.get(shift)
            if initial_states:
                kwargs["initial_states"] = initial_states
            with DeltaModulator(model, log_a_shift=shift):
                out = model(chunk, **kwargs)
            logits = out["logits"] if isinstance(out, dict) else out
            head_log_probs = F.log_softmax(logits, dim=-1)
            log_probs.append(head_log_probs)
            per_head_loss_nats[shift] = float(_nll_from_log_probs(head_log_probs, targets).item())
            final_states_by_shift[shift] = (
                [state.detach().clone() for state in out["final_states"]]
                if isinstance(out, dict) and "final_states" in out
                else []
            )

        mixed_log_probs = uniform_logprob_mixture(log_probs)
        loss_nats = float(_nll_from_log_probs(mixed_log_probs, targets).item())

    return TemporalHeadChunkResult(
        loss_nats=loss_nats,
        tokens_scored=chunk.size(1) - 1,
        mixed_log_probs=mixed_log_probs,
        final_states_by_shift=final_states_by_shift,
        per_head_loss_nats=per_head_loss_nats,
    )


def make_same_horizon_virtual_depth_config(
    cfg: dict,
    *,
    depth_recurrence_count: int,
) -> dict:
    """Return a checkpoint config for the deterministic same-horizon control."""
    if depth_recurrence_count < 1:
        raise ValueError("depth_recurrence_count must be >= 1")
    out = dict(cfg)
    shared = list(out.get("depth_recurrence_shared_layers") or [])
    if not shared:
        num_layers = int(out["num_layers"])
        shared = list(range(num_layers))
    out["depth_recurrence_shared_layers"] = shared
    out["depth_recurrence_count"] = int(depth_recurrence_count)
    return out
