from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from chaoscontrol.eval_stream.delta_mod import DeltaModulator


HeadKey = float | str


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


def weighted_logprob_mixture(
    log_probs: list[torch.Tensor],
    *,
    weights: tuple[float, ...],
) -> torch.Tensor:
    """Probability-space mixture with fixed pre-registered head weights."""
    if not log_probs:
        raise ValueError("weighted_logprob_mixture requires at least one tensor")
    if len(log_probs) != len(weights):
        raise ValueError(
            f"weights length {len(weights)} does not match log_probs length {len(log_probs)}"
        )
    weight_tensor = torch.tensor(
        weights,
        device=log_probs[0].device,
        dtype=log_probs[0].dtype,
    )
    if torch.any(weight_tensor <= 0):
        raise ValueError("all mixture weights must be positive")
    weight_tensor = weight_tensor / weight_tensor.sum()
    view_shape = (len(weights),) + (1,) * log_probs[0].ndim
    stacked = torch.stack(log_probs, dim=0)
    return torch.logsumexp(stacked + weight_tensor.log().view(view_shape), dim=0)


def online_exp_weighted_logprob_mixture(
    log_probs: list[torch.Tensor],
    targets: torch.Tensor,
    *,
    eta: float,
    initial_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Causal token-level exponential-weights probability mixture.

    The pre-update weights score token t. Only after token t is scored are the
    per-head losses for token t used to update weights for token t+1.
    """
    if not log_probs:
        raise ValueError(
            "online_exp_weighted_logprob_mixture requires at least one tensor"
        )
    if len(log_probs) == 1:
        return log_probs[0]
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    shape = log_probs[0].shape
    if len(shape) != 3:
        raise ValueError("log_probs tensors must have shape [batch, seq, vocab]")
    if any(logp.shape != shape for logp in log_probs):
        raise ValueError("all log_probs tensors must have the same shape")
    batch, seq_len, _vocab = shape
    if targets.shape != (batch, seq_len - 1):
        raise ValueError(
            f"targets shape {tuple(targets.shape)} does not match "
            f"expected {(batch, seq_len - 1)}"
        )

    if initial_weights is None:
        initial_weights = tuple(1.0 for _ in log_probs)
    weight_tensor = torch.tensor(
        initial_weights,
        device=log_probs[0].device,
        dtype=log_probs[0].dtype,
    )
    if len(weight_tensor) != len(log_probs):
        raise ValueError(
            f"initial_weights length {len(weight_tensor)} does not match "
            f"log_probs length {len(log_probs)}"
        )
    if torch.any(weight_tensor <= 0):
        raise ValueError("all initial online weights must be positive")
    log_weights = (weight_tensor / weight_tensor.sum()).log()
    log_weights = log_weights.unsqueeze(0).expand(batch, -1).clone()
    stacked = torch.stack(log_probs, dim=1)
    head_count = len(log_probs)

    mixed_positions: list[torch.Tensor] = []
    for pos in range(seq_len):
        mixed_positions.append(
            torch.logsumexp(
                stacked[:, :, pos, :] + log_weights.unsqueeze(-1),
                dim=1,
            )
        )
        if pos < targets.size(1):
            target_idx = targets[:, pos].view(batch, 1, 1).expand(-1, head_count, 1)
            token_log_probs = (
                stacked[:, :, pos, :]
                .gather(-1, target_idx)
                .squeeze(-1)
            )
            log_weights = torch.log_softmax(
                log_weights + float(eta) * token_log_probs,
                dim=1,
            )

    return torch.stack(mixed_positions, dim=1)


@dataclass(frozen=True)
class TemporalHeadConfig:
    horizon_shifts: tuple[float, ...] = (-0.5, 0.0, 0.5)
    head_ids: tuple[str, ...] | None = None
    horizon_knob: Literal["log_a_shift"] = "log_a_shift"
    mixer: Literal[
        "uniform_logprob",
        "base_prior_logprob",
        "online_exp_weights_logprob",
    ] = "uniform_logprob"
    mixer_weights: tuple[float, ...] | None = None
    online_eta: float = 1.0
    online_initial_weights: tuple[float, ...] | None = None
    policy: Literal["always", "previous_chunk_priority"] = "always"
    threshold: float | None = None


@dataclass
class TemporalHeadChunkResult:
    loss_nats: float
    tokens_scored: int
    mixed_log_probs: torch.Tensor
    final_states_by_shift: dict[HeadKey, list[torch.Tensor]]
    per_head_loss_nats: dict[HeadKey, float]
    winner_counts_by_shift: dict[HeadKey, int]
    half_life_stats_by_shift: dict[HeadKey, list[dict[str, float | int | None]]]
    state_divergence_by_shift: dict[HeadKey, list[dict[str, float | int]]]


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


def _token_nll_from_log_probs(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _find_ssm_cores(model) -> list:
    return [m for m in model.modules() if type(m).__name__ == "ChaosSSMCore"]


def _head_keys(cfg: TemporalHeadConfig) -> tuple[HeadKey, ...]:
    if cfg.head_ids is None:
        if len(set(cfg.horizon_shifts)) != len(cfg.horizon_shifts):
            raise ValueError("duplicate horizon_shifts require explicit head_ids")
        return cfg.horizon_shifts
    if len(cfg.head_ids) != len(cfg.horizon_shifts):
        raise ValueError(
            f"head_ids length {len(cfg.head_ids)} does not match "
            f"horizon_shifts length {len(cfg.horizon_shifts)}"
        )
    if len(set(cfg.head_ids)) != len(cfg.head_ids):
        raise ValueError("head_ids must be unique")
    return cfg.head_ids


def _base_head_key(
    head_keys: tuple[HeadKey, ...],
    horizon_shifts: tuple[float, ...],
) -> HeadKey | None:
    for head_key, shift in zip(head_keys, horizon_shifts, strict=True):
        if shift == 0.0:
            return head_key
    return None


def _capture_delta_means(model) -> tuple[list, list[list[torch.Tensor]]]:
    cores = _find_ssm_cores(model)
    captures: list[list[torch.Tensor]] = [[] for _ in cores]
    handles = []
    for capture, core in zip(captures, cores, strict=True):
        handles.append(
            core.delta_proj.register_forward_hook(
                lambda _mod, _inp, out, bucket=capture: bucket.append(
                    F.softplus(out.detach()).clamp_min(1e-4)
                )
            )
        )
    return handles, captures


def _remove_hooks(handles: list) -> None:
    for handle in handles:
        handle.remove()


def _mean_delta_by_layer(captures: list[list[torch.Tensor]]) -> list[torch.Tensor | None]:
    out: list[torch.Tensor | None] = []
    for layer_captures in captures:
        if not layer_captures:
            out.append(None)
            continue
        flattened = [
            delta.reshape(-1, delta.shape[-1])
            for delta in layer_captures
        ]
        out.append(torch.cat(flattened, dim=0).mean(dim=0))
    return out


def _half_lives_for_shift(
    cores: list,
    mean_deltas: list[torch.Tensor | None],
    *,
    shift: float,
) -> list[torch.Tensor | None]:
    half_lives: list[torch.Tensor | None] = []
    for core, mean_delta in zip(cores, mean_deltas, strict=True):
        if mean_delta is None or not hasattr(core, "log_a"):
            half_lives.append(None)
            continue
        a_base = torch.sigmoid(core.log_a.detach().to(mean_delta.device) + float(shift))
        a_base = a_base.to(dtype=mean_delta.dtype)
        if mean_delta.numel() == 1 and a_base.numel() > 1:
            mean_delta = mean_delta.expand_as(a_base)
        if mean_delta.shape != a_base.shape:
            half_lives.append(None)
            continue
        ln2 = torch.log(
            torch.tensor(2.0, device=mean_delta.device, dtype=mean_delta.dtype)
        )
        half_lives.append(ln2 / (mean_delta * a_base).clamp_min(1e-8))
    return half_lives


def _summarize_half_life_stats(
    raw_half_lives_by_shift: dict[HeadKey, list[torch.Tensor | None]],
    *,
    base_key: HeadKey | None,
) -> dict[HeadKey, list[dict[str, float | int | None]]]:
    base_layers = (
        raw_half_lives_by_shift.get(base_key)
        if base_key is not None
        else None
    )
    summary: dict[HeadKey, list[dict[str, float | int | None]]] = {}
    for shift, half_lives_by_layer in raw_half_lives_by_shift.items():
        layer_stats: list[dict[str, float | int | None]] = []
        for layer_idx, half_lives in enumerate(half_lives_by_layer):
            if half_lives is None:
                layer_stats.append(
                    {
                        "layer": layer_idx,
                        "p10": None,
                        "median": None,
                        "p90": None,
                        "separated_fraction_vs_base": None,
                    }
                )
                continue
            values = half_lives.detach().float().reshape(-1).cpu()
            separated_fraction = None
            if base_layers is not None and layer_idx < len(base_layers):
                base_values = base_layers[layer_idx]
                if base_values is not None and base_values.shape == half_lives.shape:
                    ratio = torch.log2(
                        (
                            half_lives.detach().float()
                            / base_values.detach().float()
                        ).clamp_min(1e-8)
                    )
                    separated_fraction = float((ratio.abs() >= 0.5).float().mean().item())
            layer_stats.append(
                {
                    "layer": layer_idx,
                    "p10": float(torch.quantile(values, 0.10).item()),
                    "median": float(torch.quantile(values, 0.50).item()),
                    "p90": float(torch.quantile(values, 0.90).item()),
                    "separated_fraction_vs_base": separated_fraction,
                }
            )
        summary[shift] = layer_stats
    return summary


def _state_divergence_by_shift(
    final_states_by_shift: dict[HeadKey, list[torch.Tensor]],
    *,
    base_key: HeadKey | None,
) -> dict[HeadKey, list[dict[str, float | int]]]:
    if base_key is None:
        return {}
    base_states = final_states_by_shift.get(base_key)
    if not base_states:
        return {}
    out: dict[HeadKey, list[dict[str, float | int]]] = {}
    for shift, states in final_states_by_shift.items():
        if shift == base_key:
            continue
        if len(states) != len(base_states):
            continue
        layer_stats: list[dict[str, float | int]] = []
        for layer_idx, (state, base_state) in enumerate(
            zip(states, base_states, strict=True)
        ):
            delta = state.detach().float() - base_state.detach().float()
            cosine = F.cosine_similarity(
                state.detach().float(),
                base_state.detach().float(),
                dim=-1,
            )
            layer_stats.append(
                {
                    "layer": layer_idx,
                    "l2_vs_base": float(delta.norm(dim=-1).mean().item()),
                    "cosine_vs_base": float(cosine.mean().item()),
                }
            )
        out[shift] = layer_stats
    return out


def _default_base_prior_weights(horizon_shifts: tuple[float, ...]) -> tuple[float, ...]:
    if len(horizon_shifts) != 3 or 0.0 not in horizon_shifts:
        raise ValueError(
            "base_prior_logprob default weights require exactly three heads including 0.0"
        )
    side_weight = 0.1
    base_weight = 0.8
    return tuple(base_weight if shift == 0.0 else side_weight for shift in horizon_shifts)


def score_temporal_heads_chunk(
    model,
    chunk: torch.Tensor,
    *,
    states_by_shift: dict[HeadKey, list[torch.Tensor] | None],
    cfg: TemporalHeadConfig,
) -> TemporalHeadChunkResult:
    """Score one chunk under parallel horizon shifts without updating weights."""
    if chunk.size(1) < 2:
        raise ValueError(
            f"score_temporal_heads_chunk needs chunk length >= 2; got {tuple(chunk.shape)}"
        )
    if cfg.horizon_knob != "log_a_shift":
        raise ValueError(f"unsupported horizon_knob: {cfg.horizon_knob!r}")
    if cfg.mixer not in (
        "uniform_logprob",
        "base_prior_logprob",
        "online_exp_weights_logprob",
    ):
        raise ValueError(f"unsupported mixer: {cfg.mixer!r}")

    model.eval()
    targets = chunk[:, 1:]
    head_keys = _head_keys(cfg)
    base_key = _base_head_key(head_keys, cfg.horizon_shifts)
    log_probs: list[torch.Tensor] = []
    final_states_by_shift: dict[HeadKey, list[torch.Tensor]] = {}
    per_head_loss_nats: dict[HeadKey, float] = {}
    raw_half_lives_by_shift: dict[HeadKey, list[torch.Tensor | None]] = {}
    cores = _find_ssm_cores(model)

    with torch.no_grad():
        for head_key, shift in zip(head_keys, cfg.horizon_shifts, strict=True):
            kwargs = {}
            initial_states = states_by_shift.get(head_key)
            if initial_states:
                kwargs["initial_states"] = initial_states
            handles, delta_captures = _capture_delta_means(model)
            with DeltaModulator(model, log_a_shift=shift):
                try:
                    out = model(chunk, **kwargs)
                finally:
                    _remove_hooks(handles)
            raw_half_lives_by_shift[head_key] = _half_lives_for_shift(
                cores,
                _mean_delta_by_layer(delta_captures),
                shift=shift,
            )
            logits = out["logits"] if isinstance(out, dict) else out
            head_log_probs = F.log_softmax(logits, dim=-1)
            log_probs.append(head_log_probs)
            per_head_loss_nats[head_key] = float(
                _nll_from_log_probs(head_log_probs, targets).item()
            )
            final_states_by_shift[head_key] = (
                [state.detach().clone() for state in out["final_states"]]
                if isinstance(out, dict) and "final_states" in out
                else []
            )

        if cfg.mixer == "uniform_logprob":
            mixed_log_probs = uniform_logprob_mixture(log_probs)
        elif cfg.mixer == "base_prior_logprob":
            weights = cfg.mixer_weights or _default_base_prior_weights(cfg.horizon_shifts)
            mixed_log_probs = weighted_logprob_mixture(log_probs, weights=weights)
        else:
            mixed_log_probs = online_exp_weighted_logprob_mixture(
                log_probs,
                targets,
                eta=cfg.online_eta,
                initial_weights=cfg.online_initial_weights,
            )
        loss_nats = float(_nll_from_log_probs(mixed_log_probs, targets).item())
        per_token_nll = torch.stack(
            [_token_nll_from_log_probs(logp, targets) for logp in log_probs],
            dim=0,
        )
        winners = per_token_nll.argmin(dim=0)
        winner_counts_by_shift = {
            head_key: int((winners == idx).sum().item())
            for idx, head_key in enumerate(head_keys)
        }

    return TemporalHeadChunkResult(
        loss_nats=loss_nats,
        tokens_scored=int(targets.numel()),
        mixed_log_probs=mixed_log_probs,
        final_states_by_shift=final_states_by_shift,
        per_head_loss_nats=per_head_loss_nats,
        winner_counts_by_shift=winner_counts_by_shift,
        half_life_stats_by_shift=_summarize_half_life_stats(
            raw_half_lives_by_shift,
            base_key=base_key,
        ),
        state_divergence_by_shift=_state_divergence_by_shift(
            final_states_by_shift,
            base_key=base_key,
        ),
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
