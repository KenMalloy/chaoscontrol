"""Bounded learned action space for the episodic controller.

This module is the first implementation slice of the "learned controller
action-space" design. The important contract is conservative:

* a learned head may express only bounded residuals over an existing legal
  heuristic action;
* readiness gates scale those residuals from 0..1, where 0 is exact heuristic
  identity;
* hard bounds/clamps are traceable so a run can stop runaway learning without
  hiding that it happened.

The CPU SSM / simplex learners can grow into this surface one head at a time.
For now the controller uses the ``selection_rank`` head to rerank replay slots
inside the existing fixed query budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping, Sequence


TraceSink = list[dict[str, Any]]


DEFAULT_EVENT_FEATURES: tuple[str, ...] = (
    "bias",
    "pressure",
    "ce",
    "score",
    "rank",
    "utility",
    "age",
    "bucket",
    "entropy",
    "reward",
    "finite_reward",
    "cache_fill",
    "wall_pressure",
)

DEFAULT_HEADS: tuple[str, ...] = (
    "write_admission",
    "eviction",
    "replay_timing",
    "replay_budget",
    "write_budget",
    "temperature",
    "entropy_beta",
    "ema_alpha",
    "consolidation",
    "selection_rank",
)


@dataclass(frozen=True)
class BoundedScalarSpec:
    """Map an unconstrained scalar to a legal controller knob range."""

    name: str
    minimum: float
    maximum: float
    transform: str = "sigmoid"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.minimum)):
            raise ValueError(f"{self.name}.minimum must be finite")
        if not math.isfinite(float(self.maximum)):
            raise ValueError(f"{self.name}.maximum must be finite")
        if float(self.maximum) < float(self.minimum):
            raise ValueError(f"{self.name}.maximum must be >= minimum")
        if self.transform not in {"sigmoid", "tanh", "softplus"}:
            raise ValueError(
                "BoundedScalarSpec.transform must be 'sigmoid', 'tanh', "
                "or 'softplus'"
            )

    def map(self, raw: float) -> float:
        raw_f = float(raw)
        if not math.isfinite(raw_f):
            return float(self.minimum)
        lo = float(self.minimum)
        hi = float(self.maximum)
        if hi == lo:
            return lo
        if self.transform == "sigmoid":
            # Branch-stable sigmoid avoids overflow on very large logits.
            if raw_f >= 0.0:
                z = math.exp(-raw_f)
                unit = 1.0 / (1.0 + z)
            else:
                z = math.exp(raw_f)
                unit = z / (1.0 + z)
        elif self.transform == "tanh":
            unit = 0.5 * (math.tanh(raw_f) + 1.0)
        else:
            # Softplus is useful for temperatures. We still map into a closed
            # interval because the controller action space owns the hard cap.
            if raw_f > 20.0:
                sp = raw_f
            else:
                sp = math.log1p(math.exp(raw_f))
            unit = sp / (1.0 + sp)
        return lo + unit * (hi - lo)


@dataclass
class SharedEventSsm:
    """Tiny diagonal recurrent state shared by the learned action heads.

    This is intentionally CPU/Python and deterministic. It is not the final AMX
    runtime; it is the shape-compatible substrate that lets the runner consume
    one learned recurrent state for write, eviction, replay, and meta heads.
    """

    hidden_dim: int = 16
    feature_names: Sequence[str] = DEFAULT_EVENT_FEATURES
    head_names: Sequence[str] = DEFAULT_HEADS
    seed: int = 0
    decay: float = 0.95
    input_scale: float = 0.05
    head_scale: float = 0.05
    hidden: list[float] = field(init=False)
    input_weights: list[list[float]] = field(init=False)
    head_weights: dict[str, list[float]] = field(init=False)
    head_bias: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("SharedEventSsm.hidden_dim must be positive")
        if not 0.0 <= float(self.decay) <= 1.0:
            raise ValueError("SharedEventSsm.decay must be in [0, 1]")
        rng = random.Random(int(self.seed))
        self.hidden = [0.0] * int(self.hidden_dim)
        self.feature_names = tuple(str(x) for x in self.feature_names)
        self.head_names = tuple(str(x) for x in self.head_names)
        self.input_weights = [
            [
                (rng.random() * 2.0 - 1.0) * float(self.input_scale)
                for _ in self.feature_names
            ]
            for _ in range(int(self.hidden_dim))
        ]
        self.head_weights = {
            head: [
                (rng.random() * 2.0 - 1.0) * float(self.head_scale)
                for _ in range(int(self.hidden_dim))
            ]
            for head in self.head_names
        }
        self.head_bias = {head: 0.0 for head in self.head_names}

    def observe(
        self,
        features: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        x = self._feature_vector(features or {})
        keep = float(self.decay)
        for i in range(int(self.hidden_dim)):
            drive = 0.0
            weights = self.input_weights[i]
            for j, value in enumerate(x):
                drive += weights[j] * value
            self.hidden[i] = keep * self.hidden[i] + math.tanh(drive)
        return self.head_outputs()

    def head_outputs(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for head, weights in self.head_weights.items():
            raw = float(self.head_bias.get(head, 0.0))
            for h, w in zip(self.hidden, weights, strict=True):
                raw += h * w
            out[head] = raw
        return out

    def head_output(self, head_name: str) -> float:
        return float(self.head_outputs().get(str(head_name), 0.0))

    def update_head_from_hidden(
        self,
        *,
        head_name: str,
        hidden: Sequence[float],
        reward: float,
        learning_rate: float,
        reward_clip: float = 5.0,
        weight_clip: float = 1.0,
    ) -> None:
        head = str(head_name)
        if head not in self.head_weights:
            return
        lr = float(learning_rate)
        if lr <= 0.0:
            return
        r = float(reward)
        if not math.isfinite(r):
            return
        clip = abs(float(reward_clip))
        if clip > 0.0:
            r = max(-clip, min(clip, r))
        update = lr * r
        weights = self.head_weights[head]
        limit = abs(float(weight_clip))
        for i, h in enumerate(hidden):
            if i >= len(weights):
                break
            w = weights[i] + update * float(h)
            weights[i] = max(-limit, min(limit, w)) if limit > 0.0 else w
        b = float(self.head_bias.get(head, 0.0)) + update
        self.head_bias[head] = max(-limit, min(limit, b)) if limit > 0.0 else b

    def _feature_vector(self, features: Mapping[str, float]) -> list[float]:
        out: list[float] = []
        for name in self.feature_names:
            if name == "bias":
                out.append(1.0)
                continue
            value = float(features.get(name, 0.0))
            out.append(value if math.isfinite(value) else 0.0)
        return out


@dataclass
class ActionSpaceTrace:
    """Small in-memory trace sink; B4/F-style writers can swap this later."""

    rows: TraceSink = field(default_factory=list)

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


@dataclass
class ConstrainedActionSpace:
    """Learned residual action space with readiness and invariant logging."""

    trace_only: bool = False
    selection_readiness: float = 0.0
    selection_max_delta: float = 0.0
    max_tags_per_query: int | None = None
    head_readiness: dict[str, float] = field(default_factory=dict)
    head_max_delta: dict[str, float] = field(default_factory=dict)
    scalar_specs: dict[str, BoundedScalarSpec] = field(default_factory=dict)
    event_ssm: SharedEventSsm | None = None
    trace_log: TraceSink | ActionSpaceTrace | None = None
    online_learning_rate: float = 0.0
    reward_clip: float = 5.0
    credit_history: dict[int, list[tuple[str, list[float]]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.selection_readiness = _clamp01(self.selection_readiness)
        if not math.isfinite(float(self.selection_max_delta)):
            raise ValueError("selection_max_delta must be finite")
        if self.selection_max_delta < 0.0:
            raise ValueError("selection_max_delta must be non-negative")
        if self.max_tags_per_query is not None and self.max_tags_per_query < 0:
            raise ValueError("max_tags_per_query must be >= 0")
        self.head_readiness = {
            str(k): _clamp01(float(v)) for k, v in self.head_readiness.items()
        }
        self.head_max_delta = {
            str(k): _finite_nonnegative(float(v), name=f"{k}.max_delta")
            for k, v in self.head_max_delta.items()
        }
        if not self.scalar_specs:
            self.scalar_specs = {
                "temperature": BoundedScalarSpec(
                    "temperature", 0.25, 4.0, transform="softplus"
                ),
                "entropy_beta": BoundedScalarSpec(
                    "entropy_beta", 0.0, 0.2, transform="sigmoid"
                ),
                "ema_alpha": BoundedScalarSpec(
                    "ema_alpha", 0.0, 0.5, transform="sigmoid"
                ),
                "replay_budget": BoundedScalarSpec(
                    "replay_budget", 0.0, 16.0, transform="sigmoid"
                ),
                "write_budget": BoundedScalarSpec(
                    "write_budget", 0.0, 16.0, transform="sigmoid"
                ),
                "eviction": BoundedScalarSpec(
                    "eviction", 0.0, 1.0, transform="sigmoid"
                ),
            }
        if self.online_learning_rate < 0.0:
            raise ValueError("online_learning_rate must be non-negative")

    @property
    def active_selection(self) -> bool:
        return (
            not self.trace_only
            and self.selection_readiness > 0.0
            and self.selection_max_delta > 0.0
        )

    def readiness(self, head_name: str) -> float:
        head = str(head_name)
        if head == "selection_rank":
            return float(self.head_readiness.get(head, self.selection_readiness))
        return float(self.head_readiness.get(head, 0.0))

    def max_delta(self, head_name: str) -> float:
        head = str(head_name)
        if head == "selection_rank":
            return float(self.head_max_delta.get(head, self.selection_max_delta))
        return float(self.head_max_delta.get(head, 0.0))

    def active_head(self, head_name: str) -> bool:
        return (
            not self.trace_only
            and self.readiness(head_name) > 0.0
            and self.max_delta(head_name) > 0.0
        )

    def effective_scores(
        self,
        *,
        heuristic_scores: Sequence[float],
        learned_scores: Sequence[float] | None,
        gpu_step: int,
        head_name: str = "selection_rank",
        reward_context: dict[str, Any] | None = None,
    ) -> list[float]:
        """Return legal effective scores for ranking.

        With ``trace_only=True``, no learned scores, zero readiness, or zero
        delta budget, this returns the heuristic scores bit-for-bit as Python
        floats. Once active, each learned score contributes
        ``tanh(raw) * selection_max_delta * readiness``.
        """
        heuristic = [float(x) for x in heuristic_scores]
        if len(heuristic) == 0 or not self.active_head(head_name):
            return heuristic
        if learned_scores is None and self.event_ssm is not None:
            learned_scores = [
                self.event_ssm.observe(
                    _candidate_features(
                        score=float(score),
                        rank=idx,
                        extra=reward_context,
                    )
                ).get(str(head_name), 0.0)
                for idx, score in enumerate(heuristic)
            ]

        if learned_scores is None:
            return heuristic

        learned = [float(x) for x in learned_scores]
        if len(learned) != len(heuristic):
            raise ValueError(
                "learned_scores must have the same length as heuristic_scores"
            )
        readiness = self.readiness(head_name)
        scale = self.max_delta(head_name) * readiness
        deltas = [_finite_tanh(x) * scale for x in learned]
        effective = [h + d for h, d in zip(heuristic, deltas, strict=True)]
        self._record({
            "gpu_step": int(gpu_step),
            "event_type": "action_space_delta",
            "head_name": str(head_name),
            "raw_action": learned,
            "bounded_action": deltas,
            "invariant_name": "bounded_residual",
            "clamp_amount": 0.0,
            "readiness": float(readiness),
            "reward_context": dict(reward_context or {}),
            "accepted": True,
        })
        return effective

    def selected_indices(
        self,
        *,
        effective_scores: Sequence[float],
        gpu_step: int,
        requested_budget: int,
        head_name: str = "replay_budget",
        reward_context: dict[str, Any] | None = None,
    ) -> list[int]:
        """Choose legal replay indices under the per-query tag budget."""
        n = len(effective_scores)
        requested = int(requested_budget)
        if requested < 0:
            requested = 0
        cap = n if self.max_tags_per_query is None else int(self.max_tags_per_query)
        budget = min(n, requested, cap)
        if budget < min(n, requested):
            self._record({
                "gpu_step": int(gpu_step),
                "event_type": "action_space_clamp",
                "head_name": str(head_name),
                "raw_action": int(requested),
                "bounded_action": int(budget),
                "invariant_name": "max_tags_per_query",
                "clamp_amount": float(min(n, requested) - budget),
                "readiness": float(self.readiness(head_name)),
                "reward_context": dict(reward_context or {}),
                "accepted": budget > 0,
            })
        if budget <= 0:
            return []
        return sorted(
            range(n),
            key=lambda idx: (-float(effective_scores[idx]), idx),
        )[:budget]

    def scalar_value(
        self,
        *,
        head_name: str,
        raw_value: float | None = None,
        gpu_step: int,
        fallback: float,
        reward_context: dict[str, Any] | None = None,
    ) -> float:
        """Return a bounded scalar meta-action.

        Trace-only and zero-readiness return ``fallback`` exactly. When active,
        the scalar is mapped by its ``BoundedScalarSpec`` and then blended from
        fallback by readiness, which keeps meta-knobs ordered and slow-moving.
        """
        head = str(head_name)
        readiness = self.readiness(head)
        if self.trace_only or readiness <= 0.0:
            return float(fallback)
        raw = raw_value
        if raw is None and self.event_ssm is not None:
            raw = self.event_ssm.observe(reward_context or {}).get(head, 0.0)
        if raw is None:
            return float(fallback)
        spec = self.scalar_specs.get(head)
        if spec is None:
            raise KeyError(f"no scalar spec configured for head {head!r}")
        bounded = float(spec.map(float(raw)))
        value = (1.0 - readiness) * float(fallback) + readiness * bounded
        self._record({
            "gpu_step": int(gpu_step),
            "event_type": "action_space_scalar",
            "head_name": head,
            "raw_action": float(raw),
            "bounded_action": float(value),
            "invariant_name": f"{head}_range",
            "clamp_amount": 0.0,
            "readiness": float(readiness),
            "reward_context": dict(reward_context or {}),
            "accepted": True,
        })
        return float(value)

    def record_credit_assignment(
        self,
        *,
        key: int,
        head_names: Sequence[str],
        gpu_step: int,
        reward_context: dict[str, Any] | None = None,
    ) -> None:
        if self.event_ssm is None or self.online_learning_rate <= 0.0:
            return
        entries = [
            (str(head), list(self.event_ssm.hidden))
            for head in head_names
            if str(head) in self.event_ssm.head_weights
        ]
        if not entries:
            return
        self.credit_history[int(key)] = entries
        self._record({
            "gpu_step": int(gpu_step),
            "event_type": "action_space_credit_record",
            "head_name": ",".join(head for head, _ in entries),
            "raw_action": int(key),
            "bounded_action": len(entries),
            "invariant_name": "reward_key",
            "clamp_amount": 0.0,
            "readiness": 1.0,
            "reward_context": dict(reward_context or {}),
            "accepted": True,
        })

    def apply_reward(
        self,
        *,
        key: int,
        reward: float,
        gpu_step: int,
        reward_context: dict[str, Any] | None = None,
    ) -> int:
        if self.event_ssm is None or self.online_learning_rate <= 0.0:
            return 0
        entries = self.credit_history.pop(int(key), [])
        if not entries:
            return 0
        applied = 0
        for head, hidden in entries:
            self.event_ssm.update_head_from_hidden(
                head_name=head,
                hidden=hidden,
                reward=float(reward),
                learning_rate=float(self.online_learning_rate),
                reward_clip=float(self.reward_clip),
            )
            applied += 1
        self._record({
            "gpu_step": int(gpu_step),
            "event_type": "action_space_reward_update",
            "head_name": ",".join(head for head, _ in entries),
            "raw_action": float(reward),
            "bounded_action": int(applied),
            "invariant_name": "clipped_reward_update",
            "clamp_amount": 0.0,
            "readiness": 1.0,
            "reward_context": dict(reward_context or {}),
            "accepted": applied > 0,
        })
        return int(applied)

    def _record(self, row: dict[str, Any]) -> None:
        sink = self.trace_log
        if sink is None:
            return
        if isinstance(sink, ActionSpaceTrace):
            sink.append(row)
        else:
            sink.append(row)


def coerce_action_space(value: Any) -> ConstrainedActionSpace | None:
    """Build a ``ConstrainedActionSpace`` from config-like values."""
    if value is None or value is False:
        return None
    if isinstance(value, ConstrainedActionSpace):
        return value
    if isinstance(value, dict):
        return ConstrainedActionSpace(**value)
    raise TypeError(
        "action_space must be None, False, a dict, or ConstrainedActionSpace"
    )


def make_shared_event_ssm_from_config(config: Mapping[str, Any]) -> SharedEventSsm:
    return SharedEventSsm(
        hidden_dim=int(config.get("episodic_controller_ssm_hidden_dim", 16)),
        seed=int(config.get("episodic_controller_ssm_seed", 0)),
        decay=float(config.get("episodic_controller_ssm_decay", 0.95)),
        input_scale=float(config.get("episodic_controller_ssm_input_scale", 0.05)),
        head_scale=float(config.get("episodic_controller_ssm_head_scale", 0.05)),
    )


def _clamp01(value: float) -> float:
    x = float(value)
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _finite_tanh(value: float) -> float:
    x = float(value)
    if not math.isfinite(x):
        return 0.0
    return math.tanh(x)


def _finite_nonnegative(value: float, *, name: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if x < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return x


def _candidate_features(
    *,
    score: float,
    rank: int,
    extra: Mapping[str, Any] | None,
) -> dict[str, float]:
    features = {"score": float(score), "rank": float(rank)}
    if extra:
        for key, value in extra.items():
            if isinstance(value, (int, float)):
                features[str(key)] = float(value)
    return features
