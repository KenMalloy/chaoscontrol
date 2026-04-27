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
from typing import Any, Sequence


TraceSink = list[dict[str, Any]]


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
        if self.transform not in {"sigmoid", "tanh"}:
            raise ValueError(
                "BoundedScalarSpec.transform must be 'sigmoid' or 'tanh'"
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
        else:
            unit = 0.5 * (math.tanh(raw_f) + 1.0)
        return lo + unit * (hi - lo)


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
    trace_log: TraceSink | ActionSpaceTrace | None = None

    def __post_init__(self) -> None:
        self.selection_readiness = _clamp01(self.selection_readiness)
        if not math.isfinite(float(self.selection_max_delta)):
            raise ValueError("selection_max_delta must be finite")
        if self.selection_max_delta < 0.0:
            raise ValueError("selection_max_delta must be non-negative")
        if self.max_tags_per_query is not None and self.max_tags_per_query < 0:
            raise ValueError("max_tags_per_query must be >= 0")

    @property
    def active_selection(self) -> bool:
        return (
            not self.trace_only
            and self.selection_readiness > 0.0
            and self.selection_max_delta > 0.0
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
        if (
            learned_scores is None
            or len(heuristic) == 0
            or not self.active_selection
        ):
            return heuristic

        learned = [float(x) for x in learned_scores]
        if len(learned) != len(heuristic):
            raise ValueError(
                "learned_scores must have the same length as heuristic_scores"
            )
        scale = float(self.selection_max_delta) * float(self.selection_readiness)
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
            "readiness": float(self.selection_readiness),
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
                "readiness": float(self.selection_readiness),
                "reward_context": dict(reward_context or {}),
                "accepted": budget > 0,
            })
        if budget <= 0:
            return []
        return sorted(
            range(n),
            key=lambda idx: (-float(effective_scores[idx]), idx),
        )[:budget]

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
