"""Concrete calc_type implementations for ``ttt_eval``.

Each module below registers exactly one calc_type via
``@register_calc_type``. Importing this package eagerly imports every
calc_type module so registration is complete by the time any caller
queries ``CALC_TYPE_REGISTRY``.

Registered calc_types:

- ``score_only_reset`` — reset SSM state per doc, no params changed (the floor).
- ``carry_state`` — SSM state continues across docs (with optional decay).
- ``dreamworld_eval`` — per-doc dream-rollout + backward + SGD step.
"""
from __future__ import annotations

# Order matters only for deterministic registration sequence; collisions
# raise at registration time so duplicates surface immediately.
from chaoscontrol.eval.calc_types import score_only_reset  # noqa: F401
from chaoscontrol.eval.calc_types import carry_state  # noqa: F401
from chaoscontrol.eval.calc_types import dreamworld_eval  # noqa: F401
