"""Learned controller action-space gates.

The gates make learned controller behavior ordered rather than runaway:
zero readiness is exact heuristic identity, nonzero readiness contributes only
bounded residuals, and hard budget clamps are logged.
"""
from __future__ import annotations

import math

import pytest

from chaoscontrol.episodic.learned_action_space import (
    BoundedScalarSpec,
    ConstrainedActionSpace,
    SharedEventSsm,
)


def test_zero_readiness_reproduces_heuristic_scores_exactly():
    heuristic = [0.25, -0.0, 1.5, -3.25]
    action_space = ConstrainedActionSpace(
        selection_readiness=0.0,
        selection_max_delta=100.0,
    )

    out = action_space.effective_scores(
        heuristic_scores=heuristic,
        learned_scores=[1000.0, -1000.0, 7.0, -7.0],
        gpu_step=17,
    )

    assert out == heuristic


def test_trace_only_reproduces_heuristic_scores_exactly():
    heuristic = [0.5, 0.25, 0.125]
    trace: list[dict] = []
    action_space = ConstrainedActionSpace(
        trace_only=True,
        selection_readiness=1.0,
        selection_max_delta=10.0,
        trace_log=trace,
    )

    out = action_space.effective_scores(
        heuristic_scores=heuristic,
        learned_scores=[-10.0, 10.0, 0.0],
        gpu_step=99,
    )

    assert out == heuristic
    assert trace == []


def test_bounded_delta_never_exceeds_configured_max_delta():
    action_space = ConstrainedActionSpace(
        selection_readiness=0.5,
        selection_max_delta=0.2,
    )

    out = action_space.effective_scores(
        heuristic_scores=[1.0, 1.0, 1.0],
        learned_scores=[1e9, -1e9, float("nan")],
        gpu_step=1,
    )

    deltas = [score - 1.0 for score in out]
    assert deltas[0] == pytest.approx(0.1)
    assert deltas[1] == pytest.approx(-0.1)
    assert deltas[2] == pytest.approx(0.0)
    assert all(abs(delta) <= 0.100001 for delta in deltas)


def test_budget_clamp_logs_trace_row():
    trace: list[dict] = []
    action_space = ConstrainedActionSpace(
        selection_readiness=1.0,
        selection_max_delta=1.0,
        max_tags_per_query=2,
        trace_log=trace,
    )

    selected = action_space.selected_indices(
        effective_scores=[0.1, 0.9, 0.8, 0.2],
        gpu_step=44,
        requested_budget=4,
    )

    assert selected == [1, 2]
    assert len(trace) == 1
    row = trace[0]
    assert row["event_type"] == "action_space_clamp"
    assert row["head_name"] == "replay_budget"
    assert row["raw_action"] == 4
    assert row["bounded_action"] == 2
    assert row["invariant_name"] == "max_tags_per_query"
    assert row["clamp_amount"] == pytest.approx(2.0)
    assert row["accepted"] is True


def test_bounded_scalar_spec_maps_to_closed_interval():
    spec = BoundedScalarSpec(
        name="entropy_beta",
        minimum=0.0,
        maximum=0.2,
    )

    values = [spec.map(x) for x in [-1000.0, -1.0, 0.0, 1.0, 1000.0]]

    assert all(0.0 <= value <= 0.2 for value in values)
    assert values[2] == pytest.approx(0.1)
    assert values == sorted(values)


def test_bounded_scalar_spec_nan_maps_to_minimum():
    spec = BoundedScalarSpec(
        name="write_rate",
        minimum=0.01,
        maximum=0.2,
        transform="tanh",
    )

    assert math.isfinite(spec.map(float("nan")))
    assert spec.map(float("nan")) == pytest.approx(0.01)


def test_shared_event_ssm_is_deterministic_for_same_stream():
    a = SharedEventSsm(hidden_dim=8, seed=123)
    b = SharedEventSsm(hidden_dim=8, seed=123)
    events = [
        {"pressure": 0.2, "ce": 1.5, "score": 0.3, "bucket": 0.0},
        {"pressure": 0.7, "ce": 2.0, "score": 1.4, "bucket": 3.0},
    ]

    out_a = [a.observe(event) for event in events]
    out_b = [b.observe(event) for event in events]

    assert out_a == out_b


def test_shared_event_ssm_is_full_a_and_tiny_vs_bare_trunk_work():
    ssm = SharedEventSsm(hidden_dim=16, seed=7)

    diag = ssm.diagnostics()
    ratio = ssm.work_ratio_vs_bare_ssm(
        batch=1024,
        seq=512,
        dim=384,
        layers=4,
    )

    assert diag["recurrence_mode"] == "full_a_event_ssm"
    assert diag["top_log_sv"] <= ssm.target_log_sv + 1.0e-5
    assert ratio < 1.0e-5


def test_scalar_head_blends_from_fallback_by_readiness():
    trace: list[dict] = []
    action_space = ConstrainedActionSpace(
        head_readiness={"temperature": 0.5},
        trace_log=trace,
    )

    value = action_space.scalar_value(
        head_name="temperature",
        raw_value=0.0,
        gpu_step=5,
        fallback=1.0,
        reward_context={"score": 0.25},
    )

    # temperature raw=0 maps to the middle-ish closed range, then readiness
    # blends it halfway with the legal fallback.
    assert 1.0 < value < 4.0
    assert trace[-1]["head_name"] == "temperature"
    assert trace[-1]["event_type"] == "action_space_scalar"


def test_consolidation_scalar_head_is_available_for_fast_slow_judge():
    action_space = ConstrainedActionSpace(
        head_readiness={"consolidation": 1.0},
    )

    gate = action_space.scalar_value(
        head_name="consolidation",
        raw_value=0.0,
        gpu_step=7,
        fallback=0.0,
        reward_context={"steps_since_slow_sync": 3.0},
    )

    assert 0.0 <= gate <= 1.0
    assert gate == pytest.approx(0.5)


def test_shared_event_ssm_head_updates_from_recorded_reward():
    ssm = SharedEventSsm(hidden_dim=4, seed=5)
    action_space = ConstrainedActionSpace(
        head_readiness={"replay_timing": 1.0},
        head_max_delta={"replay_timing": 1.0},
        event_ssm=ssm,
        online_learning_rate=0.1,
    )
    action_space.effective_scores(
        heuristic_scores=[0.1, 0.2],
        learned_scores=None,
        gpu_step=1,
        head_name="replay_timing",
        reward_context={"score": 0.15},
    )
    before = list(ssm.head_weights["replay_timing"])
    action_space.record_credit_assignment(
        key=123,
        head_names=["replay_timing"],
        gpu_step=1,
    )

    applied = action_space.apply_reward(
        key=123,
        reward=1.0,
        gpu_step=2,
    )

    assert applied == 1
    assert ssm.head_weights["replay_timing"] != before
