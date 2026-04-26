"""Online learning loop skeleton for CPU SSM controller C10.

The current wire events do not carry saved controller hidden states, so C10
must be honest: replay outcomes can append replay-selection history and
compute credit for prior selections, but the backward pass is skipped until a
future event schema supplies checkpoints.
"""
from __future__ import annotations

import math

import pytest

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def _replay_outcome(
    replay_id: int,
    *,
    gpu_step: int,
    slot_id: int = 1,
    selection_step: int | None = None,
    selected_rank: int = 0,
    controller_logit: float = 0.75,
    reward_shaped: float = 0.5,
) -> dict:
    return {
        "event_type": 3,
        "selected_rank": selected_rank,
        "outcome_status": 0,
        "replay_id": replay_id,
        "gpu_step": gpu_step,
        "query_event_id": 123,
        "source_write_id": 456,
        "slot_id": slot_id,
        "policy_version": 7,
        "selection_step": gpu_step if selection_step is None else selection_step,
        "teacher_score": 0.5,
        "controller_logit": controller_logit,
        "ce_before_replay": 4.0,
        "ce_after_replay": 3.5,
        "ce_delta_raw": 0.5,
        "bucket_baseline": 0.0,
        "reward_shaped": reward_shaped,
        "grad_cos_rare": math.nan,
        "grad_cos_total": math.nan,
        "flags": 0,
    }


def test_online_learning_records_history_and_credits_prior_replay_selection():
    controller = _ext.OnlineLearningController(
        num_slots=4,
        max_entries_per_slot=8,
        gamma=0.995,
        gerber_c=0.5,
    )

    first = _replay_outcome(1, gpu_step=100, selection_step=90)
    second = _replay_outcome(2, gpu_step=120, selection_step=110)

    controller.on_replay_outcome(first)
    telemetry_after_first = controller.telemetry()
    assert telemetry_after_first["replay_outcomes"] == 1
    assert telemetry_after_first["history_appends"] == 1
    assert telemetry_after_first["credited_actions"] == 0

    controller.on_replay_outcome(second)

    history = controller.history(1)
    assert [entry.gpu_step for entry in history] == [110, 90]
    assert [entry.action_type for entry in history] == [1, 1]
    assert [entry.output_logit for entry in history] == pytest.approx([0.75, 0.75])

    telemetry = controller.telemetry()
    assert telemetry["replay_outcomes"] == 2
    assert telemetry["history_appends"] == 2
    assert telemetry["credited_actions"] == 1
    assert telemetry["nonzero_credit_actions"] == 1
    assert telemetry["backward_skipped_missing_state"] == 1
    assert telemetry["sgd_steps"] == 0

    expected_credit = 0.5 * (0.995 ** (120 - 90))
    assert controller.last_credit_sum == pytest.approx(expected_credit)


def test_online_learning_preserves_negative_credit_sign_and_rank_factor():
    controller = _ext.OnlineLearningController(num_slots=4, max_entries_per_slot=8)

    controller.on_replay_outcome(
        _replay_outcome(
            1,
            gpu_step=100,
            selection_step=90,
            selected_rank=3,
            reward_shaped=0.5,
        )
    )
    controller.on_replay_outcome(
        _replay_outcome(2, gpu_step=120, selection_step=110, reward_shaped=-0.5)
    )

    expected_credit = -0.5 * (0.995 ** (120 - 90)) * (1.0 / 4.0)
    assert controller.last_credit_sum == pytest.approx(expected_credit)
    assert controller.telemetry()["nonzero_credit_actions"] == 1


def test_online_learning_skips_out_of_range_slot_without_throwing():
    controller = _ext.OnlineLearningController(num_slots=2, max_entries_per_slot=4)

    controller.on_replay_outcome(_replay_outcome(1, gpu_step=100, slot_id=99))

    telemetry = controller.telemetry()
    assert telemetry["replay_outcomes"] == 1
    assert telemetry["invalid_slot_skips"] == 1
    assert telemetry["history_appends"] == 0
    assert controller.history(0) == []
