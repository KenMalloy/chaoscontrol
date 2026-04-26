"""Assembled replay-outcome credit attribution for CPU SSM controller C6."""
from __future__ import annotations

import math

import pytest

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


REPLAY_SELECTION = 1


def _entry(
    step: int,
    *,
    action_type: int,
    output_logit: float,
    selected_rank: int = 0,
) -> _ext.ActionHistoryEntry:
    entry = _ext.ActionHistoryEntry()
    entry.action_type = action_type
    entry.gpu_step = step
    entry.policy_version = 7
    entry.output_logit = output_logit
    entry.selected_rank = selected_rank
    entry.neighbor_slot = 0
    entry.global_state = [0.0]
    entry.slot_state = [1.0]
    return entry


def _history_with_five_actions() -> _ext.PerSlotActionHistory:
    history = _ext.PerSlotActionHistory(num_slots=1, max_entries_per_slot=8)
    history.append(
        0,
        _entry(
            900,
            action_type=REPLAY_SELECTION,
            output_logit=1.20,
            selected_rank=3,
        ),
    )
    history.append(
        0,
        _entry(940, action_type=2, output_logit=0.80, selected_rank=5),
    )
    history.append(
        0,
        _entry(
            950,
            action_type=REPLAY_SELECTION,
            output_logit=1.50,
            selected_rank=0,
        ),
    )
    history.append(
        0,
        _entry(980, action_type=3, output_logit=-1.25, selected_rank=0),
    )
    history.append(
        0,
        _entry(990, action_type=2, output_logit=0.60, selected_rank=2),
    )
    return history


def test_attribute_credit_returns_newest_first_assembled_credit():
    history = _history_with_five_actions()
    # C6 uses the replay outcome controller logit as a temporary current-policy
    # sentinel. C10 will replace this with recomputed current-policy logits.
    outcome = {
        "gpu_step": 1000,
        "reward_shaped": 0.5,
        "controller_logit": 0.75,
    }
    sigma_by_action_type = [0.0, 0.20, 0.30, 0.40]
    gamma = 0.995
    gerber_c = 0.5

    credited = _ext.attribute_credit(
        0, outcome, history, sigma_by_action_type, gamma=gamma, gerber_c=gerber_c
    )

    assert [item.entry.gpu_step for item in credited] == [990, 980, 950, 940, 900]

    expected_by_step = {
        990: 0.5 * math.pow(gamma, 10) * 1.0,
        980: 0.0,
        950: 0.5 * math.pow(gamma, 50) * 1.0,
        940: 0.5 * math.pow(gamma, 60) * 1.0,
        900: 0.5 * math.pow(gamma, 100) * (1.0 / (3 + 1)),
    }
    for item in credited:
        assert item.credit == pytest.approx(expected_by_step[item.entry.gpu_step])


def test_attribute_credit_preserves_negative_reward_sign():
    history = _history_with_five_actions()
    outcome = {
        "gpu_step": 1000,
        "reward_shaped": -0.5,
        "controller_logit": 0.75,
    }

    credited = _ext.attribute_credit(0, outcome, history, [0.0, 0.20, 0.30, 0.40])

    by_step = {item.entry.gpu_step: item.credit for item in credited}
    assert by_step[990] < 0.0
    assert by_step[900] < 0.0
    assert by_step[980] == 0.0


def test_attribute_credit_rejects_missing_slot_and_outcome_key():
    history = _history_with_five_actions()
    outcome = {
        "gpu_step": 1000,
        "reward_shaped": 0.5,
        "controller_logit": 0.75,
    }

    with pytest.raises(IndexError, match="slot_id"):
        _ext.attribute_credit(1, outcome, history, [0.0, 0.20, 0.30, 0.40])

    missing_controller_logit = {"gpu_step": 1000, "reward_shaped": 0.5}
    with pytest.raises(KeyError, match="controller_logit"):
        _ext.attribute_credit(
            0, missing_controller_logit, history, [0.0, 0.20, 0.30, 0.40]
        )
