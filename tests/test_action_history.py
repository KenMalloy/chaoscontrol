"""Per-slot action history substrate for CPU SSM controller C3."""
from __future__ import annotations

import pytest

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def _entry(step: int, *, action_type: int = 1) -> _ext.ActionHistoryEntry:
    entry = _ext.ActionHistoryEntry()
    entry.action_type = action_type
    entry.gpu_step = step
    entry.policy_version = step % 7
    entry.output_logit = step + 0.25
    entry.selected_rank = step % 4
    entry.neighbor_slot = 1000 + step
    entry.global_state = [float(step), float(step + 1)]
    entry.slot_state = [float(-step)]
    return entry


def test_append_returns_newest_first_history_and_keeps_slots_isolated():
    history = _ext.PerSlotActionHistory(num_slots=2, max_entries_per_slot=128)

    for step in range(100):
        history.append(0, _entry(step))

    slot0_steps = [entry.gpu_step for entry in history.history(0)]
    assert slot0_steps == list(reversed(range(100)))
    assert history.size(0) == 100

    history.append(1, _entry(900))

    assert [entry.gpu_step for entry in history.history(1)] == [900]
    assert [entry.gpu_step for entry in history.history(0)] == slot0_steps


def test_overflow_retains_newest_max_entries():
    history = _ext.PerSlotActionHistory(num_slots=1, max_entries_per_slot=5)

    for step in range(10):
        history.append(0, _entry(step))

    assert [entry.gpu_step for entry in history.history(0)] == [9, 8, 7, 6, 5]
    assert history.size(0) == 5


def test_action_history_entry_round_trips_all_python_fields():
    history = _ext.PerSlotActionHistory(num_slots=1, max_entries_per_slot=4)
    entry = _ext.ActionHistoryEntry()
    entry.action_type = 3
    entry.gpu_step = 123456789
    entry.policy_version = 42
    entry.output_logit = -1.5
    entry.selected_rank = 2
    entry.neighbor_slot = 77
    entry.global_state = [0.125, 0.5, 1.25]
    entry.slot_state = [-3.0, 4.0]

    history.append(0, entry)
    [stored] = history.history(0)

    assert stored.action_type == 3
    assert stored.gpu_step == 123456789
    assert stored.policy_version == 42
    assert stored.output_logit == pytest.approx(-1.5)
    assert stored.selected_rank == 2
    assert stored.neighbor_slot == 77
    assert stored.global_state == pytest.approx([0.125, 0.5, 1.25])
    assert stored.slot_state == pytest.approx([-3.0, 4.0])


def test_eviction_retains_history_until_gc_lookahead_then_drops():
    history = _ext.PerSlotActionHistory(num_slots=2, max_entries_per_slot=8)
    history.append(0, _entry(1))
    history.append(1, _entry(20))

    history.mark_evicted(0, current_event_id=10)
    assert history.is_evicted(0) is True
    assert [entry.gpu_step for entry in history.history(0)] == [1]

    history.gc(current_event_id=14, gc_lookahead=5)
    assert history.is_evicted(0) is True
    assert [entry.gpu_step for entry in history.history(0)] == [1]

    history.gc(current_event_id=15, gc_lookahead=5)
    assert history.is_evicted(0) is False
    assert history.history(0) == []
    assert [entry.gpu_step for entry in history.history(1)] == [20]


def test_append_to_evicted_slot_revives_without_waiting_for_gc():
    history = _ext.PerSlotActionHistory(num_slots=1, max_entries_per_slot=4)
    history.append(0, _entry(1))
    history.mark_evicted(0, current_event_id=100)

    history.append(0, _entry(2))
    history.gc(current_event_id=1_000, gc_lookahead=1)

    assert history.is_evicted(0) is False
    assert [entry.gpu_step for entry in history.history(0)] == [2, 1]


def test_invalid_ctor_and_slot_raise():
    with pytest.raises(ValueError, match="num_slots"):
        _ext.PerSlotActionHistory(num_slots=0, max_entries_per_slot=8)
    with pytest.raises(ValueError, match="max_entries_per_slot"):
        _ext.PerSlotActionHistory(num_slots=1, max_entries_per_slot=0)

    history = _ext.PerSlotActionHistory(num_slots=1, max_entries_per_slot=8)
    with pytest.raises(IndexError, match="slot_id"):
        history.append(1, _entry(1))
    with pytest.raises(IndexError, match="slot_id"):
        history.history(1)
    with pytest.raises(IndexError, match="slot_id"):
        history.mark_evicted(1, current_event_id=0)
    with pytest.raises(IndexError, match="slot_id"):
        history.is_evicted(1)
