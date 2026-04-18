import pytest

from chaoscontrol.eval_stream.budget import BudgetTracker, compute_usable_ttt_budget


def test_compute_usable_ttt_budget_subtracts_floor_and_margin():
    assert compute_usable_ttt_budget(
        total_budget_seconds=600.0,
        score_floor_seconds=410.0,
        safety_margin_seconds=30.0,
    ) == pytest.approx(160.0)


def test_compute_usable_ttt_budget_clamps_at_zero():
    assert compute_usable_ttt_budget(
        total_budget_seconds=600.0,
        score_floor_seconds=590.0,
        safety_margin_seconds=30.0,
    ) == pytest.approx(0.0)


def test_budget_tracker_stops_adaptation_when_slack_is_exhausted():
    tracker = BudgetTracker(
        total_budget_seconds=10.0,
        score_floor_seconds=7.0,
        safety_margin_seconds=1.0,
    )

    assert tracker.can_adapt()
    tracker.add_adapt_time(2.0)

    assert not tracker.can_adapt()
    assert tracker.slack_remaining_seconds == pytest.approx(0.0)


def test_score_only_summary_uses_elapsed_time_as_floor():
    tracker = BudgetTracker(
        total_budget_seconds=10.0,
        score_floor_seconds=0.0,
        safety_margin_seconds=1.0,
    )
    tracker.add_score_time(3.0)

    summary = tracker.summary(
        docs_scored=2,
        chunks_scored=5,
        tokens_scored=100,
        adapt_steps=0,
        timed_out=False,
        collapsed=False,
        score_only_mode=True,
        elapsed_seconds=3.5,
    )

    assert summary["score_floor_seconds"] == pytest.approx(3.5)
    assert summary["usable_ttt_budget_seconds"] == pytest.approx(5.5)
    assert summary["ttt_budget_used_seconds"] == pytest.approx(0.0)
    assert summary["score_only_mode"] is True
