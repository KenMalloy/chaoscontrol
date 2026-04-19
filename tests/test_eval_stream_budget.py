import time

import pytest

from chaoscontrol.eval_stream.budget import (
    BudgetTracker,
    EvalDeadline,
    compute_usable_ttt_budget,
)


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


def test_summary_provenance_defaults_to_none():
    """Backwards-compat: callers that don't pass provenance still get a
    ``provenance`` sub-dict with all fields set to None. Keeps summary JSON
    schema stable so downstream readers (paper tables, plots) don't branch
    on key-presence."""
    tracker = BudgetTracker(total_budget_seconds=10.0)
    summary = tracker.summary(
        docs_scored=1,
        chunks_scored=1,
        tokens_scored=10,
        adapt_steps=0,
        timed_out=False,
        collapsed=False,
        score_only_mode=True,
        elapsed_seconds=1.0,
    )
    assert "provenance" in summary
    prov = summary["provenance"]
    for key in (
        "ckpt_sha256", "ckpt_cfg_hash", "stream_seed", "gpu_name",
        "torch_version", "cuda_version", "chunk_size", "max_docs",
    ):
        assert key in prov, f"missing provenance field: {key}"
        assert prov[key] is None, f"expected None default for {key}, got {prov[key]!r}"


def test_eval_deadline_fresh_not_expired():
    """A just-constructed deadline with a non-trivial budget is live —
    elapsed is tiny, remaining is close to the full budget, is_expired
    is False."""
    d = EvalDeadline(budget_seconds=5.0)
    assert d.elapsed() < 0.1
    assert d.remaining() > 4.5
    assert not d.is_expired()


def test_eval_deadline_expires_after_sleep():
    """Deadline expires once elapsed passes the budget. Uses a short budget
    + a short sleep so the test is deterministic without being slow."""
    d = EvalDeadline(budget_seconds=0.05)
    time.sleep(0.1)
    assert d.is_expired()
    assert d.remaining() == pytest.approx(0.0)


def test_eval_deadline_zero_budget_is_expired():
    """A zero-budget deadline is expired immediately. Useful as a
    sentinel — pass ``EvalDeadline(0.0)`` to force-break a loop."""
    d = EvalDeadline(budget_seconds=0.0)
    # elapsed will be a microsecond or two, but strictly > 0
    assert d.is_expired()
    assert d.remaining() == pytest.approx(0.0)


def test_summary_provenance_fields_passed_through():
    """When callers provide provenance, values land under ``provenance`` as-is.
    This is the common case: run_exp20_eval.py computes sha256 / cfg_hash /
    GPU name / library versions at end-of-eval and threads them here. No
    coercion, no interpretation — just pin-what-produced-the-measurement."""
    tracker = BudgetTracker(total_budget_seconds=10.0)
    summary = tracker.summary(
        docs_scored=1,
        chunks_scored=1,
        tokens_scored=10,
        adapt_steps=0,
        timed_out=False,
        collapsed=False,
        score_only_mode=True,
        elapsed_seconds=1.0,
        ckpt_sha256="deadbeef" * 8,
        ckpt_cfg_hash="feedface" * 8,
        stream_seed=1337,
        gpu_name="NVIDIA H100 80GB HBM3",
        torch_version="2.5.0+cu121",
        cuda_version="12.1",
        chunk_size=256,
        max_docs=50_000,
    )
    prov = summary["provenance"]
    assert prov["ckpt_sha256"] == "deadbeef" * 8
    assert prov["ckpt_cfg_hash"] == "feedface" * 8
    assert prov["stream_seed"] == 1337
    assert prov["gpu_name"] == "NVIDIA H100 80GB HBM3"
    assert prov["torch_version"] == "2.5.0+cu121"
    assert prov["cuda_version"] == "12.1"
    assert prov["chunk_size"] == 256
    assert prov["max_docs"] == 50_000
