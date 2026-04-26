"""Cross-process shm ring producer tests (Phase B4)."""
from __future__ import annotations

import importlib.util
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import pytest

from chaoscontrol.kernels import _cpu_ssm_controller as _ext

REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("runner_b4", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _force_unlink(cls: Any, name: str) -> None:
    try:
        cls.unlink(name)
    except Exception:
        pass


def _write_event(seq: int = 0) -> dict[str, Any]:
    return {
        "event_type": 1,
        "source_rank": 2,
        "write_bucket": 1,
        "candidate_id": (2 << 56) | int(seq),
        "gpu_step": 123,
        "key_fp": 987654321,
        "key_rep": [seq % 65536 for _ in range(256)],
        "value_tok_ids": [11, 12, 13, 14],
        "value_anchor_id": 11,
        "pressure_at_write": 0.75,
        "pre_write_ce": 2.5,
    }


def _query_event(seq: int = 0) -> dict[str, Any]:
    return {
        "event_type": 2,
        "source_rank": 1,
        "bucket": 3,
        "query_id": (1 << 56) | int(seq),
        "gpu_step": 124,
        "query_rep": [(seq + i) % 65536 for i in range(256)],
        "pressure": 0.875,
        "pre_query_ce": 1.25,
    }


def _replay_outcome(seq: int = 0) -> dict[str, Any]:
    return {
        "event_type": 3,
        "selected_rank": 0,
        "outcome_status": 0,
        "replay_id": int(seq),
        "gpu_step": 125,
        "query_event_id": 1000 + int(seq),
        "source_write_id": 2000 + int(seq),
        "slot_id": 7,
        "policy_version": 4,
        "selection_step": 120,
        "teacher_score": 0.5,
        "controller_logit": -0.25,
        "ce_before_replay": 4.0,
        "ce_after_replay": 3.5,
        "ce_delta_raw": 0.5,
        "bucket_baseline": 0.125,
        "reward_shaped": 0.375,
        "grad_cos_rare": float("nan"),
        "grad_cos_total": float("nan"),
        "flags": 0,
    }


def _attach_and_push(kind: str, name: str, event: dict[str, Any], q: Any) -> None:
    try:
        cls = {
            "write": _ext.ShmRingWriteEvent,
            "query": _ext.ShmRingQueryEvent,
            "replay": _ext.ShmRingReplayOutcome,
        }[kind]
        ring = cls.attach(name)
        q.put(("ok", bool(ring.push(event))))
    except Exception as exc:  # pragma: no cover - child-process diagnostics
        q.put(("error", repr(exc)))


def _round_trip_cross_process(
    *,
    cls: Any,
    kind: str,
    name: str,
    event: dict[str, Any],
) -> dict[str, Any]:
    _force_unlink(cls, name)
    ring = cls.create(name)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_attach_and_push, args=(kind, name, event, q))
    try:
        proc.start()
        proc.join(timeout=10)
        assert proc.exitcode == 0
        status, payload = q.get(timeout=2)
        assert status == "ok", payload
        assert payload is True
        popped = ring.pop()
        assert popped is not None
        assert ring.pop() is None
        return popped
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        _force_unlink(cls, name)


def test_write_event_ring_cross_process_round_trip():
    event = _write_event(42)
    popped = _round_trip_cross_process(
        cls=_ext.ShmRingWriteEvent,
        kind="write",
        name=f"/cc_test_b4_write_pid{os.getpid()}",
        event=event,
    )
    assert popped == event


def test_query_event_ring_cross_process_round_trip():
    event = _query_event(43)
    popped = _round_trip_cross_process(
        cls=_ext.ShmRingQueryEvent,
        kind="query",
        name=f"/cc_test_b4_query_pid{os.getpid()}",
        event=event,
    )
    assert popped == event


def test_replay_outcome_ring_cross_process_round_trip():
    event = _replay_outcome(44)
    popped = _round_trip_cross_process(
        cls=_ext.ShmRingReplayOutcome,
        kind="replay",
        name=f"/cc_test_b4_replay_pid{os.getpid()}",
        event=event,
    )
    assert popped["replay_id"] == event["replay_id"]
    assert popped["reward_shaped"] == pytest.approx(event["reward_shaped"])
    assert math.isnan(popped["grad_cos_rare"])
    assert math.isnan(popped["grad_cos_total"])


def test_ring_full_drops_increment_counter_does_not_crash():
    """A saturated producer increments drops and keeps running."""
    mod = _load_runner()
    name = f"/cc_test_b4_full_pid{os.getpid()}"
    _force_unlink(_ext.ShmRingWriteEvent, name)
    ring = _ext.ShmRingWriteEvent.create(name)

    class Owner:
        write_ring = ring
        write_ring_drops = 0

    try:
        capacity = int(_ext.ShmRingWriteEvent.capacity)
        for i in range(capacity + 7):
            mod._push_event_ring(
                Owner,
                ring_attr="write_ring",
                drops_attr="write_ring_drops",
                event=_write_event(i),
            )
        assert Owner.write_ring_drops == 7
        assert ring.size() == capacity
    finally:
        _force_unlink(_ext.ShmRingWriteEvent, name)
