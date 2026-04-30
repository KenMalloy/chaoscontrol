from __future__ import annotations

import os

import pytest

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def _shm_name(suffix: str) -> str:
    return f"/cc_{os.getpid()}_{suffix}"


def _force_unlink(cls, name: str) -> None:
    try:
        cls.unlink(name)
    except Exception:
        pass


def _slice(
    *,
    offset: int,
    nbytes: int,
    dtype: int,
    rank: int,
    shape: tuple[int, int, int, int],
) -> dict[str, object]:
    return {
        "offset_bytes": offset,
        "nbytes": nbytes,
        "dtype": dtype,
        "rank": rank,
        "shape": list(shape),
    }


def test_teacher_request_ring_round_trips_payload_descriptor():
    constants = _ext.wire_event_constants()
    name = _shm_name("treq")
    _force_unlink(_ext.ShmRingTeacherRequest, name)
    ring = _ext.ShmRingTeacherRequest.create(name)
    try:
        event = {
            "event_type": 6,
            "source_rank": 0,
            "status": 0,
            "flags": 3,
            "slice_count": constants["TEACHER_REQUEST_SLICES"],
            "request_id": 123,
            "step": 456,
            "weight_snapshot_version": 444,
            "full_ids": _slice(
                offset=4096,
                nbytes=2 * 6 * 4,
                dtype=constants["TEACHER_DTYPE_INT32"],
                rank=2,
                shape=(2, 6, 0, 0),
            ),
        }
        assert ring.push(event)
        assert ring.size() == 1
        assert ring.pop() == event
        assert ring.pop() is None
    finally:
        _ext.ShmRingTeacherRequest.unlink(name)


def test_teacher_result_ring_round_trips_all_packet_slices():
    constants = _ext.wire_event_constants()
    name = _shm_name("tres")
    _force_unlink(_ext.ShmRingTeacherResult, name)
    ring = _ext.ShmRingTeacherResult.create(name)
    try:
        slices = [
            _slice(
                offset=i * 1024,
                nbytes=128 + i,
                dtype=constants["TEACHER_DTYPE_BFLOAT16"],
                rank=2 if i < 4 else 3,
                shape=(2, 5, 8 if i >= 4 else 0, 0),
            )
            for i in range(constants["TEACHER_RESULT_SLICES"])
        ]
        event = {
            "event_type": 7,
            "source_rank": 3,
            "status": 0,
            "flags": 1,
            "slice_count": constants["TEACHER_RESULT_SLICES"],
            "request_id": 123,
            "step": 456,
            "weight_snapshot_version": 444,
            "payload_version": 12,
            "score_seconds": 0.125,
            "packet_seconds": 0.005,
            "target_token_count": 10,
            "hidden_dim": 8,
            "plasticity_dim": 8,
            "fast_slow_mode": 1,
            "fast_slow_accepted": 1,
            "fast_slow_step": 457,
            "fast_slow_alpha": 0.5,
            "fast_slow_gate": 0.75,
            "fast_slow_effective_alpha": 0.375,
            "fast_slow_reason": 1,
            "slices": slices,
        }
        assert ring.push(event)
        out = ring.pop()
        assert out is not None
        assert out["event_type"] == event["event_type"]
        assert out["source_rank"] == event["source_rank"]
        assert out["request_id"] == event["request_id"]
        assert out["weight_snapshot_version"] == event["weight_snapshot_version"]
        assert out["payload_version"] == event["payload_version"]
        assert out["target_token_count"] == event["target_token_count"]
        assert out["hidden_dim"] == event["hidden_dim"]
        assert out["plasticity_dim"] == event["plasticity_dim"]
        assert out["fast_slow_mode"] == event["fast_slow_mode"]
        assert out["fast_slow_accepted"] == event["fast_slow_accepted"]
        assert out["fast_slow_step"] == event["fast_slow_step"]
        assert out["fast_slow_reason"] == event["fast_slow_reason"]
        assert out["slices"] == event["slices"]
        assert out["score_seconds"] == pytest.approx(event["score_seconds"])
        assert out["packet_seconds"] == pytest.approx(event["packet_seconds"])
        assert out["fast_slow_alpha"] == pytest.approx(event["fast_slow_alpha"])
        assert out["fast_slow_gate"] == pytest.approx(event["fast_slow_gate"])
        assert out["fast_slow_effective_alpha"] == pytest.approx(
            event["fast_slow_effective_alpha"]
        )
        assert ring.pop() is None
    finally:
        _ext.ShmRingTeacherResult.unlink(name)


def test_teacher_result_rejects_wrong_slice_count_shape():
    constants = _ext.wire_event_constants()
    name = _shm_name("terr")
    _force_unlink(_ext.ShmRingTeacherResult, name)
    ring = _ext.ShmRingTeacherResult.create(name)
    try:
        bad = {
            "event_type": 7,
            "source_rank": 3,
            "status": 0,
            "flags": 0,
            "slice_count": constants["TEACHER_RESULT_SLICES"],
            "request_id": 1,
            "step": 2,
            "weight_snapshot_version": 3,
            "payload_version": 4,
            "score_seconds": 0.0,
            "packet_seconds": 0.0,
            "target_token_count": 0,
            "hidden_dim": 0,
            "plasticity_dim": 0,
            "fast_slow_mode": 0,
            "fast_slow_accepted": 0,
            "fast_slow_step": 0,
            "fast_slow_alpha": 0.0,
            "fast_slow_gate": 0.0,
            "fast_slow_effective_alpha": 0.0,
            "fast_slow_reason": 0,
            "slices": [],
        }
        with pytest.raises(ValueError, match="slices"):
            ring.push(bad)
    finally:
        _ext.ShmRingTeacherResult.unlink(name)
