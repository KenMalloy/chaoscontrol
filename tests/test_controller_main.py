"""Controller main-loop polling over real shm rings (Phase C1)."""
from __future__ import annotations

import math
import os
import time

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def _force_unlink(cls, name: str) -> None:
    try:
        cls.unlink(name)
    except Exception:
        pass


def _sample_write_event(candidate_id: int) -> dict:
    return {
        "event_type": 1,
        "source_rank": 0,
        "write_bucket": 2,
        "candidate_id": candidate_id,
        "gpu_step": 100 + candidate_id,
        "key_fp": 0xABC000 + candidate_id,
        "key_rep": [i % 256 for i in range(256)],
        "value_tok_ids": [11, 22, 33, 44],
        "value_anchor_id": 7,
        "pressure_at_write": 1.25,
        "pre_write_ce": 2.34,
    }


def _sample_query_event(query_id: int) -> dict:
    return {
        "event_type": 2,
        "source_rank": 1,
        "bucket": 4,
        "query_id": query_id,
        "gpu_step": 200 + query_id,
        "query_rep": [(i * 3) % 65536 for i in range(256)],
        "pressure": 0.875,
        "pre_query_ce": 3.5,
    }


def _sample_replay_outcome(replay_id: int) -> dict:
    return {
        "event_type": 3,
        "selected_rank": 2,
        "outcome_status": 0,
        "replay_id": replay_id,
        "gpu_step": 300 + replay_id,
        "query_event_id": 0x5555_6666_7777_8888,
        "source_write_id": 0x9999_AAAA_BBBB_CCCC,
        "slot_id": 12345,
        "policy_version": 7,
        "selection_step": 999,
        "teacher_score": 0.5,
        "controller_logit": -1.25,
        "ce_before_replay": 4.1,
        "ce_after_replay": 3.9,
        "ce_delta_raw": -0.2,
        "bucket_baseline": 4.0,
        "reward_shaped": 0.05,
        "grad_cos_rare": math.nan,
        "grad_cos_total": math.nan,
        "flags": 0xABCD,
    }


def test_controller_main_polls_all_event_rings_until_exit_flag():
    """Forked controller attaches to all C1 rings, drains mixed events,
    and returns the total number of stub-handled records."""
    pid_suffix = os.getpid()
    write_names = [f"/cc_c1_w0_{pid_suffix}", f"/cc_c1_w1_{pid_suffix}"]
    query_name = f"/cc_c1_q_{pid_suffix}"
    replay_name = f"/cc_c1_r_{pid_suffix}"
    exit_name = f"/cc_c1_x_{pid_suffix}"

    for name in write_names:
        _force_unlink(_ext.ShmRingWriteEvent, name)
    _force_unlink(_ext.ShmRingQueryEvent, query_name)
    _force_unlink(_ext.ShmRingReplayOutcome, replay_name)
    _force_unlink(_ext.PosixShm, exit_name)

    write_rings = [_ext.ShmRingWriteEvent.create(name) for name in write_names]
    query_ring = _ext.ShmRingQueryEvent.create(query_name)
    replay_ring = _ext.ShmRingReplayOutcome.create(replay_name)
    exit_flag = _ext.PosixShm(exit_name, 1, True)
    exit_flag.write_bytes(0, b"\x00")

    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            total = _ext.controller_main(
                write_names,
                query_name,
                replay_name,
                exit_name,
                1_000,
            )
            os.write(write_fd, str(total).encode("ascii"))
            os._exit(0)
        except Exception as exc:
            os.write(write_fd, f"ERR:{type(exc).__name__}:{exc}".encode("utf-8"))
            os._exit(2)
    os.close(write_fd)

    try:
        for i in range(10):
            assert write_rings[i % len(write_rings)].push(_sample_write_event(i))
        for i in range(2):
            assert query_ring.push(_sample_query_event(i))
        for i in range(3):
            assert replay_ring.push(_sample_replay_outcome(i))

        deadline = time.monotonic() + 5.0
        early_status: int | None = None
        while time.monotonic() < deadline:
            exited_pid, status = os.waitpid(child_pid, os.WNOHANG)
            if exited_pid == child_pid:
                early_status = status
                break
            if (
                all(ring.size() == 0 for ring in write_rings)
                and query_ring.size() == 0
                and replay_ring.size() == 0
            ):
                break
            time.sleep(0.001)
        if early_status is not None:
            payload = os.read(read_fd, 256).decode("utf-8")
            assert os.WIFEXITED(early_status), (
                f"controller did not exit cleanly: {early_status}"
            )
            assert os.WEXITSTATUS(early_status) == 0, payload
        assert all(ring.size() == 0 for ring in write_rings)
        assert query_ring.size() == 0
        assert replay_ring.size() == 0

        exit_flag.write_bytes(0, b"\x01")
        _, status = os.waitpid(child_pid, 0)
        payload = os.read(read_fd, 256).decode("utf-8")
        assert os.WIFEXITED(status), f"controller did not exit cleanly: {status}"
        assert os.WEXITSTATUS(status) == 0, payload
        assert payload == "15"
    finally:
        try:
            exit_flag.write_bytes(0, b"\x01")
            _, _ = os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            pass
        os.close(read_fd)
        for name in write_names:
            _force_unlink(_ext.ShmRingWriteEvent, name)
        _force_unlink(_ext.ShmRingQueryEvent, query_name)
        _force_unlink(_ext.ShmRingReplayOutcome, replay_name)
        _force_unlink(_ext.PosixShm, exit_name)
