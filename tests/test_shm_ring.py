"""ShmRing<T, Capacity> composing SpscRing (A2) + PosixShm (A3) into a
cross-process shared-memory ring (Phase A4 of CPU SSM controller plan).

Tests:
  - test_shm_ring_single_process_round_trip: creator pushes, creator
    pops (in-process). Verifies the placement-new'd ring is functional
    before involving a second process.
  - test_shm_ring_cross_process_round_trip: creator process pushes
    1024 events; consumer process attaches by name, pops 1024,
    verifies all match. The real Phase B/C use case — producer in
    train-rank or episodic-rank process, consumer in the controller
    process.
  - test_shm_ring_attach_size_mismatch_raises: attach with a name that
    points to a region smaller than the required REGION_BYTES raises
    rather than silently treating the smaller region as a valid ring
    (catches name collisions and stale regions from a different
    ShmRing<T, Capacity> instantiation).

Spec: docs/plans/2026-04-26-cpu-ssm-controller.md (Task A4).
"""
from __future__ import annotations

import os
import sys
import time

import pytest

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


SHM_NAME = "/cc_test_shm_ring"  # leading slash per POSIX; <30 chars for macOS


def _force_unlink() -> None:
    """Idempotent unlink — a previous failed run can leave the kernel
    region persistent; safe to call when the region does not exist."""
    try:
        _ext.ShmRingU64x1024.unlink(SHM_NAME)
    except Exception:
        pass


def test_shm_ring_single_process_round_trip():
    """Creator pushes 1024, pops 1024 in-process — verifies the
    placement-new'd ring is functional before involving a second
    process. Pins the basic create→push→pop→unlink lifecycle."""
    _force_unlink()
    ring = _ext.ShmRingU64x1024.create(SHM_NAME)
    try:
        assert ring.name() == SHM_NAME, (
            f"expected name {SHM_NAME!r}, got {ring.name()!r}"
        )
        for i in range(1024):
            assert ring.push(i), f"push {i} failed (size={ring.size()})"
        assert ring.size() == 1024, f"expected size 1024, got {ring.size()}"
        for i in range(1024):
            popped = ring.pop()
            assert popped == i, f"expected {i}, got {popped!r}"
        assert ring.pop() is None, "pop on empty should return None"
    finally:
        _ext.ShmRingU64x1024.unlink(SHM_NAME)


def test_shm_ring_cross_process_round_trip():
    """Creator process pushes 1024 events; consumer process attaches
    by name, pops 1024, verifies all arrive in order. The real Phase
    B/C use case — producer in train-rank or episodic-rank process,
    consumer in the controller process. Pins that placement-new on the
    creator side correctly initializes the SPSC state for the
    attacher's mapping."""
    _force_unlink()
    pid = os.fork()
    if pid == 0:  # consumer
        # Wait for producer to create + start populating.
        time.sleep(0.5)
        try:
            ring = _ext.ShmRingU64x1024.attach(SHM_NAME)
            received: list[int] = []
            deadline = time.monotonic() + 5.0
            while len(received) < 1024 and time.monotonic() < deadline:
                v = ring.pop()
                if v is not None:
                    received.append(v)
            ok = received == list(range(1024))
            os._exit(0 if ok else 1)
        except Exception:
            os._exit(2)
    else:  # producer
        try:
            ring = _ext.ShmRingU64x1024.create(SHM_NAME)
            for i in range(1024):
                while not ring.push(i):
                    time.sleep(0.001)  # ring full — wait for consumer
            _, status = os.waitpid(pid, 0)
            assert os.WIFEXITED(status), (
                f"child did not exit cleanly: status={status}"
            )
            assert os.WEXITSTATUS(status) == 0, (
                f"consumer exit status {os.WEXITSTATUS(status)} "
                "(1=value mismatch / timeout, 2=exception in child)"
            )
        finally:
            _ext.ShmRingU64x1024.unlink(SHM_NAME)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS page-rounds POSIX shm to 16KB pages; "
           "ShmRingU64x1024.REGION_BYTES (8320) is smaller than one "
           "macOS page, so no sub-region can be created to trigger "
           "the size-mismatch path. The check is correct (see "
           "shm_ring.h note on page rounding) — production collisions "
           "between WriteEvent / QueryEvent / ReplayOutcome rings are "
           "many pages apart in size and trip the check on any platform. "
           "Linux CI exercises the path natively (4KB pages).",
)
def test_shm_ring_attach_size_mismatch_raises():
    """Attach to a name that points to an undersized region — raise
    rather than silently treating the smaller region as a valid ring.
    Catches name collisions where two different ShmRing<T, Capacity>
    instantiations end up using the same shm name."""
    _force_unlink()
    # Create a tiny region with the same name via PosixShm directly.
    # 4096 bytes is below REGION_BYTES (8320 for u64x1024) on Linux
    # where pages are 4KB, so fstat reports the original 4096 and the
    # check fires.
    tiny_size = 4096
    assert tiny_size < _ext.ShmRingU64x1024.REGION_BYTES, (
        f"test premise broken: tiny_size {tiny_size} >= REGION_BYTES "
        f"{_ext.ShmRingU64x1024.REGION_BYTES}"
    )
    tiny = _ext.PosixShm(SHM_NAME, tiny_size, True)
    try:
        with pytest.raises(Exception):
            _ext.ShmRingU64x1024.attach(SHM_NAME)
    finally:
        del tiny
        _ext.PosixShm.unlink(SHM_NAME)
