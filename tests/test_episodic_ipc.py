"""Tests for the SPSC shared-memory ring buffer (episodic IPC, Task 1.1).

The ring is single-producer-single-consumer with fixed-shape slots over POSIX
shared memory. Producer writes are non-blocking — on full, drops the OLDEST
slot (overwrites head), bumps a `dropped_count` counter. Consumer reads in
FIFO order. Multi-writer is a non-feature; multi-producer scenarios are served
by N rings, not multi-writer support.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import uuid

import numpy as np
import pytest

from chaoscontrol.episodic.ipc import ShmRing


def _unique_name(tag: str) -> str:
    """POSIX shm name short enough for darwin's PSHMNAMLEN ~30-char cap."""
    return f"cc_{tag}_{os.getpid() & 0xFFFF:04x}_{uuid.uuid4().hex[:6]}"


@pytest.fixture
def ring_name():
    """Yield a unique shm name; cleanup happens via test's close_and_unlink."""
    return _unique_name("t")


def test_ring_writes_and_reads_in_order(ring_name):
    ring = ShmRing.create(
        name=ring_name,
        slot_shape=(4,),
        dtype=np.dtype(np.float32),
        capacity=8,
    )
    try:
        # Empty ring returns None.
        assert ring.try_read() is None

        items = [
            np.array([float(i), float(i) + 0.1, float(i) + 0.2, float(i) + 0.3],
                     dtype=np.float32)
            for i in range(5)
        ]
        for it in items:
            ring.try_write(it)

        # FIFO order.
        for expected in items:
            got = ring.try_read()
            assert got is not None
            np.testing.assert_array_equal(got, expected)

        assert ring.try_read() is None
        assert ring.dropped_count() == 0
    finally:
        ring.close_and_unlink()


def test_ring_dtype_and_shape_validation(ring_name):
    ring = ShmRing.create(
        name=ring_name,
        slot_shape=(3,),
        dtype=np.dtype(np.float32),
        capacity=4,
    )
    try:
        # Wrong shape.
        with pytest.raises(ValueError, match="shape"):
            ring.try_write(np.zeros((4,), dtype=np.float32))
        with pytest.raises(ValueError, match="shape"):
            ring.try_write(np.zeros((3, 1), dtype=np.float32))

        # Wrong dtype.
        with pytest.raises(ValueError, match="dtype"):
            ring.try_write(np.zeros((3,), dtype=np.int64))

        # Non-ndarray input.
        with pytest.raises(ValueError):
            ring.try_write([0.0, 0.0, 0.0])  # type: ignore[arg-type]

        # Correct shape + dtype works.
        ring.try_write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    finally:
        ring.close_and_unlink()


def test_ring_drops_on_overflow_and_bumps_counter(ring_name):
    capacity = 4
    overflow = 5  # write capacity + overflow items total
    total = capacity + overflow

    ring = ShmRing.create(
        name=ring_name,
        slot_shape=(2,),
        dtype=np.dtype(np.int64),
        capacity=capacity,
    )
    try:
        items = [
            np.array([i, i * 10], dtype=np.int64) for i in range(total)
        ]
        for it in items:
            ring.try_write(it)

        # Dropped count = excess writes.
        assert ring.dropped_count() == overflow

        # The most recent `capacity` items remain in FIFO order.
        kept = items[-capacity:]
        for expected in kept:
            got = ring.try_read()
            assert got is not None
            np.testing.assert_array_equal(got, expected)
        assert ring.try_read() is None

        # Wraparound integrity: after partial drain, fill again past capacity.
        # Read all → ring empty → write 7 more → 3 should drop, 4 remain.
        for i in range(total, total + 7):
            ring.try_write(np.array([i, i * 10], dtype=np.int64))
        assert ring.dropped_count() == overflow + (7 - capacity)

        last_four = [
            np.array([i, i * 10], dtype=np.int64)
            for i in range(total + 7 - capacity, total + 7)
        ]
        for expected in last_four:
            got = ring.try_read()
            assert got is not None
            np.testing.assert_array_equal(got, expected)
        assert ring.try_read() is None
    finally:
        ring.close_and_unlink()


def _producer_proc(name: str, n_items: int, ready_q, done_q):
    """Producer subprocess: create the ring, write n items, signal done."""
    import numpy as np  # re-imported in spawned interpreter

    from chaoscontrol.episodic.ipc import ShmRing

    ring = ShmRing.create(
        name=name,
        slot_shape=(3,),
        dtype=np.dtype(np.float64),
        capacity=16,
    )
    try:
        ready_q.put("ready")  # tell parent the shm exists
        for i in range(n_items):
            ring.try_write(
                np.array([float(i), float(i) * 2.0, float(i) * 3.0],
                         dtype=np.float64),
            )
        done_q.put("done")
        # Block until parent confirms it has read everything; only then unlink.
        ready_q.get()  # wait for parent's "drained" sentinel
    finally:
        ring.close_and_unlink()


def test_ring_round_trips_across_processes(ring_name):
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue()
    done_q = ctx.Queue()
    n_items = 6

    proc = ctx.Process(
        target=_producer_proc,
        args=(ring_name, n_items, ready_q, done_q),
    )
    proc.start()
    try:
        # Wait until producer has created the shm.
        assert ready_q.get(timeout=10.0) == "ready"

        # Now safe to attach.
        reader = ShmRing.attach(
            name=ring_name,
            slot_shape=(3,),
            dtype=np.dtype(np.float64),
            capacity=16,
        )
        try:
            # Wait until producer signals done so reads aren't racing.
            assert done_q.get(timeout=10.0) == "done"

            for i in range(n_items):
                got = reader.try_read()
                assert got is not None, f"missing item {i}"
                np.testing.assert_array_equal(
                    got,
                    np.array([float(i), float(i) * 2.0, float(i) * 3.0],
                             dtype=np.float64),
                )
            assert reader.try_read() is None
            assert reader.dropped_count() == 0
        finally:
            reader.close()  # consumer side: handles only, no unlink
            ready_q.put("drained")  # release the producer to unlink
            proc.join(timeout=10.0)
            assert proc.exitcode == 0, f"producer exited {proc.exitcode}"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)


def test_ring_capacity_one_edge_case(ring_name):
    """Tightest-constraint edge: cap=1 must still preserve drop-oldest FIFO."""
    ring = ShmRing.create(
        name=ring_name,
        slot_shape=(1,),
        dtype=np.dtype(np.int64),
        capacity=1,
    )
    try:
        ring.try_write(np.array([1], dtype=np.int64))
        got = ring.try_read()
        assert got is not None and got[0] == 1
        assert ring.try_read() is None

        # Refill, then overflow with B,C,D — D is the only retained item.
        ring.try_write(np.array([10], dtype=np.int64))
        ring.try_write(np.array([20], dtype=np.int64))  # drops 10
        ring.try_write(np.array([30], dtype=np.int64))  # drops 20
        assert ring.dropped_count() == 2
        got = ring.try_read()
        assert got is not None and got[0] == 30
        assert ring.try_read() is None
    finally:
        ring.close_and_unlink()


def test_ring_close_and_unlink_is_idempotent(ring_name):
    ring = ShmRing.create(
        name=ring_name,
        slot_shape=(2,),
        dtype=np.dtype(np.float32),
        capacity=4,
    )
    ring.try_write(np.array([1.0, 2.0], dtype=np.float32))
    # Repeated close_and_unlink must not raise.
    ring.close_and_unlink()
    ring.close_and_unlink()
    ring.close_and_unlink()


def test_attach_rejects_wrong_dtype(ring_name):
    """Producer creates float32; consumer attach with float64 must raise."""
    producer = ShmRing.create(
        name=ring_name,
        slot_shape=(4,),
        dtype=np.dtype(np.float32),
        capacity=8,
    )
    try:
        with pytest.raises(ValueError, match="dtype"):
            ShmRing.attach(
                name=ring_name,
                slot_shape=(4,),
                dtype=np.dtype(np.float64),
                capacity=8,
            )
    finally:
        producer.close_and_unlink()


def test_attach_rejects_wrong_shape(ring_name):
    """Producer creates slot_shape=(8,); consumer attach (7,) must raise."""
    producer = ShmRing.create(
        name=ring_name,
        slot_shape=(8,),
        dtype=np.dtype(np.float32),
        capacity=4,
    )
    try:
        with pytest.raises(ValueError, match="shape"):
            ShmRing.attach(
                name=ring_name,
                slot_shape=(7,),
                dtype=np.dtype(np.float32),
                capacity=4,
            )
    finally:
        producer.close_and_unlink()


def test_attach_rejects_wrong_capacity(ring_name):
    """Producer creates capacity=4; consumer attach capacity=8 must raise."""
    producer = ShmRing.create(
        name=ring_name,
        slot_shape=(4,),
        dtype=np.dtype(np.float32),
        capacity=4,
    )
    try:
        with pytest.raises(ValueError, match="capacity"):
            ShmRing.attach(
                name=ring_name,
                slot_shape=(4,),
                dtype=np.dtype(np.float32),
                capacity=8,
            )
    finally:
        producer.close_and_unlink()


def _matching_metadata_producer(name: str, ready_q, done_q):
    """Producer subprocess for matching-metadata round-trip test."""
    import numpy as np

    from chaoscontrol.episodic.ipc import ShmRing

    ring = ShmRing.create(
        name=name,
        slot_shape=(2, 3),
        dtype=np.dtype(np.float32),
        capacity=4,
    )
    try:
        ready_q.put("ready")
        for i in range(3):
            ring.try_write(
                np.full((2, 3), float(i), dtype=np.float32),
            )
        done_q.put("done")
        ready_q.get()  # wait for "drained" sentinel
    finally:
        ring.close_and_unlink()


def test_attach_succeeds_with_matching_metadata(ring_name):
    """Sanity: matching metadata still round-trips across processes.

    Proves the metadata-write path doesn't break the happy path. Uses 2-D
    slot_shape to exercise multi-dim metadata storage end-to-end.
    """
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue()
    done_q = ctx.Queue()

    proc = ctx.Process(
        target=_matching_metadata_producer,
        args=(ring_name, ready_q, done_q),
    )
    proc.start()
    try:
        assert ready_q.get(timeout=10.0) == "ready"

        reader = ShmRing.attach(
            name=ring_name,
            slot_shape=(2, 3),
            dtype=np.dtype(np.float32),
            capacity=4,
        )
        try:
            assert done_q.get(timeout=10.0) == "done"
            for i in range(3):
                got = reader.try_read()
                assert got is not None
                np.testing.assert_array_equal(
                    got,
                    np.full((2, 3), float(i), dtype=np.float32),
                )
            assert reader.try_read() is None
        finally:
            reader.close()
            ready_q.put("drained")
            proc.join(timeout=10.0)
            assert proc.exitcode == 0, f"producer exited {proc.exitcode}"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)


def test_create_rejects_oversized_ndim(ring_name):
    """slot_shape with more than MAX_NDIM=8 dims must raise on create()."""
    with pytest.raises(ValueError, match="MAX_NDIM"):
        ShmRing.create(
            name=ring_name,
            slot_shape=(2,) * 9,
            dtype=np.dtype(np.float32),
            capacity=2,
        )
