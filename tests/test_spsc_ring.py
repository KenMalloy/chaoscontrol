"""Lock-free SPSC ring buffer (Phase A2 of CPU SSM controller plan).

Pins single-threaded correctness (push/pop/size/capacity) and
multi-threaded correctness (producer thread races consumer thread,
no torn reads, all events arrive in order). The wire-event structs
from Phase A1 (WriteEvent / QueryEvent / ReplayOutcome) will use
this ring template in Phase A4 (ShmRing); Phase A2's ring is in-
process only.

Spec: docs/plans/2026-04-26-cpu-ssm-controller.md (Task A2).
"""
from __future__ import annotations

import threading

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_ring_capacity_is_1024():
    """Capacity is a compile-time template parameter (power-of-two);
    the binding exposes it as a static class property."""
    ring = _ext.SpscRingU64x1024()
    assert _ext.SpscRingU64x1024.capacity == 1024, (
        f"expected capacity 1024, got {_ext.SpscRingU64x1024.capacity}"
    )
    assert ring.size() == 0, f"new ring should be empty, got size={ring.size()}"


def test_ring_push_pop_single_threaded_drains_in_order():
    """Producer writes 1024 events, consumer reads 1024, all match,
    ring is empty after. push beyond capacity returns False (not None,
    not raises) — this catches a binding that returns Python None
    instead of bool."""
    ring = _ext.SpscRingU64x1024()
    for i in range(1024):
        pushed = ring.push(i)
        assert pushed is True, f"push {i} returned {pushed!r} (size={ring.size()})"
    assert ring.size() == 1024, f"expected size 1024, got {ring.size()}"

    overflow = ring.push(2024)
    assert overflow is False, (
        f"push beyond capacity should return False, got {overflow!r}"
    )

    for i in range(1024):
        popped = ring.pop()
        assert popped == i, f"expected {i}, got {popped!r}"
    assert ring.size() == 0, f"expected empty, got size {ring.size()}"
    empty = ring.pop()
    assert empty is None, f"pop on empty should return None, got {empty!r}"


def test_ring_multithreaded_producer_consumer_no_torn_reads():
    """Producer thread pushes 100k events; consumer thread pops them.
    All values must arrive in monotonic order — any out-of-order or
    duplicate read indicates a torn read or memory-ordering bug.

    Note: the GIL serializes Python-side push/pop calls, so this test
    pins FIFO + no-loss correctness rather than truly stressing
    release/acquire memory ordering. That's acceptable for Phase A2 —
    A4's ShmRing will get the cross-process workout."""
    ring = _ext.SpscRingU64x1024()
    n_events = 100_000

    received: list[int] = []

    def producer() -> None:
        i = 0
        while i < n_events:
            if ring.push(i):
                i += 1
            # else: ring is full, retry until consumer drains a slot

    def consumer() -> None:
        while len(received) < n_events:
            v = ring.pop()
            if v is not None:
                received.append(v)

    p = threading.Thread(target=producer)
    c = threading.Thread(target=consumer)
    p.start()
    c.start()
    p.join()
    c.join()

    assert len(received) == n_events, (
        f"expected {n_events} events, got {len(received)}"
    )
    if received != list(range(n_events)):
        # Find first divergence to give a useful failure message.
        first_bad = next(
            i for i, (a, b) in enumerate(zip(received, range(n_events))) if a != b
        )
        raise AssertionError(
            f"out-of-order at index {first_bad}: "
            f"got {received[first_bad]}, expected {first_bad}"
        )
