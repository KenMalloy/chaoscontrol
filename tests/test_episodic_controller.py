"""Tests for the Phase 2 episodic controller (Tasks 2.1 + 2.2).

The controller drains the episodic rank's ``controller_query_queue``
(populated by ``_drain_episodic_payloads_gpu``), invokes ``query_topk``
on each candidate residual against the cache, and pushes the resulting
slot indices to ``tagged_replay_queue`` for the Phase 3 replay path
(Y worktree's territory).

Phase 2 simplification (per the plan's "do this in-process if
multiprocessing.spawn adds too much complexity"):
the controller runs as an in-process daemon ``threading.Thread`` rather
than a ``multiprocessing.spawn`` child. Justification: Pass C dropped
POSIX shm in favor of in-process Python lists carrying GPU fp32
residual tensors; marshalling those across a process boundary would
need CUDA IPC handles and re-architects the producer side. A daemon
thread shares the cache + queues natively, and the producer side
(``_drain_episodic_payloads_gpu``) is single-writer per step, so a
single ``threading.Lock`` around the queue swap is sufficient.

Tests:

  1. ``test_controller_main_drains_queries_and_tags`` — single-cycle
     smoke: pre-populate ``controller_query_queue``, run one cycle,
     assert ``tagged_replay_queue`` grew and ``controller_query_queue``
     drained.
  2. ``test_controller_main_handles_empty_queue`` — zero candidates per
     cycle is a no-op (no exceptions, no tags).
  3. ``test_controller_main_handles_empty_cache`` — non-zero candidates
     against an empty cache produces zero tags (per ``query_topk``
     empty-cache contract) without crashing.
  4. ``test_controller_main_respects_score_mode`` — the controller
     forwards ``score_mode`` to ``query_topk``; running once under each
     mode against the same cache + query produces the expected slot
     ordering.
  5. ``test_controller_thread_starts_and_stops_cleanly`` — long-running
     loop variant: start the controller thread, push a few candidates,
     wait for the queue to drain, set the stop event, join. The thread
     must terminate within a short wall budget.
  6. ``test_controller_main_drops_residuals_after_processing`` — pinned
     hygiene: after a query candidate is processed, its GPU residual
     reference is dropped from the controller's local state so it
     doesn't pin VRAM beyond one cycle.
"""
from __future__ import annotations

import threading
import time

import torch

from chaoscontrol.episodic.controller import (
    controller_main,
    run_controller_cycle,
)
from chaoscontrol.optim.episodic_cache import EpisodicCache


def _make_cache(*, capacity: int = 8, key_rep_dim: int = 4) -> EpisodicCache:
    """Tiny cache for controller tests."""
    return EpisodicCache(
        capacity=capacity,
        span_length=2,
        key_rep_dim=key_rep_dim,
        grace_steps=10,
    )


def _populate_cache(cache: EpisodicCache, *, key_reps: list[torch.Tensor],
                    utilities: list[float]) -> None:
    """Append the given (key_rep, utility) entries into the cache.

    The append API initializes ``utility_u`` to 1.0 (Decision 0.2
    cold-start), so we override directly after each append.
    """
    span = cache.span_length
    for i, kr in enumerate(key_reps):
        cache.append(
            key_fp=1000 + i,
            key_rep=kr,
            value_tok_ids=torch.zeros(span, dtype=torch.int64),
            value_anchor_id=0,
            current_step=0,
            embedding_version=0,
        )
        cache.utility_u[i] = float(utilities[i])


def test_controller_main_drains_queries_and_tags():
    """Pre-populate the query queue with 3 entries; one cycle drains
    all 3 and produces 3 tag entries (one per candidate, k=2 so each
    candidate produces up to 2 tagged slots).
    """
    cache = _make_cache(key_rep_dim=4)
    _populate_cache(
        cache,
        key_reps=[
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
        ],
        utilities=[0.9, 0.5, 0.3],
    )
    controller_query_queue: list[dict] = [
        {
            "step": 100,
            "rank": 0,
            "k": 0,
            "pressure": 0.5,
            "residual": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        },
        {
            "step": 100,
            "rank": 1,
            "k": 0,
            "pressure": 0.4,
            "residual": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        },
        {
            "step": 101,
            "rank": 0,
            "k": 0,
            "pressure": 0.3,
            "residual": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        },
    ]
    tagged_replay_queue: list[dict] = []
    n_processed = run_controller_cycle(
        controller_query_queue=controller_query_queue,
        tagged_replay_queue=tagged_replay_queue,
        cache=cache,
        k=2,
        score_mode="cosine_utility_weighted",
    )
    # All 3 candidates were drained.
    assert n_processed == 3
    assert controller_query_queue == []
    # Each candidate produces up to k=2 tags (3 occupied slots in cache,
    # so all 3 produce 2 tags = 6 total).
    assert len(tagged_replay_queue) == 6
    # Tag schema pin.
    for tag in tagged_replay_queue:
        assert set(tag.keys()) >= {"step", "slot", "score", "selected_at"}
        assert isinstance(tag["slot"], int)
        assert isinstance(tag["score"], float)
        assert isinstance(tag["step"], int)
        assert isinstance(tag["selected_at"], int)
    # First candidate aligned with key_rep[0] → top-1 slot is 0.
    first_tag = tagged_replay_queue[0]
    assert first_tag["slot"] == 0
    assert first_tag["step"] == 100  # carries the producer step


def test_controller_main_handles_empty_queue():
    """Empty queue → zero processed, zero tags, no exception."""
    cache = _make_cache()
    controller_query_queue: list[dict] = []
    tagged_replay_queue: list[dict] = []
    n = run_controller_cycle(
        controller_query_queue=controller_query_queue,
        tagged_replay_queue=tagged_replay_queue,
        cache=cache,
        k=2,
        score_mode="cosine_utility_weighted",
    )
    assert n == 0
    assert tagged_replay_queue == []


def test_controller_main_handles_empty_cache():
    """Non-zero candidates against an empty cache produces zero tags."""
    cache = _make_cache()  # nothing appended
    controller_query_queue: list[dict] = [
        {
            "step": 5,
            "rank": 0,
            "k": 0,
            "pressure": 0.5,
            "residual": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        },
    ]
    tagged_replay_queue: list[dict] = []
    n = run_controller_cycle(
        controller_query_queue=controller_query_queue,
        tagged_replay_queue=tagged_replay_queue,
        cache=cache,
        k=2,
        score_mode="cosine_utility_weighted",
    )
    # Drained the candidate but produced no tags (empty cache → empty
    # query_topk output).
    assert n == 1
    assert controller_query_queue == []
    assert tagged_replay_queue == []


def test_controller_main_respects_score_mode():
    """Same cache, same query, two different score modes — slot
    ordering follows ``score_mode``."""
    cache = _make_cache(key_rep_dim=4)
    _populate_cache(
        cache,
        key_reps=[
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
        ],
        # Slot 0 has lowest utility; slot 1 has highest.
        utilities=[0.1, 0.9, 0.5],
    )
    # Query exactly aligned with slot 0's key_rep.
    candidate = {
        "step": 1,
        "rank": 0,
        "k": 0,
        "pressure": 0.5,
        "residual": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    }
    # Cosine-utility weighted: cosine_0 = 1.0 * util 0.1 = 0.1 (winner
    # against cosine_1 = 0 * 0.9 = 0 and cosine_2 = 0 * 0.5 = 0).
    # Top-1 = slot 0.
    cuw_q: list[dict] = [dict(candidate)]
    cuw_tags: list[dict] = []
    run_controller_cycle(
        controller_query_queue=cuw_q,
        tagged_replay_queue=cuw_tags,
        cache=cache,
        k=1,
        score_mode="cosine_utility_weighted",
    )
    assert cuw_tags[0]["slot"] == 0

    # Pressure-only: cosine ignored. Top-1 = highest utility slot 1.
    po_q: list[dict] = [dict(candidate)]
    po_tags: list[dict] = []
    run_controller_cycle(
        controller_query_queue=po_q,
        tagged_replay_queue=po_tags,
        cache=cache,
        k=1,
        score_mode="pressure_only",
    )
    assert po_tags[0]["slot"] == 1


def test_controller_thread_starts_and_stops_cleanly():
    """Long-running variant: spawn a daemon thread, push candidates,
    wait for them to drain, signal stop, join. Thread must exit within
    a short wall budget so a stuck loop doesn't hang the runner.
    """
    cache = _make_cache(key_rep_dim=4)
    _populate_cache(
        cache,
        key_reps=[
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
        ],
        utilities=[0.7, 0.3],
    )
    controller_query_queue: list[dict] = []
    tagged_replay_queue: list[dict] = []
    queue_lock = threading.Lock()
    stop_event = threading.Event()

    thread = threading.Thread(
        target=controller_main,
        kwargs={
            "controller_query_queue": controller_query_queue,
            "tagged_replay_queue": tagged_replay_queue,
            "cache": cache,
            "queue_lock": queue_lock,
            "stop_event": stop_event,
            "k": 2,
            "score_mode": "cosine_utility_weighted",
            "cycle_idle_sleep_s": 0.001,
        },
        daemon=True,
        name="episodic_controller_test",
    )
    thread.start()
    try:
        # Push 3 candidates over a short window. Use the lock to mirror
        # the runner's drain pattern.
        for i in range(3):
            with queue_lock:
                controller_query_queue.append({
                    "step": i,
                    "rank": 0,
                    "k": 0,
                    "pressure": 0.5,
                    "residual": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                })
            time.sleep(0.005)

        # Wait up to 1s for the controller to drain everything.
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            with queue_lock:
                if not controller_query_queue and tagged_replay_queue:
                    break
            time.sleep(0.005)

        with queue_lock:
            assert controller_query_queue == [], (
                f"controller did not drain queue: "
                f"{len(controller_query_queue)} entries remain"
            )
            assert len(tagged_replay_queue) > 0, (
                "controller produced no tags despite non-empty cache"
            )
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), (
            "controller thread did not stop within 2s of stop_event"
        )


def test_controller_main_drops_residuals_after_processing():
    """After the controller processes a candidate, the residual tensor
    reference must not leak into the tagged_replay_queue (the tags only
    carry slot + score + step). This protects against the slow-OOM
    regression that gated controller_query_enabled to False in Phase 1.
    """
    cache = _make_cache(key_rep_dim=4)
    _populate_cache(
        cache,
        key_reps=[torch.tensor([1.0, 0.0, 0.0, 0.0])],
        utilities=[0.5],
    )
    residual = torch.tensor([1.0, 0.0, 0.0, 0.0])
    controller_query_queue = [{
        "step": 1,
        "rank": 0,
        "k": 0,
        "pressure": 0.5,
        "residual": residual,
    }]
    tagged_replay_queue: list[dict] = []
    run_controller_cycle(
        controller_query_queue=controller_query_queue,
        tagged_replay_queue=tagged_replay_queue,
        cache=cache,
        k=1,
        score_mode="cosine_utility_weighted",
    )
    # The tag must NOT carry a tensor field.
    for tag in tagged_replay_queue:
        for v in tag.values():
            assert not isinstance(v, torch.Tensor), (
                f"tag carried a tensor: {tag} — would pin VRAM"
            )
