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

import pytest
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
            pressure_at_write=float(i) / 10.0,
            source_write_id=10_000 + i,
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
        assert set(tag.keys()) >= {
            "step",
            "slot",
            "score",
            "selected_at",
            "replay_id",
            "query_event_id",
            "source_write_id",
            "selected_rank",
            "teacher_score",
            "controller_logit",
            "selection_step",
            "policy_version",
            "outcome_status",
        }
        assert isinstance(tag["slot"], int)
        assert isinstance(tag["score"], float)
        assert isinstance(tag["step"], int)
        assert isinstance(tag["selected_at"], int)
        assert isinstance(tag["replay_id"], int)
        assert isinstance(tag["query_event_id"], int)
        assert isinstance(tag["source_write_id"], int)
        assert tag["outcome_status"] == "pending"
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
    for i, (key_rep, utility, pressure) in enumerate([
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), 0.1, 0.2),
        (torch.tensor([0.0, 1.0, 0.0, 0.0]), 0.9, 0.1),
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), 0.5, 0.8),
    ]):
        cache.append(
            key_fp=1000 + i,
            key_rep=key_rep,
            value_tok_ids=torch.zeros(cache.span_length, dtype=torch.int64),
            value_anchor_id=0,
            current_step=0,
            embedding_version=0,
            pressure_at_write=pressure,
        )
        cache.utility_u[i] = utility
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

    # Pressure-only: cosine and utility ignored. Top-1 = highest pressure slot 2.
    po_q: list[dict] = [dict(candidate)]
    po_tags: list[dict] = []
    run_controller_cycle(
        controller_query_queue=po_q,
        tagged_replay_queue=po_tags,
        cache=cache,
        k=1,
        score_mode="pressure_only",
    )
    assert po_tags[0]["slot"] == 2


def test_controller_main_builds_rank_prefixed_ids_when_query_id_absent():
    """Legacy in-process candidates do not carry QUERY_EVENT ids yet. The
    controller must still emit durable rank-prefixed ids so replay outcomes
    can be joined to selections.
    """
    cache = _make_cache(key_rep_dim=4)
    _populate_cache(
        cache,
        key_reps=[torch.tensor([1.0, 0.0, 0.0, 0.0])],
        utilities=[0.5],
    )
    q = [{
        "step": 7,
        "rank": 3,
        "k": 5,
        "pressure": 0.5,
        "residual": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    }]
    tags: list[dict] = []
    run_controller_cycle(
        controller_query_queue=q,
        tagged_replay_queue=tags,
        cache=cache,
        k=1,
        score_mode="cosine_utility_weighted",
        selected_at=123,
        policy_version=4,
    )
    assert len(tags) == 1
    tag = tags[0]
    assert tag["query_event_id"] == (3 << 56) | 5
    # replay_id clamps query_event_id to 56 bits before the 8-bit shift
    # so the packed value fits in u64 (DuckDB BIGINT). The high 8 bits of
    # query_event_id are the source_rank and are preserved separately on
    # the same tag dict via the unclamped query_event_id field above.
    assert tag["replay_id"] == (((3 << 56) | 5) & ((1 << 56) - 1)) << 8
    assert tag["replay_id"] < (1 << 64)
    assert tag["selection_step"] == 123
    assert tag["policy_version"] == 4


def test_controller_main_uses_optional_runtime_for_controller_logit():
    class RuntimeStub:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, int]] = []

        def score_slot(self, features: torch.Tensor, *, slot: int) -> torch.Tensor:
            self.calls.append((features.clone(), int(slot)))
            return torch.tensor(100.0 + float(slot), dtype=torch.float32)

    cache = _make_cache(key_rep_dim=4)
    _populate_cache(
        cache,
        key_reps=[torch.tensor([1.0, 0.0, 0.0, 0.0])],
        utilities=[0.5],
    )
    runtime = RuntimeStub()
    tags: list[dict] = []
    run_controller_cycle(
        controller_query_queue=[{
            "step": 1,
            "rank": 0,
            "k": 0,
            "pressure": 0.25,
            "residual": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }],
        tagged_replay_queue=tags,
        cache=cache,
        k=1,
        score_mode="cosine_utility_weighted",
        controller_runtime=runtime,
    )
    assert len(runtime.calls) == 1
    assert tags[0]["teacher_score"] == tags[0]["score"]
    assert tags[0]["controller_logit"] == 100.0


def test_controller_main_serializes_runtime_snapshots_without_tensors():
    """Runtime-backed tags must carry delayed-learning snapshots as plain
    Python lists, not tensors that can keep device memory alive.
    """
    from chaoscontrol.episodic.cpu_ssm_controller import (
        CpuSsmControllerRuntime,
        CpuSsmControllerWeights,
    )

    weights = CpuSsmControllerWeights(
        w_global_in=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        w_slot_in=torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        decay_global=torch.tensor([0.5, 0.25], dtype=torch.float32),
        decay_slot=torch.tensor([0.75], dtype=torch.float32),
        w_global_out=torch.tensor([1.0, 2.0], dtype=torch.float32),
        w_slot_out=torch.tensor([3.0], dtype=torch.float32),
        bias=torch.tensor(0.0, dtype=torch.float32),
    )
    runtime = CpuSsmControllerRuntime(weights, capacity=2, prefer_cpp=False)
    runtime.global_state.copy_(torch.tensor([4.0, 5.0], dtype=torch.float32))
    runtime.slot_state[0].copy_(torch.tensor([6.0], dtype=torch.float32))

    cache = _make_cache(capacity=2, key_rep_dim=4)
    _populate_cache(
        cache,
        key_reps=[torch.tensor([1.0, 0.0, 0.0, 0.0])],
        utilities=[0.5],
    )
    tags: list[dict] = []
    run_controller_cycle(
        controller_query_queue=[{
            "step": 1,
            "rank": 0,
            "k": 0,
            "pressure": 0.25,
            "residual": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }],
        tagged_replay_queue=tags,
        cache=cache,
        k=1,
        score_mode="cosine_utility_weighted",
        controller_runtime=runtime,
    )

    assert len(tags) == 1
    tag = tags[0]
    assert tag["controller_features"] == pytest.approx([0.5, 0.25, 0.5, 0.0])
    assert tag["controller_global_state"] == pytest.approx([4.0, 5.0])
    assert tag["controller_slot_state"] == pytest.approx([6.0])
    for value in tag.values():
        assert not isinstance(value, torch.Tensor)


def test_controller_main_records_snapshot_to_action_history_recorder():
    from chaoscontrol.episodic.cpu_ssm_controller import (
        CpuSsmControllerRuntime,
        CpuSsmControllerWeights,
    )

    class RecorderStub:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def record_replay_selection(self, **kwargs) -> None:
            self.calls.append(kwargs)

    weights = CpuSsmControllerWeights(
        w_global_in=torch.eye(2, 4, dtype=torch.float32),
        w_slot_in=torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        decay_global=torch.ones(2, dtype=torch.float32),
        decay_slot=torch.ones(1, dtype=torch.float32),
        w_global_out=torch.ones(2, dtype=torch.float32),
        w_slot_out=torch.ones(1, dtype=torch.float32),
        bias=torch.tensor(0.0, dtype=torch.float32),
    )
    runtime = CpuSsmControllerRuntime(weights, capacity=2, prefer_cpp=False)
    runtime.global_state.copy_(torch.tensor([1.0, 2.0], dtype=torch.float32))
    runtime.slot_state[0].copy_(torch.tensor([3.0], dtype=torch.float32))
    recorder = RecorderStub()

    cache = _make_cache(capacity=2, key_rep_dim=4)
    _populate_cache(
        cache,
        key_reps=[torch.tensor([1.0, 0.0, 0.0, 0.0])],
        utilities=[0.5],
    )
    tags: list[dict] = []
    run_controller_cycle(
        controller_query_queue=[{
            "step": 1,
            "rank": 0,
            "k": 0,
            "pressure": 0.25,
            "residual": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }],
        tagged_replay_queue=tags,
        cache=cache,
        k=1,
        score_mode="cosine_utility_weighted",
        selected_at=77,
        policy_version=9,
        controller_runtime=runtime,
        action_recorder=recorder,
    )

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["slot_id"] == tags[0]["slot"]
    assert call["gpu_step"] == 77
    assert call["policy_version"] == 9
    assert call["output_logit"] == pytest.approx(tags[0]["controller_logit"])
    assert call["selected_rank"] == 0
    assert call["features"] == pytest.approx(tags[0]["controller_features"])
    assert call["global_state"] == pytest.approx(
        tags[0]["controller_global_state"]
    )
    assert call["slot_state"] == pytest.approx(tags[0]["controller_slot_state"])


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


def test_controller_main_releases_residual_tensor_via_weakref():
    """Stronger version of the above: the controller's local
    ``cand["residual"] = None`` reference-drop MUST actually release the
    tensor. Without this test, the previous one would pass even if the
    reference-drop line were deleted (because the queue itself never
    held a tensor — just the stack frame inside the controller did).

    Use a weakref to confirm that after run_controller_cycle returns AND
    the test's local reference is dropped, the residual is garbage-
    collected. If the controller pinned it past one cycle (in any local
    list, snapshot, or tag), the weakref stays live.
    """
    import gc
    import weakref

    cache = _make_cache(key_rep_dim=4)
    _populate_cache(
        cache,
        key_reps=[torch.tensor([1.0, 0.0, 0.0, 0.0])],
        utilities=[0.5],
    )
    residual = torch.tensor([1.0, 0.0, 0.0, 0.0])
    wr = weakref.ref(residual)
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
    # Drop our test's local references to the residual.
    del residual
    del controller_query_queue  # also drops the dict's reference
    gc.collect()
    assert wr() is None, (
        "controller pinned the residual past one cycle; "
        "the cand['residual'] = None reference-drop is missing or broken"
    )


def test_replay_id_packs_into_u64_at_max_rank():
    """At extreme query_event_id values combined with the largest legal
    selected_rank, the packed replay_id must still fit in u64. The schema
    declares ``replay_id u64`` and DuckDB BIGINT silently truncates a 72-bit
    overflow — without the 56-bit clamp on query_event_id, a tag emitted at
    high source_rank with selected_rank=255 would round-trip to a different
    integer.
    """
    cache = _make_cache(key_rep_dim=4)
    _populate_cache(
        cache,
        key_reps=[torch.tensor([1.0, 0.0, 0.0, 0.0])],
        utilities=[0.5],
    )
    # source_rank=255 (max 8-bit) + rank_seq=(2**56)-1 → query_event_id
    # is exactly the 64-bit ceiling, the worst case for packing.
    high_rank = 255
    high_seq = (1 << 56) - 1
    expected_qid = (high_rank << 56) | high_seq
    tags: list[dict] = []
    run_controller_cycle(
        controller_query_queue=[{
            "step": 99,
            "source_rank": high_rank,
            "rank_seq": high_seq,
            "pressure": 0.0,
            "residual": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }],
        tagged_replay_queue=tags,
        cache=cache,
        k=1,
        score_mode="cosine_utility_weighted",
    )
    assert len(tags) == 1
    tag = tags[0]
    assert tag["query_event_id"] == expected_qid
    assert tag["selected_rank"] == 0
    # Pack at the largest legal selected_rank as well — the controller
    # currently emits one tag per slot (selected_rank starts at 0), so we
    # also synthesize a max-rank tag by packing directly with the same
    # formula and confirming the bound holds.
    max_rank = 0xFF
    packed_at_max = ((expected_qid & ((1 << 56) - 1)) << 8) | max_rank
    assert packed_at_max < (1 << 64)
    assert tag["replay_id"] < (1 << 64)
