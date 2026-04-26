"""Phase 2 episodic controller (Tasks 2.1 + 2.2 of the memory-aware optimizer plan).

The controller is the "control plane" of the curated-Dreamworld
architecture: it sits off the SSM step path, drains the per-rank
``controller_query_queue`` populated by ``_drain_episodic_payloads_gpu``,
runs ``query_topk`` against the episodic cache for each candidate, and
pushes the resulting slot indices to ``tagged_replay_queue`` for the
Phase 3 replay path (consumed by the Y worktree's dreamworld replay
backward).

Design choice — in-process daemon thread, NOT ``multiprocessing.spawn``:
the plan's Decision 0.7 specifies a child process for the controller,
but Pass C dropped POSIX shm in favor of in-process Python lists that
carry **GPU fp32 residual tensors**. Marshalling those across a process
boundary would require CUDA IPC handles (fragile, NCCL-dependent) or a
device-to-host copy on every candidate (expensive). The plan
authorizes the simplification: "do this in-process if
multiprocessing.spawn adds too much complexity for the smoke test —
flag in the report."

This module exposes two entry points:

  ``run_controller_cycle`` — single-cycle helper. Drains the query
  queue, calls ``query_topk`` per candidate, appends tags to the replay
  queue. Pure function; no threading. Used directly by tests and by the
  long-running loop.

  ``controller_main`` — long-running loop. Wraps ``run_controller_cycle``
  with a ``stop_event`` check + idle sleep. Designed to run as a
  ``threading.Thread(daemon=True, target=controller_main, ...)``
  spawned from the runner's init region.

Both helpers take the queue lock as an explicit parameter so tests can
exercise the lock-free single-threaded path. The runner-side spawn
binds a real ``threading.Lock`` shared with the producer side (the
drain function runs on the episodic rank's main thread).

Phase 5+ extensions (not implemented now): per-cycle wall-clock budget,
priority-based candidate selection from DuckDB analytics (Phase 3.5
hook), heartbeat counter for the runner's outer loop, crash recovery.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import torch

from chaoscontrol.episodic.query import query_topk
from chaoscontrol.optim.episodic_cache import EpisodicCache


def run_controller_cycle(
    *,
    controller_query_queue: list[dict[str, Any]],
    tagged_replay_queue: list[dict[str, Any]],
    cache: EpisodicCache,
    k: int = 16,
    score_mode: str = "cosine_utility_weighted",
    queue_lock: threading.Lock | None = None,
    selected_at: int | None = None,
) -> int:
    """Run a single controller cycle: drain queries, push tags.

    Args:
        controller_query_queue: producer-side queue populated by
            ``_drain_episodic_payloads_gpu``. Each entry has keys
            ``step, rank, k, pressure, residual`` where ``residual`` is
            an fp32 tensor of shape ``[D]``. The cycle drains the entire
            queue under the lock (snapshot-and-clear).
        tagged_replay_queue: consumer-side queue. Each appended tag is
            a dict with keys ``step, slot, score, selected_at``.
            Tensors MUST NOT be stored on the tag (slow-OOM hazard);
            this is a pinned invariant.
        cache: the episodic cache. Read-only from the controller's
            perspective for the cycle.
        k: top-K passed to ``query_topk``. Default 16 per Decision 0.2.
        score_mode: ``"cosine_utility_weighted"`` (default) or
            ``"pressure_only"`` (Phase 3 Arm B').
        queue_lock: protects the snapshot-and-clear of
            ``controller_query_queue``. None means "single-threaded
            test mode — no lock needed."
        selected_at: wall-clock-ish stamp written into each tag. None
            means use ``time.monotonic_ns() // 1_000_000`` (millisecond
            granularity, monotonic across the cycle). Tests pass an
            explicit value for determinism.

    Returns:
        Number of candidates drained from the query queue this cycle.
        Note: the count of tags appended to ``tagged_replay_queue`` is
        ``num_drained * k_eff_per_query``; if the cache is small or
        empty, ``k_eff_per_query`` may be < k. Callers wanting both
        numbers should diff ``len(tagged_replay_queue)`` before/after.
    """
    # Snapshot-and-clear under the lock so a concurrent producer
    # (the drain function on the same rank's main thread) doesn't see
    # a partially-mutated list. The default Python list mutation isn't
    # atomic across append + clear, so the lock is load-bearing whenever
    # the controller runs in a separate thread.
    if queue_lock is not None:
        with queue_lock:
            candidates = list(controller_query_queue)
            controller_query_queue.clear()
    else:
        candidates = list(controller_query_queue)
        controller_query_queue.clear()

    if not candidates:
        return 0

    if selected_at is None:
        # Monotonic millisecond stamp. Cheap, unambiguous, doesn't
        # collide with the producer's ``step`` field (which is the
        # training step at queue-emit time).
        selected_at = int(time.monotonic_ns() // 1_000_000)

    for cand in candidates:
        residual = cand["residual"]
        # ``query_topk`` returns the ranked slot indices; we re-derive
        # the score from the cache to log alongside each tag. The
        # cosine_utility_weighted score is what Phase 3 logs as
        # ``query_cosine`` × ``utility_pre`` per Decision 0.9 — same
        # quantity, computed twice for now (test-pin clarity > one-shot
        # micro-optimization).
        slots = query_topk(residual, cache, k=int(k), score_mode=score_mode)
        if slots.numel() == 0:
            continue
        # Recompute scores for the returned slots so we can log them
        # without re-ranking. Two GPU vector ops per query — same cost
        # as the inner ``query_topk`` call, but it stays out of the
        # query helper's API surface (which only contracts on slot
        # indices, not scores).
        scores = _scores_for_slots(
            residual=residual,
            cache=cache,
            slots=slots,
            score_mode=score_mode,
        )
        # ``slots`` and ``scores`` are tensors; ``.tolist()`` does the
        # device-to-host sync once per candidate (cheap — slots is at
        # most ``k`` int64 entries, scores is the same).
        slot_list = slots.tolist()
        score_list = scores.tolist()
        producer_step = int(cand.get("step", -1))
        for slot_i, score_i in zip(slot_list, score_list, strict=True):
            tagged_replay_queue.append({
                "step": producer_step,
                "slot": int(slot_i),
                "score": float(score_i),
                "selected_at": int(selected_at),
            })
        # Drop the residual reference explicitly. ``cand`` goes out of
        # scope at end of loop iteration, but explicit clearing makes
        # the slow-OOM invariant visible in the source.
        cand["residual"] = None

    return len(candidates)


def _scores_for_slots(
    *,
    residual: torch.Tensor,
    cache: EpisodicCache,
    slots: torch.Tensor,
    score_mode: str,
) -> torch.Tensor:
    """Compute the per-slot score on a small set of slots.

    Mirrors the score formula in ``query.query_topk`` but only over the
    ``slots`` returned by it. Used to populate the tag's ``score`` field
    for the Phase 3 diagnostic log. Returns an fp32 tensor on the
    residual's device.
    """
    device = residual.device
    if score_mode == "cosine_utility_weighted":
        keys = cache.key_rep.to(device)[slots]  # [k_eff, D]
        util = cache.utility_u.to(device)[slots]  # [k_eff]
        q = residual / (residual.norm() + 1e-8)
        keys_n = keys / (keys.norm(dim=1, keepdim=True) + 1e-8)
        cosines = keys_n @ q
        return cosines * util
    if score_mode == "pressure_only":
        return cache.utility_u.to(device)[slots]
    raise ValueError(f"unknown score_mode={score_mode!r}")


def controller_main(
    *,
    controller_query_queue: list[dict[str, Any]],
    tagged_replay_queue: list[dict[str, Any]],
    cache: EpisodicCache,
    queue_lock: threading.Lock,
    stop_event: threading.Event,
    k: int = 16,
    score_mode: str = "cosine_utility_weighted",
    cycle_idle_sleep_s: float = 0.005,
    heartbeat: list[int] | None = None,
) -> None:
    """Long-running controller loop. Designed for ``threading.Thread``.

    Polls ``controller_query_queue`` in a tight loop until
    ``stop_event`` is set. Each iteration calls ``run_controller_cycle``;
    if the cycle drained zero candidates, sleeps ``cycle_idle_sleep_s``
    seconds before checking again so the loop doesn't peg a CPU core.

    The ``heartbeat`` parameter is an optional single-element list of
    int (mirrors ``_EpisodicConsumerState.heartbeat``); incremented
    once per cycle so the runner's outer loop can monitor controller
    liveness via telemetry. None means heartbeat is not tracked.

    The loop does NOT enforce a per-cycle wall budget. Phase 1 design
    deliberately defers production hardening (signal handling, crash
    recovery, ack channel) to Phase 4+. Test 5
    (``test_controller_thread_starts_and_stops_cleanly``) pins the
    stop-event-driven exit path.
    """
    while not stop_event.is_set():
        n = run_controller_cycle(
            controller_query_queue=controller_query_queue,
            tagged_replay_queue=tagged_replay_queue,
            cache=cache,
            k=k,
            score_mode=score_mode,
            queue_lock=queue_lock,
        )
        if heartbeat is not None:
            heartbeat[0] += 1
        if n == 0:
            # Idle cycle — sleep briefly to keep CPU usage off the
            # train-step path. ``stop_event.wait(...)`` returns True if
            # the event fires during the sleep, which gives us a clean
            # exit without a separate poll.
            if stop_event.wait(timeout=cycle_idle_sleep_s):
                break
