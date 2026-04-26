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


_SIMPLEX_K_V = 16  # vertex feature dim — must match the pretrain spec.
_SIMPLEX_K_S = 4   # simplex feature dim.
_SIMPLEX_AGE_NORMALIZER = 1000.0  # divide age in steps by this for V[i, 2].


def _is_simplex_runtime(runtime: Any) -> bool:
    # Duck-type: SimplexOnlineLearner has record_simplex_decision +
    # fast_weights; V0 CpuSsmControllerRuntime / _OnlineLearningRuntimeBridge
    # have score_slot / score_slot_with_snapshot. The two surfaces don't
    # overlap, so a single attribute check picks the right path without
    # the controller importing the kernel extension.
    return runtime is not None and hasattr(runtime, "record_simplex_decision")


def _build_simplex_inputs(
    *,
    residual: torch.Tensor,
    cache: EpisodicCache,
    slots: torch.Tensor,
    scores: torch.Tensor,
    current_step: int,
    pad_to: int = 16,
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Build (V, E, simplex_features, padded_slot_ids) for one query.

    V[i, 0..K_v-1]: per-vertex features. Column 0 is utility (matches
    the BC pretrain target = argmax(V[:, 0])); other columns carry
    cosine, normalized age, pressure-at-write. Unused columns zero-pad
    to ``K_v=16``.

    E[i, j]: pairwise cosine over key_rep. Diagonal = 1.

    simplex_features[K_s=4]: top-1 utility, mean utility, top-1
    cosine_q, cosine_q spread (max - min).

    Returns the V/E/sf vectors as flat float lists ready for the C++
    record_simplex_decision call, plus the padded slot_ids list (length
    ``pad_to``, zero-padded when fewer than 16 candidates were
    retrieved). The simplex forward takes the padded shape; vertices
    beyond the actual candidate count have zero features and contribute
    zero gradient (the policy can't favor them).
    """
    n_actual = int(slots.numel())
    n = int(pad_to)

    # Per-slot scalars
    slot_indices = slots.detach().to(device="cpu", dtype=torch.long).tolist()
    score_list = scores.detach().to(device="cpu", dtype=torch.float32).tolist()
    utilities = [0.0] * n
    cosines = [0.0] * n
    ages = [0.0] * n
    pressures = [0.0] * n
    padded_slot_ids = [0] * n
    for rank, slot_i in enumerate(slot_indices):
        utilities[rank] = float(cache.utility_u[int(slot_i)].item())
        cosines[rank] = float(score_list[rank])
        ages[rank] = float(
            (int(current_step) - int(cache.write_step[int(slot_i)].item()))
            / _SIMPLEX_AGE_NORMALIZER
        )
        pressures[rank] = float(
            cache.pressure_at_write[int(slot_i)].item()
            if hasattr(cache, "pressure_at_write") else 0.0
        )
        padded_slot_ids[rank] = int(slot_i)

    # V[i, 0]=utility, V[i, 1]=cosine, V[i, 2]=age, V[i, 3]=pressure;
    # K_v=16 → indices 4..15 are zero. Column 0 is the BC-pretrain
    # target signal, so the pretrained W_vp leans on it most.
    V = [0.0] * (n * _SIMPLEX_K_V)
    for i in range(n_actual):
        base = i * _SIMPLEX_K_V
        V[base + 0] = utilities[i]
        V[base + 1] = cosines[i]
        V[base + 2] = ages[i]
        V[base + 3] = pressures[i]

    # E[i, j] = cosine(key_rep[i], key_rep[j]). Diagonal = 1 (own dot
    # over own norm).
    if n_actual > 0 and hasattr(cache, "key_rep"):
        key_reps = cache.key_rep[slots.to(dtype=torch.long)]
        norms = key_reps.norm(dim=1, keepdim=True).clamp_min(1e-12)
        normed = key_reps / norms
        cos_actual = (normed @ normed.T).clamp(-1.0, 1.0)
    else:
        cos_actual = torch.zeros(n_actual, n_actual, dtype=torch.float32)

    E = [0.0] * (n * n)
    for i in range(n_actual):
        for j in range(n_actual):
            E[i * n + j] = float(cos_actual[i, j].item())
    # Diagonal beyond n_actual: leave at 0 (padded vertices have no
    # self-similarity to anything real).

    # Simplex features
    if n_actual > 0:
        util_slice = utilities[:n_actual]
        cos_slice = cosines[:n_actual]
        sf = [
            float(max(util_slice)),
            float(sum(util_slice) / len(util_slice)),
            float(max(cos_slice)),
            float(max(cos_slice) - min(cos_slice)),
        ]
    else:
        sf = [0.0] * _SIMPLEX_K_S
    return V, E, sf, padded_slot_ids


def _record_simplex_decision_snapshot(
    *,
    simplex_runtime: Any,
    chosen_slot_id: int,
    gpu_step: int,
    policy_version: int,
    chosen_idx: int,
    p_chosen: float,
    V: list[float],
    E: list[float],
    sf: list[float],
) -> None:
    simplex_runtime.record_simplex_decision(
        chosen_slot_id=int(chosen_slot_id),
        gpu_step=int(gpu_step),
        policy_version=int(policy_version),
        chosen_idx=int(chosen_idx),
        p_chosen_decision=float(p_chosen),
        V=V,
        E=E,
        simplex_features=sf,
    )


def run_controller_cycle(
    *,
    controller_query_queue: list[dict[str, Any]],
    tagged_replay_queue: list[dict[str, Any]],
    cache: EpisodicCache,
    k: int = 16,
    score_mode: str = "cosine_utility_weighted",
    queue_lock: threading.Lock | None = None,
    selected_at: int | None = None,
    policy_version: int = 0,
    controller_runtime: Any | None = None,
    action_recorder: Any | None = None,
) -> int:
    """Run a single controller cycle: drain queries, push tags.

    Args:
        controller_query_queue: producer-side queue populated by
            ``_drain_episodic_payloads_gpu``. Each entry has keys
            ``step, rank, k, pressure, residual`` where ``residual`` is
            an fp32 tensor of shape ``[D]``. The cycle drains the entire
            queue under the lock (snapshot-and-clear).
        tagged_replay_queue: consumer-side queue. Each appended tag is
            a dict with replay/action identity plus ``step, slot, score,
            selected_at``.
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
        action_recorder: optional C10 learning-loop bridge. When supplied and
            the controller runtime exposes decision snapshots, each selected
            replay action is appended with its decision-time input/state.

    Returns:
        Number of candidates drained from the query queue this cycle.
        Note: the count of tags appended to ``tagged_replay_queue`` is
        ``num_drained * k_eff_per_query``; if the cache is small or
        empty, ``k_eff_per_query`` may be < k. Callers wanting both
        numbers should diff ``len(tagged_replay_queue)`` before/after.
    """
    # Drain via repeated ``pop(0)``. Each ``pop(0)`` is a single CPython
    # bytecode and therefore atomic w.r.t. the GIL — it cannot interleave
    # with the producer's ``append()`` call (also single-bytecode, also
    # atomic). This is the race-free pattern for "I cannot lock the
    # producer side": ``snapshot = list(q); q.clear()`` would NOT be
    # atomic across the two calls, so an ``append()`` racing in between
    # would silently drop the newly appended item.
    #
    # A ``queue_lock`` is still respected when supplied: tests pass an
    # explicit lock so they can deterministically synchronize a producer
    # thread against the controller. In production the producer side
    # (``_drain_episodic_payloads_gpu`` on the episodic rank's main
    # thread) does NOT take the lock, so the controller relies on the
    # GIL atomicity of ``pop(0)`` + ``append()``.
    candidates: list[dict[str, Any]] = []
    if queue_lock is not None:
        with queue_lock:
            while controller_query_queue:
                candidates.append(controller_query_queue.pop(0))
    else:
        while True:
            try:
                candidates.append(controller_query_queue.pop(0))
            except IndexError:
                break

    if not candidates:
        return 0

    if selected_at is None:
        # Monotonic millisecond stamp. Cheap, unambiguous, doesn't
        # collide with the producer's ``step`` field (which is the
        # training step at queue-emit time).
        selected_at = int(time.monotonic_ns() // 1_000_000)

    is_simplex = _is_simplex_runtime(controller_runtime)

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
        producer_step = int(cand.get("step", -1))
        query_event_id = _query_event_id_for_candidate(cand)

        if is_simplex:
            # Simplex V1 path: build the candidate matrix from the top-K
            # slots, run simplex_forward once over the whole set, choose
            # one vertex (argmax for V1; sampling is V2). Append a
            # SINGLE tag for the chosen slot — fundamental output-shape
            # difference vs the V0 per-slot loop, which appended K tags
            # per query.
            from chaoscontrol.kernels import _cpu_ssm_controller as _ext

            V, E, sf, padded_slot_ids = _build_simplex_inputs(
                residual=residual,
                cache=cache,
                slots=slots,
                scores=scores,
                current_step=producer_step,
                pad_to=16,
            )
            fwd = _ext.simplex_forward(
                controller_runtime.fast_weights(), V, E, sf,
            )
            # argmax across the actual candidate count (don't pick a
            # zero-padded vertex by accident if the policy somehow
            # weighs them above the real ones).
            n_actual = int(slots.numel())
            best = 0
            best_p = float(fwd.p[0])
            for idx in range(n_actual):
                if float(fwd.p[idx]) > best_p:
                    best = idx
                    best_p = float(fwd.p[idx])
            chosen_idx = best
            chosen_slot = int(padded_slot_ids[chosen_idx])
            p_chosen = best_p
            controller_logit = float(fwd.logits[chosen_idx])

            if action_recorder is not None:
                _record_simplex_decision_snapshot(
                    simplex_runtime=action_recorder,
                    chosen_slot_id=chosen_slot,
                    gpu_step=producer_step,
                    policy_version=int(policy_version),
                    chosen_idx=chosen_idx,
                    p_chosen=p_chosen,
                    V=V, E=E, sf=sf,
                )

            replay_id = (
                (int(query_event_id) & ((1 << 56) - 1)) << 8
            ) | (int(chosen_idx) & 0xFF)
            tag = {
                "step": producer_step,
                "slot": int(chosen_slot),
                "score": float(scores[chosen_idx].item())
                if chosen_idx < int(scores.numel()) else 0.0,
                "selected_at": int(selected_at),
                "replay_id": replay_id,
                "query_event_id": int(query_event_id),
                "source_write_id": int(
                    cache.source_write_id[int(chosen_slot)].item()
                    if hasattr(cache, "source_write_id")
                    else cand.get("source_write_id", -1)
                ),
                "selected_rank": int(chosen_idx),
                "teacher_score": float(
                    scores[chosen_idx].item()
                    if chosen_idx < int(scores.numel()) else 0.0
                ),
                "controller_logit": float(controller_logit),
                "selection_step": int(selected_at),
                "policy_version": int(policy_version),
                "outcome_status": "pending",
                "simplex_p_chosen": float(p_chosen),
                "simplex_chosen_idx": int(chosen_idx),
            }
            tagged_replay_queue.append(tag)
            cand["residual"] = None
            continue

        # V0 per-slot path (unchanged below). ``slots`` and ``scores``
        # are tensors; ``.tolist()`` does the device-to-host sync once
        # per candidate (cheap — slots is at most ``k`` int64 entries,
        # scores is the same).
        slot_list = slots.tolist()
        score_list = scores.tolist()
        for selected_rank, (slot_i, score_i) in enumerate(
            zip(slot_list, score_list, strict=True)
        ):
            controller_logit = float(score_i)
            controller_snapshot: dict[str, Any] = {}
            if controller_runtime is not None:
                controller_logit, controller_snapshot = (
                    _score_runtime_with_optional_snapshot(
                        controller_runtime=controller_runtime,
                        features=_controller_feature_vector(
                            heuristic_score=float(score_i),
                            pressure=float(cand.get("pressure", 0.0)),
                            utility=float(cache.utility_u[int(slot_i)].item()),
                            selected_rank=int(selected_rank),
                        ),
                        slot=int(slot_i),
                    )
                )
            # Clamp query_event_id to 56 bits before the 8-bit shift so the
            # packed replay_id fits in u64. The high 8 bits of query_event_id
            # are the source_rank, which is at most 8 bits anyway, so the clamp
            # is lossless in practice. DuckDB BIGINT silently truncates the
            # 72-bit overflow today; the clamp makes the Python side honest
            # about what survives the storage round trip. The full
            # rank-prefixed query_event_id is preserved on the same tag dict
            # as a separate field, so no information is actually lost.
            replay_id = (
                (int(query_event_id) & ((1 << 56) - 1)) << 8
            ) | (int(selected_rank) & 0xFF)
            tag = {
                "step": producer_step,
                "slot": int(slot_i),
                "score": float(score_i),
                "selected_at": int(selected_at),
                "replay_id": replay_id,
                "query_event_id": int(query_event_id),
                "source_write_id": int(
                    cache.source_write_id[int(slot_i)].item()
                    if hasattr(cache, "source_write_id")
                    else cand.get("source_write_id", -1)
                ),
                "selected_rank": int(selected_rank),
                "teacher_score": float(score_i),
                "controller_logit": float(controller_logit),
                "selection_step": int(selected_at),
                "policy_version": int(policy_version),
                "outcome_status": "pending",
            }
            tag.update(controller_snapshot)
            tagged_replay_queue.append(tag)
            if action_recorder is not None and controller_snapshot:
                _record_controller_action_snapshot(
                    action_recorder=action_recorder,
                    tag=tag,
                )
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
        return cache.pressure_at_write.to(device)[slots]
    raise ValueError(f"unknown score_mode={score_mode!r}")


def _query_event_id_for_candidate(cand: dict[str, Any]) -> int:
    """Return a durable rank-prefixed id for a controller query event.

    The CPU SSM design gives QUERY_EVENT a rank-prefixed monotonic id. The
    current in-process Python queue predates that schema, so legacy rows only
    carry ``rank`` + ``k``. Prefer explicit ids when present and synthesize the
    old shape only as a compatibility bridge.
    """
    for key in ("query_event_id", "candidate_id", "event_id"):
        if key in cand:
            return int(cand[key])
    source_rank = int(cand.get("source_rank", cand.get("rank", 0)))
    rank_seq = int(cand.get("rank_seq", cand.get("k", 0)))
    return (source_rank << 56) | rank_seq


def _controller_feature_vector(
    *,
    heuristic_score: float,
    pressure: float,
    utility: float,
    selected_rank: int,
) -> torch.Tensor:
    """Compact V1 bridge features for the learned controller runtime."""
    return torch.tensor(
        [
            float(heuristic_score),
            float(pressure),
            float(utility),
            float(selected_rank),
        ],
        dtype=torch.float32,
    )


def _score_runtime_with_optional_snapshot(
    *,
    controller_runtime: Any,
    features: torch.Tensor,
    slot: int,
) -> tuple[float, dict[str, Any]]:
    """Score a slot and, when supported, serialize C10 replay snapshots."""
    scorer = getattr(controller_runtime, "score_slot_with_snapshot", None)
    if scorer is None:
        logit = controller_runtime.score_slot(features, slot=slot)
        return float(logit.item()), {}

    decision = scorer(features, slot=slot)
    return (
        float(decision.logit.item()),
        {
            "controller_features": _tensor_to_float_list(decision.features),
            "controller_global_state": _tensor_to_float_list(
                decision.global_state,
            ),
            "controller_slot_state": _tensor_to_float_list(decision.slot_state),
        },
    )


def _tensor_to_float_list(value: torch.Tensor) -> list[float]:
    return [
        float(x)
        for x in value.detach().to(device="cpu", dtype=torch.float32).tolist()
    ]


def _record_controller_action_snapshot(
    *,
    action_recorder: Any,
    tag: dict[str, Any],
) -> None:
    action_recorder.record_replay_selection(
        slot_id=int(tag["slot"]),
        gpu_step=int(tag["selection_step"]),
        policy_version=int(tag["policy_version"]),
        output_logit=float(tag["controller_logit"]),
        selected_rank=int(tag["selected_rank"]),
        features=list(tag["controller_features"]),
        global_state=list(tag["controller_global_state"]),
        slot_state=list(tag["controller_slot_state"]),
    )


def controller_main(
    *,
    controller_query_queue: list[dict[str, Any]],
    tagged_replay_queue: list[dict[str, Any]],
    cache: EpisodicCache,
    stop_event: threading.Event,
    queue_lock: threading.Lock | None = None,
    k: int = 16,
    score_mode: str = "cosine_utility_weighted",
    controller_runtime: Any | None = None,
    action_recorder: Any | None = None,
    cycle_idle_sleep_s: float = 0.005,
    heartbeat: list[int] | None = None,
) -> None:
    """Long-running controller loop. Designed for ``threading.Thread``.

    Polls ``controller_query_queue`` in a tight loop until
    ``stop_event`` is set. Each iteration calls ``run_controller_cycle``;
    if the cycle drained zero candidates, sleeps ``cycle_idle_sleep_s``
    seconds before checking again so the loop doesn't peg a CPU core.

    ``queue_lock`` defaults to ``None``: the runner's production spawn
    relies on GIL atomicity of ``list.append()`` (producer side) and
    ``list.pop(0)`` (controller side) to coordinate without a lock.
    Tests pass an explicit lock for deterministic synchronization with
    a producer thread.

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
            controller_runtime=controller_runtime,
            action_recorder=action_recorder,
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
