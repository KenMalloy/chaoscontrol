"""Tests for the Phase 2 controller wiring in the exp23 runner init.

The runner's init region must:

  - Add a ``tagged_replay_queue: list[dict]`` field to
    ``_EpisodicConsumerState`` (Y worktree's replay path reads from it).
  - On the episodic rank with ``episodic_enabled=True`` and
    ``controller_query_enabled=True``, spawn a daemon ``threading.Thread``
    running ``controller_main(...)`` to drain the query queue.
  - The thread is gated to ``finally``-safe shutdown: stop event is
    set on runner exit, thread joins within a small wall budget.

These tests pin the back-compat invariants without exercising the
per-step path (Y's territory).

Tests:

  1. ``test_runner_creates_tagged_replay_queue_on_episodic_rank`` —
     ``_attach_episodic_consumer`` returns a state with an empty
     ``tagged_replay_queue`` attribute on the episodic rank.
  2. ``test_runner_does_not_create_tagged_replay_queue_on_train_rank`` —
     train-rank invocation also returns the empty queue (so the
     no-op path is uniform across ranks). Per the task spec, the
     attribute exists thanks to ``__slots__``; what differs is whether
     the queue ever fills (it doesn't, because the controller thread
     only spawns on the episodic rank).
  3. ``test_runner_does_not_create_tagged_replay_queue_when_disabled`` —
     ``episodic_enabled=False`` returns the no-op shape.
  4. ``test_consumer_state_has_tagged_replay_queue_field`` — Pass C-
     style shape pin: any code reaching for the old field name surfaces
     immediately.
  5. ``test_consumer_state_back_compat_with_disabled_episodic`` — the
     bit-identical default still produces an empty cache + empty
     queues + heartbeat at zero when episodic is off.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


def _load_runner_module():
    path = (
        Path(__file__).resolve().parent.parent
        / "experiments" / "23_fast_path" / "runner_fast_path.py"
    )
    spec = importlib.util.spec_from_file_location("runner_fast_path", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_creates_tagged_replay_queue_on_episodic_rank():
    """Episodic rank with episodic_enabled=True gets an empty
    tagged_replay_queue list ready for the Y-side replay drain."""
    mod = _load_runner_module()
    consumer = mod._attach_episodic_consumer(
        episodic_enabled=True,
        is_episodic_rank=True,
        world_size=4,
        config={
            "episodic_capacity": 32,
            "episodic_span_length": 4,
            "episodic_key_rep_dim": 16,
            "episodic_grace_steps": 50,
            "episodic_utility_ema_decay": 0.95,
        },
        model_dim=16,
        all_group=None,
    )
    assert hasattr(consumer, "tagged_replay_queue")
    assert consumer.tagged_replay_queue == []


def test_runner_does_not_create_tagged_replay_queue_on_train_rank():
    """Train-rank invocation: the attribute exists (uniform __slots__)
    but the queue stays empty for the lifetime of the run, since the
    controller thread only spawns on the episodic rank."""
    mod = _load_runner_module()
    consumer = mod._attach_episodic_consumer(
        episodic_enabled=True,
        is_episodic_rank=False,
        world_size=4,
        config={"episodic_capacity": 16, "episodic_span_length": 4},
        model_dim=16,
        all_group=None,
    )
    # Pass C: no cache or queues populated on train ranks.
    assert consumer.cache is None
    assert consumer.controller_query_queue == []
    # New: tagged_replay_queue must also exist + be empty on train rank.
    assert hasattr(consumer, "tagged_replay_queue")
    assert consumer.tagged_replay_queue == []


def test_runner_does_not_create_tagged_replay_queue_when_disabled():
    """episodic_enabled=False is a no-op: no queues fill, but the
    attribute is uniform-present for downstream code that may always
    reach for it (telemetry, diagnostics)."""
    mod = _load_runner_module()
    for is_epr in (False, True):
        consumer = mod._attach_episodic_consumer(
            episodic_enabled=False,
            is_episodic_rank=is_epr,
            world_size=4,
            config={},
            model_dim=16,
            all_group=None,
        )
        assert consumer.cache is None
        assert consumer.controller_query_queue == []
        assert hasattr(consumer, "tagged_replay_queue")
        assert consumer.tagged_replay_queue == []


def test_consumer_state_has_tagged_replay_queue_field():
    """Shape pin: ``_EpisodicConsumerState`` exposes
    ``tagged_replay_queue`` as a documented field. Mirrors the Pass C
    test_consumer_state_no_longer_has_write_rings shape pin so a future
    rename surfaces here loudly.
    """
    mod = _load_runner_module()
    consumer = mod._attach_episodic_consumer(
        episodic_enabled=False,
        is_episodic_rank=False,
        world_size=2,
        config={},
        model_dim=4,
        all_group=None,
    )
    # Must have the new attribute alongside the existing ones.
    assert hasattr(consumer, "tagged_replay_queue")
    assert hasattr(consumer, "controller_query_queue")
    assert hasattr(consumer, "cache")
    assert hasattr(consumer, "heartbeat")


def test_consumer_state_back_compat_with_disabled_episodic():
    """When episodic is disabled, the consumer state should be
    bit-identically empty: no cache, both queues empty lists, heartbeat
    at zero. The Phase 1 cells with episodic_enabled=False must keep
    seeing the exact same no-op shape."""
    mod = _load_runner_module()
    consumer = mod._attach_episodic_consumer(
        episodic_enabled=False,
        is_episodic_rank=False,
        world_size=2,
        config={},
        model_dim=4,
        all_group=None,
    )
    assert consumer.cache is None
    assert consumer.heartbeat == [0]
    assert consumer.controller_query_queue == []
    assert consumer.tagged_replay_queue == []


def test_online_learning_bridge_updates_scoring_runtime_after_sgd():
    mod = _load_runner_module()
    from chaoscontrol.episodic.cpu_ssm_controller import (
        CpuSsmControllerRuntime,
        CpuSsmControllerWeights,
    )

    runtime = CpuSsmControllerRuntime(
        CpuSsmControllerWeights(
            w_global_in=torch.tensor([[0.0]], dtype=torch.float32),
            w_slot_in=torch.tensor([[0.0]], dtype=torch.float32),
            decay_global=torch.tensor([1.0], dtype=torch.float32),
            decay_slot=torch.tensor([1.0], dtype=torch.float32),
            w_global_out=torch.tensor([1.0], dtype=torch.float32),
            w_slot_out=torch.tensor([0.0], dtype=torch.float32),
            bias=torch.tensor(0.0, dtype=torch.float32),
        ),
        capacity=2,
        prefer_cpp=False,
    )
    bridge = mod._OnlineLearningRuntimeBridge(
        runtime=runtime,
        capacity=2,
        config={
            "episodic_controller_learning_rate": 0.1,
            "episodic_controller_sgd_interval": 1,
            "episodic_controller_ema_interval": 999,
            "episodic_controller_credit_gamma": 1.0,
            "episodic_controller_gerber_c": 0.0,
        },
    )
    bridge.record_replay_selection(
        slot_id=1,
        gpu_step=90,
        policy_version=7,
        output_logit=1.0,
        selected_rank=0,
        features=[2.0],
        global_state=[3.0],
        slot_state=[0.0],
    )

    bridge.on_replay_outcome({
        "event_type": 3,
        "selected_rank": 0,
        "outcome_status": 0,
        "replay_id": 1,
        "gpu_step": 120,
        "query_event_id": 123,
        "source_write_id": 456,
        "slot_id": 1,
        "policy_version": 7,
        "selection_step": 110,
        "teacher_score": 0.5,
        "controller_logit": 1.0,
        "ce_before_replay": 4.0,
        "ce_after_replay": 3.0,
        "ce_delta_raw": 1.0,
        "bucket_baseline": 0.0,
        "reward_shaped": 1.0,
        "grad_cos_rare": math.nan,
        "grad_cos_total": math.nan,
        "flags": 0,
    })

    assert runtime.weights.w_global_in.reshape(-1).tolist() == pytest.approx([0.2])
    assert runtime.weights.decay_global.tolist() == pytest.approx([1.3])
    assert runtime.weights.w_global_out.tolist() == pytest.approx([1.3])
    assert float(runtime.weights.bias.item()) == pytest.approx(0.1)


def test_trainer_cache_checkpoint_preserves_configured_fingerprint_window(
    tmp_path,
):
    """The trainer-built cache must save the writer's configured W.

    C1 flagged a silent-miss hazard where the writer could hash with a
    non-default ``episodic_fingerprint_window`` while the saved cache carried
    the constructor default W=8. Drive the trainer's current cache-construction
    surface and round-trip the checkpoint payload so that mismatch fails here.
    """
    mod = _load_runner_module()
    consumer = mod._attach_episodic_consumer(
        episodic_enabled=True,
        is_episodic_rank=True,
        world_size=2,
        config={
            "episodic_capacity": 16,
            "episodic_span_length": 4,
            "episodic_key_rep_dim": 8,
            "episodic_fingerprint_window": 16,
        },
        model_dim=8,
        all_group=None,
    )
    assert consumer.cache is not None

    ckpt_path = tmp_path / "trainer_cache.pt"
    torch.save({"episodic_cache": consumer.cache.to_dict()}, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    loaded_cache = mod.EpisodicCache.from_dict(ckpt["episodic_cache"])

    assert loaded_cache.fingerprint_window == 16


def _make_disabled_consumer(mod, *, with_cache: bool = True):
    """Build a consumer state where every gate-enabling precondition
    is met EXCEPT one. Each test below flips a single gate to False to
    pin that the spawn returns None for that flag specifically.
    """
    consumer = mod._attach_episodic_consumer(
        episodic_enabled=True,
        is_episodic_rank=True,
        world_size=2,
        config={
            "episodic_capacity": 16,
            "episodic_span_length": 2,
            "episodic_key_rep_dim": 4,
            "controller_query_enabled": True,
        },
        model_dim=4,
        all_group=None,
    )
    if not with_cache:
        consumer.cache = None
    return consumer


def test_spawn_returns_none_when_episodic_disabled():
    """Gate cascade #1: episodic_enabled=False short-circuits the spawn."""
    mod = _load_runner_module()
    consumer = _make_disabled_consumer(mod)
    handle = mod._spawn_episodic_controller(
        consumer=consumer,
        is_episodic_rank=True,
        episodic_enabled=False,
        config={},
    )
    assert handle is None


def test_spawn_returns_none_when_not_episodic_rank():
    """Gate cascade #2: train ranks never spawn the controller."""
    mod = _load_runner_module()
    consumer = _make_disabled_consumer(mod)
    handle = mod._spawn_episodic_controller(
        consumer=consumer,
        is_episodic_rank=False,
        episodic_enabled=True,
        config={},
    )
    assert handle is None


def test_spawn_returns_none_when_cache_missing():
    """Gate cascade #3: no cache → no controller (the consumer
    state's no-op shape on a misconfigured episodic rank)."""
    mod = _load_runner_module()
    consumer = _make_disabled_consumer(mod, with_cache=False)
    handle = mod._spawn_episodic_controller(
        consumer=consumer,
        is_episodic_rank=True,
        episodic_enabled=True,
        config={},
    )
    assert handle is None


def test_spawn_returns_none_when_controller_query_disabled():
    """Gate cascade #4: controller_query_enabled=False (Pass C default)
    silently disables both the drain and the spawn. Single flag, double
    gate — verified end-to-end."""
    mod = _load_runner_module()
    # Override the consumer to flip controller_query_enabled off.
    consumer = mod._attach_episodic_consumer(
        episodic_enabled=True,
        is_episodic_rank=True,
        world_size=2,
        config={
            "episodic_capacity": 16,
            "episodic_span_length": 2,
            "episodic_key_rep_dim": 4,
            "controller_query_enabled": False,
        },
        model_dim=4,
        all_group=None,
    )
    handle = mod._spawn_episodic_controller(
        consumer=consumer,
        is_episodic_rank=True,
        episodic_enabled=True,
        config={},
    )
    assert handle is None


def test_controller_score_mode_accepts_exp24_alias_and_rejects_conflict():
    mod = _load_runner_module()

    assert mod._controller_score_mode_from_config({
        "controller_query_mode": "pressure_only",
    }) == "pressure_only"
    assert mod._controller_score_mode_from_config({
        "episodic_controller_score_mode": "cosine_utility_weighted",
        "controller_query_mode": "cosine_utility_weighted",
    }) == "cosine_utility_weighted"
    with pytest.raises(ValueError, match="conflicting controller score mode"):
        mod._controller_score_mode_from_config({
            "episodic_controller_score_mode": "cosine_utility_weighted",
            "controller_query_mode": "pressure_only",
        })


def _make_trained_runtime():
    """Build a minimal CpuSsmControllerRuntime that the bridge can
    initialize from. The Python reference path (prefer_cpp=False) is
    sufficient — _wire_online_learning_bridge doesn't dispatch on the
    backend, just on whether the runtime is non-None and the config
    flag.
    """
    from chaoscontrol.episodic.cpu_ssm_controller import (
        CpuSsmControllerRuntime,
        CpuSsmControllerWeights,
    )
    return CpuSsmControllerRuntime(
        CpuSsmControllerWeights(
            w_global_in=torch.tensor([[0.0]], dtype=torch.float32),
            w_slot_in=torch.tensor([[0.0]], dtype=torch.float32),
            decay_global=torch.tensor([1.0], dtype=torch.float32),
            decay_slot=torch.tensor([1.0], dtype=torch.float32),
            w_global_out=torch.tensor([1.0], dtype=torch.float32),
            w_slot_out=torch.tensor([0.0], dtype=torch.float32),
            bias=torch.tensor(0.0, dtype=torch.float32),
        ),
        capacity=2,
        prefer_cpp=False,
    )


def test_wire_bridge_short_circuits_when_runtime_is_none():
    """Heuristic mode: no trained runtime → no bridge to wrap. Returns
    (None, None) and leaves consumer.online_learning_bridge unset."""
    mod = _load_runner_module()
    consumer = _make_disabled_consumer(mod)
    consumer.online_learning_bridge = None
    bridge, runtime_for_thread = mod._wire_online_learning_bridge(
        consumer=consumer,
        controller_runtime=None,
        config={"controller_train_online": True},
    )
    assert bridge is None
    assert runtime_for_thread is None
    assert consumer.online_learning_bridge is None


def test_wire_bridge_wraps_runtime_when_train_online_true():
    """controller_train_online=True (default): the runtime is wrapped in
    _OnlineLearningRuntimeBridge and the consumer points at it. Replay
    outcomes will route through the C++ online-learning path."""
    mod = _load_runner_module()
    consumer = _make_disabled_consumer(mod)
    consumer.online_learning_bridge = None
    runtime = _make_trained_runtime()
    bridge, runtime_for_thread = mod._wire_online_learning_bridge(
        consumer=consumer,
        controller_runtime=runtime,
        config={"controller_train_online": True},
    )
    assert isinstance(bridge, mod._OnlineLearningRuntimeBridge)
    assert runtime_for_thread is bridge
    assert consumer.online_learning_bridge is bridge


def test_wire_bridge_skips_wrap_when_train_online_false():
    """controller_train_online=False (F1 arm_c): the runtime is used
    raw, no bridge constructed, consumer.online_learning_bridge stays
    unset so _notify_online_learning_bridge no-ops on every replay."""
    mod = _load_runner_module()
    consumer = _make_disabled_consumer(mod)
    consumer.online_learning_bridge = None
    runtime = _make_trained_runtime()
    bridge, runtime_for_thread = mod._wire_online_learning_bridge(
        consumer=consumer,
        controller_runtime=runtime,
        config={"controller_train_online": False},
    )
    assert bridge is None
    assert runtime_for_thread is runtime
    assert consumer.online_learning_bridge is None


class _StubQueryRing:
    """Captures pushed dicts so the producer-side wire payload can be
    asserted in-process without a real shared-memory ring. Mirrors the
    push()/size() surface that ``_emit_query_event`` reaches for."""

    def __init__(self) -> None:
        self.pushed: list[dict] = []

    def push(self, event: dict) -> bool:
        self.pushed.append(dict(event))
        return True


def _consumer_with_stub_query_ring(mod):
    """Fresh _EpisodicConsumerState with a stub query ring + zeroed seq
    counter so ``_emit_query_event`` returns a deterministic query_id
    on the first push."""
    consumer = mod._EpisodicConsumerState(
        cache=None,
        heartbeat=[0],
        controller_query_queue=[],
        controller_query_enabled=False,
    )
    consumer.query_ring = _StubQueryRing()
    consumer.rank_query_seq = {}
    return consumer


def test_runner_emit_query_event_threads_simplex_candidates():
    """Phase S3 producer pin: when ``_emit_query_event`` is called with
    a populated 16-id candidate list, the emitted wire-event dict carries
    those slot ids in order (with cosines following the same shape).
    The actual top-K retrieval that produces the candidate set lives in
    Phase S5; S3 only verifies the producer threads the list through to
    the wire payload unchanged."""
    mod = _load_runner_module()
    consumer = _consumer_with_stub_query_ring(mod)
    residual = torch.zeros(16, dtype=torch.float32)
    candidate_slot_ids = [10 * (i + 1) for i in range(16)]   # 10..160
    candidate_cosines = [0.9 - 0.05 * i for i in range(16)]  # 0.90..0.15

    qid = mod._emit_query_event(
        consumer=consumer,
        source_rank=0,
        gpu_step=42,
        query_residual=residual,
        pressure=0.5,
        pre_query_ce=2.0,
        bucket=1,
        candidate_slot_ids=candidate_slot_ids,
        candidate_cosines=candidate_cosines,
    )
    assert qid is not None
    assert len(consumer.query_ring.pushed) == 1
    payload = consumer.query_ring.pushed[0]
    assert payload["candidate_slot_ids"] == candidate_slot_ids
    assert payload["candidate_cosines"] == pytest.approx(candidate_cosines)


def test_runner_emit_query_event_sentinel_pads_when_candidates_omitted():
    """V0 heuristic-only path: caller doesn't pass candidates → producer
    sentinel-pads both arrays so the wire payload is always 16-wide. The
    C++ controller reads ``candidate_slot_ids[0] == UINT64_MAX`` as the
    fallback signal."""
    mod = _load_runner_module()
    consumer = _consumer_with_stub_query_ring(mod)
    residual = torch.zeros(16, dtype=torch.float32)

    mod._emit_query_event(
        consumer=consumer,
        source_rank=0,
        gpu_step=7,
        query_residual=residual,
        pressure=0.25,
        pre_query_ce=1.0,
        bucket=2,
    )
    assert len(consumer.query_ring.pushed) == 1
    payload = consumer.query_ring.pushed[0]
    sentinel = (1 << 64) - 1
    assert payload["candidate_slot_ids"] == [sentinel] * 16
    assert payload["candidate_cosines"] == [0.0] * 16


def test_runner_emit_query_event_sentinel_pads_short_candidate_list():
    """Producer accepts a sub-16 candidate list (heuristic returned
    fewer hits) and sentinel-pads the trailing slots so the wire
    payload is always 16-wide."""
    mod = _load_runner_module()
    consumer = _consumer_with_stub_query_ring(mod)
    residual = torch.zeros(16, dtype=torch.float32)
    short_ids = [11, 22, 33]
    short_cosines = [0.9, 0.8, 0.7]

    mod._emit_query_event(
        consumer=consumer,
        source_rank=0,
        gpu_step=11,
        query_residual=residual,
        pressure=0.5,
        pre_query_ce=2.0,
        bucket=1,
        candidate_slot_ids=short_ids,
        candidate_cosines=short_cosines,
    )
    payload = consumer.query_ring.pushed[0]
    sentinel = (1 << 64) - 1
    assert payload["candidate_slot_ids"][:3] == short_ids
    assert payload["candidate_slot_ids"][3:] == [sentinel] * 13
    assert payload["candidate_cosines"][:3] == pytest.approx(short_cosines)
    assert payload["candidate_cosines"][3:] == [0.0] * 13


def test_runner_emit_query_event_rejects_oversized_candidate_list():
    """Caller passing >16 candidates is a programmer error, not a
    silent truncation. The producer raises ValueError before it
    constructs a payload."""
    mod = _load_runner_module()
    consumer = _consumer_with_stub_query_ring(mod)
    residual = torch.zeros(16, dtype=torch.float32)

    with pytest.raises(ValueError, match="exceeds simplex capacity"):
        mod._emit_query_event(
            consumer=consumer,
            source_rank=0,
            gpu_step=11,
            query_residual=residual,
            pressure=0.5,
            pre_query_ce=2.0,
            bucket=1,
            candidate_slot_ids=[i for i in range(17)],
        )
    assert consumer.query_ring.pushed == []


def test_wire_bridge_defaults_to_train_online_true_when_flag_absent():
    """Backwards compat: existing configs that don't set
    controller_train_online keep their pre-fix behavior (bridge wraps).
    Only F1's frozen arm explicitly opts out."""
    mod = _load_runner_module()
    consumer = _make_disabled_consumer(mod)
    consumer.online_learning_bridge = None
    runtime = _make_trained_runtime()
    bridge, runtime_for_thread = mod._wire_online_learning_bridge(
        consumer=consumer,
        controller_runtime=runtime,
        config={},
    )
    assert isinstance(bridge, mod._OnlineLearningRuntimeBridge)
    assert runtime_for_thread is bridge
    assert consumer.online_learning_bridge is bridge
