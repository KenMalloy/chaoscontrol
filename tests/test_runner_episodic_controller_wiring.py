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
from pathlib import Path

import pytest


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
