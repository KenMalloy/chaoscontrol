"""REPLAY_OUTCOME producer in runner_fast_path (Phase B4)."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from chaoscontrol.kernels import _cpu_ssm_controller as _ext
from chaoscontrol.optim.episodic_cache import EpisodicCache

REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"

REPLAY_OUTCOME_KEYS = (
    "event_type",
    "selected_rank",
    "outcome_status",
    "replay_id",
    "gpu_step",
    "query_event_id",
    "source_write_id",
    "slot_id",
    "policy_version",
    "selection_step",
    "teacher_score",
    "controller_logit",
    "ce_before_replay",
    "ce_after_replay",
    "ce_delta_raw",
    "bucket_baseline",
    "reward_shaped",
    "grad_cos_rare",
    "grad_cos_total",
    "flags",
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("runner_b3", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _TinyTokenModel(nn.Module):
    def __init__(self, vocab: int = 16, dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def encode(
        self,
        inputs: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.embed(inputs.to(torch.long))
        if return_final_states:
            return h, []
        return h


def _consumer(mod, *, event_log_enabled: bool, cache: EpisodicCache):
    return mod._EpisodicConsumerState(
        cache=cache,
        heartbeat=[0],
        controller_query_queue=[],
        episodic_event_log_enabled=event_log_enabled,
    )


def _append_slot(
    cache: EpisodicCache,
    *,
    key_fp: int,
    write_bucket: int,
    source_write_id: int,
    span_tokens: list[int] | None = None,
) -> int:
    return cache.append(
        key_fp=key_fp,
        key_rep=torch.zeros(cache.key_rep_dim, dtype=torch.float32),
        value_tok_ids=torch.tensor(
            span_tokens or [3, 5, 7, 9], dtype=torch.int64,
        ),
        value_anchor_id=2,
        current_step=10,
        embedding_version=0,
        pressure_at_write=1.25,
        source_write_id=source_write_id,
        write_bucket=write_bucket,
    )


def _run_replay_drain(mod, *, consumer, model) -> int:
    return mod._run_episodic_replay_from_tagged_queue(
        consumer=consumer,
        model=model,
        current_step=17,
        weight=1.0,
        lm_head_backward_mode="single",
        lm_head_tile_size=1024,
        logger=None,
        max_replays_per_step=0,
    )


def test_replay_outcome_emitted_with_wire_schema_and_reward_fields():
    """Two tagged replays produce two REPLAY_OUTCOME records in wire order.

    With ``compute_replay_ce_pair`` OFF (B3 default), the reward fields stay
    NaN — the upstream-supplied ``entry["ce_before_replay"]`` is preserved
    in the dict but ``ce_after_replay`` is NaN until the post-step pass owns
    the pair (Phase B5, exercised in ``tests/test_replay_ce_pair.py``).
    """
    mod = _load_runner()
    torch.manual_seed(7)
    model = _TinyTokenModel()
    cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
    slot_a = _append_slot(
        cache, key_fp=42, write_bucket=2, source_write_id=1200,
        span_tokens=[3, 5, 7, 9],
    )
    slot_b = _append_slot(
        cache, key_fp=99, write_bucket=2, source_write_id=1201,
        span_tokens=[1, 2, 3, 4],
    )
    consumer = _consumer(mod, event_log_enabled=True, cache=cache)
    consumer.tagged_replay_queue.extend([
        {
            "slot": int(slot_a),
            "replay_id": 700,
            "query_event_id": 600,
            "source_write_id": 1200,
            "selection_step": 16,
            "policy_version": 3,
            "selected_rank": 0,
            "teacher_score": 0.75,
            "controller_logit": 0.25,
            "ce_before_replay": 4.0,
        },
        {
            "slot": int(slot_b),
            "replay_id": 701,
            "query_event_id": 601,
            "source_write_id": 1201,
            "selection_step": 16,
            "policy_version": 3,
            "selected_rank": 1,
            "teacher_score": 0.5,
            "controller_logit": 0.125,
            "ce_before_replay": 5.0,
        },
    ])

    replayed = _run_replay_drain(mod, consumer=consumer, model=model)

    try:
        assert replayed == 2
        assert consumer.replay_ring_name is not None
        ring = _ext.ShmRingReplayOutcome.attach(consumer.replay_ring_name)
        first = ring.pop()
        second = ring.pop()
        assert ring.pop() is None
    finally:
        mod._cleanup_episodic_event_rings(consumer)

    assert first is not None
    assert second is not None
    assert consumer.replay_ring_drops == 0
    assert tuple(first.keys()) == REPLAY_OUTCOME_KEYS
    assert tuple(second.keys()) == REPLAY_OUTCOME_KEYS

    assert first["event_type"] == 3
    assert first["replay_id"] == 700
    assert first["gpu_step"] == 17
    assert first["query_event_id"] == 600
    assert first["source_write_id"] == 1200
    assert first["slot_id"] == slot_a
    assert first["selection_step"] == 16
    assert first["policy_version"] == 3
    assert first["selected_rank"] == 0
    assert first["teacher_score"] == pytest.approx(0.75)
    assert first["controller_logit"] == pytest.approx(0.25)
    # B3 default: entry's ``ce_before_replay`` survives, ``ce_after_replay``
    # stays NaN until the B5 post-step pair pass lands a real value.
    assert first["ce_before_replay"] == pytest.approx(4.0)
    assert math.isnan(first["ce_after_replay"])
    assert math.isnan(first["ce_delta_raw"])
    assert math.isnan(first["reward_shaped"])
    assert math.isnan(first["grad_cos_rare"])
    assert math.isnan(first["grad_cos_total"])
    assert first["outcome_status"] == 0
    assert first["flags"] == 0

    # NaN ``ce_delta_raw`` must not poison the EMA — the producer skips
    # the EMA update on non-finite deltas, so the bucket baseline stays
    # at its zero-init value across both replays.
    assert math.isnan(second["ce_after_replay"])
    assert math.isnan(second["ce_delta_raw"])
    assert math.isnan(second["reward_shaped"])
    assert second["bucket_baseline"] == pytest.approx(0.0)
    assert consumer.bucket_baseline_ema is not None
    assert consumer.bucket_baseline_ema[2] == pytest.approx(0.0)
    # Flag off ⇒ no pending list allocated; the deferred pass is a no-op.
    assert consumer.compute_replay_ce_pair is False
    assert consumer.pending_post_step_replays is None


def test_replay_id_derives_from_query_event_id_when_controller_field_missing():
    """Fallback replay_id matches the current controller packing pattern."""
    mod = _load_runner()
    torch.manual_seed(11)
    model = _TinyTokenModel()
    cache = EpisodicCache(capacity=2, span_length=4, key_rep_dim=8)
    slot = _append_slot(cache, key_fp=7, write_bucket=1, source_write_id=88)
    consumer = _consumer(mod, event_log_enabled=True, cache=cache)
    consumer.tagged_replay_queue.append({
        "slot": int(slot),
        "query_event_id": 0x0200_0000_0000_0005,
        "selected_rank": 2,
        "ce_before_replay": 3.0,
    })

    _run_replay_drain(mod, consumer=consumer, model=model)

    try:
        assert consumer.replay_ring_name is not None
        ring = _ext.ShmRingReplayOutcome.attach(consumer.replay_ring_name)
        event = ring.pop()
        assert ring.pop() is None
    finally:
        mod._cleanup_episodic_event_rings(consumer)
    assert event is not None
    expected = ((0x0200_0000_0000_0005 & ((1 << 56) - 1)) << 8) | 2
    assert event["replay_id"] == expected
    assert event["selection_step"] == 17
    assert event["policy_version"] == 0
    assert event["controller_logit"] == pytest.approx(0.0)


def test_no_replay_outcome_state_when_disabled():
    """With episodic_event_log_enabled=False, no list or EMA is allocated."""
    mod = _load_runner()
    torch.manual_seed(13)
    model = _TinyTokenModel()
    cache = EpisodicCache(capacity=2, span_length=4, key_rep_dim=8)
    slot = _append_slot(cache, key_fp=42, write_bucket=0, source_write_id=91)
    consumer = _consumer(mod, event_log_enabled=False, cache=cache)
    assert consumer.replay_ring is None
    assert consumer.replay_ring_name is None
    assert consumer.bucket_baseline_ema is None
    consumer.tagged_replay_queue.append({
        "slot": int(slot),
        "query_event_id": 10,
        "selected_rank": 0,
        "ce_before_replay": 3.0,
    })

    replayed = _run_replay_drain(mod, consumer=consumer, model=model)

    assert replayed == 1
    assert consumer.replay_ring is None
    assert consumer.replay_ring_name is None
    assert consumer.bucket_baseline_ema is None


def test_slot_missing_outcome_is_recorded_without_replay():
    """Enabled event logging records slot_missing instead of silently vanishing."""
    mod = _load_runner()
    model = _TinyTokenModel()
    cache = EpisodicCache(capacity=2, span_length=4, key_rep_dim=8)
    consumer = _consumer(mod, event_log_enabled=True, cache=cache)
    consumer.tagged_replay_queue.append({
        "slot": 1,
        "query_event_id": 12,
        "selected_rank": 0,
        "source_write_id": 44,
    })

    replayed = _run_replay_drain(mod, consumer=consumer, model=model)

    try:
        assert replayed == 0
        assert consumer.replay_ring_name is not None
        ring = _ext.ShmRingReplayOutcome.attach(consumer.replay_ring_name)
        event = ring.pop()
        assert ring.pop() is None
    finally:
        mod._cleanup_episodic_event_rings(consumer)
    assert event is not None
    assert tuple(event.keys()) == REPLAY_OUTCOME_KEYS
    assert event["outcome_status"] == 1
    assert event["slot_id"] == 1
    assert math.isnan(event["ce_after_replay"])
    assert consumer.bucket_baseline_ema == [0.0, 0.0, 0.0, 0.0]
