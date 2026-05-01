"""WRITE_EVENT producer for the episodic writer path (Phase B4)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext
from chaoscontrol.optim.episodic_writer import (
    _reset_admission_trace_seq,
    select_writes,
)

REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"

WRITE_EVENT_KEYS = (
    "event_type",
    "source_rank",
    "write_bucket",
    "candidate_id",
    "gpu_step",
    "key_fp",
    "key_rep",
    "value_tok_ids",
    "value_anchor_id",
    "pressure_at_write",
    "pre_write_ce",
)


def _force_unlink(cls, name: str | None) -> None:
    if not name:
        return
    try:
        cls.unlink(name)
    except Exception:
        pass


def _fp16_bits(values: torch.Tensor) -> list[int]:
    return values.detach().cpu().to(torch.float16).view(torch.uint16).tolist()


def _load_runner():
    spec = importlib.util.spec_from_file_location("runner_b1", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _WriteAdmissionFakeSsm:
    def observe(self, features):
        return {
            "write_admission": (
                1000.0 if int(features.get("rank", -1)) == 1 else -1000.0
            ),
            "eviction": 0.0,
        }


class _ProtectionFakeSsm:
    def observe(self, features):
        return {"eviction": 10.0}


def _mixed_admit_reject_inputs():
    B, T, D = 1, 12, 4
    input_ids = torch.arange(B * T, dtype=torch.int64).reshape(B, T)
    target_ids = (input_ids + 1) % 100
    key_rep = torch.randn(B, T, D, generator=torch.Generator().manual_seed(0))
    pressure = torch.zeros(B, T)
    ce = torch.zeros(B, T)
    # Two interior admits, followed by two boundary rejects in score order.
    pressure[0, 5] = 1.0
    ce[0, 5] = 9.0
    pressure[0, 6] = 1.0
    ce[0, 6] = 8.0
    pressure[0, 0] = 1.0
    ce[0, 0] = 7.0
    pressure[0, 11] = 1.0
    ce[0, 11] = 6.0
    return input_ids, target_ids, pressure, ce, key_rep


def test_select_writes_emits_write_events_for_admits_only(tmp_path):
    """Trace rows include rejects; WRITE_EVENT records are admit-only.

    The admitted WRITE_EVENT candidate_id must match the corresponding
    D1 trace row so offline traces and the future online ring can join
    on the same rank-prefixed id.
    """
    _reset_admission_trace_seq()
    inputs, targets, pressure, ce, rep = _mixed_admit_reject_inputs()
    trace_path = tmp_path / "admission.ndjson"
    ring_name = f"/cc_test_write_select_pid{os.getpid()}"
    _force_unlink(_ext.ShmRingWriteEvent, ring_name)
    ring = _ext.ShmRingWriteEvent.create(ring_name)
    drops = [0]

    try:
        payloads = select_writes(
            input_ids=inputs,
            target_ids=targets,
            pressure=pressure,
            per_token_ce=ce,
            key_rep_per_position=rep,
            top_p=4.0 / 12.0,
            fingerprint_window=4,
            span_length=4,
            source_rank=2,
            gpu_step=137,
            write_bucket=1,
            admission_trace_path=str(trace_path),
            write_ring=ring,
            write_ring_drops=drops,
        )

        assert len(payloads) == 2
        write_events = [ring.pop(), ring.pop()]
        assert ring.pop() is None
        assert drops == [0]
    finally:
        _force_unlink(_ext.ShmRingWriteEvent, ring_name)

    assert {p.position for p in payloads} == {5, 6}

    trace_rows = [
        json.loads(line) for line in trace_path.read_text().splitlines()
    ]
    admitted_trace_ids = [
        int(row["candidate_id"]) for row in trace_rows if row["decision"] == 1
    ]
    assert [event["candidate_id"] for event in write_events] == admitted_trace_ids

    for event in write_events:
        assert event is not None
        assert tuple(event.keys()) == WRITE_EVENT_KEYS
        assert event["event_type"] == 1
        assert event["source_rank"] == 2
        assert event["gpu_step"] == 137
        assert event["write_bucket"] == 1
        assert (event["candidate_id"] >> 56) == 2
        assert len(event["key_rep"]) == 256
        assert len(event["value_tok_ids"]) == 4
        assert event["pressure_at_write"] == pytest.approx(1.0)
    assert [event["pre_write_ce"] for event in write_events] == pytest.approx(
        [9.0, 8.0]
    )
    assert write_events[0]["key_rep"][:4] == _fp16_bits(rep[0, 5])
    assert write_events[1]["key_rep"][:4] == _fp16_bits(rep[0, 6])


def test_select_writes_without_write_ring_is_side_effect_free(tmp_path):
    """Back-compat: None means no ring push and unchanged payloads."""
    _reset_admission_trace_seq()
    inputs, targets, pressure, ce, rep = _mixed_admit_reject_inputs()

    payloads_off = select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=4.0 / 12.0,
        fingerprint_window=4,
        span_length=4,
    )
    _reset_admission_trace_seq()
    payloads_trace_only = select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=4.0 / 12.0,
        fingerprint_window=4,
        span_length=4,
        admission_trace_path=str(tmp_path / "trace.ndjson"),
    )

    assert len(payloads_off) == len(payloads_trace_only)
    for a, b in zip(payloads_off, payloads_trace_only):
        assert a.batch_index == b.batch_index
        assert a.position == b.position
        assert a.key_fp == b.key_fp
        assert torch.equal(a.key_rep, b.key_rep)
        assert torch.equal(a.value_tok_ids, b.value_tok_ids)


def test_runner_emit_allocates_write_ring_only_on_train_rank_when_enabled():
    """B4's write ring lives on the train-rank emit handle."""
    mod = _load_runner()
    ring_id = f"unit_write_alloc_{os.getpid()}"
    base_config = {
        "episodic_enabled": True,
        "episodic_span_length": 2,
        "episodic_fingerprint_window": 1,
        "episodic_key_rep_dim": 4,
        "episodic_k_max": 4,
        "episodic_event_ring_id": ring_id,
        "model_dim": 4,
    }

    disabled = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config={**base_config, "episodic_async_write_rings_enabled": False},
    )
    assert disabled is not None
    assert disabled.write_ring is None
    assert disabled.write_ring_name is None

    enabled_train = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config={**base_config, "episodic_event_log_enabled": True},
    )
    assert enabled_train is not None
    try:
        assert enabled_train.write_ring is not None
        assert enabled_train.write_ring_name.startswith("/cc_e_w_")
        assert enabled_train.write_ring_name.endswith("_r0")
        assert enabled_train.write_ring_drops == 0
        attached = _ext.ShmRingWriteEvent.attach(enabled_train.write_ring_name)
        assert attached.pop() is None
    finally:
        mod._cleanup_episodic_event_rings(enabled_train)

    enabled_episodic = mod._create_episodic_emit(
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        config={**base_config, "episodic_event_log_enabled": True},
    )
    assert enabled_episodic is not None
    assert enabled_episodic.write_ring is None
    assert enabled_episodic.write_ring_name is None


def test_runner_emit_pushes_write_event_for_packed_slots():
    """The current runner producer packs directly; it still emits B1 records."""
    _reset_admission_trace_seq()
    mod = _load_runner()
    ring_id = f"unit_write_push_{os.getpid()}"
    handle = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config={
            "episodic_enabled": True,
            "episodic_span_length": 2,
            "episodic_fingerprint_window": 1,
            "episodic_key_rep_dim": 4,
            "episodic_top_p": 0.5,
            "episodic_k_max": 4,
            "episodic_event_log_enabled": True,
            "episodic_event_ring_id": ring_id,
            "model_dim": 4,
        },
    )
    assert handle is not None
    assert handle.write_ring is not None
    assert handle.write_ring_name is not None
    try:
        inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
        targets = torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.int64)
        pressure = torch.ones_like(targets, dtype=torch.float32)
        ce = torch.tensor([[0.1, 9.0, 8.0, 0.2, 0.3]], dtype=torch.float32)
        hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)

        mod._emit_episodic_payloads_gpu(
            emit=handle,
            inputs=inputs,
            targets=targets,
            pressure=pressure,
            per_token_ce=ce,
            hidden=hidden,
            rank=3,
            world_size=4,
            all_group=None,
            current_step=19,
            write_bucket=2,
        )

        attached = _ext.ShmRingWriteEvent.attach(handle.write_ring_name)
        events = [attached.pop(), attached.pop()]
        assert attached.pop() is None
        assert handle.write_ring_drops == 0
    finally:
        mod._cleanup_episodic_event_rings(handle)

    ids = [event["candidate_id"] for event in events]
    assert [qid >> 56 for qid in ids] == [3, 3]
    assert [qid & ((1 << 56) - 1) for qid in ids] == [0, 1]
    for event in events:
        assert event is not None
        assert tuple(event.keys()) == WRITE_EVENT_KEYS
        assert event["event_type"] == 1
        assert event["gpu_step"] == 19
        assert event["source_rank"] == 3
        assert event["write_bucket"] == 2
        assert len(event["key_rep"]) == 256
        assert len(event["value_tok_ids"]) == 4
    assert events[0]["value_tok_ids"][:2] == [3, 4]
    assert events[1]["value_tok_ids"][:2] == [4, 5]


def test_runner_emit_is_publish_only_even_when_all_group_is_present(monkeypatch):
    """WRITE_EVENT production must not call ``dist.gather`` anymore.

    This is the trunk-throughput invariant: memory writes are local ring pushes
    with backpressure-by-drop, never a synchronous train-step collective.
    """
    _reset_admission_trace_seq()
    mod = _load_runner()
    ring_id = f"unit_no_gather_{os.getpid()}"
    handle = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config={
            "episodic_enabled": True,
            "episodic_span_length": 2,
            "episodic_fingerprint_window": 1,
            "episodic_key_rep_dim": 4,
            "episodic_top_p": 0.5,
            "episodic_k_max": 4,
            "episodic_event_ring_id": ring_id,
            "model_dim": 4,
        },
    )
    assert handle is not None
    assert handle.write_ring is not None

    def _boom(*_args, **_kwargs):
        raise AssertionError("dist.gather must not be on the write path")

    monkeypatch.setattr(mod.dist, "gather", _boom)
    try:
        inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
        targets = torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.int64)
        pressure = torch.ones_like(targets, dtype=torch.float32)
        ce = torch.tensor([[0.1, 9.0, 8.0, 0.2, 0.3]], dtype=torch.float32)
        hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
        mod._emit_episodic_payloads_gpu(
            emit=handle,
            inputs=inputs,
            targets=targets,
            pressure=pressure,
            per_token_ce=ce,
            hidden=hidden,
            rank=0,
            world_size=2,
            all_group=object(),
            current_step=7,
            write_bucket=1,
        )
        attached = _ext.ShmRingWriteEvent.attach(handle.write_ring_name)
        assert attached.pop() is not None
    finally:
        mod._cleanup_episodic_event_rings(handle)


def test_async_write_ring_drain_populates_cache_and_query_queue():
    """Episodic rank lazily attaches train-rank write rings and drains them."""
    _reset_admission_trace_seq()
    mod = _load_runner()
    ring_id = f"unit_write_drain_{os.getpid()}"
    config = {
        "episodic_enabled": True,
        "episodic_span_length": 2,
        "episodic_fingerprint_window": 1,
        "episodic_key_rep_dim": 4,
        "episodic_top_p": 0.5,
        "episodic_k_max": 4,
        "episodic_event_ring_id": ring_id,
        "controller_query_enabled": True,
        "model_dim": 4,
    }
    producer = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config=config,
    )
    consumer = mod._attach_episodic_consumer(
        episodic_enabled=True,
        is_episodic_rank=True,
        world_size=2,
        config=config,
        model_dim=4,
        all_group=None,
    )
    assert producer is not None and producer.write_ring is not None
    assert consumer.cache is not None
    try:
        inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
        targets = torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.int64)
        pressure = torch.ones_like(targets, dtype=torch.float32)
        ce = torch.tensor([[0.1, 9.0, 8.0, 0.2, 0.3]], dtype=torch.float32)
        hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
        mod._emit_episodic_payloads_gpu(
            emit=producer,
            inputs=inputs,
            targets=targets,
            pressure=pressure,
            per_token_ce=ce,
            hidden=hidden,
            rank=0,
            world_size=2,
            all_group=None,
            current_step=11,
            write_bucket=2,
        )
        drained = mod._drain_episodic_write_rings(
            consumer=consumer,
            current_step=12,
            embedding_version=0,
        )
        assert drained == 2
        assert consumer.write_ring_events_drained == 2
        assert len(consumer.cache) == 2
        assert len(consumer.controller_query_queue) == 2
        first = consumer.controller_query_queue[0]
        assert first["rank"] == 0
        assert first["step"] == 11
        assert first["write_bucket"] == 2
        assert first["source_write_id"] >> 56 == 0
    finally:
        mod._cleanup_episodic_event_rings(producer)
        mod._cleanup_episodic_event_rings(consumer)


def test_write_drain_daemon_survives_drain_exception():
    """A raise inside ``_drain_episodic_write_rings`` must not kill the
    daemon thread — bump ``write_ring_drain_errors``, log to stderr, and
    keep iterating so a transient malformed event doesn't leave the
    cache permanently empty for the rest of a multi-hour run.
    """
    import threading
    import types

    mod = _load_runner()
    consumer = types.SimpleNamespace(write_ring_drain_errors=0)
    stop_event = threading.Event()
    heartbeat = [0]
    call_count = [0]

    def _flaky_drain(**_kwargs):
        call_count[0] += 1
        if call_count[0] <= 3:
            raise RuntimeError(f"injected drain failure #{call_count[0]}")
        # Recover: stop the daemon after we've proven it kept iterating.
        stop_event.set()
        return 0

    original = mod._drain_episodic_write_rings
    mod._drain_episodic_write_rings = _flaky_drain
    try:
        thread = threading.Thread(
            target=mod._write_drain_main,
            kwargs={
                "consumer": consumer,
                "stop_event": stop_event,
                "embedding_version_ref": [0],
                "controller_score_mode": "pressure_only",
                "controller_topk_k": 16,
                "heartbeat": heartbeat,
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "daemon thread did not exit cleanly"
        assert call_count[0] == 4, (
            f"daemon stopped iterating after exceptions; got {call_count[0]} calls"
        )
        assert consumer.write_ring_drain_errors == 3
        assert heartbeat[0] == 4
    finally:
        mod._drain_episodic_write_rings = original


def test_learned_write_admission_can_rerank_candidate_positions():
    mod = _load_runner()
    action_space = mod.ConstrainedActionSpace(
        head_readiness={"write_admission": 1.0},
        head_max_delta={"write_admission": 1000.0},
        event_ssm=_WriteAdmissionFakeSsm(),
    )
    signal = torch.tensor([[10.0, 1.0, 3.0]], dtype=torch.float32)

    positions = mod._select_write_positions_with_action_space(
        action_space=action_space,
        write_signal=signal,
        pressure_full=torch.ones_like(signal),
        ce_full=signal,
        top_p=1.0 / 3.0,
        k_max=1,
        current_step=5,
        write_bucket=0,
    )

    # Learned admission now sees a bounded top-M pool rather than the full
    # flattened B*T grid. rank=1 therefore means the second heuristic candidate
    # (flat index 2), proving it can rerank inside the stream without a full-grid
    # CPU copy.
    assert positions.tolist() == [[0, 2]]


def test_learned_eviction_head_sets_slot_protection_score():
    mod = _load_runner()
    handle = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config={
            "episodic_enabled": True,
            "episodic_span_length": 2,
            "episodic_fingerprint_window": 1,
            "episodic_key_rep_dim": 4,
            "episodic_top_p": 1.0 / 5.0,
            "episodic_k_max": 1,
            "episodic_async_write_rings_enabled": False,
            "model_dim": 4,
            "episodic_controller_action_space_enabled": True,
            "episodic_controller_eviction_readiness": 1.0,
        },
    )
    assert handle is not None
    assert handle.controller_action_space is not None
    handle.controller_action_space.event_ssm = _ProtectionFakeSsm()

    inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    targets = torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.int64)
    pressure = torch.ones_like(targets, dtype=torch.float32)
    ce = torch.tensor([[0.1, 9.0, 8.0, 0.2, 0.3]], dtype=torch.float32)
    hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    mod._emit_episodic_payloads_gpu(
        emit=handle,
        inputs=inputs,
        targets=targets,
        pressure=pressure,
        per_token_ce=ce,
        hidden=hidden,
        rank=0,
        world_size=2,
        all_group=None,
        current_step=5,
    )

    out = mod.unpack_payload(handle.slot_tensor[0], span_length=2, key_rep_dim=4)
    assert out["protection_score"] > 0.99
