"""WRITE_EVENT producer for the episodic writer path (Phase B1).

Phase A4 will replace these in-process placeholder lists with real
shared-memory ring pushes. Until then, the writer appends one dict per
admitted candidate when ``episodic_event_log_enabled=True`` and does
nothing when the placeholder list is absent.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from chaoscontrol.optim.episodic_writer import (
    _reset_admission_trace_seq,
    select_writes,
)

REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"

WRITE_EVENT_KEYS = (
    "event_type",
    "candidate_id",
    "gpu_step",
    "source_rank",
    "key_fp",
    "key_rep",
    "value_tok_ids",
    "value_anchor_id",
    "pressure_at_write",
    "pre_write_ce",
    "write_bucket",
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("runner_b1", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


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
    write_event_log: list[dict] = []

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
        write_event_log=write_event_log,
    )

    assert len(payloads) == 2
    assert len(write_event_log) == 2
    assert {p.position for p in payloads} == {5, 6}

    trace_rows = [
        json.loads(line) for line in trace_path.read_text().splitlines()
    ]
    admitted_trace_ids = [
        int(row["candidate_id"]) for row in trace_rows if row["decision"] == 1
    ]
    assert [event["candidate_id"] for event in write_event_log] == admitted_trace_ids

    for event in write_event_log:
        assert tuple(event.keys()) == WRITE_EVENT_KEYS
        assert event["event_type"] == 1
        assert event["source_rank"] == 2
        assert event["gpu_step"] == 137
        assert event["write_bucket"] == 1
        assert (event["candidate_id"] >> 56) == 2
        assert len(event["key_rep"]) == 4
        assert len(event["value_tok_ids"]) == 4
        assert event["pressure_at_write"] == pytest.approx(1.0)
    assert [event["pre_write_ce"] for event in write_event_log] == pytest.approx(
        [9.0, 8.0]
    )


def test_select_writes_without_write_event_log_is_side_effect_free(tmp_path):
    """Back-compat: None means no list allocation and unchanged payloads."""
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


def test_runner_emit_allocates_write_event_log_only_on_train_rank_when_enabled():
    """B1's placeholder log lives on the train-rank emit handle."""
    mod = _load_runner()
    base_config = {
        "episodic_enabled": True,
        "episodic_span_length": 2,
        "episodic_fingerprint_window": 1,
        "episodic_key_rep_dim": 4,
        "episodic_k_max": 4,
        "model_dim": 4,
    }

    disabled = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config=base_config,
    )
    assert disabled is not None
    assert disabled.write_event_log is None

    enabled_train = mod._create_episodic_emit(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        config={**base_config, "episodic_event_log_enabled": True},
    )
    assert enabled_train is not None
    assert enabled_train.write_event_log == []

    enabled_episodic = mod._create_episodic_emit(
        rank=1,
        world_size=2,
        device=torch.device("cpu"),
        config={**base_config, "episodic_event_log_enabled": True},
    )
    assert enabled_episodic is not None
    assert enabled_episodic.write_event_log is None


def test_runner_emit_appends_write_event_for_packed_slots():
    """The current runner producer packs directly; it still emits B1 records."""
    _reset_admission_trace_seq()
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
            "episodic_top_p": 0.5,
            "episodic_k_max": 4,
            "episodic_event_log_enabled": True,
            "model_dim": 4,
        },
    )
    assert handle is not None
    assert handle.write_event_log is not None
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

    assert len(handle.write_event_log) == 2
    ids = [event["candidate_id"] for event in handle.write_event_log]
    assert [qid >> 56 for qid in ids] == [3, 3]
    assert [qid & ((1 << 56) - 1) for qid in ids] == [0, 1]
    for event in handle.write_event_log:
        assert tuple(event.keys()) == WRITE_EVENT_KEYS
        assert event["event_type"] == 1
        assert event["gpu_step"] == 19
        assert event["source_rank"] == 3
        assert event["write_bucket"] == 2
        assert len(event["key_rep"]) == 4
        assert len(event["value_tok_ids"]) == 2
