"""Admission trace logging on the heuristic writer (Phase D1).

When ``episodic_admission_trace_path`` is set, every ``select_writes``
decision (admit AND reject) is appended as one NDJSON row matching the
WRITE_EVENT wire schema column order. When the path is None, no file
is created and the writer is bit-identical to pre-D1.

The "reject" path inside ``select_writes`` is not a pressure threshold
— it is the boundary check inside ``build_write_payload``. A position
that survives top-K selection but lacks a full fingerprint window or a
full target span returns ``None`` and is dropped. Those drops are the
rejects we need to capture for the offline pretrain dataset; downstream
analysis can filter on ``decision=0`` to study what the heuristic is
choosing not to write, not just what it commits.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from chaoscontrol.optim.episodic_writer import (
    _reset_admission_trace_seq,
    select_writes,
)


_EXPECTED_KEYS = [
    "candidate_id",
    "decision",
    "gpu_step",
    "source_rank",
    "key_fp",
    "key_rep_l2",
    "value_anchor_id",
    "pressure_at_write",
    "pre_write_ce",
    "write_bucket",
]


def _mixed_admit_reject_inputs():
    """Build a 1x8 batch where top-K=4 will straddle two interior positions
    (admits) and two boundary positions (rejects).

    fingerprint_window=4, span_length=4 means valid positions are
    4 <= t < T-span_length = 4. With T=8 the only in-bounds position is
    t=4, so we use T=12 to give us a window of valid positions [4..7]
    and force boundary rejects at t=0 and t=11.
    """
    B, T, D = 1, 12, 4
    input_ids = torch.arange(B * T, dtype=torch.int64).reshape(B, T)
    target_ids = (input_ids + 1) % 100
    key_rep = torch.randn(B, T, D, generator=torch.Generator().manual_seed(0))
    pressure = torch.zeros(B, T)
    ce = torch.zeros(B, T)
    # Two interior admits — high signal at t=5, t=6 (both pass boundary).
    pressure[0, 5] = 1.0
    ce[0, 5] = 9.0
    pressure[0, 6] = 1.0
    ce[0, 6] = 8.0
    # Two boundary rejects — high signal at t=0 (left) and t=11 (right).
    pressure[0, 0] = 1.0
    ce[0, 0] = 7.0
    pressure[0, 11] = 1.0
    ce[0, 11] = 6.0
    return input_ids, target_ids, pressure, ce, key_rep


def test_admission_trace_writes_one_row_per_candidate(tmp_path):
    """Run select_writes with 4 top-K candidates (2 admits, 2 boundary
    rejects); verify the trace file has 4 rows in WRITE_EVENT column
    order, with decisions matching the boundary-survival outcome."""
    _reset_admission_trace_seq()
    inputs, targets, pressure, ce, rep = _mixed_admit_reject_inputs()
    trace_path = tmp_path / "admission_rank0.ndjson"
    payloads = select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=4.0 / (1 * 12),  # k=4
        fingerprint_window=4,
        span_length=4,
        source_rank=2,
        gpu_step=137,
        write_bucket=1,
        admission_trace_path=str(trace_path),
    )
    # Two admits survive (positions 5, 6); two boundary rejects (0, 11).
    assert len(payloads) == 2
    assert {p.position for p in payloads} == {5, 6}

    rows = [
        json.loads(line) for line in trace_path.read_text().splitlines()
    ]
    assert len(rows) == 4
    for row in rows:
        assert list(row.keys()) == _EXPECTED_KEYS, row
        assert row["source_rank"] == 2
        assert row["gpu_step"] == 137
        assert row["write_bucket"] == 1

    # Decisions: positions 5 and 6 admit (1); positions 0 and 11 reject (0).
    by_position = {}
    # Recover position from value_anchor_id (target_ids = (arange + 1) % 100).
    # target_ids[0, t] = (t + 1) % 100 — so anchor==t+1 gives back t.
    for row in rows:
        t = (int(row["value_anchor_id"]) - 1) % 12
        by_position[t] = row
    assert by_position[5]["decision"] == 1
    assert by_position[6]["decision"] == 1
    assert by_position[0]["decision"] == 0
    assert by_position[11]["decision"] == 0

    # candidate_id has source_rank in the high byte.
    for row in rows:
        assert (int(row["candidate_id"]) >> 56) == 2

    # candidate_ids are unique within the batch.
    cids = [row["candidate_id"] for row in rows]
    assert len(set(cids)) == 4

    # key_rep_l2 matches torch.linalg.vector_norm of the slice.
    for row in rows:
        t = (int(row["value_anchor_id"]) - 1) % 12
        expected_l2 = float(torch.linalg.vector_norm(rep[0, t]).item())
        assert abs(row["key_rep_l2"] - expected_l2) < 1e-5


def test_admission_trace_path_none_creates_no_file(tmp_path):
    """With trace path None/unset, no file is created and behavior is
    identical to pre-D1."""
    _reset_admission_trace_seq()
    inputs, targets, pressure, ce, rep = _mixed_admit_reject_inputs()
    select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=4.0 / (1 * 12),
        fingerprint_window=4,
        span_length=4,
    )
    assert list(tmp_path.iterdir()) == []


def test_admission_trace_back_compat_bit_identical(tmp_path):
    """The decisions select_writes makes don't change when tracing is
    enabled vs disabled (tracing is a side effect only)."""
    _reset_admission_trace_seq()
    inputs, targets, pressure, ce, rep = _mixed_admit_reject_inputs()
    payloads_off = select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=4.0 / (1 * 12),
        fingerprint_window=4,
        span_length=4,
    )
    _reset_admission_trace_seq()
    trace_path = tmp_path / "admission.ndjson"
    payloads_on = select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=4.0 / (1 * 12),
        fingerprint_window=4,
        span_length=4,
        admission_trace_path=str(trace_path),
    )
    assert len(payloads_off) == len(payloads_on)
    for a, b in zip(payloads_off, payloads_on):
        assert a.batch_index == b.batch_index
        assert a.position == b.position
        assert a.key_fp == b.key_fp
        assert a.value_anchor_id == b.value_anchor_id
        assert torch.equal(a.value_tok_ids, b.value_tok_ids)
        assert torch.equal(a.key_rep, b.key_rep)


def test_admission_trace_appends_across_calls(tmp_path):
    """A second call to select_writes appends to the same trace file —
    so a runner can call it once per training step and accumulate rows
    in one rank-local NDJSON. rank_local_seq must keep advancing across
    calls so candidate_ids stay unique within a rank."""
    _reset_admission_trace_seq()
    inputs, targets, pressure, ce, rep = _mixed_admit_reject_inputs()
    trace_path = tmp_path / "admission.ndjson"
    select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=4.0 / (1 * 12),
        fingerprint_window=4,
        span_length=4,
        source_rank=0,
        gpu_step=0,
        admission_trace_path=str(trace_path),
    )
    select_writes(
        input_ids=inputs,
        target_ids=targets,
        pressure=pressure,
        per_token_ce=ce,
        key_rep_per_position=rep,
        top_p=4.0 / (1 * 12),
        fingerprint_window=4,
        span_length=4,
        source_rank=0,
        gpu_step=1,
        admission_trace_path=str(trace_path),
    )
    rows = [
        json.loads(line) for line in trace_path.read_text().splitlines()
    ]
    assert len(rows) == 8
    cids = [row["candidate_id"] for row in rows]
    assert len(set(cids)) == 8  # all unique within the rank's stream
