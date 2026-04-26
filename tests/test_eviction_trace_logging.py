"""Eviction trace logging on EpisodicCache (Phase D2).

When ``eviction_trace_path`` is set, every eviction (slot displacement
in ``append()`` at capacity) writes one NDJSON row carrying the
displaced slot's pre-overwrite state and the displacing write's id.
When unset (default), no file is created and the cache is bit-identical
to pre-D2 behavior.

The schema mirrors what the trained CPU SSM controller (Phase C) needs
to learn an eviction policy from heuristic traces (Phase D, offline
bootstrap).
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from chaoscontrol.optim.episodic_cache import EpisodicCache


def _fill_to_capacity(cache: EpisodicCache, *, base_step: int = 10) -> None:
    """Fill `cache` exactly to capacity with key_fps 10, 11, 12, ..."""
    for i in range(cache.capacity):
        cache.append(
            key_fp=10 + i,
            key_rep=torch.zeros(cache.key_rep_dim),
            value_tok_ids=torch.tensor(
                [1, 2, 3, 4][: cache.span_length], dtype=torch.int64,
            ),
            value_anchor_id=i,
            current_step=base_step + i,
            embedding_version=0,
            pressure_at_write=0.5,
            write_bucket=0,
        )


def test_eviction_trace_records_displaced_slot():
    """Fill a cache to capacity 4, append one more; verify one NDJSON row
    lands with the documented schema and key insertion order.

    With ``grace_steps=0`` every slot is past grace immediately; all four
    utilities are at the init value 1.0, so ``argmin()`` returns slot 0 —
    the first inserted key (10) — which matches the test's assertion.
    """
    with TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "eviction_rank0.ndjson"
        cache = EpisodicCache(
            capacity=4,
            span_length=4,
            key_rep_dim=8,
            grace_steps=0,
            eviction_trace_path=str(trace_path),
        )
        _fill_to_capacity(cache)
        # 5th append must displace exactly one slot.
        cache.append(
            key_fp=99,
            key_rep=torch.zeros(8),
            value_tok_ids=torch.tensor([5, 6, 7, 8], dtype=torch.int64),
            value_anchor_id=99,
            current_step=20,
            embedding_version=0,
            pressure_at_write=0.9,
            write_bucket=2,
            displacing_candidate_id=(7 << 56) | 42,
        )
        rows = [
            json.loads(line) for line in trace_path.read_text().splitlines()
        ]
        assert len(rows) == 1, rows
        row = rows[0]

        # Documented schema and KEY ORDER. Python 3.7+ dicts preserve
        # insertion order; controller training reads the schema from
        # column position, so the order is part of the contract.
        expected_keys = [
            "evicted_slot_id",
            "evicted_key_fp",
            "evicted_utility_at_eviction",
            "evicted_write_step",
            "evicted_last_fired_step",
            "gpu_step",
            "displacing_candidate_id",
            "displacing_key_fp",
        ]
        assert list(row.keys()) == expected_keys

        assert row["evicted_slot_id"] == 0
        assert row["evicted_key_fp"] == 10  # first inserted = first evicted
        assert row["evicted_utility_at_eviction"] == 1.0  # init utility
        assert row["evicted_write_step"] == 10
        assert row["evicted_last_fired_step"] == -1  # never fired
        assert row["gpu_step"] == 20
        assert row["displacing_candidate_id"] == (7 << 56) | 42
        assert row["displacing_key_fp"] == 99


def test_eviction_trace_appends_one_row_per_eviction():
    """Two over-capacity appends produce two rows."""
    with TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "evict.ndjson"
        cache = EpisodicCache(
            capacity=4,
            span_length=4,
            key_rep_dim=8,
            grace_steps=0,
            eviction_trace_path=str(trace_path),
        )
        _fill_to_capacity(cache)
        for j in range(2):
            cache.append(
                key_fp=200 + j,
                key_rep=torch.zeros(8),
                value_tok_ids=torch.zeros(4, dtype=torch.int64),
                value_anchor_id=j,
                current_step=30 + j,
                embedding_version=0,
            )
        rows = [
            json.loads(line) for line in trace_path.read_text().splitlines()
        ]
        assert len(rows) == 2
        # Default displacing_candidate_id sentinel = -1 when caller omits it.
        assert rows[0]["displacing_candidate_id"] == -1
        assert rows[1]["displacing_candidate_id"] == -1
        assert rows[0]["displacing_key_fp"] == 200
        assert rows[1]["displacing_key_fp"] == 201


def test_eviction_trace_path_none_creates_no_file(tmp_path):
    """No trace path → no file lands on disk, even when evictions happen."""
    cache = EpisodicCache(
        capacity=4, span_length=4, key_rep_dim=8, grace_steps=0,
    )
    _fill_to_capacity(cache)
    cache.append(
        key_fp=99,
        key_rep=torch.zeros(8),
        value_tok_ids=torch.tensor([5, 6, 7, 8], dtype=torch.int64),
        value_anchor_id=99,
        current_step=20,
        embedding_version=0,
    )
    # tmp_path is the test's per-test tmp dir; we passed no path to the
    # cache, so absolutely nothing should appear there.
    assert list(tmp_path.iterdir()) == []


def test_eviction_trace_no_rows_below_capacity(tmp_path):
    """No evictions → no file is touched (we don't pre-create it)."""
    trace_path = tmp_path / "eviction.ndjson"
    cache = EpisodicCache(
        capacity=4,
        span_length=4,
        key_rep_dim=8,
        eviction_trace_path=str(trace_path),
    )
    # Fill to capacity but do not over-append.
    _fill_to_capacity(cache)
    assert not trace_path.exists()


def test_eviction_trace_back_compat_bit_identical():
    """Cache state evolution is identical with tracing on vs off.

    Two parallel caches built with the same construction args + the same
    sequence of ``append()`` calls must produce equal ``to_dict()``
    payloads at the end. The trace-on cache also writes a side-effect
    file; the trace-off cache does not. Field tensors and the hash index
    must match element-wise.
    """
    with TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "evict.ndjson"

        kwargs = dict(
            capacity=3, span_length=2, key_rep_dim=4, grace_steps=2,
        )
        cache_off = EpisodicCache(**kwargs)
        cache_on = EpisodicCache(**kwargs, eviction_trace_path=str(trace_path))

        # Drive both caches through the same sequence: fill, then 3
        # over-capacity appends to force evictions.
        events = [
            (10, 0, 0.1, 0),
            (11, 1, 0.2, 1),
            (12, 2, 0.3, 0),
            (13, 10, 0.4, 1),  # past grace=2 by step 10 → eviction
            (14, 11, 0.5, 0),
            (15, 12, 0.6, 1),
        ]
        for key_fp, step, pressure, bucket in events:
            for cache in (cache_off, cache_on):
                cache.append(
                    key_fp=key_fp,
                    key_rep=torch.full((4,), float(key_fp)),
                    value_tok_ids=torch.tensor(
                        [key_fp, key_fp + 1], dtype=torch.int64,
                    ),
                    value_anchor_id=key_fp,
                    current_step=step,
                    embedding_version=0,
                    pressure_at_write=pressure,
                    write_bucket=bucket,
                )

        blob_off = cache_off.to_dict()
        blob_on = cache_on.to_dict()
        for key in blob_off:
            v_off = blob_off[key]
            v_on = blob_on[key]
            if isinstance(v_off, torch.Tensor):
                assert torch.equal(v_off, v_on), f"tensor mismatch on {key}"
            else:
                assert v_off == v_on, f"scalar/dict mismatch on {key}"

        # And the trace-on side did record evictions (3 over-cap appends).
        assert trace_path.exists()
        rows = [
            json.loads(line) for line in trace_path.read_text().splitlines()
        ]
        assert len(rows) == 3
