"""QUERY_EVENT producer in runner_fast_path (Phase B2).

When episodic_event_log_enabled=True, the runner records one
QUERY_EVENT dict per query emission. When the flag is unset, no
list is allocated and behavior is bit-identical to pre-B2.

Phase B4 will replace the in-process list with shm-ring pushes
once Phase A4 (ShmRing) lands.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from chaoscontrol.episodic.gpu_slot import make_slot_tensor, pack_payload

REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"

QUERY_EVENT_KEYS = (
    "event_type",
    "query_id",
    "gpu_step",
    "source_rank",
    "query_rep",
    "pressure",
    "pre_query_ce",
    "bucket",
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("runner_b2", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _consumer(mod, *, event_log_enabled: bool):
    return mod._attach_episodic_consumer(
        episodic_enabled=True,
        is_episodic_rank=True,
        world_size=2,
        config={
            "episodic_capacity": 16,
            "episodic_span_length": 2,
            "episodic_key_rep_dim": 4,
            "controller_query_enabled": True,
            "episodic_event_log_enabled": event_log_enabled,
        },
        model_dim=4,
        all_group=None,
    )


def _gather_list_with_valid_query(
    *,
    source_rank: int,
    k_max: int = 1,
    query_residual: torch.Tensor | None = None,
    pressure: float = 0.75,
) -> list[torch.Tensor]:
    span_length = 2
    key_rep_dim = 4
    gather_list: list[torch.Tensor] = []
    residual = (
        query_residual
        if query_residual is not None
        else torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    )
    for rank in range(source_rank + 1):
        t = make_slot_tensor(
            k_max=k_max,
            span_length=span_length,
            key_rep_dim=key_rep_dim,
            device=torch.device("cpu"),
        )
        if rank == source_rank:
            for k in range(k_max):
                pack_payload(
                    t[k],
                    valid_mask=1.0,
                    pressure=pressure + float(k),
                    key_fp=1000 + k,
                    value_anchor_id=7,
                    value_tok_ids=torch.tensor([7, 8], dtype=torch.int64),
                    key_rep=torch.zeros(key_rep_dim),
                    residual=residual + float(k),
                    span_length=span_length,
                    key_rep_dim=key_rep_dim,
                )
        gather_list.append(t)
    return gather_list


def test_query_event_emitted_when_log_enabled():
    """Run one query-emission cycle with episodic_event_log_enabled=True;
    verify exactly one QUERY_EVENT dict appears with the right fields
    and the dict matches the wire-event schema column order."""
    mod = _load_runner()
    consumer = _consumer(mod, event_log_enabled=True)
    residual = torch.tensor([1.25, 2.5, 3.75, 5.0], dtype=torch.float32)

    mod._drain_episodic_payloads_gpu(
        consumer=consumer,
        gather_list=_gather_list_with_valid_query(
            source_rank=1,
            query_residual=residual,
            pressure=0.875,
        ),
        span_length=2,
        key_rep_dim=4,
        k_max=1,
        current_step=11,
        embedding_version=0,
        pre_query_ce=1.5,
        query_bucket=3,
    )

    assert consumer.query_event_log is not None
    assert len(consumer.query_event_log) == 1
    event = consumer.query_event_log[0]
    assert tuple(event.keys()) == QUERY_EVENT_KEYS
    assert event["event_type"] == 2
    assert event["query_id"] == (1 << 56)
    assert event["gpu_step"] == 11
    assert event["source_rank"] == 1
    assert event["pressure"] == pytest.approx(0.875)
    assert event["pre_query_ce"] == pytest.approx(1.5)
    assert event["bucket"] == 3
    assert len(event["query_rep"]) == 4
    torch.testing.assert_close(
        torch.tensor(event["query_rep"], dtype=torch.float16),
        residual.to(torch.float16),
    )


def test_no_query_event_log_when_disabled():
    """With episodic_event_log_enabled=False (default), no list is
    allocated on the consumer state."""
    mod = _load_runner()
    consumer = _consumer(mod, event_log_enabled=False)

    assert getattr(consumer, "query_event_log", None) is None
    mod._drain_episodic_payloads_gpu(
        consumer=consumer,
        gather_list=_gather_list_with_valid_query(source_rank=0),
        span_length=2,
        key_rep_dim=4,
        k_max=1,
        current_step=5,
        embedding_version=0,
    )

    assert getattr(consumer, "query_event_log", None) is None
    assert len(consumer.controller_query_queue) == 1


def test_query_id_is_rank_prefixed_monotonic():
    """query_id high 8 bits == source_rank, low 56 bits monotonic
    per rank."""
    mod = _load_runner()
    consumer = _consumer(mod, event_log_enabled=True)

    mod._drain_episodic_payloads_gpu(
        consumer=consumer,
        gather_list=_gather_list_with_valid_query(source_rank=2, k_max=3),
        span_length=2,
        key_rep_dim=4,
        k_max=3,
        current_step=9,
        embedding_version=0,
    )

    assert consumer.query_event_log is not None
    assert len(consumer.query_event_log) == 3
    query_ids = [row["query_id"] for row in consumer.query_event_log]
    assert [qid >> 56 for qid in query_ids] == [2, 2, 2]
    assert [qid & ((1 << 56) - 1) for qid in query_ids] == [0, 1, 2]
