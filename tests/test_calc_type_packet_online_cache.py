"""Tests for the ``packet_online_cache`` calc_type.

The submission-facing eval path: prefix-safe cue + packet-lane score +
post-score cache write.  These tests pin the contract:

  1. The packet lane is used (no ``model.forward()`` calls).
  2. Score-before-write ordering: a chunk's loss is counted before its
     hidden states enter the cache, so no token is scored against its own
     evidence.
  3. ``seeded=False`` clears the cache before eval starts.
  4. Cache count grows monotonically across chunks.
  5. Result metadata exposes read/write/chunk counts and slot-count delta.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from chaoscontrol.eval.calc_types.packet_online_cache import packet_online_cache
from chaoscontrol.eval.ttt_eval import (
    CALC_TYPE_METADATA,
    CALC_TYPE_REGISTRY,
    CalcTypeContext,
)
from chaoscontrol.eval_stream.val_cache import DOC_DTYPE, TOKEN_DTYPE, ValCache


VOCAB = 19
DIM = 8


class _CountingLMHead(nn.Module):
    def __init__(self, dim: int, vocab: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, vocab, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class TraceOuterModel:
    """Tiny outer model that logs reads/appends and tracks slot count."""

    def __init__(self, events: list[tuple[str, object]], dim: int) -> None:
        self.events = events
        self.dim = dim
        self._slots: list[torch.Tensor] = []
        self._survival: list[float] = []
        self._slot_buckets: list[int] = []
        self._slot_event_ids: list[int] = []

    def read(
        self,
        batch_size: int,
        *,
        cue: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected = int(slot_mask.sum().item()) if slot_mask is not None else 0
        self.events.append(("read", {"slots": len(self._slots), "selected": selected}))
        scale = 0.25 * max(1, selected or len(self._slots))
        return torch.full((batch_size, self.dim), scale)


class PacketOnlineModel(nn.Module):
    """Recurrent model whose ``forward`` is forbidden.

    The calc_type must use ``encode(memory_mode='packet')`` only.  Slot-count
    growth is driven by ``append_memory_from_hidden`` calls.
    """

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.w_in = nn.Linear(DIM, DIM, bias=False)
        self.w_state = nn.Linear(DIM, DIM, bias=False)
        self.lm_head = _CountingLMHead(DIM, VOCAB)
        self.events: list[tuple[str, object]] = []
        self.outer_model = TraceOuterModel(self.events, DIM)
        self.memory_modes: list[str] = []
        self.packet_flags: list[bool] = []

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        memory_mode: str = "packet",
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
        episodic_residual: torch.Tensor | None = None,
        episodic_gate: torch.Tensor | None = None,
    ):
        self.memory_modes.append(memory_mode)
        has_packet = episodic_residual is not None
        self.packet_flags.append(has_packet)
        self.events.append(("encode", has_packet))
        x = self.embed(input_ids)
        if has_packet:
            residual = episodic_residual.to(device=x.device, dtype=x.dtype)
            if residual.dim() == 2:
                residual = residual.unsqueeze(1)
            gate = episodic_gate.to(device=x.device, dtype=x.dtype)
            if gate.dim() == 1:
                gate = gate[:, None]
            x = x + residual * gate.unsqueeze(-1)
        if initial_states is None:
            state = torch.zeros(input_ids.size(0), DIM, device=x.device, dtype=x.dtype)
        else:
            state = initial_states[0].to(device=x.device, dtype=x.dtype)
        outs = []
        for pos in range(input_ids.size(1)):
            state = self.w_state(state) + self.w_in(x[:, pos, :])
            outs.append(state)
        hidden = torch.stack(outs, dim=1)
        if return_final_states:
            return hidden, [state]
        return hidden

    def forward(self, *_args, **_kwargs):  # pragma: no cover - failure path
        raise AssertionError("packet_online_cache must not call model.forward()")

    def append_memory_from_hidden(
        self,
        hidden: torch.Tensor,
        *,
        score: torch.Tensor | None = None,
        max_tokens: int | None = None,
        event_ids: torch.Tensor | None = None,
    ) -> list[dict[str, object]]:
        slots_before = len(self.outer_model._slots)
        self.events.append(("append", int(hidden.shape[1])))
        self.outer_model._slots.append(hidden.detach().mean(dim=1))
        self.outer_model._survival.append(1.0)
        self.outer_model._slot_buckets.append(0)
        self.outer_model._slot_event_ids.append(0)
        return [
            {
                "slot_id": slots_before,
                "tensor": self.outer_model._slots[-1],
                "bucket_id": 0,
                "event_id": 0,
                "generation": 0,
            }
        ]


def make_val_cache(
    tmp_path: Path,
    *,
    doc_lens: list[int],
    seed: int = 1234,
) -> ValCache:
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    rows: list[tuple[int, int, int, int]] = []
    offset = 0
    for idx, length in enumerate(doc_lens):
        chunk = rng.integers(0, VOCAB, size=length, dtype=np.int64).astype(TOKEN_DTYPE)
        chunks.append(chunk)
        rows.append((idx, offset, length, max(1, 4 * length)))
        offset += length
    tokens = np.concatenate(chunks) if chunks else np.zeros(0, dtype=TOKEN_DTYPE)
    docs = np.asarray(rows, dtype=DOC_DTYPE)
    return ValCache(
        cache_dir=tmp_path,
        manifest={"schema_version": 1, "synthetic": True},
        tokens=tokens,
        docs=docs,
    )


def make_ctx(
    model: nn.Module,
    val_cache: ValCache,
    *,
    config: dict | None = None,
) -> CalcTypeContext:
    return CalcTypeContext(
        model=model,
        val_cache=val_cache,
        device=torch.device("cpu"),
        base_bytes_lut=torch.zeros(VOCAB, dtype=torch.long),
        has_leading_space_lut=torch.zeros(VOCAB, dtype=torch.bool),
        is_boundary_token_lut=torch.zeros(VOCAB, dtype=torch.bool),
        config=config or {},
    )


def test_packet_online_cache_is_registered():
    assert "packet_online_cache" in CALC_TYPE_REGISTRY
    meta = CALC_TYPE_METADATA["packet_online_cache"]
    assert meta["requires_source_order"] is True
    assert meta["requires_grad"] is False


def test_packet_online_cache_returns_finite_metrics(tmp_path):
    model = PacketOnlineModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[16, 24, 32])
    ctx = make_ctx(
        model,
        cache,
        config={
            "chunk_tokens": 8,
            "write_tokens_per_chunk": 2,
            "gate_value": 1.0,
        },
    )

    result = packet_online_cache(ctx)

    assert math.isfinite(result.bpb)
    assert math.isfinite(result.loss)
    assert result.docs_scored == 3
    assert result.tokens_scored == (16 - 1) + (24 - 1) + (32 - 1)
    assert result.extra["chunks_scored"] > 0
    assert result.extra["episodic_writes"] > 0
    # Slot count must be strictly larger after eval (cache grows online).
    assert result.extra["slot_count_final"] > result.extra["slot_count_initial"]
    # Packet lane only — no model.forward() should ever be invoked.
    assert set(model.memory_modes) == {"packet"}


def test_packet_online_cache_score_before_write_ordering(tmp_path):
    """First chunk has no packet (cache empty); writes precede later reads."""
    model = PacketOnlineModel(seed=7)
    cache = make_val_cache(tmp_path, doc_lens=[10])
    ctx = make_ctx(
        model,
        cache,
        config={
            "chunk_tokens": 3,
            "write_tokens_per_chunk": 2,
            "gate_value": 1.0,
        },
    )

    result = packet_online_cache(ctx)

    # First encode of an empty cache must not carry an episodic packet.
    assert model.packet_flags[0] is False
    # At least one later encode reads from a non-empty cache.
    assert any(model.packet_flags[1:])
    assert result.extra["episodic_reads"] > 0
    assert result.extra["episodic_writes"] > 0

    first_append_idx = next(
        i for i, event in enumerate(model.events) if event[0] == "append"
    )
    # No read sees a non-empty cache before the first append.
    for i, event in enumerate(model.events[:first_append_idx]):
        if event[0] == "read":
            assert event[1]["slots"] == 0, (
                f"read at event index {i} saw {event[1]} before any append"
            )


def test_packet_online_cache_seeded_false_clears_cache(tmp_path):
    model = PacketOnlineModel(seed=2)
    # Pre-seed the outer model with a fake slot to simulate a checkpoint.
    seeded_slot = torch.full((1, DIM), 0.7)
    model.outer_model._slots.append(seeded_slot)
    model.outer_model._survival.append(1.0)
    model.outer_model._slot_buckets.append(0)
    model.outer_model._slot_event_ids.append(0)
    assert len(model.outer_model._slots) == 1

    cache = make_val_cache(tmp_path, doc_lens=[8, 8])
    ctx = make_ctx(
        model,
        cache,
        config={
            "chunk_tokens": 4,
            "write_tokens_per_chunk": 1,
            "gate_value": 1.0,
            "seeded": False,
        },
    )

    result = packet_online_cache(ctx)

    # seeded=False clears the seed before eval starts.
    assert result.extra["slot_count_initial"] == 0
    # Online accumulation still happened.
    assert result.extra["slot_count_final"] >= 1


def test_packet_online_cache_gate_zero_disables_residual(tmp_path):
    model = PacketOnlineModel(seed=3)
    # Pre-seed so a non-zero gate would otherwise fire reads.
    seeded_slot = torch.full((1, DIM), 0.7)
    model.outer_model._slots.append(seeded_slot)
    model.outer_model._survival.append(1.0)
    model.outer_model._slot_buckets.append(0)
    model.outer_model._slot_event_ids.append(0)

    cache = make_val_cache(tmp_path, doc_lens=[12])
    ctx = make_ctx(
        model,
        cache,
        config={
            "chunk_tokens": 4,
            "write_tokens_per_chunk": 0,
            "gate_value": 0.0,
        },
    )

    result = packet_online_cache(ctx)

    # gate_value=0 disables the residual entirely, so no reads happen.
    assert result.extra["episodic_reads"] == 0
    assert all(flag is False for flag in model.packet_flags)


def test_packet_online_cache_max_docs_caps_source_order_smoke(tmp_path):
    model = PacketOnlineModel(seed=4)
    cache = make_val_cache(tmp_path, doc_lens=[8, 10, 12])
    ctx = make_ctx(
        model,
        cache,
        config={
            "chunk_tokens": 4,
            "write_tokens_per_chunk": 1,
            "max_docs": 2,
        },
    )

    result = packet_online_cache(ctx)

    assert result.docs_scored == 2
    assert result.tokens_scored == (8 - 1) + (10 - 1)
    assert result.hyperparams["max_docs"] == 2


def test_packet_online_cache_batched_score_before_write_ordering(tmp_path):
    model = PacketOnlineModel(seed=5)
    # Seed one slot so the first batched score can read memory, then verify no
    # append happens until every doc in the first microbatch has been encoded.
    model.outer_model._slots.append(torch.full((1, DIM), 0.25))
    model.outer_model._survival.append(1.0)
    model.outer_model._slot_buckets.append(0)
    model.outer_model._slot_event_ids.append(0)
    cache = make_val_cache(tmp_path, doc_lens=[8, 10, 12, 14])
    ctx = make_ctx(
        model,
        cache,
        config={
            "batch_docs": 2,
            "batch_token_budget": 64,
            "write_tokens_per_chunk": 2,
        },
    )

    result = packet_online_cache(ctx)

    assert result.docs_scored == 4
    assert result.extra["chunks_scored"] == 2
    assert result.extra["episodic_reads"] == 2
    assert result.extra["episodic_writes"] == 2
    assert result.hyperparams["batch_docs"] == 2
    first_append_idx = next(
        i for i, event in enumerate(model.events) if event[0] == "append"
    )
    assert [event[0] for event in model.events[:first_append_idx]].count("encode") == 1


def test_packet_online_cache_batched_write_zero_is_seeded_read_only(tmp_path):
    model = PacketOnlineModel(seed=6)
    model.outer_model._slots.append(torch.full((1, DIM), 0.25))
    model.outer_model._survival.append(1.0)
    model.outer_model._slot_buckets.append(0)
    model.outer_model._slot_event_ids.append(0)
    cache = make_val_cache(tmp_path, doc_lens=[8, 10])
    ctx = make_ctx(
        model,
        cache,
        config={
            "batch_docs": 2,
            "batch_token_budget": 64,
            "write_tokens_per_chunk": 0,
        },
    )

    result = packet_online_cache(ctx)

    assert result.docs_scored == 2
    assert result.extra["episodic_reads"] == 1
    assert result.extra["episodic_writes"] == 0
    assert result.extra["slot_count_initial"] == 1
    assert result.extra["slot_count_final"] == 1
    assert "append" not in [event[0] for event in model.events]


def test_packet_online_cache_controller_read_selects_bounded_slots(tmp_path):
    model = PacketOnlineModel(seed=8)
    for i in range(5):
        model.outer_model._slots.append(torch.full((1, DIM), float(i + 1) / 10.0))
        model.outer_model._survival.append(float(i + 1))
        model.outer_model._slot_buckets.append(0)
        model.outer_model._slot_event_ids.append(0)
    cache = make_val_cache(tmp_path, doc_lens=[8, 10])
    ctx = make_ctx(
        model,
        cache,
        config={
            "batch_docs": 2,
            "batch_token_budget": 64,
            "write_tokens_per_chunk": 0,
            "controller_read_enabled": True,
            "controller_topk_k": 2,
        },
    )

    result = packet_online_cache(ctx)

    assert result.extra["episodic_reads"] == 1
    assert result.extra["controller_reads"] == 1
    assert result.extra["controller_selected_slots"] == 4  # batch 2 * topk 2
    read_events = [payload for name, payload in model.events if name == "read"]
    assert read_events[0]["selected"] == 4
    assert result.hyperparams["controller_read_enabled"] is True
    assert result.hyperparams["controller_topk_k"] == 2


def test_packet_online_cache_rejects_bad_config(tmp_path):
    model = PacketOnlineModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[4])

    with pytest.raises(ValueError, match="chunk_tokens"):
        packet_online_cache(make_ctx(model, cache, config={"chunk_tokens": 0}))
    with pytest.raises(ValueError, match="write_tokens_per_chunk"):
        packet_online_cache(
            make_ctx(model, cache, config={"write_tokens_per_chunk": -1})
        )
    with pytest.raises(ValueError, match="gate_value"):
        packet_online_cache(make_ctx(model, cache, config={"gate_value": -0.5}))
    with pytest.raises(ValueError, match="batch_docs"):
        packet_online_cache(make_ctx(model, cache, config={"batch_docs": 0}))
    with pytest.raises(ValueError, match="batch_token_budget"):
        packet_online_cache(make_ctx(model, cache, config={"batch_token_budget": -1}))
    with pytest.raises(ValueError, match="controller_topk_k"):
        packet_online_cache(
            make_ctx(
                model,
                cache,
                config={"controller_read_enabled": True, "controller_topk_k": 0},
            )
        )
    with pytest.raises(ValueError, match="controller_score_mode"):
        packet_online_cache(
            make_ctx(
                model,
                cache,
                config={
                    "controller_read_enabled": True,
                    "controller_score_mode": "not_real",
                },
            )
        )
