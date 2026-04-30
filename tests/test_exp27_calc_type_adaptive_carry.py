"""Tests for the Exp27 ``adaptive_carry`` calc_type."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from chaoscontrol.eval.calc_types.adaptive_carry import adaptive_carry
from chaoscontrol.eval.calc_types.carry_state import carry_state
from chaoscontrol.eval.ttt_eval import CalcTypeContext
from chaoscontrol.eval_stream.val_cache import DOC_DTYPE, TOKEN_DTYPE, ValCache


VOCAB = 19
DIM = 8


class _CountingLMHead(nn.Module):
    def __init__(self, dim: int, vocab: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, vocab, bias=False)
        self.call_count = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return self.linear(hidden)


class PacketTinyModel(nn.Module):
    """Tiny recurrent model whose forward path is forbidden.

    ``adaptive_carry`` should use ``encode(memory_mode="packet")`` directly,
    not ``model.forward()``.  If it accidentally calls forward, tests fail.
    """

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.w_in = nn.Linear(DIM, DIM, bias=False)
        self.w_state = nn.Linear(DIM, DIM, bias=False)
        self.lm_head = _CountingLMHead(DIM, VOCAB)
        self.memory_modes: list[str] = []
        self.init_states_log: list[list[torch.Tensor] | None] = []
        self.final_states_log: list[torch.Tensor] = []

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
        if initial_states is None:
            self.init_states_log.append(None)
        else:
            self.init_states_log.append([s.detach().clone() for s in initial_states])
        x = self.embed(input_ids)
        if episodic_residual is not None:
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
        self.final_states_log.append(state.detach().clone())
        if return_final_states:
            return hidden, [state]
        return hidden

    def forward(self, *_args, **_kwargs):  # pragma: no cover - failure path
        raise AssertionError("adaptive_carry must not call model.forward()")


class TraceOuterModel:
    def __init__(self, events: list[tuple[str, object]], dim: int) -> None:
        self.events = events
        self.dim = dim
        self._slots: list[torch.Tensor] = []

    def read(
        self,
        batch_size: int,
        *,
        cue: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.events.append(("read", len(self._slots)))
        scale = 0.25 * max(1, len(self._slots))
        return torch.full((batch_size, self.dim), scale)


class EpisodicPacketTinyModel(PacketTinyModel):
    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self.events: list[tuple[str, object]] = []
        self.packet_flags: list[bool] = []
        self.outer_model = TraceOuterModel(self.events, DIM)

    def encode(self, input_ids: torch.Tensor, **kwargs):
        has_packet = kwargs.get("episodic_residual") is not None
        self.packet_flags.append(bool(has_packet))
        self.events.append(("encode", bool(has_packet)))
        return super().encode(input_ids, **kwargs)

    def append_memory_from_hidden(
        self,
        hidden: torch.Tensor,
        *,
        score: torch.Tensor | None = None,
        max_tokens: int | None = None,
    ) -> bool:
        self.events.append(("append", int(hidden.shape[1])))
        self.outer_model._slots.append(hidden.detach().mean(dim=1))
        return True


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


def test_adaptive_carry_returns_finite_and_telemetry(tmp_path):
    model = PacketTinyModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[16, 24, 32])
    ctx = make_ctx(model, cache)

    result = adaptive_carry(ctx)

    assert math.isfinite(result.bpb)
    assert math.isfinite(result.loss)
    assert result.docs_scored == 3
    assert result.tokens_scored == (16 - 1) + (24 - 1) + (32 - 1)
    assert result.hyperparams["horizon_shifts"] == [-0.5, 0.0, 0.5]
    assert set(result.extra["winner_counts_by_shift"]) == {"-0.5", "0.0", "0.5"}
    assert sum(result.extra["winner_counts_by_shift"].values()) == result.tokens_scored
    assert len(result.extra["online_final_weights"]) == 3
    assert set(model.memory_modes) == {"packet"}


def test_adaptive_carry_single_zero_shift_matches_raw_carry(tmp_path):
    cache = make_val_cache(tmp_path, doc_lens=[12, 16, 20])

    adaptive_model = PacketTinyModel(seed=4)
    adaptive = adaptive_carry(
        make_ctx(
            adaptive_model,
            cache,
            config={"horizon_shifts": [0.0], "online_eta": 1.0, "decay": 1.0},
        )
    )

    raw_model = PacketTinyModel(seed=4)
    raw = carry_state(make_ctx(raw_model, cache, config={"decay": 1.0}))

    assert adaptive.bpb == pytest.approx(raw.bpb, rel=0.0, abs=1e-6)
    assert adaptive.loss == pytest.approx(raw.loss, rel=0.0, abs=1e-6)
    assert adaptive.tokens_scored == raw.tokens_scored
    assert set(adaptive_model.memory_modes) == {"packet"}


def test_adaptive_carry_rejects_bad_online_weight_count(tmp_path):
    model = PacketTinyModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[8, 8])
    ctx = make_ctx(
        model,
        cache,
        config={
            "horizon_shifts": [-0.5, 0.0, 0.5],
            "online_initial_weights": [1.0, 1.0],
        },
    )

    with pytest.raises(ValueError, match="online_initial_weights length"):
        adaptive_carry(ctx)


def test_adaptive_carry_online_episodic_eval_is_prequential(tmp_path):
    """Episodic eval reads only cache entries committed by earlier chunks.

    First chunk has no packet. After its loss is counted, adaptive_carry
    appends hidden evidence. Later chunks can then read that cache and pass an
    episodic packet into ``encode(memory_mode="packet")``.
    """
    model = EpisodicPacketTinyModel(seed=7)
    cache = make_val_cache(tmp_path, doc_lens=[10])
    ctx = make_ctx(
        model,
        cache,
        config={
            "horizon_shifts": [0.0],
            "online_episodic_chunk_tokens": 3,
            "online_episodic_write_tokens_per_chunk": 2,
            "online_episodic_gate": 1.0,
        },
    )

    result = adaptive_carry(ctx)

    assert result.tokens_scored == 9
    assert result.extra["online_episodic_writes"] > 0
    assert result.extra["online_episodic_reads"] > 0
    assert model.packet_flags[0] is False
    assert any(model.packet_flags[1:])

    first_append = next(
        idx for idx, event in enumerate(model.events) if event[0] == "append"
    )
    first_read = next(
        idx for idx, event in enumerate(model.events) if event[0] == "read"
    )
    assert first_append < first_read
