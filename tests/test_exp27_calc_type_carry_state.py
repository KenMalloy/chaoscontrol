"""Unit tests for the ``carry_state`` calc_type.

Reuses the ``TinyModel`` + synthetic ``ValCache`` pattern from the
``score_only_reset`` test file. Each test instantiates its own model so
the call counter / init-state log starts clean.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from chaoscontrol.eval.calc_types.carry_state import carry_state
from chaoscontrol.eval.calc_types.score_only_reset import score_only_reset
from chaoscontrol.eval.ttt_eval import CalcTypeContext
from chaoscontrol.eval_stream.val_cache import DOC_DTYPE, TOKEN_DTYPE, ValCache


VOCAB = 17
DIM = 8


class _CountingLMHead(nn.Module):
    def __init__(self, dim: int, vocab: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, vocab, bias=False)
        self.call_count = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return self.linear(hidden)


class TinyModel(nn.Module):
    """Same minimal SSM-ish model as the score_only_reset test file."""

    def __init__(self, vocab: int = VOCAB, dim: int = DIM, *, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, dim)
        self.w_in = nn.Linear(dim, dim, bias=False)
        self.w_state = nn.Linear(dim, dim, bias=False)
        self.lm_head = _CountingLMHead(dim, vocab)
        self.layers = (None,)
        self._init_states_log: list[list[torch.Tensor] | None] = []
        self._final_states_log: list[torch.Tensor] = []
        self._memory_modes_log: list[str | None] = []
        self.forward_call_count = 0

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        memory_mode: str | None = None,
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ):
        self._memory_modes_log.append(memory_mode)
        if initial_states is None:
            self._init_states_log.append(None)
        else:
            self._init_states_log.append([s.detach().clone() for s in initial_states])
        if initial_states is not None and len(initial_states) != 1:
            raise ValueError(
                f"TinyModel expects 1 layer state, got {len(initial_states)}"
            )
        batch, seq = input_ids.shape
        x = self.embed(input_ids)
        if initial_states is None:
            state = torch.zeros(batch, x.size(-1), device=x.device, dtype=x.dtype)
        else:
            state = initial_states[0].to(device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq):
            state = self.w_state(state) + self.w_in(x[:, t, :])
            outputs.append(state)
        hidden = torch.stack(outputs, dim=1)
        self._final_states_log.append(state.detach().clone())
        if return_final_states:
            return hidden, [state]
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self.forward_call_count += 1
        hidden = self.encode(input_ids, initial_states=initial_states)
        return self.lm_head(hidden)


def make_val_cache(
    tmp_path: Path,
    *,
    doc_lens: list[int],
    seed: int = 1234,
) -> ValCache:
    rng = np.random.default_rng(seed)
    tokens_chunks: list[np.ndarray] = []
    rows: list[tuple[int, int, int, int]] = []
    offset = 0
    for i, length in enumerate(doc_lens):
        chunk = rng.integers(0, VOCAB, size=length, dtype=np.int64).astype(TOKEN_DTYPE)
        tokens_chunks.append(chunk)
        rows.append((i, offset, length, max(1, 4 * length)))
        offset += length
    tokens = (
        np.concatenate(tokens_chunks)
        if tokens_chunks
        else np.zeros(0, dtype=TOKEN_DTYPE)
    )
    docs = np.asarray(rows, dtype=DOC_DTYPE)
    return ValCache(
        cache_dir=Path(tmp_path),
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
    device = torch.device("cpu")
    return CalcTypeContext(
        model=model,
        val_cache=val_cache,
        device=device,
        base_bytes_lut=torch.zeros(VOCAB, dtype=torch.long),
        has_leading_space_lut=torch.zeros(VOCAB, dtype=torch.bool),
        is_boundary_token_lut=torch.zeros(VOCAB, dtype=torch.bool),
        config=config or {},
    )


# ---- Tests -----------------------------------------------------------------


def test_carry_state_returns_finite(tmp_path):
    model = TinyModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[16, 24, 32, 48])
    ctx = make_ctx(model, cache)

    result = carry_state(ctx)

    assert math.isfinite(result.bpb)
    assert math.isfinite(result.loss)
    assert result.bpb > 0.0
    assert result.docs_scored == 4
    assert result.tokens_scored == (16 - 1) + (24 - 1) + (32 - 1) + (48 - 1)
    assert result.raw_bytes == 4 * (16 + 24 + 32 + 48)
    assert result.hyperparams == {"decay": 1.0}


def test_carry_state_decay_one_threads_state_unchanged(tmp_path):
    """With decay=1.0 every doc N>0 must start from doc N-1's final state."""
    model = TinyModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[12, 12, 12, 12])
    ctx = make_ctx(model, cache, config={"decay": 1.0})

    carry_state(ctx)

    # First doc receives a None init; subsequent docs receive the prior final.
    assert model._init_states_log[0] is None
    assert len(model._final_states_log) == 4
    for i in range(1, 4):
        carried_in = model._init_states_log[i]
        assert carried_in is not None
        assert torch.equal(carried_in[0], model._final_states_log[i - 1])


def test_carry_state_decay_half_scales_carried_state(tmp_path):
    model = TinyModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[16, 16, 16])
    ctx = make_ctx(model, cache, config={"decay": 0.5})

    carry_state(ctx)

    assert model._init_states_log[0] is None
    for i in range(1, 3):
        carried_in = model._init_states_log[i]
        assert carried_in is not None
        expected = model._final_states_log[i - 1] * 0.5
        assert torch.allclose(carried_in[0], expected, rtol=0.0, atol=0.0)


def test_carry_state_decay_zero_matches_floor(tmp_path):
    """decay=0.0 zeros the carried state, which is bit-equivalent to reset."""
    cache = make_val_cache(tmp_path, doc_lens=[16, 24, 32])

    model_carry = TinyModel(seed=0)
    ctx_carry = make_ctx(model_carry, cache, config={"decay": 0.0})
    r_carry = carry_state(ctx_carry)

    model_reset = TinyModel(seed=0)
    ctx_reset = make_ctx(model_reset, cache)
    r_reset = score_only_reset(ctx_reset)

    assert r_carry.bpb == pytest.approx(r_reset.bpb, rel=0.0, abs=0.0)
    assert r_carry.loss == pytest.approx(r_reset.loss, rel=0.0, abs=0.0)
    assert r_carry.tokens_scored == r_reset.tokens_scored
    assert r_carry.raw_bytes == r_reset.raw_bytes


def test_carry_state_differs_from_floor(tmp_path):
    """Sanity: with decay=1.0 the carried state must actually change scoring."""
    cache = make_val_cache(tmp_path, doc_lens=[16, 24, 32, 48])

    model_carry = TinyModel(seed=0)
    ctx_carry = make_ctx(model_carry, cache)  # decay defaults to 1.0
    r_carry = carry_state(ctx_carry)

    model_reset = TinyModel(seed=0)
    ctx_reset = make_ctx(model_reset, cache)
    r_reset = score_only_reset(ctx_reset)

    assert r_carry.bpb != pytest.approx(r_reset.bpb, rel=0.0, abs=0.0)


def test_carry_state_skips_short_docs_without_resetting_carry(tmp_path):
    """A doc with token_len<2 must not nuke the carried state."""
    model = TinyModel(seed=0)
    # docs: 12, 1 (skipped), 12, 12.
    cache = make_val_cache(tmp_path, doc_lens=[12, 1, 12, 12])
    ctx = make_ctx(model, cache, config={"decay": 1.0})

    carry_state(ctx)

    # Three encode calls (the len=1 doc is skipped before encode runs).
    assert len(model._init_states_log) == 3
    # Doc 0: None init. Doc 1 (the second scored doc, original index 2):
    # carries doc-0 final. Doc 2 (original index 3): carries doc-1 final.
    assert model._init_states_log[0] is None
    assert model._init_states_log[1] is not None
    assert torch.equal(
        model._init_states_log[1][0], model._final_states_log[0]
    )
    assert torch.equal(
        model._init_states_log[2][0], model._final_states_log[1]
    )


def test_carry_state_preserves_train_mode(tmp_path):
    model = TinyModel(seed=0)
    model.train()
    cache = make_val_cache(tmp_path, doc_lens=[12, 12])
    ctx = make_ctx(model, cache)

    carry_state(ctx)

    assert model.training is True


def test_carry_state_uses_packet_encode_lane(tmp_path):
    model = TinyModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[8, 8, 8])
    ctx = make_ctx(model, cache)

    carry_state(ctx)

    assert model.forward_call_count == 0
    assert model._memory_modes_log == ["packet", "packet", "packet"]
