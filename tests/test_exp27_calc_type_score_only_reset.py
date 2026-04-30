"""Unit tests for the ``score_only_reset`` calc_type.

The tests build a tiny synthetic ``nn.Module`` that mimics the bare-SSM
model contract (``model(input_ids) -> logits`` and
``model.encode(input_ids, *, initial_states, return_final_states)``)
plus a small in-memory ``ValCache``. The production model build path is
intentionally avoided — these tests are about the calc_type's per-doc
loop, not the SSM internals.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from chaoscontrol.eval.calc_types.score_only_reset import score_only_reset
from chaoscontrol.eval.ttt_eval import CalcTypeContext
from chaoscontrol.eval_stream.val_cache import DOC_DTYPE, TOKEN_DTYPE, ValCache


VOCAB = 17
DIM = 8


class _CountingLMHead(nn.Module):
    """Linear lm_head that records how many times it has been invoked."""

    def __init__(self, dim: int, vocab: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, vocab, bias=False)
        self.call_count = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return self.linear(hidden)


class TinyModel(nn.Module):
    """Tiny stand-in for the production SSM-bearing model.

    The recurrence is intentionally trivial — what matters is that
    ``__call__`` produces the same logits as ``encode + lm_head`` so
    the contract used by the production code is mirrored exactly. No
    ``final_norm``: the production model has one, but for unit tests
    that asymmetry would break the R=1-equals-floor invariant the
    state_replay test relies on.
    """

    def __init__(
        self,
        vocab: int = VOCAB,
        dim: int = DIM,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, dim)
        self.w_in = nn.Linear(dim, dim, bias=False)
        self.w_state = nn.Linear(dim, dim, bias=False)
        self.lm_head = _CountingLMHead(dim, vocab)
        # Number of "physical layers" — for our purposes a single state.
        self.layers = (None,)
        self._init_states_log: list[list[torch.Tensor] | None] = []
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
        """Tiny SSM-ish recurrence: state' = w_state(state) + w_in(embed)."""
        # Snapshot the initial state input for inspection by tests.
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
        x = self.embed(input_ids)  # (batch, seq, dim)
        if initial_states is None:
            state = torch.zeros(batch, x.size(-1), device=x.device, dtype=x.dtype)
        else:
            state = initial_states[0].to(device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq):
            state = self.w_state(state) + self.w_in(x[:, t, :])
            outputs.append(state)
        hidden = torch.stack(outputs, dim=1)  # (batch, seq, dim)
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
    """Build a tiny in-memory ValCache with random tokens.

    Doc raw_bytes is fixed at 4 * token_len — the actual byte count
    doesn't matter for these tests, only that it is positive and stable.
    """
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


def make_ctx(model: nn.Module, val_cache: ValCache) -> CalcTypeContext:
    """Bundle the few pieces the calc_type actually reads."""
    device = torch.device("cpu")
    return CalcTypeContext(
        model=model,
        val_cache=val_cache,
        device=device,
        base_bytes_lut=torch.zeros(VOCAB, dtype=torch.long),
        has_leading_space_lut=torch.zeros(VOCAB, dtype=torch.bool),
        is_boundary_token_lut=torch.zeros(VOCAB, dtype=torch.bool),
        config={},
    )


# ---- Tests -----------------------------------------------------------------


def test_score_only_reset_returns_finite(tmp_path):
    model = TinyModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[16, 24, 32, 48])
    ctx = make_ctx(model, cache)

    result = score_only_reset(ctx)

    assert math.isfinite(result.bpb)
    assert math.isfinite(result.loss)
    assert result.bpb > 0.0
    assert result.docs_scored == 4
    assert result.tokens_scored == (16 - 1) + (24 - 1) + (32 - 1) + (48 - 1)
    assert result.raw_bytes == 4 * (16 + 24 + 32 + 48)
    assert result.hyperparams == {}


def test_score_only_reset_is_deterministic(tmp_path):
    model = TinyModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[20, 30, 40])
    ctx = make_ctx(model, cache)

    r1 = score_only_reset(ctx)
    r2 = score_only_reset(ctx)

    assert r1.bpb == pytest.approx(r2.bpb, rel=0.0, abs=0.0)
    assert r1.loss == pytest.approx(r2.loss, rel=0.0, abs=0.0)


def test_score_only_reset_skips_short_docs(tmp_path):
    model = TinyModel(seed=0)
    # token_len < 2 has no scoreable next-token target.
    cache = make_val_cache(tmp_path, doc_lens=[1, 16, 0, 24])
    ctx = make_ctx(model, cache)

    result = score_only_reset(ctx)

    assert result.docs_scored == 2  # only the >=2-token docs counted
    assert result.tokens_scored == (16 - 1) + (24 - 1)


def test_score_only_reset_resets_state_each_doc(tmp_path):
    """Fresh state per doc means encode never sees a non-None initial_states."""
    model = TinyModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[8, 8, 8])
    ctx = make_ctx(model, cache)

    score_only_reset(ctx)

    # Three docs scored, one encode call per doc, every initial_states None.
    assert len(model._init_states_log) == 3
    for entry in model._init_states_log:
        assert entry is None


def test_score_only_reset_uses_packet_encode_lane(tmp_path):
    """The floor must stay on the same packet lane as adaptive calc_types."""
    model = TinyModel(seed=0)
    cache = make_val_cache(tmp_path, doc_lens=[8, 8])
    ctx = make_ctx(model, cache)

    score_only_reset(ctx)

    assert model.forward_call_count == 0
    assert model._memory_modes_log == ["packet", "packet"]


def test_score_only_reset_preserves_train_mode(tmp_path):
    model = TinyModel(seed=0)
    model.train()
    cache = make_val_cache(tmp_path, doc_lens=[12, 12])
    ctx = make_ctx(model, cache)

    score_only_reset(ctx)

    assert model.training is True
