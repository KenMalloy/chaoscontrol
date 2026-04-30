"""Tests for ``chaoscontrol.eval.runner_dispatch.dispatch_eval_for_config``.

The dispatcher picks one of two eval paths per cell. These tests exercise
the routing decision itself and the backward-compat ``result["eval"]``
schema, using a tiny fake model + in-memory ``ValCache``. The legacy eval
is injected as a callable so we don't need to import any runner module.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import chaoscontrol.eval.calc_types  # noqa: F401  (triggers registration)
from chaoscontrol.eval.runner_dispatch import dispatch_eval_for_config
from chaoscontrol.eval_stream.val_cache import (
    DOC_DTYPE,
    TOKEN_DTYPE,
    ValCache,
)


VOCAB = 16
DIM = 4


class _TinyModel(nn.Module):
    """Minimum surface a calc_type touches: ``model(input_ids)`` returns logits;
    ``encode(...)`` returns ``(hidden, [state])`` when ``return_final_states``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.lm_head = nn.Linear(DIM, VOCAB)
        self.w_state = nn.Linear(DIM, DIM, bias=False)
        with torch.no_grad():
            self.w_state.weight.copy_(torch.eye(DIM) + 0.05 * self.w_state.weight)
        self.layers = [self]

    def encode(self, input_ids, *, initial_states=None, return_final_states=False):
        h = self.embed(input_ids)
        if initial_states is not None:
            state = initial_states[0]
        else:
            state = torch.zeros(h.size(0), DIM, device=h.device, dtype=h.dtype)
        outs = []
        for t in range(h.size(1)):
            state = self.w_state(state) + h[:, t, :]
            outs.append(state)
        hidden = torch.stack(outs, dim=1)
        if return_final_states:
            return hidden, [state.detach()]
        return hidden

    def forward(self, input_ids, *, initial_states=None):
        if initial_states is not None:
            hidden = self.encode(input_ids, initial_states=initial_states)
        else:
            hidden = self.encode(input_ids)
        return self.lm_head(hidden)


def _tiny_val_cache(tmp_path: Path, n_docs: int = 4, seed: int = 0) -> ValCache:
    rng = np.random.default_rng(seed)
    chunks = []
    docs = []
    cursor = 0
    for d in range(n_docs):
        n = int(rng.integers(8, 24))
        chunk = rng.integers(0, VOCAB, size=n, dtype=np.int64).astype(TOKEN_DTYPE)
        chunks.append(chunk)
        docs.append((d, cursor, n, n * 4))
        cursor += n
    return ValCache(
        cache_dir=tmp_path,
        manifest={"vocab": VOCAB, "synthetic": True},
        tokens=np.concatenate(chunks),
        docs=np.array(docs, dtype=DOC_DTYPE),
    )


@pytest.fixture
def tiny_setup(tmp_path):
    torch.manual_seed(0)
    return {
        "model": _TinyModel(),
        "val_cache": _tiny_val_cache(tmp_path),
        "val_tokens": torch.zeros(64, dtype=torch.long),
        "eval_starts": [0, 16, 32],
        "batch_size": 1,
        "seq_len": 8,
        "device": torch.device("cpu"),
        "base_bytes_lut": torch.zeros(VOCAB),
        "has_leading_space_lut": torch.zeros(VOCAB, dtype=torch.bool),
        "is_boundary_token_lut": torch.zeros(VOCAB, dtype=torch.bool),
    }


# ---- legacy fallback path ---------------------------------------------------


def test_dispatch_falls_back_to_legacy_when_no_calc_types_key(tiny_setup):
    sentinel = {"bpb": 1.234, "loss": 0.5, "from": "legacy"}
    captured: dict = {}

    def legacy(model, **kwargs):
        captured.update(kwargs)
        captured["model_passed"] = model is tiny_setup["model"]
        return sentinel

    out = dispatch_eval_for_config(
        config={},  # no calc_types
        legacy_evaluate_fn=legacy,
        **tiny_setup,
    )
    assert out is sentinel
    assert captured["model_passed"]
    assert captured["tokens"] is tiny_setup["val_tokens"]
    assert captured["eval_starts"] == [0, 16, 32]


def test_dispatch_falls_back_when_calc_types_is_empty(tiny_setup):
    """Empty list should be treated the same as missing — legacy path."""
    sentinel = {"bpb": 9.0, "loss": 6.0}

    def legacy(model, **kwargs):
        return sentinel

    out = dispatch_eval_for_config(
        config={"calc_types": []},
        legacy_evaluate_fn=legacy,
        **tiny_setup,
    )
    assert out is sentinel


def test_dispatch_falls_back_works_with_val_cache_none(tiny_setup):
    """Legacy path must not require a ValCache."""
    sentinel = {"bpb": 1.0, "loss": 0.5}
    setup = {**tiny_setup, "val_cache": None}

    def legacy(model, **kwargs):
        return sentinel

    out = dispatch_eval_for_config(
        config={},
        legacy_evaluate_fn=legacy,
        **setup,
    )
    assert out is sentinel


# ---- calc_types path --------------------------------------------------------


def test_dispatch_uses_calc_types_path_with_score_only_reset_headline(tiny_setup):
    def legacy(model, **kwargs):
        raise AssertionError("legacy path must not be called when calc_types set")

    out = dispatch_eval_for_config(
        config={
            "calc_types": ["score_only_reset", "carry_state"],
            "calc_type_configs": {"carry_state": {"decay": 1.0}},
        },
        legacy_evaluate_fn=legacy,
        **tiny_setup,
    )
    assert "calc_types" in out
    assert set(out["calc_types"].keys()) == {"score_only_reset", "carry_state"}
    # Headline must be score_only_reset's bpb/loss.
    assert out["bpb"] == out["calc_types"]["score_only_reset"]["bpb"]
    assert out["loss"] == out["calc_types"]["score_only_reset"]["loss"]
    assert out["headline_calc_type"] == "score_only_reset"
    assert math.isfinite(out["bpb"])


def test_dispatch_uses_first_calc_type_when_score_only_reset_absent(tiny_setup):
    """If the floor calc_type wasn't requested, headline = first in list (insertion order)."""
    def legacy(model, **kwargs):
        raise AssertionError("legacy path must not be called when calc_types set")

    out = dispatch_eval_for_config(
        config={
            "calc_types": ["carry_state"],
            "calc_type_configs": {"carry_state": {"decay": 1.0}},
        },
        legacy_evaluate_fn=legacy,
        **tiny_setup,
    )
    assert out["headline_calc_type"] == "carry_state"
    assert out["bpb"] == out["calc_types"]["carry_state"]["bpb"]


def test_dispatch_uses_configured_headline_calc_type(tiny_setup):
    def legacy(model, **kwargs):
        raise AssertionError("legacy path must not be called when calc_types set")

    out = dispatch_eval_for_config(
        config={
            "calc_types": ["score_only_reset", "carry_state"],
            "headline_calc_type": "carry_state",
            "calc_type_configs": {"carry_state": {"decay": 1.0}},
        },
        legacy_evaluate_fn=legacy,
        **tiny_setup,
    )
    assert out["headline_calc_type"] == "carry_state"
    assert out["bpb"] == out["calc_types"]["carry_state"]["bpb"]


def test_dispatch_rejects_headline_not_in_calc_types(tiny_setup):
    def legacy(model, **kwargs):
        raise AssertionError("legacy path must not be called when calc_types set")

    with pytest.raises(ValueError, match="headline_calc_type"):
        dispatch_eval_for_config(
            config={
                "calc_types": ["score_only_reset"],
                "headline_calc_type": "adaptive_carry",
            },
            legacy_evaluate_fn=legacy,
            **tiny_setup,
        )


def test_dispatch_raises_when_calc_types_set_but_val_cache_none(tiny_setup):
    setup = {**tiny_setup, "val_cache": None}

    def legacy(model, **kwargs):
        raise AssertionError("legacy path must not be called")

    with pytest.raises(ValueError, match="val_cache is None"):
        dispatch_eval_for_config(
            config={"calc_types": ["score_only_reset"]},
            legacy_evaluate_fn=legacy,
            **setup,
        )


def test_dispatch_raises_on_nonfinite_calc_type_result(tiny_setup):
    """A calc_type that returns NaN must not slip through silently."""
    from chaoscontrol.eval.ttt_eval import (
        CALC_TYPE_REGISTRY,
        CalcTypeResult,
        register_calc_type,
    )

    name = "__test_nonfinite_calc_type__"

    @register_calc_type(name)
    def _nan_calc_type(ctx) -> CalcTypeResult:
        return CalcTypeResult(
            bpb=float("nan"),
            loss=float("nan"),
            docs_scored=0,
            tokens_scored=0,
            raw_bytes=0,
            hyperparams={},
        )

    try:
        def legacy(model, **kwargs):
            raise AssertionError("legacy path must not be called when calc_types set")

        with pytest.raises(RuntimeError, match="non-finite"):
            dispatch_eval_for_config(
                config={"calc_types": [name]},
                legacy_evaluate_fn=legacy,
                **tiny_setup,
            )
    finally:
        CALC_TYPE_REGISTRY.pop(name, None)
        from chaoscontrol.eval.ttt_eval import CALC_TYPE_METADATA
        CALC_TYPE_METADATA.pop(name, None)
