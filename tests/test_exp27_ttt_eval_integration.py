"""Integration tests for ``evaluate_with_calc_types`` dispatching all
four registered calc_types over a shared tiny model + tiny ValCache.

Per-calc_type unit tests already cover semantics in isolation; this
suite covers the dispatcher contract:
- calling all four calc_types at once returns a dict with one entry per name
- the returned dict preserves insertion order of the requested calc_types
- per-calc_type ``calc_type_configs`` route to the corresponding body
- the result entries carry the schema fields the orchestrator expects
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import chaoscontrol.eval.calc_types  # noqa: F401  (triggers registration)
from chaoscontrol.eval.ttt_eval import (
    CALC_TYPE_REGISTRY,
    evaluate_with_calc_types,
)
from chaoscontrol.eval_stream.val_cache import (
    DOC_DTYPE,
    TOKEN_DTYPE,
    ValCache,
)


class _TinyModel(nn.Module):
    """Single-layer SSM-shaped fake model.

    Mirrors the minimum surface every calc_type touches:
    ``model(input_ids)`` returns logits; ``model.encode(input_ids,
    initial_states=..., return_final_states=True)`` returns
    ``(hidden, [state])`` where ``state`` is the post-doc recurrent
    summary. ``model.lm_head(hidden)`` projects hidden to vocab.

    The recurrence is non-contractive (identity-plus-small-perturbation)
    so state-replay tests can detect a real R>1 effect.
    """

    def __init__(self, vocab: int = 16, dim: int = 4) -> None:
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.embed = nn.Embedding(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab)
        self.w_state = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.w_state.weight.copy_(
                torch.eye(dim) + 0.05 * self.w_state.weight
            )
        self.layers = [self]  # encode contract: len(initial_states) == len(layers)

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ):
        h = self.embed(input_ids)  # (B, T, D)
        if initial_states is not None:
            state = initial_states[0]
        else:
            state = torch.zeros(h.size(0), self.dim, device=h.device, dtype=h.dtype)
        outs = []
        for t in range(h.size(1)):
            state = self.w_state(state) + h[:, t, :]
            outs.append(state)
        hidden = torch.stack(outs, dim=1)
        if return_final_states:
            return hidden, [state.detach()]
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if initial_states is not None:
            hidden = self.encode(input_ids, initial_states=initial_states)
        else:
            hidden = self.encode(input_ids)
        return self.lm_head(hidden)


def _tiny_val_cache(tmp_path: Path, *, vocab: int = 16, n_docs: int = 4, seed: int = 0) -> ValCache:
    rng = np.random.default_rng(seed)
    docs_meta = []
    token_chunks = []
    cursor = 0
    for d in range(n_docs):
        token_len = int(rng.integers(low=8, high=24))
        chunk = rng.integers(low=0, high=vocab, size=token_len, dtype=np.int64)
        token_chunks.append(chunk.astype(TOKEN_DTYPE))
        docs_meta.append((d, cursor, token_len, token_len * 4))  # raw_bytes ~ doc len * 4
        cursor += token_len
    tokens = np.concatenate(token_chunks).astype(TOKEN_DTYPE)
    docs = np.array(docs_meta, dtype=DOC_DTYPE)
    return ValCache(
        cache_dir=tmp_path,
        manifest={"vocab": vocab, "synthetic": True},
        tokens=tokens,
        docs=docs,
    )


@pytest.fixture
def tiny_setup(tmp_path):
    torch.manual_seed(0)
    model = _TinyModel(vocab=16, dim=4)
    cache = _tiny_val_cache(tmp_path, vocab=16, n_docs=4, seed=0)
    device = torch.device("cpu")
    base_lut = torch.zeros(16)
    has_space_lut = torch.zeros(16, dtype=torch.bool)
    is_boundary_lut = torch.zeros(16, dtype=torch.bool)
    return {
        "model": model,
        "val_cache": cache,
        "device": device,
        "base_bytes_lut": base_lut,
        "has_leading_space_lut": has_space_lut,
        "is_boundary_token_lut": is_boundary_lut,
    }


def test_dispatcher_runs_all_three_calc_types(tiny_setup) -> None:
    out = evaluate_with_calc_types(
        **tiny_setup,
        calc_types=[
            "score_only_reset",
            "carry_state",
            "dreamworld_eval",
        ],
        calc_type_configs={
            "dreamworld_eval": {"K": 2, "L": 4, "lr": 1e-3, "steps": 1, "prefix_len": 4},
        },
    )
    assert set(out.keys()) == {
        "score_only_reset",
        "carry_state",
        "dreamworld_eval",
    }
    for name, entry in out.items():
        assert "bpb" in entry, f"{name} missing bpb"
        assert "loss" in entry, f"{name} missing loss"
        assert "docs_scored" in entry
        assert "tokens_scored" in entry
        assert "raw_bytes" in entry
        assert "hyperparams" in entry
        assert isinstance(entry["bpb"], float)
        assert torch.isfinite(torch.tensor(entry["bpb"])).item(), f"{name} bpb non-finite"


def test_dispatcher_preserves_request_order(tiny_setup) -> None:
    requested = ["dreamworld_eval", "score_only_reset", "carry_state"]
    out = evaluate_with_calc_types(
        **tiny_setup,
        calc_types=requested,
        calc_type_configs={
            "dreamworld_eval": {"K": 2, "L": 4, "lr": 1e-3, "steps": 1, "prefix_len": 4},
        },
    )
    assert list(out.keys()) == requested


def test_dispatcher_routes_per_calc_type_configs(tiny_setup) -> None:
    out_no_decay = evaluate_with_calc_types(
        **tiny_setup,
        calc_types=["carry_state"],
        calc_type_configs={"carry_state": {"decay": 1.0}},
    )
    out_half_decay = evaluate_with_calc_types(
        **tiny_setup,
        calc_types=["carry_state"],
        calc_type_configs={"carry_state": {"decay": 0.5}},
    )
    assert out_no_decay["carry_state"]["hyperparams"]["decay"] == 1.0
    assert out_half_decay["carry_state"]["hyperparams"]["decay"] == 0.5
    # decay 1.0 vs 0.5 must produce different aggregate BPBs (otherwise
    # the config wasn't actually consulted).
    assert (
        out_no_decay["carry_state"]["bpb"]
        != pytest.approx(out_half_decay["carry_state"]["bpb"])
    )


def test_dispatcher_unknown_calc_type_lists_registered_names(tiny_setup) -> None:
    with pytest.raises(ValueError, match="unknown calc_type"):
        evaluate_with_calc_types(
            **tiny_setup,
            calc_types=["score_only_reset", "no_such_calc_type"],
            calc_type_configs={},
        )


def test_dispatcher_idempotent_on_repeated_call(tiny_setup) -> None:
    """Running the dispatcher twice with the same inputs returns matching
    BPB on the deterministic calc_types (sanity that no calc_type
    permanently mutates the model)."""
    cfg = {
        "dreamworld_eval": {
            "K": 2, "L": 4, "lr": 1e-3, "steps": 1,
            "prefix_len": 4, "per_doc_reset": True,
        },
    }
    out_a = evaluate_with_calc_types(
        **tiny_setup,
        calc_types=["score_only_reset", "carry_state"],
        calc_type_configs=cfg,
    )
    out_b = evaluate_with_calc_types(
        **tiny_setup,
        calc_types=["score_only_reset", "carry_state"],
        calc_type_configs=cfg,
    )
    for name in ("score_only_reset", "carry_state"):
        assert out_a[name]["bpb"] == pytest.approx(out_b[name]["bpb"], abs=1e-9), name


def test_registry_holds_three_canonical_calc_types() -> None:
    """The registry should hold (at least) the three canonical calc_types.
    More may be present from other tests' ad-hoc registrations; the three
    exp27 names must always be there. state_replay_within_doc was removed
    for causality reasons and must not reappear."""
    expected = {
        "score_only_reset",
        "carry_state",
        "dreamworld_eval",
    }
    assert expected <= set(CALC_TYPE_REGISTRY)
    assert "state_replay_within_doc" not in CALC_TYPE_REGISTRY
