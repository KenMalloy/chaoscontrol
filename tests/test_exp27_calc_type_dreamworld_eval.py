"""Unit tests for ``chaoscontrol.eval.calc_types.dreamworld_eval``.

A focused-fixture test suite for the per-doc dream-rollout calc_type.
We use a minimal ``nn.Module`` that exposes the surface
``dreamworld_eval`` actually touches:

- ``model.encode(input_ids, *, memory_mode="packet", initial_states=None,
  return_final_states=True)``
  → ``(hidden, [state])``
- ``model.lm_head`` and ``model.final_norm`` — present so the symbol
  contract matches even though Dreamworld now calls ``lm_head`` directly.

The synthetic ValCache is built by hand from ``ValCache``'s dataclass
constructor; we do not exercise the on-disk write/load path.
"""
from __future__ import annotations

from pathlib import Path

import math

import numpy as np
import pytest
import torch
from torch import nn

from chaoscontrol.eval.calc_types.dreamworld_eval import dreamworld_eval
from chaoscontrol.eval.ttt_eval import CalcTypeContext, CalcTypeResult
from chaoscontrol.eval_stream.val_cache import DOC_DTYPE, TOKEN_DTYPE, ValCache


# ---- synthetic model -------------------------------------------------------


class TinyRecurrentLM(nn.Module):
    """One-layer recurrent stub that mimics the API ``dreamworld_eval`` needs.

    ``encode`` returns ``(hidden, [state])`` with state = running-sum
    over the embedded inputs (with a learnable mixing matrix). The
    "recurrence" is differentiable, so SGD on dream loss has a real
    parameter gradient signal.
    """

    def __init__(self, vocab: int = 32, dim: int = 8) -> None:
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.embed = nn.Embedding(vocab, dim)
        self.mix = nn.Linear(dim, dim, bias=False)
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.memory_modes: list[str] = []

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        memory_mode: str = "packet",
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        self.memory_modes.append(memory_mode)
        x = self.embed(input_ids.to(torch.long))
        x = self.mix(x)
        b, t, d = x.shape
        if initial_states is not None:
            state = initial_states[0]
            if state.shape[0] != b:
                raise ValueError(
                    f"initial state batch {state.shape[0]} != input batch {b}"
                )
        else:
            state = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        outs = []
        cur = state
        for i in range(t):
            cur = cur + x[:, i, :]
            outs.append(cur)
        hidden = torch.stack(outs, dim=1)
        if return_final_states:
            return hidden, [cur]
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:  # pragma: no cover
        raise AssertionError("dreamworld_eval must use packet-clean encode")


# ---- synthetic ValCache fixture --------------------------------------------


def _make_val_cache(
    *,
    num_docs: int = 3,
    tokens_per_doc: int = 64,
    vocab: int = 32,
    seed: int = 0,
) -> ValCache:
    rng = np.random.default_rng(seed)
    all_tokens: list[int] = []
    rows: list[tuple[int, int, int, int]] = []
    offset = 0
    for doc_id in range(num_docs):
        toks = rng.integers(low=0, high=vocab, size=tokens_per_doc, dtype=np.int64)
        rows.append((doc_id, offset, tokens_per_doc, tokens_per_doc * 4))
        all_tokens.extend(int(t) for t in toks)
        offset += tokens_per_doc
    tokens = np.asarray(all_tokens, dtype=TOKEN_DTYPE)
    docs = np.asarray(rows, dtype=DOC_DTYPE)
    manifest = {
        "schema_version": 1,
        "num_docs": int(docs.shape[0]),
        "total_tokens": int(tokens.shape[0]),
        "total_raw_bytes": int(docs["raw_bytes"].sum()),
    }
    return ValCache(
        cache_dir=Path("/tmp/exp27-fixture-not-on-disk"),
        manifest=manifest,
        tokens=tokens,
        docs=docs,
    )


def _make_ctx(
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
        # The current dreamworld_eval implementation does not consume
        # these LUTs, but the contract requires the field; placeholders
        # are fine and document that intent.
        base_bytes_lut=torch.zeros(1),
        has_leading_space_lut=torch.zeros(1),
        is_boundary_token_lut=torch.zeros(1),
        config=dict(config or {}),
    )


# ---- tests -----------------------------------------------------------------


def test_returns_finite_bpb_and_loss():
    torch.manual_seed(0)
    model = TinyRecurrentLM(vocab=32, dim=8)
    val_cache = _make_val_cache(num_docs=3, tokens_per_doc=48, vocab=32)
    ctx = _make_ctx(
        model,
        val_cache,
        config={"K": 2, "L": 8, "lr": 1e-3, "steps": 1, "prefix_len": 4},
    )
    result = dreamworld_eval(ctx)
    assert isinstance(result, CalcTypeResult)
    assert math.isfinite(result.bpb)
    assert math.isfinite(result.loss)
    assert result.docs_scored == 3
    assert result.tokens_scored > 0
    assert result.raw_bytes > 0
    assert set(model.memory_modes) == {"packet"}


def test_per_doc_reset_keeps_params_bit_equal_at_start_of_each_doc():
    torch.manual_seed(1)
    model = TinyRecurrentLM(vocab=16, dim=4)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    val_cache = _make_val_cache(num_docs=2, tokens_per_doc=32, vocab=16)
    ctx = _make_ctx(
        model,
        val_cache,
        config={
            "K": 2,
            "L": 4,
            "lr": 1e-2,
            "steps": 1,
            "prefix_len": 4,
            "per_doc_reset": True,
        },
    )
    dreamworld_eval(ctx)
    # After the calc, with per_doc_reset=True, every parameter must
    # equal its pre-calc value.
    for name, p in model.named_parameters():
        assert torch.equal(p, initial[name]), (
            f"per_doc_reset=True did not restore {name!r}"
        )


def test_per_doc_reset_false_is_rejected():
    """Continual mode is order-sensitive but the calc_type's metadata
    advertises ``requires_source_order=False``. Allowing per_doc_reset=False
    silently violates Param Golf order semantics, so the calc_type
    refuses the unsafe path. If continual eval is ever wanted, it must
    be a separate calc_type registered with the source-order flag."""
    torch.manual_seed(2)
    model = TinyRecurrentLM(vocab=16, dim=4)
    val_cache = _make_val_cache(num_docs=1, tokens_per_doc=32, vocab=16)
    ctx = _make_ctx(
        model,
        val_cache,
        config={
            "K": 4,
            "L": 8,
            "lr": 1e-1,
            "steps": 1,
            "prefix_len": 4,
            "per_doc_reset": False,
        },
    )
    with pytest.raises(ValueError, match="per_doc_reset=False is not supported"):
        dreamworld_eval(ctx)


def test_hyperparams_round_trip():
    torch.manual_seed(3)
    model = TinyRecurrentLM(vocab=16, dim=4)
    val_cache = _make_val_cache(num_docs=1, tokens_per_doc=48, vocab=16)
    ctx = _make_ctx(
        model,
        val_cache,
        config={
            "K": 4,
            "L": 32,
            "lr": 5e-4,
            "steps": 2,
            "prefix_len": 8,
            "per_doc_reset": True,
            "dream_target_mode": "argmax",
            "dream_temperature": 1.0,
        },
    )
    result = dreamworld_eval(ctx)
    hp = result.hyperparams
    assert hp["K"] == 4
    assert hp["L"] == 32
    assert hp["lr"] == pytest.approx(5e-4)
    assert hp["steps"] == 2
    assert hp["prefix_len"] == 8
    assert hp["per_doc_reset"] is True
    assert hp["dream_target_mode"] == "argmax"


def test_snapshot_restore_is_exhaustive():
    """Mutate every parameter externally; restore must zero the diff.

    This guards against snapshot/restore drift caused by missing param
    names or by reassignment-vs-copy_ choice. We snapshot, mutate
    every param, then call ``_restore_params`` directly; bit-equality
    must hold for every param afterwards.
    """
    from chaoscontrol.eval.calc_types.dreamworld_eval import (
        _restore_params,
        _snapshot_params,
    )

    torch.manual_seed(4)
    model = TinyRecurrentLM(vocab=16, dim=4)
    snap = _snapshot_params(model)
    # Mutate every param.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    # Verify mutation actually fired before restore (sanity).
    n_diff = sum(
        1
        for n, p in model.named_parameters()
        if not torch.equal(p, snap[n])
    )
    assert n_diff == sum(1 for _ in model.parameters())
    _restore_params(model, snap)
    for n, p in model.named_parameters():
        assert torch.equal(p, snap[n]), f"restore did not return {n!r}"


def test_softmax_target_mode_runs_and_finite():
    torch.manual_seed(5)
    model = TinyRecurrentLM(vocab=32, dim=8)
    val_cache = _make_val_cache(num_docs=2, tokens_per_doc=48, vocab=32)
    ctx = _make_ctx(
        model,
        val_cache,
        config={
            "K": 2,
            "L": 8,
            "lr": 1e-3,
            "steps": 1,
            "prefix_len": 4,
            "dream_target_mode": "softmax",
            "dream_temperature": 1.0,
        },
    )
    result = dreamworld_eval(ctx)
    assert math.isfinite(result.bpb)
    assert math.isfinite(result.loss)
    assert result.docs_scored == 2


def test_softmax_and_argmax_produce_different_loss():
    """Same seed + same config except mode → different aggregate loss.

    If they were identical, the softmax branch would silently be doing
    argmax. We use a fresh model in both arms with the same init
    seed so the only difference is the rollout-target sampling.
    """
    val_cache = _make_val_cache(num_docs=2, tokens_per_doc=48, vocab=32, seed=7)

    def _run(mode: str) -> float:
        torch.manual_seed(11)
        model = TinyRecurrentLM(vocab=32, dim=8)
        ctx = _make_ctx(
            model,
            val_cache,
            config={
                "K": 4,
                "L": 16,
                "lr": 1e-2,
                "steps": 1,
                "prefix_len": 4,
                "dream_target_mode": mode,
                "dream_temperature": 1.0,
                # per_doc_reset stays True (the only supported mode);
                # the modes still produce different scoring losses
                # because doc-scoring happens after the SGD step and
                # before the param restore, so the post-SGD params
                # differ between argmax-driven and softmax-driven dream.
            },
        )
        return float(dreamworld_eval(ctx).loss)

    loss_argmax = _run("argmax")
    loss_softmax = _run("softmax")
    assert loss_argmax != loss_softmax, (
        "argmax and softmax modes produced identical loss; one branch may be dead"
    )


def test_invalid_target_mode_raises():
    torch.manual_seed(6)
    model = TinyRecurrentLM(vocab=16, dim=4)
    val_cache = _make_val_cache(num_docs=1, tokens_per_doc=32, vocab=16)
    ctx = _make_ctx(
        model,
        val_cache,
        config={"dream_target_mode": "no_such_mode"},
    )
    with pytest.raises(ValueError, match="dream_target_mode"):
        dreamworld_eval(ctx)


def test_runs_under_no_grad_outer_context():
    """Calc must work even when the dispatcher is in a no_grad scope.

    The contract requires the calc_type to manage its own grad scope
    when ``requires_grad=True``; we wrap the whole call in
    ``torch.no_grad()``. If the inner ``torch.enable_grad()`` weren't
    firing, ``loss.backward()`` would raise ``RuntimeError`` (element
    0 of tensors does not require grad), so completing the call without
    exception is sufficient evidence that backward actually ran.
    """
    torch.manual_seed(7)
    model = TinyRecurrentLM(vocab=16, dim=4)
    val_cache = _make_val_cache(num_docs=1, tokens_per_doc=32, vocab=16)
    ctx = _make_ctx(
        model,
        val_cache,
        config={
            "K": 2,
            "L": 4,
            "lr": 1e-2,
            "steps": 1,
            "prefix_len": 4,
            # per_doc_reset stays at default True (the only supported mode).
        },
    )
    with torch.no_grad():
        result = dreamworld_eval(ctx)
    assert math.isfinite(result.bpb)
    assert math.isfinite(result.loss)
