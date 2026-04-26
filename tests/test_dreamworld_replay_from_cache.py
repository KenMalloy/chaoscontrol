"""Tests for ``dreamworld_replay_from_cache_entry`` (Phase 3.1).

The function is the bridge between the curated episodic cache and the
existing Dreamworld replay backward path. Given an occupied cache slot
and the live model, it builds a synthetic input from the cached value
tokens, runs forward → CE → backward, and accumulates the resulting
gradients into ``param.grad`` so the runner's existing all-reduce
sweeps them up.

Tests:

* ``test_dreamworld_replay_from_cache_entry_returns_loss_and_grads``
  — small CPU model with a hand-built cache entry; one call produces
  a numeric loss and accumulates non-zero grads on the model params.
* ``test_dreamworld_replay_from_cache_entry_handles_missing_slot``
  — calling on an unoccupied slot is a no-op (returns None) so the
  runner's per-step drain can race against eviction without crashing.
* ``test_replay_from_cache_returns_documented_diagnostic_fields`` —
  return dict carries the keys the diagnostic log row needs (loss,
  grad norm, grad cosines, utility signal pre-/transformed). NaN is
  acceptable for Phase 1 cosines.
* ``test_replay_from_cache_accumulates_into_existing_grads`` — if
  ``param.grad`` is non-zero before the call (e.g. residual replay
  from earlier in the step), the function adds to it rather than
  replacing.
"""
from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from chaoscontrol.optim.episodic_cache import EpisodicCache


REPO = Path(__file__).resolve().parents[1]
DREAMWORLD_PATH = REPO / "experiments" / "23_fast_path" / "dreamworld.py"


def _load_dreamworld():
    spec = importlib.util.spec_from_file_location("exp23_dreamworld", DREAMWORLD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _TinyTokenModel(nn.Module):
    """Token-in / hidden-out model with a real norm + lm_head.

    Mirrors the production model interface (``encode``, ``final_norm``,
    ``lm_head``) closely enough that the LM-head backward path runs
    end-to-end without needing the SSM core.
    """

    def __init__(self, vocab: int = 16, dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def encode(
        self,
        inputs: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # The cache-driven replay path doesn't carry SSM state — it
        # synthesizes a fresh input from the value_tok_ids span — so
        # ``initial_states`` is unused here.
        h = self.proj(self.embed(inputs.to(torch.long)))
        if return_final_states:
            return h, []
        return h


def _populate_cache_slot(
    cache: EpisodicCache,
    *,
    slot_value_anchor: int,
    span_tokens: torch.Tensor,
    key_fp: int,
    write_step: int,
) -> int:
    """Append one entry into the cache and return the slot index."""
    return cache.append(
        key_fp=key_fp,
        key_rep=torch.zeros(cache.key_rep_dim, dtype=torch.float32),
        value_tok_ids=span_tokens.to(dtype=torch.int64),
        value_anchor_id=slot_value_anchor,
        current_step=write_step,
        embedding_version=0,
    )


class TestDreamworldReplayFromCache(unittest.TestCase):

    def setUp(self) -> None:
        self.mod = _load_dreamworld()
        torch.manual_seed(0)

    def test_dreamworld_replay_from_cache_entry_returns_loss_and_grads(self):
        """Replay backward: numeric loss + non-zero grads on the model."""
        model = _TinyTokenModel(vocab=16, dim=8)
        cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
        # span_length=4 → enough tokens to slice off targets.
        slot = _populate_cache_slot(
            cache,
            slot_value_anchor=2,
            span_tokens=torch.tensor([3, 5, 7, 9], dtype=torch.int64),
            key_fp=42,
            write_step=10,
        )
        # Pre-clear grads so the post-call values are unambiguous.
        model.zero_grad(set_to_none=True)
        out = self.mod.dreamworld_replay_from_cache_entry(
            model=model,
            cache=cache,
            slot=slot,
            current_step=11,
            weight=1.0,
        )
        self.assertIsNotNone(out)
        self.assertIn("replay_loss", out)
        self.assertIsInstance(out["replay_loss"], float)
        self.assertTrue(math.isfinite(out["replay_loss"]))
        # At least one parameter ended up with a non-zero grad.
        any_nonzero = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                any_nonzero = True
                break
        self.assertTrue(
            any_nonzero,
            "expected replay backward to accumulate non-zero grads "
            "into at least one model parameter",
        )

    def test_dreamworld_replay_from_cache_entry_handles_missing_slot(self):
        """Unoccupied slot → return None, no grads touched."""
        model = _TinyTokenModel()
        cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
        # Slot 0 is unoccupied (cache is empty).
        for p in model.parameters():
            self.assertIsNone(p.grad)
        out = self.mod.dreamworld_replay_from_cache_entry(
            model=model,
            cache=cache,
            slot=0,
            current_step=1,
            weight=1.0,
        )
        self.assertIsNone(out)
        # Grads stayed None — the call was a no-op.
        for p in model.parameters():
            self.assertIsNone(p.grad)

    def test_replay_from_cache_returns_documented_diagnostic_fields(self):
        """The returned dict carries the keys the runner needs to write
        a per-replay diagnostic log row (Decision 0.9). Phase 1 logs
        NaN for the three replay-grad cosines (Decision 0.10's Phase 1
        simplification — no live rare-grad EMA in scope), but the keys
        must still be present so the schema-pinned writer accepts the
        row."""
        model = _TinyTokenModel()
        cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
        slot = _populate_cache_slot(
            cache,
            slot_value_anchor=1,
            span_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            key_fp=7,
            write_step=0,
        )
        model.zero_grad(set_to_none=True)
        out = self.mod.dreamworld_replay_from_cache_entry(
            model=model,
            cache=cache,
            slot=slot,
            current_step=1,
            weight=1.0,
        )
        for col in (
            "replay_loss",
            "replay_grad_norm",
            "replay_grad_cos_common",
            "replay_grad_cos_rare",
            "replay_grad_cos_total",
            "utility_signal_raw",
            "utility_signal_transformed",
        ):
            self.assertIn(col, out, msg=f"missing diagnostic field: {col}")
        # Phase 1 simplification: cosines + raw signal NaN, transformed
        # is the deterministic 0.0 fallback so update_utility never sees NaN.
        self.assertTrue(math.isnan(out["replay_grad_cos_common"]))
        self.assertTrue(math.isnan(out["replay_grad_cos_rare"]))
        self.assertTrue(math.isnan(out["replay_grad_cos_total"]))
        self.assertTrue(math.isnan(out["utility_signal_raw"]))
        self.assertEqual(out["utility_signal_transformed"], 0.0)
        # Non-zero replay_grad_norm proves we measured the grads after
        # backward.
        self.assertGreater(out["replay_grad_norm"], 0.0)

    def test_replay_from_cache_accumulates_into_existing_grads(self):
        """If grads are non-zero pre-call (e.g. another replay item
        already fired this step), the function ADDS rather than
        replaces. The runner's all-reduce sums replay-only grads from
        every drained tagged-replay item, so additive accumulation is
        load-bearing.

        Pin it strictly: compute the replay-only grad first (from a
        zero baseline), then re-run the same replay from a 0.5
        baseline and verify at least one param's post-grad equals
        ``0.5 + replay_only_grad`` to float tolerance. A "replace"
        implementation would yield ``replay_only_grad`` alone and the
        equality would fail.
        """
        cache = EpisodicCache(capacity=4, span_length=4, key_rep_dim=8)
        # Two parallel models with identical init so the replay-only
        # grad transfers cleanly to the additive baseline run.
        torch.manual_seed(31)
        model_replay_only = _TinyTokenModel()
        torch.manual_seed(31)
        model_additive = _TinyTokenModel()

        slot = _populate_cache_slot(
            cache,
            slot_value_anchor=2,
            span_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            key_fp=99,
            write_step=0,
        )

        # Run 1: zero baseline → grads ARE replay_only_grad.
        model_replay_only.zero_grad(set_to_none=True)
        self.mod.dreamworld_replay_from_cache_entry(
            model=model_replay_only,
            cache=cache,
            slot=slot,
            current_step=1,
            weight=1.0,
        )
        replay_only = {
            name: p.grad.clone() if p.grad is not None else None
            for name, p in model_replay_only.named_parameters()
        }

        # Run 2: 0.5 baseline → grads SHOULD be 0.5 + replay_only.
        for p in model_additive.parameters():
            p.grad = torch.full_like(p, 0.5)
        self.mod.dreamworld_replay_from_cache_entry(
            model=model_additive,
            cache=cache,
            slot=slot,
            current_step=1,
            weight=1.0,
        )

        any_strict_match = False
        for name, p in model_additive.named_parameters():
            ref = replay_only.get(name)
            if ref is None or p.grad is None:
                continue
            expected = 0.5 + ref
            if torch.allclose(p.grad, expected, atol=1e-5, rtol=1e-4):
                any_strict_match = True
                break
        self.assertTrue(
            any_strict_match,
            "expected replay backward to be ADDITIVE: at least one "
            "param's post-grad must equal (baseline + replay_only_grad). "
            "If this fails, the function may be REPLACING instead "
            "of accumulating, which would corrupt multi-replay drains "
            "in the runner's per-step body.",
        )

    def test_compute_utility_signal_clamps_negative_to_zero(self):
        """Per Decision 0.10's clamp rule: negative cosines transform
        to 0.0. Pin every documented case (negative, positive,
        boundary, NaN) so the contract can't drift."""
        cases = [
            (-1.0, 0.0),
            (-0.5, 0.0),
            (0.0, 0.0),
            (0.25, 0.25),
            (1.0, 1.0),
        ]
        for raw_in, expected_t in cases:
            raw_out, transformed = self.mod.compute_utility_signal(raw_in)
            self.assertEqual(raw_out, raw_in)
            self.assertEqual(transformed, expected_t)
        # NaN raw (Phase 1 simplification) → deterministic 0.0
        # transformed so update_utility never sees NaN.
        raw_nan, t_nan = self.mod.compute_utility_signal(float("nan"))
        self.assertTrue(math.isnan(raw_nan))
        self.assertEqual(t_nan, 0.0)

    def test_utility_update_clamps_negative_to_zero(self):
        """End-to-end pinning of test 7 from the Phase 3.1+3.2 spec:
        a replay event with ``replay_grad_cos_rare = -0.5`` must
        contribute ``utility_signal_transformed = 0.0`` to the cache's
        utility EMA. With the EMA's ``decay * cur + (1-decay) * 0.0``
        update, the slot's utility decays toward 0 over time but never
        flips negative — preserving the ``score = cosine × utility_u``
        ordering invariant."""
        cache = EpisodicCache(capacity=2, span_length=4, key_rep_dim=8)
        slot = _populate_cache_slot(
            cache,
            slot_value_anchor=0,
            span_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            key_fp=99,
            write_step=0,
        )
        # Cache append seeds utility_u to 1.0 (per Decision 0.2's
        # cold-start fix); a negative-cosine replay event must NOT
        # drive utility below zero.
        utility_pre = float(cache.utility_u[slot].item())
        self.assertEqual(utility_pre, 1.0)

        # Apply the same clamp pipeline the runner uses for negative
        # cosines ("this replay was anti-aligned with rare-grad").
        cos_rare = -0.5
        raw, transformed = self.mod.compute_utility_signal(cos_rare)
        self.assertEqual(raw, -0.5)
        self.assertEqual(transformed, 0.0)
        cache.update_utility(slot, ce_delta=transformed)

        utility_post = float(cache.utility_u[slot].item())
        # EMA of (decay=0.99 * 1.0) + (0.01 * 0.0) = 0.99 — strictly
        # less than utility_pre but non-negative (clamp invariant).
        self.assertGreater(utility_post, 0.0)
        self.assertLess(utility_post, utility_pre)

    def test_replay_from_cache_skips_when_value_span_too_short(self):
        """``cache.span_length=1`` leaves no targets after the
        ``inputs/targets`` split (the replay forward needs a non-empty
        ``inputs[:, :-1]`` so the autograd graph carries through). The
        function returns None rather than crashing."""
        model = _TinyTokenModel()
        cache = EpisodicCache(capacity=4, span_length=1, key_rep_dim=8)
        slot = _populate_cache_slot(
            cache,
            slot_value_anchor=0,
            span_tokens=torch.tensor([3], dtype=torch.int64),
            key_fp=7,
            write_step=0,
        )
        model.zero_grad(set_to_none=True)
        out = self.mod.dreamworld_replay_from_cache_entry(
            model=model,
            cache=cache,
            slot=slot,
            current_step=1,
            weight=1.0,
        )
        self.assertIsNone(out)
        for p in model.parameters():
            self.assertTrue(p.grad is None or p.grad.abs().sum().item() == 0.0)


if __name__ == "__main__":
    unittest.main()
