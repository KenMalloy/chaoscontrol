"""Unit tests for ``chaoscontrol.cache_utility``.

The rank-3 oracle scoring module computes per-token utility signals
(NLL with-memory minus NLL without-memory) and converts them into a
controller probability target plus a positive-only loss reweighting.

Most tests are intentionally self-contained: the lightweight mocks document
the contract between the rank-3 scorer, ``TransactionalWakeCache``, and
``model.encode(memory_mode=..., cache_read_cutoff=...)``. A small real-model
regression test pins the same-batch causal cutoff end to end.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from chaoscontrol.cache_utility import (
    ScarcityAwareMemoryOptimizer,
    alpha_ramp,
    assign_memory_credit,
    chunked_nll_from_hidden,
    positive_only_lm_weight,
    rank3_score_batch_causal,
)
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.wake_cache_txn import TransactionalWakeCache


# ---------------------------------------------------------------------------
# Mocks documenting the foundation-piece contracts.
# ---------------------------------------------------------------------------


@dataclass
class _MockTxn:
    """Stand-in for ``CacheTxn`` from ``wake_cache_txn``.

    ``read_cutoff`` is the only field this module observes — the
    integer monotone clock value snapshotted at ``begin_batch`` so the
    encoder filters out events newer than this batch.
    """

    read_cutoff: int


class _MockCache:
    """Stand-in for ``TransactionalWakeCache``.

    The scoring module only calls ``begin_batch`` and ``commit``. We
    record both so tests can verify the lifecycle exactly matches the
    documented protocol (one begin → two encodes share the cutoff →
    one commit).
    """

    def __init__(self, *, cutoff: int = 7) -> None:
        self._cutoff = cutoff
        self.begin_calls: int = 0
        self.commit_calls: list[Any] = []

    def begin_batch(self) -> _MockTxn:
        self.begin_calls += 1
        return _MockTxn(read_cutoff=self._cutoff)

    def commit(self, txn: Any) -> None:
        self.commit_calls.append(txn)


class _MockModel(nn.Module):
    """Stand-in for the SSM encoder + LM head.

    ``encode`` returns one hidden state for ``memory_mode='off'`` and a
    different one for ``'force_on'`` so utility is non-trivial. We
    record every kwarg the scoring module passes so tests can assert
    the call signature.
    """

    def __init__(self, *, dim: int = 8, vocab: int = 32, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.randn(vocab, dim, generator=g) * (dim ** -0.5)
            )
        self._dim = dim
        self._vocab = vocab
        self.encode_calls: list[dict[str, Any]] = []
        self.append_calls: list[dict[str, Any]] = []
        # Two random hidden tensors keyed by memory_mode so utility ≠ 0.
        self._hidden_off_seed = 11
        self._hidden_mem_seed = 22

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        memory_mode: str,
        cache_read_cutoff: int,
    ) -> torch.Tensor:
        self.encode_calls.append(
            {"memory_mode": memory_mode, "cache_read_cutoff": cache_read_cutoff}
        )
        batch, seq = input_ids.shape
        seed = (
            self._hidden_off_seed
            if memory_mode == "off"
            else self._hidden_mem_seed
        )
        g = torch.Generator(device=input_ids.device).manual_seed(seed)
        return torch.randn(
            batch, seq, self._dim, generator=g, device=input_ids.device
        )

    def append_memory_from_hidden(
        self,
        hidden: torch.Tensor,
        *,
        score: torch.Tensor | None = None,
        max_tokens: int | None = None,
        event_ids: torch.Tensor | None = None,
    ) -> bool:
        self.append_calls.append(
            {
                "hidden": hidden.detach().clone(),
                "score": None if score is None else score.detach().clone(),
                "max_tokens": max_tokens,
                "event_ids": None if event_ids is None else event_ids.detach().clone(),
            }
        )
        return True


# ---------------------------------------------------------------------------
# positive_only_lm_weight
# ---------------------------------------------------------------------------


class TestPositiveOnlyLmWeight:
    def test_uniform_positive_utility_normalizes_to_one(self) -> None:
        # When every position is upweighted by the same factor, mean-1
        # normalization restores them all to exactly 1.0 — the upweight
        # is relative, not absolute.
        u = torch.full((2, 4), 1e6)  # saturates sigmoid
        mask = torch.ones_like(u, dtype=torch.bool)
        w = positive_only_lm_weight(u, mask, tau=0.1, strength=0.1, w_max=1.15)
        assert torch.allclose(w[mask], torch.full_like(w[mask], 1.0), atol=1e-4)

    def test_large_negative_utility_does_not_downweight(self) -> None:
        u = torch.full((2, 4), -1e6)
        mask = torch.ones_like(u, dtype=torch.bool)
        w = positive_only_lm_weight(u, mask, tau=0.1, strength=0.1, w_max=1.15)
        # relu(tanh(.)) → 0, raw → 1, mean → 1, normalized → 1.
        assert torch.allclose(w[mask], torch.full_like(w[mask], 1.0), atol=1e-4)

    def test_matches_positive_only_relu_tanh_reference(self) -> None:
        u = torch.tensor([[-0.50, 0.0, 0.25, 1.0]])
        mask = torch.ones_like(u, dtype=torch.bool)
        tau = 0.5
        strength = 0.2
        w = positive_only_lm_weight(u, mask, tau=tau, strength=strength, w_max=2.0)

        raw = 1.0 + strength * torch.relu(torch.tanh(u.float() / tau))
        expected = raw / raw[mask].mean()
        torch.testing.assert_close(w, expected, rtol=0, atol=1e-6)
        assert raw[0, 0].item() == 1.0
        assert raw[0, 1].item() == 1.0

    def test_mixed_signs_upweight_positives_relative_to_negatives(self) -> None:
        # Half saturated positive, half saturated negative.
        # raw = [1.1, 1.0, 1.1, 1.0]; mean = 1.05.
        u = torch.tensor([[1e6, -1e6, 1e6, -1e6]])
        mask = torch.ones_like(u, dtype=torch.bool)
        w = positive_only_lm_weight(u, mask, tau=0.1, strength=0.1, w_max=1.15)
        expected_pos = 1.1 / 1.05
        expected_neg = 1.0 / 1.05
        assert math.isclose(w[0, 0].item(), expected_pos, abs_tol=1e-4)
        assert math.isclose(w[0, 1].item(), expected_neg, abs_tol=1e-4)
        assert w[0, 0].item() > w[0, 1].item()

    def test_w_max_clamps_raw_form(self) -> None:
        # strength=10 would push raw to 11; clamp at w_max=1.15 first.
        # With mixed signs the clamp matters: positives go to 1.15, negatives to 1.0.
        u = torch.tensor([[1e6, -1e6]])
        mask = torch.ones_like(u, dtype=torch.bool)
        w = positive_only_lm_weight(u, mask, tau=0.1, strength=10.0, w_max=1.15)
        # raw = [11, 1] → clamp = [1.15, 1.0] → mean = 1.075 → norm = [1.0698, 0.9302]
        assert (w[mask] <= 1.15 + 1e-6).all()
        # Without the w_max clamp the post-norm max would be 11/6 ≈ 1.83.
        assert w[0, 0].item() < 1.10

    def test_invalid_positions_get_zero(self) -> None:
        u = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        mask = torch.tensor([[True, False, True, False]])
        w = positive_only_lm_weight(u, mask, tau=0.1, strength=0.1, w_max=1.15)
        assert w[0, 1].item() == 0.0
        assert w[0, 3].item() == 0.0
        assert w[0, 0].item() > 0.0
        assert w[0, 2].item() > 0.0

    def test_mean_one_normalization(self) -> None:
        # Mixed positive utilities — after normalization, mean over valid = 1.
        torch.manual_seed(0)
        u = torch.randn(4, 16)
        mask = torch.ones_like(u, dtype=torch.bool)
        w = positive_only_lm_weight(u, mask, tau=0.1, strength=0.1, w_max=1.15)
        assert math.isclose(w[mask].mean().item(), 1.0, abs_tol=1e-5)

    def test_mean_one_with_partial_mask(self) -> None:
        torch.manual_seed(1)
        u = torch.randn(4, 16)
        mask = torch.zeros_like(u, dtype=torch.bool)
        mask[0, :8] = True  # only 8 valid positions
        w = positive_only_lm_weight(u, mask, tau=0.1, strength=0.1, w_max=1.15)
        # Mean-1 invariant computed only over valid positions.
        assert math.isclose(w[mask].mean().item(), 1.0, abs_tol=1e-5)
        # Invalid positions are zero.
        assert (w[~mask] == 0.0).all()

    def test_all_invalid_does_not_nan(self) -> None:
        u = torch.tensor([[0.5, -0.5]])
        mask = torch.zeros_like(u, dtype=torch.bool)
        w = positive_only_lm_weight(u, mask, tau=0.1, strength=0.1, w_max=1.15)
        assert torch.isfinite(w).all()
        assert (w == 0.0).all()

    def test_shape_matches_input(self) -> None:
        u = torch.randn(3, 7)
        mask = torch.ones_like(u, dtype=torch.bool)
        w = positive_only_lm_weight(u, mask, tau=0.1, strength=0.1, w_max=1.15)
        assert w.shape == (3, 7)


# ---------------------------------------------------------------------------
# chunked_nll_from_hidden
# ---------------------------------------------------------------------------


class TestChunkedNllFromHidden:
    def _build_model(self, dim: int = 8, vocab: int = 16, seed: int = 0) -> _MockModel:
        return _MockModel(dim=dim, vocab=vocab, seed=seed)

    def test_returns_per_token_nll_shape(self) -> None:
        model = self._build_model()
        hidden = torch.randn(2, 5, 8)
        targets = torch.randint(0, 16, (2, 5))
        nll = chunked_nll_from_hidden(model, hidden, targets)
        assert nll.shape == (2, 5)
        assert nll.dtype == torch.float32

    def test_nll_is_non_negative(self) -> None:
        model = self._build_model()
        hidden = torch.randn(2, 5, 8)
        targets = torch.randint(0, 16, (2, 5))
        nll = chunked_nll_from_hidden(model, hidden, targets)
        assert (nll >= 0.0).all()

    def test_chunk_size_does_not_change_result(self) -> None:
        torch.manual_seed(42)
        model = self._build_model()
        hidden = torch.randn(2, 13, 8)
        targets = torch.randint(0, 16, (2, 13))
        nll_full = chunked_nll_from_hidden(model, hidden, targets, chunk_size=64)
        nll_small = chunked_nll_from_hidden(model, hidden, targets, chunk_size=3)
        nll_one = chunked_nll_from_hidden(model, hidden, targets, chunk_size=1)
        assert torch.allclose(nll_full, nll_small, atol=1e-6)
        assert torch.allclose(nll_full, nll_one, atol=1e-6)

    def test_matches_reference_cross_entropy(self) -> None:
        torch.manual_seed(7)
        model = self._build_model(dim=4, vocab=8)
        hidden = torch.randn(3, 6, 4)
        targets = torch.randint(0, 8, (3, 6))
        # Reference: do the head ourselves, no chunking.
        with torch.no_grad():
            logits = model.lm_head(model.final_norm(hidden)).float()
            ref = F.cross_entropy(
                logits.reshape(-1, 8),
                targets.reshape(-1),
                reduction="none",
            ).reshape(3, 6)
        out = chunked_nll_from_hidden(model, hidden, targets, chunk_size=2)
        assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# alpha_ramp
# ---------------------------------------------------------------------------


class TestAlphaRamp:
    def test_alpha_zero_step_is_small(self) -> None:
        # sigmoid(8 * (0 - 0.3)) = sigmoid(-2.4) ≈ 0.083
        a = alpha_ramp(0, 1000, alpha_max=1.0)
        assert a == pytest.approx(1.0 / (1.0 + math.exp(2.4)), abs=1e-6)

    def test_alpha_at_thirty_percent_is_half(self) -> None:
        # At t/T = 0.3, sigmoid(0) = 0.5
        a = alpha_ramp(300, 1000, alpha_max=1.0)
        assert a == pytest.approx(0.5, abs=1e-6)

    def test_alpha_full_step_approaches_alpha_max(self) -> None:
        # sigmoid(8 * 0.7) = sigmoid(5.6) ≈ 0.9963
        a = alpha_ramp(1000, 1000, alpha_max=1.0)
        assert 0.99 < a <= 1.0

    def test_alpha_scales_with_alpha_max(self) -> None:
        a05 = alpha_ramp(500, 1000, alpha_max=0.5)
        a10 = alpha_ramp(500, 1000, alpha_max=1.0)
        assert a05 == pytest.approx(a10 * 0.5, rel=1e-6)

    def test_alpha_is_monotone_increasing(self) -> None:
        T = 1000
        prev = -1.0
        for t in range(0, T + 1, 50):
            a = alpha_ramp(t, T, alpha_max=1.0)
            assert a >= prev
            prev = a

    def test_alpha_zero_total_steps_does_not_raise(self) -> None:
        # T=0 would div-by-zero if not guarded.
        a = alpha_ramp(0, 0, alpha_max=1.0)
        assert math.isfinite(a)


# ---------------------------------------------------------------------------
# rank3_score_batch_causal
# ---------------------------------------------------------------------------


class TestRank3ScoreBatchCausal:
    def _setup(self, batch: int = 2, seq: int = 6) -> tuple[_MockModel, _MockCache, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        model = _MockModel(dim=8, vocab=32, seed=3)
        cache = _MockCache(cutoff=42)
        input_ids = torch.randint(0, 32, (batch, seq))
        valid_mask = torch.ones((batch, seq), dtype=torch.bool)
        return model, cache, input_ids, valid_mask

    def test_returns_expected_keys(self) -> None:
        model, cache, ids, mask = self._setup()
        out = rank3_score_batch_causal(
            model=model, cache=cache, input_ids=ids, valid_mask=mask
        )
        assert set(out.keys()) == {
            "utility",
            "controller_target",
            "loss_weight",
            "confidence",
        }

    def test_output_shapes_are_per_target_token(self) -> None:
        model, cache, ids, mask = self._setup(batch=2, seq=6)
        out = rank3_score_batch_causal(
            model=model, cache=cache, input_ids=ids, valid_mask=mask
        )
        # x = ids[:, :-1], y = ids[:, 1:] → outputs are (B, seq-1)
        assert out["utility"].shape == (2, 5)
        assert out["controller_target"].shape == (2, 5)
        assert out["loss_weight"].shape == (2, 5)
        assert out["confidence"].shape == (2, 5)

    def test_controller_target_is_clamped_to_valid_range(self) -> None:
        model, cache, ids, mask = self._setup()
        out = rank3_score_batch_causal(
            model=model, cache=cache, input_ids=ids, valid_mask=mask
        )
        ct = out["controller_target"]
        assert (ct >= 0.05 - 1e-6).all()
        assert (ct <= 0.95 + 1e-6).all()

    def test_cache_lifecycle_is_begin_then_commit(self) -> None:
        model, cache, ids, mask = self._setup()
        rank3_score_batch_causal(
            model=model, cache=cache, input_ids=ids, valid_mask=mask
        )
        assert cache.begin_calls == 1
        assert len(cache.commit_calls) == 1
        # The same txn returned by begin_batch is what gets committed.
        assert cache.commit_calls[0].read_cutoff == 42

    def test_encode_called_with_off_and_force_on(self) -> None:
        model, cache, ids, mask = self._setup()
        rank3_score_batch_causal(
            model=model, cache=cache, input_ids=ids, valid_mask=mask
        )
        modes = [c["memory_mode"] for c in model.encode_calls]
        assert sorted(modes) == ["force_on", "off"]

    def test_encode_uses_same_cache_read_cutoff(self) -> None:
        model, cache, ids, mask = self._setup()
        rank3_score_batch_causal(
            model=model, cache=cache, input_ids=ids, valid_mask=mask
        )
        cutoffs = {c["cache_read_cutoff"] for c in model.encode_calls}
        assert cutoffs == {42}

    def test_invalid_positions_zeroed_across_all_outputs(self) -> None:
        # Without masking on every output the wiring task would multiply
        # BCE × confidence at padding positions and leak gradient through
        # tokens that aren't even valid targets.
        model, cache, ids, _mask = self._setup(batch=1, seq=4)
        valid_mask = torch.tensor([[True, False, True, False]])
        out = rank3_score_batch_causal(
            model=model, cache=cache, input_ids=ids, valid_mask=valid_mask
        )
        # mask[:, 1:] = [[False, True, False]] → positions 0 and 2 invalid in target.
        for key in ("utility", "controller_target", "confidence", "loss_weight"):
            assert out[key][0, 0].item() == 0.0, f"{key} not zeroed at invalid pos 0"
            assert out[key][0, 2].item() == 0.0, f"{key} not zeroed at invalid pos 2"

    def test_invalid_positions_zeroed_under_scarcity_price(self) -> None:
        # With a positive read_price the unmasked formula would set
        # confidence ≈ tanh(price/tau) — large — at invalid positions.
        # Verify masking holds even when the price is non-zero.
        opt = ScarcityAwareMemoryOptimizer(tau=0.10, max_price=1.0)
        opt.read_price = 0.5
        model, cache, ids, _mask = self._setup(batch=1, seq=4)
        valid_mask = torch.tensor([[True, False, True, False]])
        out = rank3_score_batch_causal(
            model=model,
            cache=cache,
            input_ids=ids,
            valid_mask=valid_mask,
            scarcity_optimizer=opt,
        )
        for key in ("utility", "controller_target", "confidence", "loss_weight"):
            assert out[key][0, 0].item() == 0.0
            assert out[key][0, 2].item() == 0.0

    def test_inference_mode_no_grad_required(self) -> None:
        model, cache, ids, mask = self._setup()
        out = rank3_score_batch_causal(
            model=model, cache=cache, input_ids=ids, valid_mask=mask
        )
        # All outputs detached / no grad: requires_grad must be False everywhere.
        for k, v in out.items():
            assert not v.requires_grad, f"{k} unexpectedly requires_grad"

    def test_loss_weight_mean_one_over_valid(self) -> None:
        model, cache, ids, mask = self._setup(batch=4, seq=17)
        out = rank3_score_batch_causal(
            model=model, cache=cache, input_ids=ids, valid_mask=mask
        )
        target_mask = mask[:, 1:].bool()
        if target_mask.any():
            assert math.isclose(
                out["loss_weight"][target_mask].mean().item(), 1.0, abs_tol=1e-5
            )

    def test_update_model_memory_appends_predictor_states_only(self) -> None:
        model, cache, ids, mask = self._setup(batch=2, seq=6)
        out = rank3_score_batch_causal(
            model=model,
            cache=cache,
            input_ids=ids,
            valid_mask=mask,
            update_model_memory_after=True,
            memory_write_tokens=3,
        )
        assert len(model.append_calls) == 1
        call = model.append_calls[0]
        assert call["hidden"].shape == (2, 5, model._dim)
        assert call["score"].shape == out["utility"].shape
        assert call["max_tokens"] == 3
        assert call["event_ids"] is None

    def test_update_model_memory_reserves_event_ids_when_cache_supports_it(self) -> None:
        class _ClockedMockCache(_MockCache):
            def __init__(self) -> None:
                super().__init__(cutoff=9)
                self.reserved: tuple[int, torch.device] | None = None

            def reserve_event_ids(
                self,
                n: int,
                *,
                device: torch.device | str | None = None,
            ) -> torch.Tensor:
                dev = torch.device("cpu" if device is None else device)
                self.reserved = (int(n), dev)
                return torch.arange(10, 10 + int(n), dtype=torch.long, device=dev)

        model = _MockModel(dim=8, vocab=32, seed=3)
        cache = _ClockedMockCache()
        ids = torch.randint(0, 32, (2, 6))
        mask = torch.ones((2, 6), dtype=torch.bool)

        rank3_score_batch_causal(
            model=model,
            cache=cache,
            input_ids=ids,
            valid_mask=mask,
            update_model_memory_after=True,
            memory_write_tokens=3,
        )

        assert cache.reserved == (10, ids.device)
        event_ids = model.append_calls[0]["event_ids"]
        assert event_ids is not None
        assert event_ids.tolist() == list(range(10, 20))

    def test_update_model_memory_writes_slots_newer_than_oracle_cutoff(self) -> None:
        torch.manual_seed(17)
        model = ChaosStudentLM(
            vocab_size=64,
            dim=8,
            num_layers=1,
            ff_mult=2,
            a_mode="diag",
            rich_b_mode="none",
            outer_model_dim=8,
            outer_model_type="multislot",
            outer_max_slots=8,
            buffer_mode="append_only",
            retrieval_mode="bucket_mean",
        )
        assert model.outer_model is not None
        with torch.no_grad():
            model.outer_model.decoder.weight.copy_(torch.eye(8))
        cache = TransactionalWakeCache(max_moments=0, max_hidden_buffer=0)
        ids = torch.randint(0, 64, (1, 5))
        mask = torch.ones_like(ids, dtype=torch.bool)

        rank3_score_batch_causal(
            model=model,
            cache=cache,
            input_ids=ids,
            valid_mask=mask,
            update_model_memory_after=True,
            memory_write_tokens=2,
        )

        assert len(model.outer_model._slot_event_ids) == 2
        assert min(model.outer_model._slot_event_ids) > 0
        hidden_at_old_cutoff = model.outer_model.read_bucket(
            1,
            bucket_id=0,
            read_cutoff=0,
        )
        hidden_after_commit = model.outer_model.read_bucket(
            1,
            bucket_id=0,
            read_cutoff=cache.clock.current,
        )
        torch.testing.assert_close(
            hidden_at_old_cutoff,
            torch.zeros_like(hidden_at_old_cutoff),
            rtol=0,
            atol=0,
        )
        assert hidden_after_commit.norm().item() > 0.0

    def test_confidence_in_unit_interval(self) -> None:
        model, cache, ids, mask = self._setup()
        out = rank3_score_batch_causal(
            model=model, cache=cache, input_ids=ids, valid_mask=mask
        )
        # confidence = tanh(|net|/tau); tanh ∈ [0, 1] for non-negative input.
        assert (out["confidence"] >= 0.0).all()
        assert (out["confidence"] <= 1.0 + 1e-6).all()

    def test_rank3_forwards_optimizer_call(self) -> None:
        # A spy subclass records every controller_target invocation so
        # we can assert the rank3 function actually delegates to the
        # optimizer instead of falling back to the plain sigmoid.
        class _Spy(ScarcityAwareMemoryOptimizer):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.calls: int = 0

            def controller_target(self, utility):  # type: ignore[override]
                self.calls += 1
                return super().controller_target(utility)

        spy = _Spy(tau=0.10)
        model, cache, ids, mask = self._setup()
        rank3_score_batch_causal(
            model=model,
            cache=cache,
            input_ids=ids,
            valid_mask=mask,
            scarcity_optimizer=spy,
        )
        assert spy.calls == 1


# ---------------------------------------------------------------------------
# ScarcityAwareMemoryOptimizer
# ---------------------------------------------------------------------------


class TestScarcityAwareMemoryOptimizer:
    def test_initial_state(self) -> None:
        opt = ScarcityAwareMemoryOptimizer(
            tau=0.1, target_read_rate=0.25, target_write_rate=0.10
        )
        assert opt.read_price == 0.0
        assert opt.write_price == 0.0
        assert opt.read_rate_ema == pytest.approx(0.25)
        assert opt.write_rate_ema == pytest.approx(0.10)

    def test_controller_target_shape_and_range(self) -> None:
        opt = ScarcityAwareMemoryOptimizer()
        utility = torch.randn(3, 5)
        target, confidence = opt.controller_target(utility)
        assert target.shape == (3, 5)
        assert confidence.shape == (3, 5)
        assert (target >= 0.05 - 1e-6).all()
        assert (target <= 0.95 + 1e-6).all()
        assert (confidence >= 0.0).all()
        assert (confidence <= 1.0 + 1e-6).all()

    def test_controller_target_is_detached(self) -> None:
        opt = ScarcityAwareMemoryOptimizer()
        utility = torch.randn(2, 3, requires_grad=True)
        target, confidence = opt.controller_target(utility)
        assert not target.requires_grad
        assert not confidence.requires_grad

    def test_write_target_shape_and_range(self) -> None:
        opt = ScarcityAwareMemoryOptimizer()
        write_utility = torch.randn(3, 5)
        target, confidence = opt.write_target(write_utility)
        assert target.shape == (3, 5)
        assert (target >= 0.05 - 1e-6).all()
        assert (target <= 0.95 + 1e-6).all()
        assert (confidence >= 0.0).all()

    def test_dual_step_raises_price_when_actual_exceeds_target(self) -> None:
        opt = ScarcityAwareMemoryOptimizer(
            target_read_rate=0.20, dual_lr=0.1, ema_beta=0.5, max_price=1.0
        )
        # Repeatedly observe over-reading; price should rise from 0.
        for _ in range(20):
            opt.dual_step(actual_read_rate=0.6)
        assert opt.read_price > 0.0
        assert opt.read_price <= 1.0  # max_price

    def test_dual_step_lowers_price_when_actual_below_target(self) -> None:
        opt = ScarcityAwareMemoryOptimizer(
            target_read_rate=0.30, dual_lr=0.1, ema_beta=0.5, max_price=1.0
        )
        opt.read_price = 0.5  # start mid-range
        # Repeatedly observe under-reading; price should fall.
        for _ in range(20):
            opt.dual_step(actual_read_rate=0.05)
        assert opt.read_price < 0.5
        assert opt.read_price >= 0.0  # clamped at zero

    def test_dual_step_clamps_at_zero(self) -> None:
        opt = ScarcityAwareMemoryOptimizer(
            target_read_rate=0.30, dual_lr=10.0, ema_beta=0.0, max_price=1.0
        )
        opt.dual_step(actual_read_rate=0.0)  # huge negative error
        assert opt.read_price == 0.0  # cannot go negative

    def test_dual_step_clamps_at_max_price(self) -> None:
        opt = ScarcityAwareMemoryOptimizer(
            target_read_rate=0.0, dual_lr=10.0, ema_beta=0.0, max_price=0.50
        )
        opt.dual_step(actual_read_rate=1.0)  # huge positive error
        assert opt.read_price == 0.50

    def test_dual_step_write_rate_optional(self) -> None:
        opt = ScarcityAwareMemoryOptimizer(
            target_read_rate=0.30, target_write_rate=0.10,
            dual_lr=0.1, ema_beta=0.5, max_price=1.0,
        )
        prev_write = opt.write_price
        prev_write_ema = opt.write_rate_ema
        opt.dual_step(actual_read_rate=0.5)  # no write_rate provided
        # Read price moved; write price did not.
        assert opt.read_price > 0.0
        assert opt.write_price == prev_write
        assert opt.write_rate_ema == pytest.approx(prev_write_ema)

    def test_dual_step_updates_both_when_provided(self) -> None:
        opt = ScarcityAwareMemoryOptimizer(
            target_read_rate=0.30, target_write_rate=0.10,
            dual_lr=0.1, ema_beta=0.5, max_price=1.0,
        )
        opt.dual_step(actual_read_rate=0.6, actual_write_rate=0.5)
        assert opt.read_price > 0.0
        assert opt.write_price > 0.0

    def test_ema_smooths_rate_estimates(self) -> None:
        opt = ScarcityAwareMemoryOptimizer(
            target_read_rate=0.25, ema_beta=0.9
        )
        # First step: ema = 0.9*0.25 + 0.1*0.5 = 0.275
        opt.dual_step(actual_read_rate=0.5)
        assert opt.read_rate_ema == pytest.approx(0.9 * 0.25 + 0.1 * 0.5, abs=1e-6)

    def test_read_price_shifts_target_at_moderate_utility(self) -> None:
        # Pick utility magnitudes well below sigmoid saturation so the
        # price delta moves the target.
        opt_zero = ScarcityAwareMemoryOptimizer(tau=0.5)
        opt_high = ScarcityAwareMemoryOptimizer(tau=0.5, max_price=1.0)
        opt_high.read_price = 0.4
        utility = torch.tensor([[-0.1, 0.0, 0.1, 0.2, 0.3]])
        target_zero, _ = opt_zero.controller_target(utility)
        target_high, _ = opt_high.controller_target(utility)
        # Higher read price → strictly lower target at every position.
        assert (target_high < target_zero).all()

    def test_confidence_peaks_at_extreme_net(self) -> None:
        # At net = 0 confidence is exactly 0; far from 0 it saturates near 1.
        opt = ScarcityAwareMemoryOptimizer(tau=0.10)
        utility = torch.tensor([[0.0, 1.0]])
        _, confidence = opt.controller_target(utility)
        assert confidence[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert confidence[0, 1].item() > 0.99

# ---------------------------------------------------------------------------
# assign_memory_credit
# ---------------------------------------------------------------------------


class TestAssignMemoryCredit:
    def test_positive_utility_writes_credit_only(self) -> None:
        n_entries = 10
        entry_credit = torch.zeros(n_entries)
        entry_debit = torch.zeros(n_entries)
        # Two tokens, each retrieved one entry.
        entry_ids = torch.tensor([[[3], [5]]])  # (B=1, T=2, K=1)
        weights = torch.ones(1, 2, 1)
        utility = torch.tensor([[1.0, 2.0]])
        assign_memory_credit(
            entry_credit, entry_debit, entry_ids, weights, utility
        )
        assert entry_credit[3].item() == pytest.approx(1.0)
        assert entry_credit[5].item() == pytest.approx(2.0)
        assert (entry_debit == 0.0).all()

    def test_negative_utility_writes_debit_only(self) -> None:
        n_entries = 10
        entry_credit = torch.zeros(n_entries)
        entry_debit = torch.zeros(n_entries)
        entry_ids = torch.tensor([[[3], [5]]])
        weights = torch.ones(1, 2, 1)
        utility = torch.tensor([[-1.0, -2.0]])
        assign_memory_credit(
            entry_credit, entry_debit, entry_ids, weights, utility
        )
        assert entry_debit[3].item() == pytest.approx(1.0)
        assert entry_debit[5].item() == pytest.approx(2.0)
        assert (entry_credit == 0.0).all()

    def test_zero_utility_writes_neither(self) -> None:
        n_entries = 10
        entry_credit = torch.zeros(n_entries)
        entry_debit = torch.zeros(n_entries)
        entry_ids = torch.tensor([[[3]]])
        weights = torch.ones(1, 1, 1)
        utility = torch.tensor([[0.0]])
        assign_memory_credit(
            entry_credit, entry_debit, entry_ids, weights, utility
        )
        assert (entry_credit == 0.0).all()
        assert (entry_debit == 0.0).all()

    def test_accumulates_across_calls(self) -> None:
        n_entries = 10
        entry_credit = torch.zeros(n_entries)
        entry_debit = torch.zeros(n_entries)
        entry_ids = torch.tensor([[[3]]])
        weights = torch.ones(1, 1, 1)
        utility = torch.tensor([[1.0]])
        assign_memory_credit(
            entry_credit, entry_debit, entry_ids, weights, utility
        )
        assign_memory_credit(
            entry_credit, entry_debit, entry_ids, weights, utility
        )
        assert entry_credit[3].item() == pytest.approx(2.0)

    def test_multiple_entries_per_token(self) -> None:
        # Two tokens × three retrieved entries each.
        n_entries = 10
        entry_credit = torch.zeros(n_entries)
        entry_debit = torch.zeros(n_entries)
        entry_ids = torch.tensor([[[1, 2, 3], [2, 4, 6]]])  # (1, 2, 3)
        weights = torch.tensor([[[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]]])
        utility = torch.tensor([[2.0, 3.0]])
        assign_memory_credit(
            entry_credit, entry_debit, entry_ids, weights, utility
        )
        # Token 0 (utility=2.0): entry 1 += 0.5*2=1.0, entry 2 += 0.3*2=0.6, entry 3 += 0.2*2=0.4
        # Token 1 (utility=3.0): entry 2 += 0.1*3=0.3, entry 4 += 0.4*3=1.2, entry 6 += 0.5*3=1.5
        assert entry_credit[1].item() == pytest.approx(1.0)
        assert entry_credit[2].item() == pytest.approx(0.9)  # 0.6 + 0.3
        assert entry_credit[3].item() == pytest.approx(0.4)
        assert entry_credit[4].item() == pytest.approx(1.2)
        assert entry_credit[6].item() == pytest.approx(1.5)
        assert (entry_debit == 0.0).all()

    def test_no_grad_required(self) -> None:
        # Inputs may require_grad; the in-place scatter_add_ on the
        # credit tensors must not break (decorator should detach).
        n_entries = 5
        entry_credit = torch.zeros(n_entries)
        entry_debit = torch.zeros(n_entries)
        entry_ids = torch.tensor([[[0]]])
        weights = torch.ones(1, 1, 1, requires_grad=True)
        utility = torch.tensor([[1.0]], requires_grad=True)
        assign_memory_credit(
            entry_credit, entry_debit, entry_ids, weights, utility
        )
        assert entry_credit[0].item() == pytest.approx(1.0)
