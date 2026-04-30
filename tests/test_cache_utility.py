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
import json
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from chaoscontrol.cache_utility import (
    CrctGradientConflictMonitor,
    ScarcityAwareMemoryOptimizer,
    alpha_ramp,
    assign_memory_credit,
    chunked_nll_from_hidden,
    plasticity_budget_from_hidden_delta,
    positive_only_lm_weight,
    rank3_score_batch_causal,
)
from chaoscontrol.model import CareStudentLM
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


class _PacketMockModel(_MockModel):
    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        memory_mode: str,
        cache_read_cutoff: int,
        return_memory_meta: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        hidden = super().encode(
            input_ids,
            memory_mode=memory_mode,
            cache_read_cutoff=cache_read_cutoff,
        )
        if return_memory_meta and memory_mode == "force_on":
            return {
                "hidden": hidden,
                "memory_meta": {
                    "memory_residual": torch.ones(
                        input_ids.shape[0], 1, self._dim, device=input_ids.device
                    ),
                },
            }
        return hidden


class _SequencePacketMockModel(_PacketMockModel):
    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        memory_mode: str,
        cache_read_cutoff: int,
        return_memory_meta: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        hidden = _MockModel.encode(
            self,
            input_ids,
            memory_mode=memory_mode,
            cache_read_cutoff=cache_read_cutoff,
        )
        if return_memory_meta and memory_mode == "force_on":
            return {
                "hidden": hidden,
                "memory_meta": {
                    "memory_residual": torch.ones(
                        input_ids.shape[0],
                        input_ids.shape[1] - 1,
                        self._dim,
                        device=input_ids.device,
                    ),
                },
            }
        return hidden


class _PairedMockModel(_MockModel):
    def __init__(self, *, dim: int = 8, vocab: int = 32, seed: int = 0) -> None:
        super().__init__(dim=dim, vocab=vocab, seed=seed)
        self.paired_calls: list[dict[str, Any]] = []

    def encode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("rank3_score_batch_causal should use paired encode")

    def encode_paired_for_score(
        self,
        input_ids: torch.Tensor,
        *,
        cache_read_cutoff: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.paired_calls.append({"cache_read_cutoff": cache_read_cutoff})
        batch, seq = input_ids.shape
        g_off = torch.Generator(device=input_ids.device).manual_seed(self._hidden_off_seed)
        g_mem = torch.Generator(device=input_ids.device).manual_seed(self._hidden_mem_seed)
        h_off = torch.randn(batch, seq, self._dim, generator=g_off, device=input_ids.device)
        h_mem = torch.randn(batch, seq, self._dim, generator=g_mem, device=input_ids.device)
        return (
            h_off,
            h_mem,
            {
                "memory_residual": torch.ones(
                    batch,
                    1,
                    self._dim,
                    device=input_ids.device,
                ),
            },
        )


# ---------------------------------------------------------------------------
# plasticity_budget_from_hidden_delta
# ---------------------------------------------------------------------------


class TestPlasticityBudgetFromHiddenDelta:
    def test_coverage_is_positive_when_residual_tracks_utility(self) -> None:
        h_off = torch.zeros(1, 4, 3)
        h_mem = torch.zeros_like(h_off)
        h_mem[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        h_mem[0, :, 1] = torch.tensor([3.0, 2.0, 1.0, 0.0])
        utility = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        mask = torch.ones_like(utility, dtype=torch.bool)

        out = plasticity_budget_from_hidden_delta(
            h_off=h_off,
            h_mem=h_mem,
            utility=utility,
            mask=mask,
            tau=0.5,
        )

        assert out["plasticity_coverage"].shape == (3,)
        assert out["plasticity_confidence"].shape == (3,)
        assert out["plasticity_budget"].shape == (3,)
        assert out["plasticity_coverage"][0] > 0.99
        assert out["plasticity_coverage"][1] < -0.99
        assert out["plasticity_budget"][0] > 0.9
        assert out["plasticity_budget"][1].item() == 0.0

    def test_invalid_or_zero_utility_positions_do_not_open_budget(self) -> None:
        h_off = torch.zeros(2, 3, 4)
        h_mem = torch.randn_like(h_off)
        utility = torch.zeros(2, 3)
        mask = torch.zeros_like(utility, dtype=torch.bool)

        out = plasticity_budget_from_hidden_delta(
            h_off=h_off,
            h_mem=h_mem,
            utility=utility,
            mask=mask,
            tau=0.1,
        )

        assert torch.equal(out["plasticity_coverage"], torch.zeros(4))
        assert torch.equal(out["plasticity_confidence"], torch.zeros(4))
        assert torch.equal(out["plasticity_budget"], torch.zeros(4))

    def test_matches_centered_reference_implementation(self) -> None:
        # The matmul-reduction form must agree with the centered
        # ``E[(X-xm)(Y-ym)]`` reference within fp32 round-off across a
        # range of batch shapes, partial masks, and dtypes — that's the
        # promise the rank-3 path relies on when shipping the budget to
        # the trunk plasticity gate.
        torch.manual_seed(123)
        cases = [
            (1, 4, 3, torch.float32),
            (2, 8, 5, torch.float32),
            (3, 16, 7, torch.float32),
            (2, 12, 4, torch.bfloat16),
        ]
        for B, T, D, dtype in cases:
            h_off = torch.randn(B, T, D, dtype=dtype)
            h_mem = h_off + 0.05 * torch.randn(B, T, D, dtype=dtype)
            utility = torch.randn(B, T)
            mask = torch.zeros(B, T, dtype=torch.bool)
            mask[:, : max(1, T // 2)] = True

            actual = plasticity_budget_from_hidden_delta(
                h_off=h_off,
                h_mem=h_mem,
                utility=utility,
                mask=mask,
                tau=0.1,
            )

            x = (h_mem.float() - h_off.float()).abs()
            y = torch.relu(utility.float()) * mask.float()
            w = mask.float()
            n = w.sum().clamp_min(1.0)
            d = D
            w3 = w.unsqueeze(-1)
            x_mean = (x * w3).sum(dim=(0, 1)) / n
            y_mean = y.sum() / n
            dx = (x - x_mean.view(1, 1, d)) * w3
            dy = (y - y_mean).unsqueeze(-1) * w3
            cov = (dx * dy).sum(dim=(0, 1)) / n
            x_var = (
                ((x - x_mean.view(1, 1, d)).square() * w3).sum(dim=(0, 1)) / n
            )
            y_var = ((y - y_mean).square() * w).sum() / n
            ref_coverage = cov / torch.sqrt(
                (x_var * y_var).clamp_min(1e-6)
            )
            ref_coverage = ref_coverage.clamp(min=-1.0, max=1.0)
            ref_energy = torch.sqrt((y.square() * w).sum() / n)
            ref_conf_scalar = torch.tanh(ref_energy / 0.1)
            ref_confidence = (ref_coverage.abs() * ref_conf_scalar).clamp(0.0, 1.0)
            ref_budget = torch.relu(ref_coverage) * ref_confidence

            torch.testing.assert_close(
                actual["plasticity_coverage"],
                ref_coverage,
                rtol=1e-4,
                atol=1e-5,
            )
            torch.testing.assert_close(
                actual["plasticity_confidence"],
                ref_confidence,
                rtol=1e-4,
                atol=1e-5,
            )
            torch.testing.assert_close(
                actual["plasticity_budget"],
                ref_budget,
                rtol=1e-4,
                atol=1e-5,
            )


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

    def test_chunk_size_clamped_against_memory_budget(self) -> None:
        # When chunk_size * batch * vocab * 4 exceeds the per-chunk
        # budget the function must clamp chunk_size down and still
        # produce identical per-token NLL.
        torch.manual_seed(11)
        from chaoscontrol.cache_utility import _NLL_CHUNK_BUDGET_BYTES

        model = self._build_model(dim=4, vocab=8)
        batch, seq = 3, 17
        hidden = torch.randn(batch, seq, 4)
        targets = torch.randint(0, 8, (batch, seq))
        ref = chunked_nll_from_hidden(model, hidden, targets, chunk_size=1)

        # Pick a chunk_size big enough that the loop would run once
        # without the cap, and a vocab/batch combo that hits the budget.
        # Force the cap to a tiny value to exercise the clamp.
        from unittest.mock import patch

        with patch("chaoscontrol.cache_utility._NLL_CHUNK_BUDGET_BYTES", 1):
            # batch * vocab * 4 = 96 bytes; budget=1 → effective_chunk = 1
            out = chunked_nll_from_hidden(
                model, hidden, targets, chunk_size=10_000
            )
        assert torch.allclose(out, ref, atol=1e-6), (
            "chunk_size budget clamp must not change the per-token NLL output"
        )

    def test_chunk_budget_bytes_override_does_not_change_result(self) -> None:
        # GPU3 maintenance/oracle callers raise the per-chunk logits budget
        # to amortise launch overhead at large gathered batch sizes. The
        # override must be a launch-count knob only — the per-token NLL is
        # still bit-stable (within the same chunk decomposition the global
        # default would produce when both fit one chunk).
        torch.manual_seed(13)
        model = self._build_model(dim=4, vocab=8)
        batch, seq = 3, 17
        hidden = torch.randn(batch, seq, 4)
        targets = torch.randint(0, 8, (batch, seq))
        ref = chunked_nll_from_hidden(model, hidden, targets, chunk_size=1)
        out = chunked_nll_from_hidden(
            model,
            hidden,
            targets,
            chunk_size=10_000,
            chunk_budget_bytes=8 << 30,
        )
        assert torch.allclose(out, ref, atol=1e-6)

    def test_chunk_budget_bytes_must_be_positive(self) -> None:
        model = self._build_model(dim=4, vocab=8)
        hidden = torch.randn(2, 5, 4)
        targets = torch.randint(0, 8, (2, 5))
        with pytest.raises(ValueError, match="chunk_budget_bytes"):
            chunked_nll_from_hidden(
                model, hidden, targets, chunk_budget_bytes=0
            )

    def test_casts_hidden_to_lm_head_dtype(self) -> None:
        # Rank-3 maintenance probes may produce fp32 hidden variants even
        # when the trained model head is bf16. The helper owns that dtype
        # boundary so callers do not need to keep autocast alive around the
        # chunked LM-head pass.
        model = self._build_model(dim=4, vocab=8).to(dtype=torch.bfloat16)
        hidden = torch.randn(2, 5, 4, dtype=torch.float32)
        targets = torch.randint(0, 8, (2, 5))

        nll = chunked_nll_from_hidden(model, hidden, targets, chunk_size=2)

        assert nll.shape == (2, 5)
        assert nll.dtype == torch.float32
        assert torch.isfinite(nll).all()

    def test_chunked_nll_has_no_cpu_scorer_escape_hatch(self) -> None:
        import inspect

        assert "cpu_scorer" not in inspect.signature(chunked_nll_from_hidden).parameters
        module = __import__("chaoscontrol.cache_utility").cache_utility
        assert not hasattr(module, "CpuMemoryScorer")


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


class TestCrctGradientConflictMonitor:
    def test_cold_start_observes_without_changing_write_score(self) -> None:
        model = _MockModel(dim=8, vocab=32, seed=4)
        monitor = CrctGradientConflictMonitor(enabled=True)
        hidden = torch.randn(1, 4, 8)
        targets = torch.tensor([[1, 2, 3, 4]])
        utility = torch.tensor([[0.1, 0.5, -0.2, 0.3]])
        mask = torch.ones_like(targets, dtype=torch.bool)

        write_score, write_limit = monitor.apply_to_write_scores(
            model=model,
            hidden=hidden,
            targets=targets,
            utility=utility,
            mask=mask,
            max_tokens=2,
        )

        torch.testing.assert_close(write_score, utility)
        assert write_limit == 2
        diag = monitor.diagnostics()
        assert diag["cold_start_calls"] == 1
        assert diag["candidates_seen"] == 2
        assert diag["admitted_candidates"] == 2
        assert diag["guardrail_suppressed_candidates"] == 0
        assert diag["has_reference"] is True

    def test_guardrail_suppresses_catastrophic_anti_alignment(self) -> None:
        model = _MockModel(dim=8, vocab=32, seed=5)
        monitor = CrctGradientConflictMonitor(
            enabled=True,
            catastrophic_threshold=-0.50,
        )
        hidden = torch.randn(1, 1, 8)
        targets = torch.tensor([[7]])
        utility = torch.tensor([[1.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        sketch = monitor._lm_head_gradient_sketches(
            model=model,
            hidden=hidden,
            targets=targets,
            selected=torch.tensor([0]),
        )[0]
        monitor._ema = (-sketch).detach().cpu()

        write_score, write_limit = monitor.apply_to_write_scores(
            model=model,
            hidden=hidden,
            targets=targets,
            utility=utility,
            mask=mask,
            max_tokens=1,
        )

        assert write_limit == 0
        assert write_score[0, 0].item() == float("-inf")
        diag = monitor.diagnostics()
        assert diag["candidates_compared"] == 1
        assert diag["guardrail_suppressed_candidates"] == 1
        assert diag["last_reason"] == "guardrail_suppressed"
        assert diag["last_conflict_mean"] < -0.99

    def test_trace_writes_bounded_candidate_rows(self, tmp_path) -> None:
        path = tmp_path / "conflict.ndjson"
        model = _MockModel(dim=8, vocab=32, seed=6)
        monitor = CrctGradientConflictMonitor(
            enabled=True,
            trace_path=str(path),
            trace_max_rows=2,
            trace_flush_rows=1,
        )
        hidden = torch.randn(1, 4, 8)
        targets = torch.tensor([[1, 2, 3, 4]])
        utility = torch.tensor([[0.1, 0.5, -0.2, 0.3]])
        mask = torch.ones_like(targets, dtype=torch.bool)

        monitor.apply_to_write_scores(
            model=model,
            hidden=hidden,
            targets=targets,
            utility=utility,
            mask=mask,
            max_tokens=3,
            step=17,
        )
        monitor.flush_trace()

        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 2
        assert rows[0]["row_type"] == "crct_gradient_conflict_candidate"
        assert rows[0]["step"] == 17
        assert rows[0]["candidate_rank"] == 0
        assert "utility" in rows[0]
        assert "conflict_cos" in rows[0]
        assert "gate" in rows[0]
        assert rows[0]["reason"] == "admitted"
        diag = monitor.diagnostics()
        assert diag["trace_rows_written"] == 2
        assert diag["trace_rows_dropped"] == 1


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
            "plasticity_coverage",
            "plasticity_confidence",
            "plasticity_budget",
        }

    def test_exports_memory_packet_when_encoder_returns_meta(self) -> None:
        cache = _MockCache(cutoff=42)
        model = _PacketMockModel(dim=8, vocab=32, seed=3)
        input_ids = torch.randint(0, 32, (2, 6))
        valid_mask = torch.ones((2, 6), dtype=torch.bool)

        out = rank3_score_batch_causal(
            model=model,
            cache=cache,
            input_ids=input_ids,
            valid_mask=valid_mask,
        )

        assert out["memory_residual"].shape == (2, 1, 8)
        assert torch.all(out["memory_residual"] == 1.0)
        assert torch.equal(out["memory_gate"], out["controller_target"])

    def test_rejects_sequence_memory_packet_from_encoder_meta(self) -> None:
        cache = _MockCache(cutoff=42)
        model = _SequencePacketMockModel(dim=8, vocab=32, seed=3)
        input_ids = torch.randint(0, 32, (2, 6))
        valid_mask = torch.ones((2, 6), dtype=torch.bool)

        with pytest.raises(ValueError, match="compact"):
            rank3_score_batch_causal(
                model=model,
                cache=cache,
                input_ids=input_ids,
                valid_mask=valid_mask,
            )

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
        assert out["plasticity_coverage"].shape == (model._dim,)
        assert out["plasticity_confidence"].shape == (model._dim,)
        assert out["plasticity_budget"].shape == (model._dim,)

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

    def test_paired_encode_is_used_when_model_provides_it(self) -> None:
        model = _PairedMockModel()
        cache = _MockCache(cutoff=42)
        ids = torch.randint(0, 32, (2, 6))
        mask = torch.ones_like(ids, dtype=torch.bool)
        out = rank3_score_batch_causal(
            model=model,
            cache=cache,
            input_ids=ids,
            valid_mask=mask,
        )

        assert model.paired_calls == [{"cache_read_cutoff": 42}]
        assert out["memory_residual"].shape == (2, 1, model._dim)

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

        assert cache.reserved == (3, ids.device)
        event_ids = model.append_calls[0]["event_ids"]
        assert event_ids is not None
        assert event_ids.tolist() == [10, 11, 12]

    def test_record_stage_seconds_populates_named_stages(self) -> None:
        # Opt-in stage timer fills a caller-provided dict; if omitted, the
        # call has no host-side sync overhead. We can only smoke-check
        # presence and non-negativity from a CPU mock — the CUDA-event
        # contract is exercised on the GPU3 hot path itself.
        model, cache, ids, mask = self._setup(batch=2, seq=6)
        record: dict[str, float] = {}
        rank3_score_batch_causal(
            model=model,
            cache=cache,
            input_ids=ids,
            valid_mask=mask,
            record_stage_seconds=record,
        )
        for stage in (
            "encode_off",
            "encode_force_on",
            "nll_off",
            "nll_mem",
            "plasticity",
        ):
            assert stage in record, f"stage {stage} missing from record"
            assert record[stage] >= 0.0

    def test_record_stage_seconds_default_is_no_op(self) -> None:
        # Without the dict, no aggregation runs and the result still has the
        # documented top-level keys. Verifies the timing path doesn't leak
        # into the canonical output schema.
        model, cache, ids, mask = self._setup()
        out = rank3_score_batch_causal(
            model=model,
            cache=cache,
            input_ids=ids,
            valid_mask=mask,
        )
        assert "_stage_seconds" not in out

    def test_gradient_conflict_guardrail_can_skip_memory_write(self) -> None:
        model, cache, ids, mask = self._setup(batch=1, seq=5)
        monitor = CrctGradientConflictMonitor(
            enabled=True,
            catastrophic_threshold=1.1,
        )
        monitor._ema = F.normalize(torch.ones(model._dim), dim=0).cpu()

        out = rank3_score_batch_causal(
            model=model,
            cache=cache,
            input_ids=ids,
            valid_mask=mask,
            update_model_memory_after=True,
            memory_write_tokens=3,
            gradient_conflict_monitor=monitor,
        )

        assert len(model.append_calls) == 0
        assert cache.commit_calls, "skipped write still commits the scoring txn"
        assert "write_score" in out
        diag = monitor.diagnostics()
        assert diag["guardrail_suppressed_candidates"] == 3
        assert diag["last_write_token_limit"] == 0

    def test_update_model_memory_writes_slots_newer_than_oracle_cutoff(self) -> None:
        torch.manual_seed(17)
        model = CareStudentLM(
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
