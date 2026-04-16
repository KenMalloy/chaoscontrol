"""Gradient-equivalence tests for ``train_ssm.chunked_lm_head_backward``.

The chunked LM-head backward path is what unlocks bs=1024/seq=512 and
bs=512/seq=1024 at V=16384 by never materializing the full ``(B, T, V)``
logits (and its grad) — peak drops 64x at V=16384, dim=256.

These tests lock in the bit-equivalence claim on an isolated tiny model:
real ``RMSNorm`` + ``nn.Linear`` lm_head, comparing the chunked path's
parameter + input gradients against a naive full-logits backward. Any
future regression in the chunking math (reduction order, per-chunk
normalization, detach/require_grad wiring) surfaces here before it
reaches a pod run.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from chaoscontrol.core import RMSNorm
from chaoscontrol.train_ssm import chunked_lm_head_backward


def _make_inputs(
    batch: int,
    seq: int,
    dim: int,
    vocab: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, seq, dim, generator=g, dtype=torch.float32).to(dtype=dtype)
    targets = torch.randint(0, vocab, (batch, seq), generator=g)
    return hidden, targets


def _build_head(dim: int, vocab: int, dtype: torch.dtype, seed: int = 42) -> tuple[RMSNorm, nn.Linear]:
    g = torch.Generator().manual_seed(seed)
    norm = RMSNorm(dim)
    # Initialize norm weight to something non-trivial so its gradient is exercised
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim, generator=g, dtype=torch.float32) * 0.1 + 1.0)
    lm_head = nn.Linear(dim, vocab, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(
            torch.randn(vocab, dim, generator=g, dtype=torch.float32) * (dim ** -0.5)
        )
    norm = norm.to(dtype=dtype)
    lm_head = lm_head.to(dtype=dtype)
    return norm, lm_head


def _naive_backward(
    hidden: torch.Tensor,
    norm: RMSNorm,
    lm_head: nn.Linear,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference path: materialize full logits, single CE + backward.

    Returns (loss, hidden_grad, lm_head_weight_grad, norm_weight_grad).
    """
    vocab = lm_head.out_features
    h = hidden.clone().detach().requires_grad_(True)
    logits = lm_head(norm(h))  # (B, T, V)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab).float(),
        targets.reshape(-1),
        reduction="mean",
    )
    loss.backward()
    return (
        loss.detach(),
        h.grad.detach().clone(),
        lm_head.weight.grad.detach().clone(),
        norm.weight.grad.detach().clone(),
    )


class TestChunkedLMBackwardFP32:
    """Gradient parity at fp32 — the tight-tolerance regime."""

    def test_hidden_and_param_grads_match_naive(self) -> None:
        hidden, targets = _make_inputs(batch=4, seq=32, dim=16, vocab=64, dtype=torch.float32, seed=1)

        # Naive path: shared weights state A
        norm_a, head_a = _build_head(dim=16, vocab=64, dtype=torch.float32, seed=42)
        loss_naive, hgrad_naive, whead_naive, wnorm_naive = _naive_backward(
            hidden, norm_a, head_a, targets,
        )

        # Chunked path: independent copies of the same weights (state B)
        norm_b = copy.deepcopy(norm_a)
        head_b = copy.deepcopy(head_a)
        # Zero grads explicitly (deepcopy preserves them)
        norm_b.weight.grad = None
        head_b.weight.grad = None

        h_for_ce = hidden.clone().detach().requires_grad_(True)
        loss_chunked = chunked_lm_head_backward(
            hidden=h_for_ce,
            final_norm=norm_b,
            lm_head=head_b,
            targets=targets,
            chunk_size=8,  # 4 chunks over seq=32
        )

        # Forward loss parity — fp32 should match within a few ULPs
        assert torch.allclose(loss_naive, loss_chunked, atol=0.0, rtol=1e-5), (
            f"loss mismatch: naive={loss_naive.item()}, chunked={loss_chunked.item()}"
        )
        # hidden grad parity
        hdiff = (hgrad_naive - h_for_ce.grad).abs().max().item()
        assert hdiff < 1e-6, f"hidden.grad max diff {hdiff}"
        # lm_head weight grad parity
        wdiff = (whead_naive - head_b.weight.grad).abs().max().item()
        assert wdiff < 1e-6, f"lm_head.weight.grad max diff {wdiff}"
        # final_norm weight grad parity
        ndiff = (wnorm_naive - norm_b.weight.grad).abs().max().item()
        assert ndiff < 1e-6, f"final_norm.weight.grad max diff {ndiff}"

    def test_chunk_size_not_divisor_of_seq(self) -> None:
        # seq=30 / chunk=8 = 3 full chunks + 1 partial of 6
        hidden, targets = _make_inputs(batch=2, seq=30, dim=8, vocab=32, dtype=torch.float32, seed=2)
        norm_a, head_a = _build_head(dim=8, vocab=32, dtype=torch.float32, seed=43)
        loss_naive, hgrad_naive, whead_naive, wnorm_naive = _naive_backward(
            hidden, norm_a, head_a, targets,
        )
        norm_b = copy.deepcopy(norm_a)
        head_b = copy.deepcopy(head_a)
        norm_b.weight.grad = None
        head_b.weight.grad = None

        h_for_ce = hidden.clone().detach().requires_grad_(True)
        loss_chunked = chunked_lm_head_backward(
            hidden=h_for_ce,
            final_norm=norm_b,
            lm_head=head_b,
            targets=targets,
            chunk_size=8,
        )

        assert torch.allclose(loss_naive, loss_chunked, atol=0.0, rtol=1e-5)
        assert (hgrad_naive - h_for_ce.grad).abs().max().item() < 1e-6
        assert (whead_naive - head_b.weight.grad).abs().max().item() < 1e-6
        assert (wnorm_naive - norm_b.weight.grad).abs().max().item() < 1e-6

    def test_chunk_size_equal_to_seq_one_chunk(self) -> None:
        # Degenerate case: chunk_size >= seq → single chunk, should still match.
        hidden, targets = _make_inputs(batch=2, seq=16, dim=8, vocab=32, dtype=torch.float32, seed=3)
        norm_a, head_a = _build_head(dim=8, vocab=32, dtype=torch.float32, seed=44)
        loss_naive, hgrad_naive, whead_naive, wnorm_naive = _naive_backward(
            hidden, norm_a, head_a, targets,
        )
        norm_b = copy.deepcopy(norm_a)
        head_b = copy.deepcopy(head_a)
        norm_b.weight.grad = None
        head_b.weight.grad = None

        h_for_ce = hidden.clone().detach().requires_grad_(True)
        loss_chunked = chunked_lm_head_backward(
            hidden=h_for_ce,
            final_norm=norm_b,
            lm_head=head_b,
            targets=targets,
            chunk_size=999,  # > seq, so one chunk
        )

        assert torch.allclose(loss_naive, loss_chunked, atol=0.0, rtol=1e-5)
        assert (hgrad_naive - h_for_ce.grad).abs().max().item() < 1e-6
        assert (whead_naive - head_b.weight.grad).abs().max().item() < 1e-6
        assert (wnorm_naive - norm_b.weight.grad).abs().max().item() < 1e-6


class TestChunkedLMBackwardBF16:
    """Gradient parity at bf16 — tolerance tracks the ~3-digit mantissa.

    bf16 max-abs grad diffs in the 1e-3 range are numerical-equivalence;
    tighter tolerances would be noise-sensitive.
    """

    def test_bf16_grads_within_tolerance(self) -> None:
        hidden, targets = _make_inputs(
            batch=4, seq=32, dim=16, vocab=64, dtype=torch.bfloat16, seed=5,
        )
        norm_a, head_a = _build_head(dim=16, vocab=64, dtype=torch.bfloat16, seed=42)
        loss_naive, hgrad_naive, whead_naive, wnorm_naive = _naive_backward(
            hidden, norm_a, head_a, targets,
        )
        norm_b = copy.deepcopy(norm_a)
        head_b = copy.deepcopy(head_a)
        norm_b.weight.grad = None
        head_b.weight.grad = None

        h_for_ce = hidden.clone().detach().requires_grad_(True)
        loss_chunked = chunked_lm_head_backward(
            hidden=h_for_ce,
            final_norm=norm_b,
            lm_head=head_b,
            targets=targets,
            chunk_size=8,
        )

        # bf16 loss scalar: 1e-2 tolerance at loss magnitudes ~4-5
        assert torch.allclose(loss_naive.float(), loss_chunked.float(), atol=5e-2, rtol=1e-2), (
            f"bf16 loss: naive={loss_naive.item()}, chunked={loss_chunked.item()}"
        )
        hdiff = (hgrad_naive.float() - h_for_ce.grad.float()).abs().max().item()
        assert hdiff < 5e-3, f"bf16 hidden.grad max diff {hdiff}"
        wdiff = (whead_naive.float() - head_b.weight.grad.float()).abs().max().item()
        assert wdiff < 5e-3, f"bf16 lm_head.weight.grad max diff {wdiff}"
        ndiff = (wnorm_naive.float() - norm_b.weight.grad.float()).abs().max().item()
        assert ndiff < 5e-3, f"bf16 final_norm.weight.grad max diff {ndiff}"


class TestChunkedLMBackwardContract:
    """API-contract checks — returned dtype, grad accumulation, no leaks."""

    def test_loss_returned_is_fp32_scalar(self) -> None:
        # Consistent with chunked_cross_entropy: helper returns fp32 regardless
        # of input dtype, so the caller gets a stably-typed scalar for logging.
        hidden, targets = _make_inputs(batch=2, seq=8, dim=8, vocab=16, dtype=torch.bfloat16, seed=6)
        norm, head = _build_head(dim=8, vocab=16, dtype=torch.bfloat16, seed=46)
        h_for_ce = hidden.clone().detach().requires_grad_(True)
        loss = chunked_lm_head_backward(
            hidden=h_for_ce,
            final_norm=norm,
            lm_head=head,
            targets=targets,
            chunk_size=4,
        )
        assert loss.dtype == torch.float32, f"expected fp32 loss, got {loss.dtype}"
        assert loss.dim() == 0, f"expected scalar, got shape {tuple(loss.shape)}"

    def test_rejects_zero_chunk_size(self) -> None:
        # chunk_size <= 0 would cause the chunk loop to never advance,
        # wedging the process until a timeout. Guard at entry so a bad
        # config fails fast and surfaces clearly rather than hanging.
        hidden, targets = _make_inputs(batch=1, seq=4, dim=4, vocab=8, dtype=torch.float32, seed=20)
        norm, head = _build_head(dim=4, vocab=8, dtype=torch.float32, seed=50)
        h = hidden.clone().detach().requires_grad_(True)
        with pytest.raises(ValueError, match="chunk_size"):
            chunked_lm_head_backward(
                hidden=h, final_norm=norm, lm_head=head,
                targets=targets, chunk_size=0,
            )

    def test_rejects_negative_chunk_size(self) -> None:
        hidden, targets = _make_inputs(batch=1, seq=4, dim=4, vocab=8, dtype=torch.float32, seed=21)
        norm, head = _build_head(dim=4, vocab=8, dtype=torch.float32, seed=51)
        h = hidden.clone().detach().requires_grad_(True)
        with pytest.raises(ValueError, match="chunk_size"):
            chunked_lm_head_backward(
                hidden=h, final_norm=norm, lm_head=head,
                targets=targets, chunk_size=-4,
            )

    def test_hidden_grad_populated_after_call(self) -> None:
        # After the call, the passed-in hidden tensor must have .grad populated
        # — the caller relies on this to feed gradient into the encoder.
        hidden, targets = _make_inputs(batch=2, seq=8, dim=8, vocab=16, dtype=torch.float32, seed=7)
        norm, head = _build_head(dim=8, vocab=16, dtype=torch.float32, seed=47)
        h_for_ce = hidden.clone().detach().requires_grad_(True)
        assert h_for_ce.grad is None, "precondition: hidden has no grad yet"
        chunked_lm_head_backward(
            hidden=h_for_ce,
            final_norm=norm,
            lm_head=head,
            targets=targets,
            chunk_size=4,
        )
        assert h_for_ce.grad is not None, "hidden.grad must be populated"
        assert h_for_ce.grad.shape == hidden.shape
