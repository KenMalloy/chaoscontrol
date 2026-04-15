"""Parity tests for ``chaoscontrol.training.chunked_cross_entropy_mean``.

The chunked helper is a drop-in replacement for
``F.cross_entropy(logits, targets, reduction='mean')`` that avoids the
34 GiB fp32 upcast allocation at bs=1024/seq=512/V=16384 by computing
the loss in position-dim chunks. These tests lock in the bit-exactness
claim from the docstring so any future regression — e.g. someone
accidentally changes the accumulator dtype or the reduction — is caught
before a pod run wastes time on drifted numerics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from chaoscontrol.training import chunked_cross_entropy, chunked_cross_entropy_mean


def _make_inputs(
    n: int,
    v: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, v, generator=g, dtype=torch.float32).to(dtype=dtype)
    targets = torch.randint(0, v, (n,), generator=g)
    return logits, targets


class TestChunkedCrossEntropyForward:
    """Forward value parity at multiple chunk sizes and shapes."""

    def test_fp32_within_ulp_small_shapes(self) -> None:
        # At fp32, sum-then-divide is mathematically equivalent to mean
        # but NOT bit-exact against F.cross_entropy's internal tree
        # reduction — the chunked path's summation order (tree-reduce
        # within each chunk + left-to-right across chunks) can differ
        # from F.cross_entropy's internal ordering by a few ULPs. 1e-5
        # relative is ~20 ULPs at loss magnitudes ~5, well inside
        # numerical-equivalence territory for a loss value.
        logits, targets = _make_inputs(n=256, v=128, dtype=torch.float32, seed=1)
        full = F.cross_entropy(logits, targets, reduction="mean")
        for chunk_size in (1, 16, 64, 128, 256, 1024):
            chunked = chunked_cross_entropy_mean(logits, targets, chunk_size=chunk_size)
            assert torch.allclose(full, chunked, atol=0.0, rtol=1e-5), (
                f"fp32 chunk_size={chunk_size}: full={full.item()}, "
                f"chunked={chunked.item()}, diff={(full - chunked).item()}"
            )

    def test_fp32_within_ulp_large_n(self) -> None:
        # Exercise a shape more representative of the Exp 18 regime
        # (n >> chunk_size so multiple chunks actually form).
        logits, targets = _make_inputs(n=4096, v=512, dtype=torch.float32, seed=2)
        full = F.cross_entropy(logits, targets, reduction="mean")
        chunked = chunked_cross_entropy_mean(logits, targets, chunk_size=128)
        assert torch.allclose(full, chunked, atol=0.0, rtol=1e-5), (
            f"fp32 n=4096/chunk=128: full={full.item()}, chunked={chunked.item()}"
        )

    def test_bf16_within_roundtrip_tolerance(self) -> None:
        # bf16 has ~7-bit mantissa, so the chunked path's explicit
        # fp32-upcast per chunk followed by scalar accumulation is not
        # guaranteed to be bit-exact against F.cross_entropy's internal
        # single-upcast path — the non-chunked reference upcasts once
        # and reduces in fp32, then downcasts to bf16 at the very end,
        # while the chunked path sums fp32 per-chunk scalars. Both are
        # mathematically equivalent but floating-point summation order
        # can differ in the last bit of bf16 precision.
        #
        # The helper returns fp32 scalar unconditionally (matches autocast
        # contract); stock F.cross_entropy returns bf16 for bf16 input
        # outside autocast. Upcast the reference to compare.
        logits, targets = _make_inputs(n=1024, v=128, dtype=torch.bfloat16, seed=3)
        full = F.cross_entropy(logits, targets, reduction="mean").float()
        chunked = chunked_cross_entropy_mean(logits, targets, chunk_size=64)
        assert chunked.dtype == torch.float32, (
            f"helper should return fp32 unconditionally; got {chunked.dtype}"
        )
        # bf16 unit-in-last-place at loss magnitudes ~4-5 is ~1/64 ≈ 0.015
        assert torch.allclose(full, chunked, atol=2e-2, rtol=1e-2), (
            f"bf16: full={full.item()}, chunked={chunked.item()}, "
            f"diff={(full - chunked).item()}"
        )

    def test_chunk_size_equal_to_n_degenerates_to_single_pass(self) -> None:
        # With chunk_size >= n, there's exactly one chunk. The helper
        # still upcasts the chunk to fp32 explicitly (which is a no-op
        # when the input is already fp32) and wraps the reduction in
        # its own accumulator, so the result may differ from the
        # non-chunked reference by a ULP or two in the division order.
        logits, targets = _make_inputs(n=100, v=64, dtype=torch.float32, seed=4)
        full = F.cross_entropy(logits, targets, reduction="mean")
        chunked_single_pass = chunked_cross_entropy_mean(
            logits, targets, chunk_size=100_000,
        )
        assert torch.allclose(full, chunked_single_pass, atol=0.0, rtol=1e-5)


class TestChunkedCrossEntropyGradient:
    """Gradient parity — the load-bearing property for training."""

    def test_fp32_gradient_matches(self) -> None:
        logits, targets = _make_inputs(n=512, v=64, dtype=torch.float32, seed=5)
        logits_a = logits.clone().requires_grad_(True)
        logits_b = logits.clone().requires_grad_(True)
        full_loss = F.cross_entropy(logits_a, targets, reduction="mean")
        chunked_loss = chunked_cross_entropy_mean(logits_b, targets, chunk_size=64)
        full_grad = torch.autograd.grad(full_loss, logits_a)[0]
        chunked_grad = torch.autograd.grad(chunked_loss, logits_b)[0]
        max_diff = (full_grad - chunked_grad).abs().max().item()
        assert max_diff < 1e-6, (
            f"fp32 grad max_diff={max_diff}, full_loss={full_loss.item()}, "
            f"chunked_loss={chunked_loss.item()}"
        )

    def test_bf16_gradient_within_tolerance(self) -> None:
        # bf16 gradients have ~3 digit precision per element. The chunked
        # path's per-chunk fp32 upcast means the local gradient within
        # each chunk is computed at fp32 precision and then downcast.
        # Total accumulated gradient difference vs the non-chunked path
        # is bounded by bf16's last-bit precision at logit scale (~1e-2
        # absolute, ~1e-3 relative for well-conditioned inputs).
        logits, targets = _make_inputs(n=512, v=128, dtype=torch.bfloat16, seed=6)
        logits_a = logits.clone().requires_grad_(True)
        logits_b = logits.clone().requires_grad_(True)
        full_loss = F.cross_entropy(logits_a, targets, reduction="mean")
        chunked_loss = chunked_cross_entropy_mean(logits_b, targets, chunk_size=64)
        full_grad = torch.autograd.grad(full_loss, logits_a)[0]
        chunked_grad = torch.autograd.grad(chunked_loss, logits_b)[0]
        max_diff = (full_grad.float() - chunked_grad.float()).abs().max().item()
        # bf16 ulp at typical logit magnitudes is ~1e-3; full scan of
        # 512x128 elements can accumulate to ~1e-2 max-abs difference.
        assert max_diff < 5e-3, (
            f"bf16 grad max_diff={max_diff}, full_loss={full_loss.item()}, "
            f"chunked_loss={chunked_loss.item()}"
        )


class TestChunkedCrossEntropySumReduction:
    """Parity for the reduction='sum' variant used by the eval path in
    ``runner_exp17.evaluate_bpb_sp``. Eval accumulates per-batch CE nats
    across many batches and can't use the mean reduction directly.
    """

    def test_sum_matches_stock_ce_fp32(self) -> None:
        logits, targets = _make_inputs(n=1024, v=128, dtype=torch.float32, seed=11)
        full = F.cross_entropy(logits, targets, reduction="sum")
        chunked = chunked_cross_entropy(
            logits, targets, reduction="sum", chunk_size=64,
        )
        # Same ULP tolerance as the mean tests — sum-path precision is
        # driven by the same fp32 reduction semantics.
        assert torch.allclose(full, chunked, atol=0.0, rtol=1e-5), (
            f"fp32 sum: full={full.item()}, chunked={chunked.item()}, "
            f"diff={(full - chunked).item()}"
        )

    def test_sum_equals_mean_times_n_fp32(self) -> None:
        # Consistency check: sum reduction equals mean reduction * N.
        logits, targets = _make_inputs(n=512, v=64, dtype=torch.float32, seed=12)
        mean_val = chunked_cross_entropy(logits, targets, reduction="mean", chunk_size=64)
        sum_val = chunked_cross_entropy(logits, targets, reduction="sum", chunk_size=64)
        n = logits.size(0)
        assert torch.allclose(sum_val, mean_val * n, atol=0.0, rtol=1e-6), (
            f"sum={sum_val.item()} should equal mean * n = {mean_val.item() * n}"
        )

    def test_sum_matches_stock_ce_bf16(self) -> None:
        logits, targets = _make_inputs(n=1024, v=128, dtype=torch.bfloat16, seed=13)
        full = F.cross_entropy(logits, targets, reduction="sum").float()
        chunked = chunked_cross_entropy(
            logits, targets, reduction="sum", chunk_size=64,
        )
        assert chunked.dtype == torch.float32
        # bf16 sum at ~5 per element × 1024 elements = ~5000, ulp ~16
        # across the sum; chunked path should stay within a few ULPs.
        assert torch.allclose(full, chunked, atol=50.0, rtol=1e-2), (
            f"bf16 sum: full={full.item()}, chunked={chunked.item()}, "
            f"diff={(full - chunked).item()}"
        )

    def test_sum_reduction_rejects_invalid(self) -> None:
        logits, targets = _make_inputs(n=16, v=8, dtype=torch.float32, seed=14)
        try:
            chunked_cross_entropy(logits, targets, reduction="none")
        except ValueError as e:
            assert "reduction" in str(e)
        else:
            raise AssertionError("should have raised on reduction='none'")


class TestChunkedCrossEntropyAutocast:
    """The drop-in-under-autocast contract that the Exp 18 training loop
    actually relies on.

    Inside ``torch.autocast(dtype=torch.bfloat16)``, stock F.cross_entropy
    upcasts its scalar loss to fp32 even when the input logits are bf16.
    The helper must behave the same way so the caller at training.py:472
    (which runs under maybe_autocast) gets a consistently-typed scalar
    loss regardless of which path it's using. Catches any future edit
    that accidentally reintroduces a ``.to(dtype=logits_flat.dtype)``
    downcast at the end of the helper.
    """

    def test_helper_returns_fp32_scalar_under_autocast(self) -> None:
        logits, targets = _make_inputs(n=512, v=128, dtype=torch.float32, seed=9)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            chunked = chunked_cross_entropy_mean(logits, targets, chunk_size=64)
        assert chunked.dtype == torch.float32, (
            f"expected fp32 scalar under autocast, got {chunked.dtype}"
        )

    def test_helper_matches_stock_ce_under_autocast(self) -> None:
        # Both operations are wrapped in autocast(bf16). Stock F.cross_entropy
        # returns fp32 under autocast; the helper also returns fp32
        # unconditionally. They should agree to numerical-equivalence
        # tolerance on both forward and gradient.
        logits_a, targets_a = _make_inputs(n=512, v=128, dtype=torch.float32, seed=10)
        logits_b = logits_a.clone()
        logits_a = logits_a.detach().requires_grad_(True)
        logits_b = logits_b.detach().requires_grad_(True)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            full_loss = F.cross_entropy(logits_a, targets_a, reduction="mean")
            chunked_loss = chunked_cross_entropy_mean(
                logits_b, targets_a, chunk_size=64,
            )

        assert full_loss.dtype == torch.float32, (
            f"stock F.cross_entropy should return fp32 under autocast, "
            f"got {full_loss.dtype}"
        )
        assert chunked_loss.dtype == torch.float32, (
            f"helper should return fp32 under autocast, got {chunked_loss.dtype}"
        )
        # Forward values match within bf16 noise — both paths compute the
        # softmax in bf16 internally under autocast.
        forward_diff = (full_loss - chunked_loss).abs().item()
        assert forward_diff < 5e-2, (
            f"autocast forward diff {forward_diff} exceeds bf16 noise floor; "
            f"full={full_loss.item()}, chunked={chunked_loss.item()}"
        )

        # Gradients should match within bf16 gradient noise.
        full_grad = torch.autograd.grad(full_loss, logits_a, retain_graph=True)[0]
        chunked_grad = torch.autograd.grad(chunked_loss, logits_b)[0]
        grad_diff = (full_grad.float() - chunked_grad.float()).abs().max().item()
        assert grad_diff < 5e-3, (
            f"autocast gradient max diff {grad_diff} exceeds bf16 noise floor"
        )


class TestChunkedCrossEntropyEdgeCases:
    """Shapes and configurations that could trip up the helper."""

    def test_n_equal_one(self) -> None:
        # Single-element input — no reduction needed but the helper
        # should still match F.cross_entropy.
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        targets = torch.tensor([2])
        full = F.cross_entropy(logits, targets, reduction="mean")
        chunked = chunked_cross_entropy_mean(logits, targets, chunk_size=16)
        assert torch.allclose(full, chunked, atol=0.0, rtol=1e-6)

    def test_n_not_divisible_by_chunk_size(self) -> None:
        # Last chunk is smaller than chunk_size — caller of the helper
        # must correctly handle the final partial slice without dropping
        # or double-counting any elements.
        logits, targets = _make_inputs(n=257, v=32, dtype=torch.float32, seed=7)
        full = F.cross_entropy(logits, targets, reduction="mean")
        chunked = chunked_cross_entropy_mean(logits, targets, chunk_size=64)
        assert torch.allclose(full, chunked, atol=0.0, rtol=1e-5), (
            f"n=257, chunk=64 (4 full chunks of 64 + 1 chunk of 1): "
            f"full={full.item()}, chunked={chunked.item()}"
        )

    def test_deterministic_across_calls(self) -> None:
        # Re-running the helper on the same inputs must produce the
        # same output — no hidden state, no nondeterministic reductions.
        # This one IS bit-exact because same summation order both times.
        logits, targets = _make_inputs(n=128, v=64, dtype=torch.float32, seed=8)
        a = chunked_cross_entropy_mean(logits, targets, chunk_size=32)
        b = chunked_cross_entropy_mean(logits, targets, chunk_size=32)
        assert torch.equal(a, b)
