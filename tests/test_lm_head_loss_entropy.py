"""Smoke tests for the per-token-entropy LM-head forward API.

These checks only verify that the Python symbol exists and has the expected
signature. The build + numerical equivalence check against a reference
`H[softmax(logits)]` runs on the CUDA pod in Stage D.4; this file is the
macOS-side guard that catches import/API regressions without CUDA.
"""
from __future__ import annotations

import inspect

import pytest
import torch


def test_fused_lm_head_forward_with_ce_entropy_exists_and_has_expected_signature():
    from chaoscontrol.kernels._lm_head_loss import (
        fused_lm_head_forward_with_ce_entropy,
    )

    sig = inspect.signature(fused_lm_head_forward_with_ce_entropy)
    params = sig.parameters
    assert "x" in params
    assert "weight" in params
    assert "target" in params
    assert "tile_size" in params
    # Return type annotation should be a 4-tuple.
    assert (
        "tuple" in str(sig.return_annotation).lower()
        or sig.return_annotation.__class__.__name__ == "_GenericAlias"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_entropy_matches_softmax_reference_within_fp32_tolerance():
    """Kernel per-token entropy must match -(softmax(logits) * log_softmax(logits)).sum(-1)
    within fp32 tolerance. Pod-only: the kernel is CUDA-only and the entropy-emitting
    entrypoint is compiled into the extension by `pip install -e .` with
    TORCH_CUDA_ARCH_LIST set."""
    import torch.nn.functional as F
    from chaoscontrol.kernels._lm_head_loss import fused_lm_head_forward_with_ce_entropy

    torch.manual_seed(0)
    # Shapes: B*T rows, D features, V vocab. tile_size must divide V for the
    # streaming_cached kernel (validated in the wrapper).
    B, T, D, V = 2, 8, 32, 128
    tile_size = 128  # == V; can also try a smaller divisor.
    x = torch.randn(B * T, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (B * T,), device="cuda", dtype=torch.int64)

    loss, lse, per_token_ce, per_token_entropy = fused_lm_head_forward_with_ce_entropy(
        x, weight, target, tile_size=tile_size,
    )

    # Reference via softmax.
    logits = x @ weight.T  # [B*T, V]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ref_entropy = -(probs * log_probs).sum(dim=-1)

    max_diff = (per_token_entropy - ref_entropy).abs().max().item()
    assert torch.allclose(per_token_entropy, ref_entropy, atol=1e-4, rtol=1e-4), (
        f"max abs diff = {max_diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_entropy_and_ce_coexist_and_are_both_correct():
    """Both outputs from the same call must be individually correct —
    entropy matches softmax-reference AND per-token CE matches F.cross_entropy
    reference. Pins the kernel's dual-output correctness."""
    import torch.nn.functional as F
    from chaoscontrol.kernels._lm_head_loss import fused_lm_head_forward_with_ce_entropy

    torch.manual_seed(1)
    B, T, D, V = 4, 16, 64, 256
    tile_size = 256
    x = torch.randn(B * T, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32)
    target = torch.randint(0, V, (B * T,), device="cuda", dtype=torch.int64)

    loss, lse, per_token_ce, per_token_entropy = fused_lm_head_forward_with_ce_entropy(
        x, weight, target, tile_size=tile_size,
    )

    logits = x @ weight.T
    ref_ce = F.cross_entropy(logits, target, reduction="none")
    ref_entropy = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)

    assert torch.allclose(per_token_ce, ref_ce, atol=1e-4, rtol=1e-4), (
        f"CE mismatch, max abs diff {(per_token_ce - ref_ce).abs().max().item()}"
    )
    assert torch.allclose(per_token_entropy, ref_entropy, atol=1e-4, rtol=1e-4), (
        f"entropy mismatch, max abs diff {(per_token_entropy - ref_entropy).abs().max().item()}"
    )
    # Sanity: scalar loss equals mean of per-token CE.
    assert torch.allclose(loss, per_token_ce.mean(), atol=1e-5, rtol=1e-5)
