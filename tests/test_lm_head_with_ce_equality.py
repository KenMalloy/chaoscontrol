"""Regression test pinning ``fused_lm_head_backward`` vs ``_with_ce`` equality.

Phase 1 Task 1.4 introduces a conditional swap to
``fused_lm_head_backward_with_ce`` when ``episodic_rings is not None``.
The two helpers MUST produce bit-identical loss and gradients on the
production fused path (``final_norm`` has a weight). Without this test,
a future kernel-dispatch refactor that touches one but not the other
silently desyncs episodic mode from non-episodic — only Phase 3's
falsifier matrix would catch it, and that's a 5-hour 4×H100 wakeup.
This is a sub-second regression guard.

Originally this test ran on CPU via the silent dispatcher fallback. The
fallback is gone (it OOM'd in production via fp32 logits materialization),
so the equality check now requires CUDA + the built native extension and
skips on dev macs.
"""
from __future__ import annotations

import pytest
import torch

import chaoscontrol.kernels._lm_head_loss as _lm_head_loss
from chaoscontrol.train_ssm import (
    fused_lm_head_backend_for_mode,
    fused_lm_head_backward,
    fused_lm_head_backward_with_ce,
)


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="fused LM-head dispatchers no longer fall back on CPU",
    ),
    pytest.mark.skipif(
        _lm_head_loss._C is None,
        reason="chaoscontrol.kernels._lm_head_loss._C not built on this machine",
    ),
]


_FUSED_BACKWARD_MODES = (
    "fused",
    "fused_streaming",
    "fused_streaming_v2",
    "fused_streaming_cached",
    "fused_norm_streaming_v2",
)


def _make_inputs(seed: int = 17):
    """Build matched (hidden, RMSNorm, Linear, targets) on CUDA in bf16 so
    the dispatcher's eligibility predicates accept us on a built pod."""
    torch.manual_seed(seed)
    B, T, D, V = 2, 4, 16, 32
    device = torch.device("cuda")
    hidden = torch.randn(B, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    final_norm = torch.nn.RMSNorm(D, eps=1e-6).to(device=device, dtype=torch.bfloat16)
    lm_head = torch.nn.Linear(D, V, bias=False).to(device=device, dtype=torch.bfloat16)
    targets = torch.randint(0, V, (B, T), dtype=torch.long, device=device)
    return hidden, final_norm, lm_head, targets


def _run_backward(fn, mode, *, hidden, final_norm, lm_head, targets):
    """Invoke ``fn`` once and snapshot loss + grads on every parameter
    that participated in the backward pass."""
    backend = fused_lm_head_backend_for_mode(mode)
    # tile_size must divide vocab (V=32) for the streaming_cached predicate;
    # 8 divides 32 and matches the other backends as well.
    result = fn(
        hidden=hidden,
        final_norm=final_norm,
        lm_head=lm_head,
        targets=targets,
        backend=backend,
        tile_size=8,
    )
    loss = result[0] if isinstance(result, tuple) else result
    return {
        "loss": loss.detach().clone(),
        "hidden_grad": (
            hidden.grad.detach().clone() if hidden.grad is not None else None
        ),
        "norm_grad": (
            final_norm.weight.grad.detach().clone()
            if final_norm.weight.grad is not None
            else None
        ),
        "lm_head_grad": (
            lm_head.weight.grad.detach().clone()
            if lm_head.weight.grad is not None
            else None
        ),
    }


@pytest.mark.parametrize("mode", _FUSED_BACKWARD_MODES)
def test_with_ce_matches_plain_loss_and_grads_per_backend(mode):
    """Both helpers must produce identical loss + grads on the production
    fused path. Pinned across all five fused backends."""
    h_a, n_a, l_a, t_a = _make_inputs()
    out_a = _run_backward(
        fused_lm_head_backward, mode,
        hidden=h_a, final_norm=n_a, lm_head=l_a, targets=t_a,
    )

    h_b, n_b, l_b, t_b = _make_inputs()  # same seed → identical init
    out_b = _run_backward(
        fused_lm_head_backward_with_ce, mode,
        hidden=h_b, final_norm=n_b, lm_head=l_b, targets=t_b,
    )

    torch.testing.assert_close(
        out_a["loss"], out_b["loss"], rtol=1e-5, atol=1e-6,
        msg=f"loss mismatch in backend={mode!r}",
    )
    torch.testing.assert_close(
        out_a["hidden_grad"], out_b["hidden_grad"], rtol=1e-5, atol=1e-6,
        msg=f"hidden.grad mismatch in backend={mode!r}",
    )
    torch.testing.assert_close(
        out_a["norm_grad"], out_b["norm_grad"], rtol=1e-5, atol=1e-6,
        msg=f"final_norm.weight.grad mismatch in backend={mode!r}",
    )
    torch.testing.assert_close(
        out_a["lm_head_grad"], out_b["lm_head_grad"], rtol=1e-5, atol=1e-6,
        msg=f"lm_head.weight.grad mismatch in backend={mode!r}",
    )


@pytest.mark.parametrize("mode", _FUSED_BACKWARD_MODES)
def test_with_ce_per_token_ce_means_to_scalar_loss(mode):
    """The per-token CE returned by ``_with_ce`` must mean to the same
    scalar that the helper returns. This is a contract check on the
    kernel — if the per-token tensor and the scalar disagree, ScOpt's
    pressure signal would silently lie."""
    hidden, final_norm, lm_head, targets = _make_inputs()
    backend = fused_lm_head_backend_for_mode(mode)
    loss, per_token_ce = fused_lm_head_backward_with_ce(
        hidden=hidden,
        final_norm=final_norm,
        lm_head=lm_head,
        targets=targets,
        backend=backend,
        tile_size=8,
    )
    torch.testing.assert_close(
        per_token_ce.float().mean(),
        loss.float(),
        rtol=1e-5,
        atol=1e-6,
        msg=f"per_token_ce.mean() != loss in backend={mode!r}",
    )
