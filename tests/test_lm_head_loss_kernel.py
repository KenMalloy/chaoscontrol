"""Tests for native LM-head/loss helper fallbacks and build wiring."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import chaoscontrol.kernels._lm_head_loss as lm_head_loss
from chaoscontrol.kernels._lm_head_loss import (
    fused_linear_cross_entropy,
    fused_linear_cross_entropy_with_ce,
    fused_linear_cross_entropy_weighted,
    fused_linear_cross_entropy_weighted_with_ce,
    fused_rms_linear_cross_entropy,
    fused_rms_linear_cross_entropy_with_ce,
    fused_rms_linear_cross_entropy_weighted,
    fused_rms_linear_cross_entropy_weighted_with_ce,
    fused_rms_norm,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_setup_ext():
    path = ROOT / "src" / "chaoscontrol" / "kernels" / "_lm_head_loss" / "setup_ext.py"
    spec = importlib.util.spec_from_file_location("lm_head_loss_setup_ext_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.rms_norm(x.float(), (x.size(-1),), eps=eps).to(x.dtype) * weight


def _reference_linear_ce(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = x.reshape(-1, x.size(-1)) @ weight.t()
    return F.cross_entropy(logits.float(), targets.reshape(-1), reduction="mean")


def _reference_weighted_linear_ce(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    token_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = x.reshape(-1, x.size(-1)) @ weight.t()
    per_token = F.cross_entropy(
        logits.float(),
        targets.reshape(-1),
        reduction="none",
    )
    flat_weight = token_weight.reshape(-1).detach().float()
    loss = (per_token * flat_weight).sum() / flat_weight.sum().clamp_min(1.0)
    return loss, per_token


def test_fused_rms_norm_cpu_fallback_matches_reference_forward_and_backward():
    torch.manual_seed(0)
    x_ref = torch.randn(3, 5, 7, requires_grad=True)
    w_ref = (torch.randn(7) * 0.1 + 1.0).requires_grad_(True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_ref)

    y_ref = _reference_rms_norm(x_ref, w_ref, eps=1e-6)
    y_new = fused_rms_norm(x_new, w_new, eps=1e-6)

    assert torch.allclose(y_ref, y_new, atol=0.0, rtol=0.0)
    y_ref.backward(grad)
    y_new.backward(grad)

    assert torch.allclose(x_ref.grad, x_new.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=0.0, rtol=0.0)


def test_lm_head_loss_build_hook_honors_cuda_arch_env(monkeypatch):
    module = _load_setup_ext()
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9;9.0")

    assert module._nvcc_gencode_args() == [
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]


def test_fused_linear_cross_entropy_cpu_fallback_matches_reference_backward():
    torch.manual_seed(2)
    x_ref = torch.randn(3, 5, 7, requires_grad=True)
    w_ref = (torch.randn(11, 7) * 0.1).requires_grad_(True)
    targets = torch.randint(0, 11, (3, 5), dtype=torch.long)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    loss_ref = _reference_linear_ce(x_ref, w_ref, targets)
    loss_new = fused_linear_cross_entropy(
        x_new,
        w_new,
        targets,
        reduction="mean",
        tile_size=4,
    )

    assert torch.allclose(loss_ref, loss_new, atol=0.0, rtol=0.0)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=0.0, rtol=0.0)


def test_fused_linear_cross_entropy_streaming_cpu_fallback_matches_reference_backward():
    torch.manual_seed(22)
    x_ref = torch.randn(3, 5, 7, requires_grad=True)
    w_ref = (torch.randn(11, 7) * 0.1).requires_grad_(True)
    targets = torch.randint(0, 11, (3, 5), dtype=torch.long)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    loss_ref = _reference_linear_ce(x_ref, w_ref, targets)
    loss_new = fused_linear_cross_entropy(
        x_new,
        w_new,
        targets,
        reduction="mean",
        tile_size=4,
        backend="streaming",
    )

    assert torch.allclose(loss_ref, loss_new, atol=0.0, rtol=0.0)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=0.0, rtol=0.0)


def test_fused_linear_cross_entropy_streaming_v2_cpu_fallback_matches_reference_backward():
    torch.manual_seed(222)
    x_ref = torch.randn(3, 5, 7, requires_grad=True)
    w_ref = (torch.randn(11, 7) * 0.1).requires_grad_(True)
    targets = torch.randint(0, 11, (3, 5), dtype=torch.long)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    loss_ref = _reference_linear_ce(x_ref, w_ref, targets)
    loss_new = fused_linear_cross_entropy(
        x_new,
        w_new,
        targets,
        reduction="mean",
        tile_size=4,
        backend="streaming_v2",
    )

    assert torch.allclose(loss_ref, loss_new, atol=0.0, rtol=0.0)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=0.0, rtol=0.0)


def test_fused_linear_cross_entropy_streaming_cached_cpu_fallback_matches_reference_backward():
    torch.manual_seed(224)
    x_ref = torch.randn(3, 5, 7, requires_grad=True)
    w_ref = (torch.randn(12, 7) * 0.1).requires_grad_(True)
    targets = torch.randint(0, 12, (3, 5), dtype=torch.long)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    loss_ref = _reference_linear_ce(x_ref, w_ref, targets)
    loss_new = fused_linear_cross_entropy(
        x_new,
        w_new,
        targets,
        reduction="mean",
        tile_size=4,
        backend="streaming_cached",
    )

    assert torch.allclose(loss_ref, loss_new, atol=0.0, rtol=0.0)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("backend", ["auto", "streaming", "streaming_v2", "streaming_cached"])
def test_fused_linear_cross_entropy_weighted_cpu_fallback_matches_reference_backward(
    backend: str,
):
    torch.manual_seed(225)
    x_ref = torch.randn(3, 5, 7, requires_grad=True)
    w_ref = (torch.randn(12, 7) * 0.1).requires_grad_(True)
    targets = torch.randint(0, 12, (3, 5), dtype=torch.long)
    token_weight = torch.rand(3, 5).requires_grad_(True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    loss_ref, per_token_ref = _reference_weighted_linear_ce(
        x_ref,
        w_ref,
        targets,
        token_weight,
    )
    loss_new, per_token_ce = fused_linear_cross_entropy_weighted_with_ce(
        x_new,
        w_new,
        targets,
        token_weight=token_weight,
        tile_size=4,
        backend=backend,
    )

    assert torch.allclose(loss_ref, loss_new, atol=0.0, rtol=0.0)
    assert torch.allclose(per_token_ce, per_token_ref.detach(), atol=0.0, rtol=0.0)
    assert not per_token_ce.requires_grad
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=0.0, rtol=0.0)
    assert token_weight.grad is None


def test_fused_linear_cross_entropy_weighted_wrapper_matches_with_ce_loss():
    torch.manual_seed(226)
    x = torch.randn(2, 4, 6, requires_grad=True)
    w = (torch.randn(9, 6) * 0.1).requires_grad_(True)
    targets = torch.randint(0, 9, (2, 4), dtype=torch.long)
    token_weight = torch.rand(2, 4)

    loss_only = fused_linear_cross_entropy_weighted(
        x,
        w,
        targets,
        token_weight=token_weight,
        tile_size=3,
    )
    loss_with_ce, _ = fused_linear_cross_entropy_weighted_with_ce(
        x,
        w,
        targets,
        token_weight=token_weight,
        tile_size=3,
    )

    assert torch.allclose(loss_only, loss_with_ce, atol=0.0, rtol=0.0)


def test_fused_rms_linear_cross_entropy_cpu_fallback_matches_reference_backward():
    torch.manual_seed(223)
    x_ref = torch.randn(3, 5, 7, requires_grad=True)
    norm_ref = (torch.randn(7) * 0.1 + 1.0).requires_grad_(True)
    w_ref = (torch.randn(11, 7) * 0.1).requires_grad_(True)
    targets = torch.randint(0, 11, (3, 5), dtype=torch.long)
    x_new = x_ref.detach().clone().requires_grad_(True)
    norm_new = norm_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    normed = _reference_rms_norm(x_ref, norm_ref, eps=1e-6)
    loss_ref = _reference_linear_ce(normed, w_ref, targets)
    loss_new = fused_rms_linear_cross_entropy(
        x_new,
        norm_new,
        w_new,
        targets,
        eps=1e-6,
        reduction="mean",
        tile_size=4,
        backend="streaming_v2",
    )

    assert torch.allclose(loss_ref, loss_new, atol=0.0, rtol=0.0)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(norm_ref.grad, norm_new.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=0.0, rtol=0.0)


def test_fused_rms_linear_cross_entropy_weighted_cpu_fallback_matches_reference_backward():
    torch.manual_seed(227)
    x_ref = torch.randn(3, 5, 7, requires_grad=True)
    norm_ref = (torch.randn(7) * 0.1 + 1.0).requires_grad_(True)
    w_ref = (torch.randn(11, 7) * 0.1).requires_grad_(True)
    targets = torch.randint(0, 11, (3, 5), dtype=torch.long)
    token_weight = torch.rand(3, 5)
    x_new = x_ref.detach().clone().requires_grad_(True)
    norm_new = norm_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    normed = _reference_rms_norm(x_ref, norm_ref, eps=1e-6)
    loss_ref, per_token_ref = _reference_weighted_linear_ce(
        normed,
        w_ref,
        targets,
        token_weight,
    )
    loss_new, per_token_ce = fused_rms_linear_cross_entropy_weighted_with_ce(
        x_new,
        norm_new,
        w_new,
        targets,
        token_weight=token_weight,
        eps=1e-6,
        tile_size=4,
        backend="streaming_v2",
    )

    assert torch.allclose(loss_ref, loss_new, atol=0.0, rtol=0.0)
    assert torch.allclose(per_token_ce, per_token_ref.detach(), atol=0.0, rtol=0.0)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(norm_ref.grad, norm_new.grad, atol=0.0, rtol=0.0)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=0.0, rtol=0.0)

    loss_only = fused_rms_linear_cross_entropy_weighted(
        x_new.detach(),
        norm_new.detach(),
        w_new.detach(),
        targets,
        token_weight=token_weight,
        eps=1e-6,
        tile_size=4,
    )
    assert torch.allclose(loss_only, loss_new.detach(), atol=0.0, rtol=0.0)


def test_fused_linear_cross_entropy_rejects_bad_reduction():
    x = torch.randn(2, 3, requires_grad=True)
    w = torch.randn(5, 3, requires_grad=True)
    targets = torch.randint(0, 5, (2,), dtype=torch.long)

    with pytest.raises(ValueError, match="reduction"):
        fused_linear_cross_entropy(x, w, targets, reduction="none")


def test_public_exports_include_norm_linear_ce_entry_point():
    assert "fused_linear_cross_entropy" in lm_head_loss.__all__
    assert "fused_linear_cross_entropy_with_ce" in lm_head_loss.__all__
    assert "fused_linear_cross_entropy_weighted" in lm_head_loss.__all__
    assert "fused_linear_cross_entropy_weighted_with_ce" in lm_head_loss.__all__
    assert "fused_rms_linear_cross_entropy" in lm_head_loss.__all__
    assert "fused_rms_linear_cross_entropy_with_ce" in lm_head_loss.__all__
    assert "fused_rms_linear_cross_entropy_weighted" in lm_head_loss.__all__
    assert "fused_rms_linear_cross_entropy_weighted_with_ce" in lm_head_loss.__all__
    assert "fused_rms_norm" in lm_head_loss.__all__


def test_native_sources_expose_weighted_linear_ce_abi():
    src_root = ROOT / "src" / "chaoscontrol" / "kernels" / "_lm_head_loss" / "src"

    header = (src_root / "linear_ce.h").read_text()
    cuda = (src_root / "linear_ce.cu").read_text()
    binding = (src_root / "rms_norm_binding.cpp").read_text()

    assert "launch_linear_ce_fill_grad_logits_weighted" in header
    assert "linear_ce_fill_grad_logits_weighted_kernel" in cuda
    assert "linear_ce_weighted_backward" in binding
    assert "linear_ce_streaming_weighted_backward" in binding
    assert "linear_ce_streaming_v2_weighted_backward" in binding
    assert "linear_ce_streaming_cached_weighted_backward" in binding


def test_fused_linear_cross_entropy_with_ce_cpu_fallback_matches_reference_none():
    """Per-token CE from the ``_with_ce`` API must match
    ``F.cross_entropy(..., reduction='none')`` exactly on the CPU fallback,
    and its mean must match the scalar loss. This is what ScOpt's pressure
    computation relies on — shape ``(rows,)`` aligned with ``targets.reshape(-1)``.
    """
    torch.manual_seed(7)
    x = torch.randn(3, 5, 7)
    w = torch.randn(11, 7) * 0.1
    targets = torch.randint(0, 11, (3, 5), dtype=torch.long)

    logits_ref = x.reshape(-1, x.size(-1)) @ w.t()
    per_token_ref = F.cross_entropy(
        logits_ref.float(),
        targets.reshape(-1),
        reduction="none",
    )

    loss, per_token_ce = fused_linear_cross_entropy_with_ce(
        x,
        w,
        targets,
        reduction="mean",
        tile_size=4,
    )

    assert per_token_ce.shape == (3 * 5,)
    assert torch.allclose(per_token_ce, per_token_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(per_token_ce.mean(), loss, atol=1e-6, rtol=1e-6)


def test_fused_linear_cross_entropy_with_ce_sum_reduction_matches_per_token_sum():
    torch.manual_seed(8)
    x = torch.randn(2, 4, 6)
    w = torch.randn(9, 6) * 0.1
    targets = torch.randint(0, 9, (2, 4), dtype=torch.long)

    loss, per_token_ce = fused_linear_cross_entropy_with_ce(
        x,
        w,
        targets,
        reduction="sum",
        tile_size=3,
    )

    assert torch.allclose(per_token_ce.sum(), loss, atol=1e-5, rtol=1e-5)


def test_fused_linear_cross_entropy_with_ce_per_token_is_detached():
    """Per-token CE is a non-differentiable forward output; leaking autograd
    would quietly double-count gradients when a caller sums it. The
    ``mark_non_differentiable`` + fallback ``.detach()`` contract makes
    ``requires_grad=False`` either way."""
    torch.manual_seed(9)
    x = torch.randn(2, 3, 5, requires_grad=True)
    w = (torch.randn(7, 5) * 0.1).requires_grad_(True)
    targets = torch.randint(0, 7, (2, 3), dtype=torch.long)

    loss, per_token_ce = fused_linear_cross_entropy_with_ce(
        x,
        w,
        targets,
        reduction="mean",
        tile_size=2,
    )

    assert not per_token_ce.requires_grad
    # The scalar loss must still be differentiable; sanity-check by
    # running backward through it.
    loss.backward()
    assert x.grad is not None
    assert w.grad is not None


def test_fused_rms_linear_cross_entropy_with_ce_cpu_fallback_matches_reference():
    torch.manual_seed(10)
    x = torch.randn(2, 3, 7)
    nw = (torch.randn(7) * 0.1 + 1.0)
    lw = torch.randn(9, 7) * 0.1
    targets = torch.randint(0, 9, (2, 3), dtype=torch.long)

    # Reference: RMSNorm → linear → CE(reduction='none').
    normed_ref = F.rms_norm(x.float(), (x.size(-1),), eps=1e-6).to(x.dtype) * nw
    logits_ref = normed_ref.reshape(-1, x.size(-1)) @ lw.t()
    per_token_ref = F.cross_entropy(
        logits_ref.float(),
        targets.reshape(-1),
        reduction="none",
    )

    loss, per_token_ce = fused_rms_linear_cross_entropy_with_ce(
        x,
        nw,
        lw,
        targets,
        eps=1e-6,
        reduction="mean",
        tile_size=3,
    )

    assert per_token_ce.shape == (2 * 3,)
    assert torch.allclose(per_token_ce, per_token_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(per_token_ce.mean(), loss, atol=1e-6, rtol=1e-6)


def test_fused_linear_cross_entropy_cuda_kernel_matches_reference_if_available():
    if (
        not torch.cuda.is_available()
        or lm_head_loss._C is None
        or not hasattr(lm_head_loss._C, "linear_ce_forward")
    ):
        pytest.skip(
            "CUDA fused linear+CE kernel is not available in this environment; "
            "run fallback tests locally and this test on a CUDA pod."
        )

    torch.manual_seed(3)
    x_ref = torch.randn(
        4, 6, 9, device="cuda", dtype=torch.float32, requires_grad=True,
    )
    w_ref = (torch.randn(17, 9, device="cuda") * 0.1).requires_grad_(True)
    targets = torch.randint(0, 17, (4, 6), device="cuda", dtype=torch.long)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    loss_ref = _reference_linear_ce(x_ref, w_ref, targets)
    loss_new = fused_linear_cross_entropy(
        x_new,
        w_new,
        targets,
        reduction="mean",
        tile_size=5,
    )

    assert torch.allclose(loss_ref, loss_new, atol=2e-5, rtol=2e-5)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=2e-5, rtol=2e-5)


def test_fused_linear_cross_entropy_cuda_streaming_matches_reference_if_available():
    if (
        not torch.cuda.is_available()
        or lm_head_loss._C is None
    ):
        pytest.skip(
            "CUDA fused linear+CE kernel is not available in this environment; "
            "run fallback tests locally and this test on a CUDA pod."
        )
    assert hasattr(lm_head_loss._C, "linear_ce_streaming_forward")
    assert hasattr(lm_head_loss._C, "linear_ce_streaming_backward")

    torch.manual_seed(33)
    x_ref = torch.randn(
        4, 6, 9, device="cuda", dtype=torch.float32, requires_grad=True,
    )
    w_ref = (torch.randn(17, 9, device="cuda") * 0.1).requires_grad_(True)
    targets = torch.randint(0, 17, (4, 6), device="cuda", dtype=torch.long)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    loss_ref = _reference_linear_ce(x_ref, w_ref, targets)
    loss_new = fused_linear_cross_entropy(
        x_new,
        w_new,
        targets,
        reduction="mean",
        tile_size=5,
        backend="streaming",
    )

    assert torch.allclose(loss_ref, loss_new, atol=2e-5, rtol=2e-5)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=2e-5, rtol=2e-5)


def test_fused_linear_cross_entropy_cuda_streaming_v2_matches_reference_if_available():
    if (
        not torch.cuda.is_available()
        or lm_head_loss._C is None
    ):
        pytest.skip(
            "CUDA fused linear+CE kernel is not available in this environment; "
            "run fallback tests locally and this test on a CUDA pod."
        )
    assert hasattr(lm_head_loss._C, "linear_ce_streaming_v2_forward")
    assert hasattr(lm_head_loss._C, "linear_ce_streaming_v2_backward")

    torch.manual_seed(333)
    x_ref = torch.randn(
        4, 6, 9, device="cuda", dtype=torch.float32, requires_grad=True,
    )
    w_ref = (torch.randn(17, 9, device="cuda") * 0.1).requires_grad_(True)
    targets = torch.randint(0, 17, (4, 6), device="cuda", dtype=torch.long)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    loss_ref = _reference_linear_ce(x_ref, w_ref, targets)
    loss_new = fused_linear_cross_entropy(
        x_new,
        w_new,
        targets,
        reduction="mean",
        tile_size=5,
        backend="streaming_v2",
    )

    assert torch.allclose(loss_ref, loss_new, atol=2e-5, rtol=2e-5)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=2e-5, rtol=2e-5)


def test_fused_linear_cross_entropy_cuda_streaming_cached_matches_reference_if_available():
    if (
        not torch.cuda.is_available()
        or lm_head_loss._C is None
    ):
        pytest.skip(
            "CUDA fused linear+CE kernel is not available in this environment; "
            "run fallback tests locally and this test on a CUDA pod."
        )
    assert hasattr(lm_head_loss._C, "linear_ce_streaming_cached_forward")
    assert hasattr(lm_head_loss._C, "linear_ce_streaming_cached_backward")

    torch.manual_seed(335)
    x_ref = torch.randn(
        4, 6, 9, device="cuda", dtype=torch.float32, requires_grad=True,
    )
    w_ref = (torch.randn(20, 9, device="cuda") * 0.1).requires_grad_(True)
    targets = torch.randint(0, 20, (4, 6), device="cuda", dtype=torch.long)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    loss_ref = _reference_linear_ce(x_ref, w_ref, targets)
    loss_new = fused_linear_cross_entropy(
        x_new,
        w_new,
        targets,
        reduction="mean",
        tile_size=5,
        backend="streaming_cached",
    )

    assert torch.allclose(loss_ref, loss_new, atol=2e-5, rtol=2e-5)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    ("backend", "backward_name"),
    [
        ("auto", "linear_ce_weighted_backward"),
        ("streaming", "linear_ce_streaming_weighted_backward"),
        ("streaming_v2", "linear_ce_streaming_v2_weighted_backward"),
        ("streaming_cached", "linear_ce_streaming_cached_weighted_backward"),
    ],
)
def test_fused_linear_cross_entropy_weighted_cuda_matches_reference_if_available(
    backend: str,
    backward_name: str,
):
    if not torch.cuda.is_available() or lm_head_loss._C is None:
        pytest.skip(
            "CUDA fused weighted linear+CE kernel is not available in this "
            "environment; run fallback tests locally and this test on a CUDA pod."
        )
    assert hasattr(lm_head_loss._C, backward_name)

    torch.manual_seed(336)
    x_ref = torch.randn(
        4, 6, 9, device="cuda", dtype=torch.float32, requires_grad=True,
    )
    w_ref = (torch.randn(20, 9, device="cuda") * 0.1).requires_grad_(True)
    targets = torch.randint(0, 20, (4, 6), device="cuda", dtype=torch.long)
    token_weight = torch.rand(4, 6, device="cuda")
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    loss_ref, per_token_ref = _reference_weighted_linear_ce(
        x_ref,
        w_ref,
        targets,
        token_weight,
    )
    loss_new, per_token_ce = fused_linear_cross_entropy_weighted_with_ce(
        x_new,
        w_new,
        targets,
        token_weight=token_weight,
        tile_size=5,
        backend=backend,
    )

    assert torch.allclose(loss_ref, loss_new, atol=2e-5, rtol=2e-5)
    assert torch.allclose(per_token_ce, per_token_ref.detach(), atol=2e-5, rtol=2e-5)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=2e-5, rtol=2e-5)


def test_fused_rms_linear_cross_entropy_cuda_streaming_v2_matches_reference_if_available():
    if (
        not torch.cuda.is_available()
        or lm_head_loss._C is None
    ):
        pytest.skip(
            "CUDA fused norm+linear+CE kernel is not available in this environment; "
            "run fallback tests locally and this test on a CUDA pod."
        )
    assert hasattr(lm_head_loss._C, "rms_norm_forward")
    assert hasattr(lm_head_loss._C, "rms_norm_backward")
    assert hasattr(lm_head_loss._C, "linear_ce_streaming_v2_forward")
    assert hasattr(lm_head_loss._C, "linear_ce_streaming_v2_backward")

    torch.manual_seed(334)
    x_ref = torch.randn(
        4, 6, 9, device="cuda", dtype=torch.float32, requires_grad=True,
    )
    norm_ref = (
        torch.randn(9, device="cuda", dtype=torch.float32) * 0.1 + 1.0
    ).requires_grad_(True)
    w_ref = (torch.randn(17, 9, device="cuda") * 0.1).requires_grad_(True)
    targets = torch.randint(0, 17, (4, 6), device="cuda", dtype=torch.long)
    x_new = x_ref.detach().clone().requires_grad_(True)
    norm_new = norm_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    normed = _reference_rms_norm(x_ref, norm_ref, eps=1e-6)
    loss_ref = _reference_linear_ce(normed, w_ref, targets)
    loss_new = fused_rms_linear_cross_entropy(
        x_new,
        norm_new,
        w_new,
        targets,
        eps=1e-6,
        reduction="mean",
        tile_size=5,
        backend="streaming_v2",
    )

    assert torch.allclose(loss_ref, loss_new, atol=2e-5, rtol=2e-5)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(norm_ref.grad, norm_new.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=2e-5, rtol=2e-5)


def test_fused_rms_linear_cross_entropy_weighted_cuda_streaming_v2_matches_reference_if_available():
    if not torch.cuda.is_available() or lm_head_loss._C is None:
        pytest.skip(
            "CUDA fused weighted norm+linear+CE kernel is not available in "
            "this environment; run fallback tests locally and this test on a pod."
        )
    assert hasattr(lm_head_loss._C, "rms_norm_forward")
    assert hasattr(lm_head_loss._C, "rms_norm_backward")
    assert hasattr(lm_head_loss._C, "linear_ce_streaming_v2_forward")
    assert hasattr(lm_head_loss._C, "linear_ce_streaming_v2_weighted_backward")

    torch.manual_seed(337)
    x_ref = torch.randn(
        4, 6, 9, device="cuda", dtype=torch.float32, requires_grad=True,
    )
    norm_ref = (
        torch.randn(9, device="cuda", dtype=torch.float32) * 0.1 + 1.0
    ).requires_grad_(True)
    w_ref = (torch.randn(17, 9, device="cuda") * 0.1).requires_grad_(True)
    targets = torch.randint(0, 17, (4, 6), device="cuda", dtype=torch.long)
    token_weight = torch.rand(4, 6, device="cuda")
    x_new = x_ref.detach().clone().requires_grad_(True)
    norm_new = norm_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    normed = _reference_rms_norm(x_ref, norm_ref, eps=1e-6)
    loss_ref, per_token_ref = _reference_weighted_linear_ce(
        normed,
        w_ref,
        targets,
        token_weight,
    )
    loss_new, per_token_ce = fused_rms_linear_cross_entropy_weighted_with_ce(
        x_new,
        norm_new,
        w_new,
        targets,
        token_weight=token_weight,
        eps=1e-6,
        tile_size=5,
        backend="streaming_v2",
    )

    assert torch.allclose(loss_ref, loss_new, atol=2e-5, rtol=2e-5)
    assert torch.allclose(per_token_ce, per_token_ref.detach(), atol=2e-5, rtol=2e-5)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(x_ref.grad, x_new.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(norm_ref.grad, norm_new.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(w_ref.grad, w_new.grad, atol=2e-5, rtol=2e-5)


def test_fused_linear_cross_entropy_cuda_mixed_dtype_matches_autocast_reference_if_available():
    if (
        not torch.cuda.is_available()
        or lm_head_loss._C is None
        or not hasattr(lm_head_loss._C, "linear_ce_forward")
    ):
        pytest.skip(
            "CUDA fused linear+CE kernel is not available in this environment; "
            "run fallback tests locally and this test on a CUDA pod."
        )
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 CUDA autocast is not supported on this GPU")

    torch.manual_seed(4)
    x_ref = torch.randn(
        3, 5, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    w_ref = (torch.randn(13, 8, device="cuda") * 0.1).requires_grad_(True)
    target_tokens = torch.randint(0, 13, (3, 6), device="cuda", dtype=torch.long)
    targets = target_tokens[:, 1:]
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)

    assert not targets.is_contiguous()
    assert lm_head_loss._is_linear_ce_kernel_eligible(
        x_new,
        w_new,
        targets,
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = F.linear(x_ref, w_ref)
        loss_ref = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction="mean",
        )
        loss_new = fused_linear_cross_entropy(
            x_new,
            w_new,
            targets,
            reduction="mean",
            tile_size=5,
        )

    assert torch.allclose(loss_ref.float(), loss_new.float(), atol=3e-2, rtol=2e-2)
    loss_ref.backward()
    loss_new.backward()

    assert torch.allclose(
        x_ref.grad.float(), x_new.grad.float(), atol=4e-2, rtol=2e-2,
    )
    assert torch.allclose(
        w_ref.grad.float(), w_new.grad.float(), atol=5e-2, rtol=2e-2,
    )


def test_fused_rms_norm_cuda_kernel_matches_reference_if_available():
    if not torch.cuda.is_available() or lm_head_loss._C is None:
        pytest.skip(
            "CUDA fused RMSNorm kernel is not available in this environment; "
            "run the focused CPU fallback test locally and this test on a pod."
        )

    torch.manual_seed(1)
    x_ref = torch.randn(
        4, 9, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    w_ref = (
        torch.randn(16, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
    ).requires_grad_(True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    w_new = w_ref.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_ref)

    y_ref = _reference_rms_norm(x_ref, w_ref, eps=1e-6)
    y_new = fused_rms_norm(x_new, w_new, eps=1e-6)

    assert torch.allclose(y_ref.float(), y_new.float(), atol=4e-2, rtol=1e-2)
    y_ref.backward(grad)
    y_new.backward(grad)

    assert torch.allclose(
        x_ref.grad.float(), x_new.grad.float(), atol=4e-2, rtol=1e-2,
    )
    assert torch.allclose(
        w_ref.grad.float(), w_new.grad.float(), atol=7.5e-1, rtol=1e-2,
    )
