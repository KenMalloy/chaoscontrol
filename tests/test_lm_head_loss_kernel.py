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
    fused_rms_linear_cross_entropy,
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


def test_fused_linear_cross_entropy_rejects_bad_reduction():
    x = torch.randn(2, 3, requires_grad=True)
    w = torch.randn(5, 3, requires_grad=True)
    targets = torch.randint(0, 5, (2,), dtype=torch.long)

    with pytest.raises(ValueError, match="reduction"):
        fused_linear_cross_entropy(x, w, targets, reduction="none")


def test_public_exports_include_norm_linear_ce_entry_point():
    assert "fused_linear_cross_entropy" in lm_head_loss.__all__
    assert "fused_rms_linear_cross_entropy" in lm_head_loss.__all__
    assert "fused_rms_norm" in lm_head_loss.__all__


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
