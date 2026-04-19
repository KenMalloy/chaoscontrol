"""CPU-safe boundary tests for the bespoke FusedFP8Linear wrapper."""
from __future__ import annotations

import torch

from chaoscontrol.kernels import fp8_linear
from chaoscontrol.kernels.fp8_linear import FusedFP8Linear


def test_forward_casts_fp32_input_to_kernel_dtype(monkeypatch) -> None:
    """SSM recurrences emit fp32; the fused fp8 kernel expects bf16 input."""
    captured: dict[str, torch.dtype | None] = {}

    def fake_apply(
        x_flat,
        weight,
        bias,
        x_scale,
        w_scale,
        gy_scale,
        x_pending,
        w_pending,
        gy_pending,
        gx_pending,
    ):
        captured["x_dtype"] = x_flat.dtype
        captured["weight_dtype"] = weight.dtype
        captured["bias_dtype"] = bias.dtype if bias is not None else None
        return torch.zeros(
            x_flat.shape[0],
            weight.shape[0],
            dtype=weight.dtype,
            device=x_flat.device,
        )

    monkeypatch.setattr(
        fp8_linear._FusedFP8LinearFn, "apply", staticmethod(fake_apply)
    )

    layer = FusedFP8Linear(4, 3, dtype=torch.bfloat16)
    x = torch.randn(2, 5, 4, dtype=torch.float32)

    out = layer(x)

    assert captured == {
        "x_dtype": torch.bfloat16,
        "weight_dtype": torch.bfloat16,
        "bias_dtype": torch.bfloat16,
    }
    assert out.shape == (2, 5, 3)
    assert out.dtype == torch.bfloat16
