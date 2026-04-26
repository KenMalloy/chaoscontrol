"""C7 -- CPU SSM online-step backward parity."""
from __future__ import annotations

import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


pytestmark = pytest.mark.skipif(
    _ext._C is None,
    reason="cpu_ssm_controller C++ extension not built",
)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "features": torch.tensor([0.25, -0.50, 1.25], dtype=torch.float32),
        "global_state": torch.tensor([0.75, -1.50], dtype=torch.float32),
        "slot_state": torch.tensor([-0.25, 1.00], dtype=torch.float32),
        "w_global_in": torch.tensor(
            [[0.10, -0.20, 0.30], [-0.40, 0.50, -0.60]],
            dtype=torch.float32,
        ),
        "w_slot_in": torch.tensor(
            [[0.70, -0.80, 0.90], [-1.00, 1.10, -1.20]],
            dtype=torch.float32,
        ),
        "decay_global": torch.tensor([0.95, 0.60], dtype=torch.float32),
        "decay_slot": torch.tensor([0.25, 0.80], dtype=torch.float32),
        "w_global_out": torch.tensor([1.30, -0.70], dtype=torch.float32),
        "w_slot_out": torch.tensor([-1.10, 0.40], dtype=torch.float32),
        "bias": torch.tensor(-0.15, dtype=torch.float32),
    }


def _reference_logit(args: dict[str, torch.Tensor]) -> torch.Tensor:
    out_g = (
        args["decay_global"] * args["global_state"]
        + args["w_global_in"] @ args["features"]
    )
    out_s = (
        args["decay_slot"] * args["slot_state"]
        + args["w_slot_in"] @ args["features"]
    )
    return (
        out_g @ args["w_global_out"]
        + out_s @ args["w_slot_out"]
        + args["bias"]
    )


def _cpp_backward(
    args: dict[str, torch.Tensor],
    *,
    grad_logit: float,
) -> dict[str, torch.Tensor]:
    return _ext.backward_step(
        args["features"],
        args["global_state"],
        args["slot_state"],
        args["w_global_in"],
        args["w_slot_in"],
        args["decay_global"],
        args["decay_slot"],
        args["w_global_out"],
        args["w_slot_out"],
        float(args["bias"].item()),
        grad_logit,
    )


def test_backward_step_matches_torch_autograd():
    args = _inputs()
    ref_args = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in args.items()
    }
    grad_logit = torch.tensor(0.75, dtype=torch.float32)
    _reference_logit(ref_args).backward(grad_logit)

    got = _cpp_backward(args, grad_logit=float(grad_logit.item()))

    assert set(got) == set(args)
    for name, value in got.items():
        assert value.dtype == torch.float32
        assert value.device.type == "cpu"
        torch.testing.assert_close(
            value,
            ref_args[name].grad,
            atol=1e-5,
            rtol=1e-5,
            msg=f"{name} gradient mismatch",
        )


def test_backward_step_zero_grad_logit_returns_zero_grads():
    got = _cpp_backward(_inputs(), grad_logit=0.0)

    for name, grad in got.items():
        torch.testing.assert_close(
            grad,
            torch.zeros_like(grad),
            atol=0.0,
            rtol=0.0,
            msg=f"{name} should be exactly zero",
        )


def test_backward_step_rejects_bad_weight_shape():
    args = _inputs()
    args["w_global_in"] = torch.zeros(3, 3, dtype=torch.float32)

    with pytest.raises(RuntimeError, match=r"w_global_in must have shape \[2, 3\]"):
        _cpp_backward(args, grad_logit=1.0)
