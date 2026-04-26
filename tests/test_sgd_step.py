import pytest
import torch

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_sgd_step_mutates_single_weight_in_place():
    w = torch.tensor([1.0], dtype=torch.float32)
    g = torch.tensor([1.0], dtype=torch.float32)

    _ext.SgdStep(0.1).apply(w, g)

    assert torch.allclose(w, torch.tensor([0.9], dtype=torch.float32))


def test_sgd_step_mutates_vector_weights_in_place():
    w = torch.ones(100, dtype=torch.float32)
    g = torch.ones(100, dtype=torch.float32)

    _ext.SgdStep(0.1).apply(w, g)

    assert torch.allclose(w, torch.full((100,), 0.9, dtype=torch.float32))


def test_sgd_step_handles_mixed_gradient_signs():
    w = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    g = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)

    _ext.SgdStep(0.1).apply(w, g)

    assert torch.allclose(
        w,
        torch.tensor([0.9, 1.2, 0.95], dtype=torch.float32),
    )


def test_sgd_step_rejects_mismatched_shapes():
    w = torch.ones(2, dtype=torch.float32)
    g = torch.ones(3, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="same shape"):
        _ext.SgdStep(0.1).apply(w, g)
