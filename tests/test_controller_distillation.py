"""CRCT controller distillation unit tests."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from chaoscontrol.controller_distillation import (
    ControllerMLP,
    controller_loss,
    gate_from_logits,
)


def test_controller_mlp_returns_one_logit_per_token():
    model = ControllerMLP(model_dim=8, hidden_dim=4)
    x = torch.randn(2, 5, 8)

    logits = model(x)

    assert logits.shape == (2, 5)


def test_gate_from_logits_is_non_negative_continuous_gate():
    logits = torch.tensor([-100.0, 0.0, torch.logit(torch.tensor(0.75)), 100.0])

    gate = gate_from_logits(logits)

    assert gate[0].item() == pytest.approx(0.0)
    assert gate[1].item() == pytest.approx(0.0)
    assert gate[2].item() == pytest.approx(0.5, abs=1e-6)
    assert gate[3].item() == pytest.approx(1.0)


def test_controller_loss_matches_weighted_bce_reference():
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    target = torch.tensor([[0.5, 1.0, 0.0]])
    confidence = torch.tensor([[1.0, 0.25, 0.0]])
    mask = torch.tensor([[True, True, True]])

    out = controller_loss(
        logits,
        target,
        confidence=confidence,
        mask=mask,
    )
    ref = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    ref = (ref * confidence).sum() / confidence.sum().clamp_min(1.0)

    assert out.item() == pytest.approx(ref.item(), abs=1e-7)


def test_controller_loss_rejects_shape_drift():
    logits = torch.zeros(2, 3)
    target = torch.zeros(2, 4)

    with pytest.raises(ValueError, match="shape mismatch"):
        controller_loss(logits, target)
