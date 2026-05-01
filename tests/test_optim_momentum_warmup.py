"""Tests for Muon momentum warmup schedule."""
import pytest
import torch
from chaoscontrol.optim.muon import Muon
from chaoscontrol.optim.momentum_warmup import apply_momentum_warmup


def _make_muon(momentum: float = 0.95):
    p = torch.nn.Parameter(torch.randn(4, 4))
    return Muon([p], lr=0.01, momentum=momentum), p


def test_warmup_at_step_zero_is_start_value():
    opt, _ = _make_muon()
    apply_momentum_warmup(opt, step=0, target=0.99, start=0.92, steps=1500)
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.92)


def test_warmup_at_full_steps_is_target_value():
    opt, _ = _make_muon()
    apply_momentum_warmup(opt, step=1500, target=0.99, start=0.92, steps=1500)
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.99)


def test_warmup_clamps_after_full_steps():
    opt, _ = _make_muon()
    apply_momentum_warmup(opt, step=5000, target=0.99, start=0.92, steps=1500)
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.99)


def test_warmup_linear_at_midpoint():
    opt, _ = _make_muon()
    apply_momentum_warmup(opt, step=750, target=0.99, start=0.92, steps=1500)
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.955)


def test_warmup_zero_steps_is_target():
    """warmup_steps=0 means no warmup — momentum should be target immediately."""
    opt, _ = _make_muon()
    apply_momentum_warmup(opt, step=0, target=0.99, start=0.92, steps=0)
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.99)


def test_warmup_applies_to_all_param_groups():
    p1 = torch.nn.Parameter(torch.randn(4, 4))
    p2 = torch.nn.Parameter(torch.randn(2))
    opt = Muon([{"params": [p1]}, {"params": [p2]}], lr=0.01, momentum=0.95)
    apply_momentum_warmup(opt, step=0, target=0.99, start=0.92, steps=1500)
    for group in opt.param_groups:
        assert group["momentum"] == pytest.approx(0.92)
