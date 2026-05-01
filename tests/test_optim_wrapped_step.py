"""Tests for optimizer.step wrapping with momentum warmup + EMA update."""
import pytest
import torch
from chaoscontrol.optim.muon import Muon
from chaoscontrol.optim.weight_ema import WeightEMA
from chaoscontrol.optim.step_wrapper import wrap_optimizer_step


class _Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


def _step_with_dummy_loss(opt, model):
    opt.zero_grad()
    x = torch.randn(2, 4)
    y = model.linear(x).sum()
    y.backward()
    opt.step()


def test_wrapped_step_applies_momentum_schedule():
    model = _Trunk()
    opt = Muon(list(model.parameters()), lr=0.01, momentum=0.95)
    wrap_optimizer_step(
        opt,
        model=model,
        target_momentum=0.99,
        warmup_start=0.92,
        warmup_steps=10,
        weight_ema_decay=0.0,  # disabled
        is_rank_zero=True,
        ema_exclude_prefixes=(),
    )
    _step_with_dummy_loss(opt, model)
    # First call evaluated schedule with step=0 -> momentum = start.
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.92)
    _step_with_dummy_loss(opt, model)
    # Second call evaluated schedule with step=1 -> frac = 1/10 = 0.1.
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.92 + 0.07 * 0.1)


def test_wrapped_step_updates_ema_when_enabled():
    model = _Trunk()
    opt = Muon(list(model.parameters()), lr=0.01, momentum=0.95)
    wrap_optimizer_step(
        opt,
        model=model,
        target_momentum=0.95,
        warmup_start=0.95,
        warmup_steps=0,
        weight_ema_decay=0.9,
        is_rank_zero=True,
        ema_exclude_prefixes=(),
    )
    initial_shadow = opt._weight_ema.shadow["linear.weight"].clone()
    _step_with_dummy_loss(opt, model)
    # EMA shadow should have moved (model weights changed via gradient step).
    assert not torch.allclose(opt._weight_ema.shadow["linear.weight"], initial_shadow)


def test_wrapped_step_skips_ema_on_non_rank_zero():
    model = _Trunk()
    opt = Muon(list(model.parameters()), lr=0.01, momentum=0.95)
    wrap_optimizer_step(
        opt,
        model=model,
        target_momentum=0.95,
        warmup_start=0.95,
        warmup_steps=0,
        weight_ema_decay=0.9,
        is_rank_zero=False,
        ema_exclude_prefixes=(),
    )
    assert opt._weight_ema is None


def test_wrapped_step_returns_original_step_value():
    """Muon.step returns None; wrapper must preserve that."""
    model = _Trunk()
    opt = Muon(list(model.parameters()), lr=0.01, momentum=0.95)
    wrap_optimizer_step(
        opt, model=model, target_momentum=0.95, warmup_start=0.95,
        warmup_steps=0, weight_ema_decay=0.0, is_rank_zero=True,
        ema_exclude_prefixes=(),
    )
    opt.zero_grad()
    x = torch.randn(2, 4)
    y = model.linear(x).sum()
    y.backward()
    assert opt.step() is None


def test_wrapped_step_does_not_break_newton_schulz_path():
    """Muon's matrix path runs Newton-Schulz iterations; verify the wrap
    doesn't disturb iterative compute on a non-trivial matrix."""
    class _Wide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(64, 64, bias=False)

    model = _Wide()
    # The 64x64 weight is 2D, so Muon's default classifier routes it through
    # the Newton-Schulz path without needing matrix_param_names + bind.
    opt = Muon(list(model.parameters()), lr=0.01, momentum=0.95)
    wrap_optimizer_step(
        opt, model=model, target_momentum=0.99, warmup_start=0.92,
        warmup_steps=10, weight_ema_decay=0.997, is_rank_zero=True,
        ema_exclude_prefixes=(),
    )
    pre = model.linear.weight.detach().clone()
    for _ in range(20):
        opt.zero_grad()
        x = torch.randn(8, 64)
        y = model.linear(x).pow(2).sum()
        y.backward()
        opt.step()
    # Weights moved (training happened), shadow exists, momentum reached target.
    assert not torch.allclose(model.linear.weight, pre)
    assert torch.isfinite(model.linear.weight).all()
    assert opt._weight_ema is not None
    assert opt.param_groups[0]["momentum"] == pytest.approx(0.99)
