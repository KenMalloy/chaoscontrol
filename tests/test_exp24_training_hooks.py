"""Tests for Exp 23 training-time hook helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "23_fast_path" / "training_hooks.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("exp23_training_hooks", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FastSlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)


class _LogAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectrum = nn.Module()
        self.spectrum.log_a = nn.Parameter(torch.tensor([-4.0, 0.0, 5.0]))


class _EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)


def test_fast_slow_consolidator_from_config_and_sync_and_copy():
    mod = _load_module()
    model = _FastSlowModel()
    consolidator = mod.FastSlowConsolidator.from_config(
        model,
        {
            "fast_slow_enabled": True,
            "fast_slow_interval": 1,
            "fast_slow_alpha": 0.5,
        },
    )

    assert consolidator.enabled
    assert consolidator.interval == 1
    assert consolidator.alpha == 0.5
    assert torch.equal(consolidator.slow_state["linear.weight"], model.linear.weight.detach())

    with torch.no_grad():
        model.linear.weight.add_(1.0)
    consolidator.after_optimizer_step(model, step=1)

    diagnostics = consolidator.diagnostics(model)
    assert diagnostics["enabled"]
    assert diagnostics["sync_count"] == 1
    assert diagnostics["fast_slow_l2"] > 0.0

    consolidator.copy_slow_to_model(model)
    assert torch.equal(model.linear.weight, consolidator.slow_state["linear.weight"])


def test_fastslow_eval_copy_is_not_ready_until_a_real_sync():
    mod = _load_module()
    model = _FastSlowModel()
    consolidator = mod.FastSlowConsolidator.from_config(
        model,
        {
            "fast_slow_enabled": True,
            "fast_slow_interval": 0,
            "fast_slow_alpha": 0.5,
        },
    )

    assert consolidator.sync_count == 0
    assert consolidator.should_copy_slow_to_model_for_eval() is False
    assert consolidator.diagnostics(model)["eval_copy_ready"] is False

    with torch.no_grad():
        model.linear.weight.add_(1.0)
    consolidator.apply_decision(
        model,
        mod.FastSlowDecision(
            mode="learned",
            accepted=True,
            alpha=0.5,
            gate=1.0,
            effective_alpha=0.5,
            step=1,
            reason="test",
        ),
    )

    assert consolidator.sync_count == 1
    assert consolidator.should_copy_slow_to_model_for_eval() is True
    assert consolidator.diagnostics(model)["eval_copy_ready"] is True


def test_fastslow_after_optimizer_step_matches_reference_lerp():
    mod = _load_module()
    FastSlowConsolidator = mod.FastSlowConsolidator

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    consolidator = FastSlowConsolidator.from_config(
        model,
        {
            "fast_slow_enabled": True,
            "fast_slow_interval": 1,
            "fast_slow_alpha": 0.1,
        },
    )
    pre_slow = {
        name: tensor.clone() for name, tensor in consolidator.slow_state.items()
    }

    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param))

    consolidator.after_optimizer_step(model, step=1)

    model_params = dict(model.named_parameters())
    for name, pre_slow_tensor in pre_slow.items():
        expected = torch.lerp(pre_slow_tensor, model_params[name].detach(), 0.1)
        actual = consolidator.slow_state[name]
        assert torch.equal(actual, expected), (
            f"{name} lerp mismatch: max diff "
            f"{(actual - expected).abs().max().item()}"
        )


def test_fastslow_controller_head_uses_gate_times_alpha_without_interval():
    mod = _load_module()
    from chaoscontrol.episodic.learned_action_space import (
        ConstrainedActionSpace,
        SharedEventSsm,
    )

    model = _FastSlowModel()
    consolidator = mod.FastSlowConsolidator.from_config(
        model,
        {
            "fast_slow_enabled": True,
            "fast_slow_interval": 0,
            "fast_slow_alpha": 0.2,
        },
    )
    ssm = SharedEventSsm(hidden_dim=4, input_scale=0.0, head_scale=0.0, seed=1)
    action_space = ConstrainedActionSpace(
        head_readiness={"consolidation": 1.0, "ema_alpha": 1.0},
        event_ssm=ssm,
    )
    pre_slow = {
        name: tensor.clone() for name, tensor in consolidator.slow_state.items()
    }
    with torch.no_grad():
        model.linear.weight.add_(1.0)

    decision = consolidator.after_optimizer_step(
        model,
        step=1,
        action_space=action_space,
        reward_context={"steps_since_slow_sync": 1.0},
    )

    assert decision.mode == "learned"
    assert decision.gate == pytest.approx(0.5)
    assert decision.alpha == pytest.approx(0.25)
    assert decision.effective_alpha == pytest.approx(0.125)
    expected = torch.lerp(
        pre_slow["linear.weight"],
        model.linear.weight.detach(),
        decision.effective_alpha,
    )
    assert torch.equal(consolidator.slow_state["linear.weight"], expected)
    diag = consolidator.diagnostics(model)
    assert diag["learned_decision_count"] == 1
    assert diag["learned_sync_count"] == 1
    assert diag["last_decision"]["mode"] == "learned"


def test_fastslow_controller_head_records_loss_feedback():
    mod = _load_module()
    from chaoscontrol.episodic.learned_action_space import (
        ConstrainedActionSpace,
        SharedEventSsm,
    )

    model = _FastSlowModel()
    ssm = SharedEventSsm(hidden_dim=4, seed=1)
    action_space = ConstrainedActionSpace(
        head_readiness={"consolidation": 1.0},
        event_ssm=ssm,
        online_learning_rate=0.1,
    )
    consolidator = mod.FastSlowConsolidator.from_config(
        model,
        {
            "fast_slow_enabled": True,
            "fast_slow_interval": 0,
            "fast_slow_alpha": 0.2,
        },
    )
    with torch.no_grad():
        model.linear.weight.add_(1.0)

    consolidator.after_optimizer_step(
        model,
        step=1,
        action_space=action_space,
        loss_value=2.0,
    )
    with torch.no_grad():
        model.linear.weight.add_(0.5)
    consolidator.after_optimizer_step(
        model,
        step=2,
        action_space=action_space,
        loss_value=1.5,
    )

    assert consolidator.reward_update_count >= 1
    assert consolidator.last_reward == pytest.approx(0.5)


def test_spectral_regularization_loss_and_summary():
    mod = _load_module()
    model = _LogAModel()
    penalty = mod.spectral_regularization_loss(
        model=model,
        lambda_dead=2.0,
        lambda_sticky=3.0,
        min_a=0.1,
        max_a=0.9,
    )

    assert penalty is not None
    assert penalty.item() > 0

    summary = mod.spectral_summary(model)
    assert summary["log_a_param_count"] == 1
    assert summary["a_min"] < 0.1
    assert summary["a_max"] > 0.9


def test_zero_embedding_grad_until():
    mod = _load_module()
    model = _EmbeddingModel()
    model.embed.weight.grad = torch.ones_like(model.embed.weight)

    mod.zero_embedding_grad_until(model, step=2, freeze_steps=3)
    assert torch.count_nonzero(model.embed.weight.grad).item() == 0

    model.embed.weight.grad = torch.ones_like(model.embed.weight)
    mod.zero_embedding_grad_until(model, step=3, freeze_steps=3)
    assert torch.count_nonzero(model.embed.weight.grad).item() == model.embed.weight.grad.numel()


def test_predictive_auxiliary_loss_uses_detached_future_hidden():
    hooks = _load_module()
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    proj = nn.Linear(4, 4, bias=False)

    loss = hooks.predictive_auxiliary_loss(
        hidden,
        projection=proj,
        horizon=2,
    )

    assert loss is not None
    loss.backward()
    assert hidden.grad is not None
    assert proj.weight.grad is not None
