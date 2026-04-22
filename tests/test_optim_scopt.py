"""Tests for the scarcity-aware optimizer design.

CPU-only and float32 throughout: these tests pin the optimizer semantics,
not H100 throughput. The fast-runner integration can build on this API
without making the core optimizer depend on a particular training loop.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from chaoscontrol.optim import ScarcityAwareOptimizer as PackageScOpt
from chaoscontrol.optim.muon import Muon
from chaoscontrol.optim.scopt import (
    ScarcityAwareOptimizer,
    scarcity_pressure_from_ce,
)


def test_package_exports_scarcity_optimizer() -> None:
    assert PackageScOpt is ScarcityAwareOptimizer


def test_pressure_is_detached_positive_excess() -> None:
    ce = torch.tensor(
        [[0.25, 1.50, 0.75]], dtype=torch.float32, requires_grad=True
    )
    targets = torch.tensor([[0, 1, 2]])
    token_frequencies = torch.tensor([1.0, 8.0, 3.0])
    baseline = torch.tensor([[0.50, 1.00, 0.25]])

    pressure = scarcity_pressure_from_ce(
        ce,
        targets,
        token_frequencies=token_frequencies,
        baseline=baseline,
    )

    expected_rarity = 1.0 / torch.log1p(token_frequencies[targets])
    expected = expected_rarity * torch.tensor([[0.0, 0.50, 0.50]])
    assert torch.allclose(pressure, expected)
    assert not pressure.requires_grad


def test_reduces_to_muon_when_scarcity_inactive() -> None:
    torch.manual_seed(0)
    layer_a = nn.Linear(6, 4, bias=True)
    layer_b = copy.deepcopy(layer_a)
    xs = torch.randn(8, 6)
    target = torch.randn(8, 4)

    muon = Muon(
        layer_a.parameters(),
        lr=0.01,
        momentum=0.9,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
        adamw_lr=0.01,
        adamw_weight_decay=0.0,
        compute_dtype=torch.float32,
    )
    scopt = ScarcityAwareOptimizer(
        layer_b.parameters(),
        lr=0.01,
        momentum=0.9,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
        adamw_lr=0.01,
        adamw_weight_decay=0.0,
        warmup_steps=0,
        compute_dtype=torch.float32,
    )

    for _ in range(4):
        for layer, opt in ((layer_a, muon), (layer_b, scopt)):
            opt.zero_grad(set_to_none=True)
            loss = ((layer(xs) - target) ** 2).mean()
            loss.backward()
            opt.step()

    assert torch.allclose(layer_a.weight, layer_b.weight, atol=1e-6)
    assert torch.allclose(layer_a.bias, layer_b.bias, atol=1e-6)


def test_rare_orthogonal_component_is_projected_against_common_direction(monkeypatch) -> None:
    monkeypatch.setattr(
        "chaoscontrol.optim.scopt.newton_schulz_orthogonalize",
        lambda grad, **_: grad,
    )
    w = nn.Parameter(torch.zeros(2, 2))
    opt = ScarcityAwareOptimizer(
        [w],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        rare_orthogonal_weight=1.0,
        compute_dtype=torch.float32,
    )
    opt.bind_param_names([("w", w)])
    opt.set_rare_grad_ema({"w": torch.tensor([[1.0, 1.0], [0.0, 0.0]])})

    w.grad = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    opt.step()

    # rare=[1,1,0,0], common=[1,0,0,0] -> orthogonal component=[0,1,0,0].
    expected_applied = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    assert torch.allclose(-w.detach(), expected_applied, atol=1e-6)


def test_row_scarcity_prescales_token_indexed_matrix_before_ns(monkeypatch) -> None:
    monkeypatch.setattr(
        "chaoscontrol.optim.scopt.newton_schulz_orthogonalize",
        lambda grad, **_: grad,
    )
    weight = nn.Parameter(torch.zeros(3, 2))
    opt = ScarcityAwareOptimizer(
        [weight],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        row_param_names={"embed.weight"},
        row_scarcity_power=1.0,
        tau_row_floor=1.0,
        tau_std_scale=0.0,
        compute_dtype=torch.float32,
    )
    opt.bind_param_names([("embed.weight", weight)])
    opt.set_row_pressure_ema(torch.tensor([0.0, 1.0, 4.0]))

    weight.grad = torch.ones_like(weight)
    opt.step()

    expected_factor = torch.tanh(torch.tensor([0.0, 1.0, 4.0])) + 1.0
    rectangular_scale = (3.0 / 2.0) ** 0.5
    applied = -weight.detach()
    assert torch.allclose(applied[:, 0], expected_factor * rectangular_scale, atol=1e-6)
    assert torch.allclose(applied[:, 1], expected_factor * rectangular_scale, atol=1e-6)
    assert applied[2, 0] > applied[1, 0] > applied[0, 0]


def test_warmup_suppresses_rare_ema_writes_and_reads() -> None:
    """Design spec line 170: rare_grad_ema is undefined during warm-start.

    During warmup both the read gate (_scarcity_enabled) and the write
    gate (_writes_enabled) should keep the EMA at zero so the mechanism
    is fully inert. At step warmup_steps + 1 the first write populates
    the EMA and the read path sees it.
    """
    w = nn.Parameter(torch.zeros(2, 2))
    opt = ScarcityAwareOptimizer(
        [w],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=3,
        compute_dtype=torch.float32,
    )
    opt.bind_param_names([("w", w)])

    for _ in range(3):
        opt.update_rare_grad_ema({"w": torch.ones(2, 2)})
        w.grad = torch.zeros_like(w)
        opt.step()

    # After three warmup step()s, _writes_enabled became True only at the
    # last call (step_count=2 pre-step, 2 < 3 = True, still suppressed).
    # The EMA must still be zero / absent.
    assert "rare_grad_ema" not in opt.state[w] or opt.state[w]["rare_grad_ema"].abs().sum() == 0

    # Now at step_count == warmup_steps, the next write proceeds.
    opt.update_rare_grad_ema({"w": torch.ones(2, 2)})
    w.grad = torch.zeros_like(w)
    opt.step()
    assert opt.state[w]["rare_grad_ema"].abs().sum() > 0


def test_reduces_to_muon_through_warmup() -> None:
    """Parity with Muon through the full warmup phase.

    Sets a non-zero rare_grad_ema directly — while scarcity is gated off
    (step_count <= warmup_steps) the update must ignore it and match
    Muon exactly. This is the ablation-fidelity test the reviewer asked
    for: confirm warm-start is a true no-op, not "scarcity enabled but
    empty state".
    """
    import copy
    from chaoscontrol.optim.muon import Muon

    torch.manual_seed(7)
    layer_a = nn.Linear(6, 4, bias=True)
    layer_b = copy.deepcopy(layer_a)
    xs = torch.randn(8, 6)
    target = torch.randn(8, 4)

    muon = Muon(
        layer_a.parameters(),
        lr=0.01,
        momentum=0.9,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
        adamw_lr=0.01,
        adamw_weight_decay=0.0,
        compute_dtype=torch.float32,
    )
    scopt = ScarcityAwareOptimizer(
        layer_b.parameters(),
        lr=0.01,
        momentum=0.9,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
        adamw_lr=0.01,
        adamw_weight_decay=0.0,
        warmup_steps=4,
        compute_dtype=torch.float32,
    )
    scopt.bind_param_names(list(layer_b.named_parameters()))
    # Inject nonzero rare EMA directly; warmup gate should make it inert.
    scopt.state[layer_b.weight]["rare_grad_ema"] = torch.randn_like(layer_b.weight)

    for _ in range(3):
        for layer, opt in ((layer_a, muon), (layer_b, scopt)):
            opt.zero_grad(set_to_none=True)
            loss = ((layer(xs) - target) ** 2).mean()
            loss.backward()
            opt.step()

    assert torch.allclose(layer_a.weight, layer_b.weight, atol=1e-6)
    assert torch.allclose(layer_a.bias, layer_b.bias, atol=1e-6)


def test_grad_clip_skip_blocks_rare_ema_update() -> None:
    """Design spec line 186: skipping a clipped step keeps rare EMA clean."""
    w = nn.Parameter(torch.zeros(2, 2))
    opt = ScarcityAwareOptimizer(
        [w],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        compute_dtype=torch.float32,
    )
    opt.bind_param_names([("w", w)])

    opt.update_rare_grad_ema({"w": torch.ones(2, 2)}, skip=True)
    assert "rare_grad_ema" not in opt.state[w] or opt.state[w]["rare_grad_ema"].abs().sum() == 0

    opt.update_rare_grad_ema({"w": torch.ones(2, 2)}, skip=False)
    assert opt.state[w]["rare_grad_ema"].abs().sum() > 0


def test_recurrence_scarcity_raises_on_shape_mismatch() -> None:
    """L4: silent no-op on mismatched shapes was masking bugs; now raises."""
    log_a = nn.Parameter(torch.zeros(4))
    opt = ScarcityAwareOptimizer(
        [log_a],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        recurrence_scarcity_map={"log_a": "mismatched"},
        compute_dtype=torch.float32,
    )
    opt.bind_param_names([("log_a", log_a)])
    opt.set_channel_pressure({"mismatched": torch.ones(7)})
    log_a.grad = torch.zeros_like(log_a)

    with pytest.raises(ValueError, match="broadcast is not supported"):
        opt.step()


def test_rare_adjusted_direction_records_alignment_telemetry() -> None:
    w = nn.Parameter(torch.zeros(2, 2))
    opt = ScarcityAwareOptimizer(
        [w],
        lr=0.1,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        rare_orthogonal_weight=1.0,
        compute_dtype=torch.float32,
    )
    opt.bind_param_names([("w", w)])
    opt.set_rare_grad_ema({"w": torch.tensor([[1.0, 1.0], [0.0, 0.0]])})

    w.grad = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    opt.step()
    trace = opt.scarcity_trace()
    assert "cos_rare_common" in trace
    assert "r_orth_over_common" in trace
    assert trace["cos_rare_common"]["count"] >= 1
