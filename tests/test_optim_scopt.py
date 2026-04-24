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
        # Disable the macro-Gerber cap: this test asserts the algebraic
        # projection identity at rare_orthogonal_weight=1.0, which only
        # holds without the post-fix cap. The cap's own behavior is
        # covered by test_rare_macro_c_caps_orthogonal_when_larger_than_common.
        rare_macro_c=0.0,
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


def test_two_sided_matrix_scarcity_applies_sqrt_factors_before_ns(monkeypatch) -> None:
    """Design spec lines 122-135: L·G·R with L=sqrt(out), R=sqrt(in)."""
    monkeypatch.setattr(
        "chaoscontrol.optim.scopt.newton_schulz_orthogonalize",
        lambda grad, **_: grad,
    )
    w = nn.Parameter(torch.zeros(3, 2))
    opt = ScarcityAwareOptimizer(
        [w],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        row_param_names=set(),  # Turn off row scarcity to isolate two-sided path.
        matrix_scarcity_map={"w": ("w.__out__", "w.__in__")},
        tau_out_floor=1.0,
        tau_in_floor=1.0,
        tau_std_scale=0.0,
        compute_dtype=torch.float32,
    )
    opt.bind_param_names([("w", w)])
    # Pressures chosen so tanh(p/τ)+1 produces clean factors.
    # out_pressure has 3 rows; in_pressure has 2 cols.
    opt.set_channel_pressure({
        "w.__out__": torch.tensor([0.0, 1.0, 4.0]),
        "w.__in__": torch.tensor([0.0, 2.0]),
    })

    w.grad = torch.ones_like(w)
    opt.step()

    out_factor = torch.tanh(torch.tensor([0.0, 1.0, 4.0])) + 1.0
    in_factor = torch.tanh(torch.tensor([0.0, 2.0])) + 1.0
    expected_scaled = out_factor.sqrt()[:, None] * torch.ones(3, 2) * in_factor.sqrt()[None, :]
    rectangular_scale = (3.0 / 2.0) ** 0.5  # rows/cols scale from Muon update.
    applied = -w.detach()
    assert torch.allclose(applied, expected_scaled * rectangular_scale, atol=1e-6)


def test_energy_enrichment_ratio_greater_than_one_when_scarce_rows_dominate(monkeypatch) -> None:
    """Design spec lines 278-284: enrichment ratio should exceed 1 when
    scarce rows receive disproportionate NS-output energy.

    Uses 4 distinct per-row pressures so torch.median picks an unambiguous
    split — bimodal pressure (e.g. [0,0,3,3]) would put the median at the
    low cluster and leave the common branch empty, silently skipping the
    telemetry record.
    """
    monkeypatch.setattr(
        "chaoscontrol.optim.scopt.newton_schulz_orthogonalize",
        lambda grad, **_: grad,
    )
    w = nn.Parameter(torch.zeros(4, 2))
    opt = ScarcityAwareOptimizer(
        [w],
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
    opt.bind_param_names([("embed.weight", w)])
    opt.set_row_pressure_ema(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    w.grad = torch.ones_like(w)
    opt.step()
    trace = opt.scarcity_trace()

    assert "ns_row_energy_enrichment" in trace
    ratio = trace["ns_row_energy_enrichment"]["median"]
    # Analytical ratio is ~3.6× (scarce rows get factor ≈ [1.76, 1.96, 1.99]
    # while common row stays at 1.0); use 1.5× as a loose floor.
    assert ratio > 1.5, f"scarce rows should carry >1.5× common-row energy, got {ratio}"


def test_ns_row_factor_corr_is_one_when_ns_preserves_pre_scaling(monkeypatch) -> None:
    """Tail-bypass diagnostic: when NS is the identity (no whitening) and
    pre-NS direction rows are equal, log(post-NS row norm) = log(row_factor)
    + const, so the Pearson correlation is exactly 1.

    Pins the measurement semantics on the lower end of possible NS
    behavior (identity). The Tier 0 smoke on a real pod will observe
    whatever real NS does; this test only guards that the telemetry is
    wired up and interprets preserved scaling as a correlation of 1.
    """
    monkeypatch.setattr(
        "chaoscontrol.optim.scopt.newton_schulz_orthogonalize",
        lambda grad, **_: grad,
    )
    w = nn.Parameter(torch.zeros(4, 2))
    opt = ScarcityAwareOptimizer(
        [w],
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
    opt.bind_param_names([("embed.weight", w)])
    opt.set_row_pressure_ema(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    w.grad = torch.ones_like(w)
    opt.step()
    trace = opt.scarcity_trace()

    assert "ns_row_factor_corr" in trace
    corr_median = trace["ns_row_factor_corr"]["median"]
    assert corr_median > 0.99, (
        f"identity NS preserves pre-NS row scaling, so log-scale correlation "
        f"should be ~1.0, got {corr_median}"
    )


def test_recurrence_scarcity_with_timescale_modulates_per_channel() -> None:
    """Design spec lines 148-161: long-half-life channels with high pressure
    get amplified updates; zero timescale disables the modulation channel-wise.

    Exercises ``_apply_recurrence_scarcity`` directly. AdamW's denom
    normalization inside ``step()`` would wash out the per-channel factor,
    so step()-level observation is the wrong scope for this assertion.
    """
    log_a = nn.Parameter(torch.zeros(4))
    opt = ScarcityAwareOptimizer(
        [log_a],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        recurrence_scarcity_map={"log_a": "rec"},
        recurrence_weight=1.0,
        tau_out_floor=1.0,
        tau_std_scale=0.0,
        compute_dtype=torch.float32,
    )
    opt.bind_param_names([("log_a", log_a)])

    pressure = torch.tensor([0.0, 1.0, 2.0, 3.0])
    timescale = torch.tensor([0.0, 0.0, 1.0, 1.0])
    opt.set_channel_pressure({"rec": pressure})
    opt.set_recurrence_timescale({"rec": timescale})

    # Bump past warmup so _scarcity_enabled() returns True inside the helper.
    opt._step_count = 1

    direction = torch.ones(4)
    adjusted = opt._apply_recurrence_scarcity("log_a", direction)

    factor_raw = torch.tanh(pressure) + 1.0  # [1, ~1.76, ~1.96, ~1.995]
    factor_expected = 1.0 + (factor_raw - 1.0) * timescale

    assert torch.allclose(adjusted, direction * factor_expected, atol=1e-6)
    # Channels 0-1 unmodulated (timescale=0 → factor=1.0).
    assert adjusted[0].item() == pytest.approx(1.0, abs=1e-6)
    assert adjusted[1].item() == pytest.approx(1.0, abs=1e-6)
    # Channels 2-3 see full scarcity amplification.
    assert adjusted[2].item() > 1.5
    assert adjusted[3].item() > 1.5


def test_rare_macro_c_caps_orthogonal_when_larger_than_common(monkeypatch) -> None:
    """Macro Gerber cap: rare-orthogonal contribution magnitude must be
    bounded by ``rare_macro_c * common_norm``. This is the direction-scale
    guard that the existing factor-scale Gerber doesn't provide, and its
    absence caused the ScOpt runaway observed on 2026-04-24.

    Setup: NS is the identity, rare EMA set to a direction orthogonal
    to the common gradient with a much larger norm. The applied update
    reads as ``-lr * (common + effective_weight * orthogonal)``. At
    ``rare_macro_c=0.5`` the orthogonal portion must be capped at half
    the common norm.
    """
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
        rare_macro_c=0.5,
        rare_orthogonal_weight=1.0,
        compute_dtype=torch.float32,
    )
    opt.bind_param_names([("w", w)])
    # common gradient in direction e1, orth direction e2 with 100× norm.
    opt.set_rare_grad_ema({"w": torch.tensor([[1.0, 100.0], [0.0, 0.0]])})
    w.grad = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    opt.step()
    # -w.detach() is the applied update: common_f + ew * orth_f, scaled
    # by Muon's rectangular rows/cols factor (rows==cols here → 1.0).
    applied = -w.detach()
    common = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    orth_contribution = applied - common
    orth_norm = float(orth_contribution.square().sum().sqrt())
    common_norm = float(common.square().sum().sqrt())
    cap = 0.5 * common_norm
    assert orth_norm <= cap * (1.0 + 1e-5), (
        f"orth contribution norm={orth_norm:.4f} exceeds cap={cap:.4f}"
    )
    trace = opt.scarcity_trace()
    assert "rare_macro_cap_fired" in trace
    # The cap should have fired: rare was 100× common, well above c=0.5.
    assert trace["rare_macro_cap_fired"]["median"] > 0.5


def test_rare_macro_c_zero_disables_cap(monkeypatch) -> None:
    """``rare_macro_c=0.0`` preserves the pre-fix behavior (no cap), so
    existing configs that predate the guard keep their bitwise updates."""
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
        rare_macro_c=0.0,
        rare_orthogonal_weight=1.0,
        compute_dtype=torch.float32,
    )
    opt.bind_param_names([("w", w)])
    opt.set_rare_grad_ema({"w": torch.tensor([[1.0, 100.0], [0.0, 0.0]])})
    w.grad = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    opt.step()
    # Without cap, orth_contribution ≈ rare_orthogonal = [0, 100, 0, 0].
    applied = -w.detach()
    common = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    orth = applied - common
    # The orthogonal component should be the full [0, 100, 0, 0] (up to
    # projection cleanup — rare was [1, 100, 0, 0] with parallel = [1,0,0,0]).
    assert abs(float(orth[0, 1]) - 100.0) < 1e-4


def test_scarcity_pressure_upper_clamp_gerber_catches_outliers() -> None:
    """Gerber-style upper clamp on pressure: a handful of exploded-CE
    tokens get clamped so they can't dominate the rare-gradient
    accumulation. The clamp uses a winsorized-std scale so the
    outliers themselves don't inflate the threshold they'd be caught by.
    """
    # 20 tokens: 19 well-calibrated (ce − baseline = 0.5), one exploded
    # (ce − baseline = 100). The outlier must shrink; the normal-scale
    # values must survive.
    ce_values = [0.75] * 19 + [100.25]
    ce = torch.tensor([ce_values], dtype=torch.float32)
    targets = torch.arange(20).reshape(1, -1)
    token_frequencies = torch.ones(20)
    pressure_no_clamp = scarcity_pressure_from_ce(
        ce,
        targets,
        token_frequencies=token_frequencies,
        baseline=0.25,
    )
    assert float(pressure_no_clamp.max()) > 50.0
    pressure_clamped = scarcity_pressure_from_ce(
        ce,
        targets,
        token_frequencies=token_frequencies,
        baseline=0.25,
        upper_c=3.0,
        upper_floor=1.0,
    )
    # Outlier should shrink to O(1), normal-scale tokens unchanged.
    assert float(pressure_clamped.max()) < 10.0
    # Normal tokens (value ≈ 0.72 after rarity scaling) are well under
    # the clamp, so they pass through unchanged.
    normal_max_no_clamp = float(pressure_no_clamp[0, :19].max())
    normal_max_clamped = float(pressure_clamped[0, :19].max())
    assert abs(normal_max_clamped - normal_max_no_clamp) < 1e-6


def test_scarcity_pressure_upper_clamp_floor_is_respected_when_std_zero() -> None:
    """When pressure has zero std (uniform), the Gerber clamp collapses
    to the floor — guaranteeing a non-zero upper bound."""
    ce = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([[0, 1, 2]])
    token_frequencies = torch.ones(3)
    pressure = scarcity_pressure_from_ce(
        ce,
        targets,
        token_frequencies=token_frequencies,
        baseline=0.0,
        upper_c=3.0,
        upper_floor=0.5,
    )
    # All pressures equal, so clamp is 3 * 0 = 0 unless floor kicks in.
    assert float(pressure.max()) <= 0.5 + 1e-6
