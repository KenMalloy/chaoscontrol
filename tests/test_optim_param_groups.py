"""Tests for SSM-aware parameter grouping.

The grouping helper is the static half of the "two-timescale" recipe that
S4/S5/HOPE primary sources converge on. These tests pin:

  * classification rules (shape-based + spectral-name override)
  * group hyperparameter math (dynamics_lr_mul, zero WD on spectral/no_decay)
  * back-compat (``"flat"`` mode returns the pre-grouping param list)
  * end-to-end: Muon with grouped params moves log_a less than matrix weights
    per identical gradient, and doesn't decay log_a at all.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from chaoscontrol.optim import (
    build_optimizer_params,
    classify_param,
    ssm_three_group_params,
)
from chaoscontrol.optim.muon import Muon


class _TinySSM(nn.Module):
    """Minimal shape-match for the classifier.

    Mirrors the CareSSMCore parameter naming: ``layers.{i}.core.log_a``
    is the spectral param; ``final_norm.weight`` is a 1D norm gain;
    everything else is a 2D matrix weight.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 6)
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].core = nn.Module()
        self.layers[0].core.log_a = nn.Parameter(torch.zeros(6))
        self.layers[0].core.in_proj = nn.Linear(6, 6, bias=False)
        self.layers[0].core.out_proj = nn.Linear(6, 6, bias=False)
        self.final_norm_weight = nn.Parameter(torch.ones(6))
        self.lm_head = nn.Linear(6, 8, bias=False)


class TestClassifyParam:
    def test_log_a_is_dynamics(self) -> None:
        p = nn.Parameter(torch.zeros(6))
        assert classify_param("layers.0.core.log_a", p) == "dynamics"

    def test_all_spectral_suffixes_classify_as_dynamics(self) -> None:
        for suffix in ("log_a", "log_r", "theta", "skew_params", "log_gamma", "U", "V"):
            p = nn.Parameter(torch.zeros(3))
            assert classify_param(f"layers.0.core.{suffix}", p) == "dynamics", suffix

    def test_1d_nonspectral_is_no_decay(self) -> None:
        p = nn.Parameter(torch.ones(6))
        assert classify_param("final_norm.weight", p) == "no_decay"
        p_bias = nn.Parameter(torch.zeros(4))
        assert classify_param("some.module.bias", p_bias) == "no_decay"

    def test_2d_matrix_is_main(self) -> None:
        p = nn.Parameter(torch.zeros(6, 6))
        assert classify_param("layers.0.core.in_proj.weight", p) == "main"
        p_embed = nn.Parameter(torch.zeros(8, 6))
        assert classify_param("embed.weight", p_embed) == "main"
        p_head = nn.Parameter(torch.zeros(6, 8))
        assert classify_param("lm_head.weight", p_head) == "main"

    def test_scalar_is_no_decay(self) -> None:
        """0-D tensors hit the ``ndim <= 1`` branch, not dynamics."""
        p = nn.Parameter(torch.tensor(1.0))
        assert classify_param("gate_bias", p) == "no_decay"


class TestSSMThreeGroupParams:
    def test_three_groups_produced_for_full_ssm(self) -> None:
        model = _TinySSM()
        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=0.064,
            weight_decay=0.01,
            dynamics_lr_mul=0.1,
        )
        names = {group["name"]: group for group in groups}
        assert set(names) == {"dynamics", "no_decay", "main"}, names.keys()

    def test_dynamics_group_contains_only_log_a(self) -> None:
        model = _TinySSM()
        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=0.064,
            weight_decay=0.01,
        )
        dynamics = next(g for g in groups if g["name"] == "dynamics")
        assert len(dynamics["params"]) == 1
        assert dynamics["params"][0] is model.layers[0].core.log_a

    def test_no_decay_group_contains_norm_weight(self) -> None:
        model = _TinySSM()
        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=0.064,
            weight_decay=0.01,
        )
        no_decay = next(g for g in groups if g["name"] == "no_decay")
        assert len(no_decay["params"]) == 1
        assert no_decay["params"][0] is model.final_norm_weight

    def test_main_group_contains_all_matrix_weights(self) -> None:
        model = _TinySSM()
        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=0.064,
            weight_decay=0.01,
        )
        main = next(g for g in groups if g["name"] == "main")
        main_params = set(id(p) for p in main["params"])
        expected = {
            id(model.embed.weight),
            id(model.layers[0].core.in_proj.weight),
            id(model.layers[0].core.out_proj.weight),
            id(model.lm_head.weight),
        }
        assert main_params == expected

    def test_dynamics_lr_is_base_times_mul(self) -> None:
        model = _TinySSM()
        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=0.064,
            weight_decay=0.01,
            dynamics_lr_mul=0.25,
        )
        dynamics = next(g for g in groups if g["name"] == "dynamics")
        assert dynamics["lr"] == pytest.approx(0.064 * 0.25)
        assert dynamics["adamw_lr"] == pytest.approx(0.064 * 0.25)

    def test_zero_wd_on_dynamics_and_no_decay(self) -> None:
        model = _TinySSM()
        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=0.064,
            weight_decay=0.01,
        )
        dynamics = next(g for g in groups if g["name"] == "dynamics")
        no_decay = next(g for g in groups if g["name"] == "no_decay")
        main = next(g for g in groups if g["name"] == "main")
        assert dynamics["weight_decay"] == 0.0
        assert dynamics["adamw_weight_decay"] == 0.0
        assert no_decay["weight_decay"] == 0.0
        assert no_decay["adamw_weight_decay"] == 0.0
        assert main["weight_decay"] == 0.01
        assert main["adamw_weight_decay"] == 0.01

    def test_main_group_uses_base_lr(self) -> None:
        model = _TinySSM()
        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=0.064,
            weight_decay=0.01,
        )
        main = next(g for g in groups if g["name"] == "main")
        no_decay = next(g for g in groups if g["name"] == "no_decay")
        assert main["lr"] == pytest.approx(0.064)
        assert no_decay["lr"] == pytest.approx(0.064)

    def test_no_grad_params_are_skipped(self) -> None:
        model = _TinySSM()
        model.embed.weight.requires_grad_(False)
        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=0.064,
            weight_decay=0.01,
        )
        all_params = [id(p) for g in groups for p in g["params"]]
        assert id(model.embed.weight) not in all_params

    def test_empty_groups_are_dropped(self) -> None:
        """A model with no spectral params shouldn't have an empty
        ``dynamics`` group in the output list.
        """

        class _NoSpectral(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Linear(4, 4, bias=False)

        model = _NoSpectral()
        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=0.01,
            weight_decay=0.0,
        )
        names = [g["name"] for g in groups]
        assert "dynamics" not in names
        assert names == ["main"]


class TestBuildOptimizerParams:
    def test_flat_returns_list_of_tensors(self) -> None:
        model = _TinySSM()
        result = build_optimizer_params(
            list(model.named_parameters()),
            grouping="flat",
            base_lr=0.064,
            weight_decay=0.01,
        )
        assert all(isinstance(item, torch.Tensor) for item in result)
        assert len(result) == sum(1 for p in model.parameters() if p.requires_grad)

    def test_ssm_three_group_returns_list_of_dicts(self) -> None:
        model = _TinySSM()
        result = build_optimizer_params(
            list(model.named_parameters()),
            grouping="ssm_three_group",
            base_lr=0.064,
            weight_decay=0.01,
        )
        assert all(isinstance(item, dict) and "params" in item for item in result)

    def test_unknown_grouping_raises(self) -> None:
        model = _TinySSM()
        with pytest.raises(ValueError, match="optimizer_param_grouping"):
            build_optimizer_params(
                list(model.named_parameters()),
                grouping="not_a_mode",
                base_lr=0.064,
                weight_decay=0.01,
            )


class TestMuonEndToEndWithGrouping:
    """Integration check: the group hyperparameters actually propagate
    into Muon's per-group lr/wd reads, not just the initial dict.
    """

    def test_log_a_moves_less_than_matrix_under_identical_grads(self) -> None:
        model = _TinySSM()
        with torch.no_grad():
            model.layers[0].core.log_a.fill_(0.0)
            model.layers[0].core.in_proj.weight.fill_(0.0)

        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=1.0,
            weight_decay=0.0,
            dynamics_lr_mul=0.1,
        )
        opt = Muon(
            groups,
            lr=1.0,
            momentum=0.0,
            nesterov=False,
            ns_steps=1,
            weight_decay=0.0,
            compute_dtype=torch.float32,
            fused=False,
        )
        opt.bind_param_names(list(model.named_parameters()))
        model.layers[0].core.log_a.grad = torch.ones(6)
        model.layers[0].core.in_proj.weight.grad = torch.ones(6, 6)
        opt.step()

        log_a_disp = model.layers[0].core.log_a.detach().abs().max().item()
        in_proj_disp = (
            model.layers[0].core.in_proj.weight.detach().abs().max().item()
        )
        # With dynamics_lr_mul=0.1, log_a's effective lr is 10× smaller.
        # AdamW fallback adds an adaptive-moment term for log_a, but the
        # first step with uniform grad still respects the 0.1× lr ratio
        # in the final update magnitude ordering.
        assert log_a_disp < in_proj_disp, (
            f"expected log_a displacement < matrix displacement, "
            f"got log_a={log_a_disp:.6f}, in_proj={in_proj_disp:.6f}"
        )

    def test_log_a_receives_no_weight_decay(self) -> None:
        model = _TinySSM()
        with torch.no_grad():
            model.layers[0].core.log_a.fill_(1.0)  # so WD would bite

        groups = ssm_three_group_params(
            list(model.named_parameters()),
            base_lr=0.064,
            weight_decay=0.1,
            dynamics_lr_mul=0.1,
        )
        opt = Muon(
            groups,
            lr=0.064,
            momentum=0.0,
            nesterov=False,
            ns_steps=1,
            weight_decay=0.1,
            compute_dtype=torch.float32,
            fused=False,
        )
        opt.bind_param_names(list(model.named_parameters()))
        # Zero grad + nonzero WD would shrink p.data under the WD rule.
        # With WD=0 on the dynamics group, log_a's value must not change.
        model.layers[0].core.log_a.grad = torch.zeros(6)
        model.layers[0].core.in_proj.weight.grad = torch.zeros(6, 6)
        log_a_before = model.layers[0].core.log_a.detach().clone()
        opt.step()
        log_a_after = model.layers[0].core.log_a.detach()
        assert torch.allclose(log_a_before, log_a_after, atol=0.0), (
            f"log_a moved under zero grad — WD leaked past the dynamics group: "
            f"before={log_a_before}, after={log_a_after}"
        )
