"""Tests for Exp 23 fastest-path training helpers."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from chaoscontrol.data import batch_from_starts
from chaoscontrol import train_ssm


_BACKEND_ENV_KEYS = (
    "CHAOSCONTROL_DIAG_SCAN_BACKEND",
    "CHAOSCONTROL_POST_SCAN_BACKEND",
)


@pytest.fixture(autouse=True)
def _restore_backend_env_after_test():
    """Restore Exp23 backend env vars after every test in this module.

    ``_load_runner_module`` invokes ``configure_exp23_fast_backend_defaults``
    at module import time, which mutates ``os.environ`` via ``setdefault``.
    That is invisible to pytest's ``monkeypatch`` fixture and leaks the
    runner's default backend selection into later tests — specifically
    ``tests/test_core.py`` resolvers would pick up
    ``CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan`` and emit a harmless-but-noisy
    RuntimeWarning on dev machines without the extension built. This fixture
    snapshots the relevant keys before each test and restores them after,
    so the leak does not cross test boundaries.
    """
    snapshot = {key: os.environ.get(key) for key in _BACKEND_ENV_KEYS}
    try:
        yield
    finally:
        for key, value in snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "23_fast_path" / "fast_path.py"
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"
LAUNCH_PATH = REPO / "experiments" / "23_fast_path" / "launch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("exp23_fast_path", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("exp23_runner_fast_path", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_crct_object_collectives_use_gloo_side_group() -> None:
    """Telemetry objects should not ride the NCCL tensor group.

    On 8xH100, the training step completed but rank 0 died while gathering
    CRCT diagnostics with ``gather_object`` over the NCCL all-rank group.
    Keep a Gloo side-group for Python objects so telemetry cannot poison the
    final train/eval run.
    """
    source = RUNNER_PATH.read_text()
    assert 'object_group = dist.new_group(list(range(world_size_)), backend="gloo")' in source
    assert "group=object_group or all_group" in source


def test_score_stage_timing_config_reaches_main_train_call() -> None:
    """The Exp26 stage profiler must wire through the real train call.

    A previous bug set ``crct_score_stage_timing_enabled`` in the matrix but
    only forwarded it through the warmup call, so the 8xH100 pulse silently
    produced zero stage samples.
    """
    source = RUNNER_PATH.read_text()
    assert source.count("crct_score_stage_timing_enabled=bool(") >= 2
    assert source.count('config.get("crct_score_stage_timing_enabled", False)') >= 2


def test_crct_memory_rank_checks_wall_clock_even_when_idle() -> None:
    """Idle memory rank must stop from wall clock, not local score steps."""
    mod = _load_runner_module()

    assert mod._should_stop_memory_rank_loop(
        steps=0,
        elapsed_s=10.0,
        budget_seconds=10.0,
        stop_margin_seconds=0.0,
        max_steps=1,
    )
    assert not mod._should_stop_memory_rank_loop(
        steps=0,
        elapsed_s=9.9,
        budget_seconds=10.0,
        stop_margin_seconds=0.0,
        max_steps=1,
    )
    assert mod._should_stop_memory_rank_loop(
        steps=1,
        elapsed_s=0.0,
        budget_seconds=10.0,
        stop_margin_seconds=0.0,
        max_steps=1,
    )


class _TinyTrainStepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, 6, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)


class _TinyPacketTrainStepModel(_TinyTrainStepModel):
    def encode(
        self,
        inputs: torch.Tensor,
        *,
        memory_mode: str = "packet",
        episodic_residual: torch.Tensor | None = None,
        episodic_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encoder(inputs)
        if episodic_residual is None or episodic_gate is None:
            return hidden
        assert memory_mode == "packet"
        return hidden + episodic_residual * episodic_gate.unsqueeze(-1)


class _TinySemanticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].core = nn.Module()
        self.layers[0].core.log_a = nn.Parameter(torch.randn(3))
        self.layers[0].core.in_proj = nn.Linear(4, 3, bias=False)
        self.layers[0].core.select_proj = nn.Linear(4, 3, bias=False)
        self.layers[0].core.gate_proj = nn.Linear(4, 3, bias=False)
        self.layers[0].core.delta_proj = nn.Linear(4, 3, bias=False)
        self.layers[0].core.out_proj = nn.Linear(3, 4, bias=False)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, 6, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class _TinyTokenTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(6, 4)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, 6, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embed(inputs)


class _TinyScOptLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(6, 4)
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].core = nn.Linear(4, 4, bias=False)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, 6, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers[0].core(self.embed(inputs))


def _load_launch_module():
    spec = importlib.util.spec_from_file_location("exp23_launch", LAUNCH_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clear_backend_env() -> None:
    """Clear backend env vars so ``setdefault`` in the runner actually fires.

    Pair this with the autouse ``_restore_backend_env_after_test`` fixture —
    the fixture handles post-test restoration, so we only need to clear at
    the start of tests that depend on a clean slate.
    """
    for key in _BACKEND_ENV_KEYS:
        os.environ.pop(key, None)


def test_build_optimizer_can_create_semantic_optimizer(monkeypatch):
    mod = _load_runner_module()

    class FakeSemanticOptimizer(torch.optim.SGD):
        def __init__(self, params, **kwargs):
            self.kwargs = dict(kwargs)
            super().__init__(params, lr=kwargs["lr"])
            self.bound_named_params = None

        def bind_param_names(self, named_params):
            self.bound_named_params = list(named_params)

        def beta_trace(self):
            return {
                "beta_min": 0.1,
                "beta_max": 0.9,
                "beta_mean": 0.5,
            }

    monkeypatch.setattr(mod, "SemanticOptimizer", FakeSemanticOptimizer)

    model = _TinySemanticModel()
    optimizer = mod._build_optimizer(
        {
            "optimizer": "semantic",
            "base_lr": 0.01,
            "weight_decay": 0.02,
            "semantic_layer_index": 0,
            "semantic_momentum_min": 0.25,
        },
        model,
    )

    assert isinstance(optimizer, FakeSemanticOptimizer)
    assert optimizer.kwargs["a_param_name"] == "layers.0.core.log_a"
    assert optimizer.kwargs["channel_map"] == {
        "layers.0.core.in_proj.weight": 0,
        "layers.0.core.select_proj.weight": 0,
        "layers.0.core.gate_proj.weight": 0,
        "layers.0.core.delta_proj.weight": 0,
        "layers.0.core.out_proj.weight": 1,
    }
    assert optimizer.kwargs["momentum_min"] == 0.25
    assert optimizer.bound_named_params is not None


def test_build_optimizer_defaults_to_flat_grouping():
    """Back-compat: no ``optimizer_param_grouping`` knob in config ⇒
    every param lands in ``param_groups[0]`` with uniform lr/wd.

    This pins the invariant that the Phase 0 locked config (which has
    no grouping knob) continues to train exactly the same way after the
    grouping code lands.
    """
    mod = _load_runner_module()
    model = _TinySemanticModel()
    optimizer = mod._build_optimizer(
        {"optimizer": "muon", "base_lr": 0.05, "weight_decay": 0.02},
        model,
    )
    # One flat group. lr/wd/adamw_lr/adamw_wd should all match the defaults.
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(0.05)
    assert group["adamw_lr"] == pytest.approx(0.05)
    assert group["weight_decay"] == pytest.approx(0.02)
    assert group["adamw_weight_decay"] == pytest.approx(0.02)


def test_build_optimizer_crct_uses_role_dispatch_for_muon():
    mod = _load_runner_module()
    model = _TinySemanticModel()
    model.outer_model = nn.Linear(4, 4, bias=False)
    model.bucket_prototypes_module = nn.Linear(4, 4, bias=False)
    optimizer = mod._build_optimizer(
        {
            "optimizer": "muon",
            "base_lr": 0.05,
            "weight_decay": 0.02,
            "crct_enabled": True,
            "optimizer_log_a_beta_ema": 0.98,
            "optimizer_log_a_beta_min": 0.25,
        },
        model,
    )

    assert "layers.0.core.in_proj.weight" in optimizer._matrix_param_names
    assert "layers.0.core.out_proj.weight" in optimizer._matrix_param_names
    assert "layers.0.core.delta_proj.weight" not in optimizer._matrix_param_names
    assert all(
        not name.startswith(("outer_model.", "bucket_prototypes_module."))
        for name in optimizer._param_name_by_id.values()
    )
    trace = optimizer.ssm_role_trace()
    assert trace["log_a_beta_coupling"] is True
    assert trace["log_a_beta_ema"] == pytest.approx(0.98)
    assert trace["log_a_beta_min_config"] == pytest.approx(0.25)
    diagnostics = mod._optimizer_diagnostics(optimizer)
    assert diagnostics["excluded_params"]["outer_model"] == 1
    assert diagnostics["excluded_params"]["bucket_prototypes"] == 1


def test_build_optimizer_crct_keeps_low_rank_delta_on_adamw_path():
    mod = _load_runner_module()
    from chaoscontrol.model import CareStudentLM

    model = CareStudentLM(
        vocab_size=16,
        dim=8,
        num_layers=1,
        ff_mult=2,
        a_mode="diag",
        ssm_delta_rank=2,
        outer_model_dim=4,
        outer_model_type="multislot",
        buffer_mode="append_only",
    )
    optimizer = mod._build_optimizer(
        {
            "optimizer": "muon",
            "base_lr": 0.05,
            "weight_decay": 0.02,
            "crct_enabled": True,
        },
        model,
    )

    names = optimizer._param_name_by_id.values()
    assert "layers.0.core.delta_proj.down.weight" in names
    assert "layers.0.core.delta_proj.up.weight" in names
    assert "layers.0.core.delta_proj.down.weight" not in optimizer._matrix_param_names
    assert "layers.0.core.delta_proj.up.weight" not in optimizer._matrix_param_names


def test_build_optimizer_ssm_three_group_splits_by_role():
    """``optimizer_param_grouping='ssm_three_group'`` produces three
    param groups (dynamics / no_decay / main) with distinct lr+wd
    values. This is the SSM-aware stack rule from S4/S5/HOPE.

    Uses _TinySemanticModel which has log_a (dynamics) + matrix weights
    (main). No 1D non-spectral param, so ``no_decay`` is dropped — the
    grouping helper prunes empty groups.
    """
    mod = _load_runner_module()
    model = _TinySemanticModel()
    optimizer = mod._build_optimizer(
        {
            "optimizer": "muon",
            "base_lr": 0.064,
            "weight_decay": 0.01,
            "optimizer_param_grouping": "ssm_three_group",
            "optimizer_dynamics_lr_mul": 0.1,
        },
        model,
    )
    group_names = [g.get("name") for g in optimizer.param_groups]
    assert "dynamics" in group_names
    assert "main" in group_names
    dynamics = next(g for g in optimizer.param_groups if g.get("name") == "dynamics")
    main = next(g for g in optimizer.param_groups if g.get("name") == "main")
    # log_a is the only dynamics-class param in the tiny model.
    assert dynamics["lr"] == pytest.approx(0.064 * 0.1)
    assert dynamics["weight_decay"] == 0.0
    assert dynamics["adamw_weight_decay"] == 0.0
    assert main["lr"] == pytest.approx(0.064)
    assert main["weight_decay"] == pytest.approx(0.01)
    dynamics_ids = {id(p) for p in dynamics["params"]}
    assert id(model.layers[0].core.log_a) in dynamics_ids


def test_build_optimizer_ssm_three_group_propagates_to_scopt():
    """The grouping is optimizer-agnostic — ScOpt should see three
    groups with the same lr/wd split as Muon would.
    """
    mod = _load_runner_module()
    model = _TinySemanticModel()
    optimizer = mod._build_optimizer(
        {
            "optimizer": "scopt",
            "base_lr": 0.064,
            "weight_decay": 0.01,
            "optimizer_param_grouping": "ssm_three_group",
            "optimizer_dynamics_lr_mul": 0.1,
            "scopt_layer_index": 0,
            "scopt_warmup_steps": 5,
        },
        model,
    )
    group_names = [g.get("name") for g in optimizer.param_groups]
    assert "dynamics" in group_names
    dynamics = next(g for g in optimizer.param_groups if g.get("name") == "dynamics")
    assert dynamics["lr"] == pytest.approx(0.064 * 0.1)
    assert dynamics["weight_decay"] == 0.0


def test_build_optimizer_can_create_scarcity_optimizer(monkeypatch):
    mod = _load_runner_module()

    class FakeScOpt(torch.optim.SGD):
        def __init__(self, params, **kwargs):
            self.kwargs = dict(kwargs)
            super().__init__(params, lr=kwargs["lr"])
            self.bound_named_params = None

        def bind_param_names(self, named_params):
            self.bound_named_params = list(named_params)

    monkeypatch.setattr(mod, "ScarcityAwareOptimizer", FakeScOpt)

    model = _TinySemanticModel()
    optimizer = mod._build_optimizer(
        {
            "optimizer": "scopt",
            "base_lr": 0.01,
            "weight_decay": 0.02,
            "scopt_layer_index": 0,
            "scopt_warmup_steps": 12,
            "scopt_rare_ema_decay": 0.8,
            "scopt_rare_orthogonal_weight": 0.75,
        },
        model,
    )

    assert isinstance(optimizer, FakeScOpt)
    assert optimizer.kwargs["warmup_steps"] == 12
    assert optimizer.kwargs["rare_ema_decay"] == 0.8
    assert optimizer.kwargs["rare_orthogonal_weight"] == 0.75
    assert optimizer.kwargs["row_param_names"] == {"embed.weight", "lm_head.weight"}
    assert optimizer.kwargs["matrix_scarcity_map"] == {
        "layers.0.core.in_proj.weight": (
            "layers.0.core.in_proj.__out__",
            "layers.0.core.in_proj.__in__",
        ),
        "layers.0.core.select_proj.weight": (
            "layers.0.core.select_proj.__out__",
            "layers.0.core.select_proj.__in__",
        ),
        "layers.0.core.gate_proj.weight": (
            "layers.0.core.gate_proj.__out__",
            "layers.0.core.gate_proj.__in__",
        ),
        "layers.0.core.delta_proj.weight": (
            "layers.0.core.delta_proj.__out__",
            "layers.0.core.delta_proj.__in__",
        ),
        "layers.0.core.out_proj.weight": (
            "layers.0.core.out_proj.__out__",
            "layers.0.core.out_proj.__in__",
        ),
    }
    assert optimizer.kwargs["recurrence_scarcity_map"] == {
        "layers.0.core.log_a": "layers.0.core.in_proj.__out__",
    }
    assert optimizer.bound_named_params is not None


def test_scopt_train_step_updates_rare_and_row_state():
    mod = _load_runner_module()
    from chaoscontrol.optim.scopt import ScarcityAwareOptimizer

    torch.manual_seed(0)
    model = _TinyTokenTrainModel()
    optimizer = ScarcityAwareOptimizer(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        compute_dtype=torch.float32,
    )
    optimizer.bind_param_names(list(model.named_parameters()))
    inputs = torch.tensor([[0, 1, 2], [1, 2, 3]])
    targets = torch.tensor([[1, 2, 3], [2, 3, 4]])
    token_frequencies = torch.ones(6)

    loss, pending = mod._run_scopt_train_step(
        model=model,
        optimizer=optimizer,
        inputs=inputs,
        targets=targets,
        token_frequencies=token_frequencies,
        precision="fp32",
        ddp_active=False,
        world_size=1,
        step=0,
        split_interval=1,
    )
    mod._apply_scopt_pending(optimizer, pending, skip=False)

    assert loss.ndim == 0
    assert optimizer.state[model.embed.weight]["rare_grad_ema"].abs().sum() > 0
    assert optimizer.state[model.lm_head.weight]["rare_grad_ema"].abs().sum() > 0
    trace = optimizer.scarcity_trace()
    assert trace["row_pressure"]["max"] > 0.0
    assert "pressure_stats" in trace


def test_scopt_train_step_sets_channel_pressure_from_activation_gradients():
    mod = _load_runner_module()
    from chaoscontrol.optim.scopt import ScarcityAwareOptimizer

    torch.manual_seed(1)
    model = _TinyScOptLayerModel()
    optimizer = ScarcityAwareOptimizer(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        matrix_scarcity_map={
            "layers.0.core.weight": (
                "layers.0.core.__out__",
                "layers.0.core.__in__",
            ),
        },
        compute_dtype=torch.float32,
    )
    optimizer.bind_param_names(list(model.named_parameters()))

    _, pending = mod._run_scopt_train_step(
        model=model,
        optimizer=optimizer,
        inputs=torch.tensor([[0, 1, 2]]),
        targets=torch.tensor([[1, 2, 3]]),
        token_frequencies=torch.ones(6),
        precision="fp32",
        ddp_active=False,
        world_size=1,
        step=0,
        split_interval=1,
    )
    mod._apply_scopt_pending(optimizer, pending, skip=False)

    trace = optimizer.scarcity_trace()
    assert "layers.0.core.__out__" in trace["channel_pressure_keys"]
    assert "layers.0.core.__in__" in trace["channel_pressure_keys"]


def test_scopt_common_step_uses_fused_ce_and_updates_bucket_baseline(monkeypatch):
    mod = _load_runner_module()
    from chaoscontrol.core import RMSNorm
    from chaoscontrol.optim.scopt import FrequencyBucketBaseline

    class _TinyNormTokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(6, 4)
            self.final_norm = RMSNorm(4)
            self.lm_head = nn.Linear(4, 6, bias=False)

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.embed(inputs)

    model = _TinyNormTokenModel()
    baseline = FrequencyBucketBaseline(torch.ones(6), decay=0.5)
    calls = []

    def fake_fused_lm_head_backward_with_ce(
        hidden,
        final_norm,
        lm_head,
        targets,
        *,
        backend,
        tile_size,
    ):
        calls.append((backend, tile_size))
        loss = hidden.square().mean()
        loss.backward()
        per_token_ce = torch.arange(
            targets.numel(),
            dtype=torch.float32,
            device=targets.device,
        )
        return loss.detach(), per_token_ce

    monkeypatch.setattr(
        mod,
        "fused_lm_head_backward_with_ce",
        fake_fused_lm_head_backward_with_ce,
    )

    loss = mod._run_scopt_common_train_step(
        model=model,
        inputs=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        targets=torch.tensor([[1, 2, 3], [2, 3, 4]]),
        precision="fp32",
        ddp_active=False,
        world_size=1,
        compile_full_path=False,
        lm_head_backward_mode="fused_streaming_cached",
        lm_head_tile_size=8192,
        grad_allreduce_mode="bulk",
        baseline=baseline,
    )

    assert loss.ndim == 0
    assert calls == [("streaming_cached", 8192)]
    state = baseline.state_dict()
    assert bool(state["initialized"].any())
    assert float(state["ema"].max()) > 0.0


def test_scopt_split_step_batches_rare_param_and_activation_grads(monkeypatch):
    mod = _load_runner_module()
    from chaoscontrol.optim.scopt import ScarcityAwareOptimizer

    torch.manual_seed(3)
    model = _TinyScOptLayerModel()
    optimizer = ScarcityAwareOptimizer(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        matrix_scarcity_map={
            "layers.0.core.weight": (
                "layers.0.core.__out__",
                "layers.0.core.__in__",
            ),
        },
        compute_dtype=torch.float32,
    )
    optimizer.bind_param_names(list(model.named_parameters()))

    original_grad = torch.autograd.grad
    calls = []

    def counting_grad(outputs, inputs, *args, **kwargs):
        calls.append(inputs)
        return original_grad(outputs, inputs, *args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counting_grad)

    _, pending = mod._run_scopt_train_step(
        model=model,
        optimizer=optimizer,
        inputs=torch.tensor([[0, 1, 2]]),
        targets=torch.tensor([[1, 2, 3]]),
        token_frequencies=torch.ones(6),
        precision="fp32",
        ddp_active=False,
        world_size=1,
        step=0,
        split_interval=1,
    )

    assert pending is not None
    assert len(calls) == 2  # common grads, then rare params + activations together.


def test_scopt_split_step_uses_fused_weighted_lm_head(monkeypatch):
    mod = _load_runner_module()
    from chaoscontrol.core import RMSNorm
    from chaoscontrol.optim.scopt import ScarcityAwareOptimizer

    class _TinyNormTokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(6, 4)
            self.final_norm = RMSNorm(4)
            self.lm_head = nn.Linear(4, 6, bias=False)

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.embed(inputs)

    model = _TinyNormTokenModel()
    optimizer = ScarcityAwareOptimizer(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        compute_dtype=torch.float32,
    )
    optimizer.bind_param_names(list(model.named_parameters()))
    common_calls = []
    rare_calls = []

    def fake_fused_lm_head_loss_with_ce(
        *,
        hidden,
        final_norm,
        lm_head,
        targets,
        backend,
        tile_size,
    ):
        common_calls.append((backend, tile_size))
        per_token_ce = torch.arange(
            1,
            targets.numel() + 1,
            dtype=torch.float32,
            device=targets.device,
        )
        return hidden.square().mean(), per_token_ce

    def fake_fused_lm_head_weighted_loss_with_ce(
        *,
        hidden,
        final_norm,
        lm_head,
        targets,
        token_weight,
        backend,
        tile_size,
    ):
        rare_calls.append((backend, tile_size, token_weight.detach().clone()))
        rare_scale = token_weight.detach().float().mean().clamp_min(1e-6)
        return hidden.square().mean() * rare_scale, torch.zeros(
            targets.numel(),
            dtype=torch.float32,
            device=targets.device,
        )

    monkeypatch.setattr(
        mod,
        "fused_lm_head_loss_with_ce",
        fake_fused_lm_head_loss_with_ce,
        raising=False,
    )
    monkeypatch.setattr(
        mod,
        "fused_lm_head_weighted_loss_with_ce",
        fake_fused_lm_head_weighted_loss_with_ce,
        raising=False,
    )

    _, pending = mod._run_scopt_train_step(
        model=model,
        optimizer=optimizer,
        inputs=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        targets=torch.tensor([[1, 2, 3], [2, 3, 4]]),
        token_frequencies=torch.ones(6),
        precision="fp32",
        ddp_active=False,
        world_size=1,
        step=0,
        split_interval=1,
        lm_head_backward_mode="fused_norm_streaming_v2",
        lm_head_tile_size=4096,
    )

    assert pending is not None
    assert common_calls == [("norm_streaming_v2", 4096)]
    assert len(rare_calls) == 1
    backend, tile_size, token_weight = rare_calls[0]
    assert (backend, tile_size) == ("norm_streaming_v2", 4096)
    assert token_weight.shape == (2, 3)
    assert token_weight.max() > 0


def test_train_fast_for_budget_routes_scopt_warmup_to_common_fast_path(monkeypatch):
    mod = _load_runner_module()
    from chaoscontrol.optim.scopt import ScarcityAwareOptimizer

    model = _TinyScOptLayerModel()
    optimizer = ScarcityAwareOptimizer(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=10,
        matrix_scarcity_map={
            "layers.0.core.weight": (
                "layers.0.core.__out__",
                "layers.0.core.__in__",
            ),
        },
        compute_dtype=torch.float32,
    )
    optimizer.bind_param_names(list(model.named_parameters()))
    common_calls = []

    def fail_split(**_kwargs):
        raise AssertionError("warmup/non-split ScOpt should not enter split path")

    def fake_common(**kwargs):
        common_calls.append(kwargs["lm_head_backward_mode"])
        for param in kwargs["model"].parameters():
            param.grad = torch.zeros_like(param)
        return torch.tensor(1.0)

    monkeypatch.setattr(mod, "_run_scopt_train_step", fail_split)
    monkeypatch.setattr(mod, "_run_scopt_common_train_step", fake_common)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="fp32",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=False,
        lm_head_backward_mode="fused_streaming_cached",
        scopt_split_interval=1,
    )

    assert result["steps"] == 2
    assert common_calls == ["fused_streaming_cached", "fused_streaming_cached"]


def test_train_fast_for_budget_rejects_scopt_non_fused_baseline(monkeypatch):
    mod = _load_runner_module()
    from chaoscontrol.optim.scopt import ScarcityAwareOptimizer

    model = _TinyScOptLayerModel()
    optimizer = ScarcityAwareOptimizer(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=10,
        matrix_scarcity_map={
            "layers.0.core.weight": (
                "layers.0.core.__out__",
                "layers.0.core.__in__",
            ),
        },
        compute_dtype=torch.float32,
    )
    optimizer.bind_param_names(list(model.named_parameters()))

    def fail_common(**_kwargs):
        raise AssertionError("ScOpt non-fused baseline should fail validation")

    monkeypatch.setattr(mod, "_run_scopt_common_train_step", fail_common)

    with pytest.raises(ValueError, match="ScOpt.*frequency baseline.*fused"):
        mod.train_fast_for_budget(
            model,
            train_tokens=torch.arange(128, dtype=torch.int16) % 6,
            train_num_tokens=128,
            stride=4,
            seq_len=3,
            batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer,
            budget_seconds=300.0,
            chunk_size=2,
            grad_clip_norm=0.0,
            fused_grad_clip=False,
            rank=0,
            world_size=1,
            seed=123,
            precision="fp32",
            stop_check_interval=1,
            stop_margin_seconds=0.0,
            vocab_size=6,
            max_steps=2,
            prefetch_batches=False,
            lm_head_backward_mode="chunked",
            scopt_split_interval=1,
            scopt_baseline_buckets=16,
        )


def test_scopt_optimizer_config_matches_real_chaos_ssm_core():
    """Smoke-test against a real CareSSMCore. The tiny test models
    above cover submodule naming convention but can't verify the map
    actually hits the real block's projections — reviewer flagged this
    gap after C5/C6 shipped broken last time against the real core."""
    mod = _load_runner_module()
    from chaoscontrol.core import CareSSMCore

    class _RealCoreShell(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 6)
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].core = CareSSMCore(dim=6, a_mode="diag")
            self.final_norm = nn.Identity()
            self.lm_head = nn.Linear(6, 8, bias=False)

        def encode(self, inputs):
            x = self.embed(inputs)
            return self.layers[0].core(x)

    model = _RealCoreShell()
    config = mod._scopt_optimizer_config(model, layer_index=0)
    matrix_map = config["matrix_scarcity_map"]
    expected_weights = {
        "layers.0.core.in_proj.weight",
        "layers.0.core.select_proj.weight",
        "layers.0.core.gate_proj.weight",
        "layers.0.core.delta_proj.weight",
        "layers.0.core.out_proj.weight",
    }
    assert set(matrix_map.keys()) == expected_weights
    for param_name, (out_key, in_key) in matrix_map.items():
        assert out_key is not None and in_key is not None
        base = param_name.removesuffix(".weight")
        assert out_key == f"{base}.__out__"
        assert in_key == f"{base}.__in__"
    assert config["recurrence_scarcity_map"] == {
        "layers.0.core.log_a": "layers.0.core.in_proj.__out__",
    }


def test_scopt_realistic_core_hooks_fire_on_real_projections(monkeypatch):
    """Every mapped submodule's pre+post hooks must actually capture
    tensors during the real CareSSMCore forward pass."""
    monkeypatch.setenv("CHAOSCONTROL_DIAG_SCAN_BACKEND", "chunked")
    monkeypatch.setenv("CHAOSCONTROL_POST_SCAN_BACKEND", "eager")
    _reload_backend_modules()

    mod = _load_runner_module()
    from chaoscontrol.core import CareSSMCore
    from chaoscontrol.optim.scopt import ScarcityAwareOptimizer

    class _RealCoreShell(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 6)
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].core = CareSSMCore(dim=6, a_mode="diag")
            self.final_norm = nn.Identity()
            self.lm_head = nn.Linear(6, 8, bias=False)

        def encode(self, inputs):
            x = self.embed(inputs)
            return self.layers[0].core(x)

    torch.manual_seed(2)
    model = _RealCoreShell()
    opt_cfg = mod._scopt_optimizer_config(model, layer_index=0)
    optimizer = ScarcityAwareOptimizer(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        weight_decay=0.0,
        warmup_steps=0,
        compute_dtype=torch.float32,
        **opt_cfg,
    )
    optimizer.bind_param_names(list(model.named_parameters()))

    _, pending = mod._run_scopt_train_step(
        model=model,
        optimizer=optimizer,
        inputs=torch.tensor([[0, 1, 2, 3]]),
        targets=torch.tensor([[1, 2, 3, 4]]),
        token_frequencies=torch.ones(8),
        precision="fp32",
        ddp_active=False,
        world_size=1,
        step=0,
        split_interval=1,
    )
    assert pending is not None
    keys = {key for key, _ in pending.channel_pressure_items}
    for submodule in ("in_proj", "select_proj", "gate_proj", "delta_proj", "out_proj"):
        assert f"layers.0.core.{submodule}.__in__" in keys, (
            f"missing pre-hook for {submodule}"
        )
        assert f"layers.0.core.{submodule}.__out__" in keys, (
            f"missing post-hook for {submodule}"
        )


def test_runner_defaults_to_native_scan_without_inductor():
    _clear_backend_env()

    _load_runner_module()

    assert os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] == "ssm_scan"
    assert os.environ["CHAOSCONTROL_POST_SCAN_BACKEND"] == "eager"


def test_runner_preserves_explicit_backend_overrides(monkeypatch):
    monkeypatch.setenv("CHAOSCONTROL_DIAG_SCAN_BACKEND", "chunked")
    monkeypatch.setenv("CHAOSCONTROL_POST_SCAN_BACKEND", "compile")

    _load_runner_module()

    assert os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] == "chunked"
    assert os.environ["CHAOSCONTROL_POST_SCAN_BACKEND"] == "compile"


def _reload_backend_modules() -> None:
    """Reload the lazy backend modules so their resolver caches start clean.

    Without this, a prior test that triggered resolution leaves a cached
    backend in place and subsequent tests would not actually exercise the
    env-var path.
    """
    for mod_name in ("chaoscontrol.core_fused", "chaoscontrol.core"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])


def test_runner_configure_steers_resolvers_even_when_core_imported_first():
    """Runner's backend defaults must take effect even when
    ``chaoscontrol.core`` is imported *before* the runner module.

    The resolvers in ``chaoscontrol.core`` / ``chaoscontrol.core_fused`` are
    lazy — env vars are read on the first resolver call, not at import
    time. Exp23's ``configure_exp23_fast_backend_defaults`` sets
    ``CHAOSCONTROL_DIAG_SCAN_BACKEND`` and
    ``CHAOSCONTROL_POST_SCAN_BACKEND`` at runner module load, BEFORE
    its own chaoscontrol imports. This test pins the ordering contract:
    even when core is already imported into the process, the runner's
    configure still steers the eventual resolver call away from Inductor.

    Without this, the earlier ``test_runner_defaults_to_native_scan_without_inductor``
    check would pass even if the env-var set were moved after the
    chaoscontrol imports, because it only asserts the final env state,
    not the resolver-vs-configure ordering the fix actually depends on.
    """
    _clear_backend_env()

    _reload_backend_modules()

    # Import core and core_fused BEFORE the runner. This is the ordering
    # under test: a consumer of chaoscontrol that touches the SSM core
    # before loading the Exp23 runner must still end up on the
    # no-Inductor path once the runner's configure runs.
    import chaoscontrol.core as cc_core
    import chaoscontrol.core_fused as cc_fused

    # Importing the modules does NOT resolve the backend; resolution is
    # deferred to the first resolver call. This is the invariant that
    # lets the lazy-env-var fix work.
    assert cc_core._diag_recurrence_impl is None
    assert cc_fused._post_scan_impl is None

    # Load the runner. configure_exp23_fast_backend_defaults runs at
    # module top and sets the env vars before any chaoscontrol import
    # in the runner's own body.
    _load_runner_module()

    assert os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] == "ssm_scan"
    assert os.environ["CHAOSCONTROL_POST_SCAN_BACKEND"] == "eager"

    # The key assertion: resolving the backend NOW, on the already-imported
    # core module, honours the env var the runner set after import. If
    # configure ran too late, or if the resolver cached a pre-configure
    # decision, this would land on ``compile`` and Inductor would fire at
    # the next training step. On H100 pods the native extension is
    # available and backend == "ssm_scan"; on dev machines without the
    # built wheel the resolver warns and falls back to "chunked". Both
    # are acceptable — the invariant is that neither is "compile".
    with warnings.catch_warnings():
        # The ssm_scan fallback on dev machines emits a legitimate
        # RuntimeWarning; it is informational, not a failure. Silence it
        # here so the test stays green on both pod and dev.
        warnings.simplefilter("ignore", RuntimeWarning)
        diag_info = cc_core.get_diag_recurrence_backend()
        post_info = cc_fused.get_post_scan_backend()

    assert diag_info["backend"] in {"ssm_scan", "chunked"}, diag_info
    assert diag_info["backend"] != "compile", diag_info
    assert post_info["backend"] == "eager", post_info

    # Reset so later tests start from a clean resolver cache.
    _reload_backend_modules()


def test_vectorized_batch_matches_reference_batcher():
    mod = _load_module()
    tokens = torch.arange(40, dtype=torch.int16)
    starts = torch.tensor([0, 5, 13, 21], dtype=torch.long)

    got_inputs, got_targets = mod.batch_from_start_tensor(
        tokens=tokens,
        starts=starts,
        seq_len=6,
        device=torch.device("cpu"),
    )
    ref_inputs, ref_targets = batch_from_starts(
        tokens=tokens,
        starts=[int(x) for x in starts.tolist()],
        seq_len=6,
        device=torch.device("cpu"),
    )

    assert torch.equal(got_inputs, ref_inputs)
    assert torch.equal(got_targets, ref_targets)
    assert got_inputs.dtype == torch.int32
    assert got_targets.dtype == torch.long


def test_vectorized_batch_can_clamp_header_contamination():
    mod = _load_module()
    tokens = torch.tensor([-7, 1, 2, 999, 4, 5, 6, 7], dtype=torch.int16)
    starts = torch.tensor([0, 3], dtype=torch.long)

    got_inputs, got_targets = mod.batch_from_start_tensor(
        tokens=tokens,
        starts=starts,
        seq_len=3,
        device=torch.device("cpu"),
        vocab_size=8,
    )

    assert got_inputs.tolist() == [[0, 1, 2], [7, 4, 5]]
    assert got_targets.tolist() == [[1, 2, 7], [4, 5, 6]]


def test_lazy_lm_start_sampling_matches_sharded_range():
    mod = _load_module()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)

    starts = mod.sample_sharded_lm_starts(
        num_tokens=10_000,
        seq_len=32,
        stride=16,
        batch_size=128,
        rank=2,
        world_size=4,
        generator=generator,
    )

    total_starts = len(range(0, 10_000 - 32 - 1, 16))
    valid_rank_starts = {
        global_idx * 16
        for global_idx in range(total_starts)
        if global_idx % 4 == 2
    }
    assert starts.shape == (128,)
    assert starts.dtype == torch.long
    assert all(int(start) in valid_rank_starts for start in starts.tolist())


def test_sequential_epoch_start_sampling_covers_shards_with_fixed_batches():
    mod = _load_module()
    num_tokens = 34
    seq_len = 3
    stride = 3
    batch_size = 2
    world_size = 3

    total_starts = mod.count_lm_starts(num_tokens, seq_len, stride)
    epoch_steps = mod.sequential_epoch_steps(
        num_tokens=num_tokens,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        world_size=world_size,
    )

    seen_by_rank: dict[int, list[int]] = {}
    for rank in range(world_size):
        rank_seen: list[int] = []
        for step in range(epoch_steps):
            starts = mod.sequential_sharded_lm_starts(
                num_tokens=num_tokens,
                seq_len=seq_len,
                stride=stride,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                step=step,
            )
            assert starts.shape == (batch_size,)
            assert starts.dtype == torch.long
            rank_seen.extend(int(start // stride) for start in starts.tolist())
        seen_by_rank[rank] = rank_seen

    covered = sorted({idx for values in seen_by_rank.values() for idx in values})
    assert covered == list(range(total_starts))

    for rank, values in seen_by_rank.items():
        sharded_count = mod.count_sharded_lm_starts(
            total_starts=total_starts,
            rank=rank,
            world_size=world_size,
        )
        expected_without_padding = [
            rank + local_idx * world_size
            for local_idx in range(sharded_count)
        ]
        assert values[:sharded_count] == expected_without_padding
        assert values[sharded_count:] == [expected_without_padding[-1]] * (
            len(values) - sharded_count
        )


def test_shuffled_epoch_sampling_covers_each_rank_without_materializing():
    mod = _load_module()
    num_tokens = 34
    seq_len = 3
    stride = 3
    batch_size = 2
    world_size = 3

    total_starts = mod.count_lm_starts(num_tokens, seq_len, stride)
    epoch_steps = mod.sequential_epoch_steps(
        num_tokens=num_tokens,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        world_size=world_size,
    )

    covered = []
    shuffled_order = []
    sequential_order = []
    for rank in range(world_size):
        for step in range(epoch_steps):
            starts = mod.shuffled_epoch_sharded_lm_starts(
                num_tokens=num_tokens,
                seq_len=seq_len,
                stride=stride,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                step=step,
                seed=1337,
            )
            seq_starts = mod.sequential_sharded_lm_starts(
                num_tokens=num_tokens,
                seq_len=seq_len,
                stride=stride,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                step=step,
            )
            covered.extend(int(start // stride) for start in starts.tolist())
            shuffled_order.extend(int(start // stride) for start in starts.tolist())
            sequential_order.extend(int(start // stride) for start in seq_starts.tolist())

    assert sorted(set(covered)) == list(range(total_starts))
    assert shuffled_order != sequential_order


def test_lazy_eval_start_selection_matches_eager_shape():
    mod = _load_module()

    starts = mod.choose_lm_starts_lazy(
        num_tokens=10_000,
        seq_len=32,
        stride=16,
        batch_size=7,
        eval_batches=3,
        seed=123,
    )

    assert len(starts) == 21
    assert len(set(starts)) == 21
    assert all(start % 16 == 0 for start in starts)
    assert all(0 <= start < 10_000 - 32 - 1 for start in starts)


def test_stage_a_matrix_names_and_fast_defaults():
    mod = _load_module()
    entries = mod.build_stage_a_matrix(
        seeds=[1337],
        vocab_sizes=[16384],
        batch_sizes=[1024, 2048],
        chunk_sizes=[64, 256],
        activation_checkpoints=[True, False],
        world_size=8,
    )

    assert [entry["name"] for entry in entries] == [
        "stageA_v16384_b1024_c64_ckpt_s1337",
        "stageA_v16384_b1024_c64_nockpt_s1337",
        "stageA_v16384_b1024_c256_ckpt_s1337",
        "stageA_v16384_b1024_c256_nockpt_s1337",
        "stageA_v16384_b2048_c64_ckpt_s1337",
        "stageA_v16384_b2048_c64_nockpt_s1337",
        "stageA_v16384_b2048_c256_ckpt_s1337",
        "stageA_v16384_b2048_c256_nockpt_s1337",
    ]
    for entry in entries:
        assert entry["mode"] == "speed_sweep"
        assert entry["world_size"] == 8
        assert entry["model_type"] == "ssm"
        assert entry["precision"] == "bf16"
        assert entry["fused_grad_clip"] is True
        assert entry["fused_muon"] is True
        assert entry["compile_full_path"] is False
        assert entry["cuda_graph_mode"] == "none"
        assert entry["cuda_graph_min_total_speedup"] == 0.05
        assert entry["cuda_graph_max_capture_seconds"] == 30.0
        assert entry["lm_head_backward_mode"] == "fused_streaming_v2"
        assert entry["lm_head_tile_size"] == 8192
        assert entry["prefetch_batches"] is True
        assert entry["eval_batches"] == 0
        assert entry["warmup_steps"] == 5
        assert entry["stop_check_interval"] == 4


def test_stage_b_matrix_crosses_vocab_and_embedding_inits():
    mod = _load_module()
    speed_cfg = {
        "batch_size": 2048,
        "chunk_size": 256,
        "activation_checkpoint": True,
        "base_lr": 0.2,
        "eval_batches": 0,
    }
    init_paths = {
        8192: {
            "meanstd": "artifacts/v8192_meanstd.pt",
            "fullcov": "artifacts/v8192_fullcov.pt",
        },
        16384: {
            "meanstd": "artifacts/v16384_meanstd.pt",
            "fullcov": "artifacts/v16384_fullcov.pt",
        },
    }

    entries = mod.build_stage_b_matrix(
        speed_config=speed_cfg,
        seeds=[1337, 2674],
        vocab_sizes=[8192, 16384],
        init_paths=init_paths,
        world_size=8,
    )

    assert len(entries) == 12  # 2 vocabs x 3 init arms x 2 seeds
    names = [entry["name"] for entry in entries]
    assert "stageB_v8192_random_s1337" in names
    assert "stageB_v16384_meanstd_s2674" in names
    assert "stageB_v16384_fullcov_s1337" in names

    random_entries = [entry for entry in entries if entry["embed_init"] == "random"]
    assert random_entries
    assert all(entry.get("embed_init_path") is None for entry in random_entries)

    meanstd_16384 = next(
        entry for entry in entries
        if entry["vocab_size"] == 16384
        and entry["embed_init"] == "meanstd"
        and entry["seed"] == 2674
    )
    assert meanstd_16384["embed_init_path"] == "artifacts/v16384_meanstd.pt"
    assert meanstd_16384["batch_size"] == 2048
    assert meanstd_16384["chunk_size"] == 256
    assert meanstd_16384["activation_checkpoint"] is True
    assert meanstd_16384["prefetch_batches"] is True
    assert meanstd_16384["eval_batches"] == 16
    assert meanstd_16384["budget_seconds"] == 600.0


def test_token_accounting_summary_uses_global_tokens():
    mod = _load_module()
    summary = mod.summarize_train_timing(
        steps=10,
        elapsed_s=2.0,
        batch_size=4,
        seq_len=8,
        world_size=3,
    )

    assert summary["tokens_per_step"] == 96
    assert summary["aggregate_tokens_per_sec"] == 480.0
    assert summary["per_gpu_tokens_per_sec"] == 160.0


def test_token_accounting_summary_subtracts_episodic_rank_when_enabled():
    """Under ``episodic_enabled=True`` the throughput math must divide by
    ``world_size - 1`` (the train-rank count), not ``world_size``. Rank
    ``world_size - 1`` skips the main forward+backward (Task 1.3 skip-main
    flow); only the train ranks consume tokens.

    With world_size=4, batch=4, seq=8: legacy math reports
    ``4 * 8 * 4 = 128`` tokens/step (33% inflation). Episodic-aware math
    reports ``4 * 8 * 3 = 96``. ``per_gpu_tokens_per_sec`` must use the
    same train-rank denominator so per-rank throughput stays internally
    consistent with aggregate.
    """
    mod = _load_module()
    summary = mod.summarize_train_timing(
        steps=10,
        elapsed_s=2.0,
        batch_size=4,
        seq_len=8,
        world_size=4,
        episodic_enabled=True,
    )

    # 3 train ranks consume tokens: 4 * 8 * 3 = 96 per step.
    assert summary["tokens_per_step"] == 96
    # 96 tokens/step * 10 steps / 2.0 s = 480.0 tok/s aggregate.
    assert summary["aggregate_tokens_per_sec"] == 480.0
    # 480 / 3 train ranks = 160 tok/s per train rank.
    assert summary["per_gpu_tokens_per_sec"] == 160.0


def test_token_accounting_default_matches_pre_episodic_behavior():
    """Back-compat invariant: omitting ``episodic_enabled`` (or passing
    ``False``) must produce identical numbers to the pre-Task-1.3 math.
    Every existing exp23/exp24 cell relies on this — they default to
    ``episodic_enabled=False`` and would silently shift if the
    keyword's default behavior changed.
    """
    mod = _load_module()
    legacy = mod.summarize_train_timing(
        steps=10, elapsed_s=2.0, batch_size=4, seq_len=8, world_size=3,
    )
    explicit_off = mod.summarize_train_timing(
        steps=10, elapsed_s=2.0, batch_size=4, seq_len=8, world_size=3,
        episodic_enabled=False,
    )

    assert legacy == explicit_off


def test_cuda_graph_gate_counts_capture_against_budget():
    mod = _load_module()

    accepted = mod.summarize_cuda_graph_gate(
        budget_seconds=600.0,
        capture_seconds=10.0,
        warmup_seconds=5.0,
        warmup_steps=3,
        eager_step_seconds=0.25,
        graph_step_seconds=0.20,
        min_total_speedup=0.05,
        max_capture_seconds=30.0,
    )
    rejected_slow_capture = mod.summarize_cuda_graph_gate(
        budget_seconds=600.0,
        capture_seconds=31.0,
        warmup_seconds=0.0,
        warmup_steps=0,
        eager_step_seconds=0.25,
        graph_step_seconds=0.10,
        min_total_speedup=0.05,
        max_capture_seconds=30.0,
    )
    rejected_small_win = mod.summarize_cuda_graph_gate(
        budget_seconds=600.0,
        capture_seconds=10.0,
        warmup_seconds=5.0,
        warmup_steps=3,
        eager_step_seconds=0.25,
        graph_step_seconds=0.245,
        min_total_speedup=0.05,
        max_capture_seconds=30.0,
    )

    assert accepted["accepted"] is True
    assert accepted["overhead_seconds"] == 15.0
    assert accepted["projected_eager_steps"] == 2400.0
    assert accepted["projected_graph_steps"] == 2925.0
    assert accepted["projected_total_speedup"] > 0.05
    assert abs(accepted["break_even_steps"] - 300.0) < 1e-9
    assert abs(accepted["break_even_seconds"] - 75.0) < 1e-9

    assert rejected_slow_capture["accepted"] is False
    assert "capture_seconds_exceeds_limit" in rejected_slow_capture["reasons"]
    assert rejected_small_win["accepted"] is False
    assert "projected_speedup_below_minimum" in rejected_small_win["reasons"]


def test_steady_state_step_seconds_drops_first_step_spike():
    mod = _load_module()

    # Typical H100 warmup shape: first step is ~3× slower than steady
    # state due to cuBLAS algo selection and allocator growth. The
    # steady-state estimator must not let that first sample drag the
    # projection upward.
    samples = [0.62, 0.21, 0.20, 0.21]
    assert mod.steady_state_step_seconds(samples) == 0.21

    # With exactly two samples the drop-first-then-median reduces to
    # the single trailing sample — that is the right call because a
    # single warm sample is still more honest than the first-call spike.
    assert mod.steady_state_step_seconds([0.50, 0.18]) == 0.18


def test_steady_state_step_seconds_handles_single_sample():
    mod = _load_module()

    # Only one warmup step gives the caller nothing to discard. Return
    # the sample as-is rather than raising; the call site (CUDA graph
    # gate) still applies its own conservative checks downstream.
    assert mod.steady_state_step_seconds([0.25]) == 0.25


def test_steady_state_step_seconds_rejects_empty():
    mod = _load_module()

    with pytest.raises(ValueError):
        mod.steady_state_step_seconds([])


def test_cuda_graph_gate_rejects_when_steady_state_baseline_is_honest():
    """Regression for the 2026-04-21 native-scan probe.

    With the legacy ``warmup_seconds / warmup_steps`` baseline, the gate
    saw the probe's ``eager_step_seconds = 0.212`` (inflated by a single
    first-call spike in a 3-step warmup) vs ``graph_step_seconds = 0.174``
    and accepted the graph. The true steady-state eager step on the same
    pod was ``~0.170`` — graphs were actually slower than a warm eager
    step. Feeding the median-based baseline into the gate flips the
    decision to the right one.
    """
    mod = _load_module()

    # Four samples: one first-call spike plus three steady-state.
    # Dropping the first leaves [0.170, 0.171, 0.172]; median is 0.171.
    per_step_warmup = [0.636, 0.170, 0.171, 0.172]
    steady_state_eager = mod.steady_state_step_seconds(per_step_warmup)

    assert steady_state_eager == 0.171

    legacy_mean = sum(per_step_warmup) / len(per_step_warmup)
    legacy_accepted = mod.summarize_cuda_graph_gate(
        budget_seconds=20.0,
        capture_seconds=0.15,
        warmup_seconds=sum(per_step_warmup),
        warmup_steps=len(per_step_warmup),
        eager_step_seconds=legacy_mean,
        graph_step_seconds=0.1739,
        min_total_speedup=0.05,
        max_capture_seconds=30.0,
    )
    honest_rejected = mod.summarize_cuda_graph_gate(
        budget_seconds=20.0,
        capture_seconds=0.15,
        warmup_seconds=sum(per_step_warmup),
        warmup_steps=len(per_step_warmup),
        eager_step_seconds=steady_state_eager,
        graph_step_seconds=0.1739,
        min_total_speedup=0.05,
        max_capture_seconds=30.0,
    )

    assert legacy_accepted["accepted"] is True, (
        "legacy baseline is the bug under test; if this starts rejecting, "
        "the probe conditions or the gate math changed and the regression "
        "scenario needs rebuilding"
    )
    assert honest_rejected["accepted"] is False
    assert "graph_step_not_faster" in honest_rejected["reasons"]


def test_cuda_graph_probe_rejects_cpu_and_runs_eager() -> None:
    mod = _load_runner_module()
    torch.manual_seed(5)
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_tokens = torch.arange(128, dtype=torch.int16)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=train_tokens,
        train_num_tokens=int(train_tokens.numel()),
        stride=4,
        seq_len=8,
        batch_size=4,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=30.0,
        chunk_size=4,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="fp32",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=False,
        lm_head_backward_mode="fused",
        cuda_graph_mode="probe",
    )

    assert result["steps"] == 2
    assert result["cuda_graph"]["accepted"] is False
    assert "cuda_required" in result["cuda_graph"]["reasons"]


def test_cuda_graph_probe_rejects_activation_checkpoint_before_capture(
    monkeypatch,
) -> None:
    mod = _load_runner_module()
    torch.manual_seed(7)
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_tokens = torch.arange(128, dtype=torch.int16)

    def unexpected_graph_runner(**_kwargs):
        raise AssertionError("activation checkpoint should reject graph capture")

    monkeypatch.setattr(mod, "_train_fast_for_budget_cuda_graph", unexpected_graph_runner)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=train_tokens,
        train_num_tokens=int(train_tokens.numel()),
        stride=4,
        seq_len=8,
        batch_size=4,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=30.0,
        chunk_size=4,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="fp32",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        activation_checkpoint=True,
        prefetch_batches=False,
        lm_head_backward_mode="fused",
        cuda_graph_mode="probe",
    )

    assert result["steps"] == 2
    assert result["cuda_graph"]["accepted"] is False
    assert "activation_checkpoint_not_supported" in result["cuda_graph"]["reasons"]


def test_cuda_graph_probe_delegates_when_eligible(monkeypatch) -> None:
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_tokens = torch.arange(128, dtype=torch.int16)
    calls: list[dict] = []

    monkeypatch.setattr(mod, "_cuda_graph_rejection_reasons", lambda **_: [])

    def fake_graph_runner(**kwargs):
        calls.append(kwargs)
        return {
            "steps": 7,
            "elapsed_s": 1.0,
            "rank": kwargs["rank"],
            "world_size": kwargs["world_size"],
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "loss_delta": -0.5,
            "peak_vram_mb": 0.0,
            "tokens_per_step": 32,
            "aggregate_tokens_per_sec": 224.0,
            "per_gpu_tokens_per_sec": 224.0,
            "cuda_graph": {"mode": "probe", "accepted": True, "reasons": []},
        }

    monkeypatch.setattr(mod, "_train_fast_for_budget_cuda_graph", fake_graph_runner)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=train_tokens,
        train_num_tokens=int(train_tokens.numel()),
        stride=4,
        seq_len=8,
        batch_size=4,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=30.0,
        chunk_size=4,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="fp32",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=False,
        lm_head_backward_mode="fused",
        cuda_graph_mode="probe",
        cuda_graph_warmup_steps=4,
    )

    assert result["steps"] == 7
    assert calls
    assert calls[0]["cuda_graph_warmup_steps"] == 4
    assert calls[0]["cuda_graph_mode"] == "probe"


def test_cuda_graph_probe_allows_compiled_encoder() -> None:
    mod = _load_runner_module()

    reasons = mod._cuda_graph_rejection_reasons(
        device=torch.device("cuda"),
        ddp_active=False,
        activation_checkpoint=False,
        compile_full_path=True,
        lm_head_backward_mode="fused_streaming_cached",
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=0.1),
    )

    assert "compile_full_path_not_supported" not in reasons


def test_cuda_graph_probe_falls_back_to_eager_on_capture_failure(monkeypatch) -> None:
    mod = _load_runner_module()
    torch.manual_seed(6)
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_tokens = torch.arange(128, dtype=torch.int16)

    monkeypatch.setattr(mod, "_cuda_graph_rejection_reasons", lambda **_: [])

    def broken_graph_runner(**_kwargs):
        raise RuntimeError("capture failed in smoke")

    monkeypatch.setattr(mod, "_train_fast_for_budget_cuda_graph", broken_graph_runner)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=train_tokens,
        train_num_tokens=int(train_tokens.numel()),
        stride=4,
        seq_len=8,
        batch_size=4,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=30.0,
        chunk_size=4,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="fp32",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=False,
        lm_head_backward_mode="fused",
        cuda_graph_mode="probe",
    )

    assert result["steps"] == 2
    assert result["cuda_graph"]["accepted"] is False
    assert "capture_failed" in result["cuda_graph"]["reasons"]
    assert "capture failed in smoke" in result["cuda_graph"]["error"]


def test_run_train_step_uses_compiled_encoder_when_enabled(monkeypatch):
    mod = _load_runner_module()
    if hasattr(train_ssm._compiled_step_fn, "cache_clear"):
        train_ssm._compiled_step_fn.cache_clear()

    compile_calls: list[tuple[str, bool, bool]] = []

    def fake_compile(fn, *, fullgraph, dynamic):
        compile_calls.append((fn.__name__, fullgraph, dynamic))

        def compiled(model, inputs):
            compile_calls.append(("compiled", True, True))
            return fn(model, inputs)

        return compiled

    monkeypatch.setattr(mod.torch, "compile", fake_compile)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    try:
        loss = mod._run_train_step(
            model=model,
            inputs=inputs,
            targets=targets,
            chunk_size=2,
            precision="bf16",
            ddp_active=False,
            world_size=1,
            compile_full_path=True,
        )

        assert loss.ndim == 0
        assert compile_calls == [
            ("_encode_only", True, False),
            ("compiled", True, True),
        ]
    finally:
        if hasattr(train_ssm._compiled_step_fn, "cache_clear"):
            train_ssm._compiled_step_fn.cache_clear()


def test_run_train_step_uses_compiled_packet_encoder_for_crct_payload(monkeypatch):
    mod = _load_runner_module()
    for fn in (train_ssm._compiled_step_fn, train_ssm._compiled_packet_step_fn):
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()

    compile_calls: list[tuple[str, bool, bool]] = []

    def fake_compile(fn, *, fullgraph, dynamic):
        compile_calls.append((fn.__name__, fullgraph, dynamic))

        def compiled(*args):
            compile_calls.append(("compiled", True, True))
            return fn(*args)

        return compiled

    monkeypatch.setattr(mod.torch, "compile", fake_compile)

    model = _TinyPacketTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)
    payload = {
        "memory_residual": torch.randn(2, 1, 4),
        "memory_gate": torch.ones(2, 3),
        "loss_weight": torch.ones(2, 3),
    }

    try:
        loss = mod._run_train_step(
            model=model,
            inputs=inputs,
            targets=targets,
            chunk_size=2,
            precision="bf16",
            ddp_active=False,
            world_size=1,
            compile_full_path=True,
            lm_head_backward_mode="single",
            crct_enabled=True,
            crct_payload=payload,
        )

        assert loss.ndim == 0
        assert compile_calls == [
            ("_encode_packet_only", True, False),
            ("compiled", True, True),
        ]
    finally:
        for fn in (train_ssm._compiled_step_fn, train_ssm._compiled_packet_step_fn):
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()


def test_run_train_step_leaves_encoder_eager_when_compile_disabled(monkeypatch):
    mod = _load_runner_module()

    compile_calls: list[tuple[str, bool, bool]] = []

    def fake_compile(fn, *, fullgraph, dynamic):
        compile_calls.append((fn.__name__, fullgraph, dynamic))
        return fn

    monkeypatch.setattr(mod.torch, "compile", fake_compile)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        compile_full_path=False,
    )

    assert loss.ndim == 0
    assert compile_calls == []


def test_run_train_step_can_use_single_backward(monkeypatch):
    mod = _load_runner_module()

    chunked_calls = 0

    def fail_if_chunked(**kwargs):
        nonlocal chunked_calls
        chunked_calls += 1
        raise AssertionError("single-backward mode must not call chunked CE")

    monkeypatch.setattr(mod, "chunked_lm_head_backward", fail_if_chunked)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        compile_full_path=False,
        lm_head_backward_mode="single",
    )

    assert loss.ndim == 0
    assert chunked_calls == 0
    assert model.encoder.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_run_train_step_can_use_fused_backward(monkeypatch):
    mod = _load_runner_module()

    calls: list[str | None] = []

    def fake_fused(**kwargs):
        calls.append(kwargs.get("backend"))
        hidden = kwargs["hidden"]
        loss = hidden.float().pow(2).mean()
        loss.backward()
        return loss.detach()

    monkeypatch.setattr(mod, "fused_lm_head_backward", fake_fused)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        compile_full_path=False,
        lm_head_backward_mode="fused",
    )

    assert loss.ndim == 0
    assert calls == ["auto"]
    assert model.encoder.weight.grad is not None


def test_run_train_step_can_use_streaming_fused_backward(monkeypatch):
    mod = _load_runner_module()

    calls: list[str | None] = []

    def fake_fused(**kwargs):
        calls.append(kwargs.get("backend"))
        hidden = kwargs["hidden"]
        loss = hidden.float().pow(2).mean()
        loss.backward()
        return loss.detach()

    monkeypatch.setattr(mod, "fused_lm_head_backward", fake_fused)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        compile_full_path=False,
        lm_head_backward_mode="fused_streaming",
    )

    assert loss.ndim == 0
    assert calls == ["streaming"]
    assert model.encoder.weight.grad is not None


def test_run_train_step_can_use_streaming_v2_fused_backward(monkeypatch):
    mod = _load_runner_module()

    calls: list[tuple[str | None, int]] = []

    def fake_fused(**kwargs):
        calls.append((kwargs.get("backend"), kwargs["tile_size"]))
        hidden = kwargs["hidden"]
        loss = hidden.float().pow(2).mean()
        loss.backward()
        return loss.detach()

    monkeypatch.setattr(mod, "fused_lm_head_backward", fake_fused)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        compile_full_path=False,
        lm_head_backward_mode="fused_streaming_v2",
        lm_head_tile_size=4096,
    )

    assert loss.ndim == 0
    assert calls == [("streaming_v2", 4096)]
    assert model.encoder.weight.grad is not None


def test_run_train_step_can_use_streaming_cached_fused_backward(monkeypatch):
    mod = _load_runner_module()

    calls: list[tuple[str | None, int]] = []

    def fake_fused(**kwargs):
        calls.append((kwargs.get("backend"), kwargs["tile_size"]))
        hidden = kwargs["hidden"]
        loss = hidden.float().pow(2).mean()
        loss.backward()
        return loss.detach()

    monkeypatch.setattr(mod, "fused_lm_head_backward", fake_fused)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        compile_full_path=False,
        lm_head_backward_mode="fused_streaming_cached",
        lm_head_tile_size=8192,
    )

    assert loss.ndim == 0
    assert calls == [("streaming_cached", 8192)]
    assert model.encoder.weight.grad is not None


def test_run_train_step_can_use_norm_streaming_v2_fused_backward(monkeypatch):
    mod = _load_runner_module()

    calls: list[tuple[str | None, int]] = []

    def fake_fused(**kwargs):
        calls.append((kwargs.get("backend"), kwargs["tile_size"]))
        hidden = kwargs["hidden"]
        loss = hidden.float().pow(2).mean()
        loss.backward()
        return loss.detach()

    monkeypatch.setattr(mod, "fused_lm_head_backward", fake_fused)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        compile_full_path=False,
        lm_head_backward_mode="fused_norm_streaming_v2",
        lm_head_tile_size=8192,
    )

    assert loss.ndim == 0
    assert calls == [("norm_streaming_v2", 8192)]
    assert model.encoder.weight.grad is not None


def test_run_train_step_applies_predictive_auxiliary(monkeypatch):
    mod = _load_runner_module()
    calls = []

    def fake_aux(hidden, *, projection, horizon):
        calls.append((tuple(hidden.shape), horizon, projection is not None))
        return hidden.float().mean() * 0.0 + 0.5

    monkeypatch.setattr(mod, "predictive_auxiliary_loss", fake_aux)
    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)
    aux = nn.Linear(4, 4, bias=False)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        predictive_aux_weight=0.1,
        predictive_aux_horizon=1,
        predictive_aux_projection=aux,
    )

    assert loss.ndim == 0
    assert calls == [((2, 3, 4), 1, True)]


def test_train_fast_for_budget_syncs_predictive_aux_projection_in_ddp(monkeypatch):
    mod = _load_runner_module()
    broadcasts = []
    allreduces = []

    monkeypatch.setattr(mod, "broadcast_params", lambda module: broadcasts.append(type(module).__name__))
    monkeypatch.setattr(
        mod,
        "allreduce_grads",
        lambda module, world_size: allreduces.append((type(module).__name__, world_size)),
    )
    monkeypatch.setattr(mod, "should_stop_now", lambda local, *_args, **_kwargs: local)
    monkeypatch.setattr(mod.dist, "barrier", lambda: None)

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=2,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=1,
        prefetch_batches=False,
        predictive_aux_weight=0.1,
        predictive_aux_horizon=1,
    )

    assert result["steps"] == 1
    assert broadcasts == ["_TinyTokenTrainModel", "Linear"]
    assert ("Linear", 2) in allreduces
    assert result["mechanisms"]["predictive_aux"]["enabled"] is True


def test_train_fast_for_budget_can_use_batch_prefetcher(monkeypatch):
    mod = _load_runner_module()
    instances = []

    class FakePrefetcher:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.prime_calls = 0
            self.next_calls = 0
            self.close_calls = 0
            instances.append(self)

        def prime(self):
            self.prime_calls += 1
            return self

        def next(self):
            self.next_calls += 1
            inputs = torch.zeros(2, 3, dtype=torch.long)
            targets = torch.zeros(2, 3, dtype=torch.long)
            return inputs, targets

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(mod, "Exp23BatchPrefetcher", FakePrefetcher)

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16),
        train_num_tokens=128,
        stride=4,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=True,
    )

    assert result["steps"] == 2
    assert len(instances) == 1
    prefetcher = instances[0]
    assert prefetcher.prime_calls == 1
    assert prefetcher.next_calls == 2
    assert prefetcher.close_calls == 1
    assert prefetcher.kwargs["seq_len"] == 3
    assert prefetcher.kwargs["batch_size"] == 2
    assert prefetcher.kwargs["vocab_size"] == 6


def test_train_fast_for_budget_serializes_loss_trajectory():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="fp32",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=3,
        prefetch_batches=False,
    )

    assert len(result["loss_trajectory"]) == result["steps"] == 3
    assert result["loss_trajectory"][0] == pytest.approx(result["initial_loss"])
    assert result["loss_trajectory"][-1] == pytest.approx(result["final_loss"])


def test_train_fast_for_budget_sequential_epoch_stops_after_epoch():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_tokens = torch.arange(25, dtype=torch.int16) % 6

    result = mod.train_fast_for_budget(
        model,
        train_tokens=train_tokens,
        train_num_tokens=int(train_tokens.numel()),
        stride=3,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=None,
        prefetch_batches=True,
        train_sampling_mode="sequential_epoch",
    )

    assert result["steps"] == 4
    assert result["sampling_mode"] == "sequential_epoch"
    assert result["epoch_steps"] == 4
    assert result["unique_start_count"] == 7
    assert result["epoch_complete"] is True


def test_train_fast_for_budget_accepts_shuffled_epoch_sampling():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(25, dtype=torch.int16) % 6,
        train_num_tokens=25,
        stride=3,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=None,
        prefetch_batches=True,
        train_sampling_mode="shuffled_epoch",
    )

    assert result["steps"] == 4
    assert result["sampling_mode"] == "shuffled_epoch"
    assert result["epoch_steps"] == 4
    assert result["unique_start_count"] == 7
    assert result["epoch_complete"] is True


def test_train_fast_for_budget_applies_spectral_extra_loss(monkeypatch):
    mod = _load_runner_module()
    calls = []

    def fake_spectral_loss(model, **kwargs):
        calls.append(kwargs)
        return torch.tensor(0.25, requires_grad=True)

    monkeypatch.setattr(mod, "spectral_regularization_loss", fake_spectral_loss)
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=1,
        prefetch_batches=False,
        spectral_reg_lambda_dead=0.1,
        spectral_reg_lambda_sticky=0.2,
    )

    assert calls == [
        {
            "lambda_dead": 0.1,
            "lambda_sticky": 0.2,
            "min_a": 0.05,
            "max_a": 0.98,
        }
    ]
    assert result["mechanisms"]["spectral"]["enabled"] is True


def test_train_fast_for_budget_zeroes_embed_grad_during_freeze(monkeypatch):
    mod = _load_runner_module()
    zero_calls = []

    def fake_zero(model, *, step, freeze_steps):
        zero_calls.append((step, freeze_steps))

    monkeypatch.setattr(mod, "zero_embedding_grad_until", fake_zero)
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=False,
        embed_freeze_steps=2,
    )

    assert zero_calls == [(0, 2), (1, 2)]


def test_train_fast_for_budget_runs_dreamworld_replay(monkeypatch):
    mod = _load_runner_module()
    events = []

    class FakeDreamBuffer:
        def __init__(self, **kwargs):
            self.entries = []
            events.append(("buffer", kwargs))

        def __len__(self):
            return len(self.entries)

        def add(self, *, step, states, replay_tokens):
            self.entries.append((step, states, replay_tokens))
            events.append(("add", step))

        def sample(self, *, generator, current_step):
            events.append(("sample", current_step))
            return object()

        def diagnostics(self, *, current_step):
            return {"size": len(self.entries), "current_step": current_step}

    monkeypatch.setattr(mod, "DreamReplayBuffer", FakeDreamBuffer)
    monkeypatch.setattr(
        mod,
        "capture_dream_entry",
        lambda model, inputs, **kwargs: type(
            "E",
            (),
            {
                "step": kwargs["step"],
                "states": [torch.zeros(inputs.size(0), 4)],
                "replay_tokens": inputs[:, :3],
            },
        )(),
    )
    monkeypatch.setattr(
        mod,
        "dreamworld_replay_backward",
        lambda model, *, entry, weight, **kwargs: torch.tensor(0.1),
    )

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=6,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=False,
        dreamworld_enabled=True,
        dreamworld_cache_interval=1,
        dreamworld_interval=1,
        dreamworld_weight=0.25,
        dreamworld_prefix_tokens=3,
        dreamworld_replay_tokens=2,
        dreamworld_min_size=1,
    )

    assert ("sample", 0) not in events
    assert ("sample", 1) in events
    assert result["mechanisms"]["dreamworld"]["enabled"] is True
    assert result["mechanisms"]["dreamworld"]["artifact_impact"] == "artifact_training_only"


def test_train_fast_for_budget_can_use_async_param_allreduce(monkeypatch):
    mod = _load_runner_module()

    events: list[str] = []

    class FakeAsyncReducer:
        def __init__(self, model, world_size):
            self.model = model
            self.world_size = world_size
            events.append(f"init:{world_size}")

        def reset(self):
            events.append("reset")

        def wait(self):
            events.append("wait")

        def close(self):
            events.append("close")

    def fail_bulk_allreduce(*_args, **_kwargs):
        raise AssertionError("async_param mode must not call bulk allreduce")

    monkeypatch.setattr(mod, "broadcast_params", lambda _model: None)
    monkeypatch.setattr(mod, "should_stop_now", lambda local, *_args, **_kwargs: local)
    monkeypatch.setattr(mod.dist, "barrier", lambda: None)
    monkeypatch.setattr(mod, "allreduce_grads", fail_bulk_allreduce)
    monkeypatch.setattr(mod, "AsyncGradAllReducer", FakeAsyncReducer)

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16),
        train_num_tokens=128,
        stride=4,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=2,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=1,
        prefetch_batches=False,
        grad_allreduce_mode="async_param",
    )

    assert result["steps"] == 1
    assert events == ["init:2", "reset", "wait", "close"]


def test_training_stop_predicate_can_stop_on_fixed_warmup_steps():
    mod = _load_module()

    assert not mod.should_stop_training_loop(
        steps=0,
        elapsed_s=999.0,
        budget_seconds=300.0,
        stop_margin_seconds=0.0,
        max_steps=5,
    )
    assert not mod.should_stop_training_loop(
        steps=4,
        elapsed_s=0.0,
        budget_seconds=300.0,
        stop_margin_seconds=0.0,
        max_steps=5,
    )
    assert mod.should_stop_training_loop(
        steps=5,
        elapsed_s=0.0,
        budget_seconds=300.0,
        stop_margin_seconds=0.0,
        max_steps=5,
    )
    assert mod.should_stop_training_loop(
        steps=1,
        elapsed_s=298.0,
        budget_seconds=300.0,
        stop_margin_seconds=2.0,
        max_steps=None,
    )


def test_read_speed_config_accepts_yaml(tmp_path):
    mod = _load_module()
    cfg = tmp_path / "speed.yaml"
    cfg.write_text("batch_size: 2048\nchunk_size: 256\n")

    assert mod.read_speed_config(cfg) == {
        "batch_size": 2048,
        "chunk_size": 256,
    }


def test_torchrun_command_uses_all_requested_gpus(tmp_path):
    mod = _load_launch_module()
    cfg = tmp_path / "cfg.yaml"
    out = tmp_path / "out.json"
    cmd = mod.build_torchrun_cmd(
        runner_path=Path("experiments/23_fast_path/runner_fast_path.py"),
        config_path=cfg,
        data_path="/data/fineweb",
        sp_model_path="/data/sp16384.model",
        output_json=out,
        world_size=8,
        rdzv_port=23456,
    )

    assert cmd[:3] == [mod.sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=8" in cmd
    assert "--rdzv-endpoint=localhost:23456" in cmd
    assert "--config" in cmd and str(cfg) in cmd
    assert "--output-json" in cmd and str(out) in cmd


def test_torchrun_command_can_pass_val_cache_dir(tmp_path):
    mod = _load_launch_module()
    cfg = tmp_path / "cfg.yaml"
    out = tmp_path / "out.json"
    val_cache = tmp_path / "val-cache"
    cmd = mod.build_torchrun_cmd(
        runner_path=Path("experiments/23_fast_path/runner_fast_path.py"),
        config_path=cfg,
        data_path="/data/fineweb",
        sp_model_path="/data/sp16384.model",
        output_json=out,
        world_size=4,
        rdzv_port=23456,
        val_cache_dir=val_cache,
    )

    assert "--val-cache-dir" in cmd
    assert cmd[cmd.index("--val-cache-dir") + 1] == str(val_cache)


def test_summarize_results_ranks_successes_and_records_errors(tmp_path):
    mod = _load_launch_module()
    results = tmp_path / "results"
    results.mkdir()
    (results / "slow.json").write_text(
        '{"config":{"name":"slow"},"train":{"aggregate_tokens_per_sec":10.0}}'
    )
    (results / "fast.json").write_text(
        '{"config":{"name":"fast"},"train":{"aggregate_tokens_per_sec":20.0},'
        '"artifact":{"artifact_impact":"artifact_training_only","submit_valid":false},'
        '"exp24":{"phase":"first_wave","mechanism":"dreamworld"}}'
    )
    (results / "bad.json").write_text(
        '{"config":{"name":"bad"},"error":"oom"}'
    )
    (results / "matrix.json").write_text('[{"name":"not-a-result"}]')

    summary = mod.summarize_result_dir(results)

    assert [row["name"] for row in summary["ranked"]] == ["fast", "slow"]
    assert summary["ranked"][0]["artifact_impact"] == "artifact_training_only"
    assert summary["ranked"][0]["submit_valid"] is False
    assert summary["ranked"][0]["exp24_phase"] == "first_wave"
    assert summary["ranked"][0]["exp24_mechanism"] == "dreamworld"
    assert summary["errors"] == [{"name": "bad", "error": "oom"}]


def test_run_condition_result_preserves_exp24_artifact_metadata(monkeypatch):
    mod = _load_runner_module()

    monkeypatch.setattr(mod, "_init_distributed", lambda _world_size: (0, 1, 0))
    monkeypatch.setattr(mod, "_pick_device", lambda _rank, _device: torch.device("cpu"))
    monkeypatch.setattr(mod, "resolve_param_dtype", lambda _dtype, _device: torch.float32)
    monkeypatch.setattr(mod, "verify_diag_recurrence", lambda _device: None)
    monkeypatch.setattr(
        mod,
        "load_fineweb_tokens",
        lambda _path: (
            torch.arange(64, dtype=torch.int16),
            torch.arange(64, dtype=torch.int16),
        ),
    )
    monkeypatch.setattr(mod, "build_sentencepiece_luts", lambda *_args: (None, None, None))
    monkeypatch.setattr(mod, "choose_lm_starts_lazy", lambda **_kwargs: [])
    monkeypatch.setattr(mod, "build_model", lambda *_args: _TinyTokenTrainModel())
    monkeypatch.setattr(mod, "_apply_embed_init", lambda *_args: None)
    monkeypatch.setattr(mod, "_reject_unsupported", lambda _model: None)
    monkeypatch.setattr(mod, "_warmup", lambda **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "_build_optimizer",
        lambda _config, model: torch.optim.SGD(model.parameters(), lr=0.01),
    )
    monkeypatch.setattr(
        mod,
        "train_fast_for_budget",
        lambda *args, **kwargs: {
            "steps": 1,
            "elapsed_s": 1.0,
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "aggregate_tokens_per_sec": 1.0,
            "peak_vram_mb": 0.0,
        },
    )

    class FakeSP:
        def Load(self, _path):
            return True

    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        type("M", (), {"SentencePieceProcessor": FakeSP}),
    )

    result = mod.run_condition(
        {
            "name": "metadata_smoke",
            "vocab_size": 6,
            "seq_len": 3,
            "stride": 1,
            "batch_size": 2,
            "artifact_impact": "artifact_training_only",
            "submit_valid": False,
            "exp24_phase": "first_wave",
            "exp24_mechanism": "predictive_aux",
        },
        data_path="unused",
        sp_model_path="unused.model",
        budget_seconds=1.0,
        output_json=None,
        output_ckpt=None,
        world_size_override=1,
    )

    assert result["artifact"]["artifact_impact"] == "artifact_training_only"
    assert result["artifact"]["submit_valid"] is False
    assert result["artifact"]["artifact_bytes_estimate"] > 0
    assert result["exp24"]["phase"] == "first_wave"
    assert result["exp24"]["mechanism"] == "predictive_aux"


def test_run_condition_threads_max_steps_from_config(monkeypatch):
    mod = _load_runner_module()
    seen_kwargs = {}

    monkeypatch.setattr(mod, "_init_distributed", lambda _world_size: (0, 1, 0))
    monkeypatch.setattr(mod, "_pick_device", lambda _rank, _device: torch.device("cpu"))
    monkeypatch.setattr(mod, "resolve_param_dtype", lambda _dtype, _device: torch.float32)
    monkeypatch.setattr(mod, "verify_diag_recurrence", lambda _device: None)
    monkeypatch.setattr(
        mod,
        "load_fineweb_tokens",
        lambda _path: (
            torch.arange(64, dtype=torch.int16),
            torch.arange(64, dtype=torch.int16),
        ),
    )
    monkeypatch.setattr(mod, "build_sentencepiece_luts", lambda *_args: (None, None, None))
    monkeypatch.setattr(mod, "choose_lm_starts_lazy", lambda **_kwargs: [])
    monkeypatch.setattr(mod, "build_model", lambda *_args: _TinyTokenTrainModel())
    monkeypatch.setattr(mod, "_apply_embed_init", lambda *_args: None)
    monkeypatch.setattr(mod, "_reject_unsupported", lambda _model: None)
    monkeypatch.setattr(mod, "_warmup", lambda **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "_build_optimizer",
        lambda _config, model: torch.optim.SGD(model.parameters(), lr=0.01),
    )

    def fake_train_fast_for_budget(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return {
            "steps": int(kwargs["max_steps"]),
            "elapsed_s": 1.0,
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "aggregate_tokens_per_sec": 1.0,
            "peak_vram_mb": 0.0,
        }

    monkeypatch.setattr(mod, "train_fast_for_budget", fake_train_fast_for_budget)

    class FakeSP:
        def Load(self, _path):
            return True

    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        type("M", (), {"SentencePieceProcessor": FakeSP}),
    )

    result = mod.run_condition(
        {
            "name": "matched_step_smoke",
            "vocab_size": 6,
            "seq_len": 3,
            "stride": 1,
            "batch_size": 2,
            "max_steps": 517,
            "episodic_cuda_write_event_stream_enabled": False,
            "episodic_cuda_write_event_stage_depth": 7,
        },
        data_path="unused",
        sp_model_path="unused.model",
        budget_seconds=999.0,
        output_json=None,
        output_ckpt=None,
        world_size_override=1,
    )

    assert seen_kwargs["max_steps"] == 517
    assert seen_kwargs["episodic_cuda_write_event_stream_enabled"] is False
    assert seen_kwargs["episodic_cuda_write_event_stage_depth"] == 7
    assert result["train"]["steps"] == 517


def test_run_condition_derives_rare_bucket_frequencies_and_uses_val_tokens(monkeypatch):
    """YAML cannot carry a tensor frequency table. The CLI path should derive
    rarity buckets from train-token counts, while the CE readout itself should
    run on val tokens rather than the training stream."""
    mod = _load_runner_module()
    seen_kwargs = {}
    train_tokens = torch.tensor([0, 1, 1, 2, 2, 2, 5], dtype=torch.int16)
    val_tokens = torch.tensor([5, 4, 4, 3, 3, 3], dtype=torch.int16)

    monkeypatch.setattr(mod, "_init_distributed", lambda _world_size: (0, 1, 0))
    monkeypatch.setattr(mod, "_pick_device", lambda _rank, _device: torch.device("cpu"))
    monkeypatch.setattr(mod, "resolve_param_dtype", lambda _dtype, _device: torch.float32)
    monkeypatch.setattr(mod, "verify_diag_recurrence", lambda _device: None)
    monkeypatch.setattr(mod, "load_fineweb_tokens", lambda _path: (train_tokens, val_tokens))
    monkeypatch.setattr(mod, "build_sentencepiece_luts", lambda *_args: (None, None, None))
    monkeypatch.setattr(mod, "choose_lm_starts_lazy", lambda **_kwargs: [])
    monkeypatch.setattr(mod, "build_model", lambda *_args: _TinyTokenTrainModel())
    monkeypatch.setattr(mod, "_apply_embed_init", lambda *_args: None)
    monkeypatch.setattr(mod, "_reject_unsupported", lambda _model: None)
    monkeypatch.setattr(mod, "_warmup", lambda **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "_build_optimizer",
        lambda _config, model: torch.optim.SGD(model.parameters(), lr=0.01),
    )

    def fake_train_fast_for_budget(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return {
            "steps": 1,
            "elapsed_s": 1.0,
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "aggregate_tokens_per_sec": 1.0,
            "peak_vram_mb": 0.0,
        }

    monkeypatch.setattr(mod, "train_fast_for_budget", fake_train_fast_for_budget)

    class FakeSP:
        def Load(self, _path):
            return True

    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        type("M", (), {"SentencePieceProcessor": FakeSP}),
    )

    mod.run_condition(
        {
            "name": "rare_bucket_cli_smoke",
            "vocab_size": 6,
            "seq_len": 3,
            "stride": 1,
            "batch_size": 2,
            "max_steps": 1,
            "rare_bucket_ce_enabled": True,
            "rare_bucket_ce_num_buckets": 4,
        },
        data_path="unused",
        sp_model_path="unused.model",
        budget_seconds=1.0,
        output_json=None,
        output_ckpt=None,
        world_size_override=1,
    )

    expected_freq = torch.tensor([1, 2, 3, 1, 1, 1], dtype=torch.float32)
    assert seen_kwargs["rare_bucket_ce_enabled"] is True
    assert torch.equal(seen_kwargs["rare_bucket_ce_token_frequencies"], expected_freq)
    assert torch.equal(seen_kwargs["rare_bucket_ce_eval_tokens"], val_tokens)
    assert seen_kwargs["rare_bucket_ce_eval_num_tokens"] == int(val_tokens.numel())


def test_train_fast_for_budget_accepts_lm_head_emit_entropy_flag():
    """Orthogonal flag; no mode-name churn. Default False."""
    mod = _load_runner_module()
    # Use the same harness other tests use — construct a minimal model+optimizer.
    import inspect
    sig = inspect.signature(mod.train_fast_for_budget)
    assert "lm_head_emit_entropy" in sig.parameters
    assert sig.parameters["lm_head_emit_entropy"].default is False


def test_train_fast_for_budget_accepts_criticality_distill_enabled_flag():
    mod = _load_runner_module()
    import inspect
    sig = inspect.signature(mod.train_fast_for_budget)
    assert "criticality_distill_enabled" in sig.parameters
    assert sig.parameters["criticality_distill_enabled"].default is False


def test_criticality_distill_enabled_requires_lm_head_emit_entropy():
    """CD enabled without entropy emission -> clear ValueError."""
    import pytest
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match=r"lm_head_emit_entropy"):
        mod.train_fast_for_budget(
            model, train_tokens=torch.arange(32, dtype=torch.int16) % 6,
            train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
            device=torch.device("cpu"), optimizer=optimizer,
            budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
            rank=0, world_size=1, seed=123, precision="fp32",
            stop_check_interval=1, stop_margin_seconds=0.0,
            vocab_size=6, max_steps=1, prefetch_batches=False,
            criticality_distill_enabled=True,
            lm_head_emit_entropy=False,
        )


def test_criticality_distill_disabled_does_not_require_lm_head_emit_entropy():
    """CD off -> entropy flag doesn't matter."""
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # Should not raise — CD off means entropy flag is don't-care.
    result = mod.train_fast_for_budget(
        model, train_tokens=torch.arange(32, dtype=torch.int16) % 6,
        train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
        device=torch.device("cpu"), optimizer=optimizer,
        budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=1, prefetch_batches=False,
        criticality_distill_enabled=False,
        lm_head_emit_entropy=False,
    )
    # Whatever the result looks like — we only need that no exception fired.
    assert result is not None


# ---------------------------------------------------------------------------
# Criticality Distillation integration harness
# ---------------------------------------------------------------------------


from contextlib import contextmanager  # noqa: E402


class _FakeSSMCore(nn.Module):
    """Minimal stand-in for CareSSMCore.

    Exposes the two contract surfaces the runner walks for CD:
      * ``capture_states()`` — context manager yielding a getter for the
        last captured ``[B, T, D]`` tensor.
      * ``log_a`` parameter — the channel-wise decay log-odds CD pushes
        toward the critical value via the seat-masked MSE loss.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Mirror the real core: log_a is a Parameter on the SSM core.
        self.log_a = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self._captured_states: torch.Tensor | None = None

    @contextmanager
    def capture_states(self):
        self._captured_states = None
        try:
            yield lambda: self._captured_states
        finally:
            # Keep the captured tensor live so the runner can read it
            # AFTER the stack exits in a test-friendly manner. The real
            # core clears here; the runner contract is to read INSIDE the
            # context, which this fake also supports.
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._captured_states = x.detach()
        return x


class _TinyCDTrainModel(nn.Module):
    """Tiny model with fake SSM cores the runner can discover via
    ``capture_states`` and hook for CD wiring."""

    def __init__(self, *, dim: int = 4, vocab_size: int = 6, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.cores = nn.ModuleList([_FakeSSMCore(dim) for _ in range(num_layers)])
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embed(inputs)
        for core in self.cores:
            x = core(x)
        return x


def test_train_fast_for_budget_wires_criticality_distillation_4_steps():
    """4-step integration: CD enabled, ingest fires every step, accumulator
    advances, seat refresh fires at the configured cadence, criticality_loss
    contributes gradient to log_a when seats exist."""
    mod = _load_runner_module()
    model = _TinyCDTrainModel(dim=4, vocab_size=6, num_layers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(256, dtype=torch.int16) % 6,
        train_num_tokens=256,
        stride=4,
        seq_len=6,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="fp32",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=4,
        prefetch_batches=False,
        lm_head_backward_mode="fused_streaming_cached",
        lm_head_emit_entropy=True,
        criticality_distill_enabled=True,
        criticality_distill_num_layers=2,
        criticality_distill_dim=4,
        criticality_distill_budget_frac=0.25,
        criticality_distill_trace_ttl_steps=8,
        criticality_distill_trace_half_life_steps=4.0,
        criticality_distill_seat_refresh_interval=2,
        criticality_distill_min_weighted_events_per_layer=0.1,
        criticality_distill_horizon_H=2,
        criticality_distill_event_frac=0.5,
        criticality_distill_weight=1.0,
        # Uniform-pressure forces every step to have events — we're
        # testing orchestration, not pressure math (pressure is validated
        # in test_runner_criticality_pressure.py).
        criticality_distill_uniform_pressure=True,
    )
    cd = result["criticality_distillation_module"]
    # Accumulator advanced per step; after steps 0..3 last_decay_step=3.
    assert int(cd.last_decay_step.item()) == 3, (
        f"expected last_decay_step=3, got {int(cd.last_decay_step.item())}"
    )
    debug = result["criticality_distill_debug"]
    assert debug["ingest_calls"] == 4, debug
    assert debug["seat_refresh_calls"] >= 1, debug
    assert cd.seat_mask.any().item(), "seats never allocated"
    # CD loss actually pushes gradient into log_a once seats exist.
    grad = model.cores[0].log_a.grad
    assert grad is not None, "cores[0].log_a has no grad"
    assert grad.abs().sum().item() > 0.0, "cores[0].log_a.grad is all zero"
    # is_pinned assertion is CUDA-gated; on CPU we can't pin, so skip.
    if torch.cuda.is_available():
        buffers = result.get("_criticality_distill_pinned_buffers")
        if buffers is not None:
            any_tensor = buffers["A"]["aggregated_excess_per_layer"]
            assert any_tensor.is_pinned()


def test_train_fast_for_budget_rejects_episodic_with_epoch_sampling_sequential():
    """``episodic_enabled=True`` is incompatible with epoch-mode sharded
    sampling in Phase 1: the episodic rank receives a 1/N start-shard
    from ``count_sharded_lm_starts`` but the skip-main flow makes it
    silently drop that shard, so 25% (at N=4) of the dataset goes
    unseen each epoch. Phase 1.6's "zero-behavior-change" comparison
    against the legacy 4-rank cell would diverge for reasons unrelated
    to the asymmetric topology. The proper fix — re-shard over N-1 —
    is a Phase 3 prerequisite. For now, refuse the combo.

    Tests both epoch modes (``sequential_epoch`` and ``shuffled_epoch``)
    because they share the same shard-and-drop pathology; the random
    sampler is fine in expectation and stays allowed.
    """
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match=r"sequential_epoch|shuffled_epoch|epoch sampling"):
        mod.train_fast_for_budget(
            model,
            train_tokens=torch.arange(32, dtype=torch.int16) % 6,
            train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
            device=torch.device("cpu"), optimizer=optimizer,
            budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0,
            fused_grad_clip=False,
            rank=0, world_size=2, seed=123, precision="fp32",
            stop_check_interval=1, stop_margin_seconds=0.0,
            vocab_size=6, max_steps=1, prefetch_batches=False,
            episodic_enabled=True,
            train_sampling_mode="sequential_epoch",
        )


def test_train_fast_for_budget_rejects_episodic_with_epoch_sampling_shuffled():
    """Mirror of the sequential_epoch test — the shuffled_epoch sampler
    has the same shard-and-drop pathology under episodic skip-main.
    """
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match=r"sequential_epoch|shuffled_epoch|epoch sampling"):
        mod.train_fast_for_budget(
            model,
            train_tokens=torch.arange(32, dtype=torch.int16) % 6,
            train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
            device=torch.device("cpu"), optimizer=optimizer,
            budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0,
            fused_grad_clip=False,
            rank=0, world_size=2, seed=123, precision="fp32",
            stop_check_interval=1, stop_margin_seconds=0.0,
            vocab_size=6, max_steps=1, prefetch_batches=False,
            episodic_enabled=True,
            train_sampling_mode="shuffled_epoch",
        )


def test_train_fast_for_budget_rejects_episodic_with_dreamworld_weight():
    """``episodic_enabled=True`` with ``dreamworld_weight > 0`` would fire
    ``dreamworld_replay_backward`` on train ranks but skip it on the
    episodic rank (because of skip-main early-return), violating the
    Phase 1 invariant that "episodic submits zeros, train ranks submit
    only main grads". The proper fix is curated-replay integration in
    Task 3.1; until then, refuse the combo.

    Defaults (``dreamworld_weight=0.0``, ``dreamworld_cache_interval=0``)
    are explicitly safe — runs with ``dreamworld_enabled=True`` but no
    knobs wired must not regress.
    """
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match=r"dreamworld|Dreamworld|Task 3\.1"):
        mod.train_fast_for_budget(
            model,
            train_tokens=torch.arange(32, dtype=torch.int16) % 6,
            train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
            device=torch.device("cpu"), optimizer=optimizer,
            budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0,
            fused_grad_clip=False,
            rank=0, world_size=2, seed=123, precision="fp32",
            stop_check_interval=1, stop_margin_seconds=0.0,
            vocab_size=6, max_steps=1, prefetch_batches=False,
            episodic_enabled=True,
            dreamworld_weight=0.5,
        )


def test_train_fast_for_budget_rejects_episodic_with_dreamworld_cache_interval():
    """``dreamworld_cache_interval > 0`` triggers
    ``capture_dream_entry`` on every rank including the episodic rank,
    which is wasteful even before the replay backward asymmetry. Same
    Task 3.1 fix path as the weight knob.
    """
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match=r"dreamworld|Dreamworld|Task 3\.1"):
        mod.train_fast_for_budget(
            model,
            train_tokens=torch.arange(32, dtype=torch.int16) % 6,
            train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
            device=torch.device("cpu"), optimizer=optimizer,
            budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0,
            fused_grad_clip=False,
            rank=0, world_size=2, seed=123, precision="fp32",
            stop_check_interval=1, stop_margin_seconds=0.0,
            vocab_size=6, max_steps=1, prefetch_batches=False,
            episodic_enabled=True,
            dreamworld_cache_interval=4,
        )


def test_train_fast_for_budget_rejects_cd_on_multi_rank():
    """CD is single-rank only in this runner — evidence/seat state is
    rank-local and cd_loss.backward() is not all-reduced. Any world_size > 1
    would produce divergent log_a updates. The runner must raise cleanly
    rather than silently diverge."""
    mod = _load_runner_module()
    model = _TinyCDTrainModel(dim=4, vocab_size=6, num_layers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match=r"single-rank|world_size"):
        mod.train_fast_for_budget(
            model,
            train_tokens=torch.arange(32, dtype=torch.int16) % 6,
            train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
            device=torch.device("cpu"), optimizer=optimizer,
            budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0,
            fused_grad_clip=False,
            rank=0, world_size=2, seed=123, precision="fp32",
            stop_check_interval=1, stop_margin_seconds=0.0,
            vocab_size=6, max_steps=1, prefetch_batches=False,
            criticality_distill_enabled=True,
            lm_head_emit_entropy=True,
            criticality_distill_num_layers=2,
            criticality_distill_dim=4,
        )


def test_train_fast_for_budget_rejects_cd_with_missing_num_layers():
    mod = _load_runner_module()
    model = _TinyCDTrainModel(dim=4, vocab_size=6, num_layers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="num_layers"):
        mod.train_fast_for_budget(
            model,
            train_tokens=torch.arange(32, dtype=torch.int16) % 6,
            train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
            device=torch.device("cpu"), optimizer=optimizer,
            budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0,
            fused_grad_clip=False,
            rank=0, world_size=1, seed=123, precision="fp32",
            stop_check_interval=1, stop_margin_seconds=0.0,
            vocab_size=6, max_steps=1, prefetch_batches=False,
            criticality_distill_enabled=True,
            lm_head_emit_entropy=True,
            criticality_distill_dim=4,
            # num_layers omitted → should raise.
        )


def test_train_fast_for_budget_rejects_cd_with_mismatched_core_count():
    mod = _load_runner_module()
    # Model has 2 cores but CD is configured for 3 layers.
    model = _TinyCDTrainModel(dim=4, vocab_size=6, num_layers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="ssm_cores|capture_states|num_layers"):
        mod.train_fast_for_budget(
            model,
            train_tokens=torch.arange(32, dtype=torch.int16) % 6,
            train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
            device=torch.device("cpu"), optimizer=optimizer,
            budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0,
            fused_grad_clip=False,
            rank=0, world_size=1, seed=123, precision="fp32",
            stop_check_interval=1, stop_margin_seconds=0.0,
            vocab_size=6, max_steps=1, prefetch_batches=False,
            criticality_distill_enabled=True,
            lm_head_emit_entropy=True,
            criticality_distill_num_layers=3,
            criticality_distill_dim=4,
        )


def test_train_result_contains_per_bucket_val_ce_when_rare_bucket_ce_enabled():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    token_frequencies = torch.tensor([100.0, 50.0, 20.0, 10.0, 5.0, 1.0])
    result = mod.train_fast_for_budget(
        model, train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128, stride=4, seq_len=3, batch_size=2,
        device=torch.device("cpu"), optimizer=optimizer,
        budget_seconds=300.0, chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=4, prefetch_batches=False,
        rare_bucket_ce_enabled=True,
        rare_bucket_ce_token_frequencies=token_frequencies,
        rare_bucket_ce_num_buckets=4,
    )
    assert "per_bucket_val_ce" in result
    assert len(result["per_bucket_val_ce"]) == 4
    assert "rare_bucket_val_ce" in result
    assert isinstance(result["rare_bucket_val_ce"], float)
    assert "val_bucket_num_buckets" in result
    assert result["val_bucket_num_buckets"] == 4
    assert "val_bucket_token_counts" in result
    assert len(result["val_bucket_token_counts"]) == 4


def test_train_result_contains_per_window_bucket_ce_when_rare_bucket_ce_enabled():
    """Within-seed paired bootstrap on b0/b1 needs per-eval-window
    bucket CE sums + counts, not just the aggregate. Aggregate must
    still equal the sum of per-window data (sanity check)."""
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    token_frequencies = torch.tensor([100.0, 50.0, 20.0, 10.0, 5.0, 1.0])
    result = mod.train_fast_for_budget(
        model, train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128, stride=4, seq_len=3, batch_size=2,
        device=torch.device("cpu"), optimizer=optimizer,
        budget_seconds=300.0, chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=4, prefetch_batches=False,
        rare_bucket_ce_enabled=True,
        rare_bucket_ce_token_frequencies=token_frequencies,
        rare_bucket_ce_num_buckets=4,
    )
    # Must emit per-window arrays alongside the aggregate.
    assert "per_window_bucket_ce_sum" in result
    assert "per_window_bucket_count" in result
    sums = result["per_window_bucket_ce_sum"]
    counts = result["per_window_bucket_count"]
    # n_windows >= 1; each entry has length=num_buckets.
    assert len(sums) >= 1
    assert len(sums) == len(counts)
    for w_sum, w_count in zip(sums, counts):
        assert len(w_sum) == 4
        assert len(w_count) == 4
        assert all(isinstance(x, float) for x in w_sum)
        assert all(isinstance(x, int) for x in w_count)
    # Aggregate per_bucket_val_ce must equal sum(window_sum) / sum(window_count)
    # (within float tolerance) for each bucket — with ws=1 there's no cross-rank
    # reduction so per-window is the full picture.
    import math
    per_bucket = result["per_bucket_val_ce"]
    counts_total = result["val_bucket_token_counts"]
    for b in range(4):
        win_sum = sum(w[b] for w in sums)
        win_cnt = sum(w[b] for w in counts)
        assert win_cnt == counts_total[b], (
            f"bucket {b}: per-window count {win_cnt} != aggregate {counts_total[b]}"
        )
        if win_cnt > 0:
            expected = win_sum / win_cnt
            assert math.isclose(expected, per_bucket[b], rel_tol=1e-9, abs_tol=1e-9), (
                f"bucket {b}: per-window mean {expected} != aggregate {per_bucket[b]}"
            )


def test_cd_diagnostics_emitted_at_every_seat_refresh():
    mod = _load_runner_module()
    model = _TinyCDTrainModel(dim=4, vocab_size=6, num_layers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(256, dtype=torch.int16) % 6,
        train_num_tokens=256, stride=4, seq_len=6, batch_size=2,
        device=torch.device("cpu"), optimizer=optimizer,
        budget_seconds=300.0, chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=8, prefetch_batches=False,
        lm_head_backward_mode="fused_streaming_cached",
        lm_head_emit_entropy=True,
        criticality_distill_enabled=True,
        criticality_distill_num_layers=2,
        criticality_distill_dim=4,
        criticality_distill_budget_frac=0.25,
        criticality_distill_trace_ttl_steps=8,
        criticality_distill_trace_half_life_steps=4.0,
        criticality_distill_seat_refresh_interval=2,
        criticality_distill_min_weighted_events_per_layer=0.1,
        criticality_distill_horizon_H=2,
        criticality_distill_event_frac=0.5,
        criticality_distill_weight=1.0,
    )
    diags = result["criticality_distillation_diagnostics"]
    # 8 steps / refresh every 2 starting at step=2 = at least 3 snapshots (steps 2, 4, 6).
    assert len(diags) >= 3
    for snap in diags:
        for key in ("step", "seat_churn_per_layer", "budget_occupancy_per_layer",
                    "score_criticality_corr_per_layer", "event_rate_per_layer",
                    "seat_mask_fraction_per_layer"):
            assert key in snap, f"diagnostic snapshot missing {key}"


def test_cd_loss_trajectory_logged_per_step():
    """cd_loss is computed each step CD ingests. Runner must record
    (step, value) pairs so we can plot the CD-loss trajectory per arm."""
    mod = _load_runner_module()
    model = _TinyCDTrainModel(dim=4, vocab_size=6, num_layers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(256, dtype=torch.int16) % 6,
        train_num_tokens=256, stride=4, seq_len=6, batch_size=2,
        device=torch.device("cpu"), optimizer=optimizer,
        budget_seconds=300.0, chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=8, prefetch_batches=False,
        lm_head_backward_mode="fused_streaming_cached",
        lm_head_emit_entropy=True,
        criticality_distill_enabled=True,
        criticality_distill_num_layers=2,
        criticality_distill_dim=4,
        criticality_distill_budget_frac=0.25,
        criticality_distill_trace_ttl_steps=8,
        criticality_distill_trace_half_life_steps=4.0,
        criticality_distill_seat_refresh_interval=2,
        criticality_distill_min_weighted_events_per_layer=0.1,
        criticality_distill_horizon_H=2,
        criticality_distill_event_frac=0.5,
        criticality_distill_weight=1.0,
    )
    traj = result.get("criticality_distill_loss_trajectory")
    assert traj is not None, "missing criticality_distill_loss_trajectory"
    # At least one entry once seats are allocated (step >= seat_refresh_interval).
    assert len(traj) >= 1
    for entry in traj:
        assert "step" in entry and "cd_loss" in entry
        assert isinstance(entry["step"], int)
        assert isinstance(entry["cd_loss"], float)
    # Steps must be monotonically nondecreasing.
    steps = [e["step"] for e in traj]
    assert steps == sorted(steps)


def test_runner_emits_topology_snapshot_when_requested():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = mod.train_fast_for_budget(
        model, train_tokens=torch.arange(32, dtype=torch.int16) % 6,
        train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
        device=torch.device("cpu"), optimizer=optimizer,
        budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=1, prefetch_batches=False,
        emit_topology_snapshot=True,
    )
    snap = result.get("topology_snapshot")
    assert snap is not None
    # CPU info must be present in some form.
    assert "lscpu" in snap or "cpu_info" in snap or snap.get("cpu_unavailable") is True
    # GPU topo may be absent on macOS — tolerate.
    assert (
        "nvidia_smi_topo" in snap
        or "gpu_topo" in snap
        or snap.get("gpu_topo_unavailable") is True
    )
    # NUMA likewise.
    assert "numactl_h" in snap or "numa_unavailable" in snap


def test_topology_snapshot_absent_when_flag_off():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = mod.train_fast_for_budget(
        model, train_tokens=torch.arange(32, dtype=torch.int16) % 6,
        train_num_tokens=32, stride=4, seq_len=3, batch_size=2,
        device=torch.device("cpu"), optimizer=optimizer,
        budget_seconds=1.0, chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=1, prefetch_batches=False,
        # emit_topology_snapshot defaults to False.
    )
    assert "topology_snapshot" not in result


def test_measure_cd_overhead_emits_cd_overhead_block():
    mod = _load_runner_module()
    # Distinct model instances for each of the two runs — otherwise
    # state carries across.
    def _make_model():
        return _TinyCDTrainModel(dim=4, vocab_size=6, num_layers=2)

    def _make_optimizer(model):
        return torch.optim.SGD(model.parameters(), lr=0.01)

    def train_fn(**kwargs):
        model = _make_model()
        optimizer = _make_optimizer(model)
        kwargs.setdefault("model", model)
        kwargs.setdefault("optimizer", optimizer)
        return mod.train_fast_for_budget(**kwargs)

    result = mod.measure_cd_overhead(
        train_fn,
        train_tokens=torch.arange(256, dtype=torch.int16) % 6,
        train_num_tokens=256, stride=4, seq_len=6, batch_size=2,
        device=torch.device("cpu"),
        budget_seconds=300.0, chunk_size=2, grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=4, prefetch_batches=False,
        lm_head_backward_mode="fused_streaming_cached",
        lm_head_emit_entropy=True,
        criticality_distill_num_layers=2,
        criticality_distill_dim=4,
        criticality_distill_budget_frac=0.25,
        criticality_distill_trace_ttl_steps=8,
        criticality_distill_trace_half_life_steps=4.0,
        criticality_distill_seat_refresh_interval=2,
        criticality_distill_min_weighted_events_per_layer=0.1,
        criticality_distill_horizon_H=2,
        criticality_distill_event_frac=0.5,
        criticality_distill_weight=1.0,
    )
    assert "tokens_per_sec_baseline" in result
    assert "tokens_per_sec_treatment" in result
    assert "overhead_fraction" in result
    assert isinstance(result["tokens_per_sec_baseline"], float)
    assert isinstance(result["tokens_per_sec_treatment"], float)
    assert isinstance(result["overhead_fraction"], float)
    # Treatment result dict has cd_overhead attached.
    tr = result["treatment_result"]
    assert "cd_overhead" in tr
    assert set(tr["cd_overhead"].keys()) == {
        "tokens_per_sec_baseline",
        "tokens_per_sec_treatment",
        "overhead_fraction",
    }
