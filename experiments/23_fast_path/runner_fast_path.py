#!/usr/bin/env python3
"""Exp 23 single-condition fastest-path DDP runner.

This is deliberately narrower than the previous experiment launchers.
It keeps only the final bare-SSM training path and makes the 600s hot
loop explicit: vectorized batch gather, fused linear+CE head/loss,
fused Muon/grad-clip knobs, amortized stop checks, and compact timing JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "17_local_attn_sidecar"))
sys.path.insert(0, str(REPO / "experiments" / "21_sgns_tokenizer"))
sys.path.insert(0, str(EXPERIMENT))


def configure_exp23_fast_backend_defaults(
    env: dict[str, str] | os._Environ[str] = os.environ,
) -> None:
    """Default Exp23 to the no-Inductor submission hot path.

    The core modules keep torch.compile as their historical default for
    standalone experiments. Exp23 is different: it is the competition-speed
    runner and already requires the native `_ssm_scan` extension on H100 pods.
    Falling back into Inductor during `verify_diag_recurrence()` can burn
    minutes before the timed loop starts, so opt into the native scan here
    unless the launcher deliberately asks for a different backend.
    """
    env.setdefault("CHAOSCONTROL_DIAG_SCAN_BACKEND", "ssm_scan")
    env.setdefault("CHAOSCONTROL_POST_SCAN_BACKEND", "eager")


configure_exp23_fast_backend_defaults()

from chaoscontrol.core import verify_diag_recurrence  # noqa: E402
from chaoscontrol.data import (  # noqa: E402
    load_fineweb_tokens,
    resolve_device,
    resolve_param_dtype,
)
from chaoscontrol.distributed import (  # noqa: E402
    AsyncGradAllReducer,
    allreduce_grads,
    broadcast_params,
    clip_grad_norm_fused,
    should_stop_now,
)
from chaoscontrol.optim.lamb import LAMB  # noqa: E402
from chaoscontrol.optim.muon import Muon  # noqa: E402
from chaoscontrol.optim.param_groups import build_optimizer_params  # noqa: E402
from chaoscontrol.optim.scopt import (  # noqa: E402
    FrequencyBucketBaseline,
    ScarcityAwareOptimizer,
    scarcity_pressure_from_ce,
)
from chaoscontrol.optim.semantic import SemanticOptimizer  # noqa: E402
from chaoscontrol.precision import autocast_context  # noqa: E402
from chaoscontrol.train_ssm import (  # noqa: E402
    _compiled_step_fn,
    _reject_unsupported,
    chunked_lm_head_backward,
    fused_lm_head_backend_for_mode,
    fused_lm_head_backward,
    fused_lm_head_backward_with_ce,
    fused_lm_head_loss_with_ce,
    fused_lm_head_weighted_loss_with_ce,
    full_lm_head_backward,
)
from fast_path import (  # noqa: E402
    Exp23BatchPrefetcher,
    SequentialShardedStartSampler,
    ShuffledEpochShardedStartSampler,
    batch_from_start_tensor,
    choose_lm_starts_lazy,
    count_lm_starts,
    count_sharded_lm_starts,
    should_stop_training_loop,
    sample_sharded_lm_starts,
    sequential_epoch_steps,
    sequential_sharded_lm_starts,
    shuffled_epoch_sharded_lm_starts,
    steady_state_step_seconds,
    summarize_cuda_graph_gate,
    summarize_train_timing,
)
from training_hooks import (  # noqa: E402
    FastSlowConsolidator,
    predictive_auxiliary_loss,
    spectral_regularization_loss,
    spectral_summary,
    zero_embedding_grad_until,
)
from dreamworld import (  # noqa: E402
    DreamReplayBuffer,
    capture_dream_entry,
    dreamworld_replay_backward,
)
from runner_exp17 import (  # noqa: E402
    build_sentencepiece_luts,
    evaluate_bpb_sp,
)
from runner_exp21 import (  # noqa: E402
    _apply_embed_init,
    _save_output_ckpt,
    build_model,
)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    return int(val)


def _init_distributed(world_size_override: int | None) -> tuple[int, int, int]:
    env_world = _env_int("WORLD_SIZE", 1)
    target_world = world_size_override if world_size_override is not None else env_world
    if target_world <= 1:
        return 0, 1, 0
    if not (dist.is_available() and dist.is_initialized()):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), _env_int("LOCAL_RANK", 0)


def _pick_device(local_rank: int, config_device: str) -> torch.device:
    if torch.cuda.is_available() and config_device in ("auto", "cuda"):
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return resolve_device(config_device)


def _build_optimizer(
    config: dict[str, Any],
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "muon")).strip()
    base_lr = float(config.get("base_lr", 0.128))
    weight_decay = float(config.get("weight_decay", 0.01))
    grouping = str(config.get("optimizer_param_grouping", "flat")).strip()
    dynamics_lr_mul = float(config.get("optimizer_dynamics_lr_mul", 0.1))
    params = build_optimizer_params(
        list(model.named_parameters()),
        grouping=grouping,
        base_lr=base_lr,
        weight_decay=weight_decay,
        dynamics_lr_mul=dynamics_lr_mul,
    )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    if name == "lamb":
        return LAMB(params, lr=base_lr, weight_decay=weight_decay)
    if name == "muon":
        opt = Muon(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
            adamw_lr=base_lr,
            adamw_weight_decay=weight_decay,
            fused=bool(config.get("fused_muon", True)),
        )
        opt.bind_param_names(list(model.named_parameters()))
        return opt
    if name == "semantic":
        semantic_cfg = _semantic_optimizer_config(
            model,
            int(config.get("semantic_layer_index", 0)),
        )
        opt = SemanticOptimizer(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
            adamw_lr=base_lr,
            adamw_weight_decay=weight_decay,
            momentum_min=float(config.get("semantic_momentum_min", 0.5)),
            **semantic_cfg,
        )
        opt.bind_param_names(list(model.named_parameters()))
        return opt
    if name in {"scopt", "scarcity", "scarcity_aware"}:
        scopt_cfg = _scopt_optimizer_config(
            model,
            int(config.get("scopt_layer_index", 0)),
        )
        opt = ScarcityAwareOptimizer(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
            adamw_lr=base_lr,
            adamw_weight_decay=weight_decay,
            warmup_steps=int(config.get("scopt_warmup_steps", 200)),
            rare_ema_decay=float(config.get("scopt_rare_ema_decay", 0.9)),
            rare_orthogonal_weight=float(
                config.get("scopt_rare_orthogonal_weight", 1.0)
            ),
            rare_macro_c=float(config.get("scopt_rare_macro_c", 0.5)),
            row_scarcity_power=float(config.get("scopt_row_scarcity_power", 0.5)),
            tau_std_scale=float(config.get("scopt_tau_std_scale", 0.5)),
            **scopt_cfg,
        )
        opt.bind_param_names(list(model.named_parameters()))
        return opt
    raise ValueError(f"unknown optimizer {name!r}")


def _semantic_optimizer_config(
    model: torch.nn.Module,
    layer_index: int,
) -> dict[str, Any]:
    prefix = f"layers.{int(layer_index)}.core"
    a_param_name = f"{prefix}.log_a"
    named_params = {name for name, _ in model.named_parameters()}
    if a_param_name not in named_params:
        raise ValueError(f"semantic optimizer requires {a_param_name!r}")

    candidate_axes = {
        "in_proj.weight": 0,
        "select_proj.weight": 0,
        "gate_proj.weight": 0,
        "delta_proj.weight": 0,
        "out_proj.weight": 1,
    }
    channel_map: dict[str, int] = {
        f"{prefix}.{name}": axis
        for name, axis in candidate_axes.items()
        if f"{prefix}.{name}" in named_params
    }
    if not channel_map:
        raise ValueError(
            "semantic optimizer requires at least one channel-coupled matrix "
            f"under {prefix!r}"
        )

    return {
        "a_param_name": a_param_name,
        "channel_map": channel_map,
    }


def _scopt_optimizer_config(
    model: torch.nn.Module,
    layer_index: int,
) -> dict[str, Any]:
    """Build ScOpt's per-submatrix two-sided scarcity map.

    For each projection submodule we register both a forward pre-hook
    (input activation) and a forward hook (output activation). The
    matrix scarcity map then references each submatrix's own input and
    output channels so ``L · G · R`` is genuinely two-sided rather than
    collapsing to a single per-block signal.
    """
    prefix = f"layers.{int(layer_index)}.core"
    named_params = {name for name, _ in model.named_parameters()}
    named_modules = {name for name, _ in model.named_modules()}
    if f"{prefix}.log_a" not in named_params:
        raise ValueError(f"ScOpt requires {prefix + '.log_a'!r}")

    projection_candidates = (
        "in_proj",
        "select_proj",
        "gate_proj",
        "delta_proj",
        "out_proj",
    )
    present = [s for s in projection_candidates if f"{prefix}.{s}" in named_modules]
    if not present:
        raise ValueError(
            f"ScOpt requires projection submodules under {prefix!r}"
        )

    matrix_scarcity_map: dict[str, tuple[str | None, str | None]] = {}
    for sub in present:
        weight_name = f"{prefix}.{sub}.weight"
        if weight_name not in named_params:
            continue
        # out_key is this submodule's own output; in_key is its input.
        # For in/select/gate/delta_proj, the input is the block's input
        # (shared across all four). For out_proj, the input is the
        # gated state (internal channel).
        matrix_scarcity_map[weight_name] = (
            f"{prefix}.{sub}.__out__",
            f"{prefix}.{sub}.__in__",
        )
    if not matrix_scarcity_map:
        raise ValueError(
            "ScOpt requires at least one channel-coupled matrix "
            f"under {prefix!r}"
        )

    # log_a is per-internal-channel (dim). The in_proj output carries
    # that axis directly; prefer it, fall back to any projection that's
    # present (all of in/select/gate_proj share the internal dim).
    for sub in ("in_proj", "select_proj", "gate_proj"):
        if sub in present:
            recurrence_source = f"{prefix}.{sub}.__out__"
            break
    else:
        recurrence_source = f"{prefix}.{present[0]}.__out__"

    recurrence_scarcity_map = {f"{prefix}.log_a": recurrence_source}

    return {
        "row_param_names": {"embed.weight", "lm_head.weight"},
        "matrix_scarcity_map": matrix_scarcity_map,
        "recurrence_scarcity_map": recurrence_scarcity_map,
    }


def _optimizer_diagnostics(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    diagnostics = {
        "type": optimizer.__class__.__name__,
    }
    beta_trace = getattr(optimizer, "beta_trace", None)
    if callable(beta_trace):
        trace = optimizer.beta_trace()
        if trace is None:
            diagnostics["beta_trace"] = None
        else:
            diagnostics["beta_trace"] = {
                key: float(trace[key])
                for key in ("beta_min", "beta_max", "beta_mean")
                if key in trace
            }
    scarcity_trace = getattr(optimizer, "scarcity_trace", None)
    if callable(scarcity_trace):
        diagnostics["scarcity_trace"] = scarcity_trace()
    return diagnostics


def _average_dense_tensors(
    tensors: list[torch.Tensor],
    *,
    world_size: int,
) -> None:
    if world_size <= 1:
        return
    for tensor in tensors:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size)


def _average_dense_tensors_coalesced(
    tensors: list[torch.Tensor],
    *,
    world_size: int,
) -> None:
    """Flatten-reduce-unflatten for a list of same-dtype dense tensors.

    Matches the ``allreduce_grads`` pattern in ``src/chaoscontrol/distributed.py``
    and collapses N per-tensor NCCL launches (each paying ~100-500µs of
    fixed overhead) into a single coalesced collective plus two cheap
    host-side copies. Tensors must share a dtype; the caller is
    responsible for grouping by dtype when that differs.

    Mutates each input tensor in place so callers that kept references
    see the averaged value.
    """
    if world_size <= 1 or not tensors:
        return
    contig = [t.contiguous() for t in tensors]
    flat = torch._utils._flatten_dense_tensors(contig)
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    synced = torch._utils._unflatten_dense_tensors(flat, contig)
    for original, s in zip(tensors, synced, strict=True):
        original.copy_(s)


def _module_by_qualified_name(
    model: torch.nn.Module,
    name: str,
) -> torch.nn.Module | None:
    modules = dict(model.named_modules())
    return modules.get(name)


def _scopt_activation_keys(optimizer: ScarcityAwareOptimizer) -> set[str]:
    """All activation keys referenced by ScOpt's scarcity maps."""
    keys: set[str] = set()
    for out_key, in_key in getattr(optimizer, "_matrix_scarcity_map", {}).values():
        if out_key is not None:
            keys.add(str(out_key))
        if in_key is not None:
            keys.add(str(in_key))
    for key in getattr(optimizer, "_recurrence_scarcity_map", {}).values():
        keys.add(str(key))
    return keys


def _base_module_names(keys: set[str]) -> set[str]:
    """Strip ``.__in__`` / ``.__out__`` suffixes to module qualnames."""
    bases: set[str] = set()
    for key in keys:
        if key.endswith(".__in__"):
            bases.add(key[: -len(".__in__")])
        elif key.endswith(".__out__"):
            bases.add(key[: -len(".__out__")])
        else:
            bases.add(key)
    return bases


def _register_scopt_activation_hooks(
    model: torch.nn.Module,
    keys: set[str],
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    """Register forward pre-hooks (inputs) and forward hooks (outputs).

    Each base module named in ``keys`` gets both hooks; the ``__in__``
    activation captures the first positional arg, and the ``__out__``
    activation captures the module's return. Only tensors that require
    grad are stored (so downstream autograd.grad can use them).
    """
    activations: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    modules = dict(model.named_modules())

    def _make_pre_hook(capture_key: str):
        def hook(_module, args):
            if not args:
                return
            tensor = args[0]
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                activations[capture_key] = tensor

        return hook

    def _make_post_hook(capture_key: str):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                activations[capture_key] = tensor

        return hook

    for base in sorted(_base_module_names(keys)):
        module = modules.get(base)
        if module is None:
            continue
        in_key = f"{base}.__in__"
        out_key = f"{base}.__out__"
        if in_key in keys:
            handles.append(module.register_forward_pre_hook(_make_pre_hook(in_key)))
        if out_key in keys or base in keys:
            # Legacy callers that pass a bare module name get the output
            # under that exact key (no suffix) for backward compatibility
            # with tests that predate the in/out split.
            legacy_key = base if base in keys else out_key
            handles.append(module.register_forward_hook(_make_post_hook(legacy_key)))
    return activations, handles


class _ScOptPending:
    """Rare-side outputs staged for post-clip commit.

    Built inside ``_run_scopt_train_step`` on split steps and applied
    after grad-clip so that clipped-batch rare gradients can be rejected
    (spec line 186). Channel-pressure keys are stored sorted so ranks
    all-reduce in matching order.
    """

    __slots__ = (
        "rare_map",
        "channel_pressure_items",
        "row_pressure",
        "pressure_stats",
    )

    def __init__(
        self,
        *,
        rare_map: dict[str, torch.Tensor],
        channel_pressure_items: list[tuple[str, torch.Tensor]],
        row_pressure: torch.Tensor | None,
        pressure_stats: dict[str, float],
    ) -> None:
        self.rare_map = rare_map
        self.channel_pressure_items = channel_pressure_items
        self.row_pressure = row_pressure
        self.pressure_stats = pressure_stats


def _pressure_summary(pressure: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        flat = pressure.detach().float().reshape(-1)
        total = int(flat.numel())
        if total == 0:
            return {}
        positive = (flat > 0).sum().item()
        return {
            "min": float(flat.min()),
            "median": float(flat.median()),
            "p95": float(torch.quantile(flat, 0.95)),
            "max": float(flat.max()),
            "fraction_positive": float(positive) / float(total),
        }


def _apply_scopt_pending(
    optimizer: ScarcityAwareOptimizer,
    pending: _ScOptPending | None,
    *,
    skip: bool,
) -> None:
    """Commit deferred rare-side updates after grad-clip decision.

    ``skip=True`` discards rare_grad_ema and channel_pressure for this
    step (clipped gradients are noisy per spec line 186). Row pressure
    is derived from CE rather than gradients and is applied either way.
    Pressure distribution stats are always recorded for telemetry.
    """
    if pending is None:
        return
    optimizer.record_pressure_stats(pending.pressure_stats)
    if pending.row_pressure is not None:
        optimizer.set_row_pressure_ema(pending.row_pressure)
    if skip:
        return
    optimizer.update_rare_grad_ema(pending.rare_map)
    if pending.channel_pressure_items:
        optimizer.set_channel_pressure(dict(pending.channel_pressure_items))


_FUSED_LM_HEAD_MODES = {
    "fused",
    "fused_streaming",
    "fused_streaming_v2",
    "fused_streaming_cached",
    "fused_norm_streaming_v2",
}


def _scopt_writes_enabled(optimizer: ScarcityAwareOptimizer) -> bool:
    writes_enabled = getattr(optimizer, "_writes_enabled", None)
    if callable(writes_enabled):
        return bool(writes_enabled())
    return True


def _should_run_scopt_split_step(
    optimizer: ScarcityAwareOptimizer,
    *,
    step: int,
    split_interval: int,
) -> bool:
    if not _scopt_writes_enabled(optimizer):
        return False
    split_every = max(1, int(split_interval))
    return int(step) % split_every == 0


def _run_scopt_common_train_step(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    precision: str,
    ddp_active: bool,
    world_size: int,
    chunk_size: int = 1024,
    compile_full_path: bool = False,
    lm_head_backward_mode: str = "fused",
    lm_head_tile_size: int = 1024,
    grad_allreduce_mode: str = "bulk",
    baseline: FrequencyBucketBaseline | torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Fast ScOpt common step for warmup/non-split iterations.

    ScOpt only needs unreduced CE on split steps for rare gradients, but
    its bucket baseline should still see the ordinary CE stream. The
    fused ``*_with_ce`` helper gives us that telemetry without leaving
    the optimized LM-head path.
    """
    mode = str(lm_head_backward_mode).strip().lower()
    if isinstance(baseline, FrequencyBucketBaseline) and mode in _FUSED_LM_HEAD_MODES:
        _reject_unsupported(model)
        with autocast_context(precision, device_type=inputs.device.type):
            hidden = (
                _compiled_step_fn()(model, inputs)
                if compile_full_path
                else model.encode(inputs)
            )
            backend_name = fused_lm_head_backend_for_mode(mode)
            loss, per_token_ce = fused_lm_head_backward_with_ce(
                hidden=hidden,
                final_norm=model.final_norm,
                lm_head=model.lm_head,
                targets=targets,
                backend=backend_name,
                tile_size=int(lm_head_tile_size),
            )
        baseline.update(per_token_ce.reshape_as(targets), targets)
        if ddp_active and world_size > 1:
            if str(grad_allreduce_mode).strip().lower() != "bulk":
                raise ValueError("ScOpt currently requires grad_allreduce_mode='bulk'")
            allreduce_grads(model, world_size)
        return loss

    return _run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=chunk_size,
        precision=precision,
        ddp_active=ddp_active,
        world_size=world_size,
        compile_full_path=compile_full_path,
        lm_head_backward_mode=lm_head_backward_mode,
        lm_head_tile_size=lm_head_tile_size,
        grad_allreduce_mode=grad_allreduce_mode,
    )


class LossTriggeredReplayDecision:
    __slots__ = (
        "_local_loss",
        "_ema_loss",
        "_local_ratio",
        "_local_pressure",
        "_global_pressure",
        "_fire_count",
        "_local_fire",
        "_triggered",
    )

    def __init__(
        self,
        *,
        local_loss: float | torch.Tensor,
        ema_loss: float | torch.Tensor,
        local_ratio: float | torch.Tensor,
        local_pressure: float | torch.Tensor,
        global_pressure: float | torch.Tensor,
        fire_count: int | torch.Tensor,
        local_fire: bool | torch.Tensor,
        triggered: bool | torch.Tensor,
    ) -> None:
        self._local_loss = local_loss
        self._ema_loss = ema_loss
        self._local_ratio = local_ratio
        self._local_pressure = local_pressure
        self._global_pressure = global_pressure
        self._fire_count = fire_count
        self._local_fire = local_fire
        self._triggered = triggered

    @staticmethod
    def _materialize_float(value: float | torch.Tensor) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().item())
        return float(value)

    @staticmethod
    def _materialize_int(value: int | torch.Tensor) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.detach().item())
        return int(value)

    @staticmethod
    def _materialize_bool(value: bool | torch.Tensor) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(value.detach().item())
        return bool(value)

    @property
    def local_loss(self) -> float:
        value = self._materialize_float(self._local_loss)
        self._local_loss = value
        return value

    @property
    def ema_loss(self) -> float:
        value = self._materialize_float(self._ema_loss)
        self._ema_loss = value
        return value

    @property
    def local_ratio(self) -> float:
        value = self._materialize_float(self._local_ratio)
        self._local_ratio = value
        return value

    @property
    def local_pressure(self) -> float:
        value = self._materialize_float(self._local_pressure)
        self._local_pressure = value
        return value

    @property
    def global_pressure(self) -> float:
        value = self._materialize_float(self._global_pressure)
        self._global_pressure = value
        return value

    @property
    def fire_count(self) -> int:
        value = self._materialize_int(self._fire_count)
        self._fire_count = value
        return value

    @property
    def local_fire(self) -> bool:
        value = self._materialize_bool(self._local_fire)
        self._local_fire = value
        return value

    @property
    def triggered(self) -> bool:
        value = self._materialize_bool(self._triggered)
        self._triggered = value
        return value


class LossTriggeredReplayEMA:
    """EMA loss-pressure gate for synchronized event-triggered replay.

    The runner uses this to schedule *next-step* dream replay. The one-step
    delay keeps the normal train-step gradient/all-reduce order intact: the
    event decision is computed from the just-finished loss, then every rank
    agrees on whether the following step should include replay.
    """

    def __init__(
        self,
        *,
        decay: float = 0.99,
        warmup_steps: int = 32,
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if int(warmup_steps) < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.eps = float(eps)
        self.value: torch.Tensor | None = None
        self.observations = 0

    def update(
        self,
        loss: torch.Tensor,
        *,
        threshold: float = 1.10,
        pressure_threshold: float = 0.05,
        ddp_active: bool = False,
        world_size: int = 1,
        device: torch.device | None = None,
    ) -> LossTriggeredReplayDecision | None:
        loss_value = loss.detach().float().reshape(())
        if self.value is None:
            self.value = loss_value.clone()
            self.observations = 1
            return None

        ema_before = self.value
        self.value = self.decay * self.value + (1.0 - self.decay) * loss_value
        self.observations += 1
        if self.observations <= self.warmup_steps:
            return None

        ratio = loss_value / ema_before.clamp_min(self.eps)
        local_pressure = (ratio - float(threshold)).clamp_min(0.0)
        local_fire = local_pressure > 0.0
        if ddp_active and int(world_size) > 1:
            reduce_device = device if device is not None else loss.device
            pressure_and_fire = torch.stack(
                (
                    local_pressure.to(device=reduce_device, dtype=torch.float32),
                    local_fire.to(device=reduce_device, dtype=torch.float32),
                )
            )
            dist.all_reduce(pressure_and_fire, op=dist.ReduceOp.SUM)
            global_pressure = pressure_and_fire[0] / float(world_size)
            fire_count = pressure_and_fire[1].round().to(torch.int64)
        else:
            global_pressure = local_pressure
            fire_count = local_fire.to(torch.int64)
        triggered = global_pressure > float(pressure_threshold)
        return LossTriggeredReplayDecision(
            local_loss=loss_value,
            ema_loss=ema_before,
            local_ratio=ratio,
            local_pressure=local_pressure,
            global_pressure=global_pressure,
            fire_count=fire_count,
            local_fire=local_fire,
            triggered=triggered,
        )


def _run_scopt_train_step(
    *,
    model: torch.nn.Module,
    optimizer: ScarcityAwareOptimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    token_frequencies: torch.Tensor,
    precision: str,
    ddp_active: bool,
    world_size: int,
    step: int,
    split_interval: int = 4,
    baseline: FrequencyBucketBaseline | torch.Tensor | float | None = None,
    pressure_upper_c: float | None = None,
    pressure_upper_floor: float = 1.0,
    lm_head_backward_mode: str = "fused",
    lm_head_tile_size: int = 1024,
) -> tuple[torch.Tensor, _ScOptPending | None]:
    """ScOpt training step with retained graph and rare-grad split.

    Returns ``(loss, pending)`` where ``pending`` is ``None`` on
    non-split steps. The caller must invoke ``_apply_scopt_pending``
    after grad-clip so a clipped step can reject its rare contribution.
    ``baseline`` may be a :class:`FrequencyBucketBaseline` (whose
    per-bucket EMA is updated in-place from the current CE), a tensor,
    a scalar, or ``None`` (bucket baseline is the expected default).
    """
    _reject_unsupported(model)
    split_every = max(1, int(split_interval))
    is_split_step = int(step) % split_every == 0
    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    params = [param for _, param in named_params]
    activations: dict[str, torch.Tensor] = {}
    hook_handles: list[Any] = []
    if is_split_step:
        activations, hook_handles = _register_scopt_activation_hooks(
            model,
            _scopt_activation_keys(optimizer),
        )
    with autocast_context(precision, device_type=inputs.device.type):
        try:
            hidden = model.encode(inputs)
        finally:
            for handle in hook_handles:
                handle.remove()
        mode = str(lm_head_backward_mode).strip().lower()
        if mode in _FUSED_LM_HEAD_MODES:
            backend_name = fused_lm_head_backend_for_mode(mode)
            total_loss, per_token_ce = fused_lm_head_loss_with_ce(
                hidden=hidden,
                final_norm=model.final_norm,
                lm_head=model.lm_head,
                targets=targets,
                backend=backend_name,
                tile_size=int(lm_head_tile_size),
            )
            ce = per_token_ce.reshape_as(targets)
        else:
            logits = model.lm_head(model.final_norm(hidden))
            vocab = logits.size(-1)
            ce = F.cross_entropy(
                logits.reshape(-1, vocab).float(),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            total_loss = ce.mean()
        if isinstance(baseline, FrequencyBucketBaseline):
            baseline_value: torch.Tensor | float | None = baseline.baseline(targets)
            baseline.update(ce, targets)
        elif baseline is None:
            baseline_value = 0.0
        else:
            baseline_value = baseline
        pressure = scarcity_pressure_from_ce(
            ce,
            targets,
            token_frequencies=token_frequencies,
            baseline=baseline_value,
            upper_c=pressure_upper_c,
            upper_floor=pressure_upper_floor,
        )
        rare_loss = None
        if is_split_step:
            if mode in _FUSED_LM_HEAD_MODES:
                rare_loss, _ = fused_lm_head_weighted_loss_with_ce(
                    hidden=hidden,
                    final_norm=model.final_norm,
                    lm_head=model.lm_head,
                    targets=targets,
                    token_weight=pressure,
                    backend=backend_name,
                    tile_size=int(lm_head_tile_size),
                )
            else:
                rare_loss = (ce * pressure).sum() / pressure.sum().clamp_min(1.0)

    common_grads = torch.autograd.grad(
        total_loss,
        params,
        retain_graph=is_split_step,
        allow_unused=True,
    )
    common_unused = {
        name for (name, _), g in zip(named_params, common_grads) if g is None
    }
    for param, grad in zip(params, common_grads, strict=True):
        param.grad = None if grad is None else grad.detach().contiguous()
    if ddp_active and world_size > 1:
        allreduce_grads(model, world_size)

    if not is_split_step:
        return total_loss.detach(), None
    assert rare_loss is not None

    # Split step: compute rare parameter gradients and activation
    # gradients in one autograd traversal. The older correctness path
    # asked autograd once for params and then once per activation hook;
    # this keeps the same tensors but removes N extra reverse walks.
    activation_items = sorted(activations.items())
    unique_activations: list[torch.Tensor] = []
    activation_lookup: dict[int, int] = {}
    activation_unique_indices: list[int] = []
    for _, activation in activation_items:
        activation_id = id(activation)
        unique_idx = activation_lookup.get(activation_id)
        if unique_idx is None:
            unique_idx = len(unique_activations)
            activation_lookup[activation_id] = unique_idx
            unique_activations.append(activation)
        activation_unique_indices.append(unique_idx)

    rare_outputs = torch.autograd.grad(
        rare_loss,
        [*params, *unique_activations],
        retain_graph=False,
        allow_unused=True,
    )
    rare_grads = rare_outputs[: len(params)]
    activation_grads = rare_outputs[len(params) :]
    rare_unused = {
        name for (name, _), g in zip(named_params, rare_grads) if g is None
    }
    extra_rare_unused = rare_unused - common_unused
    if extra_rare_unused:
        raise RuntimeError(
            "ScOpt: parameters have common-loss gradients but no rare-loss "
            "gradients — possible graph disconnect. Offending params: "
            f"{sorted(extra_rare_unused)}"
        )
    rare_map: dict[str, torch.Tensor] = {}
    dense_rare: list[torch.Tensor] = []
    for (name, _), grad in zip(named_params, rare_grads, strict=True):
        if grad is None:
            continue  # Genuinely unused in both rare and common.
        dense = grad.detach().contiguous()
        rare_map[name] = dense
        dense_rare.append(dense)
    if ddp_active and world_size > 1 and dense_rare:
        _average_dense_tensors_coalesced(dense_rare, world_size=world_size)

    # Iterate activations in sorted order so every rank issues all_reduce
    # calls in identical sequence. Duplicate activation tensors reuse the
    # same gradient, matching the old one-call-per-hook behavior without
    # asking autograd to recompute it.
    channel_pressure_items: list[tuple[str, torch.Tensor]] = []
    for (key, _activation), unique_idx in zip(
        activation_items,
        activation_unique_indices,
        strict=True,
    ):
        dh = activation_grads[unique_idx]
        if dh is None:
            continue
        reduce_dims = tuple(range(max(0, dh.ndim - 1)))
        pressure_vec = dh.detach().float().abs().mean(dim=reduce_dims)
        channel_pressure_items.append((key, pressure_vec))
    if ddp_active and world_size > 1 and channel_pressure_items:
        _average_dense_tensors_coalesced(
            [vec for _, vec in channel_pressure_items],
            world_size=world_size,
        )

    # Row pressure is derived from CE (not gradients), so it doesn't
    # need grad-clip gating. EMA update runs locally; all-reduce keeps
    # replicas consistent.
    row_pressure = optimizer.update_row_pressure_ema(
        targets,
        pressure,
        vocab_size=int(token_frequencies.numel()),
    )
    if row_pressure is not None and ddp_active and world_size > 1:
        _average_dense_tensors([row_pressure], world_size=world_size)

    pressure_stats = _pressure_summary(pressure)

    pending = _ScOptPending(
        rare_map=rare_map,
        channel_pressure_items=channel_pressure_items,
        row_pressure=row_pressure,
        pressure_stats=pressure_stats,
    )
    return total_loss.detach(), pending


def _run_train_step(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    precision: str,
    ddp_active: bool,
    world_size: int,
    compile_full_path: bool = False,
    lm_head_backward_mode: str = "fused",
    lm_head_tile_size: int = 1024,
    grad_allreduce_mode: str = "bulk",
    async_grad_reducer: AsyncGradAllReducer | None = None,
    spectral_reg_lambda_dead: float = 0.0,
    spectral_reg_lambda_sticky: float = 0.0,
    spectral_reg_min_a: float = 0.05,
    spectral_reg_max_a: float = 0.98,
    predictive_aux_weight: float = 0.0,
    predictive_aux_horizon: int = 0,
    predictive_aux_projection: torch.nn.Module | None = None,
    dreamworld_entry: Any | None = None,
    dreamworld_weight: float = 0.0,
    dreamworld_replay_batch_size: int = 0,
    dreamworld_generator: torch.Generator | None = None,
) -> torch.Tensor:
    _reject_unsupported(model)
    with autocast_context(precision, device_type=inputs.device.type):
        if compile_full_path:
            hidden = _compiled_step_fn()(model, inputs)
        else:
            hidden = model.encode(inputs)
        if (
            predictive_aux_weight > 0.0
            and predictive_aux_horizon > 0
            and predictive_aux_projection is not None
        ):
            aux = predictive_auxiliary_loss(
                hidden,
                projection=predictive_aux_projection,
                horizon=predictive_aux_horizon,
            )
            if aux is not None:
                (float(predictive_aux_weight) * aux).backward(retain_graph=True)
        if lm_head_backward_mode == "single":
            loss = full_lm_head_backward(
                hidden=hidden,
                final_norm=model.final_norm,
                lm_head=model.lm_head,
                targets=targets,
            )
        elif lm_head_backward_mode in {
            "fused",
            "fused_streaming",
            "fused_streaming_v2",
            "fused_streaming_cached",
            "fused_norm_streaming_v2",
        }:
            backend_name = fused_lm_head_backend_for_mode(lm_head_backward_mode)
            loss = fused_lm_head_backward(
                hidden=hidden,
                final_norm=model.final_norm,
                lm_head=model.lm_head,
                targets=targets,
                tile_size=int(lm_head_tile_size),
                backend=backend_name,
            )
        elif lm_head_backward_mode == "chunked":
            hidden_for_ce = hidden.detach().requires_grad_(True)
            loss = chunked_lm_head_backward(
                hidden=hidden_for_ce,
                final_norm=model.final_norm,
                lm_head=model.lm_head,
                targets=targets,
                chunk_size=chunk_size,
            )
            hidden.backward(gradient=hidden_for_ce.grad)
        else:
            raise ValueError(
                "lm_head_backward_mode must be 'chunked', 'single', "
                "'fused', 'fused_streaming', 'fused_streaming_v2', or "
                "'fused_streaming_cached', or 'fused_norm_streaming_v2', "
                f"got {lm_head_backward_mode!r}"
            )
    if spectral_reg_lambda_dead > 0.0 or spectral_reg_lambda_sticky > 0.0:
        spectral_extra = spectral_regularization_loss(
            model,
            lambda_dead=spectral_reg_lambda_dead,
            lambda_sticky=spectral_reg_lambda_sticky,
            min_a=spectral_reg_min_a,
            max_a=spectral_reg_max_a,
        )
        if spectral_extra is not None:
            spectral_extra.backward()
    if dreamworld_entry is not None and dreamworld_weight > 0.0:
        dreamworld_replay_backward(
            model,
            entry=dreamworld_entry,
            weight=dreamworld_weight,
            lm_head_backward_mode=lm_head_backward_mode,
            lm_head_tile_size=lm_head_tile_size,
            replay_batch_size=dreamworld_replay_batch_size,
            generator=dreamworld_generator,
        )
    if ddp_active and world_size > 1:
        mode = str(grad_allreduce_mode).strip().lower()
        if mode == "bulk":
            allreduce_grads(model, world_size)
        elif mode == "async_param":
            if async_grad_reducer is None:
                raise ValueError(
                    "grad_allreduce_mode='async_param' requires an "
                    "AsyncGradAllReducer instance"
                )
            async_grad_reducer.wait()
        else:
            raise ValueError(
                "grad_allreduce_mode must be 'bulk' or 'async_param', "
                f"got {grad_allreduce_mode!r}"
            )
    return loss


def _state_dict_clone(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _restore_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def _cuda_graph_rejection_reasons(
    *,
    device: torch.device,
    ddp_active: bool,
    activation_checkpoint: bool,
    compile_full_path: bool,
    lm_head_backward_mode: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> list[str]:
    reasons: list[str] = []
    if device.type != "cuda" or not torch.cuda.is_available():
        reasons.append("cuda_required")
    if ddp_active:
        reasons.append("ddp_not_supported")
    if activation_checkpoint:
        reasons.append("activation_checkpoint_not_supported")
    if compile_full_path:
        reasons.append("compile_full_path_not_supported")
    if lm_head_backward_mode not in {
        "fused",
        "fused_streaming",
        "fused_streaming_v2",
        "fused_streaming_cached",
        "fused_norm_streaming_v2",
    }:
        reasons.append("fused_lm_head_required")
    if isinstance(optimizer, ScarcityAwareOptimizer):
        reasons.append("scopt_autograd_grad_not_captured")
    return reasons


def _rejected_cuda_graph_summary(
    *,
    mode: str,
    reasons: list[str],
    budget_seconds: float,
    min_total_speedup: float,
    max_capture_seconds: float,
) -> dict[str, Any]:
    summary = summarize_cuda_graph_gate(
        budget_seconds=budget_seconds,
        capture_seconds=0.0,
        warmup_seconds=0.0,
        warmup_steps=0,
        eager_step_seconds=0.0,
        graph_step_seconds=0.0,
        min_total_speedup=min_total_speedup,
        max_capture_seconds=max_capture_seconds,
    )
    merged_reasons = list(dict.fromkeys([*reasons, *summary["reasons"]]))
    summary.update({
        "mode": mode,
        "accepted": False,
        "reasons": merged_reasons,
    })
    return summary


def _make_batch_fetcher(
    *,
    prefetch_batches: bool,
    train_tokens: torch.Tensor,
    train_num_tokens: int,
    seq_len: int,
    stride: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    generator: torch.Generator,
    vocab_size: int,
) -> tuple[Any, Any]:
    prefetcher = None
    if prefetch_batches:
        prefetcher = Exp23BatchPrefetcher(
            tokens=train_tokens,
            seq_len=seq_len,
            stride=stride,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            device=device,
            generator=generator,
            vocab_size=vocab_size,
        ).prime()

        def next_batch() -> tuple[torch.Tensor, torch.Tensor]:
            return prefetcher.next()

        return next_batch, prefetcher

    def next_batch() -> tuple[torch.Tensor, torch.Tensor]:
        starts = sample_sharded_lm_starts(
            num_tokens=train_num_tokens,
            seq_len=seq_len,
            stride=stride,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            generator=generator,
        )
        return batch_from_start_tensor(
            tokens=train_tokens,
            starts=starts,
            seq_len=seq_len,
            device=device,
            vocab_size=vocab_size,
        )

    return next_batch, None


def _apply_grad_clip(
    *,
    model: torch.nn.Module,
    grad_clip_norm: float,
    fused_grad_clip: bool,
) -> None:
    if grad_clip_norm <= 0.0:
        return
    if fused_grad_clip:
        clip_grad_norm_fused(model.parameters(), grad_clip_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)


def _train_fast_for_budget_cuda_graph(
    *,
    model: torch.nn.Module,
    train_tokens: torch.Tensor,
    train_num_tokens: int,
    stride: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    budget_seconds: float,
    chunk_size: int,
    grad_clip_norm: float,
    fused_grad_clip: bool,
    rank: int,
    world_size: int,
    seed: int,
    precision: str,
    stop_check_interval: int,
    stop_margin_seconds: float,
    vocab_size: int,
    max_steps: int | None,
    compile_full_path: bool,
    prefetch_batches: bool,
    lm_head_backward_mode: str,
    lm_head_tile_size: int,
    cuda_graph_mode: str,
    cuda_graph_min_total_speedup: float,
    cuda_graph_max_capture_seconds: float,
    cuda_graph_warmup_steps: int,
    grad_allreduce_mode: str = "bulk",
) -> dict[str, Any]:
    rank_ = int(rank)
    world_size_ = int(world_size)
    ddp_active = world_size_ > 1
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed) + rank_)
    next_batch, prefetcher = _make_batch_fetcher(
        prefetch_batches=prefetch_batches,
        train_tokens=train_tokens,
        train_num_tokens=train_num_tokens,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        rank=rank_,
        world_size=world_size_,
        device=device,
        generator=rng,
        vocab_size=vocab_size,
    )

    model.train()
    losses: list[torch.Tensor] = []
    steps = 0
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    static_inputs, static_targets = next_batch()
    static_inputs = static_inputs.contiguous()
    static_targets = static_targets.contiguous()
    graph_loss: torch.Tensor | None = None
    graph = torch.cuda.CUDAGraph()

    try:
        warmup_start = time.perf_counter()
        warmup_steps = max(1, int(cuda_graph_warmup_steps))
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        # Per-step wall times, used to estimate steady-state eager cost
        # for the gate. Drops the first-step spike (JIT / algo select /
        # allocator warmup) rather than feeding it into the projection.
        eager_step_times: list[float] = []
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup_steps):
                inputs, targets = next_batch()
                static_inputs.copy_(inputs, non_blocking=True)
                static_targets.copy_(targets, non_blocking=True)
                torch.cuda.synchronize(device)
                step_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=False)
                loss = _run_train_step(
                    model=model,
                    inputs=static_inputs,
                    targets=static_targets,
                    chunk_size=chunk_size,
                    precision=precision,
                    ddp_active=ddp_active,
                    world_size=world_size_,
                    compile_full_path=compile_full_path,
                    lm_head_backward_mode=lm_head_backward_mode,
                    lm_head_tile_size=lm_head_tile_size,
                    grad_allreduce_mode=grad_allreduce_mode,
                )
                _apply_grad_clip(
                    model=model,
                    grad_clip_norm=grad_clip_norm,
                    fused_grad_clip=fused_grad_clip,
                )
                optimizer.step()
                torch.cuda.synchronize(device)
                eager_step_times.append(time.perf_counter() - step_start)
                losses.append(loss.detach())
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start

        capture_inputs, capture_targets = next_batch()
        static_inputs.copy_(capture_inputs, non_blocking=True)
        static_targets.copy_(capture_targets, non_blocking=True)
        torch.cuda.synchronize(device)
        capture_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=False)
        with torch.cuda.graph(graph):
            optimizer.zero_grad(set_to_none=False)
            graph_loss = _run_train_step(
                model=model,
                inputs=static_inputs,
                targets=static_targets,
                chunk_size=chunk_size,
                precision=precision,
                ddp_active=ddp_active,
                world_size=world_size_,
                compile_full_path=compile_full_path,
                lm_head_backward_mode=lm_head_backward_mode,
                lm_head_tile_size=lm_head_tile_size,
                grad_allreduce_mode=grad_allreduce_mode,
            )
            _apply_grad_clip(
                model=model,
                grad_clip_norm=grad_clip_norm,
                fused_grad_clip=fused_grad_clip,
            )
        torch.cuda.synchronize(device)
        capture_seconds = time.perf_counter() - capture_start

        # The captured pass produced real gradients for the capture batch.
        # Keep optimizer.step eager so Muon's scalar Adam counters advance
        # normally rather than freezing inside the graph replay.
        optimizer.step()
        losses.append(graph_loss.detach())

        replay_start = time.perf_counter()
        while True:
            check_interval = max(1, int(stop_check_interval))
            max_steps_reached = max_steps is not None and steps >= int(max_steps)
            if steps == 0 or steps % check_interval == 0 or max_steps_reached:
                elapsed = time.perf_counter() - start_time
                local_stop = should_stop_training_loop(
                    steps=steps,
                    elapsed_s=elapsed,
                    budget_seconds=budget_seconds,
                    stop_margin_seconds=stop_margin_seconds,
                    max_steps=max_steps,
                )
                if should_stop_now(local_stop, device, ddp_active):
                    break

            inputs, targets = next_batch()
            static_inputs.copy_(inputs, non_blocking=True)
            static_targets.copy_(targets, non_blocking=True)
            graph.replay()
            optimizer.step()
            if graph_loss is not None:
                losses.append(graph_loss.detach().clone())
            steps += 1
        torch.cuda.synchronize(device)
        replay_seconds = time.perf_counter() - replay_start
    finally:
        if prefetcher is not None:
            prefetcher.close()

    if ddp_active:
        dist.barrier()

    elapsed_s = time.perf_counter() - start_time
    loss_cpu = torch.stack(losses).cpu() if losses else torch.empty(0)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    timing = summarize_train_timing(
        steps=steps,
        elapsed_s=elapsed_s,
        batch_size=batch_size,
        seq_len=seq_len,
        world_size=world_size_,
    )
    graph_step_seconds = replay_seconds / max(steps, 1)
    # Steady-state eager baseline: median of warmup step times with the
    # first sample dropped. The older ``warmup_seconds / warmup_steps``
    # mean folded first-step overhead (cuBLAS algo selection, JIT,
    # allocator growth) into the projection and caused the gate to
    # accept graphs whose replay was actually slower than a warm eager
    # step at submission scale. See ``steady_state_step_seconds`` in
    # ``fast_path.py`` for the rationale.
    eager_warmup_mean = warmup_seconds / max(warmup_steps, 1)
    eager_step_seconds = (
        steady_state_step_seconds(eager_step_times)
        if eager_step_times
        else eager_warmup_mean
    )
    graph_summary = summarize_cuda_graph_gate(
        budget_seconds=budget_seconds,
        capture_seconds=capture_seconds,
        warmup_seconds=warmup_seconds,
        warmup_steps=warmup_steps,
        eager_step_seconds=eager_step_seconds,
        graph_step_seconds=graph_step_seconds,
        min_total_speedup=cuda_graph_min_total_speedup,
        max_capture_seconds=cuda_graph_max_capture_seconds,
    )
    graph_summary.update({
        "mode": cuda_graph_mode,
        "replay_steps": steps,
        "replay_seconds": replay_seconds,
        "capture_counted_as_training_step": True,
        "optimizer_step_captured": False,
        "eager_step_times": [float(t) for t in eager_step_times],
        "eager_warmup_mean_seconds": float(eager_warmup_mean),
    })
    return {
        "steps": steps,
        "elapsed_s": elapsed_s,
        "rank": rank_,
        "world_size": world_size_,
        "initial_loss": float(loss_cpu[0]) if loss_cpu.numel() else float("nan"),
        "final_loss": float(loss_cpu[-1]) if loss_cpu.numel() else float("nan"),
        "loss_trajectory": [float(x) for x in loss_cpu.tolist()],
        "loss_delta": (
            float(loss_cpu[-1] - loss_cpu[0]) if loss_cpu.numel() >= 2 else float("nan")
        ),
        "peak_vram_mb": peak_vram_mb,
        **timing,
        "cuda_graph": graph_summary,
    }


def train_fast_for_budget(
    model: torch.nn.Module,
    *,
    train_tokens: torch.Tensor,
    train_num_tokens: int,
    stride: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    budget_seconds: float,
    chunk_size: int,
    grad_clip_norm: float,
    fused_grad_clip: bool,
    rank: int,
    world_size: int,
    seed: int,
    precision: str,
    stop_check_interval: int,
    stop_margin_seconds: float,
    vocab_size: int,
    max_steps: int | None = None,
    activation_checkpoint: bool = False,
    compile_full_path: bool = False,
    prefetch_batches: bool = False,
    lm_head_backward_mode: str = "fused",
    lm_head_tile_size: int = 1024,
    cuda_graph_mode: str = "none",
    cuda_graph_min_total_speedup: float = 0.05,
    cuda_graph_max_capture_seconds: float = 30.0,
    cuda_graph_warmup_steps: int = 3,
    grad_allreduce_mode: str = "bulk",
    train_sampling_mode: str = "random",
    fast_slow_enabled: bool = False,
    fast_slow_interval: int = 0,
    fast_slow_alpha: float = 0.0,
    fast_slow_eval_copy: str = "fast",
    spectral_reg_lambda_dead: float = 0.0,
    spectral_reg_lambda_sticky: float = 0.0,
    spectral_reg_min_a: float = 0.05,
    spectral_reg_max_a: float = 0.98,
    embed_freeze_steps: int = 0,
    predictive_aux_weight: float = 0.0,
    predictive_aux_horizon: int = 0,
    predictive_aux_dim: int = 0,
    dreamworld_enabled: bool = False,
    dreamworld_cache_interval: int = 0,
    dreamworld_interval: int = 0,
    dreamworld_weight: float = 0.0,
    dreamworld_prefix_tokens: int = 128,
    dreamworld_replay_tokens: int = 64,
    dreamworld_replay_batch_size: int = 0,
    dreamworld_buffer_size: int = 16,
    dreamworld_min_size: int = 2,
    dreamworld_max_age_steps: int = 256,
    event_sleep_enabled: bool = False,
    event_sleep_loss_ratio: float = 1.10,
    event_sleep_pressure_threshold: float = 0.05,
    event_sleep_ema_decay: float = 0.99,
    event_sleep_warmup_steps: int = 32,
    event_sleep_min_interval: int = 8,
    event_sleep_weight: float = 0.0,
    scopt_split_interval: int = 4,
    scopt_baseline_buckets: int = 16,
    scopt_baseline_decay: float = 0.99,
    scopt_trace_interval_steps: int = 0,
    scopt_pressure_upper_c: float | None = None,
    scopt_pressure_upper_floor: float = 1.0,
) -> dict[str, Any]:
    rank_ = int(rank)
    world_size_ = int(world_size)
    ddp_active = world_size_ > 1
    if ddp_active:
        broadcast_params(model)
    grad_allreduce_mode_ = str(grad_allreduce_mode).strip().lower()
    if grad_allreduce_mode_ not in {"bulk", "async_param"}:
        raise ValueError(
            "grad_allreduce_mode must be 'bulk' or 'async_param', "
            f"got {grad_allreduce_mode!r}"
        )
    scopt_active = isinstance(optimizer, ScarcityAwareOptimizer)
    if scopt_active and grad_allreduce_mode_ != "bulk":
        raise ValueError("ScOpt currently requires grad_allreduce_mode='bulk'")
    if (
        scopt_active
        and int(scopt_baseline_buckets) > 0
        and str(lm_head_backward_mode).strip().lower() not in _FUSED_LM_HEAD_MODES
    ):
        allowed = ", ".join(sorted(_FUSED_LM_HEAD_MODES))
        raise ValueError(
            "ScOpt frequency baseline requires a fused LM-head backward mode "
            "so the common path can update its CE EMA every step; got "
            f"lm_head_backward_mode={lm_head_backward_mode!r}. "
            f"Use one of: {allowed}"
        )
    if scopt_active and (
        spectral_reg_lambda_dead > 0.0
        or spectral_reg_lambda_sticky > 0.0
        or predictive_aux_weight > 0.0
        or dreamworld_weight > 0.0
        or event_sleep_enabled
    ):
        raise ValueError(
            "ScOpt training path currently supports CE-only training; "
            "disable spectral, predictive_aux, and dreamworld mechanisms"
        )
    if event_sleep_enabled and not dreamworld_enabled:
        raise ValueError("event_sleep_enabled requires dreamworld_enabled=True")
    sampling_mode = str(train_sampling_mode).strip().lower()
    if sampling_mode not in {"random", "sequential_epoch", "shuffled_epoch"}:
        raise ValueError(
            "train_sampling_mode must be 'random', 'sequential_epoch', "
            "or 'shuffled_epoch', "
            f"got {train_sampling_mode!r}"
        )
    total_starts = count_lm_starts(train_num_tokens, seq_len, stride)
    rank_start_count = count_sharded_lm_starts(
        total_starts=total_starts,
        rank=rank_,
        world_size=world_size_,
    )
    epoch_steps = None
    if sampling_mode in {"sequential_epoch", "shuffled_epoch"}:
        epoch_steps = sequential_epoch_steps(
            num_tokens=train_num_tokens,
            seq_len=seq_len,
            stride=stride,
            batch_size=batch_size,
            world_size=world_size_,
        )
        if max_steps is None:
            max_steps = epoch_steps

    graph_mode = str(cuda_graph_mode).strip().lower()
    if graph_mode not in {"none", "probe"}:
        raise ValueError(
            "cuda_graph_mode must be 'none' or 'probe', "
            f"got {cuda_graph_mode!r}"
        )
    graph_summary: dict[str, Any] | None = None
    if graph_mode != "none":
        graph_rejections = _cuda_graph_rejection_reasons(
            device=device,
            ddp_active=ddp_active,
            activation_checkpoint=activation_checkpoint,
            compile_full_path=compile_full_path,
            lm_head_backward_mode=lm_head_backward_mode,
            optimizer=optimizer,
        )
        if sampling_mode != "random":
            graph_rejections.append("train_sampling_mode_not_supported")
        if graph_rejections:
            graph_summary = _rejected_cuda_graph_summary(
                mode=graph_mode,
                reasons=graph_rejections,
                budget_seconds=budget_seconds,
                min_total_speedup=cuda_graph_min_total_speedup,
                max_capture_seconds=cuda_graph_max_capture_seconds,
            )
        else:
            try:
                return _train_fast_for_budget_cuda_graph(
                    model=model,
                    train_tokens=train_tokens,
                    train_num_tokens=train_num_tokens,
                    stride=stride,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    device=device,
                    optimizer=optimizer,
                    budget_seconds=budget_seconds,
                    chunk_size=chunk_size,
                    grad_clip_norm=grad_clip_norm,
                    fused_grad_clip=fused_grad_clip,
                    rank=rank_,
                    world_size=world_size_,
                    seed=seed,
                    precision=precision,
                    stop_check_interval=stop_check_interval,
                    stop_margin_seconds=stop_margin_seconds,
                    vocab_size=vocab_size,
                    max_steps=max_steps,
                    compile_full_path=compile_full_path,
                    prefetch_batches=prefetch_batches,
                    lm_head_backward_mode=lm_head_backward_mode,
                    lm_head_tile_size=lm_head_tile_size,
                    cuda_graph_mode=graph_mode,
                    cuda_graph_min_total_speedup=cuda_graph_min_total_speedup,
                    cuda_graph_max_capture_seconds=cuda_graph_max_capture_seconds,
                    cuda_graph_warmup_steps=cuda_graph_warmup_steps,
                    grad_allreduce_mode=grad_allreduce_mode_,
                )
            except RuntimeError as exc:
                graph_summary = _rejected_cuda_graph_summary(
                    mode=graph_mode,
                    reasons=["capture_failed"],
                    budget_seconds=budget_seconds,
                    min_total_speedup=cuda_graph_min_total_speedup,
                    max_capture_seconds=cuda_graph_max_capture_seconds,
                )
                graph_summary["error"] = str(exc)[:500]

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed) + rank_)

    model.train()
    fast_slow = FastSlowConsolidator.from_config(
        model,
        {
            "fast_slow_enabled": fast_slow_enabled,
            "fast_slow_interval": fast_slow_interval,
            "fast_slow_alpha": fast_slow_alpha,
        },
    )
    predictive_aux_projection = None
    predictive_aux_optimizer = None
    if predictive_aux_weight > 0.0 and predictive_aux_horizon > 0:
        dim = int(getattr(model, "dim", 0) or getattr(model.lm_head, "in_features"))
        aux_dim = int(predictive_aux_dim) if int(predictive_aux_dim) > 0 else dim
        if aux_dim != dim:
            raise ValueError("predictive_aux_dim must be 0 or equal to model dim in v1")
        predictive_aux_projection = torch.nn.Linear(dim, dim, bias=False).to(
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        predictive_aux_optimizer = torch.optim.AdamW(
            predictive_aux_projection.parameters(),
            lr=float(getattr(optimizer, "param_groups", [{"lr": 0.0}])[0]["lr"]),
            weight_decay=0.0,
        )
        if ddp_active:
            broadcast_params(predictive_aux_projection)
    dream_buffer = (
        DreamReplayBuffer(
            max_entries=dreamworld_buffer_size,
            max_age_steps=dreamworld_max_age_steps,
        )
        if dreamworld_enabled
        else None
    )
    event_sleep_gate = (
        LossTriggeredReplayEMA(
            decay=float(event_sleep_ema_decay),
            warmup_steps=int(event_sleep_warmup_steps),
        )
        if event_sleep_enabled
        else None
    )
    event_sleep_pending = False
    event_sleep_last_replay_step = -10**9
    event_sleep_trigger_count = 0
    event_sleep_replay_count = 0
    event_sleep_decision_count = 0
    event_sleep_pressure_sum = 0.0
    event_sleep_last_decision: dict[str, Any] | None = None
    event_sleep_queued_decision: LossTriggeredReplayDecision | None = None
    event_sleep_queued_step: int | None = None
    spectral_before = spectral_summary(model)
    losses: list[torch.Tensor] = []
    steps = 0
    start_time = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    scopt_token_frequencies = None
    scopt_baseline: FrequencyBucketBaseline | None = None
    scopt_trace_history: list[dict[str, Any]] = []
    if scopt_active:
        token_counts = torch.bincount(
            train_tokens[:train_num_tokens].detach().cpu().long(),
            minlength=int(vocab_size),
        ).clamp_min(1)
        scopt_token_frequencies = token_counts.to(device=device, dtype=torch.float32)
        # Frequency-bucket running-mean CE — the fallback baseline the
        # design spec line 197 names. Attention-based baseline is a
        # separate artifact; this keeps the optimizer runnable with a
        # principled baseline until that artifact is trained.
        scopt_baseline = FrequencyBucketBaseline(
            scopt_token_frequencies,
            num_buckets=int(scopt_baseline_buckets),
            decay=float(scopt_baseline_decay),
            device=device,
        )

    prefetcher = None
    async_grad_reducer = (
        AsyncGradAllReducer(model, world_size_)
        if ddp_active and grad_allreduce_mode_ == "async_param"
        else None
    )
    if sampling_mode == "sequential_epoch":
        batch_sampler = SequentialShardedStartSampler()
    elif sampling_mode == "shuffled_epoch":
        batch_sampler = ShuffledEpochShardedStartSampler(seed=seed)
    else:
        batch_sampler = None
    if prefetch_batches:
        prefetcher = Exp23BatchPrefetcher(
            tokens=train_tokens,
            seq_len=seq_len,
            stride=stride,
            batch_size=batch_size,
            rank=rank_,
            world_size=world_size_,
            device=device,
            generator=rng,
            vocab_size=vocab_size,
            batch_sampler=batch_sampler,
        )

    try:
        while True:
            # Resolve the previous loss decision at the step boundary so
            # ``update`` can leave CUDA loss-pressure math unmaterialized.
            if event_sleep_queued_decision is not None:
                decision = event_sleep_queued_decision
                decision_step = (
                    steps
                    if event_sleep_queued_step is None
                    else event_sleep_queued_step
                )
                event_sleep_queued_decision = None
                event_sleep_queued_step = None

                event_sleep_decision_count += 1
                event_sleep_pressure_sum += decision.global_pressure
                event_sleep_last_decision = {
                    "local_loss": decision.local_loss,
                    "ema_loss": decision.ema_loss,
                    "local_ratio": decision.local_ratio,
                    "local_pressure": decision.local_pressure,
                    "global_pressure": decision.global_pressure,
                    "fire_count": decision.fire_count,
                    "local_fire": decision.local_fire,
                    "triggered": decision.triggered,
                }
                interval_ready = (
                    decision_step - event_sleep_last_replay_step
                    >= int(event_sleep_min_interval)
                )
                buffer_ready = (
                    dream_buffer is not None
                    and len(dream_buffer) >= int(dreamworld_min_size)
                )
                if decision.triggered and interval_ready and buffer_ready:
                    event_sleep_trigger_count += 1
                    event_sleep_pending = True

            check_interval = max(1, int(stop_check_interval))
            max_steps_reached = max_steps is not None and steps >= int(max_steps)
            if steps == 0 or steps % check_interval == 0 or max_steps_reached:
                elapsed = time.perf_counter() - start_time
                local_stop = should_stop_training_loop(
                    steps=steps,
                    elapsed_s=elapsed,
                    budget_seconds=budget_seconds,
                    stop_margin_seconds=stop_margin_seconds,
                    max_steps=max_steps,
                )
                if should_stop_now(local_stop, device, ddp_active):
                    break

            if prefetcher is not None:
                if steps == 0:
                    prefetcher.prime()
                inputs, targets = prefetcher.next()
            else:
                if sampling_mode == "sequential_epoch":
                    starts = sequential_sharded_lm_starts(
                        num_tokens=train_num_tokens,
                        seq_len=seq_len,
                        stride=stride,
                        batch_size=batch_size,
                        rank=rank_,
                        world_size=world_size_,
                        step=steps,
                    )
                elif sampling_mode == "shuffled_epoch":
                    starts = shuffled_epoch_sharded_lm_starts(
                        num_tokens=train_num_tokens,
                        seq_len=seq_len,
                        stride=stride,
                        batch_size=batch_size,
                        rank=rank_,
                        world_size=world_size_,
                        step=steps,
                        seed=seed,
                    )
                else:
                    starts = sample_sharded_lm_starts(
                        num_tokens=train_num_tokens,
                        seq_len=seq_len,
                        stride=stride,
                        batch_size=batch_size,
                        rank=rank_,
                        world_size=world_size_,
                        generator=rng,
                    )
                inputs, targets = batch_from_start_tensor(
                    tokens=train_tokens,
                    starts=starts,
                    seq_len=seq_len,
                    device=device,
                    vocab_size=vocab_size,
                )

            dream_entry = None
            scheduled_dream_replay = (
                dream_buffer is not None
                and dreamworld_interval > 0
                and dreamworld_weight > 0.0
                and len(dream_buffer) >= int(dreamworld_min_size)
                and steps % dreamworld_interval == 0
            )
            event_dream_replay = (
                event_sleep_pending
                and dream_buffer is not None
                and len(dream_buffer) >= int(dreamworld_min_size)
            )
            if scheduled_dream_replay or event_dream_replay:
                dream_entry = dream_buffer.sample(generator=rng, current_step=steps)
                if event_dream_replay:
                    event_sleep_replay_count += 1
                    event_sleep_pending = False
                    event_sleep_last_replay_step = steps

            if (
                dream_buffer is not None
                and dreamworld_cache_interval > 0
                and steps % dreamworld_cache_interval == 0
            ):
                entry = capture_dream_entry(
                    model,
                    inputs,
                    step=steps,
                    prefix_tokens=dreamworld_prefix_tokens,
                    replay_tokens=dreamworld_replay_tokens,
                )
                dream_buffer.add(
                    step=entry.step,
                    states=entry.states,
                    replay_tokens=entry.replay_tokens,
                )

            optimizer.zero_grad(set_to_none=True)
            if predictive_aux_optimizer is not None:
                predictive_aux_optimizer.zero_grad(set_to_none=True)
            if async_grad_reducer is not None:
                async_grad_reducer.reset()
            scopt_pending: _ScOptPending | None = None
            if scopt_active:
                assert scopt_token_frequencies is not None
                assert isinstance(optimizer, ScarcityAwareOptimizer)
                if _should_run_scopt_split_step(
                    optimizer,
                    step=steps,
                    split_interval=scopt_split_interval,
                ):
                    loss, scopt_pending = _run_scopt_train_step(
                        model=model,
                        optimizer=optimizer,
                        inputs=inputs,
                        targets=targets,
                        token_frequencies=scopt_token_frequencies,
                        precision=precision,
                        ddp_active=ddp_active,
                        world_size=world_size_,
                        step=steps,
                        split_interval=scopt_split_interval,
                        baseline=scopt_baseline,
                        pressure_upper_c=scopt_pressure_upper_c,
                        pressure_upper_floor=scopt_pressure_upper_floor,
                        lm_head_backward_mode=lm_head_backward_mode,
                        lm_head_tile_size=lm_head_tile_size,
                    )
                else:
                    loss = _run_scopt_common_train_step(
                        model=model,
                        inputs=inputs,
                        targets=targets,
                        precision=precision,
                        ddp_active=ddp_active,
                        world_size=world_size_,
                        chunk_size=chunk_size,
                        compile_full_path=compile_full_path,
                        lm_head_backward_mode=lm_head_backward_mode,
                        lm_head_tile_size=lm_head_tile_size,
                        grad_allreduce_mode=grad_allreduce_mode_,
                        baseline=scopt_baseline,
                    )
            else:
                loss = _run_train_step(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    chunk_size=chunk_size,
                    precision=precision,
                    ddp_active=ddp_active,
                    world_size=world_size_,
                    compile_full_path=compile_full_path,
                    lm_head_backward_mode=lm_head_backward_mode,
                    lm_head_tile_size=lm_head_tile_size,
                    grad_allreduce_mode=grad_allreduce_mode_,
                    async_grad_reducer=async_grad_reducer,
                    spectral_reg_lambda_dead=spectral_reg_lambda_dead,
                    spectral_reg_lambda_sticky=spectral_reg_lambda_sticky,
                    spectral_reg_min_a=spectral_reg_min_a,
                    spectral_reg_max_a=spectral_reg_max_a,
                    predictive_aux_weight=predictive_aux_weight,
                    predictive_aux_horizon=predictive_aux_horizon,
                    predictive_aux_projection=predictive_aux_projection,
                    dreamworld_entry=dream_entry,
                    dreamworld_weight=(
                        float(event_sleep_weight)
                        if event_dream_replay and float(event_sleep_weight) > 0.0
                        else dreamworld_weight
                    ),
                    dreamworld_replay_batch_size=dreamworld_replay_batch_size,
                    dreamworld_generator=rng,
                )
            if event_sleep_gate is not None:
                decision = event_sleep_gate.update(
                    loss,
                    threshold=float(event_sleep_loss_ratio),
                    pressure_threshold=float(event_sleep_pressure_threshold),
                    ddp_active=ddp_active,
                    world_size=world_size_,
                    device=device,
                )
                if decision is not None:
                    event_sleep_queued_decision = decision
                    event_sleep_queued_step = steps
            if ddp_active and predictive_aux_projection is not None:
                allreduce_grads(predictive_aux_projection, world_size_)
            zero_embedding_grad_until(
                model,
                step=steps,
                freeze_steps=embed_freeze_steps,
            )
            clipped_this_step = False
            if grad_clip_norm > 0.0:
                if fused_grad_clip:
                    total_norm = clip_grad_norm_fused(
                        model.parameters(), grad_clip_norm
                    )
                else:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_norm
                    )
                # One GPU->CPU sync per step on the ScOpt correctness path
                # is acceptable given it already runs outside CUDA graphs
                # and does several collectives; skip it when we don't need
                # the decision.
                if scopt_active:
                    clipped_this_step = bool(
                        float(total_norm) > float(grad_clip_norm)
                    )
            if scopt_active:
                scopt_opt = optimizer
                assert isinstance(scopt_opt, ScarcityAwareOptimizer)
                scopt_opt.record_clip_event(triggered=clipped_this_step)
                _apply_scopt_pending(
                    scopt_opt, scopt_pending, skip=clipped_this_step
                )
            optimizer.step()
            if predictive_aux_optimizer is not None:
                predictive_aux_optimizer.step()
            fast_slow.after_optimizer_step(model, step=steps + 1)
            losses.append(loss.detach())
            steps += 1
            if (
                scopt_active
                and scopt_trace_interval_steps > 0
                and steps % int(scopt_trace_interval_steps) == 0
            ):
                assert isinstance(optimizer, ScarcityAwareOptimizer)
                scopt_trace_history.append(optimizer.scarcity_trace())
    finally:
        if prefetcher is not None:
            prefetcher.close()
        if async_grad_reducer is not None:
            async_grad_reducer.close()

    if ddp_active:
        dist.barrier()

    if fast_slow.enabled and str(fast_slow_eval_copy).strip().lower() == "slow":
        fast_slow.copy_slow_to_model(model)

    elapsed_s = time.perf_counter() - start_time
    loss_cpu = torch.stack(losses).cpu() if losses else torch.empty(0)
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    timing = summarize_train_timing(
        steps=steps,
        elapsed_s=elapsed_s,
        batch_size=batch_size,
        seq_len=seq_len,
        world_size=world_size_,
    )
    result = {
        "steps": steps,
        "elapsed_s": elapsed_s,
        "rank": rank_,
        "world_size": world_size_,
        "initial_loss": float(loss_cpu[0]) if loss_cpu.numel() else float("nan"),
        "final_loss": float(loss_cpu[-1]) if loss_cpu.numel() else float("nan"),
        "loss_trajectory": [float(x) for x in loss_cpu.tolist()],
        "loss_delta": (
            float(loss_cpu[-1] - loss_cpu[0]) if loss_cpu.numel() >= 2 else float("nan")
        ),
        "peak_vram_mb": peak_vram_mb,
        "optimizer": _optimizer_diagnostics(optimizer),
        "scopt_trace_history": scopt_trace_history if scopt_active else None,
        "sampling_mode": sampling_mode,
        "total_start_count": total_starts,
        "rank_start_count": rank_start_count,
        "epoch_steps": epoch_steps,
        "unique_start_count": (
            min(total_starts, steps * int(batch_size) * world_size_)
            if sampling_mode in {"sequential_epoch", "shuffled_epoch"}
            else None
        ),
        "epoch_complete": (
            steps >= int(epoch_steps)
            if epoch_steps is not None
            else None
        ),
        "mechanisms": {
            "fast_slow": fast_slow.diagnostics(model),
            "spectral": {
                "enabled": (
                    spectral_reg_lambda_dead > 0.0
                    or spectral_reg_lambda_sticky > 0.0
                ),
                "before": spectral_before,
                "after": spectral_summary(model),
            },
            "embed_freeze": {
                "freeze_steps": int(embed_freeze_steps),
                "enabled": int(embed_freeze_steps) > 0,
            },
            "predictive_aux": {
                "enabled": predictive_aux_projection is not None,
                "weight": float(predictive_aux_weight),
                "horizon": int(predictive_aux_horizon),
                "artifact_impact": "artifact_training_only",
            },
            "dreamworld": {
                "enabled": dream_buffer is not None,
                "weight": float(dreamworld_weight),
                "cache_interval": int(dreamworld_cache_interval),
                "dream_interval": int(dreamworld_interval),
                "prefix_tokens": int(dreamworld_prefix_tokens),
                "replay_tokens": int(dreamworld_replay_tokens),
                "replay_batch_size": int(dreamworld_replay_batch_size),
                "artifact_impact": "artifact_training_only",
                "buffer": (
                    dream_buffer.diagnostics(current_step=steps)
                    if dream_buffer is not None
                    else None
                ),
            },
            "event_sleep": {
                "enabled": event_sleep_gate is not None,
                "loss_ratio": float(event_sleep_loss_ratio),
                "pressure_threshold": float(event_sleep_pressure_threshold),
                "ema_decay": float(event_sleep_ema_decay),
                "warmup_steps": int(event_sleep_warmup_steps),
                "min_interval": int(event_sleep_min_interval),
                "weight": (
                    float(event_sleep_weight)
                    if float(event_sleep_weight) > 0.0
                    else float(dreamworld_weight)
                ),
                "trigger_count": int(event_sleep_trigger_count),
                "replay_count": int(event_sleep_replay_count),
                "decision_count": int(event_sleep_decision_count),
                "mean_global_pressure": (
                    event_sleep_pressure_sum / event_sleep_decision_count
                    if event_sleep_decision_count
                    else 0.0
                ),
                "pending": bool(event_sleep_pending),
                "last_decision": event_sleep_last_decision,
                "artifact_impact": "artifact_training_only",
            },
        },
        **timing,
    }
    if graph_summary is not None:
        graph_summary["warmup_steps"] = int(cuda_graph_warmup_steps)
        result["cuda_graph"] = graph_summary
    return result


def _warmup(
    *,
    model: torch.nn.Module,
    train_tokens: torch.Tensor,
    train_num_tokens: int,
    stride: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    rank: int,
    world_size: int,
    seed: int,
    vocab_size: int,
) -> None:
    steps = int(config.get("warmup_steps", 0))
    if steps <= 0:
        return
    train_fast_for_budget(
        model,
        train_tokens=train_tokens,
        train_num_tokens=train_num_tokens,
        stride=stride,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=int(config.get("chunk_size", 64)),
        grad_clip_norm=float(config.get("grad_clip_norm", 1.0)),
        fused_grad_clip=bool(config.get("fused_grad_clip", True)),
        rank=rank,
        world_size=world_size,
        seed=seed,
        precision=str(config.get("precision", "bf16")),
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=vocab_size,
        max_steps=steps,
        compile_full_path=bool(config.get("compile_full_path", False)),
        prefetch_batches=bool(config.get("prefetch_batches", True)),
        lm_head_backward_mode=str(config.get("lm_head_backward_mode", "fused")),
        lm_head_tile_size=int(config.get("lm_head_tile_size", 1024)),
        cuda_graph_mode="none",
        scopt_split_interval=int(config.get("scopt_split_interval", 4)),
        scopt_baseline_buckets=int(config.get("scopt_baseline_buckets", 16)),
        scopt_baseline_decay=float(config.get("scopt_baseline_decay", 0.99)),
        scopt_trace_interval_steps=int(config.get("scopt_trace_interval_steps", 0)),
        scopt_pressure_upper_c=(
            float(config["scopt_pressure_upper_c"])
            if config.get("scopt_pressure_upper_c") is not None
            else None
        ),
        scopt_pressure_upper_floor=float(
            config.get("scopt_pressure_upper_floor", 1.0)
        ),
    )


def run_condition(
    config: dict[str, Any],
    *,
    data_path: str,
    sp_model_path: str,
    budget_seconds: float,
    output_json: str | None,
    output_ckpt: str | None,
    world_size_override: int | None,
) -> dict[str, Any]:
    rank, world_size, local_rank = _init_distributed(world_size_override)
    is_rank0 = rank == 0
    ddp_active = world_size > 1
    device = _pick_device(local_rank, str(config.get("device", "auto")))
    dtype = resolve_param_dtype(str(config.get("dtype", "bf16")), device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    seed = int(config.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    verify_diag_recurrence(device)

    vocab_size = int(config["vocab_size"])
    train_tokens, val_tokens = load_fineweb_tokens(data_path)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, vocab_size, device)
    )

    seq_len = int(config["seq_len"])
    stride = int(config.get("stride", seq_len // 2))
    batch_size = int(config["batch_size"])
    eval_batches = int(config.get("eval_batches", 0))
    eval_starts = choose_lm_starts_lazy(
        num_tokens=int(val_tokens.numel()),
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        eval_batches=max(0, eval_batches),
        seed=seed,
    )

    model = build_model(config, device, dtype)
    _apply_embed_init(model, config, device)
    _reject_unsupported(model)
    model_params = sum(p.numel() for p in model.parameters())

    if is_rank0:
        print(
            f"[rank 0/{world_size}] {config.get('name', '<unnamed>')} "
            f"vocab={config['vocab_size']} batch={batch_size} "
            f"chunk={config.get('chunk_size', 64)} "
            f"ckpt={bool(config.get('activation_checkpoint', False))} "
            f"params={model_params:,}",
            flush=True,
        )

    saved_state = None
    if bool(config.get("restore_after_warmup", False)) and int(config.get("warmup_steps", 0)) > 0:
        saved_state = _state_dict_clone(model)
    optimizer = _build_optimizer(config, model)
    _warmup(
        model=model,
        train_tokens=train_tokens,
        train_num_tokens=int(train_tokens.numel()),
        stride=stride,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        optimizer=optimizer,
        config=config,
        rank=rank,
        world_size=world_size,
        seed=seed,
        vocab_size=vocab_size,
    )
    if saved_state is not None:
        _restore_state_dict(model, saved_state)
        optimizer = _build_optimizer(config, model)
        if ddp_active:
            dist.barrier()

    train_result = train_fast_for_budget(
        model,
        train_tokens=train_tokens,
        train_num_tokens=int(train_tokens.numel()),
        stride=stride,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        optimizer=optimizer,
        budget_seconds=budget_seconds,
        max_steps=(
            None
            if config.get("max_steps") is None
            else int(config["max_steps"])
        ),
        chunk_size=int(config.get("chunk_size", 64)),
        grad_clip_norm=float(config.get("grad_clip_norm", 1.0)),
        fused_grad_clip=bool(config.get("fused_grad_clip", True)),
        rank=rank,
        world_size=world_size,
        seed=seed,
        precision=str(config.get("precision", "bf16")),
        stop_check_interval=int(config.get("stop_check_interval", 4)),
        stop_margin_seconds=float(config.get("stop_margin_seconds", 2.0)),
        vocab_size=vocab_size,
        activation_checkpoint=bool(config.get("activation_checkpoint", False)),
        compile_full_path=bool(config.get("compile_full_path", False)),
        prefetch_batches=bool(config.get("prefetch_batches", True)),
        lm_head_backward_mode=str(config.get("lm_head_backward_mode", "fused")),
        lm_head_tile_size=int(config.get("lm_head_tile_size", 1024)),
        cuda_graph_mode=str(config.get("cuda_graph_mode", "none")),
        cuda_graph_min_total_speedup=float(
            config.get("cuda_graph_min_total_speedup", 0.05)
        ),
        cuda_graph_max_capture_seconds=float(
            config.get("cuda_graph_max_capture_seconds", 30.0)
        ),
        cuda_graph_warmup_steps=int(config.get("cuda_graph_warmup_steps", 3)),
        grad_allreduce_mode=str(config.get("grad_allreduce_mode", "bulk")),
        train_sampling_mode=str(config.get("train_sampling_mode", "random")),
        fast_slow_enabled=bool(config.get("fast_slow_enabled", False)),
        fast_slow_interval=int(config.get("fast_slow_interval", 0)),
        fast_slow_alpha=float(config.get("fast_slow_alpha", 0.0)),
        fast_slow_eval_copy=str(config.get("fast_slow_eval_copy", "fast")),
        spectral_reg_lambda_dead=float(config.get("spectral_reg_lambda_dead", 0.0)),
        spectral_reg_lambda_sticky=float(
            config.get("spectral_reg_lambda_sticky", 0.0)
        ),
        spectral_reg_min_a=float(config.get("spectral_reg_min_a", 0.05)),
        spectral_reg_max_a=float(config.get("spectral_reg_max_a", 0.98)),
        embed_freeze_steps=int(config.get("embed_freeze_steps", 0)),
        predictive_aux_weight=float(config.get("predictive_aux_weight", 0.0)),
        predictive_aux_horizon=int(config.get("predictive_aux_horizon", 0)),
        predictive_aux_dim=int(config.get("predictive_aux_dim", 0)),
        dreamworld_enabled=bool(config.get("dreamworld_enabled", False)),
        dreamworld_cache_interval=int(config.get("dreamworld_cache_interval", 0)),
        dreamworld_interval=int(config.get("dreamworld_interval", 0)),
        dreamworld_weight=float(config.get("dreamworld_weight", 0.0)),
        dreamworld_prefix_tokens=int(config.get("dreamworld_prefix_tokens", 128)),
        dreamworld_replay_tokens=int(config.get("dreamworld_replay_tokens", 64)),
        dreamworld_replay_batch_size=int(
            config.get("dreamworld_replay_batch_size", 0)
        ),
        dreamworld_buffer_size=int(config.get("dreamworld_buffer_size", 16)),
        dreamworld_min_size=int(config.get("dreamworld_min_size", 2)),
        dreamworld_max_age_steps=int(config.get("dreamworld_max_age_steps", 256)),
        event_sleep_enabled=bool(config.get("event_sleep_enabled", False)),
        event_sleep_loss_ratio=float(config.get("event_sleep_loss_ratio", 1.10)),
        event_sleep_pressure_threshold=float(
            config.get("event_sleep_pressure_threshold", 0.05)
        ),
        event_sleep_ema_decay=float(config.get("event_sleep_ema_decay", 0.99)),
        event_sleep_warmup_steps=int(config.get("event_sleep_warmup_steps", 32)),
        event_sleep_min_interval=int(config.get("event_sleep_min_interval", 8)),
        event_sleep_weight=float(config.get("event_sleep_weight", 0.0)),
        scopt_split_interval=int(config.get("scopt_split_interval", 4)),
        scopt_baseline_buckets=int(config.get("scopt_baseline_buckets", 16)),
        scopt_baseline_decay=float(config.get("scopt_baseline_decay", 0.99)),
        scopt_trace_interval_steps=int(config.get("scopt_trace_interval_steps", 0)),
        scopt_pressure_upper_c=(
            float(config["scopt_pressure_upper_c"])
            if config.get("scopt_pressure_upper_c") is not None
            else None
        ),
        scopt_pressure_upper_floor=float(
            config.get("scopt_pressure_upper_floor", 1.0)
        ),
    )

    if ddp_active:
        dist.barrier()

    eval_result: dict[str, Any] = {}
    if is_rank0 and eval_batches > 0:
        eval_result = evaluate_bpb_sp(
            model,
            tokens=val_tokens,
            eval_starts=eval_starts,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )

    if ddp_active:
        dist.barrier()

    artifact = {
        "artifact_impact": str(
            config.get("artifact_impact", "artifact_changes_weights_only")
        ),
        "submit_valid": bool(config.get("submit_valid", True)),
        "artifact_bytes_estimate": (
            int(model.artifact_bytes())
            if hasattr(model, "artifact_bytes")
            else int(model_params * 2)
        ),
        "compressed_artifact_bytes": config.get("compressed_artifact_bytes"),
    }
    exp24 = {
        "phase": config.get("exp24_phase"),
        "mechanism": config.get("exp24_mechanism"),
    }
    result = {
        "config": config,
        "params": model_params,
        "train": train_result,
        "eval": eval_result,
        "artifact": artifact,
        "exp24": exp24,
    }

    if is_rank0:
        violations: list[str] = []
        for key in ("initial_loss", "final_loss"):
            if not math.isfinite(float(train_result.get(key, float("nan")))):
                violations.append(f"train.{key} is not finite")
        if eval_result:
            for key in ("bpb", "loss"):
                if key in eval_result and not math.isfinite(float(eval_result[key])):
                    violations.append(f"eval.{key} is not finite")
        if violations:
            raise RuntimeError("; ".join(violations))
        if output_json:
            out = Path(output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2, default=str))
            tmp.rename(out)
        if output_ckpt:
            _save_output_ckpt(output_ckpt, model, config)
        print(
            f"[rank 0/{world_size}] done steps={train_result['steps']} "
            f"tok/s={train_result['aggregate_tokens_per_sec']:.0f} "
            f"loss={train_result['final_loss']:.4f} "
            f"vram={train_result['peak_vram_mb']:.1f}MB",
            flush=True,
        )

    if ddp_active and dist.is_initialized():
        dist.destroy_process_group()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exp 23 fastest-path runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-ckpt", default=None)
    parser.add_argument("--world-size", type=int, default=None)
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    budget = (
        float(args.budget)
        if args.budget is not None
        else float(config.get("budget_seconds", 90.0))
    )
    run_condition(
        config,
        data_path=args.data_path,
        sp_model_path=args.sp_model_path,
        budget_seconds=budget,
        output_json=args.output_json,
        output_ckpt=args.output_ckpt,
        world_size_override=args.world_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
