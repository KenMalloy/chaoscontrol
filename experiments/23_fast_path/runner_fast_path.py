#!/usr/bin/env python3
"""Exp 23 single-condition fastest-path DDP runner.

This is deliberately narrower than the previous experiment launchers.
It keeps only the final bare-SSM training path and makes the 600s hot
loop explicit: vectorized batch gather, fused linear+CE head/loss,
fused Muon/grad-clip knobs, amortized stop checks, and compact timing JSON.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import threading
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
sys.path.insert(0, str(REPO))
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
from chaoscontrol.episodic.controller import controller_main  # noqa: E402
from chaoscontrol.episodic.query import query_topk  # noqa: E402
from chaoscontrol.episodic.gpu_slot import (  # noqa: E402
    make_slot_tensor,
    pack_payload,
    unpack_payload,
)
from chaoscontrol.kernels import _cpu_ssm_controller as _ext  # noqa: E402
from chaoscontrol.optim.episodic_cache import EpisodicCache  # noqa: E402
from chaoscontrol.optim.episodic_writer import (  # noqa: E402
    _next_admission_trace_seq,
    build_write_event_dict,
    fingerprint_tokens,
    select_top_p_positions,
    tensor_fp16_to_u16_wire,
)
from chaoscontrol.optim.lamb import LAMB  # noqa: E402
from chaoscontrol.optim.muon import Muon  # noqa: E402
from chaoscontrol.optim.param_groups import build_optimizer_params  # noqa: E402
from chaoscontrol.optim.scopt import (  # noqa: E402
    FrequencyBucketBaseline,
    ScarcityAwareOptimizer,
    scarcity_pressure_from_ce,
    scopt_allreduce_config,
)
from chaoscontrol.optim.criticality import CriticalityDistillation  # noqa: E402
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
    dreamworld_replay_from_cache_entry,
)
from chaoscontrol.episodic.diagnostics import DiagnosticsLogger  # noqa: E402
from experiments._23_fast_path_runner_helpers import (  # noqa: E402
    _alloc_pinned_evidence_buffers,
    compute_ce_minus_entropy_pressure_from_fused,
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


def _capture_topology_snapshot() -> dict:
    """Best-effort snapshot of CPU, GPU-interconnect, and NUMA info.
    Each subprocess call is wrapped in try/except so missing binaries
    don't break training startup."""
    import subprocess

    def _run(cmd: list[str], timeout: float = 2.0) -> str | None:
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False,
            )
            if out.returncode != 0:
                return None
            return out.stdout
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
            return None

    snap: dict = {}

    # CPU info.
    lscpu = _run(["lscpu"])
    if lscpu is not None:
        snap["lscpu"] = lscpu
    else:
        # macOS fallback — sysctl.
        sysctl = _run(["sysctl", "-a"])
        if sysctl is not None:
            snap["cpu_info"] = sysctl
        else:
            snap["cpu_unavailable"] = True

    # GPU topology.
    gpu = _run(["nvidia-smi", "topo", "-m"])
    if gpu is not None:
        snap["nvidia_smi_topo"] = gpu
    else:
        snap["gpu_topo_unavailable"] = True

    # NUMA.
    numa = _run(["numactl", "-H"])
    if numa is not None:
        snap["numactl_h"] = numa
    else:
        snap["numa_unavailable"] = True

    return snap


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


def _build_token_buckets(
    token_frequencies: torch.Tensor, num_buckets: int,
) -> torch.Tensor:
    """Return a `[vocab_size]` int64 tensor assigning each token to a
    log-frequency bucket.

    Bucket 0 is rarest (lowest log-freq); bucket ``num_buckets - 1`` is
    most frequent. Matches the binning math used by
    ``FrequencyBucketBaseline`` (log1p with clamp_min(0.0), then a
    span-anchored linspace) so bucket ids are directly comparable
    between scopt and the val-time diagnostic.
    """
    if token_frequencies.ndim != 1:
        raise ValueError("token_frequencies must be 1D")
    if num_buckets < 1:
        raise ValueError(f"num_buckets must be >= 1, got {num_buckets}")
    log_freq = torch.log1p(
        token_frequencies.to(dtype=torch.float32).clamp_min(0.0)
    )
    min_lf = float(log_freq.min().item())
    max_lf = float(log_freq.max().item())
    span = max(max_lf - min_lf, 1e-6)
    edges = torch.linspace(
        min_lf,
        min_lf + span + 1e-6,
        num_buckets + 1,
        device=log_freq.device,
        dtype=torch.float32,
    )
    bucket = torch.bucketize(log_freq, edges[1:-1])
    bucket.clamp_(0, num_buckets - 1)
    return bucket.to(dtype=torch.int64)


def _compute_per_bucket_val_ce(
    *,
    model: torch.nn.Module,
    device: torch.device,
    tokens: torch.Tensor,
    num_tokens: int,
    seq_len: int,
    stride: int,
    batch_size: int,
    vocab_size: int,
    token_frequencies: torch.Tensor | None,
    num_buckets: int,
    rank: int,
    world_size: int,
    precision: str,
) -> dict[str, Any]:
    """Deterministic no_grad val pass: sum per-token CE into log-freq buckets.

    Runs over sequential windows of ``tokens`` on this rank's shard
    (rank/world_size stride), mean-reduces per bucket, and returns a
    result-dict fragment with four keys:
      * ``per_bucket_val_ce`` (list[float], length=num_buckets;
        bucket 0 is rarest)
      * ``rare_bucket_val_ce`` (float, mean of first
        ``max(1, num_buckets // 4)`` buckets)
      * ``val_bucket_num_buckets`` (int)
      * ``val_bucket_token_counts`` (list[int])
    """
    if token_frequencies is None:
        raise ValueError(
            "rare_bucket_ce_enabled=True requires rare_bucket_ce_token_frequencies"
        )
    token_bucket = _build_token_buckets(token_frequencies, num_buckets).to(device)
    bucket_sum = torch.zeros(num_buckets, dtype=torch.float64, device=device)
    bucket_count = torch.zeros(num_buckets, dtype=torch.int64, device=device)
    # Per-window (per inner-step) per-bucket sum/count for within-seed
    # paired bootstrap. Rank-local — the aggregate keys above receive the
    # cross-rank reduction; per-window arrays describe this rank's shard.
    per_window_bucket_sum: list[list[float]] = []
    per_window_bucket_count: list[list[int]] = []
    total = count_lm_starts(num_tokens, seq_len, stride)
    sharded = count_sharded_lm_starts(
        total_starts=total, rank=rank, world_size=world_size,
    )
    if sharded <= 0:
        # This rank has no val work; emit zero-count buckets so the
        # schema is populated.
        return {
            "per_bucket_val_ce": [0.0] * int(num_buckets),
            "rare_bucket_val_ce": 0.0,
            "val_bucket_num_buckets": int(num_buckets),
            "val_bucket_token_counts": [0] * int(num_buckets),
            "per_window_bucket_ce_sum": [],
            "per_window_bucket_count": [],
        }
    n_steps = (sharded + batch_size - 1) // batch_size
    model.eval()
    try:
        with torch.no_grad():
            for step in range(n_steps):
                batch_starts = sequential_sharded_lm_starts(
                    num_tokens=num_tokens,
                    seq_len=seq_len,
                    stride=stride,
                    batch_size=batch_size,
                    rank=rank,
                    world_size=world_size,
                    step=step,
                )
                inputs, targets = batch_from_start_tensor(
                    tokens=tokens,
                    starts=batch_starts,
                    seq_len=seq_len,
                    device=device,
                    vocab_size=vocab_size,
                )
                with autocast_context(precision, device_type=inputs.device.type):
                    hidden = model.encode(inputs)
                    logits = model.lm_head(model.final_norm(hidden))
                vocab = logits.size(-1)
                ce = F.cross_entropy(
                    logits.reshape(-1, vocab).float(),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
                bucket_idx = token_bucket[targets]
                flat_idx = bucket_idx.reshape(-1)
                flat_ce = ce.reshape(-1).to(dtype=torch.float64)
                flat_ones = torch.ones_like(flat_idx, dtype=torch.int64)
                window_sum = torch.zeros_like(bucket_sum)
                window_count = torch.zeros_like(bucket_count)
                window_sum.scatter_add_(0, flat_idx, flat_ce)
                window_count.scatter_add_(0, flat_idx, flat_ones)
                bucket_sum.add_(window_sum)
                bucket_count.add_(window_count)
                per_window_bucket_sum.append(
                    [float(x) for x in window_sum.detach().cpu().tolist()]
                )
                per_window_bucket_count.append(
                    [int(x) for x in window_count.detach().cpu().tolist()]
                )
    finally:
        model.train()
    # All-reduce across ranks before dividing — otherwise each rank reports
    # its own shard's bucket means and rank-0's result is skewed by shard
    # composition rather than the true val distribution.
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(bucket_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(bucket_count, op=dist.ReduceOp.SUM)
    counts_f = bucket_count.to(torch.float64).clamp_min(1.0)
    per_bucket_val_ce = (bucket_sum / counts_f).detach().cpu().tolist()
    rare_k = max(1, int(num_buckets) // 4)
    rare_bucket_val_ce = float(sum(per_bucket_val_ce[:rare_k]) / float(rare_k))
    return {
        "per_bucket_val_ce": [float(x) for x in per_bucket_val_ce],
        "rare_bucket_val_ce": rare_bucket_val_ce,
        "val_bucket_num_buckets": int(num_buckets),
        "val_bucket_token_counts": [int(x) for x in bucket_count.detach().cpu().tolist()],
        "per_window_bucket_ce_sum": per_window_bucket_sum,
        "per_window_bucket_count": per_window_bucket_count,
    }


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


def _sync_scopt_dense_tensors_coalesced(
    tensors: list[torch.Tensor],
    *,
    world_size: int,
    all_group: "dist.ProcessGroup | None" = None,
) -> None:
    """Flatten-reduce-unflatten for ScOpt dense state tensors.

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
    cfg = scopt_allreduce_config(world_size=world_size, all_group=all_group)
    if cfg.op != "sum":
        raise ValueError(f"unsupported ScOpt all-reduce op {cfg.op!r}")
    contig = [t.contiguous() for t in tensors]
    flat = torch._utils._flatten_dense_tensors(contig)
    if cfg.train_grad_scale != 1.0:
        flat.mul_(cfg.train_grad_scale)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=all_group)
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


def _allreduce_scopt_grads(
    model: torch.nn.Module,
    *,
    world_size: int,
    all_group: "dist.ProcessGroup | None",
) -> None:
    cfg = scopt_allreduce_config(world_size=world_size, all_group=all_group)
    if cfg.op != "sum":
        raise ValueError(f"unsupported ScOpt all-reduce op {cfg.op!r}")
    if cfg.train_grad_scale != 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(cfg.train_grad_scale)
    allreduce_grads(
        model,
        world_size,
        group=all_group,
        op=dist.ReduceOp.SUM,
        materialize_zeros=cfg.materialize_zeros,
    )


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
    all_group: "dist.ProcessGroup | None" = None,
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
            _allreduce_scopt_grads(
                model,
                world_size=world_size,
                all_group=all_group,
            )
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
        all_group=all_group,
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
    all_group: "dist.ProcessGroup | None" = None,
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
        _allreduce_scopt_grads(
            model,
            world_size=world_size,
            all_group=all_group,
        )

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
        _sync_scopt_dense_tensors_coalesced(
            dense_rare,
            world_size=world_size,
            all_group=all_group,
        )

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
        _sync_scopt_dense_tensors_coalesced(
            [vec for _, vec in channel_pressure_items],
            world_size=world_size,
            all_group=all_group,
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
        _sync_scopt_dense_tensors_coalesced(
            [row_pressure],
            world_size=world_size,
            all_group=all_group,
        )

    pressure_stats = _pressure_summary(pressure)

    pending = _ScOptPending(
        rare_map=rare_map,
        channel_pressure_items=channel_pressure_items,
        row_pressure=row_pressure,
        pressure_stats=pressure_stats,
    )
    return total_loss.detach(), pending


# ---------------------------------------------------------------------------
# Episodic-cache GPU-resident IPC (Perf Pass C)
# ---------------------------------------------------------------------------
#
# Replaces the POSIX shm SPSC ring path from Phase 1 Tasks 1.4 + 1.5. A
# train rank with ``episodic_enabled=True`` now packs its selected write
# + query payloads into a single contiguous fp32 tensor of shape
# ``[K_max, slot_dim]`` and emits via ``dist.gather(dst=N-1)``. The
# episodic rank receives, filters by ``valid_mask``, and routes each
# valid slot to (a) ``cache.append`` for the write side and (b) a
# Python list ``controller_query_queue`` for Phase 2 query handling.
# Slot layout lives in ``chaoscontrol.episodic.gpu_slot``; design doc
# is ``docs/plans/2026-04-25-perf-pass-c-gpu-resident-ipc.md``.

# Default K_max for the per-rank emit tensor. At Phase 1 ``top_p ≈
# 1/(B*T)`` the typical valid-row count is 1-2 per step; K_max=16 leaves
# headroom for a 16x increase before we'd silently truncate.
_DEFAULT_EPISODIC_K_MAX = 16


def _event_ring_name(prefix: str, *, rank: int | None = None) -> str:
    pid = int(os.getpid())
    if rank is None:
        return f"{prefix}_pid{pid}"
    exact = f"{prefix}_rank{int(rank)}_pid{pid}"
    if (
        sys.platform == "darwin"
        and prefix == "/cc_episodic_write"
        and len(exact) > 31
    ):
        return f"{prefix}_r{int(rank)}_pid{pid}"
    return exact


def _create_event_ring(cls: Any, name: str) -> Any:
    # POSIX shm names can outlive a crashed test process. B4 names are
    # PID-scoped, so unlink-before-create is safe cleanup for stale regions.
    try:
        cls.unlink(name)
    except Exception:
        pass
    return cls.create(name)


def _push_event_ring(
    owner: Any,
    *,
    ring_attr: str,
    drops_attr: str,
    event: dict[str, Any],
) -> bool:
    ring = getattr(owner, ring_attr, None)
    if ring is None:
        return False
    pushed = bool(ring.push(event))
    if not pushed:
        setattr(owner, drops_attr, int(getattr(owner, drops_attr, 0)) + 1)
    return pushed


def _u8_wire(value: int, *, sentinel: int = 255) -> int:
    v = int(value)
    return v if 0 <= v <= 255 else int(sentinel)


def _u64_wire(value: int, *, sentinel: int = (1 << 64) - 1) -> int:
    v = int(value)
    return v if v >= 0 else int(sentinel)


def _cleanup_episodic_event_rings(owner: Any) -> None:
    for name_attr, ring_attr, cls in (
        ("write_ring_name", "write_ring", _ext.ShmRingWriteEvent),
        ("query_ring_name", "query_ring", _ext.ShmRingQueryEvent),
        ("replay_ring_name", "replay_ring", _ext.ShmRingReplayOutcome),
    ):
        name = getattr(owner, name_attr, None)
        if name:
            try:
                cls.unlink(str(name))
            except Exception:
                pass
        if hasattr(owner, ring_attr):
            try:
                setattr(owner, ring_attr, None)
            except Exception:
                pass
        if hasattr(owner, name_attr):
            try:
                setattr(owner, name_attr, None)
            except Exception:
                pass


class EpisodicGpuEmit:
    """Train-rank handle threaded into ``_run_train_step``.

    Holds the pre-allocated ``[K_max, slot_dim]`` slot tensor (zeroed
    each step before pack), the slot-format dimensions (S, D), the
    fingerprint window, and the configured ``top_p`` (or NaN sentinel
    for "use ``1/(B*T)`` default"). The episodic rank does NOT receive
    this handle — it allocates its own ``gather_list`` of empty
    ``[K_max, slot_dim]`` tensors via ``_drain_episodic_payloads_gpu``.

    Plain ``__slots__`` class (not a ``@dataclass``) for the same
    importlib-from-spec compatibility reason ``_ScOptPending`` and the
    pre-Pass-C ``EpisodicRingsHandle`` used: Python 3.14's
    ``dataclasses`` looks up ``cls.__module__`` in ``sys.modules`` while
    resolving annotations, which fails when the runner is loaded via
    ``importlib.util.spec_from_file_location``.
    """

    __slots__ = (
        "slot_tensor",
        "k_max",
        "span_length",
        "key_rep_dim",
        "fingerprint_window",
        "top_p",
        "write_ring",
        "write_ring_name",
        "write_ring_drops",
    )

    def __init__(
        self,
        *,
        slot_tensor: torch.Tensor,
        k_max: int,
        span_length: int,
        key_rep_dim: int,
        fingerprint_window: int,
        top_p: float,
        write_ring: Any | None = None,
        write_ring_name: str | None = None,
    ) -> None:
        self.slot_tensor = slot_tensor
        self.k_max = k_max
        self.span_length = span_length
        self.key_rep_dim = key_rep_dim
        self.fingerprint_window = fingerprint_window
        self.top_p = top_p
        self.write_ring = write_ring
        self.write_ring_name = write_ring_name
        self.write_ring_drops = 0


def _create_episodic_emit(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    config: dict[str, Any],
) -> EpisodicGpuEmit | None:
    """Build the per-rank emit-tensor handle.

    Returns the handle on every rank when ``episodic_enabled=True`` and
    ``world_size > 1``. Both train ranks and the episodic rank carry
    their own emit tensor — the train ranks pack into theirs, the
    episodic rank's stays all-zeros (its valid_mask=0 rows contribute
    no cache writes, but the gather collective is symmetric and every
    rank must contribute the same shape).

    Returns ``None`` for ``episodic_enabled=False`` or ``world_size <=
    1`` so the caller's ``handle is None`` check is the back-compat
    skip.
    """
    if not bool(config.get("episodic_enabled", False)):
        return None
    if world_size <= 1:
        return None
    span_length = int(config.get("episodic_span_length", 4))
    key_rep_dim = int(config.get("episodic_key_rep_dim", config.get("model_dim", 0)))
    if key_rep_dim <= 0:
        raise ValueError(
            "episodic_enabled=True requires episodic_key_rep_dim > 0 "
            "(or a positive model_dim default); got "
            f"episodic_key_rep_dim={config.get('episodic_key_rep_dim')!r}, "
            f"model_dim={config.get('model_dim')!r}"
        )
    fingerprint_window = int(config.get("episodic_fingerprint_window", 8))
    k_max = int(config.get("episodic_k_max", _DEFAULT_EPISODIC_K_MAX))
    if k_max <= 0:
        raise ValueError(
            f"episodic_k_max must be positive; got {k_max}"
        )
    slot = make_slot_tensor(
        k_max=k_max,
        span_length=span_length,
        key_rep_dim=key_rep_dim,
        device=device,
        dtype=torch.float32,
    )
    # ``episodic_top_p`` is read lazily in-step because B*T isn't
    # known here; default is ``1.0 / (B * T)`` per the design. The
    # value stored on the handle is whatever the config sets, or NaN
    # to signal "use the per-step default".
    top_p = float(config.get("episodic_top_p", float("nan")))
    event_log_enabled = bool(config.get("episodic_event_log_enabled", False))
    # WRITE_EVENT producers live on train ranks. The episodic rank carries an
    # all-zero gather tensor only, so allocating a ring there would be a false
    # signal and would violate the default "no producer unless enabled"
    # invariant.
    write_ring = None
    write_ring_name = None
    if event_log_enabled and int(rank) != int(world_size) - 1:
        write_ring_name = _event_ring_name(
            "/cc_episodic_write",
            rank=int(rank),
        )
        write_ring = _create_event_ring(_ext.ShmRingWriteEvent, write_ring_name)
    return EpisodicGpuEmit(
        slot_tensor=slot,
        k_max=k_max,
        span_length=span_length,
        key_rep_dim=key_rep_dim,
        fingerprint_window=fingerprint_window,
        top_p=top_p,
        write_ring=write_ring,
        write_ring_name=write_ring_name,
    )


def _right_pad_per_token_signal(signal: torch.Tensor, T: int) -> torch.Tensor:
    """Right-pad a ``[B, T-1]`` per-token signal to ``[B, T]`` with zero.

    Adapter for the case where ``per_token_ce`` arrives in the next-
    token-CE convention (``[B, T-1]``, one CE per predicted target). The
    exp23 batch builder shifts targets at construction time so the
    typical signal is already ``[B, T]``; this helper is a no-op there.
    Anything more than one short of ``T`` is a coding error and raises.
    """
    if signal.dim() != 2:
        raise ValueError(
            f"_right_pad_per_token_signal expects a 2-D tensor; got shape "
            f"{tuple(signal.shape)}"
        )
    B, S = signal.shape
    if S == T:
        return signal
    if S != T - 1:
        raise ValueError(
            f"_right_pad_per_token_signal: signal width {S} must equal "
            f"T={T} (already padded) or T-1={T - 1} (transformer convention); "
            "any other width is a coding error"
        )
    return torch.cat([signal, signal.new_zeros((B, 1))], dim=1)


def _emit_episodic_payloads_gpu(
    *,
    emit: EpisodicGpuEmit,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pressure: torch.Tensor,
    per_token_ce: torch.Tensor,
    hidden: torch.Tensor,
    rank: int,
    world_size: int,
    all_group: "dist.ProcessGroup | None",
    current_step: int = 0,
    write_bucket: int = 0,
) -> None:
    """Train-rank emit: pack the slot tensor and call ``dist.gather``.

    Only train ranks reach this function (the episodic rank's
    ``is_episodic_rank`` branch in ``_run_train_step`` calls
    ``dist.gather`` directly with its own ``gather_list``). Train ranks
    pass ``gather_list=None`` to ``dist.gather`` so they emit-only.

    The gather is a hard sync point across all ranks; train ranks block
    here until the episodic rank also reaches its gather call. Packing
    is a Python loop over ``select_top_p_positions``'s output (K is
    small — default ``top_p ≈ 1/(B*T)`` produces 1-2 valid rows per
    step). Boundary-dropped positions stay at ``valid_mask=0`` so the
    drain's filter skips them.
    """
    T = int(inputs.size(1))
    pressure_full = _right_pad_per_token_signal(pressure, T)
    ce_full = _right_pad_per_token_signal(per_token_ce, T)
    # Default top_p = ``1 / (B * T)`` — one position per batch step in
    # expectation. ``emit.top_p`` is NaN when the config didn't pin a
    # value; in that case we compute the default here where B/T are
    # known. ``select_top_p_positions`` always returns at least 1 row,
    # so this guarantees at least one selection attempt per step.
    if not (emit.top_p == emit.top_p):  # NaN sentinel
        numel = int(inputs.numel())
        top_p = 1.0 / float(max(numel, 1))
    else:
        top_p = float(emit.top_p)
    write_signal = (
        pressure_full.detach().to(dtype=torch.float32)
        * ce_full.detach().to(dtype=torch.float32)
    )
    positions = select_top_p_positions(write_signal, top_p=top_p)
    K = int(positions.size(0))
    if K > emit.k_max:
        # Truncate silently — design doc says K > K_max is "config want
        # bigger than the slot can hold"; assert covers an upstream bug
        # where top_p would explode K beyond the configured cap.
        positions = positions[: emit.k_max]
        K = emit.k_max
    # Zero out the slot tensor so untouched rows have valid_mask=0 (the
    # drain's filter relies on this).
    emit.slot_tensor.zero_()
    W = int(emit.fingerprint_window)
    S = int(emit.span_length)
    D = int(emit.key_rep_dim)
    B, _ = inputs.shape
    inputs_i64 = inputs.detach().to(dtype=torch.int64)
    targets_i64 = targets.detach().to(dtype=torch.int64)
    hidden_detached = hidden.detach()
    for k in range(K):
        b = int(positions[k, 0].item())
        t = int(positions[k, 1].item())
        # Boundary skip mirrors ``build_write_payload``: need a full
        # fingerprint window to the left and a full span to the right.
        # Skip BEFORE allocating a candidate_id so the rank_seq doesn't
        # accumulate gaps for boundary-rejected positions — keeps offline
        # trace joins (writer → controller → cache) seq-contiguous.
        if t < W or t + S > T:
            continue
        candidate_id: int | None = None
        if emit.write_ring is not None:
            candidate_id = _rank_prefixed_event_id(
                source_rank=int(rank),
                rank_seq=_next_admission_trace_seq(),
            )
        fp_window = inputs_i64[b, t - W:t]
        key_fp = fingerprint_tokens(fp_window)
        anchor = int(targets_i64[b, t].item())
        value_tok_ids = targets_i64[b, t:t + S]
        # key_rep == residual in Phase 1 (same write-time hidden state);
        # the slot carries both so the controller queue can consume the
        # residual without the cache-writer blocking the controller path.
        # NOTE: pack_payload upcasts these to fp32 internally, which is
        # required for the int64-via-fp32-view reinterpret to be byte-
        # exact across all GPUs. Do NOT "optimize" by passing bf16 —
        # the slot dtype is fixed fp32 by the gpu_slot module contract.
        key_rep = hidden_detached[b, t]
        residual = hidden_detached[b, t]
        pack_payload(
            emit.slot_tensor[k],
            valid_mask=1.0,
            pressure=float(pressure_full[b, t].item()),
            key_fp=key_fp,
            value_anchor_id=anchor,
            value_tok_ids=value_tok_ids,
            key_rep=key_rep,
            residual=residual,
            span_length=S,
            key_rep_dim=D,
        )
        if emit.write_ring is not None:
            assert candidate_id is not None
            _push_event_ring(
                emit,
                ring_attr="write_ring",
                drops_attr="write_ring_drops",
                event=build_write_event_dict(
                    candidate_id=int(candidate_id),
                    gpu_step=int(current_step),
                    source_rank=int(rank),
                    key_fp=int(key_fp),
                    key_rep=key_rep,
                    value_tok_ids=value_tok_ids,
                    value_anchor_id=int(anchor),
                    pressure_at_write=float(pressure_full[b, t].item()),
                    pre_write_ce=float(ce_full[b, t].item()),
                    write_bucket=int(write_bucket),
                ),
            )
    # Single GPU-to-GPU collective. Train ranks emit-only — they pass
    # ``gather_list=None`` to ``dist.gather`` and the episodic rank
    # holds the receive list. ``all_group is None`` is the single-rank
    # test path: callers inspect ``emit.slot_tensor`` directly without
    # a collective.
    if all_group is not None:
        dist.gather(
            emit.slot_tensor,
            gather_list=None,
            dst=int(world_size) - 1,
            group=all_group,
        )


class _EpisodicConsumerState:
    """Episodic-rank consumer state: cache + heartbeat + controller queues.

    Constructed by ``_attach_episodic_consumer`` once per runner init. The
    no-op shape (``cache=None``, ``heartbeat=[0]``,
    ``controller_query_queue=[]``, ``tagged_replay_queue=[]``) is what
    the train ranks and ``episodic_enabled=False`` runs see so the
    ``_run_train_step`` consumer branch is bit-identical to pre-Pass-C
    on those code paths.

    Pass C dropped the ``write_rings`` field — the new path uses
    ``dist.gather`` instead of POSIX shm rings, so there are no rings to
    track on the consumer side. The ``controller_query_queue`` is the
    in-process Python list that the Phase 2 CPU controller reads from;
    Phase 1's drain just appends one entry per valid slot per step and
    exposes the list to the runner for telemetry.

    Phase 2 added ``tagged_replay_queue``: the controller's output. The
    Y worktree's Phase 3 replay path drains this list each episodic-rank
    step and runs Dreamworld backward on each tagged slot. The runner
    spawns a daemon ``threading.Thread`` running
    ``chaoscontrol.episodic.controller.controller_main`` that owns the
    write side of this queue.

    The heartbeat is a single-element list so increments propagate back
    to the caller (the runner's outer loop reads it for telemetry).
    """

    __slots__ = (
        "cache",
        "heartbeat",
        "controller_query_queue",
        "controller_query_enabled",
        "tagged_replay_queue",
        "query_ring",
        "query_ring_name",
        "query_ring_drops",
        "rank_query_seq",
        "replay_ring",
        "replay_ring_name",
        "replay_ring_drops",
        "bucket_baseline_ema",
        "compute_replay_ce_pair",
        "pending_post_step_replays",
        "online_learning_bridge",
    )

    def __init__(
        self,
        cache: EpisodicCache | None,
        heartbeat: list[int],
        controller_query_queue: list[dict[str, Any]],
        controller_query_enabled: bool = False,
        tagged_replay_queue: list[dict[str, Any]] | None = None,
        episodic_event_log_enabled: bool = False,
        compute_replay_ce_pair: bool = False,
    ) -> None:
        self.cache = cache
        self.heartbeat = heartbeat
        self.controller_query_queue = controller_query_queue
        # Default False: Phase 1 ships without a Phase 2 consumer wired,
        # so the queue would grow unbounded and retain GPU residual
        # tensors (~1.25 GB at world=8, 600s, D=256). Phase 2's
        # controller-bring-up task flips this True at the same time it
        # adds the consumer that drains the queue.
        self.controller_query_enabled = bool(controller_query_enabled)
        # Phase 2: tagged_replay_queue defaults to a fresh empty list on
        # every consumer state. The controller thread (Y-side) writes
        # to it; the replay path (Y-side) reads from it. Keeping the
        # default ``None → []`` rather than a class-level mutable
        # default avoids the classic Python gotcha where every consumer
        # state shares the same list instance.
        self.tagged_replay_queue = (
            [] if tagged_replay_queue is None else tagged_replay_queue
        )
        self.query_ring = None
        self.query_ring_name: str | None = None
        self.query_ring_drops = 0
        if episodic_event_log_enabled:
            self.query_ring_name = _event_ring_name("/cc_episodic_query")
            self.query_ring = _create_event_ring(
                _ext.ShmRingQueryEvent,
                self.query_ring_name,
            )
        self.rank_query_seq: dict[int, int] | None = (
            {} if episodic_event_log_enabled else None
        )
        self.replay_ring = None
        self.replay_ring_name: str | None = None
        self.replay_ring_drops = 0
        if episodic_event_log_enabled:
            self.replay_ring_name = _event_ring_name("/cc_episodic_replay")
            self.replay_ring = _create_event_ring(
                _ext.ShmRingReplayOutcome,
                self.replay_ring_name,
            )
        # Per-bucket reward baseline. Disabled keeps None: no EMA state and
        # no event emission on the default path.
        self.bucket_baseline_ema: list[float] | None = (
            [0.0, 0.0, 0.0, 0.0] if episodic_event_log_enabled else None
        )
        # Phase B5: opt-in true pre/post replay CE pair. When enabled,
        # the drain stages each successful replay (slot's value tokens +
        # the just-emitted REPLAY_OUTCOME dict) into
        # ``pending_post_step_replays``. After ``optimizer.step()`` the
        # outer loop drains the staged list, runs a no-grad forward on
        # the post-step weights, mutates the dict's
        # ``ce_after_replay`` / ``ce_delta_raw`` / ``bucket_baseline`` /
        # ``reward_shaped`` fields, and updates the bucket EMA.
        # Disabling this flag keeps the B3 default: reward fields stay
        # NaN and the EMA never updates. Allocating the list only when
        # both gates are on avoids any per-step allocation cost on the
        # default path.
        self.compute_replay_ce_pair: bool = bool(
            compute_replay_ce_pair and episodic_event_log_enabled
        )
        self.pending_post_step_replays: list[dict[str, Any]] | None = (
            [] if self.compute_replay_ce_pair else None
        )
        self.online_learning_bridge = None


def _attach_episodic_consumer(
    *,
    episodic_enabled: bool,
    is_episodic_rank: bool,
    world_size: int,
    config: dict[str, Any],
    model_dim: int,
    all_group: "dist.ProcessGroup | None",
) -> _EpisodicConsumerState:
    """Build the episodic-rank consumer (cache + controller queue).

    On non-episodic-rank or ``episodic_enabled=False`` this returns the
    no-op state (``cache=None``, ``controller_query_queue=[]``); the
    runner's outer loop and ``_run_train_step``'s consumer branch both
    treat that as a skip.

    On the episodic rank, constructs an ``EpisodicCache`` from
    config-derived defaults (Decision 0.4 of the design doc):
    capacity=4096, span_length=4, key_rep_dim=model_dim,
    grace_steps=1000, utility_ema_decay=0.99. The
    ``controller_query_queue`` starts empty and grows one entry per
    valid slot per gather drain.

    Pass C: the producer/consumer rendezvous from the POSIX shm path is
    gone — the gather collective is its own implicit barrier, and there
    are no shm segments to attach to. The ``all_group`` parameter is
    accepted (for back-compat with the call site) but unused.
    """
    no_op = _EpisodicConsumerState(
        cache=None,
        heartbeat=[0],
        controller_query_queue=[],
    )
    if not episodic_enabled or not is_episodic_rank:
        return no_op
    if int(world_size) < 2:
        # The runner's pre-collective guards reject this combination
        # before we get here, but the helper stays defensive — building
        # an empty consumer is the closest thing to a no-op for a misuse.
        return no_op

    span_length = int(config.get("episodic_span_length", 4))
    key_rep_dim = int(config.get("episodic_key_rep_dim", model_dim))
    cache = EpisodicCache(
        capacity=int(config.get("episodic_capacity", 4096)),
        span_length=span_length,
        key_rep_dim=key_rep_dim,
        grace_steps=int(config.get("episodic_grace_steps", 1000)),
        utility_ema_decay=float(
            config.get("episodic_utility_ema_decay", 0.99)
        ),
        fingerprint_window=int(config.get("episodic_fingerprint_window", 8)),
        slot_state_dim=int(config.get("episodic_slot_state_dim", 0)),
        simplex_k_max=int(config.get("episodic_simplex_k_max", 0)),
    )
    return _EpisodicConsumerState(
        cache=cache,
        heartbeat=[0],
        controller_query_queue=[],
        controller_query_enabled=bool(
            config.get("controller_query_enabled", False)
        ),
        episodic_event_log_enabled=bool(
            config.get("episodic_event_log_enabled", False)
        ),
        compute_replay_ce_pair=bool(
            config.get("episodic_compute_replay_ce_pair", False)
        ),
    )


class _EpisodicControllerHandle:
    """Runner-init handle to the episodic controller thread.

    Bundles the daemon thread + its stop event + heartbeat counter so
    the runner's ``finally`` block can cleanly shut down the loop. None
    on every code path that doesn't spawn the controller (train ranks,
    ``episodic_enabled=False``, ``controller_query_enabled=False``,
    or the episodic rank with no cache).

    Plain ``__slots__`` class for the same importlib-from-spec
    compatibility reason ``EpisodicGpuEmit`` and ``_ScOptPending`` use.
    """

    __slots__ = ("thread", "stop_event", "heartbeat")

    def __init__(
        self,
        thread: threading.Thread,
        stop_event: threading.Event,
        heartbeat: list[int],
    ) -> None:
        self.thread = thread
        self.stop_event = stop_event
        self.heartbeat = heartbeat


def _rank_prefixed_event_id(*, source_rank: int, rank_seq: int) -> int:
    """Pack ``(rank, local_seq)`` into the CPU-controller wire id shape."""
    if not 0 <= int(source_rank) < 256:
        raise ValueError(f"source_rank must fit in u8; got {source_rank}")
    seq = int(rank_seq)
    if seq < 0:
        raise ValueError(f"rank_seq must be non-negative; got {rank_seq}")
    return (int(source_rank) << 56) | (seq & ((1 << 56) - 1))


_SIMPLEX_CANDIDATES = 16
_SIMPLEX_SLOT_SENTINEL = (1 << 64) - 1   # UINT64_MAX
_SIMPLEX_COSINE_SENTINEL = 0.0


def _pad_simplex_slot_ids(
    candidate_slot_ids: list[int] | None,
) -> list[int]:
    """Sentinel-pad a candidate slot id list to length 16.

    None / empty list → all-sentinel (heuristic-only / V0 path; the C++
    controller dispatches on ``candidate_slot_ids[0] == UINT64_MAX``).
    Shorter list → trailing slots filled with the sentinel. Length > 16
    is a programmer error, not a silent truncation.
    """
    if candidate_slot_ids is None:
        return [_SIMPLEX_SLOT_SENTINEL] * _SIMPLEX_CANDIDATES
    ids = [int(s) for s in candidate_slot_ids]
    if len(ids) > _SIMPLEX_CANDIDATES:
        raise ValueError(
            f"candidate_slot_ids length {len(ids)} exceeds simplex "
            f"capacity {_SIMPLEX_CANDIDATES}"
        )
    if len(ids) < _SIMPLEX_CANDIDATES:
        ids = ids + [_SIMPLEX_SLOT_SENTINEL] * (
            _SIMPLEX_CANDIDATES - len(ids)
        )
    return ids


def _pad_simplex_cosines(
    candidate_cosines: list[float] | None,
) -> list[float]:
    """Sentinel-pad a candidate cosine list to length 16. Symmetric to
    ``_pad_simplex_slot_ids`` so the two arrays default independently —
    a producer wiring the simplex retrieval up incrementally can supply
    one without the other."""
    if candidate_cosines is None:
        return [_SIMPLEX_COSINE_SENTINEL] * _SIMPLEX_CANDIDATES
    cosines = [float(c) for c in candidate_cosines]
    if len(cosines) > _SIMPLEX_CANDIDATES:
        raise ValueError(
            f"candidate_cosines length {len(cosines)} exceeds simplex "
            f"capacity {_SIMPLEX_CANDIDATES}"
        )
    if len(cosines) < _SIMPLEX_CANDIDATES:
        cosines = cosines + [_SIMPLEX_COSINE_SENTINEL] * (
            _SIMPLEX_CANDIDATES - len(cosines)
        )
    return cosines


def _query_event_simplex_candidates(
    *,
    cache: EpisodicCache,
    query_residual: torch.Tensor,
    score_mode: str,
    k: int,
) -> tuple[list[int], list[float]]:
    """Return the cache-side simplex candidate set for a QueryEvent.

    The candidate ids follow the same top-K scorer the controller thread
    will use; the companion values are raw query/key cosines for those ids.
    This makes the wire event a real local simplex snapshot instead of a
    sentinel-padded placeholder.
    """
    k_eff = min(_SIMPLEX_CANDIDATES, max(0, int(k)))
    if k_eff <= 0 or cache is None or not cache.occupied.any():
        return [], []
    slots = query_topk(
        query_residual,
        cache,
        k=k_eff,
        score_mode=str(score_mode),
    )
    if slots.numel() == 0:
        return [], []

    device = query_residual.device
    slot_idx = slots.to(device=device, dtype=torch.long)
    keys = cache.key_rep.to(device=device, dtype=torch.float32)[slot_idx]
    q = query_residual.detach().to(device=device, dtype=torch.float32)
    q = q / (q.norm() + 1e-8)
    keys_n = keys / (keys.norm(dim=1, keepdim=True) + 1e-8)
    cosines = (keys_n @ q).clamp(-1.0, 1.0)
    cosine_list = cosines.detach().to(
        device="cpu", dtype=torch.float32,
    ).tolist()
    return (
        [int(s) for s in slots.detach().cpu().tolist()],
        [float(c) for c in cosine_list],
    )


def _emit_query_event(
    *,
    consumer: _EpisodicConsumerState,
    source_rank: int,
    gpu_step: int,
    query_residual: torch.Tensor,
    pressure: float,
    pre_query_ce: float,
    bucket: int,
    candidate_slot_ids: list[int] | None = None,
    candidate_cosines: list[float] | None = None,
) -> int | None:
    """Push a QUERY_EVENT dict to the Phase B4 shm ring.

    Returns the query_id when an event is emitted; returns None when the event
    ring is disabled. The dict field order mirrors QueryEvent in wire_events.h.

    ``candidate_slot_ids`` / ``candidate_cosines`` carry the simplex
    candidate set (Phase S3 of the simplex-controller pivot). Pass the
    heuristic top-K retrieval result (up to 16 slot ids and their query
    cosines) to enable the simplex policy path; pass ``None`` (or omit)
    to ride the V0 heuristic-only path — the wire payload is sentinel-
    padded and the C++ controller falls back to per-slot scoring.
    """
    if consumer.query_ring is None or consumer.rank_query_seq is None:
        return None
    rank = int(source_rank)
    seq = int(consumer.rank_query_seq.get(rank, 0))
    consumer.rank_query_seq[rank] = seq + 1
    query_id = _rank_prefixed_event_id(source_rank=rank, rank_seq=seq)
    _push_event_ring(
        consumer,
        ring_attr="query_ring",
        drops_attr="query_ring_drops",
        event={
            "event_type": 2,
            "source_rank": rank,
            "bucket": _u8_wire(int(bucket)),
            "query_id": int(query_id),
            "gpu_step": int(gpu_step),
            "query_rep": tensor_fp16_to_u16_wire(query_residual),
            "pressure": float(pressure),
            "pre_query_ce": float(pre_query_ce),
            "candidate_slot_ids": _pad_simplex_slot_ids(candidate_slot_ids),
            "candidate_cosines": _pad_simplex_cosines(candidate_cosines),
        },
    )
    return int(query_id)


_REPLAY_STATUS_OK = 0
_REPLAY_STATUS_SLOT_MISSING = 1
_REPLAY_STATUS_STALE = 2
_REPLAY_STATUS_NAN = 3
_REPLAY_STATUS_SKIPPED = 4


def _replay_id_from_tag(
    *,
    entry: dict[str, Any],
    query_event_id: int,
    selected_rank: int,
) -> int:
    if "replay_id" in entry:
        return int(entry["replay_id"])
    if int(query_event_id) < 0 or int(selected_rank) < 0:
        return -1
    # Sentinel until the trained controller owns replay_id: mirror
    # controller.py's 56-bit clamp before adding the selected-rank byte so
    # the derived id fits in u64 on the future wire path.
    return ((int(query_event_id) & ((1 << 56) - 1)) << 8) | (
        int(selected_rank) & 0xFF
    )


def _emit_replay_outcome(
    *,
    consumer: _EpisodicConsumerState,
    entry: dict[str, Any],
    current_step: int,
    slot: int,
    outcome_status: int,
    source_write_id: int,
    write_bucket: int,
    ce_before_replay_override: float | None = None,
    ce_after_replay: float = float("nan"),
    grad_cos_rare: float = float("nan"),
    grad_cos_total: float = float("nan"),
    defer_reward_shaping: bool = False,
) -> dict[str, Any] | None:
    """Push a REPLAY_OUTCOME dict to the Phase B4 shm ring.

    The real controller fields are threaded through when present. Sentinel
    defaults are documented inline so Phase C can replace them deliberately.

    Phase B5 adds the ``defer_reward_shaping`` knob: when True, the dict is
    emitted with placeholder NaN reward fields and the bucket EMA is NOT
    updated. The post-step CE pair pass owns the patch + EMA update once
    ``optimizer.step()`` lands (see ``_run_post_step_replay_ce``). When the
    flag is False (B3 default), the EMA still updates on the upstream-supplied
    delta (NaN-skip in place) and the dict's reward fields are computed
    inline against the supplied ``ce_after_replay``.
    """
    replay_ring = getattr(consumer, "replay_ring", None)
    bucket_baseline_ema = getattr(consumer, "bucket_baseline_ema", None)
    if replay_ring is None or bucket_baseline_ema is None:
        return None

    query_event_id = int(entry.get("query_event_id", -1))
    selected_rank = int(entry.get("selected_rank", -1))
    replay_id = _replay_id_from_tag(
        entry=entry,
        query_event_id=query_event_id,
        selected_rank=selected_rank,
    )
    # Until the trained controller is wired, selection_step is the current GPU
    # step proxy when a tag does not carry the controller's selection step.
    selection_step = int(entry.get("selection_step", int(current_step)))
    policy_version = int(entry.get("policy_version", 0))
    teacher_score = float(entry.get("teacher_score", entry.get("score", 0.0)))
    # Until the trained controller is wired, controller_logit is a zero
    # sentinel rather than reusing the heuristic score.
    controller_logit = float(entry.get("controller_logit", 0.0))
    if ce_before_replay_override is not None:
        # Phase B5: the helper-computed pre-step ``replay_loss`` is the
        # canonical ``ce_before_replay`` once the post-step pair runs.
        # Override entry-supplied placeholder so the field carries the
        # number that pairs with the post-step CE the deferred pass will
        # write.
        ce_before_replay = float(ce_before_replay_override)
    else:
        ce_before_replay = float(entry.get("ce_before_replay", float("nan")))
    ce_after = float(ce_after_replay)
    ce_delta_raw = ce_before_replay - ce_after

    bucket = int(write_bucket)
    baseline = 0.0
    if 0 <= bucket < len(bucket_baseline_ema):
        baseline = float(bucket_baseline_ema[bucket])
    reward_shaped = ce_delta_raw - baseline
    if (
        not defer_reward_shaping
        and 0 <= bucket < len(bucket_baseline_ema)
        and math.isfinite(ce_delta_raw)
    ):
        alpha = 0.05
        bucket_baseline_ema[bucket] = (
            (1.0 - alpha) * float(bucket_baseline_ema[bucket])
            + alpha * float(ce_delta_raw)
        )

    if defer_reward_shaping:
        # Post-step pass owns the final values; emit NaN placeholders so
        # downstream readers that crash on partial values raise loudly
        # if the deferred pass forgot to patch.
        ce_after = float("nan")
        ce_delta_raw = float("nan")
        baseline = float("nan")
        reward_shaped = float("nan")

    event = {
        "event_type": 3,
        "selected_rank": _u8_wire(int(selected_rank)),
        "outcome_status": _u8_wire(int(outcome_status)),
        "replay_id": _u64_wire(int(replay_id)),
        "gpu_step": int(current_step),
        "query_event_id": _u64_wire(int(query_event_id)),
        "source_write_id": _u64_wire(int(source_write_id)),
        "slot_id": int(slot),
        "policy_version": int(policy_version),
        "selection_step": int(selection_step),
        "teacher_score": float(teacher_score),
        "controller_logit": float(controller_logit),
        "ce_before_replay": float(ce_before_replay),
        "ce_after_replay": float(ce_after),
        "ce_delta_raw": float(ce_delta_raw),
        "bucket_baseline": float(baseline),
        "reward_shaped": float(reward_shaped),
        "grad_cos_rare": float(grad_cos_rare),
        "grad_cos_total": float(grad_cos_total),
        "flags": 0,
    }
    if not defer_reward_shaping:
        _push_event_ring(
            consumer,
            ring_attr="replay_ring",
            drops_attr="replay_ring_drops",
            event=event,
        )
        _notify_online_learning_bridge(consumer=consumer, event=event)
    return event


def _notify_online_learning_bridge(
    *,
    consumer: _EpisodicConsumerState,
    event: dict[str, Any],
) -> None:
    bridge = getattr(consumer, "online_learning_bridge", None)
    if bridge is None:
        return
    if int(event.get("outcome_status", -1)) != _REPLAY_STATUS_OK:
        return
    reward = float(event.get("reward_shaped", float("nan")))
    if not math.isfinite(reward):
        return
    bridge.on_replay_outcome(event)
    telemetry_fn = getattr(bridge, "telemetry", None)
    telemetry = telemetry_fn() if callable(telemetry_fn) else None
    if telemetry is None and hasattr(bridge, "learner"):
        learner_telemetry = getattr(bridge.learner, "telemetry", None)
        telemetry = learner_telemetry() if callable(learner_telemetry) else None
    if telemetry is None:
        return

    def _metric(name: str, default: float = float("nan")) -> float:
        if isinstance(telemetry, dict):
            return float(telemetry.get(name, default))
        return float(getattr(telemetry, name, default))

    if hasattr(telemetry, "last_gerber_weight"):
        event["gerber_weight"] = _metric("last_gerber_weight")
        event["advantage_raw"] = _metric("last_advantage_raw")
        event["advantage_corrected"] = float(
            getattr(bridge, "last_advantage", _metric("last_advantage_standardized"))
        )
        event["advantage_standardized"] = _metric("last_advantage_standardized")
        event["lambda_hxh"] = _metric("last_lambda_hxh", 0.0)


def _write_replay_ndjson_row(
    logger: DiagnosticsLogger,
    *,
    event_dict: dict[str, Any] | None,
    current_step: int,
    slot: int,
    key_fp: int,
    write_step: int,
    write_pressure: float,
    write_bucket: int,
    query_cosine: float,
    utility_pre: float,
    utility_post: float,
    entry: dict[str, Any],
    replay_loss: float,
    replay_grad_norm: float,
    replay_grad_cos_common: float,
    replay_grad_cos_rare: float,
    replay_grad_cos_total: float,
    utility_signal_raw: float,
    utility_signal_transformed: float,
) -> None:
    """Write one replay diagnostic row, sourcing wire fields from event_dict."""

    def _event_value(field: str, fallback: Any) -> Any:
        if event_dict is None:
            return fallback
        return event_dict.get(field, fallback)

    logger.write_row({
        "step": int(_event_value("gpu_step", int(current_step))),
        "slot": int(_event_value("slot_id", int(slot))),
        "key_fp": int(key_fp),
        "write_step": int(write_step),
        "write_pressure": float(write_pressure),
        "write_bucket": int(write_bucket),
        "query_cosine": float(query_cosine),
        "utility_pre": float(utility_pre),
        "replay_id": int(_event_value("replay_id", entry.get("replay_id", -1))),
        "query_event_id": int(
            _event_value("query_event_id", entry.get("query_event_id", -1))
        ),
        "source_write_id": int(
            _event_value("source_write_id", entry.get("source_write_id", -1))
        ),
        "selection_step": int(
            _event_value("selection_step", entry.get("selection_step", -1))
        ),
        "policy_version": int(
            _event_value("policy_version", entry.get("policy_version", 0))
        ),
        "selected_rank": int(
            _event_value("selected_rank", entry.get("selected_rank", -1))
        ),
        "teacher_score": float(
            _event_value("teacher_score", entry.get("teacher_score", query_cosine))
        ),
        "controller_logit": float(
            _event_value(
                "controller_logit",
                entry.get("controller_logit", query_cosine),
            )
        ),
        "arm": str(entry.get("arm", "")),
        "chosen_idx": int(entry.get("simplex_chosen_idx", entry.get("selected_rank", -1))),
        "p_chosen": float(entry.get("p_chosen", entry.get("simplex_p_chosen", float("nan")))),
        "p_behavior": entry.get("p_behavior", entry.get("simplex_probabilities", [])),
        "entropy": float(entry.get("entropy", float("nan"))),
        "gerber_weight": float(_event_value("gerber_weight", float("nan"))),
        "advantage_raw": float(_event_value("advantage_raw", float("nan"))),
        "advantage_corrected": float(
            _event_value("advantage_corrected", float("nan"))
        ),
        "lambda_hxh": float(
            _event_value("lambda_hxh", entry.get("lambda_hxh", 0.0))
        ),
        "feature_manifest_hash": str(entry.get("feature_manifest_hash", "")),
        "candidate_slot_ids": entry.get(
            "candidate_slot_ids", entry.get("simplex_candidate_slot_ids", [])
        ),
        "candidate_scores": entry.get(
            "candidate_scores", entry.get("simplex_candidate_scores", [])
        ),
        "logits": entry.get("logits", entry.get("simplex_logits", [])),
        "replay_loss": float(replay_loss),
        "ce_before_replay": float(
            _event_value("ce_before_replay", float("nan"))
        ),
        "ce_after_replay": float(_event_value("ce_after_replay", replay_loss)),
        "ce_delta_raw": float(_event_value("ce_delta_raw", float("nan"))),
        "bucket_baseline": float(_event_value("bucket_baseline", 0.0)),
        "reward_shaped": float(_event_value("reward_shaped", float("nan"))),
        "replay_grad_norm": float(replay_grad_norm),
        "replay_grad_cos_common": float(replay_grad_cos_common),
        "replay_grad_cos_rare": float(
            _event_value("grad_cos_rare", replay_grad_cos_rare)
        ),
        "replay_grad_cos_total": float(
            _event_value("grad_cos_total", replay_grad_cos_total)
        ),
        "utility_signal_raw": float(utility_signal_raw),
        "utility_signal_transformed": float(utility_signal_transformed),
        "utility_post": float(utility_post),
        "outcome_status": "ok",
        "flags": int(_event_value("flags", 0)),
    })


def _controller_score_mode_from_config(config: dict[str, Any]) -> str:
    """Read the controller score mode, accepting the Exp24 alias.

    Exp24 matrix builders historically emitted ``controller_query_mode`` while
    the runner consumed ``episodic_controller_score_mode``. Supporting both
    keeps older matrices runnable; a disagreement is rejected because silently
    choosing one would poison the pressure-only control.
    """
    primary = config.get("episodic_controller_score_mode")
    alias = config.get("controller_query_mode")
    if primary is not None and alias is not None and str(primary) != str(alias):
        raise ValueError(
            "conflicting controller score mode keys: "
            f"episodic_controller_score_mode={primary!r}, "
            f"controller_query_mode={alias!r}"
        )
    return str(
        primary if primary is not None else alias if alias is not None
        else "cosine_utility_weighted"
    ).strip()


def _build_simplex_learner_from_cswg(
    weights_path: str,
    *,
    config: dict[str, Any],
) -> Any:
    # Load CSWG v3 (the simplex policy artifact produced by
    # experiments/25_controller_pretrain/dump_to_cpp.py) and construct
    # an _ext.SimplexOnlineLearner with those weights. The learner IS
    # the runtime for simplex_v1 mode — it carries score-side state
    # (fast/slow weights) plus the REINFORCE backward path. No
    # _OnlineLearningRuntimeBridge wrap is needed.
    import sys
    pretrain_dir = (
        Path(__file__).resolve().parent.parent / "25_controller_pretrain"
    )
    if str(pretrain_dir) not in sys.path:
        sys.path.insert(0, str(pretrain_dir))
    from dump_to_cpp import load_cswg_v3  # noqa: E402

    artifact = load_cswg_v3(weights_path)
    header = artifact["manifest"]["dims"]
    tensors = artifact["tensors"]

    weights = _ext.SimplexWeights()
    weights.K_v = int(header["k_v"])
    weights.K_e = int(header["k_e"])
    weights.K_s = int(header["k_s"])
    weights.H = int(header["h"])
    weights.N = int(header["n_vertices"])
    weights.n_heads = int(header.get("n_heads", 0))
    # CSWG v3 stores tensors as fp16; cast to fp32 lists for the C++ side
    # which expects std::vector<float>. The bf16 cast for AMX dispatch
    # happens inside simplex_policy.cpp, not here.
    weights.W_vp = tensors["W_vp"].to(torch.float32).flatten().tolist()
    weights.b_vp = tensors["b_vp"].to(torch.float32).flatten().tolist()
    weights.W_lh = tensors["W_lh"].to(torch.float32).flatten().tolist()
    weights.b_lh = float(tensors["b_lh"].to(torch.float32).item())
    weights.W_sb = tensors["W_sb"].to(torch.float32).flatten().tolist()
    weights.alpha = float(tensors["alpha"].to(torch.float32).item())
    weights.temperature = float(tensors["temperature"].to(torch.float32).item())
    weights.bucket_embed = (
        tensors["bucket_embed"].to(torch.float32).flatten().tolist()
    )
    weights.lambda_hxh = float(tensors["lambda_hxh"].to(torch.float32).item())
    if weights.n_heads > 0:
        weights.W_q = tensors["W_q"].to(torch.float32).flatten().tolist()
        weights.W_k = tensors["W_k"].to(torch.float32).flatten().tolist()
        weights.W_v = tensors["W_v"].to(torch.float32).flatten().tolist()
        weights.W_o = tensors["W_o"].to(torch.float32).flatten().tolist()
        weights.W_e = tensors["W_e"].to(torch.float32).flatten().tolist()

    learner = _ext.SimplexOnlineLearner(
        num_slots=int(config.get("episodic_capacity", 4096)),
        max_entries_per_slot=int(
            config.get("episodic_controller_history_entries", 64)
        ),
        gamma=float(config.get("episodic_controller_credit_gamma", 0.995)),
        learning_rate=float(
            config.get("episodic_controller_learning_rate", 1.0e-3)
        ),
        sgd_interval=int(config.get("episodic_controller_sgd_interval", 256)),
        ema_alpha=float(config.get("episodic_controller_ema_alpha", 0.25)),
        ema_interval=int(config.get("episodic_controller_ema_interval", 64)),
        gerber_c=float(config.get("episodic_controller_gerber_c", 0.5)),
        lambda_hxh_warmup_events=int(
            config.get("episodic_controller_lambda_hxh_warmup_events", 1024)
        ),
        lambda_hxh_clip=float(
            config.get("episodic_controller_lambda_hxh_clip", 1.0)
        ),
        entropy_beta=float(config.get("episodic_controller_entropy_beta", 0.0)),
    )
    learner.initialize_simplex_weights(weights)
    return learner


def _build_controller_runtime_from_config(
    config: dict[str, Any],
    *,
    capacity: int,
) -> Any | None:
    mode = str(config.get("episodic_controller_runtime", "heuristic")).strip()
    if mode in {"", "heuristic", "python_heuristic"}:
        return None
    if mode == "simplex_v1":
        weights_path = config.get("episodic_controller_weights_path")
        if not weights_path:
            raise ValueError(
                f"episodic_controller_runtime={mode!r} requires "
                "episodic_controller_weights_path (CSWG v3 path produced "
                "by experiments/25_controller_pretrain/pretrain_controller.py)"
            )
        return _build_simplex_learner_from_cswg(
            str(weights_path), config=config
        )
    prefer_cpp = mode in {"cpp", "cpu_ssm_cpp", "cpp_reference"}
    prefer_reference = mode in {"reference", "cpu_ssm_reference", "python_reference"}
    if not prefer_cpp and not prefer_reference:
        raise ValueError(
            "episodic_controller_runtime must be one of "
            "'heuristic', 'cpp_reference', 'cpu_ssm_reference', or "
            "'simplex_v1'; "
            f"got {mode!r}"
        )
    from chaoscontrol.episodic.cpu_ssm_controller import (
        CpuSsmControllerRuntime,
        CpuSsmControllerWeights,
        require_cpp,
    )

    if prefer_cpp:
        require_cpp()

    # Trained-runtime modes (cpp_reference / cpu_ssm_reference) require
    # real weights. The previous default of all-zeros silently produced
    # ``controller_logit ≡ 0`` for every event, indistinguishable from
    # "controller-on, no signal" — the wiring-vs-existence failure mode
    # the design explicitly calls out. Heuristic mode short-circuits
    # above and never reaches this branch.
    weights_path = config.get("episodic_controller_weights_path")
    if not weights_path:
        raise ValueError(
            f"episodic_controller_runtime={mode!r} requires "
            "episodic_controller_weights_path; refusing to construct a "
            "trained runtime with all-zeros default weights (would "
            "silently emit controller_logit=0 for every event)"
        )
    weights = CpuSsmControllerWeights.load(str(weights_path))
    return CpuSsmControllerRuntime(
        weights,
        capacity=int(capacity),
        prefer_cpp=bool(prefer_cpp),
    )


class _OnlineLearningRuntimeBridge:
    """Bridge C++ online updates back into the Python scoring runtime."""

    __slots__ = ("runtime", "learner", "_lock", "_last_sgd_steps")

    def __init__(
        self,
        *,
        runtime: Any,
        capacity: int,
        config: dict[str, Any],
    ) -> None:
        self.runtime = runtime
        self.learner = _ext.OnlineLearningController(
            num_slots=int(capacity),
            max_entries_per_slot=int(
                config.get("episodic_controller_history_entries", 64)
            ),
            gamma=float(config.get("episodic_controller_credit_gamma", 0.995)),
            gerber_c=float(config.get("episodic_controller_gerber_c", 0.5)),
            learning_rate=float(
                config.get("episodic_controller_learning_rate", 1.0e-3)
            ),
            sgd_interval=int(config.get("episodic_controller_sgd_interval", 256)),
            ema_alpha=float(config.get("episodic_controller_ema_alpha", 0.25)),
            ema_interval=int(config.get("episodic_controller_ema_interval", 64)),
        )
        self._lock = threading.Lock()
        self._last_sgd_steps = 0
        self._initialize_learner_from_runtime()

    def score_slot_with_snapshot(
        self,
        features: torch.Tensor,
        *,
        slot: int,
    ) -> Any:
        with self._lock:
            return self.runtime.score_slot_with_snapshot(features, slot=slot)

    def score_slot(self, features: torch.Tensor, *, slot: int) -> torch.Tensor:
        with self._lock:
            return self.runtime.score_slot(features, slot=slot)

    def record_replay_selection(self, **kwargs: Any) -> None:
        with self._lock:
            self.learner.record_replay_selection(**kwargs)

    def on_replay_outcome(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.learner.on_replay_outcome(event)
            telemetry = self.learner.telemetry()
            sgd_steps = int(telemetry.get("sgd_steps", 0))
            if sgd_steps != self._last_sgd_steps:
                self._last_sgd_steps = sgd_steps
                self._sync_runtime_from_fast_weights()

    def _initialize_learner_from_runtime(self) -> None:
        weights = self.runtime.weights.to_dict()
        self.learner.initialize_weights(
            feature_dim=int(self.runtime.weights.feature_dim),
            global_dim=int(self.runtime.weights.global_dim),
            slot_dim=int(self.runtime.weights.slot_dim),
            w_global_in=_flat_float_list(weights["w_global_in"]),
            w_slot_in=_flat_float_list(weights["w_slot_in"]),
            decay_global=_flat_float_list(weights["decay_global"]),
            decay_slot=_flat_float_list(weights["decay_slot"]),
            w_global_out=_flat_float_list(weights["w_global_out"]),
            w_slot_out=_flat_float_list(weights["w_slot_out"]),
            bias=float(weights["bias"].item()),
        )

    def _sync_runtime_from_fast_weights(self) -> None:
        from chaoscontrol.episodic.cpu_ssm_controller import (
            CpuSsmControllerWeights,
        )

        blob = self.learner.fast_weights()
        feature_dim = int(blob["feature_dim"])
        global_dim = int(blob["global_dim"])
        slot_dim = int(blob["slot_dim"])
        self.runtime.weights = CpuSsmControllerWeights(
            w_global_in=torch.tensor(
                blob["w_global_in"],
                dtype=torch.float32,
            ).reshape(global_dim, feature_dim),
            w_slot_in=torch.tensor(
                blob["w_slot_in"],
                dtype=torch.float32,
            ).reshape(slot_dim, feature_dim),
            decay_global=torch.tensor(blob["decay_global"], dtype=torch.float32),
            decay_slot=torch.tensor(blob["decay_slot"], dtype=torch.float32),
            w_global_out=torch.tensor(blob["w_global_out"], dtype=torch.float32),
            w_slot_out=torch.tensor(blob["w_slot_out"], dtype=torch.float32),
            bias=torch.tensor(float(blob["bias"]), dtype=torch.float32),
        )


def _flat_float_list(value: torch.Tensor) -> list[float]:
    return [
        float(x)
        for x in value.detach().to(device="cpu", dtype=torch.float32)
        .reshape(-1)
        .tolist()
    ]


def _wire_online_learning_bridge(
    *,
    consumer: _EpisodicConsumerState,
    controller_runtime: Any | None,
    config: dict[str, Any],
) -> tuple[Any | None, Any | None]:
    """Decide whether to wrap the controller runtime with the C++ online-
    learning bridge for the upcoming spawn.

    Returns ``(action_recorder, controller_runtime_for_thread)``.

    ``controller_train_online`` (default True for backwards compat) gates
    the bridge. F1's frozen arms set it False so the trained controller
    scores from loaded weights without recording snapshots or running
    SGD; the online arms set it True so the bridge wraps the runtime
    and credit attribution + SGD + EMA all run during the 600s training
    window.

    Three runtime shapes:

    1. ``controller_runtime is None`` (heuristic-only) — returns
       ``(None, None)``; nothing to wrap.
    2. ``controller_runtime`` is a V0 ``CpuSsmControllerRuntime`` — when
       train_online, wraps in ``_OnlineLearningRuntimeBridge`` so the
       per-slot REINFORCE updates land back in the runtime's scoring
       weights. When frozen, returns the raw runtime so
       ``score_slot_with_snapshot`` still works for the query path.
    3. ``controller_runtime`` is a V1 ``_ext.SimplexOnlineLearner`` — the
       learner IS the bridge; it carries fast/slow weights and the
       REINFORCE backward path. When train_online, ``consumer.online_learning_bridge``
       points at it so ``_notify_online_learning_bridge`` routes replay
       outcomes via ``on_replay_outcome``. When frozen, ``online_learning_bridge``
       stays unset (no SGD); the controller thread still calls
       ``simplex_forward`` on the learner's frozen weights.
    """
    if controller_runtime is None:
        return None, None
    train_online = bool(config.get("controller_train_online", True))
    is_simplex_learner = isinstance(
        controller_runtime, _ext.SimplexOnlineLearner
    )
    if not train_online:
        print(
            "[runner_fast_path] controller_train_online=False — wrapping "
            "runtime without online-learning bridge; SGD + EMA + history "
            "recording disabled for this run.",
            flush=True,
        )
        return None, controller_runtime
    if is_simplex_learner:
        # Simplex V1: the learner is its own bridge. Set the consumer
        # field so _notify_online_learning_bridge routes replay outcomes
        # to it; pass the learner through to the controller thread as
        # the runtime so simplex_forward is callable on its weights.
        consumer.online_learning_bridge = controller_runtime
        return controller_runtime, controller_runtime
    bridge = _OnlineLearningRuntimeBridge(
        runtime=controller_runtime,
        capacity=int(consumer.cache.capacity),
        config=config,
    )
    consumer.online_learning_bridge = bridge
    return bridge, bridge


def _spawn_episodic_controller(
    *,
    consumer: _EpisodicConsumerState,
    is_episodic_rank: bool,
    episodic_enabled: bool,
    config: dict[str, Any],
) -> _EpisodicControllerHandle | None:
    """Spawn the Phase 2 controller thread on the episodic rank.

    Phase 2 simplification (see ``chaoscontrol.episodic.controller``
    module docstring): in-process daemon ``threading.Thread`` rather
    than ``multiprocessing.spawn`` per Decision 0.7. Pass C's
    ``controller_query_queue`` carries GPU fp32 residual tensors that
    cannot be cheaply marshalled across a process boundary; the plan
    explicitly authorizes the in-process variant. Returns ``None`` on
    every code path that doesn't spawn (train ranks, episodic disabled,
    controller_query_enabled=False, episodic rank with no cache).

    The single-flag gate is intentional: ``controller_query_enabled``
    controls BOTH (a) whether the drain pushes residuals into the
    queue and (b) whether the controller spawns to consume them. This
    matches Pass C's design — gating one without the other would either
    leak GPU residuals (drain on, controller off) or burn CPU on an
    always-empty queue (drain off, controller on).
    """
    if not episodic_enabled:
        return None
    if not is_episodic_rank:
        return None
    if consumer.cache is None:
        return None
    if not consumer.controller_query_enabled:
        return None

    score_mode = _controller_score_mode_from_config(config)
    k = int(config.get("episodic_controller_topk_k", 16))
    idle_sleep_s = float(
        config.get("episodic_controller_idle_sleep_s", 0.005)
    )
    simplex_selection_mode = str(
        config.get("episodic_controller_selection_mode", "argmax")
    ).strip().lower()
    if simplex_selection_mode not in {
        "argmax", "greedy", "sample", "soft_sample", "stochastic",
    }:
        raise ValueError(
            "episodic_controller_selection_mode must be one of "
            "'argmax', 'greedy', 'sample', 'soft_sample', or 'stochastic'; "
            f"got {simplex_selection_mode!r}"
        )
    simplex_generator = None
    if simplex_selection_mode in {"sample", "soft_sample", "stochastic"}:
        simplex_generator = torch.Generator(device="cpu")
        simplex_generator.manual_seed(
            int(config.get("episodic_controller_selection_seed", 0))
        )
    controller_runtime = _build_controller_runtime_from_config(
        config,
        capacity=int(consumer.cache.capacity),
    )
    action_recorder, controller_runtime_for_thread = (
        _wire_online_learning_bridge(
            consumer=consumer,
            controller_runtime=controller_runtime,
            config=config,
        )
    )

    stop_event = threading.Event()
    # The controller thread has its OWN heartbeat (exposed via
    # ``handle.heartbeat``), separate from ``consumer.heartbeat`` (which
    # the episodic-rank step body increments). The runner's outer loop
    # can poll BOTH for independent liveness signals: consumer.heartbeat
    # tracks step-loop drain progress; this one tracks controller-thread
    # progress. A regression that stalled the controller thread without
    # stalling the step loop would surface as drift between the two.
    controller_heartbeat: list[int] = [0]
    thread = threading.Thread(
        target=controller_main,
        kwargs={
            "controller_query_queue": consumer.controller_query_queue,
            "tagged_replay_queue": consumer.tagged_replay_queue,
            "cache": consumer.cache,
            # No queue_lock: the producer side
            # (``_drain_episodic_payloads_gpu``) does plain
            # ``list.append()`` (atomic under the GIL) and the
            # controller's drain uses ``list.pop(0)`` (also atomic).
            # See ``run_controller_cycle`` docstring. Locking only the
            # controller side would not help — the producer doesn't
            # acquire it, so a hypothetical "snapshot + clear" race
            # would still happen. Tests pass an explicit lock for
            # deterministic test-thread synchronization; runtime
            # relies on GIL atomicity of single-bytecode ops.
            "queue_lock": None,
            "stop_event": stop_event,
            "k": k,
            "score_mode": score_mode,
            "controller_runtime": controller_runtime_for_thread,
            "action_recorder": action_recorder,
            "simplex_selection_mode": simplex_selection_mode,
            "simplex_generator": simplex_generator,
            "cycle_idle_sleep_s": idle_sleep_s,
            "heartbeat": controller_heartbeat,
        },
        daemon=True,
        name="episodic_controller",
    )
    thread.start()
    return _EpisodicControllerHandle(
        thread=thread,
        stop_event=stop_event,
        heartbeat=controller_heartbeat,
    )


def _shutdown_episodic_controller(
    handle: _EpisodicControllerHandle | None,
    *,
    join_timeout_s: float = 2.0,
) -> None:
    """Signal the controller thread to stop and join it.

    Called from the runner's ``finally`` block. None handle is a no-op
    so train ranks and ``episodic_enabled=False`` runs pay nothing. If
    the thread doesn't exit within ``join_timeout_s`` seconds we leave
    it running as a daemon — the process exit will reap it. Logging the
    timeout would require the runner's logger; for now the silent
    drop-on-timeout is acceptable for Phase 2 smoke (Phase 4 hardening
    can wire a heartbeat assertion).
    """
    if handle is None:
        return
    handle.stop_event.set()
    handle.thread.join(timeout=join_timeout_s)


def _drain_episodic_payloads_gpu(
    *,
    consumer: _EpisodicConsumerState,
    gather_list: list[torch.Tensor],
    span_length: int,
    key_rep_dim: int,
    k_max: int,
    current_step: int,
    embedding_version: int,
    pre_query_ce: float = float("nan"),
    query_bucket: int = -1,
    controller_score_mode: str = "cosine_utility_weighted",
    controller_topk_k: int = 16,
) -> None:
    """Episodic-rank drain: filter gather_list by valid_mask, route to cache + queue.

    ``gather_list[r]`` is a ``[K_max, slot_dim]`` fp32 tensor — rank r's
    contribution. Iterate ``(rank, k)`` preserving the rank index so the
    per-entry controller-queue row knows which train rank produced it,
    skip rows with ``valid_mask <= 0.5``, and call ``cache.append`` /
    queue-append for each surviving slot.

    Tensor outputs from ``unpack_payload`` are already cloned so the
    gather_list buffers are reusable next step. Scalar fields (``key_fp``,
    ``value_anchor_id``, ``pressure``, ``valid_mask``) sync via
    ``.item()`` — unavoidable since the cache schema and queue both want
    Python scalars.
    """
    if consumer.cache is None:
        return
    cache = consumer.cache
    queue = consumer.controller_query_queue
    for r, gather_tensor in enumerate(gather_list):
        for k in range(int(k_max)):
            slot = gather_tensor[k]
            # Cheap fp32 read on GPU — the .item() call is the real cost
            # but it's per-row and gates the unpack work below.
            if float(slot[0].item()) <= 0.5:
                continue
            unpacked = unpack_payload(
                slot,
                span_length=span_length,
                key_rep_dim=key_rep_dim,
            )
            source_write_id = _rank_prefixed_event_id(
                source_rank=int(r),
                rank_seq=(int(current_step) << 16) | int(k),
            )
            appended_slot = cache.append(
                key_fp=int(unpacked["key_fp"]),
                key_rep=unpacked["key_rep"],
                value_tok_ids=unpacked["value_tok_ids"],
                value_anchor_id=int(unpacked["value_anchor_id"]),
                current_step=int(current_step),
                embedding_version=int(embedding_version),
                pressure_at_write=float(unpacked["pressure"]),
                source_write_id=int(source_write_id),
                write_bucket=int(query_bucket),
            )
            candidate_slot_ids = None
            candidate_cosines = None
            if (
                consumer.query_ring is not None
                and consumer.rank_query_seq is not None
            ):
                candidate_slot_ids, candidate_cosines = (
                    _query_event_simplex_candidates(
                        cache=cache,
                        query_residual=unpacked["residual"],
                        score_mode=str(controller_score_mode),
                        k=int(controller_topk_k),
                    )
                )
            query_event_id = _emit_query_event(
                consumer=consumer,
                source_rank=int(r),
                gpu_step=int(current_step),
                query_residual=unpacked["residual"],
                pressure=float(unpacked["pressure"]),
                pre_query_ce=float(pre_query_ce),
                bucket=int(query_bucket),
                candidate_slot_ids=candidate_slot_ids,
                candidate_cosines=candidate_cosines,
            )
            # Gate queue.append behind the consumer-enabled flag.
            # Phase 1 ships with the flag default False; the queue stays
            # empty (no GPU residual tensors retained, no slow OOM at
            # 600s). Phase 2's controller bring-up flips the flag True at
            # the same time it adds the consumer that drains the queue.
            if consumer.controller_query_enabled:
                queue.append({
                    "step": int(current_step),
                    "rank": int(r),
                    "k": int(k),
                    "query_event_id": int(
                        query_event_id
                        if query_event_id is not None
                        else source_write_id
                    ),
                    "source_write_id": int(source_write_id),
                    "write_bucket": int(query_bucket),
                    "slot": int(appended_slot),
                    "pressure": float(unpacked["pressure"]),
                    "residual": unpacked["residual"],
                })


def _get_tagged_replay_queue(
    consumer: "_EpisodicConsumerState | None",
) -> list[dict[str, Any]]:
    """Return the consumer's ``tagged_replay_queue`` if X has wired it.

    Phase 3.1 (this lane) is implemented in parallel with Phase 2's
    controller bring-up (X's lane). Until X merges, the consumer state
    may not carry a ``tagged_replay_queue`` attribute — ``getattr``
    with a default-empty-list fallback keeps this lane shippable on
    its own. After X merges this becomes a no-op assertion ("X
    promised the field; let's read it directly"). The function is
    intentionally tiny so the merge collapses cleanly into a single
    attribute read.
    """
    if consumer is None:
        return []
    return getattr(consumer, "tagged_replay_queue", [])


def _run_episodic_replay_from_tagged_queue(
    *,
    consumer: "_EpisodicConsumerState | None",
    model: torch.nn.Module,
    current_step: int,
    weight: float,
    lm_head_backward_mode: str,
    lm_head_tile_size: int,
    logger: DiagnosticsLogger | None,
    max_replays_per_step: int = 0,
) -> int:
    """Drain ``consumer.tagged_replay_queue`` and run replay per slot.

    For each tagged entry the controller pushed onto the queue:

    * Snapshot ``utility_pre`` for the diagnostic log row.
    * Call ``dreamworld_replay_from_cache_entry`` — replay grads
      accumulate into ``param.grad`` so the upcoming SUM all-reduce
      sweeps them up alongside the train-rank main grads.
    * Apply Decision 0.10's clamp via the dreamworld helper's
      ``utility_signal_transformed`` and call
      ``cache.update_utility`` so the EMA reflects the replay event.
    * If a logger is wired, emit one NDJSON row carrying the 16
      Decision-0.9 columns.

    Returns the number of tagged entries replayed (0 when the queue
    was empty / cache absent / consumer absent — the runner uses this
    for telemetry).
    """
    if consumer is None or consumer.cache is None:
        return 0
    tagged = _get_tagged_replay_queue(consumer)
    if not tagged:
        return 0
    cache = consumer.cache
    replayed = 0
    max_replays = int(max_replays_per_step)
    # Drain destructively. The controller is the producer; once we
    # consume an entry, it's done — the controller will push fresh
    # entries on the next cycle.
    while tagged and (max_replays <= 0 or replayed < max_replays):
        entry = tagged.pop(0)
        slot = int(entry.get("slot", -1))
        if not (0 <= slot < int(cache.capacity)):
            _emit_replay_outcome(
                consumer=consumer,
                entry=entry,
                current_step=int(current_step),
                slot=int(slot),
                outcome_status=_REPLAY_STATUS_SLOT_MISSING,
                source_write_id=int(entry.get("source_write_id", -1)),
                write_bucket=int(entry.get("write_bucket", -1)),
            )
            continue
        # Snapshot pre-replay utility so the log row pins the value
        # the EMA started from. Read from the underlying tensor field
        # rather than via ``query`` so we don't depend on the
        # fingerprint hash being live.
        if not bool(cache.occupied[slot].item()):
            # Slot evicted between the controller's tag and our
            # drain — race is documented in the design plan.
            _emit_replay_outcome(
                consumer=consumer,
                entry=entry,
                current_step=int(current_step),
                slot=int(slot),
                outcome_status=_REPLAY_STATUS_SLOT_MISSING,
                source_write_id=int(
                    entry.get(
                        "source_write_id",
                        int(cache.source_write_id[slot].item()),
                    )
                ),
                write_bucket=int(entry.get("write_bucket", -1)),
            )
            continue
        utility_pre = float(cache.utility_u[slot].item())
        write_step = int(cache.write_step[slot].item())
        key_fp = int(cache.key_fp[slot].item())
        write_pressure = float(cache.pressure_at_write[slot].item())
        write_bucket = int(cache.write_bucket[slot].item())
        source_write_id = int(
            entry.get(
                "source_write_id",
                int(cache.source_write_id[slot].item()),
            )
        )
        # X's contract says the queue entry carries ``score`` (=
        # cosine × utility). The diagnostic log wants ``query_cosine``
        # (= raw cosine). Until the controller emits the raw cosine
        # separately, log ``score`` and let Phase 3.5 backsolve cosine
        # from ``score / utility_pre``. Default 0.0 if X's contract
        # changes — the column is preserved either way.
        query_cosine = float(entry.get("score", 0.0))

        result = dreamworld_replay_from_cache_entry(
            model=model,
            cache=cache,
            slot=slot,
            current_step=int(current_step),
            weight=float(weight),
            lm_head_backward_mode=lm_head_backward_mode,
            lm_head_tile_size=int(lm_head_tile_size),
        )
        if result is None:
            # Race: slot evicted between our occupancy check and the
            # replay forward. Skip without logging.
            _emit_replay_outcome(
                consumer=consumer,
                entry=entry,
                current_step=int(current_step),
                slot=int(slot),
                outcome_status=_REPLAY_STATUS_SKIPPED,
                source_write_id=int(source_write_id),
                write_bucket=int(write_bucket),
            )
            continue
        # Decision 0.10: feed the clamped value to update_utility so
        # the cache scoring rule (cosine × utility_u) keeps a
        # well-defined ordering.
        #
        # Phase 1 NaN-skip: if utility_signal_raw is NaN (no live
        # rare-grad direction in scope — the default until Phase 4
        # wires the post-allreduce EMA snapshot), DON'T call
        # update_utility. Otherwise every replay would feed 0.0 into
        # the EMA, decaying utility_u by the EMA decay factor each
        # time. Combined with the lowest-utility eviction policy and
        # the cosine × utility_u retrieval rule, replayed entries
        # would get evicted FASTER and down-weighted at retrieval —
        # the cache would preferentially preserve UNUSED entries, and
        # Arm B would collapse to "cosine × anti-frequency-of-replay."
        # Skipping preserves utility_u=1.0 (init) for all entries until
        # Phase 4 lands a real signal. Log row's utility_post then
        # equals utility_pre, which Phase 3.5 readers can detect as
        # "this replay didn't update utility."
        raw = float(result["utility_signal_raw"])
        if not math.isnan(raw):
            cache.update_utility(
                slot, ce_delta=float(result["utility_signal_transformed"]),
            )
        utility_post = float(cache.utility_u[slot].item())
        replay_loss = float(result["replay_loss"])
        replay_outcome_status = (
            _REPLAY_STATUS_OK
            if math.isfinite(replay_loss)
            else _REPLAY_STATUS_NAN
        )
        # Phase B5: when the pre/post CE pair gate is on, stash the
        # value tokens + reference to the just-emitted dict so the
        # post-optimizer-step pass can run a no-grad forward on the
        # post-step weights and patch ``ce_after_replay`` /
        # ``ce_delta_raw`` / ``bucket_baseline`` / ``reward_shaped``
        # in place. The drain owns the EMA update gating: when deferred,
        # the EMA stays untouched until the post-step pass lands on real
        # finite deltas.
        compute_pair = (
            getattr(consumer, "compute_replay_ce_pair", False)
            and replay_outcome_status == _REPLAY_STATUS_OK
        )
        emitted = _emit_replay_outcome(
            consumer=consumer,
            entry=entry,
            current_step=int(current_step),
            slot=int(slot),
            outcome_status=int(replay_outcome_status),
            source_write_id=int(source_write_id),
            write_bucket=int(write_bucket),
            ce_before_replay_override=(
                float(replay_loss) if compute_pair else None
            ),
            ce_after_replay=float("nan"),
            grad_cos_rare=float(result["replay_grad_cos_rare"]),
            grad_cos_total=float(result["replay_grad_cos_total"]),
            defer_reward_shaping=bool(compute_pair),
        )
        if (
            compute_pair
            and emitted is not None
            and consumer.pending_post_step_replays is not None
        ):
            # Snapshot the value-token row off the cache so a slot
            # eviction between drain and post-step pass can't corrupt
            # the second forward. ``cache.value_tok_ids[slot]`` is an
            # int64 row of shape ``[span_length]``; ``.detach().clone()``
            # copies the data without moving it off-device.
            value_row = (
                cache.value_tok_ids[int(slot)].detach().clone()
            )
            staged: dict[str, Any] = {
                "event_dict": emitted,
                "value_tok_ids": value_row,
                "weight": float(weight),
                "lm_head_backward_mode": str(lm_head_backward_mode),
                "lm_head_tile_size": int(lm_head_tile_size),
                "write_bucket": int(write_bucket),
                "ce_before_replay": float(replay_loss),
            }
            if logger is not None:
                staged["ndjson_logger"] = logger
                staged["ndjson_row"] = {
                    "current_step": int(current_step),
                    "slot": int(slot),
                    "key_fp": int(key_fp),
                    "write_step": int(write_step),
                    "write_pressure": float(write_pressure),
                    "write_bucket": int(write_bucket),
                    "query_cosine": float(query_cosine),
                    "utility_pre": float(utility_pre),
                    "utility_post": float(utility_post),
                    "entry": dict(entry),
                    "replay_loss": float(replay_loss),
                    "replay_grad_norm": float(result["replay_grad_norm"]),
                    "replay_grad_cos_common": float(
                        result["replay_grad_cos_common"]
                    ),
                    "replay_grad_cos_rare": float(
                        result["replay_grad_cos_rare"]
                    ),
                    "replay_grad_cos_total": float(
                        result["replay_grad_cos_total"]
                    ),
                    "utility_signal_raw": float(result["utility_signal_raw"]),
                    "utility_signal_transformed": float(
                        result["utility_signal_transformed"]
                    ),
                }
            consumer.pending_post_step_replays.append(staged)

        elif logger is not None:
            _write_replay_ndjson_row(
                logger,
                event_dict=emitted,
                current_step=int(current_step),
                slot=int(slot),
                key_fp=int(key_fp),
                write_step=int(write_step),
                write_pressure=float(write_pressure),
                write_bucket=int(write_bucket),
                query_cosine=float(query_cosine),
                utility_pre=float(utility_pre),
                utility_post=float(utility_post),
                entry=entry,
                replay_loss=float(replay_loss),
                replay_grad_norm=float(result["replay_grad_norm"]),
                replay_grad_cos_common=float(result["replay_grad_cos_common"]),
                replay_grad_cos_rare=float(result["replay_grad_cos_rare"]),
                replay_grad_cos_total=float(result["replay_grad_cos_total"]),
                utility_signal_raw=float(result["utility_signal_raw"]),
                utility_signal_transformed=float(
                    result["utility_signal_transformed"]
                ),
            )
        replayed += 1
    return replayed


def _post_step_replay_ce(
    *,
    model: torch.nn.Module,
    value_tok_ids: torch.Tensor,
) -> float:
    """Compute mean CE on the value-token row using current weights.

    Mirrors ``dreamworld_replay_from_cache_entry``'s pre-step CE
    computation EXCEPT it runs under ``torch.no_grad`` and skips both the
    backward and the loss-weight scaling — the returned scalar is the
    unscaled mean cross-entropy that ``replay_loss`` returns from
    ``full_lm_head_backward`` / ``fused_lm_head_backward``. Keeping the
    reduction and dtype path identical is what makes
    ``ce_before_replay - ce_after_replay`` a meaningful delta.

    **Precision parity caveat.** This helper always materializes the
    full ``logits = lm_head(final_norm(hidden))`` tensor and casts to
    fp32 before CE. The pre-step ``replay_loss`` from
    ``fused_lm_head_backward`` / ``fused_rms_linear_cross_entropy`` may
    take a kernel-fused path with its own intermediate dtype handling.
    At fp32 the two surfaces match; at bf16 the small-span numerics can
    differ by ~1e-3, which is well below the reward-signal magnitudes
    the EMA tracks. If a future debugger sees a mystery delta of that
    magnitude, the place to look is here, not the optimizer.

    Returns the post-step CE as a Python float; NaN if the helper would
    produce a degenerate row (length < 2 leaves no input/target split).
    """
    if value_tok_ids.dim() != 1:
        raise ValueError(
            "_post_step_replay_ce expects a 1-D value-token row; got shape "
            f"{tuple(value_tok_ids.shape)}"
        )
    span_length = int(value_tok_ids.shape[0])
    if span_length < 2:
        return float("nan")
    device = next(model.parameters()).device
    value_row = value_tok_ids.to(device=device).reshape(1, span_length)
    inputs = value_row[:, :-1].to(torch.int32)
    targets = value_row[:, 1:].to(torch.long)
    with torch.no_grad():
        hidden = model.encode(inputs)
        logits = model.lm_head(model.final_norm(hidden))
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction="mean",
        )
    return float(ce.detach().item())


def _run_post_step_replay_ce(
    *,
    consumer: "_EpisodicConsumerState | None",
    model: torch.nn.Module,
) -> int:
    """Drain the deferred post-step CE pass for B5 reward shaping.

    For every replay that the in-step drain staged on
    ``consumer.pending_post_step_replays``, run a no-grad forward on the
    just-updated weights, compute mean CE on the same value tokens the
    pre-step ``replay_loss`` was computed on, and patch the staged
    REPLAY_OUTCOME dict in place. The bucket baseline EMA updates here
    (deferred from the in-step path) so a real finite delta is the only
    signal the EMA ever absorbs.

    Returns the number of patched events (0 when the gate is off, when
    no consumer is wired, or when the pending list is empty). Always
    clears the pending list on exit so the next step starts clean.
    """
    if consumer is None:
        return 0
    pending = getattr(consumer, "pending_post_step_replays", None)
    if pending is None or not pending:
        return 0
    bucket_baseline_ema = getattr(consumer, "bucket_baseline_ema", None)
    patched = 0
    for staged in pending:
        event_dict = staged["event_dict"]
        value_tok_ids = staged["value_tok_ids"]
        ce_before = float(staged["ce_before_replay"])
        bucket = int(staged["write_bucket"])
        ce_after = _post_step_replay_ce(
            model=model,
            value_tok_ids=value_tok_ids,
        )
        baseline = 0.0
        if (
            bucket_baseline_ema is not None
            and 0 <= bucket < len(bucket_baseline_ema)
        ):
            baseline = float(bucket_baseline_ema[bucket])
        ce_delta_raw = ce_before - ce_after
        reward_shaped = ce_delta_raw - baseline
        # Patch before pushing to the B4 ring: shm ring push copies the dict
        # into fixed wire storage, so the old placeholder-list in-place
        # mutation trick would otherwise leave the ring with NaN rewards.
        event_dict["ce_before_replay"] = float(ce_before)
        event_dict["ce_after_replay"] = float(ce_after)
        event_dict["ce_delta_raw"] = float(ce_delta_raw)
        event_dict["bucket_baseline"] = float(baseline)
        event_dict["reward_shaped"] = float(reward_shaped)
        if (
            bucket_baseline_ema is not None
            and 0 <= bucket < len(bucket_baseline_ema)
            and math.isfinite(ce_delta_raw)
        ):
            alpha = 0.05
            bucket_baseline_ema[bucket] = (
                (1.0 - alpha) * float(bucket_baseline_ema[bucket])
                + alpha * float(ce_delta_raw)
            )
        _push_event_ring(
            consumer,
            ring_attr="replay_ring",
            drops_attr="replay_ring_drops",
            event=event_dict,
        )
        _notify_online_learning_bridge(consumer=consumer, event=event_dict)
        ndjson_logger = staged.get("ndjson_logger")
        ndjson_row = staged.get("ndjson_row")
        if ndjson_logger is not None and ndjson_row is not None:
            _write_replay_ndjson_row(
                ndjson_logger,
                event_dict=event_dict,
                **ndjson_row,
            )
        patched += 1
    pending.clear()
    return patched


def _run_train_step(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    precision: str,
    ddp_active: bool,
    world_size: int,
    rank: int = 0,
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
    is_episodic_rank: bool = False,
    all_group: "dist.ProcessGroup | None" = None,
    episodic_emit: EpisodicGpuEmit | None = None,
    episodic_consumer: "_EpisodicConsumerState | None" = None,
    episodic_replay_logger: DiagnosticsLogger | None = None,
    episodic_replay_max_replays_per_step: int = 0,
    episodic_controller_score_mode: str = "cosine_utility_weighted",
    episodic_controller_topk_k: int = 16,
    current_step: int = 0,
    embedding_version: int = 0,
) -> torch.Tensor:
    _reject_unsupported(model)
    # ------------------------------------------------------------------
    # Episodic rank: skip main, prepare slot tensor (all-zeros valid_mask),
    # gather (as dst), drain into cache + controller queue, all-reduce.
    # ------------------------------------------------------------------
    if is_episodic_rank:
        device = inputs.device
        # The gather collective requires every rank to contribute a
        # tensor of identical shape and dtype. The episodic rank's
        # contribution is all zeros (valid_mask=0 for every k), so the
        # train ranks' valid slots are the only ones that survive the
        # drain's filter. Allocate from the emit handle when present
        # (Pass C convention: the runner builds an emit handle on every
        # rank including the episodic one); fall back to a zero tensor
        # of the same shape if the caller forgot to thread an emit
        # handle through.
        if episodic_emit is not None:
            episodic_emit.slot_tensor.zero_()
            slot_self = episodic_emit.slot_tensor
            k_max = int(episodic_emit.k_max)
            span_length = int(episodic_emit.span_length)
            key_rep_dim = int(episodic_emit.key_rep_dim)
        else:
            slot_self = None
            k_max = 0
            span_length = 0
            key_rep_dim = 0
        # Gather: episodic rank is the destination, allocates the
        # gather_list. Train ranks pass gather_list=None.
        gather_list: list[torch.Tensor] | None = None
        if (
            slot_self is not None
            and ddp_active
            and world_size > 1
            and all_group is not None
        ):
            gather_list = [
                torch.zeros_like(slot_self) for _ in range(int(world_size))
            ]
            dist.gather(
                slot_self,
                gather_list=gather_list,
                dst=int(world_size) - 1,
                group=all_group,
            )
        # Drain: filter by valid_mask, route each surviving row to
        # ``cache.append`` and the controller queue. Drain runs BEFORE
        # the all-reduce so the cache state is up to date before the
        # optimizer step (Phase 3+ replay backward will read the cache
        # post-drain).
        if (
            episodic_consumer is not None
            and episodic_consumer.cache is not None
            and gather_list is not None
        ):
            _drain_episodic_payloads_gpu(
                consumer=episodic_consumer,
                gather_list=gather_list,
                span_length=span_length,
                key_rep_dim=key_rep_dim,
                k_max=k_max,
                current_step=int(current_step),
                embedding_version=int(embedding_version),
                controller_score_mode=str(episodic_controller_score_mode),
                controller_topk_k=int(episodic_controller_topk_k),
            )
        # Phase 3.1: drain ``tagged_replay_queue`` (X's controller
        # fills it during Phase 2). Each tagged slot triggers one
        # replay backward; grads accumulate into ``param.grad`` so
        # the SUM all-reduce below sweeps them up alongside the train
        # ranks' main_avg contributions. The diagnostic logger emits
        # one Decision-0.9 row per replay event.
        #
        # ``dreamworld_weight`` from the runner's outer loop carries
        # the existing replay weight knob. Phase 3 falsifier matrix
        # sets this per arm; Arm C runs the topology with weight=0 so
        # replay backward fires but the loss-scaled CE contributes
        # zero grad — establishes the 3+1 topology baseline without
        # any DW signal. Single-process unit tests override the kwarg
        # explicitly.
        if episodic_consumer is not None and episodic_consumer.cache is not None:
            _run_episodic_replay_from_tagged_queue(
                consumer=episodic_consumer,
                model=model,
                current_step=int(current_step),
                weight=float(dreamworld_weight),
                lm_head_backward_mode=lm_head_backward_mode,
                lm_head_tile_size=int(lm_head_tile_size),
                logger=episodic_replay_logger,
                max_replays_per_step=int(episodic_replay_max_replays_per_step),
            )
        if episodic_consumer is not None:
            episodic_consumer.heartbeat[0] += 1
        # Train ranks reach the all-reduce after backward; the episodic
        # rank skips backward, so we must enter the same collective with
        # all-None grads → ``materialize_zeros=True`` fills them.
        if ddp_active and world_size > 1 and all_group is not None:
            n_train = world_size - 1
            if n_train < 1:
                raise ValueError(
                    "episodic rank topology requires world_size >= 2 "
                    f"(1 train + 1 episodic), got world_size={world_size}"
                )
            allreduce_grads(
                model,
                world_size,
                group=all_group,
                op=dist.ReduceOp.SUM,
                materialize_zeros=True,
            )
        return torch.zeros((), device=device, dtype=torch.float32)
    # ------------------------------------------------------------------
    # Train rank: forward + backward, pack slot tensor, gather, all-reduce.
    # ------------------------------------------------------------------
    # Per-token CE is captured only when the episodic producer is wired —
    # the standard fast path stays on ``fused_lm_head_backward`` (which
    # discards per-token CE) so the cost of the ``_with_ce`` variant is
    # not paid by ``episodic_enabled=False`` runs.
    per_token_ce_for_episodic: torch.Tensor | None = None
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
            if episodic_emit is not None:
                # Episodic-enabled fast path — the fused kernel computes
                # per-token CE on the way to the loss anyway, so swap to
                # the ``_with_ce`` sibling and stop discarding it.
                loss, per_token_ce_for_episodic = fused_lm_head_backward_with_ce(
                    hidden=hidden,
                    final_norm=model.final_norm,
                    lm_head=model.lm_head,
                    targets=targets,
                    tile_size=int(lm_head_tile_size),
                    backend=backend_name,
                )
            else:
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
            if all_group is not None:
                # 3+1 episodic topology — train ranks pre-scale their
                # grads by 1/N_train so the SUM collective with the
                # zero-materialized episodic rank reconstructs the
                # train-rank average. The ``allreduce_grads`` interface
                # owns the materialize_zeros zero-fill on the episodic
                # rank above; this branch handles the train-rank pre-
                # scale that pairs with it. Pre-scaling happens BEFORE
                # the all-reduce, so post-collective grad clipping
                # operates on the same train-rank-average value as the
                # legacy AVG path produced — clipping decisions are
                # unchanged in expectation.
                n_train = world_size - 1
                if n_train < 1:
                    raise ValueError(
                        "episodic rank topology requires world_size >= 2 "
                        f"(1 train + 1 episodic), got world_size={world_size}"
                    )
                inv_n_train = 1.0 / float(n_train)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(inv_n_train)
                # Pack slot tensor + gather BEFORE the SUM all-reduce
                # per the design doc order. ``episodic_emit is None`` on
                # ``episodic_enabled=False`` runs, in which case the
                # gather is skipped entirely (back-compat).
                if episodic_emit is not None:
                    if per_token_ce_for_episodic is None:
                        # Non-fused LM-head modes don't return per-token
                        # CE; recompute it under no_grad so the producer
                        # can score positions. This is off the documented
                        # submission fast path (which uses one of the
                        # fused modes), so the extra forward is acceptable.
                        with torch.no_grad():
                            logits_episodic = model.lm_head(
                                model.final_norm(hidden.detach())
                            )
                            vocab = logits_episodic.size(-1)
                            per_token_ce_for_episodic = F.cross_entropy(
                                logits_episodic.reshape(-1, vocab).float(),
                                targets.reshape(-1),
                                reduction="none",
                            ).reshape_as(targets)
                    # Phase 1 uses uniform pressure (= 1.0 everywhere).
                    # With the default ``episodic_top_p = 1 / (B * T)``
                    # selection rule this picks the highest-CE position
                    # per step, which matches the design intent
                    # (most-surprising target). Real ScOpt pressure is
                    # incompatible with episodic_enabled in Phase 1
                    # (caller-side guard), and a richer pressure source
                    # is Phase 2+.
                    per_token_ce_bt = per_token_ce_for_episodic.detach().reshape(
                        targets.shape
                    )
                    pressure_bt = torch.ones_like(per_token_ce_bt)
                    _emit_episodic_payloads_gpu(
                        emit=episodic_emit,
                        inputs=inputs,
                        targets=targets,
                        pressure=pressure_bt,
                        per_token_ce=per_token_ce_bt,
                        hidden=hidden,
                        rank=int(rank),
                        world_size=int(world_size),
                        all_group=all_group,
                        current_step=int(current_step),
                    )
                allreduce_grads(
                    model,
                    world_size,
                    group=all_group,
                    op=dist.ReduceOp.SUM,
                    materialize_zeros=True,
                )
            else:
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
    elif episodic_emit is not None:
        # Single-rank back-compat: the test path with ``ddp_active=False``
        # exercises packing without a gather (``all_group is None``), so
        # the runner can verify slot_tensor contents directly. This is
        # how the unit tests pin the producer's pack semantics without
        # standing up a process group.
        if per_token_ce_for_episodic is None:
            with torch.no_grad():
                logits_episodic = model.lm_head(
                    model.final_norm(hidden.detach())
                )
                vocab = logits_episodic.size(-1)
                per_token_ce_for_episodic = F.cross_entropy(
                    logits_episodic.reshape(-1, vocab).float(),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
        per_token_ce_bt = per_token_ce_for_episodic.detach().reshape(targets.shape)
        pressure_bt = torch.ones_like(per_token_ce_bt)
        _emit_episodic_payloads_gpu(
            emit=episodic_emit,
            inputs=inputs,
            targets=targets,
            pressure=pressure_bt,
            per_token_ce=per_token_ce_bt,
            hidden=hidden,
            rank=int(rank),
            world_size=int(world_size),
            all_group=all_group,
            current_step=int(current_step),
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
    # CUDA graph path is unreachable under ``episodic_enabled=True``
    # because the asymmetric topology requires DDP (world_size >= 2)
    # and ``_cuda_graph_rejection_reasons`` rejects ddp_active. Pass
    # ``episodic_enabled=False`` explicitly so the kwarg is wired even
    # though the value is constant here — keeps the call sites uniform
    # if a future task lifts the DDP-CUDA-graph constraint.
    timing = summarize_train_timing(
        steps=steps,
        elapsed_s=elapsed_s,
        batch_size=batch_size,
        seq_len=seq_len,
        world_size=world_size_,
        episodic_enabled=False,
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
    lm_head_emit_entropy: bool = False,
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
    criticality_distill_enabled: bool = False,
    criticality_distill_num_layers: int | None = None,
    criticality_distill_dim: int | None = None,
    criticality_distill_budget_frac: float = 0.15,
    criticality_distill_critical_value: float = 0.95,
    criticality_distill_trace_ttl_steps: int = 1024,
    criticality_distill_trace_half_life_steps: float = 256.0,
    criticality_distill_seat_refresh_interval: int = 64,
    criticality_distill_min_weighted_events_per_layer: float = 256.0,
    criticality_distill_horizon_H: int = 16,
    criticality_distill_event_frac: float = 0.05,
    criticality_distill_weight: float = 1e-3,
    criticality_distill_uniform_pressure: bool = False,
    criticality_distill_score_permute_before_topk: bool = False,
    criticality_distill_fixed_random_seats: bool = False,
    rare_bucket_ce_enabled: bool = False,
    rare_bucket_ce_num_buckets: int = 4,
    rare_bucket_ce_token_frequencies: torch.Tensor | None = None,
    rare_bucket_ce_eval_tokens: torch.Tensor | None = None,
    rare_bucket_ce_eval_num_tokens: int | None = None,
    emit_topology_snapshot: bool = False,
    episodic_enabled: bool = False,
    episodic_top_p: float | None = None,
    episodic_fingerprint_window: int = 8,
    episodic_span_length: int = 4,
    episodic_key_rep_dim: int | None = None,
    episodic_k_max: int = _DEFAULT_EPISODIC_K_MAX,
    episodic_capacity: int = 4096,
    episodic_grace_steps: int = 1000,
    episodic_utility_ema_decay: float = 0.99,
    controller_query_enabled: bool = False,
    episodic_event_log_enabled: bool = False,
    episodic_compute_replay_ce_pair: bool = False,
    episodic_controller_score_mode: str = "cosine_utility_weighted",
    episodic_controller_topk_k: int = 16,
    episodic_controller_idle_sleep_s: float = 0.005,
    episodic_controller_selection_mode: str = "argmax",
    episodic_controller_selection_seed: int | None = None,
    episodic_controller_runtime: str = "heuristic",
    episodic_controller_weights_path: str | None = None,
    episodic_controller_global_dim: int = 8,
    episodic_controller_slot_dim: int = 4,
    episodic_controller_learning_rate: float = 1.0e-3,
    episodic_controller_sgd_interval: int = 256,
    episodic_controller_ema_alpha: float = 0.25,
    episodic_controller_ema_interval: int = 64,
    episodic_controller_credit_gamma: float = 0.995,
    episodic_controller_gerber_c: float = 0.5,
    episodic_controller_lambda_hxh_warmup_events: int = 1024,
    episodic_controller_lambda_hxh_clip: float = 1.0,
    episodic_controller_entropy_beta: float = 0.0,
    episodic_controller_history_entries: int = 64,
    episodic_replay_max_replays_per_step: int = 0,
) -> dict[str, Any]:
    rank_ = int(rank)
    world_size_ = int(world_size)
    ddp_active = world_size_ > 1
    if criticality_distill_enabled and ddp_active:
        raise ValueError(
            "criticality_distill_enabled=True is single-rank only in this "
            "runner. CD evidence, seat allocation, and the separate "
            "cd_loss.backward() path are rank-local; running on "
            f"world_size={world_size_} would produce divergent log_a updates "
            "and non-comparable ranks. Add collectives for CD aggregates + "
            "seat masks + criticality_loss before enabling on multi-rank."
        )
    # Pre-collective episodic guards. These fire BEFORE
    # ``broadcast_params`` so a misconfigured combination raises the real
    # config error rather than a downstream "default process group has
    # not been initialized" — and so they're directly testable via the
    # same ``pytest.raises`` pattern as the CD guard above. The four
    # episodic guards landed in Task 1.3 (lines ~1903-1973) intentionally
    # stay where they are; they fire after ``broadcast_params`` and gate
    # on optimizer / aux-objective state that's only known by then.
    if episodic_enabled and str(train_sampling_mode).strip().lower() in {
        "sequential_epoch",
        "shuffled_epoch",
    }:
        raise ValueError(
            "episodic_enabled=True is incompatible with "
            f"train_sampling_mode={train_sampling_mode!r} in Phase 1: the "
            "episodic rank receives a 1/N start-shard from "
            "count_sharded_lm_starts but the skip-main flow makes it "
            "silently drop that shard, so 1/N of the dataset goes unseen "
            "each epoch and breaks the Phase 1.6 zero-behavior-change "
            "comparison. Workaround: set train_sampling_mode='random' for "
            "Phase 1 episodic runs; the proper fix (re-shard over N-1) is "
            "tracked as a Phase 3 prerequisite."
        )
    if episodic_enabled and (
        float(dreamworld_weight) > 0.0 or int(dreamworld_cache_interval) > 0
    ):
        raise ValueError(
            "episodic_enabled=True is incompatible with "
            f"dreamworld_weight={dreamworld_weight} / "
            f"dreamworld_cache_interval={dreamworld_cache_interval} in "
            "Phase 1. Defaults (both 0) are explicitly safe. Setting "
            "either knob fires capture_dream_entry on every rank "
            "including the episodic rank (wasteful) and would run "
            "dreamworld_replay_backward on train ranks but not the "
            "episodic rank (skip-main early-return), violating the "
            "Phase 1 invariant that 'episodic submits zeros, train ranks "
            "submit only main grads'. Proper Dreamworld integration with "
            "the curated-replay cache is Phase 3 work (Task 3.1)."
        )
    if ddp_active:
        broadcast_params(model)
    # 3+1 (train ranks + episodic) topology setup. ``episodic_enabled``
    # selects the asymmetric replay topology where rank world_size-1
    # skips the main forward+backward and (in Phase 3+) runs replay
    # backward instead. Phase 1 introduces the structural plumbing —
    # the episodic rank's grads are zero-filled in the SUM collective
    # so ``(main_avg + 0) = main_avg``; Phase 3 replaces the zero-fill
    # with a real replay grad so ``(main_avg + replay_full)`` flows in
    # the same single collective with no API change at the call sites.
    is_episodic_rank = False
    all_group = None
    if episodic_enabled:
        if not ddp_active:
            raise ValueError(
                "episodic_enabled=True requires world_size > 1 (need at "
                "least 1 train rank + 1 episodic rank); got "
                f"world_size={world_size_}"
            )
        if world_size_ < 2:
            raise ValueError(
                "episodic_enabled=True requires world_size >= 2 "
                f"(1 train + 1 episodic), got world_size={world_size_}"
            )
        is_episodic_rank = (rank_ == world_size_ - 1)
        # ``all_group`` is the all-rank process group as an explicit
        # handle. Phase 5 will introduce ``main_group`` (just train
        # ranks) without changing any all_group call sites. Note:
        # ``dist.new_group`` is itself a WORLD-collective on gloo/nccl
        # — every rank in WORLD must call it, even ranks not in the
        # subgroup. Here all_group is the world group, so all ranks
        # participate.
        all_group = dist.new_group(list(range(world_size_)))
    grad_allreduce_mode_ = str(grad_allreduce_mode).strip().lower()
    if grad_allreduce_mode_ not in {"bulk", "async_param"}:
        raise ValueError(
            "grad_allreduce_mode must be 'bulk' or 'async_param', "
            f"got {grad_allreduce_mode!r}"
        )
    scopt_active = isinstance(optimizer, ScarcityAwareOptimizer)
    if scopt_active and grad_allreduce_mode_ != "bulk":
        raise ValueError("ScOpt currently requires grad_allreduce_mode='bulk'")
    if episodic_enabled and grad_allreduce_mode_ != "bulk":
        raise ValueError(
            "episodic_enabled=True requires grad_allreduce_mode='bulk'; "
            "the 3+1 SUM all-reduce path is the only collective wired "
            "for the asymmetric topology in Phase 1."
        )
    if episodic_enabled and scopt_active:
        raise ValueError(
            "episodic_enabled=True is incompatible with the ScOpt "
            "optimizer in Phase 1: the ScOpt runner branch does not yet "
            "implement the episodic rank's skip-main/replay path. Add the "
            "episodic branch before combining ScOpt with the asymmetric "
            "topology."
        )
    if episodic_enabled and predictive_aux_weight > 0.0:
        raise ValueError(
            "episodic_enabled=True is incompatible with "
            "predictive_aux_weight > 0.0 in Phase 1: the predictive "
            "aux projection's grad all-reduce at the runner's outer "
            "loop still uses AVG-over-WORLD, which would over-divide "
            "by N when the episodic rank contributes None. Migrate "
            "the predictive aux all-reduce to the SUM/all_group path "
            "before combining the two."
        )
    if episodic_enabled and event_sleep_enabled:
        raise ValueError(
            "episodic_enabled=True is incompatible with "
            "event_sleep_enabled=True in Phase 1: the EMA loss-pressure "
            "gate would feed the episodic rank's zero placeholder loss "
            "into its update, corrupting the cross-rank decision. The "
            "Phase-3 replay path will replace the placeholder; gate "
            "this combination on that work landing."
        )
    # Pass C — episodic-rank consumer init. Builds the cache and the
    # in-process controller_query_queue on the episodic rank only; on
    # train ranks and ``episodic_enabled=False`` the helper returns the
    # no-op state and the runner's outer loop treats it as a skip. No
    # init-time barrier is needed under Pass C: the per-step
    # ``dist.gather`` is itself the rendezvous, and there's no shm to
    # coordinate.
    _model_dim_for_episodic = (
        int(getattr(model, "dim", 0))
        or int(getattr(model.lm_head, "in_features", 0))
    )
    _episodic_consumer_config = {
        "episodic_capacity": int(episodic_capacity),
        "episodic_span_length": int(episodic_span_length),
        "episodic_key_rep_dim": (
            int(episodic_key_rep_dim)
            if episodic_key_rep_dim is not None
            else _model_dim_for_episodic
        ),
        "episodic_grace_steps": int(episodic_grace_steps),
        "episodic_utility_ema_decay": float(episodic_utility_ema_decay),
        "episodic_fingerprint_window": int(episodic_fingerprint_window),
        "controller_query_enabled": bool(controller_query_enabled),
        "episodic_event_log_enabled": bool(episodic_event_log_enabled),
        "episodic_compute_replay_ce_pair": bool(
            episodic_compute_replay_ce_pair
        ),
    }
    episodic_consumer = _attach_episodic_consumer(
        episodic_enabled=bool(episodic_enabled),
        is_episodic_rank=bool(is_episodic_rank),
        world_size=int(world_size_),
        config=_episodic_consumer_config,
        model_dim=_model_dim_for_episodic or 1,
        all_group=all_group,
    )
    # Initialize the controller handle to None up front so the outer
    # ``finally`` block sees a defined name even if a pre-spawn raise
    # exits the function. The actual spawn happens just before the
    # ``try:`` block at the start of the train loop, AFTER all init
    # guards have fired — that way a config-validation raise can't
    # leak a running daemon thread.
    episodic_controller_handle: _EpisodicControllerHandle | None = None
    _episodic_controller_config = {
        "episodic_controller_score_mode": str(episodic_controller_score_mode),
        "episodic_controller_topk_k": int(episodic_controller_topk_k),
        "episodic_controller_idle_sleep_s": float(
            episodic_controller_idle_sleep_s
        ),
        "episodic_controller_selection_mode": str(
            episodic_controller_selection_mode
        ),
        "episodic_controller_selection_seed": (
            int(episodic_controller_selection_seed)
            if episodic_controller_selection_seed is not None
            else int(seed) + rank_
        ),
        "episodic_controller_runtime": str(episodic_controller_runtime),
        "episodic_controller_weights_path": episodic_controller_weights_path,
        "episodic_controller_global_dim": int(episodic_controller_global_dim),
        "episodic_controller_slot_dim": int(episodic_controller_slot_dim),
        "episodic_controller_learning_rate": float(
            episodic_controller_learning_rate
        ),
        "episodic_controller_sgd_interval": int(
            episodic_controller_sgd_interval
        ),
        "episodic_controller_ema_alpha": float(episodic_controller_ema_alpha),
        "episodic_controller_ema_interval": int(episodic_controller_ema_interval),
        "episodic_controller_credit_gamma": float(
            episodic_controller_credit_gamma
        ),
        "episodic_controller_gerber_c": float(episodic_controller_gerber_c),
        "episodic_controller_lambda_hxh_warmup_events": int(
            episodic_controller_lambda_hxh_warmup_events
        ),
        "episodic_controller_lambda_hxh_clip": float(
            episodic_controller_lambda_hxh_clip
        ),
        "episodic_controller_entropy_beta": float(
            episodic_controller_entropy_beta
        ),
        "episodic_controller_history_entries": int(
            episodic_controller_history_entries
        ),
    }
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
    # Criticality Distillation requires the entropy-emitting LM-head forward.
    # See docs/plans/2026-04-24-criticality-distillation-runner-design.md.
    if criticality_distill_enabled and not lm_head_emit_entropy:
        raise ValueError(
            "criticality_distill_enabled=True requires lm_head_emit_entropy=True. "
            "CD uses per-token entropy from the fused forward to build surprise pressure; "
            "pass lm_head_emit_entropy=True (the entrypoint select flag is orthogonal to "
            "lm_head_backward_mode, which stays at its default)."
        )
    cd: CriticalityDistillation | None = None
    cd_ssm_cores: list[torch.nn.Module] = []
    cd_pinned_buffers: dict[str, dict[str, torch.Tensor]] | None = None
    criticality_distillation_diagnostics: list[dict] = []
    criticality_distill_loss_trajectory: list[dict] = []
    if criticality_distill_enabled:
        if criticality_distill_num_layers is None:
            raise ValueError(
                "criticality_distill_enabled=True requires "
                "criticality_distill_num_layers (the CD module needs to know "
                "how many SSM layers to score)."
            )
        if criticality_distill_dim is None:
            raise ValueError(
                "criticality_distill_enabled=True requires "
                "criticality_distill_dim (per-layer recurrence-state dim)."
            )
        cd_ssm_cores = [
            m for m in model.modules() if hasattr(m, "capture_states")
        ]
        if len(cd_ssm_cores) != int(criticality_distill_num_layers):
            raise ValueError(
                "criticality_distill_num_layers mismatch: configured "
                f"{int(criticality_distill_num_layers)} but model exposes "
                f"{len(cd_ssm_cores)} modules with capture_states(). "
                "Every ssm_cores layer must have a capture_states context manager."
            )
        cd = CriticalityDistillation(
            num_layers=int(criticality_distill_num_layers),
            dim=int(criticality_distill_dim),
            trace_ttl_steps=int(criticality_distill_trace_ttl_steps),
            trace_half_life_steps=float(criticality_distill_trace_half_life_steps),
            seat_refresh_interval=int(criticality_distill_seat_refresh_interval),
            criticality_budget_frac=float(criticality_distill_budget_frac),
            critical_value=float(criticality_distill_critical_value),
            min_weighted_events_per_layer=float(
                criticality_distill_min_weighted_events_per_layer
            ),
            criticality_distill_weight=float(criticality_distill_weight),
            score_permute_before_topk=bool(
                criticality_distill_score_permute_before_topk
            ),
            fixed_random_seats=bool(criticality_distill_fixed_random_seats),
        )
        cd.to(device=device)
        # Simple call counters — the test asserts against these.
        cd._ingest_call_count = 0
        cd._seat_refresh_call_count = 0
        cd_pinned_buffers = _alloc_pinned_evidence_buffers(
            num_layers=int(criticality_distill_num_layers),
            dim=int(criticality_distill_dim),
            use_pinned=bool(torch.cuda.is_available()),
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
    topology_snapshot: dict | None = None
    if emit_topology_snapshot:
        topology_snapshot = _capture_topology_snapshot()
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

    # Episodic-cache GPU emit handle (Perf Pass C). Replaces the POSIX
    # shm ring setup from Phase 1 Tasks 1.4 + 1.5. Every rank (train AND
    # episodic) gets its own ``[K_max, slot_dim]`` slot tensor for the
    # ``dist.gather`` collective; train ranks pack into theirs, the
    # episodic rank's stays all-zeros (the gather symmetric-shape
    # requirement). Returns ``None`` for ``episodic_enabled=False`` or
    # ``world_size == 1`` runs.
    episodic_emit_handle: EpisodicGpuEmit | None = None
    if episodic_enabled:
        # Default ``episodic_key_rep_dim`` to the model's hidden dim if
        # the caller didn't pin it. ``model.dim`` is the canonical name
        # on real models; the toy CPU smoke models expose ``dim`` via
        # the embedding shape so probe both.
        resolved_key_rep_dim = (
            int(episodic_key_rep_dim)
            if episodic_key_rep_dim is not None
            else int(getattr(model, "dim", 0)) or int(model.lm_head.in_features)
        )
        emit_config = {
            "episodic_enabled": True,
            "episodic_top_p": (
                float(episodic_top_p)
                if episodic_top_p is not None
                else float("nan")
            ),
            "episodic_fingerprint_window": int(episodic_fingerprint_window),
            "episodic_span_length": int(episodic_span_length),
            "episodic_key_rep_dim": resolved_key_rep_dim,
            "episodic_k_max": int(episodic_k_max),
            "episodic_event_log_enabled": bool(episodic_event_log_enabled),
            "model_dim": resolved_key_rep_dim,
        }
        episodic_emit_handle = _create_episodic_emit(
            rank=rank_,
            world_size=world_size_,
            device=device,
            config=emit_config,
        )

    # Phase 2: spawn the CPU controller thread on the episodic rank
    # when controller_query_enabled=True. Returns None on every other
    # code path (train ranks, episodic disabled, controller flag off).
    # The spawn lands here — AFTER all init guards have fired — so a
    # config-validation raise above can't leak a running daemon thread.
    # The handle is consumed by ``_shutdown_episodic_controller`` in
    # the outer ``finally`` block.
    episodic_controller_handle = _spawn_episodic_controller(
        consumer=episodic_consumer,
        is_episodic_rank=bool(is_episodic_rank),
        episodic_enabled=bool(episodic_enabled),
        config=_episodic_controller_config,
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
            # Stage captured SSM states for CD. Must enter the
            # capture_states() contexts BEFORE the encode call in
            # _run_train_step so the cores write into _captured_states.
            # We keep the stack open across the train-step call and
            # read getters BEFORE exiting — the real ChaosSSMCore clears
            # _captured_states in its context finally block.
            cd_state_getters: list = []
            cd_stack = contextlib.ExitStack()
            if cd is not None:
                for core in cd_ssm_cores:
                    getter = cd_stack.enter_context(core.capture_states())
                    cd_state_getters.append(getter)
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
                        all_group=all_group,
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
                        all_group=all_group,
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
                    rank=rank_,
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
                    is_episodic_rank=is_episodic_rank,
                    all_group=all_group,
                    episodic_emit=episodic_emit_handle,
                    episodic_consumer=episodic_consumer,
                    episodic_replay_max_replays_per_step=int(
                        episodic_replay_max_replays_per_step
                    ),
                    episodic_controller_score_mode=str(
                        episodic_controller_score_mode
                    ),
                    episodic_controller_topk_k=int(episodic_controller_topk_k),
                    current_step=steps,
                    embedding_version=0,
                    dreamworld_entry=dream_entry,
                    dreamworld_weight=(
                        float(event_sleep_weight)
                        if event_dream_replay and float(event_sleep_weight) > 0.0
                        else dreamworld_weight
                    ),
                    dreamworld_replay_batch_size=dreamworld_replay_batch_size,
                    dreamworld_generator=rng,
                )
            # ------------------------------------------------------------
            # Criticality Distillation: ingest + seat refresh + loss add.
            # The capture_states() stack is still open, so reading the
            # getters is safe. After ingest we close the stack so the
            # real ChaosSSMCore can clear its _captured_states per contract.
            # ------------------------------------------------------------
            if cd is not None:
                cd_states_per_layer = [g() for g in cd_state_getters]
                if any(s is None for s in cd_states_per_layer):
                    cd_stack.close()
                    raise RuntimeError(
                        "Criticality Distillation: one or more SSM cores did "
                        "not populate _captured_states during forward. Check "
                        "that each core.a_mode='diag' and that the forward "
                        "pass actually ran under capture_states()."
                    )
                target_device = cd_states_per_layer[0].device
                # Compute per-token CE + per-token entropy for pressure.
                # On CUDA with a fully-built extension we use the fused
                # entropy-emitting kernel; otherwise fall back to plain
                # softmax. The fallback triggers on (a) CPU targets,
                # (b) no CUDA runtime, (c) missing `_C` module, OR
                # (d) the extension built without the entropy-emitting
                # entrypoint (stale install) — case (d) raises at the
                # call site, not at import, so we wrap BOTH. Stage D.4
                # pins the kernel's numerical correctness.
                per_token_ce_bt: torch.Tensor | None = None
                per_token_entropy_bt: torch.Tensor | None = None
                use_fused_entropy = (
                    target_device.type == "cuda" and torch.cuda.is_available()
                )
                fused_entropy_fn = None
                if use_fused_entropy:
                    try:
                        from chaoscontrol.kernels._lm_head_loss import (
                            fused_lm_head_forward_with_ce_entropy as fused_entropy_fn,
                        )
                    except Exception:
                        use_fused_entropy = False
                with torch.no_grad():
                    hidden_cd = model.encode(inputs)
                    normed_cd = model.final_norm(hidden_cd)
                    B_cd, T_cd = inputs.shape[0], inputs.shape[1]
                    if use_fused_entropy and fused_entropy_fn is not None:
                        try:
                            (
                                _loss_ignore,
                                _lse_ignore,
                                per_token_ce_flat,
                                per_token_entropy_flat,
                            ) = fused_entropy_fn(
                                normed_cd,
                                model.lm_head.weight,
                                targets,
                                tile_size=int(lm_head_tile_size),
                            )
                            per_token_ce_bt = per_token_ce_flat.reshape(B_cd, T_cd)
                            per_token_entropy_bt = per_token_entropy_flat.reshape(
                                B_cd, T_cd
                            )
                        except (AttributeError, RuntimeError):
                            # Stale extension missing the entropy entrypoint,
                            # or kernel launch failure — fall through to the
                            # softmax path. Same codepath as CPU.
                            use_fused_entropy = False
                    if not (use_fused_entropy and per_token_ce_bt is not None):
                        logits_cd = normed_cd @ model.lm_head.weight.t()
                        V_cd = logits_cd.shape[-1]
                        per_token_ce_flat = F.cross_entropy(
                            logits_cd.reshape(-1, V_cd),
                            targets.reshape(-1),
                            reduction="none",
                        )
                        probs_cd = F.softmax(logits_cd, dim=-1)
                        per_token_entropy_flat = -(
                            probs_cd * probs_cd.clamp_min(1e-12).log()
                        ).sum(dim=-1)
                        per_token_ce_bt = per_token_ce_flat.reshape(B_cd, T_cd)
                        per_token_entropy_bt = per_token_entropy_flat.reshape(
                            B_cd, T_cd
                        )
                    if criticality_distill_uniform_pressure:
                        cd_pressure = torch.ones(
                            B_cd, T_cd, device=target_device, dtype=torch.float32
                        )
                    else:
                        cd_pressure = compute_ce_minus_entropy_pressure_from_fused(
                            per_token_ce_bt, per_token_entropy_bt
                        )
                    cd_prepared_gpu = cd.ingest_gpu(
                        pressure=cd_pressure,
                        states_per_layer=cd_states_per_layer,
                        horizon_H=int(criticality_distill_horizon_H),
                        event_frac=float(criticality_distill_event_frac),
                    )
                # Close the capture_states stack now that we've read everything.
                cd_stack.close()
                # Async D2H copy into the pinned ping-pong slot. Parity is
                # keyed on the monotonically-advancing ingest counter — not
                # the step index — so a skipped step (OOM retry, conditional
                # early-exit) cannot desync "slot A's last copy completed"
                # from "slot A is about to be reused."
                parity_key = "A" if cd._ingest_call_count % 2 == 0 else "B"
                host_slot = cd_pinned_buffers[parity_key]
                for key, host_t in host_slot.items():
                    src = cd_prepared_gpu[key]
                    if src.device == host_t.device:
                        host_t.copy_(src)
                    else:
                        host_t.copy_(src, non_blocking=True)
                if target_device.type == "cuda" and torch.cuda.is_available():
                    copy_done = torch.cuda.Event()
                    copy_done.record()
                    copy_done.synchronize()
                cd.ingest_cpu_from_prepared(step=steps, prepared=host_slot)
                cd._ingest_call_count += 1
                # Seat refresh cadence: fire at step > 0 on the interval
                # boundary. step=0 gets nothing (no accumulator state yet).
                if (
                    steps > 0
                    and int(cd.seat_refresh_interval) > 0
                    and steps % int(cd.seat_refresh_interval) == 0
                ):
                    cd.allocate_seats_from_accumulators(current_step=steps)
                    cd._seat_refresh_call_count += 1
                    # Emit one diagnostic snapshot per seat refresh. Walk
                    # the ssm_cores for their log_a Parameters (same set
                    # E.3 uses for the criticality loss below).
                    cd_log_a_per_layer: list[torch.Tensor] = []
                    for core in cd_ssm_cores:
                        la = getattr(core, "log_a", None)
                        if la is not None:
                            cd_log_a_per_layer.append(la)
                    if len(cd_log_a_per_layer) == cd.num_layers:
                        snap = cd.diagnostics_snapshot(
                            log_a_per_layer=cd_log_a_per_layer,
                            current_step=steps,
                        )
                        criticality_distillation_diagnostics.append(snap)
                # Compose criticality_loss as a SEPARATE backward pass.
                # log_a grads are disjoint from LM-head grads, so this
                # does not conflict with the fused CE backward already done.
                # criticality_loss() already multiplies by
                # criticality_distill_weight internally — don't double-multiply.
                if cd.seat_mask.any():
                    log_a_per_layer: list[torch.Tensor] = []
                    for core in cd_ssm_cores:
                        la = getattr(core, "log_a", None)
                        if la is not None:
                            log_a_per_layer.append(la)
                    if len(log_a_per_layer) == cd.num_layers:
                        cd_loss = cd.criticality_loss(log_a_per_layer)
                        criticality_distill_loss_trajectory.append(
                            {"step": int(steps), "cd_loss": float(cd_loss.detach().item())}
                        )
                        if cd_loss.requires_grad:
                            cd_loss.backward()
            else:
                # Nothing to close when CD is disabled; the stack is empty.
                cd_stack.close()
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
            # Phase B5: deferred post-step CE pair pass on the episodic
            # rank. The in-step drain stages each successful replay and
            # the just-emitted REPLAY_OUTCOME dict; this call runs a
            # no-grad forward on the post-step weights, computes mean CE
            # on the same value tokens, and patches
            # ``ce_after_replay`` / ``ce_delta_raw`` /
            # ``bucket_baseline`` / ``reward_shaped`` in place. Bucket
            # EMA also updates here. Helper is a no-op (zero pending,
            # gate off, or non-episodic rank where the pending list is
            # always None) on every other code path.
            if is_episodic_rank and episodic_consumer is not None:
                _run_post_step_replay_ce(
                    consumer=episodic_consumer,
                    model=model,
                )
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
        # Phase 2: stop the episodic controller thread. None handle
        # (train ranks, episodic disabled, controller flag off) is a
        # no-op. Bounded join so a stuck loop can't block the runner's
        # exit indefinitely.
        _shutdown_episodic_controller(episodic_controller_handle)
        _cleanup_episodic_event_rings(episodic_emit_handle)
        _cleanup_episodic_event_rings(episodic_consumer)

    if ddp_active:
        dist.barrier()

    if fast_slow.enabled and str(fast_slow_eval_copy).strip().lower() == "slow":
        fast_slow.copy_slow_to_model(model)

    episodic_cache_payload: dict[str, Any] | None = None
    if episodic_enabled:
        local_cache_payload = (
            episodic_consumer.cache.to_dict()
            if is_episodic_rank and episodic_consumer.cache is not None
            else None
        )
        if ddp_active and dist.is_initialized():
            gathered_payloads: list[dict[str, Any] | None] | None = (
                [None for _ in range(world_size_)] if rank_ == 0 else None
            )
            dist.gather_object(
                local_cache_payload,
                object_gather_list=gathered_payloads,
                dst=0,
                group=all_group,
            )
            if rank_ == 0 and gathered_payloads is not None:
                episodic_cache_payload = next(
                    (p for p in gathered_payloads if p is not None),
                    None,
                )
        else:
            episodic_cache_payload = local_cache_payload

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
        episodic_enabled=episodic_enabled,
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
    if cd is not None:
        result["criticality_distillation_module"] = cd
        result["criticality_distill_debug"] = {
            "ingest_calls": int(cd._ingest_call_count),
            "seat_refresh_calls": int(cd._seat_refresh_call_count),
        }
        result["_criticality_distill_pinned_buffers"] = cd_pinned_buffers
        result["criticality_distillation_diagnostics"] = (
            criticality_distillation_diagnostics
        )
        result["criticality_distill_loss_trajectory"] = (
            criticality_distill_loss_trajectory
        )
    if rare_bucket_ce_enabled:
        bucket_eval_tokens = (
            rare_bucket_ce_eval_tokens
            if rare_bucket_ce_eval_tokens is not None
            else train_tokens
        )
        bucket_eval_num_tokens = (
            int(rare_bucket_ce_eval_num_tokens)
            if rare_bucket_ce_eval_num_tokens is not None
            else int(train_num_tokens)
        )
        # Cap eval batch size: training bs (1024) materializes a
        # [bs, seq, vocab] fp32 logits tensor during non-fused eval
        # forward (F.cross_entropy path), which at seq=512/vocab=16384
        # is 32 GiB and OOMs. 128 keeps it at 4 GiB.
        eval_batch_size = min(int(batch_size), 128)
        val_block = _compute_per_bucket_val_ce(
            model=model,
            device=device,
            tokens=bucket_eval_tokens,
            num_tokens=bucket_eval_num_tokens,
            seq_len=int(seq_len),
            stride=int(stride),
            batch_size=eval_batch_size,
            vocab_size=int(vocab_size),
            token_frequencies=rare_bucket_ce_token_frequencies,
            num_buckets=int(rare_bucket_ce_num_buckets),
            rank=rank_,
            world_size=world_size_,
            precision=precision,
        )
        result.update(val_block)
    if topology_snapshot is not None:
        result["topology_snapshot"] = topology_snapshot
    if episodic_cache_payload is not None:
        result["_episodic_cache_payload"] = episodic_cache_payload
    return result


def measure_cd_overhead(train_fn, **kwargs) -> dict:
    """Run ``train_fn`` twice — once with CD off, once on — using the
    same seed and step budget, then report the overhead fraction.

    Returns a dict with both result dicts plus a tokens/sec summary.
    This is opt-in (report-only, not a hard gate) and doubles wall
    time; use it only for smoke cells that care about the overhead
    number. The treatment result dict has ``cd_overhead`` attached as
    a convenience so downstream callers can keep the authoritative
    run.
    """
    baseline_kwargs = dict(kwargs)
    baseline_kwargs["criticality_distill_enabled"] = False
    baseline_result = train_fn(**baseline_kwargs)

    treatment_kwargs = dict(kwargs)
    treatment_kwargs["criticality_distill_enabled"] = True
    treatment_result = train_fn(**treatment_kwargs)

    def _tok_per_sec(d: dict) -> float:
        if "tokens_per_sec" in d:
            return float(d["tokens_per_sec"])
        if "aggregate_tokens_per_sec" in d:
            return float(d["aggregate_tokens_per_sec"])
        # Fall back to reconstruction.
        step_count = int(d.get("final_step_count", d.get("steps", d.get("step_count", 0))))
        wall = float(d.get("total_wall_time", d.get("elapsed_s", d.get("elapsed_seconds", 1.0))))
        seq_len = int(kwargs["seq_len"])
        batch_size = int(kwargs["batch_size"])
        return (step_count * seq_len * batch_size) / max(wall, 1e-9)

    baseline_tps = _tok_per_sec(baseline_result)
    treatment_tps = _tok_per_sec(treatment_result)
    overhead = 1.0 - (treatment_tps / max(baseline_tps, 1e-9))
    summary = {
        "tokens_per_sec_baseline": float(baseline_tps),
        "tokens_per_sec_treatment": float(treatment_tps),
        "overhead_fraction": float(overhead),
    }
    treatment_result["cd_overhead"] = summary
    return {
        **summary,
        "baseline_result": baseline_result,
        "treatment_result": treatment_result,
    }


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

    rare_bucket_ce_enabled = bool(config.get("rare_bucket_ce_enabled", False))
    rare_bucket_ce_token_frequencies = None
    rare_bucket_ce_eval_tokens = None
    rare_bucket_ce_eval_num_tokens = None
    if rare_bucket_ce_enabled:
        rare_bucket_ce_token_frequencies = torch.bincount(
            train_tokens.detach().cpu().long(),
            minlength=vocab_size,
        ).clamp_min(1).to(device=device, dtype=torch.float32)
        rare_bucket_ce_eval_tokens = val_tokens
        rare_bucket_ce_eval_num_tokens = int(val_tokens.numel())

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
        lm_head_emit_entropy=bool(config.get("lm_head_emit_entropy", False)),
        criticality_distill_enabled=bool(
            config.get("criticality_distill_enabled", False)
        ),
        criticality_distill_num_layers=(
            int(config["criticality_distill_num_layers"])
            if "criticality_distill_num_layers" in config
            else int(config["num_layers"])
            if "num_layers" in config
            else None
        ),
        criticality_distill_dim=(
            int(config["criticality_distill_dim"])
            if "criticality_distill_dim" in config
            else int(config["model_dim"])
            if "model_dim" in config
            else None
        ),
        criticality_distill_budget_frac=float(
            config.get("criticality_distill_budget_frac", 0.15)
        ),
        criticality_distill_critical_value=float(
            config.get("criticality_distill_critical_value", 0.95)
        ),
        criticality_distill_trace_ttl_steps=int(
            config.get("criticality_distill_trace_ttl_steps", 1024)
        ),
        criticality_distill_trace_half_life_steps=float(
            config.get("criticality_distill_trace_half_life_steps", 256.0)
        ),
        criticality_distill_seat_refresh_interval=int(
            config.get("criticality_distill_seat_refresh_interval", 64)
        ),
        criticality_distill_min_weighted_events_per_layer=float(
            config.get(
                "criticality_distill_min_weighted_events_per_layer", 256.0
            )
        ),
        criticality_distill_horizon_H=int(
            config.get("criticality_distill_horizon_H", 16)
        ),
        criticality_distill_event_frac=float(
            config.get("criticality_distill_event_frac", 0.05)
        ),
        criticality_distill_weight=float(
            config.get("criticality_distill_weight", 1e-3)
        ),
        criticality_distill_uniform_pressure=bool(
            config.get("criticality_distill_uniform_pressure", False)
        ),
        criticality_distill_score_permute_before_topk=bool(
            config.get("criticality_distill_score_permute_before_topk", False)
        ),
        criticality_distill_fixed_random_seats=bool(
            config.get("criticality_distill_fixed_random_seats", False)
        ),
        rare_bucket_ce_enabled=bool(
            rare_bucket_ce_enabled
        ),
        rare_bucket_ce_num_buckets=int(
            config.get("rare_bucket_ce_num_buckets", 4)
        ),
        rare_bucket_ce_token_frequencies=rare_bucket_ce_token_frequencies,
        rare_bucket_ce_eval_tokens=rare_bucket_ce_eval_tokens,
        rare_bucket_ce_eval_num_tokens=rare_bucket_ce_eval_num_tokens,
        emit_topology_snapshot=bool(
            config.get("emit_topology_snapshot", False)
        ),
        episodic_enabled=bool(config.get("episodic_enabled", False)),
        episodic_top_p=(
            float(config["episodic_top_p"])
            if config.get("episodic_top_p") is not None
            else None
        ),
        episodic_fingerprint_window=int(
            config.get("episodic_fingerprint_window", 8)
        ),
        episodic_span_length=int(config.get("episodic_span_length", 4)),
        episodic_key_rep_dim=(
            int(config["episodic_key_rep_dim"])
            if config.get("episodic_key_rep_dim") is not None
            else None
        ),
        episodic_k_max=int(
            config.get("episodic_k_max", _DEFAULT_EPISODIC_K_MAX)
        ),
        episodic_capacity=int(config.get("episodic_capacity", 4096)),
        episodic_grace_steps=int(config.get("episodic_grace_steps", 1000)),
        episodic_utility_ema_decay=float(
            config.get("episodic_utility_ema_decay", 0.99)
        ),
        controller_query_enabled=bool(
            config.get("controller_query_enabled", False)
        ),
        episodic_event_log_enabled=bool(
            config.get("episodic_event_log_enabled", False)
        ),
        episodic_compute_replay_ce_pair=bool(
            config.get("episodic_compute_replay_ce_pair", False)
        ),
        episodic_controller_score_mode=str(
            _controller_score_mode_from_config(config)
        ),
        episodic_controller_topk_k=int(
            config.get("episodic_controller_topk_k", 16)
        ),
        episodic_controller_idle_sleep_s=float(
            config.get("episodic_controller_idle_sleep_s", 0.005)
        ),
        episodic_controller_selection_mode=str(
            config.get("episodic_controller_selection_mode", "argmax")
        ),
        episodic_controller_selection_seed=(
            int(config["episodic_controller_selection_seed"])
            if config.get("episodic_controller_selection_seed") is not None
            else None
        ),
        episodic_controller_runtime=str(
            config.get("episodic_controller_runtime", "heuristic")
        ),
        episodic_controller_weights_path=(
            str(config["episodic_controller_weights_path"])
            if config.get("episodic_controller_weights_path") is not None
            else None
        ),
        episodic_controller_global_dim=int(
            config.get("episodic_controller_global_dim", 8)
        ),
        episodic_controller_slot_dim=int(
            config.get("episodic_controller_slot_dim", 4)
        ),
        episodic_controller_learning_rate=float(
            config.get("episodic_controller_learning_rate", 1.0e-3)
        ),
        episodic_controller_sgd_interval=int(
            config.get("episodic_controller_sgd_interval", 256)
        ),
        episodic_controller_ema_alpha=float(
            config.get("episodic_controller_ema_alpha", 0.25)
        ),
        episodic_controller_ema_interval=int(
            config.get("episodic_controller_ema_interval", 64)
        ),
        episodic_controller_credit_gamma=float(
            config.get("episodic_controller_credit_gamma", 0.995)
        ),
        episodic_controller_gerber_c=float(
            config.get("episodic_controller_gerber_c", 0.5)
        ),
        episodic_controller_lambda_hxh_warmup_events=int(
            config.get("episodic_controller_lambda_hxh_warmup_events", 1024)
        ),
        episodic_controller_lambda_hxh_clip=float(
            config.get("episodic_controller_lambda_hxh_clip", 1.0)
        ),
        episodic_controller_entropy_beta=float(
            config.get("episodic_controller_entropy_beta", 0.0)
        ),
        episodic_controller_history_entries=int(
            config.get("episodic_controller_history_entries", 64)
        ),
        episodic_replay_max_replays_per_step=int(
            config.get("episodic_replay_max_replays_per_step", 0)
        ),
    )
    episodic_cache_payload = train_result.pop("_episodic_cache_payload", None)

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
            _save_output_ckpt(
                output_ckpt,
                model,
                config,
                episodic_cache=episodic_cache_payload,
            )
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
