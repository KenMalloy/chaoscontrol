#!/usr/bin/env python3
"""Exp 23 single-condition fastest-path DDP runner.

This is deliberately narrower than the previous experiment launchers.
It keeps only the final bare-SSM training path and makes the 600s hot
loop explicit: vectorized batch gather, fused linear+CE head/loss,
fused Muon/grad-clip knobs, amortized stop checks, and compact timing JSON.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import datetime
import faulthandler
import hashlib
import json
import math
import os
import random
import signal
import struct
import sys
import threading
import time
import traceback
from collections import deque
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

_STACK_DUMP_SIGNAL_REGISTERED = False


def _enable_runner_stack_dump_signal() -> None:
    """Allow ``kill -USR1 <pid>`` to dump all Python thread stacks.

    External profilers are often blocked on managed pods. ``faulthandler`` is
    in-process, has no hot-loop cost, and gives us enough context to avoid
    rerunning blind when a distributed control path stalls.
    """
    global _STACK_DUMP_SIGNAL_REGISTERED
    if _STACK_DUMP_SIGNAL_REGISTERED:
        return
    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is None:
        _STACK_DUMP_SIGNAL_REGISTERED = True
        return
    try:
        faulthandler.register(sigusr1, file=sys.stderr, all_threads=True)
    except Exception:
        pass
    _STACK_DUMP_SIGNAL_REGISTERED = True


def _dump_runner_stacks(label: str) -> None:
    print(f"[stack-dump] {label}", file=sys.stderr, flush=True)
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:
        traceback.print_stack(file=sys.stderr)

from chaoscontrol.core import verify_diag_recurrence  # noqa: E402
from chaoscontrol.cache_utility import (  # noqa: E402
    CrctGradientConflictMonitor,
    ScarcityAwareMemoryOptimizer as CrctScarcityAwareMemoryOptimizer,
    _RANK3_NLL_CHUNK_BUDGET_BYTES,
    alpha_ramp as crct_alpha_ramp,
    chunked_nll_from_hidden,
    rank3_score_batch_causal,
)
from chaoscontrol.replay_eviction import ReplayEvictionLoop  # noqa: E402
from chaoscontrol.slot_commit import (  # noqa: E402
    SLOT_COMMIT_APPEND,
    SLOT_COMMIT_CODE_TO_ACTION,
    SlotCommit,
    apply_append_slot_commit_to_model,
    apply_slot_commit_to_model,
    slot_commit_dtype_code,
    slot_commit_dtype_from_code,
)
from chaoscontrol.data import (  # noqa: E402
    load_fineweb_tokens,
    load_fineweb_val_tokens,
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
from chaoscontrol.episodic.learned_action_space import (  # noqa: E402
    ConstrainedActionSpace,
    make_shared_event_ssm_from_config,
)
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
    _reserve_admission_trace_seq,
    build_write_event_dict,
    fingerprint_tokens,
    select_top_p_positions,
    tensor_fp16_to_u16_wire,
)
from chaoscontrol.optim.lamb import LAMB  # noqa: E402
from chaoscontrol.optim.muon import Muon  # noqa: E402
from chaoscontrol.optim.step_wrapper import wrap_optimizer_step  # noqa: E402
from chaoscontrol.optim.weight_ema import eval_with_ema  # noqa: E402
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
    _compiled_packet_step_fn,
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
from chaoscontrol.wake_cache_txn import TransactionalWakeCache  # noqa: E402
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
    FastSlowDecision,
    fast_slow_decision_from_dict,
    fast_slow_decision_to_dict,
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
from chaoscontrol.episodic.diagnostics import (  # noqa: E402
    ActionSpaceTraceLogger,
    DiagnosticsLogger,
)
from experiments._23_fast_path_runner_helpers import (  # noqa: E402
    _alloc_pinned_evidence_buffers,
    compute_ce_minus_entropy_pressure_from_fused,
)
from runner_exp17 import (  # noqa: E402
    build_sentencepiece_luts,
    evaluate_bpb_sp,
)
import chaoscontrol.eval.calc_types  # noqa: F401, E402  (registers calc_types)
from chaoscontrol.eval.runner_dispatch import dispatch_eval_for_config  # noqa: E402
from chaoscontrol.eval_stream.val_cache import load_val_cache  # noqa: E402
from runner_exp21 import (  # noqa: E402
    _apply_embed_init,
    _save_output_ckpt,
    build_model,
)


def _outer_slot_count_for_eval(model: torch.nn.Module) -> int:
    outer = getattr(model, "outer_model", None)
    table = getattr(outer, "table", None)
    if table is not None:
        return int(len(table))
    slots = getattr(outer, "_slots", None)
    if slots is not None:
        return int(len(slots))
    return 0


def _crct_packet_cache_eval_state(model: torch.nn.Module) -> dict[str, Any] | None:
    """Snapshot the packet-rank slot bank for post-train eval.

    The train ranks run the packet lane without owning CRCT slots. Exp27's
    calc-type eval runs on rank 0 after the DDP loop, so rank 0 must inherit
    the packet rank's latest complete cache before scoring. This transfer is
    outside the training timer and contains no validation data.
    """
    outer = getattr(model, "outer_model", None)
    get_extra_state = getattr(outer, "get_extra_state", None)
    if not callable(get_extra_state):
        return None
    slot_count = _outer_slot_count_for_eval(model)
    return {
        "schema_version": 1,
        "slot_count": int(slot_count),
        "outer_model_extra_state": get_extra_state(),
    }


def _apply_crct_packet_cache_eval_state(
    model: torch.nn.Module,
    state: dict[str, Any] | None,
) -> int:
    """Install a packet-rank slot-bank snapshot on the local eval model."""
    if not isinstance(state, dict):
        return 0
    extra_state = state.get("outer_model_extra_state")
    if not isinstance(extra_state, dict):
        return 0
    outer = getattr(model, "outer_model", None)
    set_extra_state = getattr(outer, "set_extra_state", None)
    if not callable(set_extra_state):
        raise AttributeError("model.outer_model must expose set_extra_state()")
    set_extra_state(extra_state)
    return _outer_slot_count_for_eval(model)


def _merge_online_eval_state_payloads(
    payloads: list[dict[str, Any] | None] | tuple[dict[str, Any] | None, ...],
) -> dict[str, Any]:
    """Merge rank-local online eval state from packet and maintenance ranks."""
    merged: dict[str, Any] = {}
    for payload in payloads:
        if isinstance(payload, dict):
            merged.update(payload)
    return merged


def _load_checkpoint_into_model_for_runner(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load a runner checkpoint and return metadata needed by eval-only runs."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint_path does not exist: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(f"checkpoint at {path} must be a dict, got {type(blob)!r}")
    state = blob.get("model")
    if not isinstance(state, dict):
        raise KeyError(f"checkpoint at {path} missing dict key 'model'")
    load_result = model.load_state_dict(state, strict=True)
    online_eval_state = blob.get("online_eval_state")
    if not isinstance(online_eval_state, dict):
        online_eval_state = {}
    return {
        "checkpoint_path": str(path),
        "checkpoint_keys": sorted(str(k) for k in blob.keys()),
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        "online_eval_state": online_eval_state,
    }


def _train_fast_eval_only_result(
    model: torch.nn.Module,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return a finite train-result shell for checkpoint-only eval runs."""
    device = kwargs.get("device", torch.device("cpu"))
    rank = int(kwargs.get("rank", 0))
    world_size = int(kwargs.get("world_size", 1))
    seq_len = int(kwargs.get("seq_len", 0))
    batch_size = int(kwargs.get("batch_size", 0))
    if isinstance(device, torch.device) and device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1 << 20))
    else:
        peak_vram_mb = 0.0
    artifact_bytes = (
        int(model.artifact_bytes())
        if hasattr(model, "artifact_bytes")
        else int(sum(p.numel() for p in model.parameters()) * 2)
    )
    return {
        "eval_only": True,
        "steps": 0,
        "elapsed_s": 0.0,
        "rank": rank,
        "world_size": world_size,
        "initial_loss": 0.0,
        "final_loss": 0.0,
        "loss_trajectory": [],
        "loss_delta": 0.0,
        "peak_vram_mb": peak_vram_mb,
        "aggregate_tokens_per_sec": 0.0,
        "per_gpu_tokens_per_sec": 0.0,
        "tokens_per_step": int(batch_size * seq_len * max(1, world_size)),
        "artifact_bytes_estimate": artifact_bytes,
    }


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


def _current_rank() -> int:
    """Return this process's distributed rank, or 0 if dist isn't initialized.

    Uses ``torch.distributed.get_rank()`` so it works correctly under torchrun
    after ``init_process_group`` has been called, and degrades gracefully in
    single-process tests where dist is unavailable.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def _build_optimizer(
    config: dict[str, Any],
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "muon")).strip()
    base_lr = float(config.get("base_lr", 0.128))
    weight_decay = float(config.get("weight_decay", 0.01))
    grouping = str(config.get("optimizer_param_grouping", "flat")).strip()
    dynamics_lr_mul = float(config.get("optimizer_dynamics_lr_mul", 0.1))
    all_named_params = list(model.named_parameters())
    named_params = all_named_params
    excluded_param_names: list[str] = []
    if bool(config.get("crct_enabled", False)):
        memory_prefixes = (
            "outer_model.",
            "semantic_tier.",
            "bucket_prototypes_module.",
        )
        named_params = [
            (param_name, param)
            for param_name, param in all_named_params
            if not param_name.startswith(memory_prefixes)
        ]
        excluded_param_names = [
            param_name
            for param_name, _ in all_named_params
            if param_name.startswith(memory_prefixes)
        ]
    params = build_optimizer_params(
        named_params,
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
        matrix_param_names: set[str] | None = None
        if bool(config.get("crct_enabled", False)):
            adamw_prefixes = (
                "embed.",
                "lm_head.",
                "outer_model.",
            )
            matrix_param_names = {
                param_name
                for param_name, param in named_params
                if param.requires_grad
                and param.ndim >= 2
                and not param_name.startswith(adamw_prefixes)
                and ".delta_proj." not in param_name
                and not param_name.endswith(".delta_proj.weight")
            }
        opt = Muon(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
            adamw_lr=base_lr,
            adamw_weight_decay=weight_decay,
            matrix_param_names=matrix_param_names,
            fused=bool(config.get("fused_muon", True)),
            log_a_beta_coupling=bool(
                config.get(
                    "optimizer_log_a_beta_coupling",
                    bool(config.get("crct_enabled", False)),
                )
            ),
            log_a_beta_ema=float(config.get("optimizer_log_a_beta_ema", 0.99)),
            log_a_beta_min=float(config.get("optimizer_log_a_beta_min", 0.5)),
        )
        opt.bind_param_names(named_params)
        opt._excluded_param_names = list(excluded_param_names)
        wrap_optimizer_step(
            opt,
            model=model,
            target_momentum=float(config.get("muon_momentum", 0.95)),
            warmup_start=float(config.get("muon_momentum_warmup_start", 0.95)),
            warmup_steps=int(config.get("muon_momentum_warmup_steps", 0)),
            weight_ema_decay=float(config.get("weight_ema_decay", 0.0)),
            is_rank_zero=_current_rank() == 0,
            ema_exclude_prefixes=(
                "outer_model.",
                "semantic_tier.",
                "bucket_prototypes_module.",
            ) if bool(config.get("crct_enabled", False)) else (),
            weight_ema_fake_quant_bits=int(config.get("weight_ema_fake_quant_bits", 0)),
        )
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
    plasticity_trace = getattr(optimizer, "plasticity_budget_trace", None)
    if callable(plasticity_trace):
        diagnostics["plasticity_budget"] = plasticity_trace()
    role_trace = getattr(optimizer, "ssm_role_trace", None)
    if callable(role_trace):
        diagnostics["ssm_role"] = role_trace()
    excluded = getattr(optimizer, "_excluded_param_names", None)
    if isinstance(excluded, (list, tuple)):
        diagnostics["excluded_params"] = {
            "count": int(len(excluded)),
            "outer_model": int(
                sum(str(name).startswith("outer_model.") for name in excluded)
            ),
            "semantic_tier": int(
                sum(str(name).startswith("semantic_tier.") for name in excluded)
            ),
            "bucket_prototypes": int(
                sum(
                    str(name).startswith("bucket_prototypes_module.")
                    for name in excluded
                )
            ),
        }
    return diagnostics


def _broadcast_model_params_coalesced(
    model: torch.nn.Module,
    *,
    src: int = 0,
    group: "dist.ProcessGroup | None" = None,
) -> None:
    """Synchronize model parameters with one broadcast per dtype/device bucket.

    CRCT-only runs keep rank 3 out of the train-rank gradient collective so
    teacher scoring cannot stall the trunk. This helper refreshes rank 3's
    teacher snapshot at a coarse cadence without paying one collective per
    parameter tensor.
    """
    buckets: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for param in model.parameters():
        buckets.setdefault((param.device, param.dtype), []).append(param.data)
    with torch.no_grad():
        for tensors in buckets.values():
            if not tensors:
                continue
            contig = [tensor.contiguous() for tensor in tensors]
            flat = torch._utils._flatten_dense_tensors(contig)
            dist.broadcast(flat, src=int(src), group=group)
            synced = torch._utils._unflatten_dense_tensors(flat, contig)
            for original, value in zip(tensors, synced, strict=True):
                original.copy_(value)


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


def _crct_full_input_ids(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Reconstruct the ``(B, T+1)`` continuation batch from fast-path slices."""
    return torch.cat([inputs[:, :1].long(), targets.long()], dim=1)


def _crct_valid_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Fast-path training batches are dense continuations with no padding."""
    return torch.ones_like(input_ids, dtype=torch.bool)


def _reject_unsupported_fast_step(
    model: torch.nn.Module,
    *,
    crct_enabled: bool = False,
) -> None:
    if not crct_enabled:
        _reject_unsupported(model)
        return
    unsupported = (
        ("wernicke", "wernicke layer"),
        ("semantic_tier", "semantic_tier bias"),
        ("posterior", "posterior correction module"),
    )
    # CRCT train ranks call encode(..., memory_mode="off"), so bucket
    # prototypes are not on the trunk hot path.  Replay maintenance may still
    # use them on the sidecar rank as the slow-prior consumer for DISTILL.
    for attr, label in unsupported:
        if getattr(model, attr, None) is not None:
            raise ValueError(
                f"crct_enabled=True fast path does not support the {label} "
                f"path (model.{attr} is not None)."
            )


def _crct_replay_cache_probe(
    loop: ReplayEvictionLoop,
    model: torch.nn.Module,
    step: int,
) -> bool:
    """Cache the most recent batch as probe data for replay-eviction."""
    outer = model.outer_model
    if len(outer.table) == 0:
        return False
    probe_ids = getattr(model, "_last_crct_probe_input_ids", None)
    if probe_ids is None:
        return False
    probe_step = int(getattr(model, "_last_crct_probe_step", step))
    if int(getattr(loop, "_last_ingested_probe_step", -1)) == probe_step:
        return False
    valid_mask = getattr(model, "_last_crct_probe_valid_mask", None)
    if valid_mask is None:
        valid_mask = torch.ones_like(probe_ids, dtype=torch.bool)
    probe_cue = getattr(model, "_last_crct_probe_outer_cue", None)
    loop.cache_probe(
        input_ids=probe_ids,
        valid_mask=valid_mask,
        cue=probe_cue,
        cache_read_cutoff=None,
        step=probe_step,
        stream_id=probe_step,
    )
    return True


def _crct_replay_tick_step(
    loop: ReplayEvictionLoop,
    model: torch.nn.Module,
    fallback_step: int,
) -> int:
    """Return the train-step clock for rank-3 replay maintenance.

    The memory rank can spin much faster than the train ranks.  Replay frames
    are produced by CRCT teacher payloads and their TTL is expressed in train
    steps, so using rank 3's local loop counter would age valid frames out
    before the sidecar has a chance to process them.
    """
    probe_step = getattr(model, "_last_crct_probe_step", None)
    if probe_step is not None:
        return int(probe_step)
    loop_probe_step = loop.latest_probe_step()
    if loop_probe_step is not None:
        return int(loop_probe_step)
    return int(fallback_step)


def _crct_score_payload_inline(
    *,
    model: torch.nn.Module,
    cache: TransactionalWakeCache,
    scarcity_optimizer: CrctScarcityAwareMemoryOptimizer | None,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    step: int,
    total_steps: int | None,
    tau: float,
    strength: float,
    w_max: float,
    alpha_max: float,
    memory_write_tokens: int,
    gradient_conflict_monitor: CrctGradientConflictMonitor | None = None,
    update_model_memory_after: bool = False,
    record_stage_seconds: dict[str, float] | None = None,
) -> dict[str, Any]:
    input_ids = _crct_full_input_ids(inputs, targets)
    valid_mask = _crct_valid_mask(input_ids)
    setattr(model, "_last_crct_probe_input_ids", input_ids.detach())
    setattr(model, "_last_crct_probe_valid_mask", valid_mask.detach())
    setattr(model, "_last_crct_probe_step", int(step))
    alpha = crct_alpha_ramp(
        int(step),
        int(total_steps or 0),
        alpha_max=float(alpha_max),
    )
    score = rank3_score_batch_causal(
        model=model,
        cache=cache,
        input_ids=input_ids,
        valid_mask=valid_mask,
        scarcity_optimizer=scarcity_optimizer,
        tau=float(tau),
        strength=float(strength) * float(alpha),
        w_max=float(w_max),
        update_model_memory_after=bool(update_model_memory_after),
        memory_write_tokens=int(memory_write_tokens),
        gradient_conflict_monitor=gradient_conflict_monitor,
        step=int(step),
        record_stage_seconds=record_stage_seconds,
    )
    outer_cue = getattr(model, "_last_outer_cue", None)
    if outer_cue is not None:
        setattr(model, "_last_crct_probe_outer_cue", outer_cue.detach())
    if scarcity_optimizer is not None:
        target = score["controller_target"]
        mask = valid_mask[:, 1:].to(device=target.device)
        actual_read_rate = (
            float(target[mask].mean().detach().item()) if bool(mask.any()) else 0.0
        )
        scarcity_optimizer.dual_step(actual_read_rate=actual_read_rate)
    payload = {
        "step_id": torch.tensor(int(step), device=inputs.device),
        "step_id_int": int(step),
        "target": score["controller_target"].detach().clone(),
        "confidence": score["confidence"].detach().clone(),
        "loss_weight": score["loss_weight"].detach().clone(),
        "utility": score["utility"].detach().clone(),
    }
    for key in (
        "plasticity_coverage",
        "plasticity_confidence",
        "plasticity_budget",
        "loss_reweight_diagnostics",
    ):
        value = score.get(key)
        if isinstance(value, torch.Tensor):
            payload[key] = value.detach().clone()
    if "memory_residual" in score and "memory_gate" in score:
        payload["memory_residual"] = score["memory_residual"].detach().clone()
        payload["memory_gate"] = score["memory_gate"].detach().clone()
        # rank3_score_batch_causal defines the packet gate as the same tensor
        # as the CRCT controller target. The mailbox can transport that once
        # and alias it back to memory_gate on the train rank.
        payload["memory_gate_alias_target"] = True
    write_records = score.get("memory_write_records")
    if isinstance(write_records, list):
        payload["memory_write_records"] = write_records
    return payload


@torch.inference_mode()
def _crct_packet_payload_inline(
    *,
    model: torch.nn.Module,
    cache: TransactionalWakeCache,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    step: int,
    memory_write_tokens: int,
) -> dict[str, Any]:
    """Serve a CRCT residual packet without exact off/force scoring.

    The packet-service rank is a low-latency serving rank, not the truth source.
    It retrieves the current episodic residual and publishes that packet with
    neutral loss/utility labels. The maintenance/evidence loop owns exact
    recurrent contrast and the plasticity/commit feedback derived from it.
    """
    input_ids = _crct_full_input_ids(inputs, targets)
    valid_mask = _crct_valid_mask(input_ids)
    setattr(model, "_last_crct_probe_input_ids", input_ids.detach())
    setattr(model, "_last_crct_probe_valid_mask", valid_mask.detach())
    setattr(model, "_last_crct_probe_step", int(step))

    packet_fn = getattr(model, "build_episodic_packet", None)
    if not callable(packet_fn):
        raise ValueError(
            "approximate CRCT packet serving requires "
            "model.build_episodic_packet(...)"
        )

    txn = cache.begin_batch()
    x = input_ids[:, :-1].to(dtype=torch.int32)
    mask = valid_mask[:, 1:].to(device=inputs.device, dtype=torch.float32)
    packet = packet_fn(x, cache_read_cutoff=txn.read_cutoff)
    outer_cue = getattr(model, "_last_outer_cue", None)
    if outer_cue is not None:
        setattr(model, "_last_crct_probe_outer_cue", outer_cue.detach())

    residual = packet.get("memory_residual")
    gate = packet.get("memory_gate")
    if not isinstance(residual, torch.Tensor) or not isinstance(gate, torch.Tensor):
        raise TypeError("build_episodic_packet must return memory_residual and memory_gate tensors")
    gate = gate.to(device=inputs.device, dtype=torch.float32) * mask

    write_records: list[dict[str, object]] | None = None
    write_limit = int(memory_write_tokens)
    if write_limit > 0:
        append_fn = getattr(model, "append_memory_from_hidden", None)
        source_hidden = packet.get("packet_source_hidden")
        if append_fn is None or not isinstance(source_hidden, torch.Tensor):
            raise ValueError(
                "approximate CRCT packet serving requires packet_source_hidden "
                "and model.append_memory_from_hidden(...) for cache bootstrap"
            )
        write_score = packet.get("packet_write_score")
        if isinstance(write_score, torch.Tensor):
            write_score = write_score.to(device=source_hidden.device, dtype=torch.float32)
            if write_score.shape == mask.shape:
                write_score = write_score * mask.to(device=source_hidden.device)
        else:
            write_score = None
        event_ids = None
        reserve_event_ids = getattr(cache, "reserve_event_ids", None)
        if callable(reserve_event_ids):
            n_write = min(int(source_hidden.shape[0] * source_hidden.shape[1]), write_limit)
            event_ids = reserve_event_ids(n_write, device=source_hidden.device)
        write_records = append_fn(
            source_hidden.detach(),
            bucket_ids=packet.get("packet_source_bucket_ids"),
            score=write_score,
            max_tokens=write_limit,
            event_ids=event_ids,
        )
        if not isinstance(write_records, list):
            raise TypeError(
                "append_memory_from_hidden must return generation-stamped records"
            )

    cache.commit(txn)
    neutral_target = torch.zeros_like(mask)
    out: dict[str, Any] = {
        "step_id": torch.tensor(int(step), device=inputs.device),
        "step_id_int": int(step),
        "target": neutral_target.detach().clone(),
        "confidence": neutral_target.detach().clone(),
        "loss_weight": mask.detach().clone(),
        "utility": torch.zeros_like(mask),
        "memory_residual": residual.detach().clone(),
        "memory_gate": gate.detach().clone(),
        "memory_gate_alias_target": False,
        "packet_source_count": int(packet.get("packet_source_count", 0)),
        "packet_approximate": True,
    }
    if write_records is not None:
        out["memory_write_records"] = write_records
        out["approx_memory_write_records"] = len(write_records)
    return out


def _record_crct_loss_reweight_metrics(
    metrics: dict[str, Any],
    scored: dict[str, Any],
) -> None:
    diag = scored.get("loss_reweight_diagnostics")
    if not isinstance(diag, torch.Tensor) or int(diag.numel()) < 8:
        return
    vals = diag.detach().float().reshape(-1).cpu()
    valid_tokens = float(vals[0].item())
    if valid_tokens <= 0.0:
        return
    plain = float(vals[1].item())
    weighted = float(vals[2].item())
    delta = float(vals[3].item())
    rel_delta = float(vals[4].item())
    abs_dev = float(vals[5].item())
    std = float(vals[6].item())
    weight_max = float(vals[7].item())

    metrics["crct_loss_reweight_samples"] += 1
    metrics["crct_loss_reweight_valid_tokens_sum"] += int(valid_tokens)
    metrics["crct_loss_reweight_plain_nll_weighted_sum"] += (
        plain * valid_tokens
    )
    metrics["crct_loss_reweight_weighted_nll_weighted_sum"] += (
        weighted * valid_tokens
    )
    metrics["crct_loss_reweight_delta_weighted_sum"] += delta * valid_tokens
    metrics["crct_loss_reweight_rel_delta_sum"] += rel_delta
    metrics["crct_loss_weight_abs_dev_mean_sum"] += abs_dev
    metrics["crct_loss_weight_std_sum"] += std
    metrics["crct_loss_weight_max"] = max(
        float(metrics["crct_loss_weight_max"]),
        weight_max,
    )


def _add_crct_loss_reweight_metric_means(out: dict[str, Any]) -> None:
    samples = int(out.get("crct_loss_reweight_samples", 0))
    valid_tokens = int(out.get("crct_loss_reweight_valid_tokens_sum", 0))
    if valid_tokens > 0:
        out["crct_loss_reweight_plain_nll_mean"] = (
            float(out.get("crct_loss_reweight_plain_nll_weighted_sum", 0.0))
            / float(valid_tokens)
        )
        out["crct_loss_reweight_weighted_nll_mean"] = (
            float(out.get("crct_loss_reweight_weighted_nll_weighted_sum", 0.0))
            / float(valid_tokens)
        )
        out["crct_loss_reweight_delta_mean"] = (
            float(out.get("crct_loss_reweight_delta_weighted_sum", 0.0))
            / float(valid_tokens)
        )
    else:
        out["crct_loss_reweight_plain_nll_mean"] = 0.0
        out["crct_loss_reweight_weighted_nll_mean"] = 0.0
        out["crct_loss_reweight_delta_mean"] = 0.0
    out["crct_loss_reweight_rel_delta_mean"] = (
        float(out.get("crct_loss_reweight_rel_delta_sum", 0.0)) / float(samples)
        if samples
        else 0.0
    )
    out["crct_loss_weight_abs_dev_mean"] = (
        float(out.get("crct_loss_weight_abs_dev_mean_sum", 0.0)) / float(samples)
        if samples
        else 0.0
    )
    out["crct_loss_weight_std_mean"] = (
        float(out.get("crct_loss_weight_std_sum", 0.0)) / float(samples)
        if samples
        else 0.0
    )


@torch.inference_mode()
def _score_fast_slow_readiness_inline(
    *,
    model: torch.nn.Module,
    slow_model: torch.nn.Module | None,
    fast_slow: FastSlowConsolidator | None,
    input_ids: torch.Tensor,
    step: int,
    chunk_size: int,
) -> dict[str, float | int] | None:
    """Exact rank-3 fast-vs-slow copy evidence on the current probe frame.

    Positive ``delta_nll`` means the slow EMA copy was better on this frame.
    This intentionally runs only where the CRCT oracle already runs: off the
    train-rank hot path, under ``torch.inference_mode()``, and against the
    latest-complete rank-3 weight mirror.
    """
    if fast_slow is None or not fast_slow.enabled or not fast_slow.slow_state:
        return None
    if slow_model is None:
        return None
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        return None
    x = input_ids[:, :-1].to(dtype=torch.int32)
    y = input_ids[:, 1:].to(dtype=torch.long)
    mask = _crct_valid_mask(input_ids)[:, 1:].to(device=input_ids.device).bool()
    valid_tokens = int(mask.sum().detach().item())
    if valid_tokens <= 0:
        return None

    def _nll_for(scored_model: torch.nn.Module) -> torch.Tensor:
        try:
            hidden = scored_model.encode(x, memory_mode="off")
        except TypeError:
            hidden = scored_model.encode(x)
        return chunked_nll_from_hidden(
            scored_model,
            hidden,
            y,
            chunk_size=int(chunk_size),
            chunk_budget_bytes=_RANK3_NLL_CHUNK_BUDGET_BYTES,
        )

    score_t0 = time.perf_counter()
    nll_fast = _nll_for(model)
    nll_slow = _nll_for(slow_model)

    mask_f = mask.to(device=nll_fast.device, dtype=torch.float32)
    denom = mask_f.sum().clamp_min(1.0)
    fast_mean = float((nll_fast.float() * mask_f).sum().detach().item() / denom.item())
    slow_mean = float((nll_slow.float() * mask_f).sum().detach().item() / denom.item())
    elapsed = time.perf_counter() - score_t0
    credit_key = (
        int(fast_slow.last_decision.step)
        if fast_slow.last_decision is not None
        else -1
    )
    return {
        "step": int(step),
        "credit_key": credit_key,
        "nll_fast": fast_mean,
        "nll_slow": slow_mean,
        "delta_nll": fast_mean - slow_mean,
        "valid_tokens": int(valid_tokens),
        "score_seconds": float(elapsed),
        "sync_count": int(fast_slow.sync_count),
        "last_sync_step": int(fast_slow.last_sync_step),
    }


def _decide_fast_slow_from_oracle_evidence(
    *,
    fast_slow: FastSlowConsolidator | None,
    action_space: ConstrainedActionSpace | None,
    evidence: dict[str, Any] | None,
    model: torch.nn.Module,
    step: int,
) -> FastSlowDecision | None:
    """Advance the learned consolidation head on the memory plane.

    This is intentionally called from the teacher/memory rank after exact
    fast-vs-slow evidence is available. The train ranks do not run the scalar
    action-space head every optimizer step.
    """
    if fast_slow is None or not fast_slow.enabled or action_space is None:
        return None
    if not isinstance(evidence, dict):
        return None

    _apply_fast_slow_oracle_feedback(
        fast_slow=fast_slow,
        action_space=action_space,
        payload={"fast_slow_readiness": evidence},
        step=int(step),
    )
    reward_context = {
        "rank": -1.0,
        "world_size": 0.0,
        "ddp_active": 1.0,
        "fast_slow_alpha": float(fast_slow.alpha),
        "oracle_delta_nll": float(evidence.get("delta_nll", 0.0)),
        "oracle_nll_fast": float(evidence.get("nll_fast", 0.0)),
        "oracle_nll_slow": float(evidence.get("nll_slow", 0.0)),
        "oracle_valid_tokens": float(evidence.get("valid_tokens", 0)),
        "oracle_score_seconds": float(evidence.get("score_seconds", 0.0)),
        "weight_snapshot_version_lag_steps": float(
            evidence.get("weight_snapshot_version_lag_steps", 0)
        ),
    }
    decision = fast_slow.decide(
        step=int(step) + 1,
        action_space=action_space,
        reward_context=reward_context,
    )
    fast_slow.apply_decision(model, decision)
    return decision


def _apply_fast_slow_oracle_feedback(
    *,
    fast_slow: FastSlowConsolidator,
    action_space: ConstrainedActionSpace | None,
    payload: dict[str, Any] | None,
    step: int,
) -> None:
    if action_space is None or not isinstance(payload, dict):
        return
    evidence = payload.get("fast_slow_readiness")
    if not isinstance(evidence, dict):
        return
    try:
        reward = float(evidence["delta_nll"])
        key = int(evidence["credit_key"])
    except Exception:
        return
    if key < 0:
        return
    fast_slow.apply_external_reward(
        action_space=action_space,
        key=key,
        reward=reward,
        step=int(step),
        reward_context={
            "source": "gpu3_fast_slow_readiness_oracle",
            "nll_fast": float(evidence.get("nll_fast", 0.0)),
            "nll_slow": float(evidence.get("nll_slow", 0.0)),
            "valid_tokens": float(evidence.get("valid_tokens", 0)),
            "score_seconds": float(evidence.get("score_seconds", 0.0)),
            "oracle_step": float(evidence.get("step", key - 1)),
        },
    )


def _apply_fast_slow_result_payload(
    *,
    model: torch.nn.Module,
    fast_slow: FastSlowConsolidator,
    payload: dict[str, Any] | None,
    metrics: dict[str, Any] | None = None,
) -> FastSlowDecision | None:
    if not isinstance(payload, dict):
        return None
    decision = fast_slow_decision_from_dict(payload.get("fast_slow_decision"))
    if decision is None:
        return None
    fast_slow.apply_decision(model, decision)
    if metrics is not None:
        metrics["fast_slow_result_decisions_applied"] = int(
            metrics.get("fast_slow_result_decisions_applied", 0)
        ) + 1
    return decision


def _apply_plasticity_budget_payload(
    *,
    optimizer: torch.optim.Optimizer,
    payload: dict[str, Any] | None,
    strength: float,
) -> bool:
    setter = getattr(optimizer, "set_plasticity_budget", None)
    if setter is None or not callable(setter) or not isinstance(payload, dict):
        return False
    budget = payload.get("plasticity_budget")
    confidence = payload.get("plasticity_confidence")
    if not isinstance(budget, torch.Tensor):
        return False
    step_tensor = payload.get("step_id")
    step_raw = payload.get("step_id_int")
    step = int(step_raw) if isinstance(step_raw, int) else None
    if step is None and isinstance(step_tensor, torch.Tensor):
        step = int(step_tensor.detach().cpu().item())
    setter(
        budget.detach(),
        confidence=confidence.detach() if isinstance(confidence, torch.Tensor) else None,
        strength=float(strength),
        step=step,
    )
    return True


def _dist_work_done(
    work: Any,
    *,
    device: torch.device | None = None,
    wait_for_progress: bool = True,
) -> bool:
    if work is None:
        return True
    is_completed = getattr(work, "is_completed", None)
    has_completion_probe = callable(is_completed)
    if callable(is_completed):
        try:
            if bool(is_completed()):
                return True
        except Exception:
            pass
    if not bool(wait_for_progress):
        return False
    wait = getattr(work, "wait", None)
    if callable(wait):
        try:
            wait_result = wait(datetime.timedelta(milliseconds=1))
            if has_completion_probe:
                try:
                    return bool(is_completed())
                except Exception:
                    return False
            return bool(wait_result)
        except TypeError:
            return False
        except Exception:
            return False
    return False


def _hotpath_yield() -> None:
    """Yield on Linux; Darwin local tests intentionally busy-spin here."""
    sched_yield = getattr(os, "sched_yield", None)
    if sched_yield is not None:
        sched_yield()


def _should_stop_memory_rank_loop(
    *,
    steps: int,
    elapsed_s: float,
    budget_seconds: float,
    stop_margin_seconds: float,
    max_steps: int | None = None,
) -> bool:
    """Stop predicate for async memory ranks.

    The shared train-rank predicate deliberately refuses to stop at
    ``steps == 0`` so a cold train rank cannot exit before doing work. Memory
    ranks count scored/served packets, not trunk steps; when no requests arrive
    they can legitimately remain at zero local steps and must still exit on the
    wall-clock budget so teardown collectives cannot hang.
    """
    if max_steps is not None and int(steps) >= int(max_steps):
        return True
    effective_budget = max(0.0, float(budget_seconds) - float(stop_margin_seconds))
    return float(elapsed_s) >= effective_budget


def _should_defer_memory_rank_stop_for_shutdown(
    *,
    local_stop: bool,
    elapsed_s: float,
    budget_seconds: float,
    stop_margin_seconds: float,
    transport_mode: str,
    active_transport: Any | None,
) -> bool:
    """Keep async memory ranks alive briefly to consume shutdown sentinels.

    Train ranks send the mailbox sentinel during teardown, after their own
    training loop exits. If memory ranks stop at the same wall-clock edge they
    can miss that sentinel and appear to shut down by timeout. The drain window
    is outside the trunk hot path: memory ranks keep polling for a bounded grace
    period, but still fall back to wall-clock exit if rank 0 never sends.
    """
    if not bool(local_stop):
        return False
    if str(transport_mode) != "async_rank0_memory_mailbox":
        return False
    if active_transport is None:
        return False
    if bool(getattr(active_transport, "shutdown_requested", False)):
        return False
    grace_s = max(5.0, float(stop_margin_seconds))
    return float(elapsed_s) < float(budget_seconds) + grace_s


def _control_barrier(
    *,
    group: "dist.ProcessGroup | None",
    label: str,
) -> None:
    """Bounded control-plane barrier for post-run bookkeeping.

    Split-memory runs use Gloo side groups for Python objects and lifecycle
    sync. A raw NCCL barrier here can spin forever and hides the missing rank;
    monitored Gloo barriers turn that into a labelled exception.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    timeout_s = float(os.environ.get("CHAOSCONTROL_CONTROL_BARRIER_TIMEOUT_S", "30"))
    timeout = datetime.timedelta(seconds=max(1.0, timeout_s))
    try:
        monitored = getattr(dist, "monitored_barrier", None)
        if callable(monitored) and group is not None:
            try:
                monitored(group=group, timeout=timeout, wait_all_ranks=True)
            except TypeError:
                monitored(group=group, timeout=timeout)
        else:
            dist.barrier(group=group)
    except Exception as exc:
        _dump_runner_stacks(f"control barrier {label!r} failure")
        raise RuntimeError(
            f"control barrier {label!r} failed or timed out after "
            f"{timeout.total_seconds():.1f}s"
        ) from exc


def _crct_rank_topology(
    *,
    world_size: int,
    replay_eviction_enabled: bool,
) -> dict[str, Any]:
    """Return the CRCT hardware roles for the current DDP world.

    Four-GPU validation keeps the historical 3+1 shape.  At 8 GPUs, learned
    maintenance gets its own coprocessor so packet serving no longer has to
    choose between low-latency residual production and slot coverage.
    """
    world = int(world_size)
    if world < 1:
        raise ValueError(f"world_size must be positive, got {world_size!r}")
    split = False
    packet_rank = world - (2 if split else 1)
    maintenance_rank = world - 1
    memory_ranks = sorted({int(packet_rank), int(maintenance_rank)})
    train_ranks = [rank for rank in range(world) if rank not in memory_ranks]
    return {
        "split_memory_ranks": bool(split),
        "packet_rank": int(packet_rank),
        "maintenance_rank": int(maintenance_rank),
        "memory_ranks": memory_ranks,
        "train_ranks": train_ranks,
        "grad_world_size": len(train_ranks),
        "memory_owner": (
            "split_packet_and_maintenance"
            if split
            else "packet_and_maintenance_shared"
        ),
    }


_SLOT_COMMIT_MAGIC = 0x4353434D544C414E  # "CSCMTLAN"
_SLOT_COMMIT_CLOSE_MAGIC = 0x4353434D544C434C  # "CSCMTLCL"
_SLOT_COMMIT_HEADER_VERSION = 1
_SLOT_COMMIT_HEADER_FIELDS = 16
_SLOT_COMMIT_SURVIVAL_SCALE = 1_000_000


class _CrctSlotCommitPeerTransport:
    """Peer slot-commit lane for split memory ranks.

    Tensor payloads use point-to-point distributed sends between the two memory
    ranks only.  On CUDA/NCCL this is the NVLink/P2P path; CPU/Gloo keeps the
    same state machine for local tests.  Train ranks never join this transport.
    The packet-service rank publishes authoritative APPEND commits to the
    maintenance rank; the maintenance rank publishes confirmed maintenance
    commits back to the packet-service rank.
    """

    def __init__(
        self,
        *,
        rank: int,
        packet_rank: int,
        maintenance_rank: int,
        group: "dist.ProcessGroup | None",
        device: torch.device,
        queue_capacity: int = 1024,
    ) -> None:
        self.rank = int(rank)
        self.packet_rank = int(packet_rank)
        self.maintenance_rank = int(maintenance_rank)
        self.group = group
        self.device = device
        self.queue_capacity = max(1, int(queue_capacity))
        self.participant = self.rank in {self.packet_rank, self.maintenance_rank}
        self._send_queue: deque[SlotCommit] = deque()
        self._send_header: torch.Tensor | None = None
        self._send_payload: torch.Tensor | None = None
        self._send_header_work: Any | None = None
        self._send_payload_work: Any | None = None
        self._recv_header: torch.Tensor | None = None
        self._recv_payload: torch.Tensor | None = None
        self._recv_header_work: Any | None = None
        self._recv_payload_work: Any | None = None
        self._recv_pending_header: torch.Tensor | None = None
        self._peer_close_seen = False
        self._seq = 0
        cuda_peer_access_possible = False
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                local_idx = int(device.index if device.index is not None else torch.cuda.current_device())
                other_idx = (
                    int(self.maintenance_rank)
                    if self.rank == self.packet_rank
                    else int(self.packet_rank)
                )
                if 0 <= other_idx < torch.cuda.device_count():
                    cuda_peer_access_possible = bool(
                        torch.cuda.can_device_access_peer(local_idx, other_idx)
                    )
            except Exception:
                cuda_peer_access_possible = False
        self.metrics: dict[str, Any] = {
            "mode": (
                "slot_commit_gloo_cpu"
                if self.device.type == "cpu"
                else "slot_commit_p2p"
            ),
            "participant": bool(self.participant),
            "packet_rank": int(self.packet_rank),
            "maintenance_rank": int(self.maintenance_rank),
            "device": str(self.device),
            "queue_capacity": int(self.queue_capacity),
            "queue_depth": 0,
            "queue_depth_max": 0,
            "submitted": 0,
            "queued": 0,
            "queue_overwrites": 0,
            "append_commits_sent": 0,
            "append_commits_applied": 0,
            "maintenance_commits_sent": 0,
            "maintenance_commits_applied": 0,
            "send_headers_started": 0,
            "send_payloads_started": 0,
            "send_completed": 0,
            "recv_headers_posted": 0,
            "recv_headers_completed": 0,
            "recv_payloads_started": 0,
            "recv_completed": 0,
            "close_headers_sent": 0,
            "close_headers_received": 0,
            "close_timeout": 0,
            "applied": 0,
            "dropped": 0,
            "stale_generation_drops": 0,
            "missing_slot_drops": 0,
            "replica_capacity_full_drops": 0,
            "errors": 0,
            "last_error": "",
            "last_drop_reason": "",
            "last_action": "",
            "last_slot_id": -1,
            "last_step": -1,
            "last_payload_numel": 0,
            "p2p_available": bool(self.participant and self.group is not None),
            "cuda_peer_access_possible": bool(cuda_peer_access_possible),
        }
        if self.participant and self.group is not None:
            self._post_header_recv()

    def close(self) -> None:
        if self.participant and self.group is not None:
            self._close_peer_lane()
        self._send_queue.clear()
        self._send_header = None
        self._send_payload = None
        self._send_header_work = None
        self._send_payload_work = None
        self._recv_header = None
        self._recv_payload = None
        self._recv_header_work = None
        self._recv_payload_work = None
        self._recv_pending_header = None

    def _close_peer_lane(self, *, timeout_s: float = 1.0) -> None:
        """Match the peer's outstanding header receive before shutdown.

        The legacy split-memory lane keeps one header ``irecv`` posted so the
        two memory ranks can exchange slot commits without polling a CPU mailbox. NCCL P2P
        receives cannot be abandoned safely before later collectives, so both
        peers send a fixed close header and wait briefly for the matching
        receive to complete.
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        self._send_queue.clear()
        if (
            not self._peer_close_seen
            and self._recv_header_work is None
            and self._recv_payload_work is None
        ):
            self._post_header_recv()
        if self._send_header_work is None and self._send_payload_work is None:
            self._send_header = self._make_close_header()
            try:
                self._send_header_work = dist.isend(
                    self._send_header,
                    dst=self._peer_rank(),
                    group=self.group,
                )
                self.metrics["close_headers_sent"] += 1
            except Exception as exc:
                self.metrics["errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                return
        while time.monotonic() < deadline:
            progressed = False
            if self._send_header_work is not None and self._work_done(
                self._send_header_work
            ):
                self._send_header_work = None
                self._send_header = None
                progressed = True
            if self._send_payload_work is not None and self._work_done(
                self._send_payload_work
            ):
                self._send_payload_work = None
                self._send_payload = None
                progressed = True
            if self._recv_header_work is not None and self._work_done(
                self._recv_header_work
            ):
                self._recv_header_work = None
                if self._recv_header is not None and self._is_close_header(
                    self._recv_header
                ):
                    self._mark_peer_close_seen()
                    self._recv_header = None
                    self._recv_pending_header = None
                else:
                    self._recv_pending_header = (
                        self._recv_header.detach().clone()
                        if self._recv_header is not None
                        else None
                    )
                    self._recv_header = None
                    self._recv_pending_header = None
                    if not self._peer_close_seen:
                        self._post_header_recv()
                progressed = True
            if (
                self._send_header_work is None
                and self._send_payload_work is None
                and self._peer_close_seen
                and self._recv_payload_work is None
            ):
                return
            if not progressed:
                _hotpath_yield()
        self.metrics["close_timeout"] += 1

    def diagnostics(self) -> dict[str, Any]:
        out = dict(self.metrics)
        out["queue_depth"] = len(self._send_queue)
        out["send_in_flight"] = self._send_header_work is not None
        out["payload_in_flight"] = (
            self._send_payload_work is not None
            or self._recv_payload_work is not None
        )
        return out

    def submit_peer(self, commit: SlotCommit) -> bool:
        if not self.participant or self.group is None:
            return False
        self.metrics["submitted"] += 1
        if len(self._send_queue) >= int(self.queue_capacity):
            self.metrics["queue_overwrites"] += 1
            self.metrics["dropped"] += 1
            self.metrics["last_drop_reason"] = "queue_full_newest_drop"
            self.metrics["last_action"] = str(commit.action)
            self.metrics["last_slot_id"] = int(commit.slot_id)
            self.metrics["last_step"] = int(commit.step)
            return False
        self._send_queue.append(commit)
        self.metrics["queued"] += 1
        if commit.action is SLOT_COMMIT_APPEND:
            self.metrics["append_commits_sent"] += 1
        else:
            self.metrics["maintenance_commits_sent"] += 1
        self.metrics["queue_depth"] = len(self._send_queue)
        self.metrics["queue_depth_max"] = max(
            int(self.metrics["queue_depth_max"]), len(self._send_queue)
        )
        self._poll_send()
        return True

    def poll(self, *, model: torch.nn.Module | None = None) -> bool:
        if not self.participant or self.group is None:
            return False
        progressed = False
        progressed = self._poll_send() or progressed
        progressed = self._poll_recv(model=model) or progressed
        return progressed

    def _work_done(
        self,
        work: Any | None,
        *,
        wait_for_progress: bool = True,
    ) -> bool:
        return _dist_work_done(
            work,
            device=self.device,
            wait_for_progress=wait_for_progress,
        )

    def _header_device(self) -> torch.device:
        return self.device if self.device.type == "cuda" else torch.device("cpu")

    def _payload_device(self) -> torch.device:
        return self.device if self.device.type == "cuda" else torch.device("cpu")

    def _make_header(self, commit: SlotCommit, payload: torch.Tensor | None) -> torch.Tensor:
        dtype_code = (
            slot_commit_dtype_code(payload.dtype) if payload is not None else 0
        )
        payload_numel = int(payload.numel()) if payload is not None else 0
        # APPEND has no parent slot generation in the Python data model; -1 is
        # only the fixed-width wire representation.
        base_generation = (
            -1 if commit.base_generation is None else int(commit.base_generation)
        )
        fields = [
            _SLOT_COMMIT_MAGIC,
            _SLOT_COMMIT_HEADER_VERSION,
            int(self._seq),
            int(commit.step),
            int(commit.action),
            int(commit.slot_id),
            base_generation,
            int(commit.new_generation),
            int(commit.bucket_id),
            int(commit.event_id),
            int(payload_numel),
            int(dtype_code),
            int(round(float(commit.survival_factor) * _SLOT_COMMIT_SURVIVAL_SCALE)),
            int(self.rank),
            0,
            0,
        ]
        self._seq += 1
        return torch.tensor(fields, device=self._header_device(), dtype=torch.int64)

    def _make_close_header(self) -> torch.Tensor:
        fields = [
            _SLOT_COMMIT_CLOSE_MAGIC,
            _SLOT_COMMIT_HEADER_VERSION,
            int(self._seq),
            0,
            0,
            -1,
            -1,
            -1,
            -1,
            0,
            0,
            0,
            0,
            int(self.rank),
            0,
            0,
        ]
        self._seq += 1
        return torch.tensor(fields, device=self._header_device(), dtype=torch.int64)

    def _is_close_header(self, header: torch.Tensor) -> bool:
        try:
            return int(header.reshape(-1)[0].detach().cpu().item()) == int(
                _SLOT_COMMIT_CLOSE_MAGIC
            )
        except Exception:
            return False

    def _mark_peer_close_seen(self) -> None:
        if not self._peer_close_seen:
            self.metrics["close_headers_received"] += 1
        self._peer_close_seen = True

    def _peer_rank(self) -> int:
        return (
            int(self.maintenance_rank)
            if int(self.rank) == int(self.packet_rank)
            else int(self.packet_rank)
        )

    def _commit_from_header(
        self,
        header: torch.Tensor,
        payload: torch.Tensor | None,
    ) -> SlotCommit:
        h = header.detach().to(device="cpu", dtype=torch.int64).tolist()
        if int(h[0]) != _SLOT_COMMIT_MAGIC:
            raise ValueError("bad slot commit magic")
        if int(h[1]) != _SLOT_COMMIT_HEADER_VERSION:
            raise ValueError("bad slot commit header version")
        action = SLOT_COMMIT_CODE_TO_ACTION[int(h[4])]
        base_generation = (
            None
            if action is SLOT_COMMIT_APPEND and int(h[6]) < 0
            else int(h[6])
        )
        survival_factor = float(h[12]) / float(_SLOT_COMMIT_SURVIVAL_SCALE)
        tensor = None
        if payload is not None:
            tensor = payload.reshape(1, -1).detach()
        return SlotCommit(
            slot_id=int(h[5]),
            action=action,
            step=int(h[3]),
            base_generation=base_generation,
            new_generation=int(h[7]),
            bucket_id=int(h[8]),
            event_id=int(h[9]),
            survival_factor=survival_factor,
            tensor=tensor,
            reason="p2p_commit",
        )

    def _start_next_send(self) -> bool:
        if not self._send_queue:
            return False
        commit = self._send_queue.popleft()
        payload = None
        if commit.tensor is not None:
            payload = commit.tensor.detach().to(
                device=self._payload_device(),
                dtype=commit.tensor.dtype,
            ).contiguous().view(-1)
        header = self._make_header(commit, payload)
        self._send_header = header
        self._send_payload = payload
        self._send_header_work = dist.isend(
            header,
            dst=self._peer_rank(),
            group=self.group,
        )
        self.metrics["send_headers_started"] += 1
        self.metrics["last_action"] = str(commit.action)
        self.metrics["last_slot_id"] = int(commit.slot_id)
        self.metrics["last_step"] = int(commit.step)
        self.metrics["last_payload_numel"] = int(payload.numel()) if payload is not None else 0
        return True

    def _poll_send(self) -> bool:
        progressed = False
        if self._send_header_work is None and self._send_payload_work is None:
            return self._start_next_send()
        if self._send_payload_work is None:
            if not self._work_done(self._send_header_work):
                return False
            self._send_header_work = None
            progressed = True
            if self._send_payload is not None:
                self._send_payload_work = dist.isend(
                    self._send_payload,
                    dst=self._peer_rank(),
                    group=self.group,
                )
                self.metrics["send_payloads_started"] += 1
                return True
        if self._send_payload_work is not None:
            if not self._work_done(self._send_payload_work):
                return progressed
            self._send_payload_work = None
            progressed = True
        self._send_header = None
        self._send_payload = None
        self.metrics["send_completed"] += 1
        return self._start_next_send() or progressed

    def _post_header_recv(self) -> None:
        if self._recv_header_work is not None:
            return
        self._recv_header = torch.empty(
            _SLOT_COMMIT_HEADER_FIELDS,
            device=self._header_device(),
            dtype=torch.int64,
        )
        self._recv_header_work = dist.irecv(
            self._recv_header,
            src=self._peer_rank(),
            group=self.group,
        )
        self.metrics["recv_headers_posted"] += 1

    def _poll_recv(self, *, model: torch.nn.Module | None) -> bool:
        if self._recv_header_work is None and self._recv_payload_work is None:
            self._post_header_recv()
            return False
        if self._recv_header_work is not None:
            if not self._work_done(
                self._recv_header_work,
                wait_for_progress=int(self.rank) == int(self.maintenance_rank),
            ):
                return False
            self._recv_header_work = None
            assert self._recv_header is not None
            self._recv_pending_header = self._recv_header.detach().clone()
            self.metrics["recv_headers_completed"] += 1
            if self._is_close_header(self._recv_pending_header):
                self._mark_peer_close_seen()
                self._recv_pending_header = None
                self._recv_header = None
                return True
            h = self._recv_pending_header.detach().to(device="cpu", dtype=torch.int64)
            payload_numel = int(h[10].item())
            dtype = slot_commit_dtype_from_code(int(h[11].item()))
            if payload_numel > 0:
                self._recv_payload = torch.empty(
                    payload_numel,
                    device=self._payload_device(),
                    dtype=dtype,
                )
                self._recv_payload_work = dist.irecv(
                    self._recv_payload,
                    src=self._peer_rank(),
                    group=self.group,
                )
                self.metrics["recv_payloads_started"] += 1
                return True
        if self._recv_payload_work is not None:
            if not self._work_done(
                self._recv_payload_work,
                wait_for_progress=int(self.rank) == int(self.maintenance_rank),
            ):
                return False
            self._recv_payload_work = None
        if self._recv_pending_header is None:
            return False
        try:
            commit = self._commit_from_header(
                self._recv_pending_header,
                self._recv_payload,
            )
            self.metrics["last_action"] = str(commit.action)
            self.metrics["last_slot_id"] = int(commit.slot_id)
            self.metrics["last_step"] = int(commit.step)
            if model is None:
                self.metrics["dropped"] += 1
                self.metrics["last_drop_reason"] = "missing_model"
            else:
                if commit.action is SLOT_COMMIT_APPEND:
                    accepted, reason = apply_append_slot_commit_to_model(model, commit)
                else:
                    accepted, reason = apply_slot_commit_to_model(model, commit)
                if accepted:
                    self.metrics["applied"] += 1
                    if commit.action is SLOT_COMMIT_APPEND:
                        self.metrics["append_commits_applied"] += 1
                    else:
                        self.metrics["maintenance_commits_applied"] += 1
                else:
                    self.metrics["dropped"] += 1
                    self.metrics["last_drop_reason"] = str(reason)
                    if reason == "stale_generation":
                        self.metrics["stale_generation_drops"] += 1
                    elif reason in {"missing_slot", "missing_record"}:
                        self.metrics["missing_slot_drops"] += 1
                    elif reason == "replica_capacity_full":
                        self.metrics["replica_capacity_full_drops"] += 1
        except Exception as exc:
            self.metrics["errors"] += 1
            self.metrics["last_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
        finally:
            self._recv_pending_header = None
            self._recv_header = None
            self._recv_payload = None
            self.metrics["recv_completed"] += 1
        return True


class _CrctAsyncTeacherTransport:
    """Nonblocking CRCT teacher transport with matched-batch replay.

    Only rank 0 and the memory rank enter this transport. Ranks 1..N-2 stay
    entirely inside the train-rank gradient subgroup; they do not know that the
    sparse teacher side channel exists. The two participating ranks enter the
    same sparse collective sequence:

    1. memory-rank broadcasts the previous scored payload (or a sentinel),
    2. rank 0 broadcasts this step's input ids to the memory rank.

    Rank 0 does not wait for either collective. It uses a payload only when a
    completed broadcast's ``request_step`` still has its original
    ``(inputs, targets)`` in the local request table; otherwise it fails open
    for that step. The memory rank scores completed requests after the
    optimizer/all-reduce window via ``after_optimizer_step()``, so teacher work
    can overlap the next train-rank forward.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        teacher_group: "dist.ProcessGroup",
        payload_shape: tuple[int, int, int],
        full_ids_shape: tuple[int, int],
        device: torch.device,
        payload_dtype: torch.dtype,
        max_local_batches: int,
        max_payload_lag_steps: int,
        score_interval_steps: int = 1,
        coordinator_rank: int = 0,
        memory_rank: int | None = None,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.memory_rank = (
            int(memory_rank) if memory_rank is not None else self.world_size - 1
        )
        self.n_train = max(1, self.world_size - 1)
        self.coordinator_rank = int(coordinator_rank)
        self.teacher_group = teacher_group
        self.payload_shape = tuple(int(x) for x in payload_shape)
        self.full_ids_shape = tuple(int(x) for x in full_ids_shape)
        self.device = device
        self.payload_dtype = payload_dtype
        self._async_collectives = self.device.type != "cpu"
        self.max_local_batches_configured = max(1, int(max_local_batches))
        self.max_payload_lag_steps_configured = max(0, int(max_payload_lag_steps))
        self.max_local_batches = self.max_local_batches_configured
        self.max_payload_lag_steps = self.max_payload_lag_steps_configured
        self.score_interval_steps = max(1, int(score_interval_steps))
        self.pending_result_broadcasts: deque[dict[str, Any]] = deque()
        self.pending_input_requests: deque[dict[str, Any]] = deque()
        self.local_batches_by_step: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.local_batch_order: deque[int] = deque()
        self.ready_result: dict[str, torch.Tensor] | None = None
        self.ready_result_request_step: int | None = None
        self.metrics: dict[str, Any] = {
            "mode": "async_rank0_memory_broadcast",
            "transport_group": "rank0_memory",
            "coordinator_rank": int(self.coordinator_rank),
            "memory_rank": int(self.memory_rank),
            "participant": True,
            "payload_dtype": str(payload_dtype).replace("torch.", ""),
            "request_shape": list(self.full_ids_shape),
            "payload_shape": list(self.payload_shape),
            "max_local_batches": int(self.max_local_batches),
            "max_payload_lag_steps": int(self.max_payload_lag_steps),
            "score_interval_steps": int(self.score_interval_steps),
            "requests_started": 0,
            "result_broadcasts_started": 0,
            "result_broadcasts_completed": 0,
            "request_broadcasts_started": 0,
            "request_broadcasts_completed": 0,
            "request_interval_skips": 0,
            "broadcast_interval_skips": 0,
            "requests_stored": 0,
            "local_batch_gpu_clones": 0,
            "local_request_evictions": 0,
            "payloads_scored": 0,
            "payloads_served": 0,
            "payloads_served_approximate": 0,
            "score_interval_skips": 0,
            "payloads_sent": 0,
            "payloads_received": 0,
            "payloads_used": 0,
            "memory_rank_pump_steps": 0,
            "memory_rank_outer_loop_seconds_sum": 0.0,
            "memory_rank_outer_loop_seconds_max": 0.0,
            "memory_rank_pre_pump_seconds_sum": 0.0,
            "memory_rank_pre_pump_seconds_max": 0.0,
            "memory_rank_replay_seconds_sum": 0.0,
            "memory_rank_replay_seconds_max": 0.0,
            "memory_rank_replay_ticks": 0,
            "memory_rank_replay_probes_ingested": 0,
            "memory_rank_replay_deferred_for_packet_work": 0,
            "memory_rank_replay_deferred_for_backpressure": 0,
            "memory_rank_pump_loop_seconds_sum": 0.0,
            "memory_rank_pump_loop_seconds_max": 0.0,
            "memory_rank_pump_idle_spins": 0,
            "memory_rank_pump_idle_yields": 0,
            "memory_rank_pump_request_pops": 0,
            "memory_rank_pump_last_request_step": -1,
            "memory_rank_request_events_superseded": 0,
            "memory_rank_pump_score_calls": 0,
            "low_priority_maintenance_checks": 0,
            "low_priority_maintenance_allows": 0,
            "low_priority_maintenance_defers": 0,
            "low_priority_maintenance_defer_pending_requests": 0,
            "low_priority_maintenance_defer_request_mailbox": 0,
            "low_priority_maintenance_pending_requests": 0,
            "low_priority_maintenance_last_reason": "",
            "memory_packets_sent": 0,
            "memory_packets_received": 0,
            "memory_packet_bytes_sent": 0,
            "memory_packet_bytes_received": 0,
            "memory_packet_bytes_sent_max": 0,
            "memory_packet_bytes_received_max": 0,
            "memory_packet_missing_payloads": 0,
            "memory_packet_compact_residuals_sent": 0,
            "memory_packet_compact_residuals_received": 0,
            "memory_packet_sequence_residual_rejections": 0,
            "memory_packet_residual_elements_max": 0,
            "memory_packet_gate_elements_max": 0,
            "memory_packet_lag_steps_sum": 0,
            "memory_packet_lag_steps_max": 0,
            "memory_packet_last_residual_shape": [],
            "memory_packet_last_gate_shape": [],
            "sentinel_broadcasts": 0,
            "sentinels_received": 0,
            "stale_payloads_dropped": 0,
            "orphan_payloads_dropped": 0,
            "superseded_payloads_dropped": 0,
            "completed_requests_dropped": 0,
            "ready_result_drops": 0,
            "shutdown_result_broadcasts_drained": 0,
            "shutdown_input_requests_drained": 0,
            "pre_sync_waits": 0,
            "pre_sync_wait_seconds_sum": 0.0,
            "pre_sync_wait_seconds_max": 0.0,
            "score_seconds_sum": 0.0,
            "score_seconds_max": 0.0,
            "crct_loss_reweight_samples": 0,
            "crct_loss_reweight_valid_tokens_sum": 0,
            "crct_loss_reweight_plain_nll_weighted_sum": 0.0,
            "crct_loss_reweight_weighted_nll_weighted_sum": 0.0,
            "crct_loss_reweight_delta_weighted_sum": 0.0,
            "crct_loss_reweight_rel_delta_sum": 0.0,
            "crct_loss_weight_abs_dev_mean_sum": 0.0,
            "crct_loss_weight_std_sum": 0.0,
            "crct_loss_weight_max": 0.0,
            "packet_service_seconds_sum": 0.0,
            "packet_service_seconds_max": 0.0,
            "packet_service_source_count_sum": 0,
            "packet_service_zero_source_packets": 0,
            "packet_service_approx_write_records": 0,
            "payload_lag_steps_sum": 0,
            "payload_lag_steps_max": 0,
            "max_pending_result_broadcasts": 0,
            "max_pending_input_requests": 0,
            "max_local_pending_batches": 0,
            "last_scored_request_step": None,
            "last_sent_request_step": None,
            "last_received_request_step": None,
            "last_used_request_step": None,
            "last_drop_reason": "",
            "errors": 0,
            "last_error": "",
        }

    def begin_step(
        self,
        *,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
        if self.rank == self.memory_rank:
            # The memory rank has a much lighter per-step body than rank 0.
            # Without side-channel backpressure it can queue teacher-group
            # collectives several steps ahead, then collide with later world
            # diagnostics. Throttle only the sidecar; ranks 1..N-2 still never
            # enter teacher traffic, and rank 0 stays on the trunk clock.
            self.wait_for_pending_collectives()
        payload_tuple = self._reap_result_broadcasts(current_step=int(step))
        interval = int(self.score_interval_steps)
        score_due = int(step) % interval == 0
        # Score at step N, then give rank 3 one train step to finish enough
        # work to publish on the next sparse all-group slot. The async
        # broadcast may complete later if scoring is still running, but train
        # ranks no longer have to wait a full score interval to see a label.
        broadcast_due = score_due or int(step) % interval == 1
        if broadcast_due:
            self._issue_result_broadcast(send_step=int(step))
        else:
            self.metrics["broadcast_interval_skips"] += 1
        if not score_due:
            self.metrics["request_interval_skips"] += 1
            return payload_tuple
        self._issue_input_request_broadcast(
            inputs=inputs,
            targets=targets,
            step=int(step),
        )
        return payload_tuple

    def after_optimizer_step(
        self,
        *,
        model: torch.nn.Module,
        cache: TransactionalWakeCache,
        scarcity_optimizer: CrctScarcityAwareMemoryOptimizer | None,
        step: int,
        total_steps: int | None,
        tau: float,
        strength: float,
        w_max: float,
        alpha_max: float,
        memory_write_tokens: int,
        gradient_conflict_monitor: CrctGradientConflictMonitor | None = None,
        replay_eviction_loop: ReplayEvictionLoop | None = None,
        fast_slow: FastSlowConsolidator | None = None,
        fast_slow_action_space: ConstrainedActionSpace | None = None,
        fast_slow_nll_chunk_size: int = 1024,
        slot_commit_transport: "_CrctSlotCommitPeerTransport | None" = None,
        update_model_memory_after: bool = True,
    ) -> None:
        # The non-mailbox transport does not own a teacher-snapshot apply
        # path, so it does not refresh the evidence engine LM head or score
        # fast/slow readiness here.
        # Accept the parameter so the polymorphic call site is uniform.
        del (
            replay_eviction_loop,
            fast_slow,
            fast_slow_action_space,
            fast_slow_nll_chunk_size,
            slot_commit_transport,
        )
        completed = self._reap_input_requests()
        if self.rank != self.memory_rank:
            return
        if not completed:
            return
        if int(step) % int(self.score_interval_steps) != 0:
            self.metrics["score_interval_skips"] += len(completed)
            self.metrics["completed_requests_dropped"] += len(completed)
            self.metrics["last_drop_reason"] = "score_interval"
            return
        if self.ready_result is not None:
            self.metrics["completed_requests_dropped"] += len(completed)
            self.metrics["ready_result_drops"] += len(completed)
            self.metrics["last_drop_reason"] = "ready_result_pending"
            return
        if len(completed) > 1:
            self.metrics["completed_requests_dropped"] += len(completed) - 1
            self.metrics["last_drop_reason"] = "newer_completed_request_won"
        slot = completed[-1]
        request_step = int(slot["step"])
        request_full_ids = slot["buffer"]
        try:
            train_inputs = request_full_ids[:, :-1].to(dtype=torch.int32)
            train_targets = request_full_ids[:, 1:].to(dtype=torch.long)
            t0 = time.perf_counter()
            scored = _crct_score_payload_inline(
                model=model,
                cache=cache,
                scarcity_optimizer=scarcity_optimizer,
                inputs=train_inputs,
                targets=train_targets,
                step=request_step,
                total_steps=total_steps,
                tau=tau,
                strength=strength,
                w_max=w_max,
                alpha_max=alpha_max,
                memory_write_tokens=int(memory_write_tokens),
                gradient_conflict_monitor=gradient_conflict_monitor,
                update_model_memory_after=bool(update_model_memory_after),
            )
            score_s = time.perf_counter() - t0
            self.metrics["score_seconds_sum"] += float(score_s)
            self.metrics["score_seconds_max"] = max(
                float(self.metrics["score_seconds_max"]),
                float(score_s),
            )
            self.metrics["payloads_scored"] += 1
            _record_crct_loss_reweight_metrics(self.metrics, scored)
            self.metrics["last_scored_request_step"] = request_step
            self.ready_result = scored
            self.ready_result_request_step = request_step
        except Exception as exc:
            self.metrics["errors"] += 1
            self.metrics["last_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            self.metrics["last_drop_reason"] = "score_exception"

    def diagnostics(self) -> dict[str, Any]:
        out = dict(self.metrics)
        out.update(
            {
                "pending_result_broadcasts": len(self.pending_result_broadcasts),
                "pending_input_requests": len(self.pending_input_requests),
                "local_pending_batches": len(self.local_batches_by_step),
                "ready_result_pending": self.ready_result is not None,
                "ready_result_request_step": self.ready_result_request_step,
            }
        )
        used = int(out.get("payloads_used", 0))
        if used:
            out["payload_lag_steps_mean"] = (
                float(out["payload_lag_steps_sum"]) / float(used)
            )
        else:
            out["payload_lag_steps_mean"] = 0.0
        scored = int(out.get("payloads_scored", 0))
        if scored:
            out["score_seconds_mean"] = float(out["score_seconds_sum"]) / float(scored)
        else:
            out["score_seconds_mean"] = 0.0
        _add_crct_loss_reweight_metric_means(out)
        return out

    def wait_for_pending_collectives(self) -> None:
        """Drain already-issued transport collectives before another all-group op.

        The transport intentionally uses async all-rank collectives. When CRCT
        refreshes rank 3's teacher snapshot with a coalesced parameter
        broadcast, wait for the previous payload/result collectives first so
        the all-group ordering stays explicit and telemetry shows any overlap
        cost instead of hiding it inside the parameter sync.
        """
        t0 = time.perf_counter()
        waited = False
        for slot in self.pending_result_broadcasts:
            for work in slot["works"]:
                wait = getattr(work, "wait", None)
                if wait is not None:
                    wait()
                    waited = True
        for slot in self.pending_input_requests:
            wait = getattr(slot["work"], "wait", None)
            if wait is not None:
                wait()
                waited = True
        if waited:
            elapsed = time.perf_counter() - t0
            self.metrics["pre_sync_waits"] += 1
            self.metrics["pre_sync_wait_seconds_sum"] += float(elapsed)
            self.metrics["pre_sync_wait_seconds_max"] = max(
                float(self.metrics["pre_sync_wait_seconds_max"]),
                float(elapsed),
            )

    def close(self) -> None:
        while self.pending_result_broadcasts:
            slot = self.pending_result_broadcasts.popleft()
            try:
                for work in slot["works"]:
                    work.wait()
                self.metrics["result_broadcasts_completed"] += 1
                self.metrics["shutdown_result_broadcasts_drained"] += 1
            except Exception as exc:
                self.metrics["errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "shutdown_broadcast_wait_error"
                break
        while self.pending_input_requests:
            slot = self.pending_input_requests.popleft()
            try:
                wait = getattr(slot["work"], "wait", None)
                if wait is not None:
                    wait()
                self.metrics["request_broadcasts_completed"] += 1
                self.metrics["shutdown_input_requests_drained"] += 1
            except Exception as exc:
                self.metrics["errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "shutdown_gather_wait_error"
                break

    def _new_payload_buffers(self) -> dict[str, torch.Tensor]:
        return {
            "target": torch.zeros(
                self.payload_shape,
                device=self.device,
                dtype=self.payload_dtype,
            ),
            "confidence": torch.zeros(
                self.payload_shape,
                device=self.device,
                dtype=self.payload_dtype,
            ),
            "loss_weight": torch.ones(
                self.payload_shape,
                device=self.device,
                dtype=self.payload_dtype,
            ),
            "utility": torch.zeros(
                self.payload_shape,
                device=self.device,
                dtype=self.payload_dtype,
            ),
            "meta": torch.zeros(6, device=self.device, dtype=torch.float32),
        }

    def _issue_result_broadcast(self, *, send_step: int) -> None:
        buffers = self._new_payload_buffers()
        valid = False
        request_step = -1
        if self.rank == self.memory_rank and self.ready_result is not None:
            request_step = int(
                self.ready_result_request_step
                if self.ready_result_request_step is not None
                else -1
            )
            try:
                for key in ("target", "confidence", "loss_weight", "utility"):
                    buffers[key].copy_(
                        self.ready_result[key].reshape(self.payload_shape).to(
                            device=self.device,
                            dtype=self.payload_dtype,
                        )
                    )
                buffers["meta"][0] = 1.0
                buffers["meta"][1] = float(request_step)
                buffers["meta"][2] = float(send_step)
                buffers["meta"][3] = float(request_step)
                valid = True
                self.metrics["payloads_sent"] += 1
                self.metrics["last_sent_request_step"] = request_step
            except Exception as exc:
                self.metrics["errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "broadcast_pack_exception"
            finally:
                self.ready_result = None
                self.ready_result_request_step = None
        if not valid:
            self.metrics["sentinel_broadcasts"] += 1
        works = [
            dist.broadcast(
                buffers["target"],
                src=self.memory_rank,
                group=self.teacher_group,
                async_op=bool(self._async_collectives),
            ),
            dist.broadcast(
                buffers["confidence"],
                src=self.memory_rank,
                group=self.teacher_group,
                async_op=bool(self._async_collectives),
            ),
            dist.broadcast(
                buffers["loss_weight"],
                src=self.memory_rank,
                group=self.teacher_group,
                async_op=bool(self._async_collectives),
            ),
            dist.broadcast(
                buffers["utility"],
                src=self.memory_rank,
                group=self.teacher_group,
                async_op=bool(self._async_collectives),
            ),
            dist.broadcast(
                buffers["meta"],
                src=self.memory_rank,
                group=self.teacher_group,
                async_op=bool(self._async_collectives),
            ),
        ]
        works = [work for work in works if work is not None]
        self.pending_result_broadcasts.append(
            {"works": works, "buffers": buffers, "send_step": int(send_step)}
        )
        self.metrics["result_broadcasts_started"] += 1
        self.metrics["max_pending_result_broadcasts"] = max(
            int(self.metrics["max_pending_result_broadcasts"]),
            len(self.pending_result_broadcasts),
        )

    def _issue_input_request_broadcast(
        self,
        *,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
    ) -> None:
        full_ids = _crct_full_input_ids(inputs, targets).to(
            dtype=torch.int32
        ).contiguous()
        if tuple(full_ids.shape) != self.full_ids_shape:
            raise ValueError(
                "CRCT async transport saw a dynamic batch shape: "
                f"{tuple(full_ids.shape)} != {self.full_ids_shape}"
            )
        if self.rank == self.coordinator_rank:
            self.local_batches_by_step[int(step)] = (
                inputs.detach().clone(),
                targets.detach().clone(),
            )
            self.metrics["local_batch_gpu_clones"] += 1
            self.local_batch_order.append(int(step))
            self.metrics["requests_stored"] += 1
            while len(self.local_batch_order) > self.max_local_batches:
                old_step = self.local_batch_order.popleft()
                if self.local_batches_by_step.pop(old_step, None) is not None:
                    self.metrics["local_request_evictions"] += 1
                    self.metrics["last_drop_reason"] = "local_request_evicted"
            request_buf = full_ids.detach().clone()
        else:
            request_buf = torch.empty(
                self.full_ids_shape,
                device=self.device,
                dtype=full_ids.dtype,
            )
        work = dist.broadcast(
            request_buf,
            src=self.coordinator_rank,
            group=self.teacher_group,
            async_op=bool(self._async_collectives),
        )
        self.pending_input_requests.append(
            {"work": work, "buffer": request_buf, "step": int(step)}
        )
        self.metrics["requests_started"] += 1
        self.metrics["request_broadcasts_started"] += 1
        self.metrics["max_pending_input_requests"] = max(
            int(self.metrics["max_pending_input_requests"]),
            len(self.pending_input_requests),
        )
        self.metrics["max_local_pending_batches"] = max(
            int(self.metrics["max_local_pending_batches"]),
            len(self.local_batches_by_step),
        )

    def _reap_result_broadcasts(
        self,
        *,
        current_step: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
        ready: tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None = None
        while self.pending_result_broadcasts and all(
            _dist_work_done(work)
            for work in self.pending_result_broadcasts[0]["works"]
        ):
            slot = self.pending_result_broadcasts.popleft()
            self.metrics["result_broadcasts_completed"] += 1
            if self.rank == self.memory_rank:
                continue
            buffers = slot["buffers"]
            meta_values = buffers["meta"].detach().cpu().tolist()
            valid = bool(float(meta_values[0]) > 0.5)
            if not valid:
                self.metrics["sentinels_received"] += 1
                continue
            request_step = int(float(meta_values[1]))
            lag = int(current_step) - request_step
            batch = self.local_batches_by_step.pop(request_step, None)
            try:
                self.local_batch_order.remove(request_step)
            except ValueError:
                pass
            if batch is None:
                self.metrics["orphan_payloads_dropped"] += 1
                self.metrics["last_drop_reason"] = "payload_without_local_batch"
                continue
            if self.max_payload_lag_steps > 0 and lag > self.max_payload_lag_steps:
                self.metrics["stale_payloads_dropped"] += 1
                self.metrics["last_drop_reason"] = "payload_too_stale"
                continue
            payload = {
                "step_id": torch.tensor(request_step, device=self.device),
                "target": buffers["target"][0].detach().clone().float(),
                "confidence": (
                    buffers["confidence"][0].detach().clone().float()
                ),
                "loss_weight": (
                    buffers["loss_weight"][0].detach().clone().float()
                ),
                "utility": buffers["utility"][0].detach().clone().float(),
            }
            if ready is not None:
                self.metrics["superseded_payloads_dropped"] += 1
            inputs, targets = batch
            ready = (payload, inputs, targets)
            self.metrics["payloads_received"] += 1
            self.metrics["payloads_used"] += 1
            self.metrics["last_received_request_step"] = request_step
            self.metrics["last_used_request_step"] = request_step
            self.metrics["payload_lag_steps_sum"] += max(0, lag)
            self.metrics["payload_lag_steps_max"] = max(
                int(self.metrics["payload_lag_steps_max"]),
                max(0, lag),
            )
        return ready

    def _reap_input_requests(self) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        while self.pending_input_requests and _dist_work_done(
            self.pending_input_requests[0]["work"]
        ):
            completed.append(self.pending_input_requests.popleft())
        if completed:
            self.metrics["request_broadcasts_completed"] += len(completed)
        return completed


_TEACHER_RESULT_SLICE_NAMES = (
    "target",
    "confidence",
    "loss_weight",
    "utility",
    "memory_residual",
    "memory_gate",
    "plasticity_coverage",
    "plasticity_confidence",
    "plasticity_budget",
)
_WEIGHT_SNAPSHOT_HEADER = struct.Struct("<IIIIQQQQIIIIQfffIQ")
_TEACHER_REQUEST_EVENT_TYPE = 6
_TEACHER_REQUEST_FLAG_SHUTDOWN = 1 << 0


def _align64(n: int) -> int:
    return (int(n) + 63) & ~63


def _teacher_shm_name(mailbox_dir: Path, suffix: str) -> str:
    digest = hashlib.blake2s(str(mailbox_dir).encode("utf-8"), digest_size=4).hexdigest()
    # macOS POSIX shm names are short; keep this under PSHMNAMLEN.
    return f"/cc{suffix}{digest}"


def _teacher_dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in {torch.float16, torch.bfloat16}:
        return 2
    if dtype in {torch.float32, torch.int32}:
        return 4
    raise ValueError(f"unsupported teacher shm dtype: {dtype}")


def _teacher_dtype_code(dtype: torch.dtype) -> int:
    constants = _ext.wire_event_constants()
    if dtype == torch.int32:
        return int(constants["TEACHER_DTYPE_INT32"])
    if dtype == torch.float32:
        return int(constants["TEACHER_DTYPE_FLOAT32"])
    if dtype == torch.float16:
        return int(constants["TEACHER_DTYPE_FLOAT16"])
    if dtype == torch.bfloat16:
        return int(constants["TEACHER_DTYPE_BFLOAT16"])
    raise ValueError(f"unsupported teacher shm dtype: {dtype}")


def _teacher_dtype_from_code(code: int) -> torch.dtype | None:
    constants = _ext.wire_event_constants()
    code_i = int(code)
    if code_i == int(constants["TEACHER_DTYPE_NONE"]):
        return None
    if code_i == int(constants["TEACHER_DTYPE_INT32"]):
        return torch.int32
    if code_i == int(constants["TEACHER_DTYPE_FLOAT32"]):
        return torch.float32
    if code_i == int(constants["TEACHER_DTYPE_FLOAT16"]):
        return torch.float16
    if code_i == int(constants["TEACHER_DTYPE_BFLOAT16"]):
        return torch.bfloat16
    raise ValueError(f"unknown teacher shm dtype code: {code_i}")


def _teacher_empty_slice() -> dict[str, object]:
    return {
        "offset_bytes": 0,
        "nbytes": 0,
        "dtype": int(_ext.wire_event_constants()["TEACHER_DTYPE_NONE"]),
        "rank": 0,
        "shape": [0, 0, 0, 0],
    }


def _fast_slow_mode_code(mode: str) -> int:
    if str(mode) == "learned":
        return 1
    if str(mode) == "interval":
        return 2
    if str(mode) == "hold":
        return 3
    if str(mode) == "disabled":
        return 4
    return 0


def _fast_slow_mode_from_code(code: int) -> str:
    return {
        1: "learned",
        2: "interval",
        3: "hold",
        4: "disabled",
    }.get(int(code), "")


def _fast_slow_reason_code(reason: str) -> int:
    if str(reason) == "controller_consolidation_head":
        return 1
    if str(reason) == "fixed_interval_fallback":
        return 2
    if str(reason) == "not_due":
        return 3
    if str(reason) == "disabled":
        return 4
    return 0


def _fast_slow_reason_from_code(code: int) -> str:
    return {
        1: "controller_consolidation_head",
        2: "fixed_interval_fallback",
        3: "not_due",
        4: "disabled",
    }.get(int(code), "")


class _CrctMailboxTeacherTransport:
    """Same-node drop mailbox for CRCT teacher labels.

    The collective transport is correct but not a true sidecar: issuing a
    PyTorch ``broadcast(async_op=True)`` can still block rank 0 if rank 3 is
    busy inside oracle scoring. This mailbox trades a small score-interval D2H
    copy for the property the CRCT headline matrix needs: train ranks never
    rendezvous with the teacher. Phase 2 can swap the file-backed payloads for
    POSIX shm/SPSC slots without changing the runner contract.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        mailbox_dir: str,
        payload_shape: tuple[int, int, int],
        full_ids_shape: tuple[int, int],
        device: torch.device,
        payload_dtype: torch.dtype,
        max_local_batches: int,
        max_payload_lag_steps: int,
        score_interval_steps: int = 1,
        coordinator_rank: int = 0,
        memory_rank: int | None = None,
        memory_role: str = "packet",
        produce_results: bool = True,
        plasticity_ema_beta: float = 0.95,
        hidden_dim: int | None = None,
        score_stage_timing_enabled: bool = False,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.memory_rank = (
            int(memory_rank) if memory_rank is not None else self.world_size - 1
        )
        self.coordinator_rank = int(coordinator_rank)
        self.memory_role = str(memory_role or "packet")
        self.produce_results = bool(produce_results)
        self.mailbox_dir = Path(mailbox_dir)
        self.mailbox_dir.mkdir(parents=True, exist_ok=True)
        self.payload_shape = tuple(int(x) for x in payload_shape)
        self.full_ids_shape = tuple(int(x) for x in full_ids_shape)
        self.hidden_dim = int(
            hidden_dim
            if hidden_dim is not None and int(hidden_dim) > 0
            else max(4096, int(self.payload_shape[-1]) if self.payload_shape else 1)
        )
        self.device = device
        self.payload_dtype = payload_dtype
        self.plasticity_ema_beta = min(max(float(plasticity_ema_beta), 0.0), 0.9999)
        self.max_local_batches_configured = max(1, int(max_local_batches))
        self.max_payload_lag_steps_configured = max(0, int(max_payload_lag_steps))
        self.max_local_batches = self.max_local_batches_configured
        self.max_payload_lag_steps = self.max_payload_lag_steps_configured
        self.score_interval_steps = max(1, int(score_interval_steps))
        self.score_stage_timing_enabled = bool(score_stage_timing_enabled)
        self.pending_input_requests: deque[dict[str, Any]] = deque()
        self.ready_results: deque[
            tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]
        ] = deque()
        self.local_batches_by_step: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.local_batch_order: deque[int] = deque()
        self._request_write_thread: threading.Thread | None = None
        self._request_write_lock = threading.Lock()
        self._request_write_cv = threading.Condition(self._request_write_lock)
        self._request_write_pending: dict[str, Any] | None = None
        self._request_write_stop = False
        self._request_write_worker_active = False
        self._request_copy_stream: torch.cuda.Stream | None = None
        self._weight_publish_thread: threading.Thread | None = None
        self._last_applied_weight_step = -1
        self._fast_slow_slow_model: torch.nn.Module | None = None
        self._weight_snapshot_host_staging: dict[str, torch.Tensor] = {}
        self._weight_snapshot_read_staging: dict[str, torch.Tensor] = {}
        self._weight_snapshot_stage_banks: list[dict[str, Any]] = []
        self._weight_snapshot_bank_count = 3
        self._weight_snapshot_pending: dict[str, Any] | None = None
        self._weight_snapshot_writer_bank: int | None = None
        self._weight_snapshot_writer_stop = False
        self._weight_snapshot_worker_active = False
        self._weight_snapshot_lock = threading.Lock()
        self._weight_snapshot_cv = threading.Condition(self._weight_snapshot_lock)
        self._request_host_staging: torch.Tensor | None = None
        self._plasticity_coverage_ema: torch.Tensor | None = None
        self._plasticity_confidence_ema: torch.Tensor | None = None
        self._plasticity_budget_ema: torch.Tensor | None = None
        self._teacher_request_ring: Any | None = None
        self._teacher_result_ring: Any | None = None
        self._teacher_request_payload: Any | None = None
        self._teacher_result_payload: Any | None = None
        ring_prefix = "tm" if self.memory_role == "maintenance" else "t"
        self._teacher_request_ring_name = _teacher_shm_name(
            self.mailbox_dir, f"{ring_prefix}q"
        )
        self._teacher_result_ring_name = _teacher_shm_name(
            self.mailbox_dir, f"{ring_prefix}r"
        )
        self._teacher_request_payload_name = _teacher_shm_name(
            self.mailbox_dir, f"{ring_prefix}qb"
        )
        self._teacher_result_payload_name = _teacher_shm_name(
            self.mailbox_dir, f"{ring_prefix}rb"
        )
        self._weight_snapshot_shm_name = _teacher_shm_name(self.mailbox_dir, "ws")
        self._weight_snapshot_shm: Any | None = None
        self._weight_snapshot_layout: list[dict[str, Any]] | None = None
        self._weight_snapshot_buffer_bytes = 0
        self._weight_snapshot_region_bytes = 0
        self._teacher_request_seq = 0
        self._teacher_result_seq = 0
        self._teacher_ring_capacity = int(_ext.ShmRingTeacherRequest.capacity)
        # Mailbox transport should honor the configured backlog/lag policy, but
        # the queue can never exceed the physical ring capacity.
        self.max_local_batches = min(
            int(self.max_local_batches_configured),
            int(self._teacher_ring_capacity),
        )
        self.max_payload_lag_steps = int(self.max_payload_lag_steps_configured)
        self._teacher_request_slot_bytes = _align64(
            int(np.prod(self.full_ids_shape)) * _teacher_dtype_nbytes(torch.int32)
        )
        bsz = int(self.payload_shape[1]) if len(self.payload_shape) >= 2 else 1
        seq = int(self.payload_shape[2]) if len(self.payload_shape) >= 3 else 1
        payload_item_bytes = _teacher_dtype_nbytes(self.payload_dtype)
        self._teacher_result_slot_bytes = _align64(
            (5 * bsz * seq * payload_item_bytes)
            + (bsz * self.hidden_dim * payload_item_bytes)
            + (3 * self.hidden_dim * payload_item_bytes)
        )
        self._teacher_request_payload_bytes = (
            self._teacher_ring_capacity * self._teacher_request_slot_bytes
        )
        self._teacher_result_payload_bytes = (
            self._teacher_ring_capacity * self._teacher_result_slot_bytes
        )
        self._request_shutdown_seen = False
        self._init_teacher_shm()
        self.metrics: dict[str, Any] = {
            "mode": "async_rank0_memory_shm",
            "transport_group": "rank0_memory_shm",
            "coordinator_rank": int(self.coordinator_rank),
            "memory_rank": int(self.memory_rank),
            "memory_role": self.memory_role,
            "produce_results": bool(self.produce_results),
            "participant": True,
            "mailbox_dir": str(self.mailbox_dir),
            "payload_dtype": str(payload_dtype).replace("torch.", ""),
            "request_shape": list(self.full_ids_shape),
            "payload_shape": list(self.payload_shape),
            "teacher_shm_request_ring": self._teacher_request_ring_name,
            "teacher_shm_result_ring": self._teacher_result_ring_name,
            "teacher_shm_request_payload": self._teacher_request_payload_name,
            "teacher_shm_result_payload": self._teacher_result_payload_name,
            "weight_snapshot_shm": self._weight_snapshot_shm_name,
            "teacher_shm_ring_capacity": int(self._teacher_ring_capacity),
            "teacher_shm_request_slot_bytes": int(self._teacher_request_slot_bytes),
            "teacher_shm_result_slot_bytes": int(self._teacher_result_slot_bytes),
            "teacher_shm_request_payload_bytes": int(
                self._teacher_request_payload_bytes
            ),
            "teacher_shm_result_payload_bytes": int(
                self._teacher_result_payload_bytes
            ),
            "max_local_batches_configured": int(self.max_local_batches_configured),
            "max_payload_lag_steps_configured": int(
                self.max_payload_lag_steps_configured
            ),
            "teacher_shm_request_ring_full_drops": 0,
            "teacher_shm_result_ring_full_drops": 0,
            "teacher_shm_request_attach_misses": 0,
            "teacher_shm_result_attach_misses": 0,
            "teacher_shm_payload_write_seconds_sum": 0.0,
            "teacher_shm_payload_write_seconds_max": 0.0,
            "teacher_shm_payload_read_seconds_sum": 0.0,
            "teacher_shm_payload_read_seconds_max": 0.0,
            "teacher_shm_request_events_pushed": 0,
            "teacher_shm_request_events_popped": 0,
            "teacher_shm_request_shutdown_events_pushed": 0,
            "teacher_shm_request_shutdown_events_popped": 0,
            "teacher_shm_request_shutdown_overwrites": 0,
            "teacher_shm_request_shutdown_ring_full_drops": 0,
            "teacher_request_shutdown_seen": False,
            "teacher_shm_result_events_pushed": 0,
            "teacher_shm_result_events_popped": 0,
            "max_local_batches": int(self.max_local_batches),
            "max_payload_lag_steps": int(self.max_payload_lag_steps),
            "score_interval_steps": int(self.score_interval_steps),
            "score_stage_timing_enabled": bool(self.score_stage_timing_enabled),
            "score_stage_samples": 0,
            "score_stage_encode_off_seconds_sum": 0.0,
            "score_stage_encode_off_seconds_max": 0.0,
            "score_stage_encode_force_on_seconds_sum": 0.0,
            "score_stage_encode_force_on_seconds_max": 0.0,
            "score_stage_nll_off_seconds_sum": 0.0,
            "score_stage_nll_off_seconds_max": 0.0,
            "score_stage_nll_mem_seconds_sum": 0.0,
            "score_stage_nll_mem_seconds_max": 0.0,
            "score_stage_plasticity_seconds_sum": 0.0,
            "score_stage_plasticity_seconds_max": 0.0,
            "score_stage_append_memory_seconds_sum": 0.0,
            "score_stage_append_memory_seconds_max": 0.0,
            "score_stage_peak_allocated_mb_max": 0.0,
            "score_stage_last_batch_size": 0,
            "score_stage_last_seq_len": 0,
            "score_stage_last_vocab_size": 0,
            "score_stage_last_hidden_dim": 0,
            "score_stage_last_nll_chunk_budget_bytes": 0,
            "score_stage_last_nll_effective_chunk": 0,
            "score_stage_last_nll_chunks_per_pass": 0,
            "score_stage_last_paired_encode": False,
            "requests_started": 0,
            "result_broadcasts_started": 0,
            "result_broadcasts_completed": 0,
            "request_broadcasts_started": 0,
            "request_broadcasts_completed": 0,
            "request_interval_skips": 0,
            "request_write_skipped_busy": 0,
            "request_write_latest_overwrites": 0,
            "request_write_publisher_busy": False,
            "request_writer_wakeups": 0,
            "request_submit_seconds_sum": 0.0,
            "request_submit_seconds_max": 0.0,
            "request_stage_started": 0,
            "request_stage_seconds_sum": 0.0,
            "request_stage_seconds_max": 0.0,
            "request_writer_cpu_copy_seconds_sum": 0.0,
            "request_writer_cpu_copy_seconds_max": 0.0,
            "request_host_pinned": False,
            "request_host_stage_bytes": 0,
            "broadcast_interval_skips": 0,
            "requests_stored": 0,
            "local_batch_gpu_clones": 0,
            "local_request_evictions": 0,
            "payloads_scored": 0,
            "payloads_served": 0,
            "payloads_served_approximate": 0,
            "score_interval_skips": 0,
            "payloads_sent": 0,
            "payloads_received": 0,
            "payloads_used": 0,
            "ready_result_queue_depth": 0,
            "ready_result_queue_max": 0,
            "ready_result_queue_drops": 0,
            "memory_rank_pump_steps": 0,
            "memory_rank_outer_loop_seconds_sum": 0.0,
            "memory_rank_outer_loop_seconds_max": 0.0,
            "memory_rank_pre_pump_seconds_sum": 0.0,
            "memory_rank_pre_pump_seconds_max": 0.0,
            "memory_rank_replay_seconds_sum": 0.0,
            "memory_rank_replay_seconds_max": 0.0,
            "memory_rank_replay_ticks": 0,
            "memory_rank_replay_probes_ingested": 0,
            "memory_rank_replay_deferred_for_packet_work": 0,
            "memory_rank_replay_deferred_for_backpressure": 0,
            "memory_rank_pump_loop_seconds_sum": 0.0,
            "memory_rank_pump_loop_seconds_max": 0.0,
            "memory_rank_pump_idle_spins": 0,
            "memory_rank_pump_idle_yields": 0,
            "memory_rank_pump_request_pops": 0,
            "memory_rank_pump_last_request_step": -1,
            "memory_rank_request_events_superseded": 0,
            "memory_rank_pump_score_calls": 0,
            "maintenance_request_frames_ingested": 0,
            "maintenance_request_frames_dropped_no_loop": 0,
            "maintenance_score_path_skips": 0,
            "maintenance_local_append_packets": 0,
            "maintenance_local_append_records": 0,
            "maintenance_local_append_seconds_sum": 0.0,
            "maintenance_local_append_seconds_max": 0.0,
            "maintenance_local_append_errors": 0,
            "memory_packets_sent": 0,
            "memory_packets_received": 0,
            "memory_packet_bytes_sent": 0,
            "memory_packet_bytes_received": 0,
            "memory_packet_bytes_sent_max": 0,
            "memory_packet_bytes_received_max": 0,
            "memory_packet_missing_payloads": 0,
            "memory_packet_compact_residuals_sent": 0,
            "memory_packet_compact_residuals_received": 0,
            "memory_packet_gate_alias_target_sent": 0,
            "memory_packet_gate_alias_target_received": 0,
            "memory_packet_sequence_residual_rejections": 0,
            "memory_packet_residual_elements_max": 0,
            "memory_packet_gate_elements_max": 0,
            "memory_packet_lag_steps_sum": 0,
            "memory_packet_lag_steps_max": 0,
            "memory_packet_last_residual_shape": [],
            "memory_packet_last_gate_shape": [],
            "slot_append_records_seen": 0,
            "slot_append_commits_published": 0,
            "slot_append_commit_publish_failures": 0,
            "slot_append_commits_local_only": 0,
            "plasticity_ema_beta": float(self.plasticity_ema_beta),
            "plasticity_packets_sent": 0,
            "plasticity_packets_received": 0,
            "plasticity_packets_missing": 0,
            "plasticity_packet_bytes_sent": 0,
            "plasticity_packet_bytes_received": 0,
            "plasticity_packet_bytes_sent_max": 0,
            "plasticity_packet_bytes_received_max": 0,
            "plasticity_packet_last_shape": [],
            "plasticity_budget_mean_sent": 0.0,
            "plasticity_budget_max_sent": 0.0,
            "plasticity_budget_mean_received": 0.0,
            "plasticity_budget_max_received": 0.0,
            "plasticity_confidence_mean_sent": 0.0,
            "plasticity_confidence_mean_received": 0.0,
            "plasticity_coverage_abs_mean_sent": 0.0,
            "plasticity_coverage_abs_mean_received": 0.0,
            "plasticity_lag_steps_sum": 0,
            "plasticity_lag_steps_max": 0,
            "sentinel_broadcasts": 0,
            "sentinels_received": 0,
            "stale_payloads_dropped": 0,
            "orphan_payloads_dropped": 0,
            "superseded_payloads_dropped": 0,
            "completed_requests_dropped": 0,
            "ready_result_drops": 0,
            "shutdown_result_broadcasts_drained": 0,
            "shutdown_input_requests_drained": 0,
            "pre_sync_waits": 0,
            "pre_sync_wait_seconds_sum": 0.0,
            "pre_sync_wait_seconds_max": 0.0,
            "score_seconds_sum": 0.0,
            "score_seconds_max": 0.0,
            "crct_loss_reweight_samples": 0,
            "crct_loss_reweight_valid_tokens_sum": 0,
            "crct_loss_reweight_plain_nll_weighted_sum": 0.0,
            "crct_loss_reweight_weighted_nll_weighted_sum": 0.0,
            "crct_loss_reweight_delta_weighted_sum": 0.0,
            "crct_loss_reweight_rel_delta_sum": 0.0,
            "crct_loss_weight_abs_dev_mean_sum": 0.0,
            "crct_loss_weight_std_sum": 0.0,
            "crct_loss_weight_max": 0.0,
            "packet_service_seconds_sum": 0.0,
            "packet_service_seconds_max": 0.0,
            "packet_service_source_count_sum": 0,
            "packet_service_zero_source_packets": 0,
            "packet_service_approx_write_records": 0,
            "payload_lag_steps_sum": 0,
            "payload_lag_steps_max": 0,
            "max_pending_result_broadcasts": 0,
            "max_pending_input_requests": 0,
            "max_local_pending_batches": 0,
            "last_scored_request_step": None,
            "last_sent_request_step": None,
            "last_received_request_step": None,
            "last_used_request_step": None,
            "last_drop_reason": "",
            "errors": 0,
            "last_error": "",
            "mailbox_request_writes": 0,
            "mailbox_result_writes": 0,
            "mailbox_request_reads": 0,
            "mailbox_result_reads": 0,
            "mailbox_unlinks": 0,
            "mailbox_write_seconds_sum": 0.0,
            "mailbox_write_seconds_max": 0.0,
            "mailbox_read_seconds_sum": 0.0,
            "mailbox_read_seconds_max": 0.0,
            "weight_snapshot_attempts": 0,
            "weight_snapshot_copy_started": 0,
            "weight_snapshot_copy_seconds_sum": 0.0,
            "weight_snapshot_copy_seconds_max": 0.0,
            "weight_snapshot_hotpath_cpu_copies": 0,
            "weight_snapshot_stage_started": 0,
            "weight_snapshot_stage_enqueue_seconds_sum": 0.0,
            "weight_snapshot_stage_enqueue_seconds_max": 0.0,
            "weight_snapshot_stage_gpu_seconds_sum": 0.0,
            "weight_snapshot_stage_gpu_seconds_max": 0.0,
            "weight_snapshot_stage_tensor_count": 0,
            "weight_snapshot_stage_bytes": 0,
            "weight_snapshot_stage_wait_seconds_sum": 0.0,
            "weight_snapshot_stage_wait_seconds_max": 0.0,
            "weight_snapshot_writer_cpu_copy_seconds_sum": 0.0,
            "weight_snapshot_writer_cpu_copy_seconds_max": 0.0,
            "weight_snapshot_host_pinned_buffers": 0,
            "weight_snapshot_host_pinned_bytes": 0,
            "weight_snapshot_host_pageable_buffers": 0,
            "weight_snapshot_host_pageable_bytes": 0,
            "weight_snapshot_pin_memory_failures": 0,
            "host_pin_memory_failures": 0,
            "weight_snapshot_publish_started": 0,
            "weight_snapshot_published": 0,
            "weight_snapshot_publish_skipped_busy": 0,
            "weight_snapshot_publish_errors": 0,
            "weight_snapshot_save_seconds_sum": 0.0,
            "weight_snapshot_save_seconds_max": 0.0,
            "weight_snapshot_apply_attempts": 0,
            "weight_snapshot_applied": 0,
            "weight_snapshot_apply_stale": 0,
            "weight_snapshot_apply_errors": 0,
            "weight_snapshot_stat_skips": 0,
            "weight_snapshot_read_seconds_sum": 0.0,
            "weight_snapshot_read_seconds_max": 0.0,
            "weight_snapshot_read_tensor_count": 0,
            "weight_snapshot_read_bytes": 0,
            "weight_snapshot_apply_seconds_sum": 0.0,
            "weight_snapshot_apply_seconds_max": 0.0,
            "weight_snapshot_last_published_step": -1,
            "weight_snapshot_last_applied_step": -1,
            "weight_snapshot_version_lag_steps": 0,
            "weight_snapshot_publisher_busy": False,
            "weight_snapshot_shm_bytes": 0,
            "weight_snapshot_shm_header_bytes": int(_WEIGHT_SNAPSHOT_HEADER.size),
            "weight_snapshot_shm_writes": 0,
            "weight_snapshot_shm_reads": 0,
            "weight_snapshot_shm_attach_misses": 0,
            "weight_snapshot_pickle_writes": 0,
            "weight_snapshot_pickle_reads": 0,
            "weight_snapshot_stage_bank_count": int(self._weight_snapshot_bank_count),
            "weight_snapshot_latest_overwrites": 0,
            "weight_snapshot_worker_wakeups": 0,
            "weight_snapshot_shm_write_seconds_sum": 0.0,
            "weight_snapshot_shm_write_seconds_max": 0.0,
            "fast_slow_snapshot_decisions_published": 0,
            "fast_slow_snapshot_decisions_applied": 0,
            "fast_slow_readiness_scores": 0,
            "fast_slow_readiness_skips_no_slow_mirror": 0,
            "fast_slow_readiness_skips_no_valid_tokens": 0,
            "fast_slow_readiness_skips_gpu3_mirror": 0,
            "fast_slow_readiness_errors": 0,
            "fast_slow_readiness_seconds_sum": 0.0,
            "fast_slow_readiness_seconds_max": 0.0,
            "fast_slow_readiness_delta_sum": 0.0,
            "fast_slow_readiness_delta_abs_sum": 0.0,
            "fast_slow_readiness_delta_last": 0.0,
            "fast_slow_readiness_delta_positive": 0,
            "fast_slow_readiness_delta_negative": 0,
            "fast_slow_readiness_delta_zero": 0,
            "fast_slow_readiness_valid_tokens_sum": 0,
            "fast_slow_readiness_last_step": -1,
            "fast_slow_readiness_result_payloads": 0,
            "fast_slow_decisions_issued": 0,
            "fast_slow_decisions_result_payloads": 0,
            "fast_slow_result_decisions_applied": 0,
            "fast_slow_slow_mirror_creations": 0,
            "fast_slow_slow_mirror_create_seconds_sum": 0.0,
            "fast_slow_slow_mirror_create_seconds_max": 0.0,
            "fast_slow_slow_mirror_apply_seconds_sum": 0.0,
            "fast_slow_slow_mirror_apply_seconds_max": 0.0,
            "fast_slow_slow_mirror_last_step": -1,
            "fast_slow_slow_mirror_version_lag_steps": 0,
            "fast_slow_slow_mirror_last_error": "",
            "low_priority_maintenance_checks": 0,
            "low_priority_maintenance_allows": 0,
            "low_priority_maintenance_defers": 0,
            "low_priority_maintenance_defer_pending_requests": 0,
            "low_priority_maintenance_defer_request_mailbox": 0,
            "low_priority_maintenance_pending_requests": 0,
            "low_priority_maintenance_last_reason": "",
        }

    def _init_teacher_shm(self) -> None:
        if self.rank == self.coordinator_rank:
            owned_resources: list[tuple[Any, str]] = [
                (_ext.ShmRingTeacherRequest, self._teacher_request_ring_name),
                (_ext.PosixShm, self._teacher_request_payload_name),
            ]
            if self.produce_results:
                owned_resources.extend(
                    [
                        (_ext.ShmRingTeacherResult, self._teacher_result_ring_name),
                        (_ext.PosixShm, self._teacher_result_payload_name),
                        (_ext.PosixShm, self._weight_snapshot_shm_name),
                    ]
                )
            for cls, name in owned_resources:
                try:
                    cls.unlink(name)
                except Exception:
                    pass
            self._teacher_request_ring = _ext.ShmRingTeacherRequest.create(
                self._teacher_request_ring_name
            )
            self._teacher_request_payload = _ext.PosixShm(
                self._teacher_request_payload_name,
                self._teacher_request_payload_bytes,
                True,
            )
            if self.produce_results:
                self._teacher_result_ring = _ext.ShmRingTeacherResult.create(
                    self._teacher_result_ring_name
                )
                self._teacher_result_payload = _ext.PosixShm(
                    self._teacher_result_payload_name,
                    self._teacher_result_payload_bytes,
                    True,
                )

    def _ensure_teacher_request_consumer(self) -> bool:
        if self._teacher_request_ring is None:
            try:
                self._teacher_request_ring = _ext.ShmRingTeacherRequest.attach(
                    self._teacher_request_ring_name
                )
            except Exception:
                self.metrics["teacher_shm_request_attach_misses"] += 1
                return False
        if self._teacher_request_payload is None:
            try:
                self._teacher_request_payload = _ext.PosixShm(
                    self._teacher_request_payload_name,
                    0,
                    False,
                )
            except Exception:
                self.metrics["teacher_shm_request_attach_misses"] += 1
                return False
        return True

    def _ensure_teacher_result_producer(self) -> bool:
        if not self.produce_results:
            return False
        if self._teacher_result_ring is None:
            try:
                self._teacher_result_ring = _ext.ShmRingTeacherResult.attach(
                    self._teacher_result_ring_name
                )
            except Exception:
                self.metrics["teacher_shm_result_attach_misses"] += 1
                return False
        if self._teacher_result_payload is None:
            try:
                self._teacher_result_payload = _ext.PosixShm(
                    self._teacher_result_payload_name,
                    0,
                    False,
                )
            except Exception:
                self.metrics["teacher_shm_result_attach_misses"] += 1
                return False
        return True

    def _write_teacher_slice(
        self,
        *,
        shm: Any,
        slot_base: int,
        cursor: int,
        tensor: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[dict[str, object], int]:
        t = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
        nbytes = int(t.numel() * t.element_size())
        offset = int(slot_base) + int(cursor)
        slot_limit = int(slot_base) + int(
            self._teacher_request_slot_bytes
            if shm is self._teacher_request_payload
            else self._teacher_result_slot_bytes
        )
        if offset + nbytes > slot_limit:
            raise ValueError(
                "teacher shm payload slot too small: "
                f"offset={offset} nbytes={nbytes} slot_limit={slot_limit}"
            )
        t0 = time.perf_counter()
        shm.write_tensor(offset, t)
        elapsed = time.perf_counter() - t0
        self.metrics["teacher_shm_payload_write_seconds_sum"] += float(elapsed)
        self.metrics["teacher_shm_payload_write_seconds_max"] = max(
            float(self.metrics["teacher_shm_payload_write_seconds_max"]),
            float(elapsed),
        )
        shape = list(t.shape)[:4]
        shape.extend([0] * (4 - len(shape)))
        return (
            {
                "offset_bytes": offset,
                "nbytes": nbytes,
                "dtype": _teacher_dtype_code(t.dtype),
                "rank": int(t.dim()),
                "shape": [int(x) for x in shape],
            },
            _align64(int(cursor) + nbytes),
        )

    def _read_teacher_slice(self, shm: Any, desc: dict[str, object]) -> torch.Tensor | None:
        dtype = _teacher_dtype_from_code(int(desc["dtype"]))
        if dtype is None or int(desc["nbytes"]) <= 0:
            return None
        rank = int(desc["rank"])
        shape = tuple(int(x) for x in list(desc["shape"])[:rank])
        out = torch.empty(shape, dtype=dtype, device="cpu").contiguous()
        t0 = time.perf_counter()
        shm.read_tensor_into(int(desc["offset_bytes"]), out)
        elapsed = time.perf_counter() - t0
        self.metrics["teacher_shm_payload_read_seconds_sum"] += float(elapsed)
        self.metrics["teacher_shm_payload_read_seconds_max"] = max(
            float(self.metrics["teacher_shm_payload_read_seconds_max"]),
            float(elapsed),
        )
        return out

    @property
    def shutdown_requested(self) -> bool:
        return bool(self._request_shutdown_seen)

    def _push_shutdown_request(self) -> None:
        if self.rank != self.coordinator_rank:
            return
        if self._teacher_request_ring is None:
            self.metrics["last_drop_reason"] = "teacher_request_shutdown_ring_missing"
            return
        request_id = int(self._teacher_request_seq)
        self._teacher_request_seq += 1
        step = max(
            0,
            int(
                self.metrics.get("last_sent_request_step")
                or self.metrics.get("weight_snapshot_last_published_step")
                or 0
            ),
        )
        event = {
            "event_type": _TEACHER_REQUEST_EVENT_TYPE,
            "source_rank": int(self.rank),
            "status": 0,
            "flags": _TEACHER_REQUEST_FLAG_SHUTDOWN,
            "slice_count": 0,
            "request_id": request_id,
            "step": step,
            "weight_snapshot_version": max(
                0,
                int(self.metrics.get("weight_snapshot_last_published_step", -1)),
            ),
            "full_ids": _teacher_empty_slice(),
        }
        for _ in range(int(self._teacher_ring_capacity) + 1):
            if self._teacher_request_ring.push(event):
                self.metrics["teacher_shm_request_shutdown_events_pushed"] += 1
                self.metrics["teacher_shm_request_events_pushed"] += 1
                self.metrics["last_drop_reason"] = "teacher_request_shutdown_sent"
                return
            dropped = self._teacher_request_ring.pop()
            if dropped is None:
                break
            self.metrics["teacher_shm_request_shutdown_overwrites"] += 1
        self.metrics["teacher_shm_request_shutdown_ring_full_drops"] += 1
        self.metrics["last_drop_reason"] = "teacher_request_shutdown_ring_full"

    def begin_step(
        self,
        *,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
        if self.rank == self.coordinator_rank:
            ready = self._poll_results(current_step=int(step))
            self._submit_request(inputs=inputs, targets=targets, step=int(step))
            return ready
        if self.rank == self.memory_rank:
            popped = self._poll_requests()
            self.metrics["memory_rank_pump_request_pops"] += int(popped)
        return None

    def after_optimizer_step(
        self,
        *,
        model: torch.nn.Module,
        cache: TransactionalWakeCache,
        scarcity_optimizer: CrctScarcityAwareMemoryOptimizer | None,
        step: int,
        total_steps: int | None,
        tau: float,
        strength: float,
        w_max: float,
        alpha_max: float,
        memory_write_tokens: int,
        gradient_conflict_monitor: CrctGradientConflictMonitor | None = None,
        replay_eviction_loop: ReplayEvictionLoop | None = None,
        fast_slow: FastSlowConsolidator | None = None,
        fast_slow_action_space: ConstrainedActionSpace | None = None,
        fast_slow_nll_chunk_size: int = 1024,
        slot_commit_transport: "_CrctSlotCommitPeerTransport | None" = None,
        update_model_memory_after: bool = True,
    ) -> None:
        if self.rank == self.memory_rank:
            self.poll_weight_snapshot(
                model=model,
                step=int(step),
                replay_eviction_loop=replay_eviction_loop,
                # GPU3 is a latest-complete trunk mirror for the packet
                # scorer.  The snapshot already includes whatever
                # consolidation rank 0 committed; applying fast/slow again
                # here makes the memory GPU own a second copy policy and slows
                # exact-match packet production.
                fast_slow=None,
            )
        if self.rank == self.memory_rank and not self.produce_results:
            self._ingest_maintenance_requests(
                model=model,
                cache=cache,
                replay_eviction_loop=replay_eviction_loop,
                memory_write_tokens=int(memory_write_tokens),
            )
            return
        if self.rank != self.memory_rank or not self.pending_input_requests:
            return
        slot = self.pending_input_requests.popleft()
        request_step = int(slot["step"])
        request_full_ids = slot["buffer"]
        try:
            train_inputs = request_full_ids[:, :-1].to(dtype=torch.int32)
            train_targets = request_full_ids[:, 1:].to(dtype=torch.long)
            t0 = time.perf_counter()
            scored = _crct_packet_payload_inline(
                model=model,
                cache=cache,
                inputs=train_inputs,
                targets=train_targets,
                step=request_step,
                memory_write_tokens=int(memory_write_tokens),
            )
            packet_s = time.perf_counter() - t0
            scored["packet_seconds"] = float(packet_s)
            self.metrics["packet_service_seconds_sum"] += float(packet_s)
            self.metrics["packet_service_seconds_max"] = max(
                float(self.metrics["packet_service_seconds_max"]),
                float(packet_s),
            )
            self.metrics["payloads_served"] += 1
            self.metrics["payloads_served_approximate"] += 1
            self.metrics["memory_rank_pump_score_calls"] += 1
            if fast_slow is not None and getattr(fast_slow, "enabled", False):
                self.metrics["fast_slow_readiness_skips_gpu3_mirror"] += 1
            self.metrics["last_scored_request_step"] = request_step
            source_count = int(scored.get("packet_source_count", 0))
            self.metrics["packet_service_source_count_sum"] += source_count
            if source_count <= 0:
                self.metrics["packet_service_zero_source_packets"] += 1
            self.metrics["packet_service_approx_write_records"] += int(
                scored.get("approx_memory_write_records", 0)
            )
            if bool(update_model_memory_after):
                self._publish_memory_write_commits(
                    request_step=request_step,
                    scored=scored,
                    slot_commit_transport=slot_commit_transport,
                )
            if bool(self.score_stage_timing_enabled):
                print(
                    "[gpu6-packet] "
                    f"step={request_step} "
                    f"B={int(train_inputs.shape[0])} "
                    f"T={int(train_inputs.shape[1])} "
                    f"sources={source_count} "
                    f"writes={int(scored.get('approx_memory_write_records', 0))} "
                    f"packet={packet_s * 1000:.1f}ms",
                    flush=True,
                )
            if self.produce_results:
                self._write_result(request_step=request_step, scored=scored)
            else:
                self.metrics["last_sent_request_step"] = request_step
        except Exception as exc:
            self.metrics["errors"] += 1
            self.metrics["last_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            self.metrics["last_drop_reason"] = "score_exception"

    def _record_score_stage_seconds(
        self,
        *,
        request_step: int,
        stage_seconds: dict[str, float],
        peak_mb: float,
    ) -> None:
        self.metrics["score_stage_samples"] += 1
        for label in (
            "encode_off",
            "encode_force_on",
            "nll_off",
            "nll_mem",
            "plasticity",
            "append_memory",
        ):
            value = float(stage_seconds.get(label, 0.0))
            sum_key = f"score_stage_{label}_seconds_sum"
            max_key = f"score_stage_{label}_seconds_max"
            self.metrics[sum_key] += value
            self.metrics[max_key] = max(float(self.metrics[max_key]), value)
        self.metrics["score_stage_peak_allocated_mb_max"] = max(
            float(self.metrics["score_stage_peak_allocated_mb_max"]),
            float(peak_mb),
        )
        for metric_name, stage_key in (
            ("score_stage_last_batch_size", "_batch_size"),
            ("score_stage_last_seq_len", "_seq_len"),
            ("score_stage_last_vocab_size", "_vocab_size"),
            ("score_stage_last_hidden_dim", "_hidden_dim"),
            ("score_stage_last_nll_chunk_budget_bytes", "_nll_chunk_budget_bytes"),
            ("score_stage_last_nll_effective_chunk", "_nll_effective_chunk"),
            ("score_stage_last_nll_chunks_per_pass", "_nll_chunks_per_pass"),
        ):
            if stage_key in stage_seconds:
                self.metrics[metric_name] = int(stage_seconds.get(stage_key, 0.0))
        if "_paired_encode" in stage_seconds:
            self.metrics["score_stage_last_paired_encode"] = bool(
                stage_seconds.get("_paired_encode", 0.0)
            )
        print(
            "[gpu3-stage] "
            f"step={int(request_step)} "
            f"B={int(self.metrics['score_stage_last_batch_size'])} "
            f"T={int(self.metrics['score_stage_last_seq_len'])} "
            f"D={int(self.metrics['score_stage_last_hidden_dim'])} "
            f"V={int(self.metrics['score_stage_last_vocab_size'])} "
            f"nll_budget={int(self.metrics['score_stage_last_nll_chunk_budget_bytes'])} "
            f"nll_chunk={int(self.metrics['score_stage_last_nll_effective_chunk'])} "
            f"nll_chunks={int(self.metrics['score_stage_last_nll_chunks_per_pass'])} "
            f"paired={bool(self.metrics['score_stage_last_paired_encode'])} "
            f"encode_off={stage_seconds.get('encode_off', 0.0) * 1000:.1f}ms "
            f"encode_force_on={stage_seconds.get('encode_force_on', 0.0) * 1000:.1f}ms "
            f"nll_off={stage_seconds.get('nll_off', 0.0) * 1000:.1f}ms "
            f"nll_mem={stage_seconds.get('nll_mem', 0.0) * 1000:.1f}ms "
            f"plasticity={stage_seconds.get('plasticity', 0.0) * 1000:.1f}ms "
            f"append={stage_seconds.get('append_memory', 0.0) * 1000:.1f}ms "
            f"peak_mb={float(peak_mb):.0f}",
            flush=True,
        )

    def _ingest_maintenance_requests(
        self,
        *,
        model: torch.nn.Module,
        cache: TransactionalWakeCache,
        replay_eviction_loop: ReplayEvictionLoop | None,
        memory_write_tokens: int,
    ) -> None:
        """Turn latest request frames into replay work without packet scoring.

        A split maintenance rank consumes the request ring as a stream of probe
        frames. The packet-service rank owns low-latency residual serving, while
        the maintenance rank mirrors the append side locally from the same
        stream so learned maintenance has a cache to massage without a fragile
        peer receive.
        """
        if self.rank != self.memory_rank or self.produce_results:
            return
        if not self.pending_input_requests:
            return
        self.metrics["maintenance_score_path_skips"] += len(self.pending_input_requests)
        if replay_eviction_loop is None:
            dropped = len(self.pending_input_requests)
            self.pending_input_requests.clear()
            self.metrics["maintenance_request_frames_dropped_no_loop"] += int(dropped)
            self.metrics["last_drop_reason"] = "maintenance_no_replay_loop"
            return
        while self.pending_input_requests:
            slot = self.pending_input_requests.popleft()
            request_step = int(slot["step"])
            request_full_ids = slot["buffer"]
            valid_mask = _crct_valid_mask(request_full_ids)
            replay_eviction_loop.cache_probe(
                input_ids=request_full_ids,
                valid_mask=valid_mask,
                cue=None,
                cache_read_cutoff=None,
                step=request_step,
                stream_id=request_step,
            )
            self.metrics["maintenance_request_frames_ingested"] += 1
            self.metrics["last_received_request_step"] = request_step
            append_t0 = time.perf_counter()
            try:
                train_inputs = request_full_ids[:, :-1].to(dtype=torch.int32)
                train_targets = request_full_ids[:, 1:].to(dtype=torch.long)
                scored = _crct_packet_payload_inline(
                    model=model,
                    cache=cache,
                    inputs=train_inputs,
                    targets=train_targets,
                    step=request_step,
                    memory_write_tokens=int(memory_write_tokens),
                )
                write_records = scored.get("memory_write_records")
                n_records = len(write_records) if isinstance(write_records, list) else 0
                self.metrics["maintenance_local_append_packets"] += 1
                self.metrics["maintenance_local_append_records"] += int(n_records)
            except Exception as exc:
                self.metrics["maintenance_local_append_errors"] += 1
                self.metrics["errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "maintenance_local_append_error"
            finally:
                append_s = time.perf_counter() - append_t0
                self.metrics["maintenance_local_append_seconds_sum"] += float(
                    append_s
                )
                self.metrics["maintenance_local_append_seconds_max"] = max(
                    float(self.metrics["maintenance_local_append_seconds_max"]),
                    float(append_s),
                )

    def diagnostics(self) -> dict[str, Any]:
        out = dict(self.metrics)
        with self._weight_snapshot_lock:
            busy = self._weight_snapshot_writer_bank is not None
            pending = self._weight_snapshot_pending
            writer_bank = self._weight_snapshot_writer_bank
        out["weight_snapshot_pending_step"] = (
            int(pending["step"]) if pending is not None else -1
        )
        out["weight_snapshot_writer_bank"] = (
            int(writer_bank) if writer_bank is not None else -1
        )
        out["weight_snapshot_publisher_busy"] = bool(busy)
        with self._request_write_lock:
            request_busy = bool(
                self._request_write_pending is not None
                or (
                    self._request_write_thread is not None
                    and self._request_write_thread.is_alive()
                    and self.metrics.get("request_write_publisher_busy", False)
                )
            )
        out["request_write_publisher_busy"] = bool(request_busy)
        out.update(
            {
                "pending_result_broadcasts": 0,
                "pending_input_requests": len(self.pending_input_requests),
                "local_pending_batches": len(self.local_batches_by_step),
                "ready_result_pending": bool(self.ready_results),
                "ready_result_request_step": (
                    int(self.ready_results[0][0].get("step_id_int", -1))
                    if self.ready_results
                    else None
                ),
                "ready_result_queue_depth": len(self.ready_results),
            }
        )
        used = int(out.get("payloads_used", 0))
        out["payload_lag_steps_mean"] = (
            float(out["payload_lag_steps_sum"]) / float(used) if used else 0.0
        )
        scored = int(out.get("payloads_scored", 0))
        out["score_seconds_mean"] = (
            float(out["score_seconds_sum"]) / float(scored) if scored else 0.0
        )
        _add_crct_loss_reweight_metric_means(out)
        served = int(out.get("payloads_served", 0))
        out["packet_service_seconds_mean"] = (
            float(out["packet_service_seconds_sum"]) / float(served)
            if served
            else 0.0
        )
        out["packet_service_source_count_mean"] = (
            float(out["packet_service_source_count_sum"]) / float(served)
            if served
            else 0.0
        )
        packets = int(out.get("memory_packets_received", 0))
        out["memory_packet_lag_steps_mean"] = (
            float(out["memory_packet_lag_steps_sum"]) / float(packets)
            if packets
            else 0.0
        )
        plasticity_packets = int(out.get("plasticity_packets_received", 0))
        out["plasticity_lag_steps_mean"] = (
            float(out["plasticity_lag_steps_sum"]) / float(plasticity_packets)
            if plasticity_packets
            else 0.0
        )
        return out

    def _refresh_fast_slow_slow_model(
        self,
        *,
        model: torch.nn.Module,
        fast_slow: FastSlowConsolidator | None,
        step: int,
    ) -> torch.nn.Module | None:
        """Keep a rank-local slow-model mirror for exact fast-vs-slow scoring."""
        if fast_slow is None or not fast_slow.enabled or not fast_slow.slow_state:
            return None
        if self._fast_slow_slow_model is None:
            t0 = time.perf_counter()
            try:
                self._fast_slow_slow_model = copy.deepcopy(model)
                self._fast_slow_slow_model.requires_grad_(False)
            except Exception as exc:
                self._fast_slow_slow_model = None
                self.metrics["fast_slow_slow_mirror_last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "fast_slow_slow_mirror_create_error"
                return None
            elapsed = time.perf_counter() - t0
            self.metrics["fast_slow_slow_mirror_creations"] += 1
            self.metrics["fast_slow_slow_mirror_create_seconds_sum"] += float(elapsed)
            self.metrics["fast_slow_slow_mirror_create_seconds_max"] = max(
                float(self.metrics["fast_slow_slow_mirror_create_seconds_max"]),
                float(elapsed),
            )

        t0 = time.perf_counter()
        try:
            self._fast_slow_slow_model.train(bool(model.training))
            fast_slow.copy_slow_to_model(self._fast_slow_slow_model)
        except Exception as exc:
            self.metrics["fast_slow_slow_mirror_last_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            self.metrics["last_drop_reason"] = "fast_slow_slow_mirror_apply_error"
            return None
        elapsed = time.perf_counter() - t0
        self.metrics["fast_slow_slow_mirror_apply_seconds_sum"] += float(elapsed)
        self.metrics["fast_slow_slow_mirror_apply_seconds_max"] = max(
            float(self.metrics["fast_slow_slow_mirror_apply_seconds_max"]),
            float(elapsed),
        )
        self.metrics["fast_slow_slow_mirror_last_step"] = int(step)
        self.metrics["fast_slow_slow_mirror_version_lag_steps"] = max(
            0, int(step) - int(self._last_applied_weight_step)
        )
        return self._fast_slow_slow_model

    def wait_for_pending_collectives(self) -> None:
        return

    def close(self) -> None:
        with self._request_write_cv:
            self._request_write_stop = True
            self._request_write_cv.notify_all()
        if self._request_write_thread is not None:
            self._request_write_thread.join(timeout=0.25)
        self._push_shutdown_request()
        with self._weight_snapshot_cv:
            self._weight_snapshot_writer_stop = True
            self._weight_snapshot_cv.notify_all()
        if self._weight_publish_thread is not None:
            self._weight_publish_thread.join(timeout=0.25)
        return

    def unlink_shared_resources(self) -> None:
        if self.rank == self.coordinator_rank:
            owned_resources: list[tuple[Any, str]] = [
                (_ext.ShmRingTeacherRequest, self._teacher_request_ring_name),
                (_ext.PosixShm, self._teacher_request_payload_name),
            ]
            if self.produce_results:
                owned_resources.extend(
                    [
                        (_ext.ShmRingTeacherResult, self._teacher_result_ring_name),
                        (_ext.PosixShm, self._teacher_result_payload_name),
                        (_ext.PosixShm, self._weight_snapshot_shm_name),
                    ]
                )
            for cls, name in owned_resources:
                try:
                    cls.unlink(name)
                    self.metrics["mailbox_unlinks"] += 1
                except Exception:
                    pass
        return

    def _host_staging_like(self, src: torch.Tensor) -> torch.Tensor:
        want_pinned = bool(torch.cuda.is_available() and src.device.type == "cuda")
        try:
            return torch.empty_strided(
                tuple(src.shape),
                tuple(src.stride()),
                dtype=src.dtype,
                device="cpu",
                pin_memory=want_pinned,
            )
        except Exception:
            if want_pinned:
                self.metrics["host_pin_memory_failures"] += 1
                self.metrics["weight_snapshot_pin_memory_failures"] += 1
            return torch.empty_strided(
                tuple(src.shape),
                tuple(src.stride()),
                dtype=src.dtype,
                device="cpu",
            )

    def _compact_memory_packet_residual_to_cpu(
        self,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        residual_d = residual.detach()
        if residual_d.dim() == 2:
            residual_d = residual_d.unsqueeze(1)
        if residual_d.dim() != 3 or int(residual_d.shape[1]) != 1:
            self.metrics["memory_packet_sequence_residual_rejections"] += 1
            self.metrics["last_drop_reason"] = "memory_packet_sequence_residual"
            raise ValueError(
                "memory_residual packets must be compact with shape "
                f"(B, D) or (B, 1, D); got {tuple(residual.shape)}"
            )
        self.metrics["memory_packet_compact_residuals_sent"] += 1
        self.metrics["memory_packet_residual_elements_max"] = max(
            int(self.metrics["memory_packet_residual_elements_max"]),
            int(residual_d.numel()),
        )
        self.metrics["memory_packet_last_residual_shape"] = list(residual_d.shape)
        return residual_d.to(device="cpu", dtype=self.payload_dtype)

    def should_defer_low_priority_maintenance(self) -> bool:
        """Return True only when packet-request substrate capacity is tight."""
        if self.rank != self.memory_rank:
            return False
        self.metrics["low_priority_maintenance_checks"] += 1
        pending = len(self.pending_input_requests)
        self.metrics["low_priority_maintenance_pending_requests"] = int(pending)
        if (
            pending >= int(self._teacher_ring_capacity)
            or (
                self._ensure_teacher_request_consumer()
                and self._teacher_request_ring is not None
                and self._teacher_request_ring.size()
                >= max(1, int(self._teacher_ring_capacity) - 1)
            )
        ):
            self.metrics["low_priority_maintenance_defers"] += 1
            if pending >= int(self._teacher_ring_capacity):
                self.metrics["low_priority_maintenance_defer_pending_requests"] += 1
                self.metrics["low_priority_maintenance_last_reason"] = (
                    "pending_request_capacity"
                )
            else:
                self.metrics["low_priority_maintenance_defer_request_mailbox"] += 1
                self.metrics["low_priority_maintenance_last_reason"] = (
                    "request_ring_capacity"
                )
            return True
        self.metrics["low_priority_maintenance_allows"] += 1
        self.metrics["low_priority_maintenance_last_reason"] = "allowed"
        return False

    def _build_weight_snapshot_layout(
        self,
        state_dict: dict[str, Any],
    ) -> list[dict[str, Any]]:
        layout: list[dict[str, Any]] = []
        cursor = _align64(_WEIGHT_SNAPSHOT_HEADER.size)
        for name, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            tensor = value.detach()
            nbytes = int(tensor.numel() * tensor.element_size())
            layout.append(
                {
                    "name": str(name),
                    "shape": tuple(int(x) for x in tensor.shape),
                    "stride": tuple(int(x) for x in tensor.stride()),
                    "dtype": tensor.dtype,
                    "offset": int(cursor),
                    "nbytes": int(nbytes),
                }
            )
            cursor = _align64(cursor + nbytes)
        return layout

    def _ensure_weight_snapshot_writer(
        self,
        state_dict: dict[str, Any],
    ) -> bool:
        if self.rank != self.coordinator_rank:
            return False
        if self._weight_snapshot_layout is None:
            layout = self._build_weight_snapshot_layout(state_dict)
            buffer_bytes = _align64(
                max(
                    _WEIGHT_SNAPSHOT_HEADER.size,
                    max(
                        (
                            int(item["offset"]) + int(item["nbytes"])
                            for item in layout
                        ),
                        default=0,
                    ),
                )
            )
            region_bytes = int(buffer_bytes) * 2
            self._weight_snapshot_layout = layout
            self._weight_snapshot_buffer_bytes = int(buffer_bytes)
            self._weight_snapshot_region_bytes = int(region_bytes)
        assert self._weight_snapshot_layout is not None
        recreate = self._weight_snapshot_shm is None
        if recreate:
            try:
                _ext.PosixShm.unlink(self._weight_snapshot_shm_name)
            except Exception:
                pass
            self._weight_snapshot_shm = _ext.PosixShm(
                self._weight_snapshot_shm_name,
                int(self._weight_snapshot_region_bytes),
                True,
            )
            self.metrics["weight_snapshot_shm_bytes"] = int(
                self._weight_snapshot_region_bytes
            )
            self._weight_snapshot_stage_banks = []
        return True

    def _ensure_weight_snapshot_stage_banks(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        if self._weight_snapshot_stage_banks:
            return
        banks: list[dict[str, Any]] = []
        for bank_idx in range(int(self._weight_snapshot_bank_count)):
            tensors: dict[str, torch.Tensor] = {}
            for name, value in state_dict.items():
                if not torch.is_tensor(value):
                    continue
                src = value.detach()
                tensors[name] = torch.empty_like(src)
            banks.append(
                {
                    "bank": int(bank_idx),
                    "tensors": tensors,
                    "passthrough": {},
                    "event": None,
                    "start_event": None,
                    "step": -1,
                    "decision_payload": None,
                }
            )
        self._weight_snapshot_stage_banks = banks

    def _ensure_weight_snapshot_worker(self) -> None:
        if (
            self._weight_snapshot_worker_active
            and self._weight_publish_thread is not None
            and self._weight_publish_thread.is_alive()
        ):
            return
        with self._weight_snapshot_cv:
            self._weight_snapshot_writer_stop = False
            self._weight_snapshot_worker_active = True
        self._weight_publish_thread = threading.Thread(
            target=self._weight_snapshot_worker_loop,
            name="crct-weight-snapshot-shm-writer",
            daemon=True,
        )
        self._weight_publish_thread.start()

    def _weight_snapshot_worker_loop(self) -> None:
        while True:
            with self._weight_snapshot_cv:
                while (
                    self._weight_snapshot_pending is None
                    and not self._weight_snapshot_writer_stop
                ):
                    self._weight_snapshot_cv.wait()
                if self._weight_snapshot_writer_stop:
                    self._weight_snapshot_worker_active = False
                    return
                pending = self._weight_snapshot_pending
                self._weight_snapshot_pending = None
                assert pending is not None
                self._weight_snapshot_writer_bank = int(pending["bank"])
                self.metrics["weight_snapshot_publisher_busy"] = True
                self.metrics["weight_snapshot_worker_wakeups"] += 1

            stage_event = pending.get("event")
            stage_start_event = pending.get("start_event")
            wait_t0 = time.perf_counter()
            if stage_event is not None:
                try:
                    stage_event.synchronize()
                    if stage_start_event is not None:
                        stage_gpu_s = float(
                            stage_start_event.elapsed_time(stage_event)
                        ) / 1000.0
                        self.metrics["weight_snapshot_stage_gpu_seconds_sum"] += (
                            stage_gpu_s
                        )
                        self.metrics["weight_snapshot_stage_gpu_seconds_max"] = max(
                            float(
                                self.metrics[
                                    "weight_snapshot_stage_gpu_seconds_max"
                                ]
                            ),
                            stage_gpu_s,
                        )
                except Exception as exc:
                    self.metrics["weight_snapshot_publish_errors"] += 1
                    self.metrics["last_error"] = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()
                    self.metrics["last_drop_reason"] = "weight_snapshot_stage_wait_error"
                    with self._weight_snapshot_cv:
                        self._weight_snapshot_writer_bank = None
                        self.metrics["weight_snapshot_publisher_busy"] = False
                        self._weight_snapshot_cv.notify_all()
                    continue
            wait_s = time.perf_counter() - wait_t0
            self.metrics["weight_snapshot_stage_wait_seconds_sum"] += float(wait_s)
            self.metrics["weight_snapshot_stage_wait_seconds_max"] = max(
                float(self.metrics["weight_snapshot_stage_wait_seconds_max"]),
                float(wait_s),
            )

            copy_t0 = time.perf_counter()
            try:
                state_cpu: dict[str, Any] = {}
                pinned_buffers = 0
                pinned_bytes = 0
                pageable_buffers = 0
                pageable_bytes = 0
                staged_tensors = pending["staged_tensors"]
                for name, tensor in staged_tensors.items():
                    host = self._weight_snapshot_host_staging.get(name)
                    if (
                        host is None
                        or tuple(host.shape) != tuple(tensor.shape)
                        or tuple(host.stride()) != tuple(tensor.stride())
                        or host.dtype != tensor.dtype
                    ):
                        host = self._host_staging_like(tensor)
                        self._weight_snapshot_host_staging[name] = host
                    host.copy_(tensor.detach(), non_blocking=False)
                    nbytes = int(host.numel() * host.element_size())
                    if bool(host.is_pinned()):
                        pinned_buffers += 1
                        pinned_bytes += nbytes
                    else:
                        pageable_buffers += 1
                        pageable_bytes += nbytes
                    state_cpu[name] = host
                state_cpu.update(pending.get("passthrough", {}))
                self.metrics["weight_snapshot_host_pinned_buffers"] = int(
                    pinned_buffers
                )
                self.metrics["weight_snapshot_host_pinned_bytes"] = int(
                    pinned_bytes
                )
                self.metrics["weight_snapshot_host_pageable_buffers"] = int(
                    pageable_buffers
                )
                self.metrics["weight_snapshot_host_pageable_bytes"] = int(
                    pageable_bytes
                )
            except Exception as exc:
                self.metrics["weight_snapshot_publish_errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "weight_snapshot_writer_cpu_copy_error"
                with self._weight_snapshot_cv:
                    self._weight_snapshot_writer_bank = None
                    self.metrics["weight_snapshot_publisher_busy"] = False
                    self._weight_snapshot_cv.notify_all()
                continue
            copy_s = time.perf_counter() - copy_t0
            self.metrics["weight_snapshot_copy_started"] += 1
            self.metrics["weight_snapshot_copy_seconds_sum"] += float(copy_s)
            self.metrics["weight_snapshot_copy_seconds_max"] = max(
                float(self.metrics["weight_snapshot_copy_seconds_max"]),
                float(copy_s),
            )
            self.metrics["weight_snapshot_writer_cpu_copy_seconds_sum"] += float(
                copy_s
            )
            self.metrics["weight_snapshot_writer_cpu_copy_seconds_max"] = max(
                float(self.metrics["weight_snapshot_writer_cpu_copy_seconds_max"]),
                float(copy_s),
            )

            save_t0 = time.perf_counter()
            try:
                self._write_weight_snapshot_shm(
                    state_cpu=state_cpu,
                    step=int(pending["step"]),
                    decision_payload=pending.get("decision_payload"),
                )
                save_s = time.perf_counter() - save_t0
                self.metrics["weight_snapshot_published"] += 1
                self.metrics["weight_snapshot_last_published_step"] = int(
                    pending["step"]
                )
                self.metrics["weight_snapshot_save_seconds_sum"] += float(save_s)
                self.metrics["weight_snapshot_save_seconds_max"] = max(
                    float(self.metrics["weight_snapshot_save_seconds_max"]),
                    float(save_s),
                )
                self.metrics["weight_snapshot_shm_write_seconds_sum"] += float(save_s)
                self.metrics["weight_snapshot_shm_write_seconds_max"] = max(
                    float(self.metrics["weight_snapshot_shm_write_seconds_max"]),
                    float(save_s),
                )
            except Exception as exc:
                self.metrics["weight_snapshot_publish_errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "weight_snapshot_shm_write_error"
            finally:
                with self._weight_snapshot_cv:
                    self._weight_snapshot_writer_bank = None
                    self.metrics["weight_snapshot_publisher_busy"] = False
                    self._weight_snapshot_cv.notify_all()

    def _ensure_weight_snapshot_reader(
        self,
        state_dict: dict[str, Any],
    ) -> bool:
        if self.rank != self.memory_rank:
            return False
        if self._weight_snapshot_layout is None:
            self._weight_snapshot_layout = self._build_weight_snapshot_layout(state_dict)
            self._weight_snapshot_buffer_bytes = _align64(
                max(
                    _WEIGHT_SNAPSHOT_HEADER.size,
                    max(
                        (
                            int(item["offset"]) + int(item["nbytes"])
                            for item in self._weight_snapshot_layout
                        ),
                        default=0,
                    ),
                )
            )
            self._weight_snapshot_region_bytes = int(self._weight_snapshot_buffer_bytes) * 2
        if self._weight_snapshot_shm is None:
            try:
                self._weight_snapshot_shm = _ext.PosixShm(
                    self._weight_snapshot_shm_name,
                    0,
                    False,
                )
            except Exception:
                self.metrics["weight_snapshot_shm_attach_misses"] += 1
                return False
            self.metrics["weight_snapshot_shm_bytes"] = int(
                self._weight_snapshot_shm.size()
            )
        return True

    def _pack_weight_snapshot_header(
        self,
        *,
        step: int,
        active_buffer: int,
        tensor_count: int,
        total_bytes: int,
        decision_payload: dict[str, object] | None,
    ) -> bytes:
        constants = _ext.wire_event_constants()
        mode = str(decision_payload.get("mode", "")) if decision_payload else ""
        reason = str(decision_payload.get("reason", "")) if decision_payload else ""
        return _WEIGHT_SNAPSHOT_HEADER.pack(
            int(constants["TEACHER_WEIGHT_SNAPSHOT_MAGIC"]),
            int(constants["TEACHER_WIRE_VERSION"]),
            int(_WEIGHT_SNAPSHOT_HEADER.size),
            0,
            int(step),
            int(step),
            int(total_bytes),
            0,
            int(tensor_count),
            int(active_buffer),
            _fast_slow_mode_code(mode),
            int(bool(decision_payload.get("accepted", False))) if decision_payload else 0,
            int(decision_payload.get("step", 0)) if decision_payload else 0,
            float(decision_payload.get("alpha", 0.0)) if decision_payload else 0.0,
            float(decision_payload.get("gate", 0.0)) if decision_payload else 0.0,
            float(decision_payload.get("effective_alpha", 0.0)) if decision_payload else 0.0,
            _fast_slow_reason_code(reason),
            0,
        )

    def _unpack_weight_snapshot_header(self, raw: bytes) -> dict[str, Any] | None:
        if len(raw) != _WEIGHT_SNAPSHOT_HEADER.size:
            return None
        values = _WEIGHT_SNAPSHOT_HEADER.unpack(raw)
        constants = _ext.wire_event_constants()
        if int(values[0]) != int(constants["TEACHER_WEIGHT_SNAPSHOT_MAGIC"]):
            return None
        if int(values[2]) != int(_WEIGHT_SNAPSHOT_HEADER.size):
            return None
        return {
            "step": int(values[4]),
            "snapshot_version": int(values[5]),
            "total_bytes": int(values[6]),
            "tensor_count": int(values[8]),
            "active_buffer": int(values[9]),
            "fast_slow_mode": int(values[10]),
            "fast_slow_accepted": int(values[11]),
            "fast_slow_step": int(values[12]),
            "fast_slow_alpha": float(values[13]),
            "fast_slow_gate": float(values[14]),
            "fast_slow_effective_alpha": float(values[15]),
            "fast_slow_reason": int(values[16]),
        }

    def _write_weight_snapshot_shm(
        self,
        *,
        state_cpu: dict[str, Any],
        step: int,
        decision_payload: dict[str, object] | None,
    ) -> None:
        if self._weight_snapshot_shm is None or self._weight_snapshot_layout is None:
            raise RuntimeError("weight snapshot shm writer is not initialized")
        active_buffer = (int(step) & 1)
        buffer_base = active_buffer * int(self._weight_snapshot_buffer_bytes)
        self._weight_snapshot_shm.write_bytes(
            buffer_base,
            bytes(_WEIGHT_SNAPSHOT_HEADER.size),
        )
        for item in self._weight_snapshot_layout:
            tensor = state_cpu[item["name"]]
            if not torch.is_tensor(tensor):
                continue
            src = tensor.detach().to(dtype=item["dtype"], device="cpu").contiguous()
            self._weight_snapshot_shm.write_tensor(
                buffer_base + int(item["offset"]),
                src,
            )
        header = self._pack_weight_snapshot_header(
            step=int(step),
            active_buffer=active_buffer,
            tensor_count=len(self._weight_snapshot_layout),
            total_bytes=int(self._weight_snapshot_buffer_bytes),
            decision_payload=decision_payload,
        )
        # Publish latest-complete last. The consumer ignores buffers whose
        # header is absent or stale.
        self._weight_snapshot_shm.write_bytes(buffer_base, header)
        self.metrics["weight_snapshot_shm_writes"] += 1

    def _read_weight_snapshot_shm(
        self,
        *,
        model: torch.nn.Module,
        min_snapshot_version: int = -1,
    ) -> tuple[int, dict[str, torch.Tensor], dict[str, object] | None] | None:
        state_template = model.state_dict()
        if not self._ensure_weight_snapshot_reader(state_template):
            return None
        assert self._weight_snapshot_shm is not None
        assert self._weight_snapshot_layout is not None
        best_header: dict[str, Any] | None = None
        best_base = 0
        for active_buffer in (0, 1):
            buffer_base = active_buffer * int(self._weight_snapshot_buffer_bytes)
            raw = self._weight_snapshot_shm.read_bytes(
                buffer_base,
                _WEIGHT_SNAPSHOT_HEADER.size,
            )
            header = self._unpack_weight_snapshot_header(raw)
            if header is None:
                continue
            if int(header["active_buffer"]) != active_buffer:
                continue
            if best_header is None or int(header["snapshot_version"]) > int(
                best_header["snapshot_version"]
            ):
                best_header = header
                best_base = buffer_base
        if best_header is None:
            return None
        if int(best_header["snapshot_version"]) <= int(min_snapshot_version):
            return int(best_header["step"]), {}, None
        read_t0 = time.perf_counter()
        state_cpu: dict[str, torch.Tensor] = {}
        read_bytes = 0
        read_tensors = 0
        for item in self._weight_snapshot_layout:
            template = state_template[item["name"]]
            if not torch.is_tensor(template):
                continue
            out = self._weight_snapshot_read_staging.get(item["name"])
            if (
                out is None
                or tuple(out.shape) != tuple(item["shape"])
                or out.dtype != item["dtype"]
                or out.device.type != "cpu"
                or not out.is_contiguous()
            ):
                want_pinned = bool(
                    torch.cuda.is_available()
                    and getattr(template, "device", torch.device("cpu")).type
                    == "cuda"
                )
                try:
                    out = torch.empty(
                        tuple(item["shape"]),
                        dtype=item["dtype"],
                        device="cpu",
                        pin_memory=want_pinned,
                    )
                except Exception:
                    if want_pinned:
                        self.metrics["host_pin_memory_failures"] += 1
                        self.metrics["weight_snapshot_pin_memory_failures"] += 1
                    out = torch.empty(
                        tuple(item["shape"]),
                        dtype=item["dtype"],
                        device="cpu",
                    )
                self._weight_snapshot_read_staging[item["name"]] = out
            self._weight_snapshot_shm.read_tensor_into(
                best_base + int(item["offset"]),
                out,
            )
            state_cpu[item["name"]] = out
            read_bytes += int(item["nbytes"])
            read_tensors += 1
        read_s = time.perf_counter() - read_t0
        self.metrics["weight_snapshot_read_seconds_sum"] += float(read_s)
        self.metrics["weight_snapshot_read_seconds_max"] = max(
            float(self.metrics["weight_snapshot_read_seconds_max"]),
            float(read_s),
        )
        self.metrics["weight_snapshot_read_tensor_count"] = int(read_tensors)
        self.metrics["weight_snapshot_read_bytes"] = int(read_bytes)
        decision_payload = None
        if int(best_header["fast_slow_mode"]) != 0:
            decision_payload = {
                "mode": _fast_slow_mode_from_code(best_header["fast_slow_mode"]),
                "accepted": bool(best_header["fast_slow_accepted"]),
                "alpha": float(best_header["fast_slow_alpha"]),
                "gate": float(best_header["fast_slow_gate"]),
                "effective_alpha": float(best_header["fast_slow_effective_alpha"]),
                "step": int(best_header["fast_slow_step"]),
                "reason": _fast_slow_reason_from_code(best_header["fast_slow_reason"]),
            }
        self.metrics["weight_snapshot_shm_reads"] += 1
        return int(best_header["step"]), state_cpu, decision_payload

    def maybe_publish_weight_snapshot(
        self,
        *,
        model: torch.nn.Module,
        step: int,
        fast_slow_decision: FastSlowDecision | None = None,
    ) -> None:
        """Publish a latest-only teacher snapshot without a rank-3 rendezvous.

        Rank 0 performs only a vectorized device-local staging copy in the step
        body.  A persistent writer thread publishes the latest completed stage
        into the double-buffered shared-memory mirror.  Newer stages overwrite
        older unpublished stages; the trunk never waits for the writer and the
        memory rank consumes by version stamp rather than by a software TTL.
        """
        if self.rank != self.coordinator_rank:
            return
        self.metrics["weight_snapshot_attempts"] += 1
        state_dict = model.state_dict()
        if not self._ensure_weight_snapshot_writer(state_dict):
            return
        self._ensure_weight_snapshot_stage_banks(state_dict)
        self._ensure_weight_snapshot_worker()

        with self._weight_snapshot_cv:
            writer_bank = self._weight_snapshot_writer_bank
            preferred = int(step) % int(self._weight_snapshot_bank_count)
            bank_idx = preferred
            if writer_bank is not None and bank_idx == int(writer_bank):
                for candidate in range(int(self._weight_snapshot_bank_count)):
                    if candidate != int(writer_bank):
                        bank_idx = int(candidate)
                        break
            if self._weight_snapshot_pending is not None:
                self.metrics["weight_snapshot_latest_overwrites"] += 1
        bank = self._weight_snapshot_stage_banks[bank_idx]

        stage_t0 = time.perf_counter()
        try:
            staged_tensors: dict[str, torch.Tensor] = bank["tensors"]
            passthrough: dict[str, Any] = {}
            src_tensors: list[torch.Tensor] = []
            dst_tensors: list[torch.Tensor] = []
            stage_bytes = 0
            for name, value in state_dict.items():
                if not torch.is_tensor(value):
                    passthrough[name] = value
                    continue
                src = value.detach()
                dst = staged_tensors.get(name)
                if (
                    dst is None
                    or tuple(dst.shape) != tuple(src.shape)
                    or dst.dtype != src.dtype
                    or dst.device != src.device
                ):
                    dst = torch.empty_like(src)
                    staged_tensors[name] = dst
                src_tensors.append(src)
                dst_tensors.append(dst)
                stage_bytes += int(src.numel() * src.element_size())
            if dst_tensors:
                first_tensor = dst_tensors[0]
                stage_start_event = None
                stage_event = None
                if (
                    first_tensor.device.type == "cuda"
                    and torch.cuda.is_available()
                ):
                    stream = torch.cuda.current_stream(first_tensor.device)
                    stage_start_event = torch.cuda.Event(enable_timing=True)
                    stage_event = torch.cuda.Event(enable_timing=True)
                    stage_start_event.record(stream)
                torch._foreach_copy_(dst_tensors, src_tensors)
                if stage_event is not None:
                    stage_event.record(torch.cuda.current_stream(first_tensor.device))
            else:
                stage_start_event = None
                stage_event = None
        except Exception as exc:
            self.metrics["weight_snapshot_publish_errors"] += 1
            self.metrics["last_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            self.metrics["last_drop_reason"] = "weight_snapshot_stage_error"
            return
        stage_s = time.perf_counter() - stage_t0
        self.metrics["weight_snapshot_stage_started"] += 1
        self.metrics["weight_snapshot_stage_enqueue_seconds_sum"] += float(stage_s)
        self.metrics["weight_snapshot_stage_enqueue_seconds_max"] = max(
            float(self.metrics["weight_snapshot_stage_enqueue_seconds_max"]),
            float(stage_s),
        )
        self.metrics["weight_snapshot_stage_tensor_count"] = len(staged_tensors)
        self.metrics["weight_snapshot_stage_bytes"] = int(stage_bytes)
        self.metrics["weight_snapshot_publish_started"] += 1
        version_step = int(step)
        decision_payload = fast_slow_decision_to_dict(fast_slow_decision)
        if decision_payload is not None:
            self.metrics["fast_slow_snapshot_decisions_published"] += 1
        bank["passthrough"] = passthrough
        bank["event"] = stage_event
        bank["start_event"] = stage_start_event
        bank["step"] = version_step
        bank["decision_payload"] = decision_payload
        with self._weight_snapshot_cv:
            self._weight_snapshot_pending = {
                "bank": int(bank_idx),
                "staged_tensors": staged_tensors,
                "passthrough": passthrough,
                "event": stage_event,
                "start_event": stage_start_event,
                "step": version_step,
                "decision_payload": decision_payload,
            }
            self._weight_snapshot_cv.notify_all()

    def poll_weight_snapshot(
        self,
        *,
        model: torch.nn.Module,
        step: int,
        replay_eviction_loop: ReplayEvictionLoop | None = None,
        fast_slow: FastSlowConsolidator | None = None,
    ) -> None:
        """Apply the latest rank0 snapshot on the memory rank if available.

        When a fresh snapshot lands and a replay_eviction_loop is given,
        also pushes the now-current LM head + final_norm into the loop's
        CPU evidence engine. This couples the engine's cue NLL to the same
        teacher-freshness contract the trunk oracle on GPU3 already obeys.
        """
        if self.rank != self.memory_rank:
            return
        try:
            snapshot = self._read_weight_snapshot_shm(
                model=model,
                min_snapshot_version=int(self._last_applied_weight_step),
            )
            if snapshot is None:
                return
            version_step, state_cpu, decision_payload = snapshot
            if version_step <= self._last_applied_weight_step:
                self.metrics["weight_snapshot_stat_skips"] += 1
                self.metrics["weight_snapshot_version_lag_steps"] = max(
                    0, int(step) - int(self._last_applied_weight_step)
                )
                return
            self.metrics["weight_snapshot_apply_attempts"] += 1
            apply_t0 = time.perf_counter()
            load_result = model.load_state_dict(
                state_cpu,
                strict=False,
            )
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            if missing or unexpected:
                self.metrics["last_drop_reason"] = "weight_snapshot_partial_load"
            if fast_slow is not None:
                decision = fast_slow_decision_from_dict(decision_payload)
                if decision is not None:
                    fast_slow.apply_decision(model, decision)
                    self.metrics["fast_slow_snapshot_decisions_applied"] += 1
            elapsed = time.perf_counter() - apply_t0
            self._last_applied_weight_step = version_step
            self.metrics["weight_snapshot_applied"] += 1
            self.metrics["weight_snapshot_last_applied_step"] = version_step
            self.metrics["weight_snapshot_version_lag_steps"] = max(
                0, int(step) - version_step
            )
            self.metrics["weight_snapshot_apply_seconds_sum"] += float(elapsed)
            self.metrics["weight_snapshot_apply_seconds_max"] = max(
                float(self.metrics["weight_snapshot_apply_seconds_max"]),
                float(elapsed),
            )
            if replay_eviction_loop is not None:
                try:
                    replay_eviction_loop.refresh_evidence_weights(
                        norm_weight=model.final_norm.weight,
                        lm_head_weight=model.lm_head.weight,
                        step=version_step,
                    )
                except Exception as exc:  # pragma: no cover - telemetry only
                    self.metrics["last_drop_reason"] = (
                        "evidence_engine_lm_head_refresh_error"
                    )
                    self.metrics["last_error"] = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()
        except Exception as exc:
            self.metrics["weight_snapshot_apply_errors"] += 1
            self.metrics["last_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            self.metrics["last_drop_reason"] = "weight_snapshot_apply_error"

    def _ensure_request_worker(self) -> None:
        if (
            self._request_write_worker_active
            and self._request_write_thread is not None
            and self._request_write_thread.is_alive()
        ):
            return
        with self._request_write_cv:
            self._request_write_stop = False
            self._request_write_worker_active = True
        self._request_write_thread = threading.Thread(
            target=self._request_writer_loop,
            name="crct-teacher-request-shm-writer",
            daemon=True,
        )
        self._request_write_thread.start()

    def _record_local_request_batch(
        self,
        *,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
    ) -> None:
        # Inputs/targets are immutable batch tensors for this step. Keep
        # references for matched-label replay instead of cloning a second
        # GPU batch on rank0's hot path.
        self.local_batches_by_step[int(step)] = (
            inputs.detach(),
            targets.detach(),
        )
        self.local_batch_order.append(int(step))
        self.metrics["requests_stored"] += 1
        while len(self.local_batch_order) > self.max_local_batches:
            old_step = self.local_batch_order.popleft()
            if self.local_batches_by_step.pop(old_step, None) is not None:
                self.metrics["local_request_evictions"] += 1
                self.metrics["last_drop_reason"] = "local_request_evicted"
        self.metrics["max_local_pending_batches"] = max(
            int(self.metrics["max_local_pending_batches"]),
            len(self.local_batches_by_step),
        )

    def _submit_request(
        self,
        *,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
    ) -> None:
        if self.rank != self.coordinator_rank:
            return
        submit_t0 = time.perf_counter()
        self._record_local_request_batch(
            inputs=inputs,
            targets=targets,
            step=int(step),
        )
        ready_event = None
        if inputs.device.type == "cuda" and torch.cuda.is_available():
            ready_event = torch.cuda.Event(enable_timing=False)
            ready_event.record(torch.cuda.current_stream(inputs.device))
        self._ensure_request_worker()
        with self._request_write_cv:
            if self._request_write_pending is not None:
                self.metrics["request_write_latest_overwrites"] += 1
            self._request_write_pending = {
                "inputs": inputs.detach(),
                "targets": targets.detach(),
                "step": int(step),
                "ready_event": ready_event,
            }
            self._request_write_cv.notify()
        elapsed = time.perf_counter() - submit_t0
        self.metrics["request_submit_seconds_sum"] += float(elapsed)
        self.metrics["request_submit_seconds_max"] = max(
            float(self.metrics["request_submit_seconds_max"]),
            float(elapsed),
        )

    def _request_writer_loop(self) -> None:
        while True:
            with self._request_write_cv:
                while (
                    self._request_write_pending is None
                    and not self._request_write_stop
                ):
                    self._request_write_cv.wait()
                if self._request_write_stop:
                    self._request_write_worker_active = False
                    return
                pending = self._request_write_pending
                self._request_write_pending = None
                self.metrics["request_write_publisher_busy"] = True
                self.metrics["request_writer_wakeups"] += 1
            assert pending is not None
            try:
                self._write_request(
                    inputs=pending["inputs"],
                    targets=pending["targets"],
                    step=int(pending["step"]),
                    ready_event=pending.get("ready_event"),
                )
            finally:
                with self._request_write_cv:
                    self.metrics["request_write_publisher_busy"] = False
                    self._request_write_cv.notify_all()

    def _write_request(
        self,
        *,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int,
        ready_event: torch.cuda.Event | None = None,
    ) -> None:
        if self._teacher_request_ring is None or self._teacher_request_payload is None:
            self.metrics["last_drop_reason"] = "teacher_shm_request_not_ready"
            return
        stage_t0 = time.perf_counter()
        stream = None
        if inputs.device.type == "cuda" and torch.cuda.is_available():
            if self._request_copy_stream is None:
                self._request_copy_stream = torch.cuda.Stream(device=inputs.device)
            stream = self._request_copy_stream
        if stream is not None:
            with torch.cuda.stream(stream):
                if ready_event is not None:
                    stream.wait_event(ready_event)
                full_ids = _crct_full_input_ids(inputs, targets).to(
                    dtype=torch.int32
                ).contiguous()
        else:
            full_ids = _crct_full_input_ids(inputs, targets).to(
                dtype=torch.int32
            ).contiguous()
        if tuple(full_ids.shape) != self.full_ids_shape:
            raise ValueError(
                "CRCT mailbox transport saw a dynamic batch shape: "
                f"{tuple(full_ids.shape)} != {self.full_ids_shape}"
            )
        full_ids_staged = full_ids.detach()
        stage_s = time.perf_counter() - stage_t0
        self.metrics["request_stage_started"] += 1
        self.metrics["request_stage_seconds_sum"] += float(stage_s)
        self.metrics["request_stage_seconds_max"] = max(
            float(self.metrics["request_stage_seconds_max"]),
            float(stage_s),
        )
        self.metrics["requests_started"] += 1
        self.metrics["request_broadcasts_started"] += 1
        request_step = int(step)
        try:
            copy_t0 = time.perf_counter()
            host = self._request_host_staging
            if (
                host is None
                or tuple(host.shape) != tuple(full_ids_staged.shape)
                or tuple(host.stride()) != tuple(full_ids_staged.stride())
                or host.dtype != full_ids_staged.dtype
            ):
                self._request_host_staging = self._host_staging_like(
                    full_ids_staged
                )
                host = self._request_host_staging
            if stream is not None:
                with torch.cuda.stream(stream):
                    host.copy_(full_ids_staged, non_blocking=True)
                stream.synchronize()
            else:
                host.copy_(full_ids_staged, non_blocking=False)
            copy_s = time.perf_counter() - copy_t0
            self.metrics["request_writer_cpu_copy_seconds_sum"] += float(copy_s)
            self.metrics["request_writer_cpu_copy_seconds_max"] = max(
                float(self.metrics["request_writer_cpu_copy_seconds_max"]),
                float(copy_s),
            )
            self.metrics["request_host_pinned"] = bool(host.is_pinned())
            self.metrics["request_host_stage_bytes"] = int(
                host.numel() * host.element_size()
            )
            request_id = int(self._teacher_request_seq)
            self._teacher_request_seq += 1
            slot = request_id % int(self._teacher_ring_capacity)
            slot_base = slot * int(self._teacher_request_slot_bytes)
            full_ids_slice, _ = self._write_teacher_slice(
                shm=self._teacher_request_payload,
                slot_base=slot_base,
                cursor=0,
                tensor=host,
                dtype=torch.int32,
            )
            event = {
                "event_type": _TEACHER_REQUEST_EVENT_TYPE,
                "source_rank": int(self.rank),
                "status": 0,
                "flags": 0,
                "slice_count": int(_ext.wire_event_constants()["TEACHER_REQUEST_SLICES"]),
                "request_id": request_id,
                "step": request_step,
                "weight_snapshot_version": max(
                    0, int(self.metrics.get("weight_snapshot_last_published_step", -1))
                ),
                "full_ids": full_ids_slice,
            }
            if self._teacher_request_ring.push(event):
                self.metrics["teacher_shm_request_events_pushed"] += 1
                self.metrics["mailbox_request_writes"] += 1
                self.metrics["request_broadcasts_completed"] += 1
                self.metrics["last_sent_request_step"] = request_step
            else:
                self.metrics["teacher_shm_request_ring_full_drops"] += 1
                self.metrics["last_drop_reason"] = "teacher_request_ring_full"
        except Exception as exc:
            self.metrics["errors"] += 1
            self.metrics["last_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            self.metrics["last_drop_reason"] = "request_shm_write_error"

    def _poll_requests(self) -> int:
        if not self._ensure_teacher_request_consumer():
            return 0
        assert self._teacher_request_ring is not None
        assert self._teacher_request_payload is not None
        popped = 0
        latest_event: dict[str, Any] | None = None
        shutdown_seen = False
        while True:
            event = self._teacher_request_ring.pop()
            if event is None:
                break
            popped += 1
            if int(event.get("flags", 0)) & _TEACHER_REQUEST_FLAG_SHUTDOWN:
                if latest_event is not None:
                    self.metrics["memory_rank_request_events_superseded"] += 1
                    self.metrics["completed_requests_dropped"] += 1
                    latest_event = None
                shutdown_seen = True
                self.metrics["teacher_shm_request_shutdown_events_popped"] += 1
                continue
            if shutdown_seen:
                self.metrics["memory_rank_request_events_superseded"] += 1
                self.metrics["completed_requests_dropped"] += 1
                continue
            if latest_event is not None:
                self.metrics["memory_rank_request_events_superseded"] += 1
                self.metrics["completed_requests_dropped"] += 1
            latest_event = event
        if shutdown_seen:
            if self.pending_input_requests:
                dropped = len(self.pending_input_requests)
                self.pending_input_requests.clear()
                self.metrics["memory_rank_request_events_superseded"] += int(dropped)
                self.metrics["completed_requests_dropped"] += int(dropped)
            self._request_shutdown_seen = True
            self.metrics["teacher_request_shutdown_seen"] = True
            self.metrics["last_drop_reason"] = "teacher_request_shutdown"
            self.metrics["teacher_shm_request_events_popped"] += int(popped)
            return int(popped)
        if latest_event is not None:
            if self.pending_input_requests:
                dropped = len(self.pending_input_requests)
                self.pending_input_requests.clear()
                self.metrics["memory_rank_request_events_superseded"] += int(dropped)
                self.metrics["completed_requests_dropped"] += int(dropped)
            try:
                full_ids_cpu = self._read_teacher_slice(
                    self._teacher_request_payload,
                    latest_event["full_ids"],  # type: ignore[arg-type]
                )
                if full_ids_cpu is None:
                    self.metrics["last_drop_reason"] = (
                        "teacher_request_empty_full_ids"
                    )
                else:
                    full_ids = full_ids_cpu.to(device=self.device, dtype=torch.int32)
                    step = int(latest_event["step"])
                    self.pending_input_requests.append(
                        {"step": step, "buffer": full_ids}
                    )
                    self.metrics["memory_rank_pump_last_request_step"] = int(step)
                    self.metrics["mailbox_request_reads"] += 1
                    if self.metrics["memory_rank_request_events_superseded"]:
                        self.metrics["last_drop_reason"] = "newer_request_won"
                    self.metrics["max_pending_input_requests"] = max(
                        int(self.metrics["max_pending_input_requests"]),
                        len(self.pending_input_requests),
                    )
            except Exception as exc:
                self.metrics["errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "request_shm_read_error"
        self.metrics["teacher_shm_request_events_popped"] += int(popped)
        return int(popped)

    def _plasticity_ema_update(
        self,
        scored: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        coverage = scored.get("plasticity_coverage")
        confidence = scored.get("plasticity_confidence")
        budget = scored.get("plasticity_budget")
        if not (
            isinstance(coverage, torch.Tensor)
            and isinstance(confidence, torch.Tensor)
            and isinstance(budget, torch.Tensor)
        ):
            return None
        cov = coverage.detach().float().reshape(-1).cpu()
        conf = confidence.detach().float().reshape(-1).cpu()
        bud = budget.detach().float().reshape(-1).cpu()
        if not (cov.numel() == conf.numel() == bud.numel()):
            raise ValueError(
                "plasticity packet tensors must share shape; got "
                f"coverage={tuple(cov.shape)} confidence={tuple(conf.shape)} "
                f"budget={tuple(bud.shape)}"
            )
        beta = float(self.plasticity_ema_beta)
        if self._plasticity_coverage_ema is None:
            self._plasticity_coverage_ema = cov.clone()
            self._plasticity_confidence_ema = conf.clone()
            self._plasticity_budget_ema = bud.clone()
        else:
            assert self._plasticity_confidence_ema is not None
            assert self._plasticity_budget_ema is not None
            if self._plasticity_coverage_ema.numel() != cov.numel():
                raise ValueError(
                    "plasticity packet channel count changed: "
                    f"{self._plasticity_coverage_ema.numel()} -> {cov.numel()}"
                )
            self._plasticity_coverage_ema.mul_(beta).add_(cov, alpha=1.0 - beta)
            self._plasticity_confidence_ema.mul_(beta).add_(
                conf, alpha=1.0 - beta
            )
            self._plasticity_budget_ema.mul_(beta).add_(bud, alpha=1.0 - beta)
        return (
            self._plasticity_coverage_ema,
            self._plasticity_confidence_ema,
            self._plasticity_budget_ema,
        )

    def _publish_memory_write_commits(
        self,
        *,
        request_step: int,
        scored: dict[str, Any],
        slot_commit_transport: "_CrctSlotCommitPeerTransport | None",
    ) -> None:
        write_records = scored.get("memory_write_records")
        if not isinstance(write_records, list) or not write_records:
            return
        self.metrics["slot_append_records_seen"] += len(write_records)
        if slot_commit_transport is None:
            self.metrics["slot_append_commits_local_only"] += len(write_records)
            return
        for record in write_records:
            if not isinstance(record, dict):
                self.metrics["slot_append_commit_publish_failures"] += 1
                self.metrics["last_drop_reason"] = "slot_append_bad_record"
                continue
            tensor = record.get("tensor")
            if not isinstance(tensor, torch.Tensor):
                self.metrics["slot_append_commit_publish_failures"] += 1
                self.metrics["last_drop_reason"] = "slot_append_missing_tensor"
                continue
            commit = SlotCommit(
                slot_id=int(record.get("slot_id", -1)),
                action=SLOT_COMMIT_APPEND,
                step=int(request_step),
                base_generation=None,
                new_generation=int(record.get("generation", 0)),
                bucket_id=int(record.get("bucket_id", -1)),
                event_id=int(record.get("event_id", 0)),
                tensor=tensor.detach(),
                reason="packet_append",
            )
            if slot_commit_transport.submit_peer(commit):
                self.metrics["slot_append_commits_published"] += 1
            else:
                self.metrics["slot_append_commit_publish_failures"] += 1
                self.metrics["last_drop_reason"] = "slot_append_submit_failed"

    def _write_result(self, *, request_step: int, scored: dict[str, Any]) -> None:
        if not self.produce_results:
            return
        if not self._ensure_teacher_result_producer():
            self.metrics["last_drop_reason"] = "teacher_shm_result_not_ready"
            return
        assert self._teacher_result_ring is not None
        assert self._teacher_result_payload is not None
        result_id = int(self._teacher_result_seq)
        self._teacher_result_seq += 1
        slot = result_id % int(self._teacher_ring_capacity)
        slot_base = slot * int(self._teacher_result_slot_bytes)
        cursor = 0
        slices_by_name: dict[str, dict[str, object]] = {
            name: _teacher_empty_slice() for name in _TEACHER_RESULT_SLICE_NAMES
        }
        try:
            for key in ("target", "confidence", "loss_weight", "utility"):
                slices_by_name[key], cursor = self._write_teacher_slice(
                    shm=self._teacher_result_payload,
                    slot_base=slot_base,
                    cursor=cursor,
                    tensor=scored[key],
                    dtype=self.payload_dtype,
                )
        except Exception as exc:
            self.metrics["errors"] += 1
            self.metrics["last_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            self.metrics["last_drop_reason"] = "result_shm_base_write_error"
            return
        readiness = scored.get("fast_slow_readiness")
        if isinstance(readiness, dict):
            self.metrics["fast_slow_readiness_result_payloads"] += 1
        decision = fast_slow_decision_from_dict(scored.get("fast_slow_decision"))
        memory_residual = scored.get("memory_residual")
        memory_gate = scored.get("memory_gate")
        if isinstance(memory_residual, torch.Tensor) and isinstance(memory_gate, torch.Tensor):
            try:
                residual_cpu = self._compact_memory_packet_residual_to_cpu(
                    memory_residual
                )
                slices_by_name["memory_residual"], cursor = self._write_teacher_slice(
                    shm=self._teacher_result_payload,
                    slot_base=slot_base,
                    cursor=cursor,
                    tensor=residual_cpu,
                    dtype=self.payload_dtype,
                )
                gate_alias_target = bool(scored.get("memory_gate_alias_target", False))
                if gate_alias_target and tuple(memory_gate.shape) == tuple(scored["target"].shape):
                    gate_bytes = 0
                    self.metrics["memory_packet_gate_alias_target_sent"] += 1
                    self.metrics["memory_packet_last_gate_shape"] = ["alias:target"]
                    slices_by_name["memory_gate"] = _teacher_empty_slice()
                else:
                    slices_by_name["memory_gate"], cursor = self._write_teacher_slice(
                        shm=self._teacher_result_payload,
                        slot_base=slot_base,
                        cursor=cursor,
                        tensor=memory_gate,
                        dtype=self.payload_dtype,
                    )
                    gate_bytes = int(slices_by_name["memory_gate"]["nbytes"])
                    self.metrics["memory_packet_last_gate_shape"] = list(
                        memory_gate.shape
                    )
                residual_bytes = int(slices_by_name["memory_residual"]["nbytes"])
                packet_bytes = residual_bytes + gate_bytes
                self.metrics["memory_packets_sent"] += 1
                self.metrics["memory_packet_bytes_sent"] += int(packet_bytes)
                self.metrics["memory_packet_bytes_sent_max"] = max(
                    int(self.metrics["memory_packet_bytes_sent_max"]),
                    int(packet_bytes),
                )
                self.metrics["memory_packet_gate_elements_max"] = max(
                    int(self.metrics["memory_packet_gate_elements_max"]),
                    int(memory_gate.numel()),
                )
            except Exception as exc:
                self.metrics["errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "result_shm_memory_packet_write_error"
                if isinstance(exc, ValueError):
                    raise
                return
        else:
            self.metrics["memory_packet_missing_payloads"] += 1
        plasticity = self._plasticity_ema_update(scored)
        if plasticity is None:
            self.metrics["plasticity_packets_missing"] += 1
        else:
            coverage_ema, confidence_ema, budget_ema = plasticity
            try:
                for key, tensor in (
                    ("plasticity_coverage", coverage_ema),
                    ("plasticity_confidence", confidence_ema),
                    ("plasticity_budget", budget_ema),
                ):
                    slices_by_name[key], cursor = self._write_teacher_slice(
                        shm=self._teacher_result_payload,
                        slot_base=slot_base,
                        cursor=cursor,
                        tensor=tensor,
                        dtype=self.payload_dtype,
                    )
            except Exception as exc:
                self.metrics["errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "result_shm_plasticity_write_error"
                return
            packet_bytes = int(
                sum(
                    int(slices_by_name[key]["nbytes"])
                    for key in ("plasticity_coverage", "plasticity_confidence", "plasticity_budget")
                )
            )
            self.metrics["plasticity_packets_sent"] += 1
            self.metrics["plasticity_packet_bytes_sent"] += int(packet_bytes)
            self.metrics["plasticity_packet_bytes_sent_max"] = max(
                int(self.metrics["plasticity_packet_bytes_sent_max"]),
                int(packet_bytes),
            )
            self.metrics["plasticity_packet_last_shape"] = list(budget_ema.shape)
            self.metrics["plasticity_budget_mean_sent"] = float(
                budget_ema.mean().item()
            )
            self.metrics["plasticity_budget_max_sent"] = float(
                budget_ema.max().item()
            )
            self.metrics["plasticity_confidence_mean_sent"] = float(
                confidence_ema.mean().item()
            )
            self.metrics["plasticity_coverage_abs_mean_sent"] = float(
                coverage_ema.abs().mean().item()
            )
        if decision is not None:
            self.metrics["fast_slow_decisions_result_payloads"] += 1
        result_event = {
            "event_type": 7,
            "source_rank": int(self.rank),
            "status": 0,
            "flags": 1 if (
                isinstance(memory_gate, torch.Tensor)
                and bool(scored.get("memory_gate_alias_target", False))
                and tuple(memory_gate.shape) == tuple(scored["target"].shape)
            ) else 0,
            "slice_count": int(_ext.wire_event_constants()["TEACHER_RESULT_SLICES"]),
            "request_id": result_id,
            "step": int(request_step),
            "weight_snapshot_version": max(0, int(self._last_applied_weight_step)),
            "payload_version": result_id,
            "score_seconds": float(scored.get("score_seconds", 0.0))
            if not isinstance(scored.get("score_seconds"), torch.Tensor)
            else float(scored["score_seconds"].detach().cpu().item()),
            "packet_seconds": float(scored.get("packet_seconds", 0.0))
            if not isinstance(scored.get("packet_seconds"), torch.Tensor)
            else float(scored["packet_seconds"].detach().cpu().item()),
            "target_token_count": int(scored["target"].numel()),
            "hidden_dim": int(self.hidden_dim),
            "plasticity_dim": int(
                plasticity[2].numel() if plasticity is not None else 0
            ),
            "fast_slow_mode": _fast_slow_mode_code(decision.mode) if decision is not None else 0,
            "fast_slow_accepted": int(bool(decision.accepted)) if decision is not None else 0,
            "fast_slow_step": int(decision.step) if decision is not None else 0,
            "fast_slow_alpha": float(decision.alpha) if decision is not None else 0.0,
            "fast_slow_gate": float(decision.gate) if decision is not None else 0.0,
            "fast_slow_effective_alpha": float(decision.effective_alpha) if decision is not None else 0.0,
            "fast_slow_reason": _fast_slow_reason_code(decision.reason) if decision is not None else 0,
            "slices": [slices_by_name[name] for name in _TEACHER_RESULT_SLICE_NAMES],
        }
        if self._teacher_result_ring.push(result_event):
            self.metrics["payloads_sent"] += 1
            self.metrics["payloads_received"] += 0
            self.metrics["mailbox_result_writes"] += 1
            self.metrics["result_broadcasts_started"] += 1
            self.metrics["result_broadcasts_completed"] += 1
            self.metrics["teacher_shm_result_events_pushed"] += 1
            self.metrics["last_sent_request_step"] = int(request_step)
        else:
            self.metrics["teacher_shm_result_ring_full_drops"] += 1
            self.metrics["last_drop_reason"] = "teacher_result_ring_full"

    def _poll_results(
        self,
        *,
        current_step: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
        if not self.produce_results:
            return None
        if self._teacher_result_ring is None or self._teacher_result_payload is None:
            return self._pop_ready_result(current_step=current_step)
        while True:
            result_event = self._teacher_result_ring.pop()
            if result_event is None:
                break
            self.metrics["teacher_shm_result_events_popped"] += 1
            request_step = int(result_event["step"])
            lag = int(current_step) - request_step
            batch = self.local_batches_by_step.pop(request_step, None)
            try:
                self.local_batch_order.remove(request_step)
            except ValueError:
                pass
            if batch is None:
                self.metrics["orphan_payloads_dropped"] += 1
                self.metrics["last_drop_reason"] = "payload_without_local_batch"
                continue
            if self.max_payload_lag_steps > 0 and lag > self.max_payload_lag_steps:
                self.metrics["stale_payloads_dropped"] += 1
                self.metrics["last_drop_reason"] = "payload_too_stale"
                continue
            slices = {
                name: desc
                for name, desc in zip(
                    _TEACHER_RESULT_SLICE_NAMES,
                    result_event["slices"],
                    strict=True,
                )
            }
            try:
                tensors_cpu = {
                    name: self._read_teacher_slice(self._teacher_result_payload, desc)
                    for name, desc in slices.items()
                }
            except Exception as exc:
                self.metrics["errors"] += 1
                self.metrics["last_error"] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.metrics["last_drop_reason"] = "result_shm_read_error"
                continue
            target_cpu = tensors_cpu.get("target")
            confidence_cpu = tensors_cpu.get("confidence")
            loss_weight_cpu = tensors_cpu.get("loss_weight")
            utility_cpu = tensors_cpu.get("utility")
            if not (
                isinstance(target_cpu, torch.Tensor)
                and isinstance(confidence_cpu, torch.Tensor)
                and isinstance(loss_weight_cpu, torch.Tensor)
                and isinstance(utility_cpu, torch.Tensor)
            ):
                self.metrics["last_drop_reason"] = "result_shm_missing_base_tensor"
                continue
            payload = {
                "step_id": torch.tensor(request_step, device=self.device),
                "step_id_int": int(request_step),
                "target": target_cpu.to(
                    device=self.device, dtype=torch.float32
                ),
                "confidence": confidence_cpu.to(
                    device=self.device, dtype=torch.float32
                ),
                "loss_weight": loss_weight_cpu.to(
                    device=self.device, dtype=torch.float32
                ),
                "utility": utility_cpu.to(
                    device=self.device, dtype=torch.float32
                ),
            }
            if int(result_event.get("fast_slow_mode", 0)) != 0:
                payload["fast_slow_decision"] = {
                    "mode": _fast_slow_mode_from_code(result_event["fast_slow_mode"]),
                    "accepted": bool(result_event["fast_slow_accepted"]),
                    "alpha": float(result_event["fast_slow_alpha"]),
                    "gate": float(result_event["fast_slow_gate"]),
                    "effective_alpha": float(
                        result_event["fast_slow_effective_alpha"]
                    ),
                    "step": int(result_event["fast_slow_step"]),
                    "reason": _fast_slow_reason_from_code(
                        result_event["fast_slow_reason"]
                    ),
                }
            memory_residual = tensors_cpu.get("memory_residual")
            memory_gate = tensors_cpu.get("memory_gate")
            gate_alias_target = bool(int(result_event.get("flags", 0)) & 1)
            if gate_alias_target:
                memory_gate = target_cpu
            if isinstance(memory_residual, torch.Tensor) and isinstance(memory_gate, torch.Tensor):
                residual = memory_residual
                if residual.dim() == 2:
                    residual = residual.unsqueeze(1)
                if residual.dim() != 3 or int(residual.shape[1]) != 1:
                    self.metrics["memory_packet_sequence_residual_rejections"] += 1
                    self.metrics["last_drop_reason"] = (
                        "memory_packet_sequence_residual_received"
                    )
                    continue
                payload["memory_residual"] = residual.to(
                    device=self.device, dtype=torch.float32
                )
                payload["memory_gate"] = memory_gate.to(
                    device=self.device, dtype=torch.float32
                )
                residual_bytes = int(residual.numel() * residual.element_size())
                gate_bytes = (
                    0
                    if gate_alias_target
                    else int(memory_gate.numel() * memory_gate.element_size())
                )
                packet_bytes = residual_bytes + gate_bytes
                self.metrics["memory_packets_received"] += 1
                self.metrics["memory_packet_compact_residuals_received"] += 1
                if gate_alias_target:
                    self.metrics["memory_packet_gate_alias_target_received"] += 1
                self.metrics["memory_packet_bytes_received"] += int(packet_bytes)
                self.metrics["memory_packet_bytes_received_max"] = max(
                    int(self.metrics["memory_packet_bytes_received_max"]),
                    int(packet_bytes),
                )
                self.metrics["memory_packet_residual_elements_max"] = max(
                    int(self.metrics["memory_packet_residual_elements_max"]),
                    int(residual.numel()),
                )
                self.metrics["memory_packet_gate_elements_max"] = max(
                    int(self.metrics["memory_packet_gate_elements_max"]),
                    int(memory_gate.numel()),
                )
                self.metrics["memory_packet_lag_steps_sum"] += max(0, lag)
                self.metrics["memory_packet_lag_steps_max"] = max(
                    int(self.metrics["memory_packet_lag_steps_max"]),
                    max(0, lag),
                )
                self.metrics["memory_packet_last_residual_shape"] = list(
                    residual.shape
                )
                self.metrics["memory_packet_last_gate_shape"] = (
                    ["alias:target"]
                    if gate_alias_target
                    else list(memory_gate.shape)
                )
            plasticity_coverage = tensors_cpu.get("plasticity_coverage")
            plasticity_confidence = tensors_cpu.get("plasticity_confidence")
            plasticity_budget = tensors_cpu.get("plasticity_budget")
            if (
                isinstance(plasticity_coverage, torch.Tensor)
                and isinstance(plasticity_confidence, torch.Tensor)
                and isinstance(plasticity_budget, torch.Tensor)
            ):
                cov_cpu = plasticity_coverage.reshape(-1).float()
                conf_cpu = plasticity_confidence.reshape(-1).float()
                budget_cpu = plasticity_budget.reshape(-1).float()
                cov = cov_cpu.to(
                    device=self.device, dtype=torch.float32
                )
                conf = conf_cpu.to(
                    device=self.device, dtype=torch.float32
                )
                budget = budget_cpu.to(
                    device=self.device, dtype=torch.float32
                )
                if not (cov.numel() == conf.numel() == budget.numel()):
                    self.metrics["last_drop_reason"] = (
                        "plasticity_packet_shape_mismatch"
                    )
                    continue
                payload["plasticity_coverage"] = cov
                payload["plasticity_confidence"] = conf
                payload["plasticity_budget"] = budget
                packet_bytes = int(
                    sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in (
                            plasticity_coverage,
                            plasticity_confidence,
                            plasticity_budget,
                        )
                    )
                )
                self.metrics["plasticity_packets_received"] += 1
                self.metrics["plasticity_packet_bytes_received"] += int(packet_bytes)
                self.metrics["plasticity_packet_bytes_received_max"] = max(
                    int(self.metrics["plasticity_packet_bytes_received_max"]),
                    int(packet_bytes),
                )
                self.metrics["plasticity_packet_last_shape"] = list(budget.shape)
                self.metrics["plasticity_budget_mean_received"] = float(
                    budget_cpu.mean().item()
                )
                self.metrics["plasticity_budget_max_received"] = float(
                    budget_cpu.max().item()
                )
                self.metrics["plasticity_confidence_mean_received"] = float(
                    conf_cpu.mean().item()
                )
                self.metrics["plasticity_coverage_abs_mean_received"] = float(
                    cov_cpu.abs().mean().item()
                )
                self.metrics["plasticity_lag_steps_sum"] += max(0, lag)
                self.metrics["plasticity_lag_steps_max"] = max(
                    int(self.metrics["plasticity_lag_steps_max"]),
                    max(0, lag),
                )
            inputs, targets = batch
            self.ready_results.append((payload, inputs, targets))
            self.metrics["payloads_received"] += 1
            self.metrics["mailbox_result_reads"] += 1
            self.metrics["last_received_request_step"] = request_step
            while len(self.ready_results) > int(self.max_local_batches):
                self.ready_results.popleft()
                self.metrics["ready_result_queue_drops"] += 1
                self.metrics["last_drop_reason"] = "ready_result_queue_overflow"
            self.metrics["ready_result_queue_depth"] = len(self.ready_results)
            self.metrics["ready_result_queue_max"] = max(
                int(self.metrics["ready_result_queue_max"]),
                len(self.ready_results),
            )
        return self._pop_ready_result(current_step=current_step)

    def _pop_ready_result(
        self,
        *,
        current_step: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor] | None:
        if not self.ready_results:
            self.metrics["ready_result_queue_depth"] = 0
            return None
        ready = self.ready_results.popleft()
        payload, _inputs, _targets = ready
        request_step = int(payload.get("step_id_int", -1))
        lag = int(current_step) - request_step
        self.metrics["payloads_used"] += 1
        self.metrics["last_used_request_step"] = request_step
        self.metrics["payload_lag_steps_sum"] += max(0, lag)
        self.metrics["payload_lag_steps_max"] = max(
            int(self.metrics["payload_lag_steps_max"]),
            max(0, lag),
        )
        self.metrics["ready_result_queue_depth"] = len(self.ready_results)
        return ready


def _resolve_crct_async_payload_dtype(
    requested: str,
    *,
    device: torch.device,
) -> torch.dtype:
    value = str(requested).strip().lower()
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(
        "crct_async_teacher_payload_dtype must be one of "
        "'auto', 'fp16', 'bf16', or 'fp32'; got "
        f"{requested!r}"
    )


def _collect_crct_teacher_payload(
    *,
    model: torch.nn.Module,
    cache: TransactionalWakeCache,
    scarcity_optimizer: CrctScarcityAwareMemoryOptimizer | None,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    rank: int,
    world_size: int,
    all_group: "dist.ProcessGroup | None",
    step: int,
    total_steps: int | None,
    tau: float,
    strength: float,
    w_max: float,
    alpha_max: float,
    memory_write_tokens: int,
    gradient_conflict_monitor: CrctGradientConflictMonitor | None = None,
) -> dict[str, torch.Tensor] | None:
    """Return this train rank's CRCT teacher payload.

    ``world_size < 4`` uses the design-doc synchronous fallback. At
    ``world_size >= 4`` the final rank is a memory coprocessor: train ranks
    synchronously fan input ids in, rank 3 scores one concatenated batch, then
    broadcasts dense per-rank payload tensors back. This is intentionally
    conservative and correctness-first; the async/drop-don't-queue transport
    can replace this helper without changing the train-step loss composition.
    """
    rank_ = int(rank)
    world_size_ = int(world_size)
    memory_rank = world_size_ - 1
    if world_size_ < 4 or all_group is None or not dist.is_initialized():
        return _crct_score_payload_inline(
            model=model,
            cache=cache,
            scarcity_optimizer=scarcity_optimizer,
            inputs=inputs,
            targets=targets,
            step=step,
            total_steps=total_steps,
            tau=tau,
            strength=strength,
            w_max=w_max,
            alpha_max=alpha_max,
            memory_write_tokens=int(memory_write_tokens),
            gradient_conflict_monitor=gradient_conflict_monitor,
            update_model_memory_after=False,
        )

    full_ids = _crct_full_input_ids(inputs, targets).contiguous()
    gathered: list[torch.Tensor] | None = None
    if rank_ == memory_rank:
        gathered = [torch.empty_like(full_ids) for _ in range(world_size_)]
    dist.gather(full_ids, gather_list=gathered, dst=memory_rank, group=all_group)

    n_train = world_size_ - 1
    bsz, seq_plus_one = int(full_ids.shape[0]), int(full_ids.shape[1])
    seq = seq_plus_one - 1
    payload_shape = (n_train, bsz, seq)
    if rank_ == memory_rank:
        assert gathered is not None
        train_full_ids = torch.cat(gathered[:n_train], dim=0)
        train_inputs = train_full_ids[:, :-1].to(dtype=inputs.dtype)
        train_targets = train_full_ids[:, 1:].to(dtype=targets.dtype)
        scored = _crct_score_payload_inline(
            model=model,
            cache=cache,
            scarcity_optimizer=scarcity_optimizer,
            inputs=train_inputs,
            targets=train_targets,
            step=step,
            total_steps=total_steps,
            tau=tau,
            strength=strength,
            w_max=w_max,
            alpha_max=alpha_max,
            memory_write_tokens=int(memory_write_tokens),
            gradient_conflict_monitor=gradient_conflict_monitor,
            update_model_memory_after=True,
        )
        target_all = scored["target"].reshape(payload_shape).contiguous()
        conf_all = scored["confidence"].reshape(payload_shape).contiguous()
        weight_all = scored["loss_weight"].reshape(payload_shape).contiguous()
        utility_all = scored["utility"].reshape(payload_shape).contiguous()
    else:
        target_all = torch.empty(payload_shape, device=inputs.device, dtype=torch.float32)
        conf_all = torch.empty_like(target_all)
        weight_all = torch.empty_like(target_all)
        utility_all = torch.empty_like(target_all)

    for tensor in (target_all, conf_all, weight_all, utility_all):
        dist.broadcast(tensor, src=memory_rank, group=all_group)

    if rank_ == memory_rank:
        return None
    return {
        "step_id": torch.tensor(int(step), device=inputs.device),
        "target": target_all[rank_].detach(),
        "confidence": conf_all[rank_].detach(),
        "loss_weight": weight_all[rank_].detach(),
        "utility": utility_all[rank_].detach(),
    }


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
# Async episodic write transport. Train ranks publish WRITE_EVENT wire records
# into per-rank POSIX shm SPSC rings and continue; the episodic rank drains on
# memory's own clock and routes records to (a) ``cache.append`` and (b) the
# controller query queue. The old per-step ``dist.gather`` was intentionally
# removed from the trunk path: write memory may drop or lag, but it may not
# stall the SSM train ranks.

# Default K_max for the per-rank emit tensor. At Phase 1 ``top_p ≈
# 1/(B*T)`` the typical valid-row count is 1-2 per step; K_max=16 leaves
# headroom for a 16x increase before we'd silently truncate.
_DEFAULT_EPISODIC_K_MAX = 16


def _event_ring_namespace(config: dict[str, Any] | None = None) -> str:
    """Return a short cross-rank namespace for POSIX shm event rings.

    Train/episodic ranks run in separate processes, so PID-scoped names are
    only useful for single-process tests. The async write path needs every rank
    to derive the same per-run namespace without an extra rendezvous. Prefer an
    explicit config/env id; fall back to torchrun's shared MASTER_PORT; finally
    fall back to the local PID for direct unit-test calls.
    """
    raw = None
    if config is not None:
        raw = config.get("episodic_event_ring_id")
    if raw is None:
        raw = (
            os.environ.get("CHAOSCONTROL_EPISODIC_RING_ID")
            or os.environ.get("TORCHELASTIC_RUN_ID")
            or os.environ.get("MASTER_PORT")
        )
    if raw is None:
        raw = f"pid{os.getpid()}"
    return hashlib.sha1(str(raw).encode("utf-8")).hexdigest()[:8]


def _event_ring_name(
    prefix: str,
    *,
    rank: int | None = None,
    namespace: str | None = None,
) -> str:
    pid = int(os.getpid())
    if namespace is not None:
        if rank is None:
            return f"{prefix}_{namespace}"
        return f"{prefix}_{namespace}_r{int(rank)}"
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


class _CompatDeque(deque):
    """Deque with list-compatible equality for older unit-test assertions."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, list):
            return list(self) == other
        return super().__eq__(other)


class _CudaWriteEventPublisher:
    """Publish completed CUDA-staged WriteEvent batches into a shm ring.

    The train rank owns CUDA production; this helper owns the CPU side of the
    stream. The hot path only calls ``submit`` after recording a CUDA event.
    The background thread polls event completion and calls the C++ raw-tensor
    ring API, so Python never materializes per-event dict/list payloads.
    """

    def __init__(
        self,
        *,
        ring: Any,
        k_max: int,
        event_size: int,
        depth: int,
        device: torch.device,
    ) -> None:
        self.ring = ring
        self.k_max = int(k_max)
        self.event_size = int(event_size)
        self.depth = max(1, int(depth))
        self.gpu_slots = [
            torch.empty(
                (self.k_max, self.event_size),
                device=device,
                dtype=torch.uint8,
            )
            for _ in range(self.depth)
        ]
        self.cpu_slots = [
            torch.empty(
                (self.k_max, self.event_size),
                device="cpu",
                dtype=torch.uint8,
                pin_memory=True,
            )
            for _ in range(self.depth)
        ]
        self.events = [torch.cuda.Event(blocking=False) for _ in range(self.depth)]
        self.free_slots: deque[int] = deque(range(self.depth))
        self.pending: deque[int] = deque()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.submitted_batches = 0
        self.pushed_events = 0
        self.skipped_events = 0
        self.dropped_events = 0
        self.dropped_batches = 0
        self.failed = False
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="cc-write-event-publisher",
            daemon=True,
        )
        self.thread.start()

    def acquire_slot(self) -> int | None:
        with self.lock:
            if self.failed:
                self.dropped_batches += 1
                return None
            if not self.free_slots:
                self.dropped_batches += 1
                return None
            return self.free_slots.popleft()

    def gpu_slot(self, slot: int) -> torch.Tensor:
        return self.gpu_slots[int(slot)]

    def submit(self, slot: int) -> None:
        slot_i = int(slot)
        with self.lock:
            if self.failed:
                self.free_slots.append(slot_i)
                self.dropped_batches += 1
                return
        self.cpu_slots[slot_i].copy_(self.gpu_slots[slot_i], non_blocking=True)
        self.events[slot_i].record(torch.cuda.current_stream())
        with self.lock:
            if self.failed:
                self.free_slots.append(slot_i)
                self.dropped_batches += 1
                return
            self.pending.append(slot_i)
            self.submitted_batches += 1
            self.cond.notify()

    def close(self, timeout: float = 2.0) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        self.thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set() or self.pending:
                slot: int | None = None
                with self.cond:
                    while not self.pending and not self.stop_event.is_set():
                        self.cond.wait()
                    if not self.pending:
                        continue
                    candidate = self.pending[0]
                    if bool(self.events[candidate].query()):
                        slot = self.pending.popleft()
                if slot is None:
                    _hotpath_yield()
                    continue
                try:
                    stats = self.ring.push_batch_tensor(self.cpu_slots[slot])
                    self.pushed_events += int(stats.get("pushed", 0))
                    self.skipped_events += int(stats.get("skipped", 0))
                    self.dropped_events += int(stats.get("dropped", 0))
                finally:
                    with self.lock:
                        self.free_slots.append(slot)
        except BaseException as exc:  # noqa: BLE001 - daemon failures must surface
            with self.lock:
                self.failed = True
                self.error = f"{type(exc).__name__}: {exc}"
                while self.pending:
                    self.free_slots.append(self.pending.popleft())


def _u8_wire(value: int, *, sentinel: int = 255) -> int:
    v = int(value)
    return v if 0 <= v <= 255 else int(sentinel)


def _u64_wire(value: int, *, sentinel: int = (1 << 64) - 1) -> int:
    v = int(value)
    return v if v >= 0 else int(sentinel)


def _cleanup_episodic_event_rings(owner: Any) -> None:
    publisher = getattr(owner, "cuda_write_event_publisher", None)
    if publisher is not None:
        try:
            publisher.close()
        except Exception:
            pass
        try:
            owner.write_ring_pushed = int(publisher.pushed_events)
            owner.write_ring_skipped = int(publisher.skipped_events)
            owner.write_ring_drops = int(publisher.dropped_events)
            owner.write_ring_drop_batches = int(publisher.dropped_batches)
            owner.write_ring_submitted_batches = int(publisher.submitted_batches)
            owner.write_ring_publisher_error = str(publisher.error or "")
        except Exception:
            pass
        try:
            owner.cuda_write_event_publisher = None
        except Exception:
            pass
    write_ring_names = getattr(owner, "write_ring_names", None)
    if write_ring_names:
        for name in write_ring_names:
            try:
                _ext.ShmRingWriteEvent.unlink(str(name))
            except Exception:
                pass
    if hasattr(owner, "write_rings"):
        try:
            owner.write_rings = []
        except Exception:
            pass
    if hasattr(owner, "write_ring_names"):
        try:
            owner.write_ring_names = []
        except Exception:
            pass
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


_ACTION_SPACE_HEADS = (
    "write_admission",
    "eviction",
    "replay_timing",
    "replay_budget",
    "write_budget",
    "temperature",
    "entropy_beta",
    "ema_alpha",
    "consolidation",
    "selection_rank",
)


def _action_trace_logger_from_config(
    config: dict[str, Any],
    *,
    rank: int | str,
) -> ActionSpaceTraceLogger | None:
    path_value = config.get("episodic_controller_action_trace_path")
    if not path_value:
        return None
    path = Path(str(path_value))
    rank_label = str(rank)
    if path.suffix:
        path = path.with_name(f"{path.stem}.rank{rank_label}{path.suffix}")
    else:
        path = path / f"action_space_rank{rank_label}.ndjson"
    return ActionSpaceTraceLogger(path)


def _head_config_float(
    config: dict[str, Any],
    *,
    head: str,
    field: str,
    default: float,
) -> float:
    table = config.get(f"episodic_controller_head_{field}")
    if isinstance(table, dict) and head in table:
        return float(table[head])
    return float(config.get(f"episodic_controller_{head}_{field}", default))


def _head_table_from_config(
    config: dict[str, Any],
    *,
    field: str,
) -> dict[str, float] | None:
    table = config.get(f"episodic_controller_head_{field}")
    out: dict[str, float] = {}
    if isinstance(table, dict):
        out.update({str(k): float(v) for k, v in table.items()})
    for head in _ACTION_SPACE_HEADS:
        key = f"episodic_controller_{head}_{field}"
        if key in config:
            out[head] = float(config[key])
    return out or None


def _build_action_space_from_config(
    config: dict[str, Any],
    *,
    trace_log: Any | None,
) -> ConstrainedActionSpace | None:
    if not bool(config.get("episodic_controller_action_space_enabled", False)):
        return None
    max_tags = config.get("episodic_controller_max_tags_per_query")
    head_readiness = {
        head: _head_config_float(
            config, head=head, field="readiness", default=0.0
        )
        for head in _ACTION_SPACE_HEADS
    }
    head_max_delta = {
        head: _head_config_float(
            config, head=head, field="max_delta", default=0.0
        )
        for head in _ACTION_SPACE_HEADS
    }
    selection_readiness = float(
        config.get(
            "episodic_controller_selection_readiness",
            head_readiness.get("selection_rank", 0.0),
        )
    )
    selection_max_delta = float(
        config.get(
            "episodic_controller_selection_max_delta",
            head_max_delta.get("selection_rank", 0.0),
        )
    )
    head_readiness["selection_rank"] = selection_readiness
    head_max_delta["selection_rank"] = selection_max_delta
    shared_ssm_enabled = bool(
        config.get("episodic_controller_shared_event_ssm_enabled", True)
    )
    return ConstrainedActionSpace(
        trace_only=bool(
            config.get("episodic_controller_action_space_trace_only", False)
        ),
        selection_readiness=selection_readiness,
        selection_max_delta=selection_max_delta,
        max_tags_per_query=(
            int(max_tags) if max_tags is not None else None
        ),
        head_readiness=head_readiness,
        head_max_delta=head_max_delta,
        event_ssm=(
            make_shared_event_ssm_from_config(config)
            if shared_ssm_enabled
            else None
        ),
        trace_log=trace_log,
        online_learning_rate=float(
            config.get("episodic_controller_action_learning_rate", 0.0)
        ),
        reward_clip=float(
            config.get("episodic_controller_action_reward_clip", 5.0)
        ),
    )


class EpisodicGpuEmit:
    """Train-rank handle threaded into ``_run_train_step``.

    Holds the pre-allocated ``[K_max, slot_dim]`` slot tensor (zeroed
    each step before pack), the slot-format dimensions (S, D), the
    fingerprint window, and the configured ``top_p`` (or NaN sentinel
    for "use ``1/(B*T)`` default"). The episodic rank does NOT receive
    this handle — WRITE_EVENT wire records move through per-train-rank
    ``ShmRingWriteEvent`` rings instead.

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
        "write_ring_pushed",
        "write_ring_skipped",
        "write_ring_drop_batches",
        "write_ring_submitted_batches",
        "write_ring_publisher_error",
        "cuda_write_event_unavailable_reason",
        "async_write_rings_enabled",
        "cuda_write_event_stream_enabled",
        "cuda_write_event_publisher",
        "cuda_write_event_positions",
        "cuda_write_event_candidate_base",
        "cuda_write_event_empty_pressure",
        "event_ring_namespace",
        "controller_action_space",
        "controller_action_trace_log",
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
        async_write_rings_enabled: bool = False,
        cuda_write_event_stream_enabled: bool = False,
        cuda_write_event_publisher: _CudaWriteEventPublisher | None = None,
        cuda_write_event_positions: torch.Tensor | None = None,
        cuda_write_event_candidate_base: torch.Tensor | None = None,
        cuda_write_event_empty_pressure: torch.Tensor | None = None,
        event_ring_namespace: str | None = None,
        controller_action_space: ConstrainedActionSpace | None = None,
        controller_action_trace_log: Any | None = None,
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
        self.write_ring_pushed = 0
        self.write_ring_skipped = 0
        self.write_ring_drop_batches = 0
        self.write_ring_submitted_batches = 0
        self.write_ring_publisher_error = ""
        self.cuda_write_event_unavailable_reason = ""
        self.async_write_rings_enabled = bool(async_write_rings_enabled)
        self.cuda_write_event_stream_enabled = bool(cuda_write_event_stream_enabled)
        self.cuda_write_event_publisher = cuda_write_event_publisher
        self.cuda_write_event_positions = cuda_write_event_positions
        self.cuda_write_event_candidate_base = cuda_write_event_candidate_base
        self.cuda_write_event_empty_pressure = cuda_write_event_empty_pressure
        self.event_ring_namespace = event_ring_namespace
        self.controller_action_space = controller_action_space
        self.controller_action_trace_log = controller_action_trace_log


def _create_episodic_emit(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    config: dict[str, Any],
) -> EpisodicGpuEmit | None:
    """Build the per-rank emit-tensor handle.

    Returns the handle on every rank when ``episodic_enabled=True`` and
    ``world_size > 1``. Train ranks carry a local slot tensor for the
    historical single-process pack tests and, in the production path, publish
    WRITE_EVENT records into async rings. The episodic rank returns a no-op
    emit handle and never participates in write transport.

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
    async_write_rings_enabled = bool(
        config.get("episodic_async_write_rings_enabled", True)
    )
    ring_namespace = _event_ring_namespace(config)
    # WRITE_EVENT producers live on train ranks. Async rings are the memory
    # write transport, not just an offline log: train ranks publish-or-drop and
    # keep going; the episodic side drains on its own clock. The legacy
    # `episodic_event_log_enabled` flag still allocates the same ring for tests
    # and wire-event logging, but ring creation no longer depends on it.
    write_ring = None
    write_ring_name = None
    cuda_write_event_publisher = None
    cuda_write_event_positions = None
    cuda_write_event_candidate_base = None
    cuda_write_event_empty_pressure = None
    cuda_write_event_stream_enabled = False
    cuda_unavailable_reason = ""
    controller_action_trace_log = None
    controller_action_space = None
    if (
        (async_write_rings_enabled or event_log_enabled)
        and int(rank) != int(world_size) - 1
    ):
        write_ring_name = _event_ring_name(
            "/cc_e_w",
            rank=int(rank),
            namespace=ring_namespace,
        )
        write_ring = _create_event_ring(_ext.ShmRingWriteEvent, write_ring_name)
        try:
            cuda_pack_available = bool(_ext.write_event_cuda_pack_available())
        except Exception as exc:
            cuda_pack_available = False
            cuda_unavailable_reason = f"availability_check_failed:{type(exc).__name__}"
        else:
            cuda_unavailable_reason = (
                "" if cuda_pack_available else "cuda_pack_extension_unavailable"
            )
        cuda_write_requested = bool(
            config.get("episodic_cuda_write_event_stream_enabled", True)
        )
        cuda_write_event_stream_enabled = (
            cuda_write_requested
            and bool(async_write_rings_enabled)
            and device.type == "cuda"
            and cuda_pack_available
        )
        if (
            cuda_write_requested
            and bool(async_write_rings_enabled)
            and device.type == "cuda"
            and not cuda_pack_available
        ):
            raise RuntimeError(
                "episodic CUDA WRITE_EVENT stream was requested on CUDA, "
                "but the _cpu_ssm_controller extension was built without "
                "write_event_pack.cu. Rebuild on the pod with CUDA available "
                "or set CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT=1 explicitly. "
                f"reason={cuda_unavailable_reason}"
            )
        if cuda_write_event_stream_enabled:
            constants = _ext.wire_event_constants()
            max_key_rep_dim = int(constants["KEY_REP_DIM_DEFAULT"])
            max_span_length = int(constants["SPAN_LENGTH_DEFAULT"])
            if int(key_rep_dim) > max_key_rep_dim:
                raise ValueError(
                    "episodic CUDA WRITE_EVENT pack requires "
                    f"episodic_key_rep_dim <= {max_key_rep_dim}; "
                    f"got {int(key_rep_dim)}"
                )
            if int(span_length) > max_span_length:
                raise ValueError(
                    "episodic CUDA WRITE_EVENT pack requires "
                    f"episodic_span_length <= {max_span_length}; "
                    f"got {int(span_length)}"
                )
            event_size = int(_ext.wire_event_sizes()["WriteEvent"])
            depth = int(config.get("episodic_cuda_write_event_stage_depth", 4))
            cuda_write_event_publisher = _CudaWriteEventPublisher(
                ring=write_ring,
                k_max=k_max,
                event_size=event_size,
                depth=depth,
                device=device,
            )
            cuda_write_event_positions = torch.empty(
                (k_max, 2),
                device=device,
                dtype=torch.long,
            )
            cuda_write_event_candidate_base = torch.empty(
                (1,),
                device=device,
                dtype=torch.long,
            )
            cuda_write_event_empty_pressure = torch.empty(
                (0,),
                device=device,
                dtype=torch.float32,
            )
    if (
        int(rank) != int(world_size) - 1
        and bool(config.get("episodic_controller_action_space_enabled", False))
    ):
        controller_action_trace_log = _action_trace_logger_from_config(
            config, rank=int(rank)
        )
        if controller_action_trace_log is None:
            controller_action_trace_log = []
        controller_action_space = _build_action_space_from_config(
            config,
            trace_log=controller_action_trace_log,
        )
    handle = EpisodicGpuEmit(
        slot_tensor=slot,
        k_max=k_max,
        span_length=span_length,
        key_rep_dim=key_rep_dim,
        fingerprint_window=fingerprint_window,
        top_p=top_p,
        write_ring=write_ring,
        write_ring_name=write_ring_name,
        async_write_rings_enabled=async_write_rings_enabled,
        cuda_write_event_stream_enabled=cuda_write_event_stream_enabled,
        cuda_write_event_publisher=cuda_write_event_publisher,
        cuda_write_event_positions=cuda_write_event_positions,
        cuda_write_event_candidate_base=cuda_write_event_candidate_base,
        cuda_write_event_empty_pressure=cuda_write_event_empty_pressure,
        event_ring_namespace=ring_namespace,
        controller_action_space=controller_action_space,
        controller_action_trace_log=controller_action_trace_log,
    )
    handle.cuda_write_event_unavailable_reason = str(cuda_unavailable_reason)
    return handle


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


def _valid_write_signal_window(
    write_signal: torch.Tensor,
    *,
    fingerprint_window: int,
    span_length: int,
) -> tuple[torch.Tensor, int]:
    """Return the boundary-valid column window and its original offset."""
    T = int(write_signal.size(1))
    W = max(0, int(fingerprint_window))
    S = max(1, int(span_length))
    first = min(W, T)
    stop = max(first, min(T, T - S + 1))
    if first == 0 and stop == T:
        return write_signal, 0
    return write_signal[:, first:stop], first


def _select_write_positions_with_action_space(
    *,
    action_space: ConstrainedActionSpace | None,
    write_signal: torch.Tensor,
    pressure_full: torch.Tensor | None,
    ce_full: torch.Tensor,
    top_p: float,
    k_max: int,
    current_step: int,
    write_bucket: int,
) -> torch.Tensor:
    n = int(write_signal.numel())
    if n <= 0:
        return torch.empty((0, 2), device=write_signal.device, dtype=torch.long)
    heuristic_positions = select_top_p_positions(write_signal, top_p=top_p)
    if action_space is None:
        return heuristic_positions
    write_head_active = action_space.active_head("write_admission")
    budget_readiness = action_space.readiness("write_budget")
    if not write_head_active and budget_readiness <= 0.0:
        return heuristic_positions

    fallback_budget = min(int(k_max), int(heuristic_positions.size(0)))
    reward_context = {
        "bucket": float(write_bucket),
        "cache_fill": 0.0,
        "pressure": 1.0 if pressure_full is None else float(
            pressure_full.detach().to(dtype=torch.float32).mean().item()
        ),
        "ce": float(ce_full.detach().to(dtype=torch.float32).mean().item()),
        "score": float(write_signal.detach().to(dtype=torch.float32).mean().item()),
    }
    requested_budget = fallback_budget
    if budget_readiness > 0.0:
        requested_budget = int(round(action_space.scalar_value(
            head_name="write_budget",
            gpu_step=int(current_step),
            fallback=float(fallback_budget),
            reward_context=reward_context,
        )))
    requested_budget = max(0, min(int(k_max), n, requested_budget))
    if requested_budget <= 0:
        return torch.empty((0, 2), device=write_signal.device, dtype=torch.long)
    if not write_head_active:
        flat = write_signal.reshape(-1)
        _, selected_flat = torch.topk(flat, k=requested_budget, largest=True)
        T = int(write_signal.shape[1])
        rows = (selected_flat // T).to(dtype=torch.int64)
        cols = (selected_flat % T).to(dtype=torch.int64)
        return torch.stack([rows, cols], dim=1)

    # Learned write admission must not pull the full B*T grid to CPU. It gets
    # a bounded heuristic candidate pool: enough slots for reranking/exploration
    # inside a fixed top-M stream, never an unbounded per-token Python list.
    pool_k = min(n, max(int(k_max), requested_budget * 4, 16))
    _, pool_flat_idx = torch.topk(
        write_signal.reshape(-1),
        k=pool_k,
        largest=True,
    )
    pool_scores = write_signal.reshape(-1).gather(0, pool_flat_idx)
    heuristic_scores = [
        float(x) for x in pool_scores.detach().to(dtype=torch.float32).cpu().tolist()
    ]
    effective_scores = action_space.effective_scores(
        heuristic_scores=heuristic_scores,
        learned_scores=None,
        gpu_step=int(current_step),
        head_name="write_admission",
        reward_context=reward_context,
    )
    selected = sorted(
        range(len(effective_scores)),
        key=lambda idx: (-float(effective_scores[idx]), idx),
    )[:requested_budget]
    T = int(write_signal.shape[1])
    selected_flat = pool_flat_idx[
        torch.tensor(selected, device=pool_flat_idx.device, dtype=torch.long)
    ]
    rows = [
        [int(idx) // T, int(idx) % T]
        for idx in selected_flat.detach().cpu().tolist()
    ]
    return torch.tensor(rows, device=write_signal.device, dtype=torch.long)


def _protection_score_for_write(
    *,
    action_space: ConstrainedActionSpace | None,
    current_step: int,
    write_bucket: int,
    pressure: float,
    ce: float,
    score: float,
) -> float:
    if action_space is None or action_space.readiness("eviction") <= 0.0:
        return 0.0
    reward_context = {
        "bucket": float(write_bucket),
        "pressure": float(pressure),
        "ce": float(ce),
        "score": float(score),
    }
    raw = None
    if action_space.event_ssm is not None:
        raw = action_space.event_ssm.observe(reward_context).get("eviction", 0.0)
    return float(action_space.scalar_value(
        head_name="eviction",
        raw_value=raw,
        gpu_step=int(current_step),
        fallback=0.0,
        reward_context=reward_context,
    ))


def _emit_write_events_cuda_stream(
    *,
    emit: EpisodicGpuEmit,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pressure_full: torch.Tensor | None,
    ce_full: torch.Tensor,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    rank: int,
    current_step: int,
    write_bucket: int,
) -> bool:
    """CUDA-resident WRITE_EVENT production into pinned CPU staging.

    Returns True when the CUDA stream path accepted the batch (or deliberately
    dropped it because all staging slots were busy). Returns False when callers
    should use the legacy Python pack path.
    """
    publisher = getattr(emit, "cuda_write_event_publisher", None)
    pos_stage = getattr(emit, "cuda_write_event_positions", None)
    base_stage = getattr(emit, "cuda_write_event_candidate_base", None)
    empty_pressure = getattr(emit, "cuda_write_event_empty_pressure", None)
    if (
        publisher is None
        or pos_stage is None
        or base_stage is None
        or empty_pressure is None
        or emit.write_ring is None
        or not inputs.is_cuda
        or not targets.is_cuda
        or not hidden.is_cuda
        or not ce_full.is_cuda
    ):
        return False
    slot = publisher.acquire_slot()
    if slot is None:
        # Publish-or-drop: a saturated staging ring must not stall the trunk.
        emit.write_ring_submitted_batches = int(publisher.submitted_batches)
        emit.write_ring_publisher_error = str(publisher.error or "")
        emit.write_ring_drop_batches += 1
        emit.write_ring_drops += int(emit.k_max)
        return True
    K = min(int(positions.size(0)), int(emit.k_max))
    pos_stage.fill_(-1)
    if K > 0:
        pos_stage[:K].copy_(positions[:K], non_blocking=True)
    base = _reserve_admission_trace_seq(int(K))
    base_stage.fill_(int(base))
    pressure_arg = (
        pressure_full.contiguous()
        if pressure_full is not None
        else empty_pressure
    )
    # The CUDA pack kernel requires int64 token tensors (TORCH_CHECK in
    # write_event_pack.cu). The trunk passes int32 in production; cast
    # here to match the legacy CPU path (`_emit_episodic_payloads_gpu`
    # converts to int64 at line ~2319). `.to(dtype=...)` is a no-op when
    # the tensor is already int64.
    _ext.pack_write_events_cuda_(
        publisher.gpu_slot(slot),
        inputs.to(dtype=torch.int64).contiguous(),
        targets.to(dtype=torch.int64).contiguous(),
        hidden.contiguous(),
        pressure_arg,
        ce_full.contiguous(),
        pos_stage,
        base_stage,
        int(current_step),
        int(rank),
        int(write_bucket),
        int(emit.fingerprint_window),
        int(emit.span_length),
        int(emit.key_rep_dim),
    )
    publisher.submit(slot)
    emit.write_ring_submitted_batches = int(publisher.submitted_batches)
    emit.write_ring_pushed = int(publisher.pushed_events)
    emit.write_ring_skipped = int(publisher.skipped_events)
    emit.write_ring_drops = int(publisher.dropped_events)
    emit.write_ring_drop_batches = int(publisher.dropped_batches)
    emit.write_ring_publisher_error = str(publisher.error or "")
    return True


def _emit_episodic_payloads_gpu(
    *,
    emit: EpisodicGpuEmit,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pressure: torch.Tensor | None,
    per_token_ce: torch.Tensor,
    hidden: torch.Tensor,
    rank: int,
    world_size: int,
    all_group: "dist.ProcessGroup | None",
    current_step: int = 0,
    write_bucket: int = 0,
) -> None:
    """Train-rank emit: pack selected writes and publish WRITE_EVENTs.

    The trunk SSM invariant is publish-or-drop, never wait. This helper keeps
    the historical ``slot_tensor`` populated for unit tests and local
    inspection, but the cross-rank transport is the per-rank
    ``ShmRingWriteEvent``. A full ring increments ``write_ring_drops`` and the
    train step continues. ``all_group`` is accepted for old call sites but is
    intentionally unused: memory writes must not introduce a train-step
    collective.
    """
    T = int(inputs.size(1))
    pressure_full = None if pressure is None else _right_pad_per_token_signal(pressure, T)
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
    if pressure_full is None:
        write_signal_full = ce_full.detach().to(dtype=torch.float32)
    else:
        write_signal_full = (
            pressure_full.detach().to(dtype=torch.float32)
            * ce_full.detach().to(dtype=torch.float32)
        )
    write_signal, col_offset = _valid_write_signal_window(
        write_signal_full,
        fingerprint_window=int(emit.fingerprint_window),
        span_length=int(emit.span_length),
    )
    positions = _select_write_positions_with_action_space(
        action_space=getattr(emit, "controller_action_space", None),
        write_signal=write_signal,
        pressure_full=pressure_full,
        ce_full=ce_full,
        top_p=top_p,
        k_max=int(emit.k_max),
        current_step=int(current_step),
        write_bucket=int(write_bucket),
    )
    if col_offset and int(positions.numel()) > 0:
        positions = positions.clone()
        positions[:, 1] += int(col_offset)
    K = int(positions.size(0))
    if K > emit.k_max:
        # Truncate silently — design doc says K > K_max is "config want
        # bigger than the slot can hold"; assert covers an upstream bug
        # where top_p would explode K beyond the configured cap.
        positions = positions[: emit.k_max]
        K = emit.k_max
    if _emit_write_events_cuda_stream(
        emit=emit,
        inputs=inputs,
        targets=targets,
        pressure_full=pressure_full,
        ce_full=ce_full,
        hidden=hidden.detach(),
        positions=positions,
        rank=int(rank),
        current_step=int(current_step),
        write_bucket=int(write_bucket),
    ):
        _ = (world_size, all_group)
        return
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
        pressure_value = 1.0 if pressure_full is None else float(pressure_full[b, t].item())
        ce_value = float(ce_full[b, t].item())
        protection_score = _protection_score_for_write(
            action_space=getattr(emit, "controller_action_space", None),
            current_step=int(current_step),
            write_bucket=int(write_bucket),
            pressure=pressure_value,
            ce=ce_value,
            score=float(write_signal_full[b, t].item()),
        )
        pack_payload(
            emit.slot_tensor[k],
            valid_mask=1.0,
            pressure=pressure_value,
            key_fp=key_fp,
            value_anchor_id=anchor,
            value_tok_ids=value_tok_ids,
            key_rep=key_rep,
            residual=residual,
            span_length=S,
            key_rep_dim=D,
            pre_write_ce=ce_value,
            protection_score=protection_score,
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
                    pressure_at_write=pressure_value,
                    pre_write_ce=ce_value,
                    write_bucket=int(write_bucket),
                ),
            )
    _ = (world_size, all_group)


class _EpisodicConsumerState:
    """Episodic-rank consumer state: cache + heartbeat + controller queues.

    Constructed by ``_attach_episodic_consumer`` once per runner init. The
    no-op shape (``cache=None``, ``heartbeat=[0]``,
    ``controller_query_queue=[]``, ``tagged_replay_queue=[]``) is what
    the train ranks and ``episodic_enabled=False`` runs see so those code
    paths stay cheap. The episodic rank owns lazy-attached write rings and the
    in-process Python ``controller_query_queue`` that the CPU controller reads
    from.

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
        "write_ring_names",
        "write_rings",
        "write_ring_attach_misses",
        "write_ring_events_drained",
        "write_ring_event_age_sum",
        "write_ring_event_age_max",
        "write_ring_max_drain_per_step",
        "write_ring_drain_errors",
        "write_ring_lock",
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
        "controller_action_trace_log",
        "controller_action_space",
    )

    def __init__(
        self,
        cache: EpisodicCache | None,
        heartbeat: list[int],
        controller_query_queue: Any,
        controller_query_enabled: bool = False,
        tagged_replay_queue: Any | None = None,
        episodic_event_log_enabled: bool = False,
        compute_replay_ce_pair: bool = False,
        controller_action_space_enabled: bool = False,
        write_ring_names: list[str] | None = None,
        write_ring_max_drain_per_step: int = 4096,
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
            _CompatDeque() if tagged_replay_queue is None else tagged_replay_queue
        )
        self.write_ring_names = list(write_ring_names or [])
        self.write_rings: list[Any | None] = [None for _ in self.write_ring_names]
        self.write_ring_attach_misses = 0
        self.write_ring_events_drained = 0
        self.write_ring_event_age_sum = 0
        self.write_ring_event_age_max = 0
        self.write_ring_max_drain_per_step = max(
            0, int(write_ring_max_drain_per_step)
        )
        self.write_ring_drain_errors = 0
        self.write_ring_lock = threading.RLock()
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
        # Learned action-space heads are opt-in. When disabled, do not
        # allocate even an empty trace list so the default controller path
        # remains allocation-identical to the pre-gate runner.
        self.controller_action_trace_log: list[dict[str, Any]] | None = (
            [] if controller_action_space_enabled else None
        )
        self.controller_action_space: ConstrainedActionSpace | None = None


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
    valid async write when controller querying is enabled.

    Async write transport: train ranks publish WRITE_EVENTs into stable
    per-rank shm rings. The episodic consumer derives the same ring names and
    attaches lazily, so there is no init-time barrier and no train-step gather.
    The ``all_group`` parameter is accepted for older call sites but unused.
    """
    no_op = _EpisodicConsumerState(
        cache=None,
        heartbeat=[0],
        controller_query_queue=_CompatDeque(),
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
    async_write_rings_enabled = bool(
        config.get("episodic_async_write_rings_enabled", True)
    )
    ring_namespace = _event_ring_namespace(config)
    write_ring_names = (
        [
            _event_ring_name(
                "/cc_e_w",
                rank=r,
                namespace=ring_namespace,
            )
            for r in range(max(0, int(world_size) - 1))
        ]
        if async_write_rings_enabled
        else []
    )
    return _EpisodicConsumerState(
        cache=cache,
        heartbeat=[0],
        controller_query_queue=_CompatDeque(),
        controller_query_enabled=bool(
            config.get("controller_query_enabled", False)
        ),
        episodic_event_log_enabled=bool(
            config.get("episodic_event_log_enabled", False)
        ),
        compute_replay_ce_pair=bool(
            config.get("episodic_compute_replay_ce_pair", False)
        ),
        controller_action_space_enabled=bool(
            config.get("episodic_controller_action_space_enabled", False)
        ),
        write_ring_names=write_ring_names,
        write_ring_max_drain_per_step=int(
            config.get("episodic_write_ring_max_drain_per_step", 4096)
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


class _EpisodicWriteDrainHandle:
    """Daemon handle for async WRITE_EVENT ring consumption."""

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


def _write_drain_main(
    *,
    consumer: _EpisodicConsumerState,
    stop_event: threading.Event,
    embedding_version_ref: list[int],
    controller_score_mode: str,
    controller_topk_k: int,
    heartbeat: list[int],
) -> None:
    current_step = 0
    while not stop_event.is_set():
        try:
            drained = _drain_episodic_write_rings(
                consumer=consumer,
                current_step=int(current_step),
                embedding_version=int(embedding_version_ref[0]),
                controller_score_mode=str(controller_score_mode),
                controller_topk_k=int(controller_topk_k),
            )
        except Exception:
            # The drain runs on a daemon thread; an uncaught exception
            # would silently kill it while train ranks keep publishing,
            # leaving the cache permanently empty for the rest of the
            # run. Bump the counter so telemetry surfaces the failure
            # and continue — the next iteration retries the drain on a
            # later batch of events.
            consumer.write_ring_drain_errors += 1
            print(
                "[runner_fast_path] episodic write-drain error "
                f"(step={current_step}, total_errors="
                f"{consumer.write_ring_drain_errors}):",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            drained = 0
        current_step += 1
        heartbeat[0] += 1
        if drained == 0:
            _hotpath_yield()


def _spawn_episodic_write_drain(
    *,
    consumer: _EpisodicConsumerState,
    is_episodic_rank: bool,
    episodic_enabled: bool,
    config: dict[str, Any],
    embedding_version_ref: list[int],
) -> _EpisodicWriteDrainHandle | None:
    if not episodic_enabled or not is_episodic_rank:
        return None
    if consumer.cache is None or not consumer.write_ring_names:
        return None
    stop_event = threading.Event()
    heartbeat = [0]
    thread = threading.Thread(
        target=_write_drain_main,
        kwargs={
            "consumer": consumer,
            "stop_event": stop_event,
            "embedding_version_ref": embedding_version_ref,
            "controller_score_mode": str(
                _controller_score_mode_from_config(config)
            ),
            "controller_topk_k": int(
                config.get("episodic_controller_topk_k", 16)
            ),
            "heartbeat": heartbeat,
        },
        daemon=True,
        name="episodic_write_drain",
    )
    thread.start()
    return _EpisodicWriteDrainHandle(
        thread=thread,
        stop_event=stop_event,
        heartbeat=heartbeat,
    )


def _shutdown_episodic_write_drain(
    handle: _EpisodicWriteDrainHandle | None,
    *,
    timeout_s: float = 2.0,
) -> None:
    if handle is None:
        return
    handle.stop_event.set()
    handle.thread.join(timeout=float(timeout_s))


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
    if int(event.get("outcome_status", -1)) != _REPLAY_STATUS_OK:
        return
    reward = float(event.get("reward_shaped", float("nan")))
    if not math.isfinite(reward):
        return
    action_space = getattr(consumer, "controller_action_space", None)
    if action_space is not None:
        action_space.apply_reward(
            key=int(event.get("replay_id", -1)),
            reward=reward,
            gpu_step=int(event.get("gpu_step", 0)),
            reward_context={
                "slot_id": float(event.get("slot_id", -1)),
                "bucket_baseline": float(event.get("bucket_baseline", 0.0)),
            },
        )
    bridge = getattr(consumer, "online_learning_bridge", None)
    if bridge is None:
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

    def _entry_value(field: str, fallback: Any) -> Any:
        value = _event_value(field, None)
        if value is not None:
            return value
        return entry.get(field, fallback)

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
        "arm": str(_entry_value("arm", "")),
        "chosen_idx": int(
            _entry_value("chosen_idx", entry.get("simplex_chosen_idx", -1))
        ),
        "p_chosen": float(
            _entry_value("p_chosen", entry.get("simplex_p_chosen", float("nan")))
        ),
        "p_behavior": _entry_value(
            "p_behavior", entry.get("simplex_probabilities", [])
        ),
        "entropy": float(_entry_value("entropy", float("nan"))),
        "gerber_weight": float(_entry_value("gerber_weight", float("nan"))),
        "advantage_raw": float(_entry_value("advantage_raw", float("nan"))),
        "advantage_corrected": float(
            _entry_value("advantage_corrected", float("nan"))
        ),
        "lambda_hxh": float(_entry_value("lambda_hxh", float("nan"))),
        "feature_manifest_hash": str(_entry_value("feature_manifest_hash", "")),
        "candidate_slot_ids": _entry_value(
            "candidate_slot_ids", entry.get("simplex_candidate_slot_ids", [])
        ),
        "candidate_scores": _entry_value(
            "candidate_scores", entry.get("simplex_candidate_scores", [])
        ),
        "logits": _entry_value("logits", entry.get("simplex_logits", [])),
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
    # Optional sharper-policy override: the BC-pretrained simplex_v1 CSWG
    # produces near-uniform initial p (max 0.0667 vs uniform 0.0625 —
    # only 7% above per the 2026-04-27 v2 trace inspection). With
    # near-uniform sampling REINFORCE gradients average to zero across
    # events and the policy can't bootstrap. Setting a smaller initial
    # temperature than what the CSWG carries (default 1.0) sharpens the
    # softmax so the sampling policy is non-uniform from step 0, which
    # gives REINFORCE a non-noise gradient direction to lock onto.
    # Default None preserves the CSWG-loaded temperature; set e.g. 0.2
    # for a 5x sharper policy.
    initial_T_override = config.get("episodic_controller_initial_temperature")
    if initial_T_override is not None:
        learner.set_temperature(float(initial_T_override))
    # NDJSON per-replay-event trace. Empty string disables. Configured
    # AFTER initialize so a misconfigured weights path can't leak an
    # opened trace file. Mirrors the entropy_beta plumbing one-for-one.
    trace_path = str(
        config.get("episodic_controller_simplex_trace_path", "") or ""
    )
    if trace_path:
        learner.set_simplex_trace_path(trace_path)
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


def _build_controller_action_space(
    *,
    consumer: _EpisodicConsumerState,
    config: dict[str, Any],
) -> ConstrainedActionSpace | None:
    """Build the optional constrained action-space gate for the controller.

    Disabled means ``None``: no object allocation on the controller hot path and
    no trace list on the consumer. Enabled means learned controller logits are
    interpreted as bounded residuals over the heuristic ranker; readiness=0 is
    an explicit traceable-but-dormant stage.
    """
    if not bool(config.get("episodic_controller_action_space_enabled", False)):
        return None
    if consumer.controller_action_trace_log is None:
        consumer.controller_action_trace_log = (
            _action_trace_logger_from_config(config, rank="controller")
        )
    if consumer.controller_action_trace_log is None:
        consumer.controller_action_trace_log = []
    action_space = _build_action_space_from_config(
        config,
        trace_log=consumer.controller_action_trace_log,
    )
    consumer.controller_action_space = action_space
    return action_space


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
    action_space = _build_controller_action_space(
        consumer=consumer,
        config=config,
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
            # Shared cache+queue lock with the async write-drain thread.
            # Queue ops are GIL-atomic, but cache.append and query_topk must
            # not interleave once writes are drained off the train-step path.
            "queue_lock": consumer.write_ring_lock,
            "stop_event": stop_event,
            "k": k,
            "score_mode": score_mode,
            "controller_runtime": controller_runtime_for_thread,
            "action_recorder": action_recorder,
            "simplex_selection_mode": simplex_selection_mode,
            "simplex_generator": simplex_generator,
            "action_space": action_space,
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
                protection_score=float(unpacked.get("protection_score", 0.0)),
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
            pre_query_ce_value = float(
                unpacked.get("pre_write_ce", float("nan"))
            )
            if not math.isfinite(pre_query_ce_value):
                pre_query_ce_value = float(pre_query_ce)
            query_event_id = _emit_query_event(
                consumer=consumer,
                source_rank=int(r),
                gpu_step=int(current_step),
                query_residual=unpacked["residual"],
                pressure=float(unpacked["pressure"]),
                pre_query_ce=pre_query_ce_value,
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


def _fp16_u16_wire_to_tensor(values: Any, *, width: int) -> torch.Tensor:
    arr = np.asarray([int(x) for x in list(values)[:int(width)]], dtype=np.uint16)
    if int(arr.size) < int(width):
        arr = np.pad(arr, (0, int(width) - int(arr.size)), constant_values=0)
    return torch.from_numpy(arr.view(np.float16).astype(np.float32, copy=True))


def _drain_episodic_write_rings(
    *,
    consumer: _EpisodicConsumerState,
    current_step: int,
    embedding_version: int,
    controller_score_mode: str = "cosine_utility_weighted",
    controller_topk_k: int = 16,
) -> int:
    """Drain async per-train-rank WRITE_EVENT rings into cache + query queue.

    This is the consumer half of the supercar invariant: memory consumes what
    the trunk published, on memory's own clock. Missing rings, empty rings, and
    backlog are observable state, never a reason for a train-rank collective.
    """
    if consumer.cache is None or not consumer.write_ring_names:
        return 0
    cache = consumer.cache
    drained = 0
    max_events = int(consumer.write_ring_max_drain_per_step)
    with consumer.write_ring_lock:
        for idx, name in enumerate(consumer.write_ring_names):
            ring = consumer.write_rings[idx]
            if ring is None:
                try:
                    ring = _ext.ShmRingWriteEvent.attach(str(name))
                except Exception:
                    consumer.write_ring_attach_misses += 1
                    continue
                consumer.write_rings[idx] = ring
            while max_events <= 0 or drained < max_events:
                event = ring.pop()
                if event is None:
                    break
                source_rank = int(event["source_rank"])
                write_step = int(event["gpu_step"])
                write_bucket = int(event["write_bucket"])
                pressure = float(event["pressure_at_write"])
                pre_write_ce = float(event["pre_write_ce"])
                key_rep = _fp16_u16_wire_to_tensor(
                    event["key_rep"],
                    width=int(cache.key_rep_dim),
                )
                value_tok_ids = torch.tensor(
                    [int(x) for x in list(event["value_tok_ids"])[:cache.span_length]],
                    dtype=torch.int64,
                )
                if int(value_tok_ids.numel()) < int(cache.span_length):
                    value_tok_ids = torch.cat(
                        [
                            value_tok_ids,
                            torch.zeros(
                                int(cache.span_length) - int(value_tok_ids.numel()),
                                dtype=torch.int64,
                            ),
                        ]
                    )
                protection_score = _protection_score_for_write(
                    action_space=getattr(consumer, "controller_action_space", None),
                    current_step=int(write_step),
                    write_bucket=int(write_bucket),
                    pressure=pressure,
                    ce=pre_write_ce,
                    score=pressure * pre_write_ce,
                )
                appended_slot = cache.append(
                    key_fp=int(event["key_fp"]),
                    key_rep=key_rep,
                    value_tok_ids=value_tok_ids,
                    value_anchor_id=int(event["value_anchor_id"]),
                    current_step=int(write_step),
                    embedding_version=int(embedding_version),
                    pressure_at_write=pressure,
                    source_write_id=int(event["candidate_id"]),
                    write_bucket=int(write_bucket),
                    displacing_candidate_id=int(event["candidate_id"]),
                    protection_score=float(protection_score),
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
                            query_residual=key_rep,
                            score_mode=str(controller_score_mode),
                            k=int(controller_topk_k),
                        )
                    )
                query_event_id = _emit_query_event(
                    consumer=consumer,
                    source_rank=source_rank,
                    gpu_step=write_step,
                    query_residual=key_rep,
                    pressure=pressure,
                    pre_query_ce=pre_write_ce,
                    bucket=write_bucket,
                    candidate_slot_ids=candidate_slot_ids,
                    candidate_cosines=candidate_cosines,
                )
                if consumer.controller_query_enabled:
                    consumer.controller_query_queue.append({
                        "step": int(write_step),
                        "rank": int(source_rank),
                        "k": 0,
                        "query_event_id": int(
                            query_event_id
                            if query_event_id is not None
                            else int(event["candidate_id"])
                        ),
                        "source_write_id": int(event["candidate_id"]),
                        "write_bucket": int(write_bucket),
                        "slot": int(appended_slot),
                        "pressure": pressure,
                        "residual": key_rep,
                    })
                age = max(0, int(current_step) - int(write_step))
                consumer.write_ring_event_age_sum += int(age)
                consumer.write_ring_event_age_max = max(
                    int(consumer.write_ring_event_age_max),
                    int(age),
                )
                drained += 1
                consumer.write_ring_events_drained += 1
            if max_events > 0 and drained >= max_events:
                break
    return int(drained)


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


def _order_replay_queue_with_action_space(
    *,
    consumer: "_EpisodicConsumerState | None",
    tagged: list[dict[str, Any]],
    current_step: int,
    max_replays_per_step: int,
) -> tuple[int, bool]:
    action_space = getattr(consumer, "controller_action_space", None)
    if action_space is None or not tagged:
        return int(max_replays_per_step), False
    timing_active = action_space.active_head("replay_timing")
    budget_readiness = action_space.readiness("replay_budget")
    if not timing_active and budget_readiness <= 0.0:
        return int(max_replays_per_step), False

    tagged_list = list(tagged)
    scores = [
        float(entry.get("action_space_score", entry.get("score", 0.0)))
        for entry in tagged_list
    ]
    reward_context = {
        "queue_depth": float(len(tagged)),
        "score": float(sum(scores) / max(1, len(scores))),
    }
    fallback_budget = (
        int(max_replays_per_step)
        if int(max_replays_per_step) > 0
        else len(tagged)
    )
    budget = fallback_budget
    if budget_readiness > 0.0:
        budget = int(round(action_space.scalar_value(
            head_name="replay_budget",
            gpu_step=int(current_step),
            fallback=float(fallback_budget),
            reward_context=reward_context,
        )))
    budget = max(0, min(len(tagged_list), int(budget)))
    if timing_active:
        effective_scores = action_space.effective_scores(
            heuristic_scores=scores,
            learned_scores=None,
            gpu_step=int(current_step),
            head_name="replay_timing",
            reward_context=reward_context,
        )
    else:
        effective_scores = scores
    selected = sorted(
        range(len(tagged_list)),
        key=lambda idx: (-float(effective_scores[idx]), idx),
    )[:budget]
    credit_heads: list[str] = []
    if timing_active:
        credit_heads.append("replay_timing")
    if budget_readiness > 0.0:
        credit_heads.append("replay_budget")
    for idx in selected:
        entry = tagged_list[idx]
        replay_id = int(
            entry.get(
                "replay_id",
                (int(entry.get("step", current_step)) << 32)
                ^ int(entry.get("slot", idx)),
            )
        )
        action_space.record_credit_assignment(
            key=replay_id,
            head_names=credit_heads,
            gpu_step=int(current_step),
            reward_context={
                **reward_context,
                "slot": float(entry.get("slot", -1)),
                "replay_id": float(replay_id),
            },
        )
    selected_set = set(selected)
    reordered = (
        [tagged_list[idx] for idx in selected]
        + [entry for idx, entry in enumerate(tagged_list) if idx not in selected_set]
    )
    if hasattr(tagged, "clear") and hasattr(tagged, "extend"):
        tagged.clear()
        tagged.extend(reordered)
    else:
        tagged[:] = reordered
    return int(budget), True


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
    max_replays, learned_replay_cap = _order_replay_queue_with_action_space(
        consumer=consumer,
        tagged=tagged,
        current_step=int(current_step),
        max_replays_per_step=int(max_replays_per_step),
    )
    # Drain destructively. The controller is the producer; once we
    # consume an entry, it's done — the controller will push fresh
    # entries on the next cycle.
    while tagged and (
        (not learned_replay_cap and max_replays <= 0)
        or replayed < max_replays
    ):
        pop_left = getattr(tagged, "popleft", None)
        entry = pop_left() if pop_left is not None else tagged.pop(0)
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
    grad_group: "dist.ProcessGroup | None" = None,
    grad_world_size: int | None = None,
    memory_rank_joins_grad: bool = True,
    episodic_emit: EpisodicGpuEmit | None = None,
    episodic_consumer: "_EpisodicConsumerState | None" = None,
    episodic_replay_logger: DiagnosticsLogger | None = None,
    episodic_replay_max_replays_per_step: int = 0,
    episodic_controller_score_mode: str = "cosine_utility_weighted",
    episodic_controller_topk_k: int = 16,
    current_step: int = 0,
    embedding_version: int = 0,
    crct_enabled: bool = False,
    crct_payload: dict[str, torch.Tensor] | None = None,
    crct_memory_write_tokens_per_step: int = 256,
) -> torch.Tensor:
    _reject_unsupported_fast_step(model, crct_enabled=bool(crct_enabled))
    # ------------------------------------------------------------------
    # Episodic rank: skip main, optionally replay, then all-reduce. WRITE_EVENT
    # ingestion happens on the background drain thread, not in this step body.
    # ------------------------------------------------------------------
    if is_episodic_rank:
        device = inputs.device
        # Async WRITE_EVENT rings replaced the per-step dist.gather. The
        # episodic step body no longer receives train-rank writes here; a
        # daemon write-drain thread consumes the rings and mutates the cache on
        # memory's own clock. This branch stays in the collective path only for
        # gradient/replay semantics, never for write admission.
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
            write_lock = getattr(
                episodic_consumer,
                "write_ring_lock",
                contextlib.nullcontext(),
            )
            with write_lock:
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
        if (
            ddp_active
            and world_size > 1
            and all_group is not None
            and bool(memory_rank_joins_grad)
        ):
            grad_group_eff = grad_group if grad_group is not None else all_group
            grad_world = (
                int(grad_world_size)
                if grad_world_size is not None
                else int(world_size)
            )
            n_train = grad_world - 1
            if n_train < 1:
                raise ValueError(
                    "episodic rank topology requires world_size >= 2 "
                    f"(1 train + 1 episodic), got world_size={world_size}"
                )
            allreduce_grads(
                model,
                grad_world,
                group=grad_group_eff,
                op=dist.ReduceOp.SUM,
                materialize_zeros=True,
            )
        return torch.zeros((), device=device, dtype=torch.float32)
    # ------------------------------------------------------------------
    # Train rank: forward + backward, publish WRITE_EVENT records, all-reduce.
    # ------------------------------------------------------------------
    # Per-token CE is captured only when the episodic producer is wired —
    # the standard fast path stays on ``fused_lm_head_backward`` (which
    # discards per-token CE) so the cost of the ``_with_ce`` variant is
    # not paid by ``episodic_enabled=False`` runs.
    per_token_ce_for_episodic: torch.Tensor | None = None
    with autocast_context(precision, device_type=inputs.device.type):
        packet_residual = (
            crct_payload.get("memory_residual")
            if crct_enabled and crct_payload is not None
            else None
        )
        packet_gate = (
            crct_payload.get("memory_gate")
            if crct_enabled and crct_payload is not None
            else None
        )
        if compile_full_path:
            if (
                crct_enabled
                and isinstance(packet_residual, torch.Tensor)
                and isinstance(packet_gate, torch.Tensor)
            ):
                hidden = _compiled_packet_step_fn()(
                    model,
                    inputs,
                    packet_residual,
                    packet_gate,
                )
            else:
                hidden = _compiled_step_fn()(model, inputs)
        elif crct_enabled:
            # GPU3/CPU own the episodic memory decision plane. Train ranks
            # consume only the latest-complete residual packet through a fixed
            # pre-recurrence lane; when the teacher lags, ``packet`` is a
            # bit-identical zero-residual no-op. No local slot reads and no
            # token-wise controller MLP run on the trunk hot path.
            hidden = model.encode(
                inputs,
                memory_mode="packet",
                episodic_residual=packet_residual,
                episodic_gate=packet_gate,
            )
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
        crct_token_weight = (
            crct_payload["loss_weight"].to(device=hidden.device, dtype=torch.float32)
            if crct_enabled and crct_payload is not None
            else None
        )
        if crct_enabled:
            mode = str(lm_head_backward_mode).strip().lower()
            loss_backward_done = False
            if mode in _FUSED_LM_HEAD_MODES:
                backend_name = fused_lm_head_backend_for_mode(mode)
                if crct_token_weight is None:
                    loss = fused_lm_head_backward(
                        hidden=hidden,
                        final_norm=model.final_norm,
                        lm_head=model.lm_head,
                        targets=targets,
                        backend=backend_name,
                        tile_size=int(lm_head_tile_size),
                    )
                    loss_backward_done = True
                else:
                    loss, per_token_ce_for_episodic = fused_lm_head_weighted_loss_with_ce(
                        hidden=hidden,
                        final_norm=model.final_norm,
                        lm_head=model.lm_head,
                        targets=targets,
                        token_weight=crct_token_weight,
                        backend=backend_name,
                        tile_size=int(lm_head_tile_size),
                    )
            else:
                logits = model.lm_head(model.final_norm(hidden))
                vocab = logits.size(-1)
                ce = F.cross_entropy(
                    logits.reshape(-1, vocab).float(),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
                per_token_ce_for_episodic = ce.detach()
                if crct_token_weight is None:
                    loss = ce.mean()
                else:
                    loss = (
                        ce * crct_token_weight.reshape_as(ce)
                    ).sum() / crct_token_weight.sum().clamp_min(1.0)
            if crct_payload is None:
                if not loss_backward_done:
                    loss.backward()
            else:
                loss.backward()
        elif lm_head_backward_mode == "single":
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
            grad_group_eff = grad_group if grad_group is not None else all_group
            if grad_group_eff is not None:
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
                if bool(memory_rank_joins_grad):
                    n_train = world_size - 1
                    materialize_zeros = True
                else:
                    n_train = (
                        int(grad_world_size)
                        if grad_world_size is not None
                        else world_size - 1
                    )
                    # Rank0-only CRCT labels/packets mean rank0 may see
                    # payload-conditioned loss weights while ranks 1/2 fail
                    # open. Materialize zeros inside the train subgroup so
                    # the flattened grad buffer is identical across train
                    # ranks without asking the memory rank to join the trunk
                    # collective.
                    materialize_zeros = bool(crct_enabled)
                if n_train < 1:
                    raise ValueError(
                        "episodic rank topology requires world_size >= 2 "
                        f"(1 train + 1 episodic), got world_size={world_size}"
                    )
                inv_n_train = 1.0 / float(n_train)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(inv_n_train)
                # Publish WRITE_EVENT records before the SUM all-reduce. This
                # is a local ring push only: no memory-side collective may sit
                # in the train-rank critical path.
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
                    _emit_episodic_payloads_gpu(
                        emit=episodic_emit,
                        inputs=inputs,
                        targets=targets,
                        pressure=None,
                        per_token_ce=per_token_ce_bt,
                        hidden=hidden,
                        rank=int(rank),
                        world_size=int(world_size),
                        all_group=all_group,
                        current_step=int(current_step),
                    )
                allreduce_grads(
                    model,
                    n_train,
                    group=grad_group_eff,
                    op=dist.ReduceOp.SUM,
                    materialize_zeros=materialize_zeros,
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
        _emit_episodic_payloads_gpu(
            emit=episodic_emit,
            inputs=inputs,
            targets=targets,
            pressure=None,
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
        # Keep full-step graph capture away from DDP/NCCL collectives. The
        # episodic memory producer has its own per-rank CUDA staging stream;
        # it must not make the trunk SSM a stricter lockstep machine.
        reasons.append("ddp_not_supported")
    if activation_checkpoint:
        reasons.append("activation_checkpoint_not_supported")
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
            memory_rank_wall_stop_check = (
                crct_enabled
                and bool(is_episodic_rank)
                and not bool(episodic_enabled)
                and not bool(memory_rank_joins_grad)
            )
            if (
                memory_rank_wall_stop_check
                or steps == 0
                or steps % check_interval == 0
                or max_steps_reached
            ):
                elapsed = time.perf_counter() - start_time
                if memory_rank_wall_stop_check:
                    local_stop = _should_stop_memory_rank_loop(
                        steps=steps,
                        elapsed_s=elapsed,
                        budget_seconds=budget_seconds,
                        stop_margin_seconds=stop_margin_seconds,
                        max_steps=max_steps,
                    )
                    active_stop_transport = (
                        crct_teacher_transport
                        if bool(is_crct_packet_rank)
                        else crct_maintenance_transport
                    )
                    if _should_defer_memory_rank_stop_for_shutdown(
                        local_stop=bool(local_stop),
                        elapsed_s=elapsed,
                        budget_seconds=budget_seconds,
                        stop_margin_seconds=stop_margin_seconds,
                        transport_mode=str(crct_teacher_transport_mode),
                        active_transport=active_stop_transport,
                    ):
                        local_stop = False
                else:
                    local_stop = should_stop_training_loop(
                        steps=steps,
                        elapsed_s=elapsed,
                        budget_seconds=budget_seconds,
                        stop_margin_seconds=stop_margin_seconds,
                        max_steps=max_steps,
                    )
                stop_ddp_active = bool(ddp_active)
                stop_group_eff = stop_group
                if crct_enabled and not episodic_enabled and is_episodic_rank:
                    # Rank 3 is deliberately off the trunk clock for CRCT.
                    # It stops from its own wall timer; train ranks sync
                    # stop decisions with each other through ``train_group``.
                    stop_ddp_active = False
                    stop_group_eff = None
                if should_stop_now(
                    local_stop,
                    device,
                    stop_ddp_active,
                    group=stop_group_eff,
                ):
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
        _control_barrier(
            group=object_group or all_group,
            label="cuda_graph_train_teardown",
        )

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
    episodic_async_write_rings_enabled: bool = True,
    episodic_cuda_write_event_stream_enabled: bool = True,
    episodic_cuda_write_event_stage_depth: int = 4,
    episodic_event_ring_id: str | None = None,
    episodic_write_ring_max_drain_per_step: int = 4096,
    episodic_compute_replay_ce_pair: bool = False,
    episodic_controller_score_mode: str = "cosine_utility_weighted",
    episodic_controller_topk_k: int = 16,
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
    episodic_controller_simplex_trace_path: str = "",
    episodic_controller_history_entries: int = 64,
    episodic_controller_action_space_enabled: bool = False,
    episodic_controller_action_space_trace_only: bool = False,
    episodic_controller_selection_readiness: float = 0.0,
    episodic_controller_selection_max_delta: float = 0.0,
    episodic_controller_max_tags_per_query: int | None = None,
    episodic_controller_action_trace_path: str | None = None,
    episodic_controller_shared_event_ssm_enabled: bool = True,
    episodic_controller_ssm_hidden_dim: int = 16,
    episodic_controller_ssm_seed: int = 0,
    episodic_controller_ssm_decay: float = 0.95,
    episodic_controller_ssm_input_scale: float = 0.05,
    episodic_controller_ssm_head_scale: float = 0.05,
    episodic_controller_head_readiness: dict[str, float] | None = None,
    episodic_controller_head_max_delta: dict[str, float] | None = None,
    episodic_controller_action_learning_rate: float = 0.0,
    episodic_controller_action_reward_clip: float = 5.0,
    episodic_replay_max_replays_per_step: int = 0,
    crct_enabled: bool = False,
    crct_lm_weight_alpha_max: float = 0.15,
    crct_lm_weight_strength: float = 0.10,
    crct_lm_weight_w_max: float = 1.20,
    crct_lm_weight_tau: float = 0.10,
    crct_target_read_rate: float = 0.25,
    crct_target_write_rate: float = 0.10,
    crct_dual_lr: float = 0.01,
    crct_ema_beta: float = 0.95,
    crct_max_price: float = 0.50,
    crct_plasticity_budget_strength: float = 0.25,
    crct_memory_write_tokens_per_step: int = 256,
    crct_async_teacher_transport: bool = True,
    crct_async_teacher_transport_backend: str = "collective",
    crct_teacher_mailbox_dir: str = "",
    crct_async_teacher_pending_batches: int = 64,
    crct_async_teacher_max_lag_steps: int = 128,
    crct_async_teacher_payload_dtype: str = "auto",
    crct_teacher_score_interval_steps: int = 1,
    crct_score_stage_timing_enabled: bool = False,
    crct_teacher_param_sync_interval_steps: int | None = None,
    crct_gradient_conflict_enabled: bool = False,
    crct_gradient_conflict_ema_beta: float = 0.95,
    crct_gradient_conflict_catastrophic_threshold: float = -0.90,
    crct_gradient_conflict_soft_gate_strength: float = 0.0,
    crct_gradient_conflict_soft_gate_floor: float = 0.05,
    crct_gradient_conflict_trace_path: str = "",
    crct_gradient_conflict_trace_stride: int = 1,
    crct_gradient_conflict_trace_max_rows: int = 0,
    crct_gradient_conflict_trace_flush_rows: int = 256,
    replay_eviction_enabled: bool = False,
    replay_eviction_mode: str = "active",
    replay_eviction_memory_streams: int = 8,
    replay_eviction_threshold: float = 0.01,
    replay_eviction_ema_beta: float = 0.9,
    replay_eviction_min_age_steps: int = 128,
    replay_eviction_max_seconds: float = 0.5,
    replay_eviction_trace_path: str = "",
    replay_eviction_trace_max_rows: int = 0,
    replay_eviction_trace_flush_rows: int = 256,
    replay_eviction_probe_chunk_size: int = 16,
    replay_eviction_scoring_mode: str = "proxy",
    replay_eviction_oracle_confirm_top_k: int = 32,
    replay_eviction_oracle_variant_chunk_size: int = 1,
    replay_eviction_drift_threshold: float = 0.3,
    replay_eviction_repr_drift_threshold: float = 0.2,
    replay_eviction_refresh_lr: float = 0.1,
    replay_eviction_refresh_candidate_count: int = 16,
    replay_eviction_refresh_proposal_rank: int = 8,
    replay_eviction_refresh_proposal_noise_scale: float = 0.04,
    replay_eviction_refresh_proposal_momentum: float = 0.9,
    replay_eviction_refresh_candidate_variant_chunk_size: int = 16,
    replay_eviction_refresh_proposal_seed: int = 1729,
    replay_eviction_controller_state_dim: int = 32,
    replay_eviction_controller_rank: int = 8,
    replay_eviction_controller_dt: float = 1.0,
    replay_eviction_controller_gamma: float = 0.08,
    replay_eviction_controller_target_log_sv: float = -0.05,
    replay_eviction_controller_max_state_norm: float = 8.0,
    replay_eviction_controller_perturbation_scale: float = 0.25,
    replay_eviction_controller_feedback_lr: float = 0.05,
    replay_eviction_quarantine_threshold: float = -0.01,
    replay_eviction_max_quarantined: int = 8,
    replay_eviction_distill_peak_threshold: float = 0.04,
    replay_eviction_peak_preserve_utility_threshold: float = 0.20,
    replay_eviction_peak_preserve_sharpness_threshold: float = 0.20,
    replay_eviction_useful_threshold: float = 0.005,
    replay_eviction_probe_buffer_size: int = 32,
    replay_eviction_frame_ttl_steps: int = 256,
    replay_eviction_slot_work_chunk_size: int = 64,
    replay_eviction_action_agreement_count: int = 2,
    replay_eviction_commit_policy: str = "learned",
    replay_eviction_commit_online_lr: float = 0.05,
    replay_eviction_commit_temperature: float = 0.75,
    replay_eviction_arm_runtime_enabled: bool = False,
    replay_eviction_arm_runtime_namespace: str = "",
    replay_eviction_evidence_engine_enabled: bool = False,
    replay_eviction_evidence_engine_d_model: int = 384,
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
    if crct_enabled and str(train_sampling_mode).strip().lower() in {
        "sequential_epoch",
        "shuffled_epoch",
    }:
        raise ValueError(
            "crct_enabled=True is incompatible with "
            f"train_sampling_mode={train_sampling_mode!r} in the current "
            "3+1 topology: the final rank is a memory coprocessor and would "
            "drop its shard of the epoch. Set train_sampling_mode='random' "
            "for CRCT runs until the start sampler shards over train ranks "
            "only."
        )
    grad_allreduce_mode_ = str(grad_allreduce_mode).strip().lower()
    if grad_allreduce_mode_ not in {"bulk", "async_param"}:
        raise ValueError(
            "grad_allreduce_mode must be 'bulk' or 'async_param', "
            f"got {grad_allreduce_mode!r}"
        )
    if crct_enabled and world_size_ >= 4 and grad_allreduce_mode_ != "bulk":
        raise ValueError(
            "crct_enabled=True with world_size>=4 requires "
            "grad_allreduce_mode='bulk'; the memory rank and train ranks "
            "must enter the same 3+1 SUM all-reduce path."
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
    is_crct_packet_rank = False
    is_crct_maintenance_rank = False
    crct_packet_rank = world_size_ - 1
    crct_maintenance_rank = world_size_ - 1
    crct_memory_ranks: list[int] = []
    crct_train_ranks: list[int] = list(range(world_size_))
    crct_memory_owner = "disabled"
    all_group = None
    train_group = None
    teacher_group = None
    slot_commit_group = None
    grad_group = None
    object_group = None
    grad_world_size = world_size_
    memory_rank_joins_grad = True
    stop_group = None
    if episodic_enabled or (crct_enabled and world_size_ >= 4):
        if not ddp_active:
            raise ValueError(
                "3+1 memory topology requires world_size > 1 (need at "
                "least 1 train rank + 1 memory rank); got "
                f"world_size={world_size_}"
            )
        if world_size_ < 2:
            raise ValueError(
                "3+1 memory topology requires world_size >= 2 "
                f"(1 train + 1 memory rank), got world_size={world_size_}"
            )
        if crct_enabled and not episodic_enabled:
            _crct_backend_for_topology = str(
                crct_async_teacher_transport_backend
            ).strip().lower()
            crct_topology = _crct_rank_topology(
                world_size=world_size_,
                replay_eviction_enabled=(
                    bool(replay_eviction_enabled)
                    and _crct_backend_for_topology
                    in {"mailbox", "file", "file_mailbox"}
                ),
            )
            crct_packet_rank = int(crct_topology["packet_rank"])
            crct_maintenance_rank = int(crct_topology["maintenance_rank"])
            crct_memory_ranks = [int(r) for r in crct_topology["memory_ranks"]]
            crct_train_ranks = [int(r) for r in crct_topology["train_ranks"]]
            crct_memory_owner = str(crct_topology["memory_owner"])
            is_crct_packet_rank = rank_ == crct_packet_rank
            is_crct_maintenance_rank = rank_ == crct_maintenance_rank
            is_episodic_rank = rank_ in set(crct_memory_ranks)
        else:
            is_episodic_rank = (rank_ == world_size_ - 1)
        # ``all_group`` is the all-rank process group as an explicit
        # handle. Phase 5 will introduce ``main_group`` (just train
        # ranks) without changing any all_group call sites. Note:
        # ``dist.new_group`` is itself a WORLD-collective on gloo/nccl
        # — every rank in WORLD must call it, even ranks not in the
        # subgroup. Here all_group is the world group, so all ranks
        # participate.
        all_group = dist.new_group(list(range(world_size_)))
        # Post-run diagnostics and online eval state are Python objects.
        # Keep those object collectives on Gloo even when tensor collectives
        # use NCCL; NCCL object gather materializes CUDA byte tensors and has
        # proven fragile on 8xH100 after asymmetric memory-rank work.
        object_group = dist.new_group(list(range(world_size_)), backend="gloo")
        grad_group = all_group
        if crct_enabled and not episodic_enabled:
            # CRCT's rank-3 teacher is an asynchronous coprocessor, not a
            # replay-gradient producer. Keep train-rank gradients and
            # stop checks on the train subgroup so oracle scoring can lag
            # without touching trunk SSM throughput.
            train_ranks = list(crct_train_ranks)
            train_group = dist.new_group(train_ranks)
            teacher_group = dist.new_group([0, crct_packet_rank])
            grad_group = train_group
            grad_world_size = len(train_ranks)
            memory_rank_joins_grad = False
            if not is_episodic_rank:
                stop_group = train_group
    scopt_active = isinstance(optimizer, ScarcityAwareOptimizer)
    if crct_enabled and scopt_active:
        raise ValueError(
            "crct_enabled=True is incompatible with ScOpt in this runner: "
            "both mechanisms own per-token LM loss reweighting."
        )
    if crct_enabled and getattr(model, "outer_model", None) is None:
        raise ValueError(
            "crct_enabled=True requires an outer memory module. The Exp24 "
            "fast constructor should set outer_model_dim > 0 for CRCT cells."
        )
    if crct_enabled and str(getattr(model, "buffer_mode", "")) != "append_only":
        raise ValueError(
            "crct_enabled=True requires buffer_mode='append_only' so the "
            "rank-3 teacher can populate its memory substrate."
        )
    if crct_enabled and world_size_ < 4:
        raise ValueError(
            "crct_enabled=True requires world_size>=4 so rank world_size-1 "
            "can own the teacher memory without coupling train-rank trunk "
            "forwards to cache reads."
        )
    if crct_enabled and not episodic_enabled and grad_world_size < 1:
        raise ValueError(
            "crct_enabled=True requires at least one train rank after "
            f"memory-role assignment; got world_size={world_size_} "
            f"memory_ranks={crct_memory_ranks}"
        )
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
    _action_space_config = {
        "episodic_controller_action_space_enabled": bool(
            episodic_controller_action_space_enabled
        ),
        "episodic_controller_action_space_trace_only": bool(
            episodic_controller_action_space_trace_only
        ),
        "episodic_controller_selection_readiness": float(
            episodic_controller_selection_readiness
        ),
        "episodic_controller_selection_max_delta": float(
            episodic_controller_selection_max_delta
        ),
        "episodic_controller_max_tags_per_query": (
            int(episodic_controller_max_tags_per_query)
            if episodic_controller_max_tags_per_query is not None
            else None
        ),
        "episodic_controller_action_trace_path": (
            str(episodic_controller_action_trace_path)
            if episodic_controller_action_trace_path is not None
            else None
        ),
        "episodic_controller_shared_event_ssm_enabled": bool(
            episodic_controller_shared_event_ssm_enabled
        ),
        "episodic_controller_ssm_hidden_dim": int(
            episodic_controller_ssm_hidden_dim
        ),
        "episodic_controller_ssm_seed": int(episodic_controller_ssm_seed),
        "episodic_controller_ssm_decay": float(episodic_controller_ssm_decay),
        "episodic_controller_ssm_input_scale": float(
            episodic_controller_ssm_input_scale
        ),
        "episodic_controller_ssm_head_scale": float(
            episodic_controller_ssm_head_scale
        ),
        "episodic_controller_action_learning_rate": float(
            episodic_controller_action_learning_rate
        ),
        "episodic_controller_action_reward_clip": float(
            episodic_controller_action_reward_clip
        ),
    }
    if episodic_controller_head_readiness is not None:
        _action_space_config["episodic_controller_head_readiness"] = dict(
            episodic_controller_head_readiness
        )
    if episodic_controller_head_max_delta is not None:
        _action_space_config["episodic_controller_head_max_delta"] = dict(
            episodic_controller_head_max_delta
        )
    # Episodic-rank consumer init. Builds the cache, in-process controller
    # queues, and lazy async write-ring attachment names on the episodic rank
    # only; train ranks and ``episodic_enabled=False`` get the no-op state.
    # No init-time barrier is needed because ring attach retries are
    # opportunistic and train ranks never wait for the memory side.
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
        "episodic_async_write_rings_enabled": bool(
            episodic_async_write_rings_enabled
        ),
        "episodic_cuda_write_event_stream_enabled": bool(
            episodic_cuda_write_event_stream_enabled
        ),
        "episodic_cuda_write_event_stage_depth": int(
            episodic_cuda_write_event_stage_depth
        ),
        "episodic_event_ring_id": episodic_event_ring_id,
        "episodic_write_ring_max_drain_per_step": int(
            episodic_write_ring_max_drain_per_step
        ),
        "episodic_compute_replay_ce_pair": bool(
            episodic_compute_replay_ce_pair
        ),
        **_action_space_config,
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
    episodic_write_drain_handle: _EpisodicWriteDrainHandle | None = None
    episodic_embedding_version_ref = [0]
    _episodic_controller_config = {
        "episodic_controller_score_mode": str(episodic_controller_score_mode),
        "episodic_controller_topk_k": int(episodic_controller_topk_k),
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
        "episodic_controller_simplex_trace_path": str(
            episodic_controller_simplex_trace_path or ""
        ),
        "episodic_controller_history_entries": int(
            episodic_controller_history_entries
        ),
        **_action_space_config,
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
        if crct_enabled:
            graph_rejections.append("crct_not_supported")
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
    fast_slow_action_trace_log = None
    fast_slow_action_space = None
    fast_slow_action_space_owner = (
        not bool(ddp_active)
        or int(rank_) == int(world_size_) - 1
    )
    if fast_slow.enabled and bool(
        _action_space_config.get("episodic_controller_action_space_enabled", False)
    ) and bool(
        fast_slow_action_space_owner
    ):
        fast_slow_action_trace_log = _action_trace_logger_from_config(
            _action_space_config,
            rank=f"fastslow{rank_}",
        )
        if fast_slow_action_trace_log is None:
            fast_slow_action_trace_log = []
        fast_slow_action_space = _build_action_space_from_config(
            _action_space_config,
            trace_log=fast_slow_action_trace_log,
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
    plasticity_budget_payloads_applied = 0
    plasticity_budget_payloads_missing = 0
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
    crct_cache: TransactionalWakeCache | None = None
    crct_scarcity: CrctScarcityAwareMemoryOptimizer | None = None
    crct_gradient_conflict: CrctGradientConflictMonitor | None = None
    crct_teacher_requests = 0
    crct_teacher_payloads = 0
    crct_teacher_fail_open = 0
    crct_teacher_param_syncs = 0
    crct_teacher_param_sync_seconds_sum = 0.0
    crct_teacher_param_sync_seconds_max = 0.0
    if str(crct_async_teacher_transport_backend).strip().lower() == "mailbox":
        # Mailbox mode uses a latest-complete mirror opportunity every step.
        # There is no independent teacher-weight cadence in the final ARM
        # path; one in-flight mirror write naturally drops extra attempts.
        crct_teacher_param_sync_interval = 1
    else:
        crct_teacher_param_sync_interval = (
            int(crct_teacher_score_interval_steps)
            if crct_teacher_param_sync_interval_steps is None
            else int(crct_teacher_param_sync_interval_steps)
        )
    crct_teacher_transport: _CrctAsyncTeacherTransport | None = None
    crct_maintenance_transport: _CrctMailboxTeacherTransport | None = None
    crct_slot_commit_transport: _CrctSlotCommitPeerTransport | None = None
    crct_memory_begin_step_by_role: dict[str, int] = {}
    crct_teacher_transport_mode = "disabled"
    crct_teacher_bypass_steps = 0
    replay_eviction_loop: ReplayEvictionLoop | None = None
    online_eval_state_payload: dict[str, Any] = {}
    # Split memory ranks ingest the same request stream and build their local
    # caches independently.  PyTorch Gloo P2P is not a safe hot commit lane on
    # this pod: timeout polling can close the pair, and unbounded waits violate
    # the nonblocking memory-rank contract.
    if bool(replay_eviction_enabled) and crct_enabled:
        replay_eviction_loop = ReplayEvictionLoop(
            action_mode=str(replay_eviction_mode),
            memory_streams=int(replay_eviction_memory_streams),
            eviction_threshold=float(replay_eviction_threshold),
            eviction_ema_beta=float(replay_eviction_ema_beta),
            min_slot_age_steps=int(replay_eviction_min_age_steps),
            max_seconds_per_tick=float(replay_eviction_max_seconds),
            trace_path=str(replay_eviction_trace_path) or None,
            trace_max_rows=int(replay_eviction_trace_max_rows),
            trace_flush_rows=int(replay_eviction_trace_flush_rows),
            probe_chunk_size=int(replay_eviction_probe_chunk_size),
            scoring_mode=str(replay_eviction_scoring_mode),
            oracle_confirm_top_k=int(replay_eviction_oracle_confirm_top_k),
            oracle_variant_chunk_size=int(replay_eviction_oracle_variant_chunk_size),
            drift_threshold=float(replay_eviction_drift_threshold),
            repr_drift_threshold=float(replay_eviction_repr_drift_threshold),
            refresh_lr=float(replay_eviction_refresh_lr),
            refresh_candidate_count=int(replay_eviction_refresh_candidate_count),
            refresh_proposal_rank=int(replay_eviction_refresh_proposal_rank),
            refresh_proposal_noise_scale=float(
                replay_eviction_refresh_proposal_noise_scale
            ),
            refresh_proposal_momentum=float(
                replay_eviction_refresh_proposal_momentum
            ),
            refresh_candidate_variant_chunk_size=int(
                replay_eviction_refresh_candidate_variant_chunk_size
            ),
            refresh_proposal_seed=int(replay_eviction_refresh_proposal_seed),
            controller_state_dim=int(replay_eviction_controller_state_dim),
            controller_rank=int(replay_eviction_controller_rank),
            controller_dt=float(replay_eviction_controller_dt),
            controller_gamma=float(replay_eviction_controller_gamma),
            controller_target_log_sv=float(
                replay_eviction_controller_target_log_sv
            ),
            controller_max_state_norm=float(
                replay_eviction_controller_max_state_norm
            ),
            controller_perturbation_scale=float(
                replay_eviction_controller_perturbation_scale
            ),
            controller_feedback_lr=float(replay_eviction_controller_feedback_lr),
            quarantine_threshold=float(replay_eviction_quarantine_threshold),
            max_quarantined=int(replay_eviction_max_quarantined),
            distill_peak_threshold=float(replay_eviction_distill_peak_threshold),
            peak_preserve_utility_threshold=float(
                replay_eviction_peak_preserve_utility_threshold
            ),
            peak_preserve_sharpness_threshold=float(
                replay_eviction_peak_preserve_sharpness_threshold
            ),
            useful_threshold=float(replay_eviction_useful_threshold),
            probe_buffer_size=int(replay_eviction_probe_buffer_size),
            frame_ttl_steps=int(replay_eviction_frame_ttl_steps),
            slot_work_chunk_size=int(replay_eviction_slot_work_chunk_size),
            action_agreement_count=int(replay_eviction_action_agreement_count),
            commit_policy=str(replay_eviction_commit_policy),
            commit_online_lr=float(replay_eviction_commit_online_lr),
            commit_temperature=float(replay_eviction_commit_temperature),
            arm_runtime_enabled=(
                bool(replay_eviction_arm_runtime_enabled)
                and int(rank_) == int(crct_maintenance_rank)
            ),
            arm_runtime_namespace=(
                str(replay_eviction_arm_runtime_namespace) or None
            ),
            evidence_engine_enabled=(
                bool(replay_eviction_evidence_engine_enabled)
                and int(rank_) == int(crct_maintenance_rank)
            ),
            evidence_engine_d_model=int(replay_eviction_evidence_engine_d_model),
        )
    crct_rank_diagnostics: list[dict[str, Any] | None] | None = None
    if crct_enabled:
        crct_cache = TransactionalWakeCache(
            max_moments=0,
            max_hidden_buffer=0,
        )
        crct_scarcity = CrctScarcityAwareMemoryOptimizer(
            tau=float(crct_lm_weight_tau),
            target_read_rate=float(crct_target_read_rate),
            target_write_rate=float(crct_target_write_rate),
            dual_lr=float(crct_dual_lr),
            ema_beta=float(crct_ema_beta),
            max_price=float(crct_max_price),
        )
        if bool(crct_gradient_conflict_enabled):
            crct_gradient_conflict = CrctGradientConflictMonitor(
                enabled=True,
                ema_beta=float(crct_gradient_conflict_ema_beta),
                catastrophic_threshold=float(
                    crct_gradient_conflict_catastrophic_threshold
                ),
                soft_gate_strength=float(
                    crct_gradient_conflict_soft_gate_strength
                ),
                soft_gate_floor=float(crct_gradient_conflict_soft_gate_floor),
                trace_path=str(crct_gradient_conflict_trace_path or ""),
                trace_stride=int(crct_gradient_conflict_trace_stride),
                trace_max_rows=int(crct_gradient_conflict_trace_max_rows),
                trace_flush_rows=int(crct_gradient_conflict_trace_flush_rows),
            )
        crct_transport_backend = str(
            crct_async_teacher_transport_backend
        ).strip().lower()
        if bool(crct_async_teacher_transport) and world_size_ >= 4 and dist.is_initialized():
            if crct_transport_backend in {"mailbox", "file", "file_mailbox"}:
                crct_teacher_transport_mode = "async_rank0_memory_mailbox"
            elif crct_transport_backend in {"collective", "broadcast", "nccl"}:
                crct_teacher_transport_mode = (
                    "async_rank0_memory_broadcast"
                    if teacher_group is not None
                    else "sync_collective"
                )
            else:
                raise ValueError(
                    "crct_async_teacher_transport_backend must be one of "
                    "'collective' or 'mailbox'; got "
                    f"{crct_async_teacher_transport_backend!r}"
                )
        else:
            crct_teacher_transport_mode = (
                "sync_collective" if world_size_ >= 4 else "inline"
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

    # Episodic write emit handle. Train ranks get a local slot tensor for
    # pack-test compatibility plus a WRITE_EVENT ring producer; the episodic
    # rank gets a no-op handle. Returns ``None`` for ``episodic_enabled=False``
    # or ``world_size == 1`` runs.
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
            "episodic_async_write_rings_enabled": bool(
                episodic_async_write_rings_enabled
            ),
            "episodic_cuda_write_event_stream_enabled": bool(
                episodic_cuda_write_event_stream_enabled
            ),
            "episodic_cuda_write_event_stage_depth": int(
                episodic_cuda_write_event_stage_depth
            ),
            "episodic_event_ring_id": episodic_event_ring_id,
            "model_dim": resolved_key_rep_dim,
            **_action_space_config,
        }
        episodic_emit_handle = _create_episodic_emit(
            rank=rank_,
            world_size=world_size_,
            device=device,
            config=emit_config,
        )

    # Async WRITE_EVENT consumer on the episodic rank. It starts before the
    # controller so the query queue can fill independently of train steps.
    episodic_write_drain_handle = _spawn_episodic_write_drain(
        consumer=episodic_consumer,
        is_episodic_rank=bool(is_episodic_rank),
        episodic_enabled=bool(episodic_enabled),
        config=_episodic_controller_config,
        embedding_version_ref=episodic_embedding_version_ref,
    )

    # Phase 2: spawn the CPU controller thread on the episodic rank when
    # controller_query_enabled=True. Returns None on every other code path
    # (train ranks, episodic disabled, controller flag off). The handle is
    # consumed by ``_shutdown_episodic_controller`` in the outer ``finally``.
    episodic_controller_handle = _spawn_episodic_controller(
        consumer=episodic_consumer,
        is_episodic_rank=bool(is_episodic_rank),
        episodic_enabled=bool(episodic_enabled),
        config=_episodic_controller_config,
    )

    def _ensure_crct_teacher_transport(
        *,
        full_ids_shape: tuple[int, ...],
        payload_shape: tuple[int, ...],
        payload_device: torch.device,
    ) -> None:
        nonlocal crct_teacher_transport
        if crct_teacher_transport is not None:
            return
        if crct_teacher_transport_mode not in {
            "async_rank0_memory_broadcast",
            "async_rank0_memory_mailbox",
        }:
            return
        payload_dtype = _resolve_crct_async_payload_dtype(
            crct_async_teacher_payload_dtype,
            device=payload_device,
        )
        if crct_teacher_transport_mode == "async_rank0_memory_mailbox":
            mailbox_dir = str(crct_teacher_mailbox_dir or "").strip()
            if not mailbox_dir:
                raise ValueError(
                    "CRCT mailbox transport requires "
                    "crct_teacher_mailbox_dir to be set"
                )
            crct_teacher_transport = _CrctMailboxTeacherTransport(
                rank=rank_,
                world_size=world_size_,
                mailbox_dir=mailbox_dir,
                payload_shape=payload_shape,
                full_ids_shape=full_ids_shape,
                device=payload_device,
                payload_dtype=payload_dtype,
                max_local_batches=int(crct_async_teacher_pending_batches),
                max_payload_lag_steps=int(crct_async_teacher_max_lag_steps),
                score_interval_steps=int(crct_teacher_score_interval_steps),
                memory_rank=int(crct_packet_rank),
                memory_role="packet",
                produce_results=True,
                plasticity_ema_beta=float(crct_ema_beta),
                hidden_dim=int(
                    getattr(model, "dim", 0)
                    or getattr(model.lm_head, "in_features", 0)
                    or 0
                ),
                score_stage_timing_enabled=bool(
                    crct_score_stage_timing_enabled
                ),
            )
            return
        if teacher_group is None:
            raise ValueError(
                "CRCT async rank0-memory transport requires "
                "teacher_group to be initialized"
            )
        crct_teacher_transport = _CrctAsyncTeacherTransport(
            rank=rank_,
            world_size=world_size_,
            teacher_group=teacher_group,
            payload_shape=payload_shape,
            full_ids_shape=full_ids_shape,
            device=payload_device,
            payload_dtype=payload_dtype,
            max_local_batches=int(crct_async_teacher_pending_batches),
            max_payload_lag_steps=int(crct_async_teacher_max_lag_steps),
            score_interval_steps=int(crct_teacher_score_interval_steps),
            memory_rank=int(crct_packet_rank),
        )

    def _ensure_crct_maintenance_transport(
        *,
        full_ids_shape: tuple[int, ...],
        payload_shape: tuple[int, ...],
        payload_device: torch.device,
    ) -> None:
        nonlocal crct_maintenance_transport
        if crct_maintenance_transport is not None:
            return
        if int(crct_maintenance_rank) == int(crct_packet_rank):
            return
        if crct_teacher_transport_mode != "async_rank0_memory_mailbox":
            return
        if rank_ not in {0, int(crct_maintenance_rank)}:
            return
        mailbox_dir = str(crct_teacher_mailbox_dir or "").strip()
        if not mailbox_dir:
            raise ValueError(
                "CRCT mailbox transport requires crct_teacher_mailbox_dir to be set"
            )
        payload_dtype = _resolve_crct_async_payload_dtype(
            crct_async_teacher_payload_dtype,
            device=payload_device,
        )
        crct_maintenance_transport = _CrctMailboxTeacherTransport(
            rank=rank_,
            world_size=world_size_,
            mailbox_dir=mailbox_dir,
            payload_shape=payload_shape,
            full_ids_shape=full_ids_shape,
            device=payload_device,
            payload_dtype=payload_dtype,
            max_local_batches=int(crct_async_teacher_pending_batches),
            max_payload_lag_steps=int(crct_async_teacher_max_lag_steps),
            score_interval_steps=int(crct_teacher_score_interval_steps),
            memory_rank=int(crct_maintenance_rank),
            memory_role="maintenance",
            produce_results=False,
            plasticity_ema_beta=float(crct_ema_beta),
            hidden_dim=int(
                getattr(model, "dim", 0)
                or getattr(model.lm_head, "in_features", 0)
                or 0
            ),
            score_stage_timing_enabled=bool(crct_score_stage_timing_enabled),
        )

    def _pump_crct_memory_rank(
        step: int,
        *,
        transport: _CrctAsyncTeacherTransport | _CrctMailboxTeacherTransport | None = None,
    ) -> bool:
        active_transport = crct_teacher_transport if transport is None else transport
        if active_transport is None:
            return False
        pump_t0 = time.perf_counter()
        assert crct_cache is not None
        try:
            score_before = int(
                active_transport.metrics.get("payloads_scored", 0)
            )
            served_before = int(
                active_transport.metrics.get("payloads_served", 0)
            )
            weight_before = int(
                active_transport.metrics.get("weight_snapshot_applied", 0)
            )
            request_pops_before = int(
                active_transport.metrics.get("memory_rank_pump_request_pops", 0)
            )
            poll_requests = getattr(active_transport, "_poll_requests", None)
            if callable(poll_requests):
                popped = int(poll_requests())
                active_transport.metrics["memory_rank_pump_request_pops"] += popped
            active_transport.after_optimizer_step(
                model=model,
                cache=crct_cache,
                scarcity_optimizer=crct_scarcity,
                step=int(step),
                total_steps=max_steps,
                tau=float(crct_lm_weight_tau),
                strength=float(crct_lm_weight_strength),
                w_max=float(crct_lm_weight_w_max),
                alpha_max=float(crct_lm_weight_alpha_max),
                memory_write_tokens=int(crct_memory_write_tokens_per_step),
                gradient_conflict_monitor=crct_gradient_conflict,
                replay_eviction_loop=replay_eviction_loop,
                fast_slow=None,
                fast_slow_action_space=None,
                fast_slow_nll_chunk_size=int(chunk_size),
                slot_commit_transport=crct_slot_commit_transport,
                update_model_memory_after=not (
                    int(crct_packet_rank) != int(crct_maintenance_rank)
                    and getattr(active_transport, "memory_role", "") == "maintenance"
                ),
            )
            request_pops_after = int(
                active_transport.metrics.get("memory_rank_pump_request_pops", 0)
            )
            return (
                request_pops_after > request_pops_before
                or int(active_transport.metrics.get("payloads_scored", 0))
                > score_before
                or int(active_transport.metrics.get("payloads_served", 0))
                > served_before
                or int(active_transport.metrics.get("weight_snapshot_applied", 0))
                > weight_before
            )
        finally:
            elapsed = time.perf_counter() - pump_t0
            active_transport.metrics["memory_rank_pump_loop_seconds_sum"] += float(
                elapsed
            )
            active_transport.metrics["memory_rank_pump_loop_seconds_max"] = max(
                float(active_transport.metrics["memory_rank_pump_loop_seconds_max"]),
                float(elapsed),
            )

    def _maybe_tick_crct_replay_maintenance(
        step: int,
        *,
        packet_work_done: bool,
        transport: _CrctAsyncTeacherTransport | _CrctMailboxTeacherTransport | None = None,
        defer_for_packet_work: bool = True,
    ) -> bool:
        """Run low-priority replay maintenance only when packet service is clear."""
        if replay_eviction_loop is None:
            return False
        active_transport = crct_teacher_transport if transport is None else transport
        probe_fresh = _crct_replay_cache_probe(
            replay_eviction_loop,
            model,
            step,
        )
        if active_transport is not None and probe_fresh:
            active_transport.metrics["memory_rank_replay_probes_ingested"] += 1
        if not replay_eviction_loop.has_probe():
            return False
        if packet_work_done and defer_for_packet_work:
            if active_transport is not None:
                active_transport.metrics[
                    "memory_rank_replay_deferred_for_packet_work"
                ] += 1
                active_transport.metrics["low_priority_maintenance_last_reason"] = (
                    "packet_work"
                )
            return False
        defer_low_priority = False
        if active_transport is not None:
            should_defer = getattr(
                active_transport,
                "should_defer_low_priority_maintenance",
                None,
            )
            if callable(should_defer):
                defer_low_priority = bool(should_defer())
        if defer_low_priority:
            if active_transport is not None:
                active_transport.metrics[
                    "memory_rank_replay_deferred_for_backpressure"
                ] += 1
            return False
        replay_step = _crct_replay_tick_step(
            replay_eviction_loop,
            model,
            step,
        )
        tick_result = replay_eviction_loop.tick(
            model=model,
            step=replay_step,
        )
        if crct_slot_commit_transport is not None:
            for commit in tick_result.slot_commits:
                crct_slot_commit_transport.submit_peer(commit)
        if active_transport is not None:
            active_transport.metrics["memory_rank_replay_ticks"] += 1
        if tick_result.evicted_indices:
            replay_eviction_loop.flush_trace()
        return True

    try:
        while True:
            memory_rank_loop_t0 = (
                time.perf_counter()
                if (
                    crct_enabled
                    and bool(is_episodic_rank)
                    and not bool(episodic_enabled)
                    and not bool(memory_rank_joins_grad)
                )
                else None
            )
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
            memory_rank_wall_stop_check = (
                crct_enabled
                and bool(is_episodic_rank)
                and not bool(episodic_enabled)
                and not bool(memory_rank_joins_grad)
            )
            if (
                memory_rank_wall_stop_check
                or steps == 0
                or steps % check_interval == 0
                or max_steps_reached
            ):
                elapsed = time.perf_counter() - start_time
                if memory_rank_wall_stop_check:
                    local_stop = _should_stop_memory_rank_loop(
                        steps=steps,
                        elapsed_s=elapsed,
                        budget_seconds=budget_seconds,
                        stop_margin_seconds=stop_margin_seconds,
                        max_steps=max_steps,
                    )
                    active_stop_transport = (
                        crct_teacher_transport
                        if bool(is_crct_packet_rank)
                        else crct_maintenance_transport
                    )
                    if _should_defer_memory_rank_stop_for_shutdown(
                        local_stop=bool(local_stop),
                        elapsed_s=elapsed,
                        budget_seconds=budget_seconds,
                        stop_margin_seconds=stop_margin_seconds,
                        transport_mode=str(crct_teacher_transport_mode),
                        active_transport=active_stop_transport,
                    ):
                        local_stop = False
                else:
                    local_stop = should_stop_training_loop(
                        steps=steps,
                        elapsed_s=elapsed,
                        budget_seconds=budget_seconds,
                        stop_margin_seconds=stop_margin_seconds,
                        max_steps=max_steps,
                    )
                stop_ddp_active = bool(ddp_active)
                stop_group_eff = stop_group
                if crct_enabled and not episodic_enabled and is_episodic_rank:
                    stop_ddp_active = False
                    stop_group_eff = None
                if should_stop_now(
                    local_stop,
                    device,
                    stop_ddp_active,
                    group=stop_group_eff,
                ):
                    break

            if (
                crct_enabled
                and bool(is_episodic_rank)
                and not bool(episodic_enabled)
                and not bool(memory_rank_joins_grad)
                and crct_teacher_transport_mode == "async_rank0_memory_mailbox"
            ):
                active_memory_transport = (
                    crct_teacher_transport
                    if bool(is_crct_packet_rank)
                    else crct_maintenance_transport
                )
                if crct_teacher_transport_mode in {
                    "async_rank0_memory_broadcast",
                    "async_rank0_memory_mailbox",
                }:
                    if bool(is_crct_packet_rank):
                        _ensure_crct_teacher_transport(
                            full_ids_shape=(int(batch_size), int(seq_len) + 1),
                            payload_shape=(1, int(batch_size), int(seq_len)),
                            payload_device=device,
                        )
                    if bool(is_crct_maintenance_rank):
                        _ensure_crct_maintenance_transport(
                            full_ids_shape=(int(batch_size), int(seq_len) + 1),
                            payload_shape=(1, int(batch_size), int(seq_len)),
                            payload_device=device,
                        )
                    active_memory_transport = (
                        crct_teacher_transport
                        if bool(is_crct_packet_rank)
                        else crct_maintenance_transport
                    )
                    crct_teacher_requests += 1
                if (
                    active_memory_transport is not None
                    and crct_teacher_transport_mode
                    == "async_rank0_memory_broadcast"
                    and bool(is_crct_packet_rank)
                ):
                    role_key = str(
                        active_memory_transport.metrics.get("memory_role", "packet")
                    )
                    if crct_memory_begin_step_by_role.get(role_key) != int(steps):
                        dummy_inputs = torch.empty(
                            (int(batch_size), int(seq_len)),
                            device=device,
                            dtype=torch.int32,
                        )
                        dummy_targets = torch.empty(
                            (int(batch_size), int(seq_len)),
                            device=device,
                            dtype=torch.long,
                        )
                        active_memory_transport.begin_step(
                            inputs=dummy_inputs,
                            targets=dummy_targets,
                            step=steps,
                        )
                        crct_memory_begin_step_by_role[role_key] = int(steps)
                if (
                    active_memory_transport is not None
                    and memory_rank_loop_t0 is not None
                ):
                    pre_pump_s = time.perf_counter() - memory_rank_loop_t0
                    active_memory_transport.metrics[
                        "memory_rank_pre_pump_seconds_sum"
                    ] += float(pre_pump_s)
                    active_memory_transport.metrics[
                        "memory_rank_pre_pump_seconds_max"
                    ] = max(
                        float(
                            active_memory_transport.metrics[
                                "memory_rank_pre_pump_seconds_max"
                            ]
                        ),
                        float(pre_pump_s),
                    )
                commit_work_done = False
                if crct_slot_commit_transport is not None:
                    commit_work_done = crct_slot_commit_transport.poll(model=model)
                pump_work_done = _pump_crct_memory_rank(
                    steps,
                    transport=active_memory_transport,
                )
                if (
                    active_memory_transport is not None
                    and bool(
                        getattr(
                            active_memory_transport,
                            "shutdown_requested",
                            False,
                        )
                    )
                ):
                    break
                if crct_slot_commit_transport is not None:
                    commit_work_done = (
                        crct_slot_commit_transport.poll(model=model)
                        or commit_work_done
                    )
                replay_t0 = time.perf_counter()
                replay_work_done = bool(is_crct_maintenance_rank) and _maybe_tick_crct_replay_maintenance(
                    steps,
                    packet_work_done=bool(pump_work_done),
                    transport=active_memory_transport,
                    defer_for_packet_work=not bool(
                        int(crct_maintenance_rank) != int(crct_packet_rank)
                    ),
                )
                if crct_slot_commit_transport is not None:
                    commit_work_done = (
                        crct_slot_commit_transport.poll(model=model)
                        or commit_work_done
                    )
                if replay_work_done:
                    pump_work_done = True
                if commit_work_done:
                    pump_work_done = True
                if active_memory_transport is not None:
                    replay_s = time.perf_counter() - replay_t0
                    active_memory_transport.metrics[
                        "memory_rank_replay_seconds_sum"
                    ] += float(replay_s)
                    active_memory_transport.metrics[
                        "memory_rank_replay_seconds_max"
                    ] = max(
                        float(
                            active_memory_transport.metrics[
                                "memory_rank_replay_seconds_max"
                            ]
                        ),
                        float(replay_s),
                    )
                if pump_work_done:
                    if active_memory_transport is not None:
                        active_memory_transport.metrics["memory_rank_pump_steps"] += 1
                    losses.append(torch.zeros((), device=device, dtype=torch.float32))
                    steps += 1
                else:
                    if active_memory_transport is not None:
                        active_memory_transport.metrics[
                            "memory_rank_pump_idle_spins"
                        ] += 1
                    if active_memory_transport is not None:
                        active_memory_transport.metrics[
                            "memory_rank_pump_idle_yields"
                        ] += 1
                    _hotpath_yield()
                if (
                    active_memory_transport is not None
                    and memory_rank_loop_t0 is not None
                ):
                    loop_s = time.perf_counter() - memory_rank_loop_t0
                    active_memory_transport.metrics[
                        "memory_rank_outer_loop_seconds_sum"
                    ] += float(loop_s)
                    active_memory_transport.metrics[
                        "memory_rank_outer_loop_seconds_max"
                    ] = max(
                        float(
                            active_memory_transport.metrics[
                                "memory_rank_outer_loop_seconds_max"
                            ]
                        ),
                        float(loop_s),
                    )
                continue

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

            crct_payload: dict[str, torch.Tensor] | None = None
            train_inputs = inputs
            train_targets = targets
            memory_rank_request_pops_before: int | None = None
            if crct_enabled:
                assert crct_cache is not None
                crct_transport_participant = (
                    crct_teacher_transport_mode
                    in {
                        "async_rank0_memory_broadcast",
                        "async_rank0_memory_mailbox",
                    }
                    and rank_ in {0, int(crct_packet_rank)}
                )
                crct_maintenance_transport_participant = (
                    crct_teacher_transport_mode == "async_rank0_memory_mailbox"
                    and int(crct_maintenance_rank) != int(crct_packet_rank)
                    and rank_ in {0, int(crct_maintenance_rank)}
                )
                if crct_teacher_transport_mode in {
                    "async_rank0_memory_broadcast",
                    "async_rank0_memory_mailbox",
                }:
                    if not (
                        crct_transport_participant
                        or crct_maintenance_transport_participant
                    ):
                        crct_teacher_bypass_steps += 1
                    else:
                        crct_teacher_requests += 1
                    if not (
                        crct_transport_participant
                        or crct_maintenance_transport_participant
                    ):
                        pass
                    elif (
                        crct_teacher_transport_mode == "async_rank0_memory_broadcast"
                        and teacher_group is None
                    ):
                        raise ValueError(
                            "CRCT async rank0-memory transport requires "
                            "teacher_group to be initialized"
                        )
                    elif crct_transport_participant and crct_teacher_transport is None:
                        _ensure_crct_teacher_transport(
                            full_ids_shape=tuple(
                                int(x)
                                for x in _crct_full_input_ids(inputs, targets).shape
                            ),
                            payload_shape=(
                                1,
                                int(targets.shape[0]),
                                int(targets.shape[1]),
                            ),
                            payload_device=inputs.device,
                        )
                    if (
                        crct_maintenance_transport_participant
                        and crct_maintenance_transport is None
                    ):
                        _ensure_crct_maintenance_transport(
                            full_ids_shape=tuple(
                                int(x)
                                for x in _crct_full_input_ids(inputs, targets).shape
                            ),
                            payload_shape=(
                                1,
                                int(targets.shape[0]),
                                int(targets.shape[1]),
                            ),
                            payload_device=inputs.device,
                        )
                    if crct_teacher_transport is None:
                        ready = None
                    else:
                        if (
                            crct_transport_participant
                            and rank_ == int(crct_packet_rank)
                            and not bool(memory_rank_joins_grad)
                        ):
                            memory_rank_request_pops_before = int(
                                crct_teacher_transport.metrics.get(
                                    "memory_rank_pump_request_pops", 0
                                )
                            )
                        skip_duplicate_memory_collective = (
                            crct_teacher_transport_mode
                            == "async_rank0_memory_broadcast"
                            and crct_transport_participant
                            and rank_ == int(crct_packet_rank)
                            and not bool(memory_rank_joins_grad)
                            and crct_memory_begin_step_by_role.get("packet")
                            == int(steps)
                        )
                        if skip_duplicate_memory_collective:
                            ready = None
                        else:
                            ready = crct_teacher_transport.begin_step(
                                inputs=inputs,
                                targets=targets,
                                step=steps,
                            )
                            if (
                                crct_teacher_transport_mode
                                == "async_rank0_memory_broadcast"
                                and crct_transport_participant
                                and rank_ == int(crct_packet_rank)
                                and not bool(memory_rank_joins_grad)
                            ):
                                crct_memory_begin_step_by_role["packet"] = int(steps)
                        if ready is not None:
                            crct_payload, train_inputs, train_targets = ready
                            _apply_fast_slow_result_payload(
                                model=model,
                                fast_slow=fast_slow,
                                payload=crct_payload,
                                metrics=crct_teacher_transport.metrics,
                            )
                    if crct_maintenance_transport is not None:
                        crct_maintenance_transport.begin_step(
                            inputs=inputs,
                            targets=targets,
                            step=steps,
                        )
                else:
                    crct_teacher_requests += 1
                    crct_payload = _collect_crct_teacher_payload(
                        model=model,
                        cache=crct_cache,
                        scarcity_optimizer=crct_scarcity,
                        inputs=inputs,
                        targets=targets,
                        rank=rank_,
                        world_size=world_size_,
                        all_group=all_group,
                        step=steps,
                        total_steps=max_steps,
                        tau=float(crct_lm_weight_tau),
                        strength=float(crct_lm_weight_strength),
                        w_max=float(crct_lm_weight_w_max),
                        alpha_max=float(crct_lm_weight_alpha_max),
                        memory_write_tokens=int(crct_memory_write_tokens_per_step),
                        gradient_conflict_monitor=crct_gradient_conflict,
                    )
                if crct_payload is None:
                    if rank_ == 0:
                        crct_teacher_fail_open += 1
                elif rank_ == 0:
                    crct_teacher_payloads += 1

            if (
                crct_enabled
                and bool(is_episodic_rank)
                and not bool(episodic_enabled)
                and not bool(memory_rank_joins_grad)
            ):
                pump_work_done = False
                if crct_teacher_transport is not None:
                    assert crct_cache is not None
                    score_before = int(
                        crct_teacher_transport.metrics.get("payloads_scored", 0)
                    )
                    served_before = int(
                        crct_teacher_transport.metrics.get("payloads_served", 0)
                    )
                    weight_before = int(
                        crct_teacher_transport.metrics.get(
                            "weight_snapshot_applied", 0
                        )
                    )
                    crct_teacher_transport.after_optimizer_step(
                        model=model,
                        cache=crct_cache,
                        scarcity_optimizer=crct_scarcity,
                        step=steps,
                        total_steps=max_steps,
                        tau=float(crct_lm_weight_tau),
                        strength=float(crct_lm_weight_strength),
                        w_max=float(crct_lm_weight_w_max),
                        alpha_max=float(crct_lm_weight_alpha_max),
                        memory_write_tokens=int(crct_memory_write_tokens_per_step),
                        gradient_conflict_monitor=crct_gradient_conflict,
                        replay_eviction_loop=replay_eviction_loop,
                        fast_slow=None,
                        fast_slow_action_space=None,
                        fast_slow_nll_chunk_size=int(chunk_size),
                    )
                    if bool(
                        getattr(crct_teacher_transport, "shutdown_requested", False)
                    ):
                        break
                    if (
                        crct_teacher_transport_mode
                        == "async_rank0_memory_broadcast"
                        and teacher_group is not None
                        and int(crct_teacher_param_sync_interval) > 0
                        and steps % int(crct_teacher_param_sync_interval) == 0
                    ):
                        crct_teacher_transport.wait_for_pending_collectives()
                        sync_t0 = time.perf_counter()
                        _broadcast_model_params_coalesced(
                            model,
                            src=0,
                            group=teacher_group,
                        )
                        sync_s = time.perf_counter() - sync_t0
                        crct_teacher_param_syncs += 1
                        crct_teacher_param_sync_seconds_sum += float(sync_s)
                        crct_teacher_param_sync_seconds_max = max(
                            float(crct_teacher_param_sync_seconds_max),
                            float(sync_s),
                        )
                    request_pops_after = int(
                        crct_teacher_transport.metrics.get(
                            "memory_rank_pump_request_pops", 0
                        )
                    )
                    request_pops_before = (
                        int(memory_rank_request_pops_before)
                        if memory_rank_request_pops_before is not None
                        else request_pops_after
                    )
                    pump_work_done = (
                        request_pops_after > request_pops_before
                        or int(
                            crct_teacher_transport.metrics.get(
                                "payloads_scored", 0
                            )
                        )
                        > score_before
                        or int(
                            crct_teacher_transport.metrics.get(
                                "payloads_served", 0
                            )
                        )
                        > served_before
                        or int(
                            crct_teacher_transport.metrics.get(
                                "weight_snapshot_applied", 0
                            )
                        )
                        > weight_before
                )
                if _maybe_tick_crct_replay_maintenance(
                    steps,
                    packet_work_done=bool(pump_work_done),
                ):
                    pump_work_done = True
                if pump_work_done:
                    if crct_teacher_transport is not None:
                        crct_teacher_transport.metrics["memory_rank_pump_steps"] += 1
                    losses.append(torch.zeros((), device=device, dtype=torch.float32))
                    steps += 1
                else:
                    if crct_teacher_transport is not None:
                        crct_teacher_transport.metrics[
                            "memory_rank_pump_idle_spins"
                        ] += 1
                    if crct_teacher_transport is not None:
                        crct_teacher_transport.metrics[
                            "memory_rank_pump_idle_yields"
                        ] += 1
                    if (
                        crct_teacher_transport is not None
                        and crct_teacher_transport_mode
                        == "async_rank0_memory_broadcast"
                    ):
                        crct_teacher_transport.wait_for_pending_collectives()
                    else:
                        _hotpath_yield()
                continue

            optimizer.zero_grad(set_to_none=True)
            if predictive_aux_optimizer is not None:
                predictive_aux_optimizer.zero_grad(set_to_none=True)
            if async_grad_reducer is not None:
                async_grad_reducer.reset()
            # Stage captured SSM states for CD. Must enter the
            # capture_states() contexts BEFORE the encode call in
            # _run_train_step so the cores write into _captured_states.
            # We keep the stack open across the train-step call and
            # read getters BEFORE exiting — the real CareSSMCore clears
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
                    inputs=train_inputs,
                    targets=train_targets,
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
                    grad_group=grad_group,
                    grad_world_size=grad_world_size,
                    memory_rank_joins_grad=memory_rank_joins_grad,
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
                    crct_enabled=bool(crct_enabled),
                    crct_payload=crct_payload,
                    crct_memory_write_tokens_per_step=int(
                        crct_memory_write_tokens_per_step
                    ),
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
            # real CareSSMCore can clear its _captured_states per contract.
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
                    hidden_cd = model.encode(train_inputs)
                    normed_cd = model.final_norm(hidden_cd)
                    B_cd, T_cd = train_inputs.shape[0], train_inputs.shape[1]
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
                                train_targets,
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
                            train_targets.reshape(-1),
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
            if crct_enabled:
                if _apply_plasticity_budget_payload(
                    optimizer=optimizer,
                    payload=crct_payload,
                    strength=float(crct_plasticity_budget_strength),
                ):
                    plasticity_budget_payloads_applied += 1
                elif crct_payload is not None:
                    plasticity_budget_payloads_missing += 1
            optimizer.step()
            if predictive_aux_optimizer is not None:
                predictive_aux_optimizer.step()
            fast_slow_decision_for_snapshot: FastSlowDecision | None = None
            if (
                fast_slow.enabled
                and int(fast_slow.interval) > 0
                and not (
                    bool(is_episodic_rank) and not bool(memory_rank_joins_grad)
                )
            ):
                # Legacy fixed-interval mode only. The final learned path owns
                # consolidation decisions on the memory/oracle rank and mirrors
                # commands back through the mailbox result, avoiding per-step
                # Python policy orchestration on the trunk ranks.
                fast_slow_decision = fast_slow.decide(
                    step=int(steps + 1),
                    action_space=None,
                    reward_context=None,
                )
                fast_slow.apply_decision(model, fast_slow_decision)
                fast_slow_decision_for_snapshot = fast_slow_decision
            if (
                crct_teacher_transport is not None
                and crct_teacher_transport_mode == "async_rank0_memory_mailbox"
                and not bool(memory_rank_joins_grad)
            ):
                publish = getattr(
                    crct_teacher_transport,
                    "maybe_publish_weight_snapshot",
                    None,
                )
                if callable(publish):
                    publish(
                        model=model,
                        step=steps,
                        fast_slow_decision=fast_slow_decision_for_snapshot,
                    )
            if (
                crct_teacher_transport is not None
                and crct_teacher_transport_mode == "async_rank0_memory_broadcast"
                and not bool(memory_rank_joins_grad)
                and teacher_group is not None
                and int(crct_teacher_param_sync_interval) > 0
                and steps % int(crct_teacher_param_sync_interval) == 0
            ):
                crct_teacher_transport.wait_for_pending_collectives()
                sync_t0 = time.perf_counter()
                _broadcast_model_params_coalesced(
                    model,
                    src=0,
                    group=teacher_group,
                )
                sync_s = time.perf_counter() - sync_t0
                crct_teacher_param_syncs += 1
                crct_teacher_param_sync_seconds_sum += float(sync_s)
                crct_teacher_param_sync_seconds_max = max(
                    float(crct_teacher_param_sync_seconds_max),
                    float(sync_s),
                )
            if crct_teacher_transport is not None:
                assert crct_cache is not None
                crct_teacher_transport.after_optimizer_step(
                    model=model,
                    cache=crct_cache,
                    scarcity_optimizer=crct_scarcity,
                    step=steps,
                    total_steps=max_steps,
                    tau=float(crct_lm_weight_tau),
                    strength=float(crct_lm_weight_strength),
                    w_max=float(crct_lm_weight_w_max),
                    alpha_max=float(crct_lm_weight_alpha_max),
                    memory_write_tokens=int(crct_memory_write_tokens_per_step),
                    gradient_conflict_monitor=crct_gradient_conflict,
                    replay_eviction_loop=replay_eviction_loop,
                    fast_slow=fast_slow,
                    fast_slow_action_space=fast_slow_action_space,
                    fast_slow_nll_chunk_size=int(chunk_size),
                )
            # Replay-eviction: rank-3 streaming maintenance. Fresh teacher
            # score batches are ingested as probe frames; ticks consume
            # bounded slot-work chunks from the rolling frame stream.
            if replay_eviction_loop is not None and rank_ == world_size_ - 1:
                _maybe_tick_crct_replay_maintenance(
                    steps,
                    packet_work_done=False,
                )
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
        if crct_teacher_transport is not None:
            crct_teacher_transport.close()
        if crct_maintenance_transport is not None:
            crct_maintenance_transport.close()
        if crct_slot_commit_transport is not None:
            crct_slot_commit_transport.close()
        # Phase 2: stop the episodic controller thread. None handle
        # (train ranks, episodic disabled, controller flag off) is a
        # no-op. Bounded join so a stuck loop can't block the runner's
        # exit indefinitely.
        _shutdown_episodic_controller(episodic_controller_handle)
        _shutdown_episodic_write_drain(episodic_write_drain_handle)
        _cleanup_episodic_event_rings(episodic_emit_handle)
        _cleanup_episodic_event_rings(episodic_consumer)
        if hasattr(fast_slow_action_trace_log, "close"):
            try:
                fast_slow_action_trace_log.close()
            except Exception:
                pass

    if ddp_active:
        _control_barrier(
            group=object_group or all_group,
            label="train_teardown",
        )
    if crct_teacher_transport is not None:
        crct_teacher_transport.close()
    if crct_maintenance_transport is not None:
        crct_maintenance_transport.unlink_shared_resources()

    if (
        fast_slow.enabled
        and str(fast_slow_eval_copy).strip().lower() == "slow"
        and fast_slow.should_copy_slow_to_model_for_eval()
    ):
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
                group=object_group or all_group,
            )
            if rank_ == 0 and gathered_payloads is not None:
                episodic_cache_payload = next(
                    (p for p in gathered_payloads if p is not None),
                    None,
                )
        else:
            episodic_cache_payload = local_cache_payload

    if crct_enabled:
        local_crct_transport = (
            crct_teacher_transport
            if crct_teacher_transport is not None
            else crct_maintenance_transport
        )
        crct_local_diagnostics: dict[str, Any] = {
            "rank": int(rank_),
            "is_memory_rank": bool(rank_ in set(crct_memory_ranks)),
            "is_packet_rank": bool(rank_ == int(crct_packet_rank)),
            "is_maintenance_rank": bool(rank_ == int(crct_maintenance_rank)),
            "packet_rank": int(crct_packet_rank),
            "maintenance_rank": int(crct_maintenance_rank),
            "memory_ranks": [int(r) for r in crct_memory_ranks],
            "train_ranks": [int(r) for r in crct_train_ranks],
            "memory_owner": str(crct_memory_owner),
            "grad_sync_group": (
                "all_ranks" if bool(memory_rank_joins_grad) else "train_ranks"
            ),
            "memory_rank_joins_grad": bool(memory_rank_joins_grad),
            "stop_sync_group": (
                "all_ranks"
                if bool(memory_rank_joins_grad)
                else ("local" if is_episodic_rank else "train_ranks")
            ),
            "train_rank_slot_reads": 0,
            "train_rank_slot_writes": 0,
            "teacher_transport_mode": str(crct_teacher_transport_mode),
            "teacher_transport_participant": bool(
                local_crct_transport is not None
            ),
            "teacher_coordinator_rank": 0,
            "teacher_memory_rank": int(crct_packet_rank),
            "teacher_maintenance_rank": int(crct_maintenance_rank),
            "teacher_requests": int(crct_teacher_requests),
            "teacher_payloads": int(crct_teacher_payloads),
            "teacher_fail_open": int(crct_teacher_fail_open),
            "teacher_bypass_steps": int(crct_teacher_bypass_steps),
            "teacher_param_syncs": int(crct_teacher_param_syncs),
            "teacher_param_sync_interval_steps": int(
                crct_teacher_param_sync_interval
            ),
            "teacher_param_sync_seconds_sum": float(
                crct_teacher_param_sync_seconds_sum
            ),
            "teacher_param_sync_seconds_max": float(
                crct_teacher_param_sync_seconds_max
            ),
            "read_price": (
                float(crct_scarcity.read_price)
                if crct_scarcity is not None
                else 0.0
            ),
            "read_rate_ema": (
                float(crct_scarcity.read_rate_ema)
                if crct_scarcity is not None
                else 0.0
            ),
            "plasticity_budget_strength": float(crct_plasticity_budget_strength),
            "plasticity_budget_payloads_applied": int(
                plasticity_budget_payloads_applied
            ),
            "plasticity_budget_payloads_missing": int(
                plasticity_budget_payloads_missing
            ),
            "memory_slots": int(len(getattr(model.outer_model, "_slots", []))),
            "replay_eviction": (
                replay_eviction_loop.diagnostics()
                if replay_eviction_loop is not None
                else {"enabled": False}
            ),
            "gradient_conflict": (
                crct_gradient_conflict.diagnostics()
                if crct_gradient_conflict is not None
                else {"enabled": False}
            ),
            "transport": (
                local_crct_transport.diagnostics()
                if local_crct_transport is not None
                else {
                    "mode": str(crct_teacher_transport_mode),
                    "transport_group": "rank0_memory",
                    "coordinator_rank": 0,
                    "memory_rank": int(crct_packet_rank),
                    "participant": False,
                    "requests_started": 0,
                    "payloads_used": 0,
                    "payloads_scored": 0,
                    "payloads_sent": 0,
                    "payloads_received": 0,
                    "errors": 0,
                    "last_error": "",
                }
            ),
            "maintenance_transport": (
                crct_maintenance_transport.diagnostics()
                if crct_maintenance_transport is not None
                else None
            ),
            "slot_commit_transport": (
                crct_slot_commit_transport.diagnostics()
                if crct_slot_commit_transport is not None
                else None
            ),
        }
        if ddp_active and dist.is_initialized() and all_group is not None:
            gathered_crct: list[dict[str, Any] | None] | None = (
                [None for _ in range(world_size_)] if rank_ == 0 else None
            )
            dist.gather_object(
                crct_local_diagnostics,
                object_gather_list=gathered_crct,
                dst=0,
                group=object_group or all_group,
            )
            if rank_ == 0:
                crct_rank_diagnostics = gathered_crct
        else:
            crct_rank_diagnostics = [crct_local_diagnostics]
        local_online_state: dict[str, Any] = {}
        if replay_eviction_loop is not None and rank_ == int(crct_maintenance_rank):
            local_online_state["replay_eviction"] = replay_eviction_loop.state_dict()
        if rank_ == int(crct_packet_rank):
            packet_cache_state = _crct_packet_cache_eval_state(model)
            if packet_cache_state is not None:
                local_online_state["packet_cache"] = packet_cache_state
        local_online_payload = local_online_state or None
        if ddp_active and dist.is_initialized() and all_group is not None:
            gathered_online: list[dict[str, Any] | None] | None = (
                [None for _ in range(world_size_)] if rank_ == 0 else None
            )
            dist.gather_object(
                local_online_payload,
                object_gather_list=gathered_online,
                dst=0,
                group=object_group or all_group,
            )
            if rank_ == 0 and gathered_online is not None:
                online_eval_state_payload = _merge_online_eval_state_payloads(
                    gathered_online
                )
        elif local_online_payload:
            online_eval_state_payload = local_online_payload

    elapsed_s = time.perf_counter() - start_time
    loss_cpu = torch.stack(losses).cpu() if losses else torch.empty(0)
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    crct_gradient_conflict_summary: dict[str, Any] = (
        crct_gradient_conflict.diagnostics()
        if crct_gradient_conflict is not None
        else {"enabled": False}
    )
    crct_teacher_memory_slots = 0
    crct_transport_summary: dict[str, Any] = {}
    if crct_rank_diagnostics:
        for diag in crct_rank_diagnostics:
            if diag is None:
                continue
            transport = diag.get("transport")
            if isinstance(transport, dict) and bool(
                diag.get("teacher_transport_participant", False)
            ):
                if bool(diag.get("is_packet_rank", False)):
                    role = "memory"
                elif bool(diag.get("is_maintenance_rank", False)):
                    role = "maintenance"
                elif int(diag.get("rank", -1)) == 0:
                    role = "coordinator"
                else:
                    role = f"rank_{int(diag.get('rank', -1))}"
                crct_transport_summary[role] = transport
            maintenance_transport = diag.get("maintenance_transport")
            slot_commit_transport = diag.get("slot_commit_transport")
            if (
                int(diag.get("rank", -1)) == 0
                and isinstance(maintenance_transport, dict)
            ):
                crct_transport_summary["maintenance_coordinator"] = (
                    maintenance_transport
                )
            if isinstance(slot_commit_transport, dict):
                if bool(diag.get("is_packet_rank", False)):
                    crct_transport_summary["slot_commit_packet"] = (
                        slot_commit_transport
                    )
                elif bool(diag.get("is_maintenance_rank", False)):
                    crct_transport_summary["slot_commit_maintenance"] = (
                        slot_commit_transport
                    )
            if diag is not None and bool(diag.get("is_packet_rank", False)):
                crct_gradient_conflict_summary = dict(
                    diag.get("gradient_conflict", crct_gradient_conflict_summary)
                )
                crct_teacher_memory_slots = int(diag.get("memory_slots", 0))
        if "memory" in crct_transport_summary and "coordinator" in crct_transport_summary:
            coord = dict(crct_transport_summary["coordinator"])
            mem = dict(crct_transport_summary["memory"])
            maintenance = dict(crct_transport_summary.get("maintenance") or {})
            maintenance_coord = dict(
                crct_transport_summary.get("maintenance_coordinator") or {}
            )
            slot_commit_packet = dict(
                crct_transport_summary.get("slot_commit_packet") or {}
            )
            slot_commit_maintenance = dict(
                crct_transport_summary.get("slot_commit_maintenance") or {}
            )
            replay_mem = maintenance or mem
            crct_transport_summary["health"] = {
                "mode": str(coord.get("mode", mem.get("mode", ""))),
                "coordinator_errors": int(coord.get("errors", 0)),
                "memory_errors": int(mem.get("errors", 0)),
                "maintenance_errors": int(maintenance.get("errors", 0)),
                "payloads_used": int(coord.get("payloads_used", 0)),
                "payloads_scored": int(mem.get("payloads_scored", 0)),
                "payloads_served": int(mem.get("payloads_served", 0)),
                "payloads_served_approximate": int(
                    mem.get("payloads_served_approximate", 0)
                ),
                "maintenance_payloads_scored": int(
                    maintenance.get("payloads_scored", 0)
                ),
                "payload_lag_steps_mean": float(
                    coord.get("payload_lag_steps_mean", 0.0)
                ),
                "maintenance_replay_ticks": int(
                    maintenance.get("memory_rank_replay_ticks", 0)
                ),
                "maintenance_replay_probes_ingested": int(
                    maintenance.get("memory_rank_replay_probes_ingested", 0)
                ),
                "maintenance_request_ring_full_drops": int(
                    maintenance_coord.get("teacher_shm_request_ring_full_drops", 0)
                ),
                "slot_commit_p2p_available": bool(
                    slot_commit_packet.get("p2p_available", False)
                    or slot_commit_maintenance.get("p2p_available", False)
                ),
                "append_commits_sent": int(
                    slot_commit_packet.get("append_commits_sent", 0)
                ),
                "append_commits_applied": int(
                    slot_commit_maintenance.get("append_commits_applied", 0)
                ),
                "maintenance_commits_sent": int(
                    slot_commit_maintenance.get("maintenance_commits_sent", 0)
                ),
                "maintenance_commits_applied": int(
                    slot_commit_packet.get("maintenance_commits_applied", 0)
                ),
                "slot_commit_drops": int(
                    slot_commit_packet.get("dropped", 0)
                    + slot_commit_maintenance.get("dropped", 0)
                ),
                "slot_commit_stale_generation_drops": int(
                    slot_commit_packet.get("stale_generation_drops", 0)
                    + slot_commit_maintenance.get("stale_generation_drops", 0)
                ),
                "slot_commit_replica_capacity_full_drops": int(
                    slot_commit_packet.get("replica_capacity_full_drops", 0)
                    + slot_commit_maintenance.get("replica_capacity_full_drops", 0)
                ),
                "slot_commit_queue_overwrites": int(
                    slot_commit_packet.get("queue_overwrites", 0)
                    + slot_commit_maintenance.get("queue_overwrites", 0)
                ),
                "payload_lag_steps_max": int(coord.get("payload_lag_steps_max", 0)),
                "score_seconds_max": float(mem.get("score_seconds_max", 0.0)),
                "crct_loss_reweight_samples": int(
                    mem.get("crct_loss_reweight_samples", 0)
                ),
                "crct_loss_reweight_valid_tokens_sum": int(
                    mem.get("crct_loss_reweight_valid_tokens_sum", 0)
                ),
                "crct_loss_reweight_plain_nll_mean": float(
                    mem.get("crct_loss_reweight_plain_nll_mean", 0.0)
                ),
                "crct_loss_reweight_weighted_nll_mean": float(
                    mem.get("crct_loss_reweight_weighted_nll_mean", 0.0)
                ),
                "crct_loss_reweight_delta_mean": float(
                    mem.get("crct_loss_reweight_delta_mean", 0.0)
                ),
                "crct_loss_reweight_rel_delta_mean": float(
                    mem.get("crct_loss_reweight_rel_delta_mean", 0.0)
                ),
                "crct_loss_weight_abs_dev_mean": float(
                    mem.get("crct_loss_weight_abs_dev_mean", 0.0)
                ),
                "crct_loss_weight_std_mean": float(
                    mem.get("crct_loss_weight_std_mean", 0.0)
                ),
                "crct_loss_weight_max": float(
                    mem.get("crct_loss_weight_max", 0.0)
                ),
                "packet_service_seconds_max": float(
                    mem.get("packet_service_seconds_max", 0.0)
                ),
                "packet_service_seconds_mean": float(
                    mem.get("packet_service_seconds_mean", 0.0)
                ),
                "packet_service_source_count_mean": float(
                    mem.get("packet_service_source_count_mean", 0.0)
                ),
                "packet_service_zero_source_packets": int(
                    mem.get("packet_service_zero_source_packets", 0)
                ),
                "packet_service_approx_write_records": int(
                    mem.get("packet_service_approx_write_records", 0)
                ),
                "memory_packet_lag_steps_mean": float(
                    mem.get("memory_packet_lag_steps_mean", 0.0)
                ),
                "memory_packet_lag_steps_max": int(
                    mem.get("memory_packet_lag_steps_max", 0)
                ),
                "score_stage_timing_enabled": bool(
                    mem.get("score_stage_timing_enabled", False)
                ),
                "score_stage_samples": int(mem.get("score_stage_samples", 0)),
                "score_stage_encode_off_seconds_sum": float(
                    mem.get("score_stage_encode_off_seconds_sum", 0.0)
                ),
                "score_stage_encode_off_seconds_max": float(
                    mem.get("score_stage_encode_off_seconds_max", 0.0)
                ),
                "score_stage_encode_force_on_seconds_sum": float(
                    mem.get("score_stage_encode_force_on_seconds_sum", 0.0)
                ),
                "score_stage_encode_force_on_seconds_max": float(
                    mem.get("score_stage_encode_force_on_seconds_max", 0.0)
                ),
                "score_stage_nll_off_seconds_sum": float(
                    mem.get("score_stage_nll_off_seconds_sum", 0.0)
                ),
                "score_stage_nll_off_seconds_max": float(
                    mem.get("score_stage_nll_off_seconds_max", 0.0)
                ),
                "score_stage_nll_mem_seconds_sum": float(
                    mem.get("score_stage_nll_mem_seconds_sum", 0.0)
                ),
                "score_stage_nll_mem_seconds_max": float(
                    mem.get("score_stage_nll_mem_seconds_max", 0.0)
                ),
                "score_stage_plasticity_seconds_sum": float(
                    mem.get("score_stage_plasticity_seconds_sum", 0.0)
                ),
                "score_stage_plasticity_seconds_max": float(
                    mem.get("score_stage_plasticity_seconds_max", 0.0)
                ),
                "score_stage_append_memory_seconds_sum": float(
                    mem.get("score_stage_append_memory_seconds_sum", 0.0)
                ),
                "score_stage_append_memory_seconds_max": float(
                    mem.get("score_stage_append_memory_seconds_max", 0.0)
                ),
                "score_stage_peak_allocated_mb_max": float(
                    mem.get("score_stage_peak_allocated_mb_max", 0.0)
                ),
                "memory_rank_pump_steps": int(
                    mem.get("memory_rank_pump_steps", 0)
                ),
                "memory_rank_pump_idle_spins": int(
                    mem.get("memory_rank_pump_idle_spins", 0)
                ),
                "memory_rank_pump_idle_yields": int(
                    mem.get("memory_rank_pump_idle_yields", 0)
                ),
                "memory_rank_pump_request_pops": int(
                    mem.get("memory_rank_pump_request_pops", 0)
                ),
                "memory_rank_request_events_superseded": int(
                    mem.get("memory_rank_request_events_superseded", 0)
                ),
                "memory_rank_outer_loop_seconds_sum": float(
                    mem.get("memory_rank_outer_loop_seconds_sum", 0.0)
                ),
                "memory_rank_outer_loop_seconds_max": float(
                    mem.get("memory_rank_outer_loop_seconds_max", 0.0)
                ),
                "memory_rank_pre_pump_seconds_sum": float(
                    mem.get("memory_rank_pre_pump_seconds_sum", 0.0)
                ),
                "memory_rank_pre_pump_seconds_max": float(
                    mem.get("memory_rank_pre_pump_seconds_max", 0.0)
                ),
                "memory_rank_replay_seconds_sum": float(
                    replay_mem.get("memory_rank_replay_seconds_sum", 0.0)
                ),
                "memory_rank_replay_seconds_max": float(
                    replay_mem.get("memory_rank_replay_seconds_max", 0.0)
                ),
                "memory_rank_replay_ticks": int(
                    replay_mem.get("memory_rank_replay_ticks", 0)
                ),
                "memory_rank_replay_probes_ingested": int(
                    replay_mem.get("memory_rank_replay_probes_ingested", 0)
                ),
                "memory_rank_replay_deferred_for_packet_work": int(
                    replay_mem.get("memory_rank_replay_deferred_for_packet_work", 0)
                ),
                "memory_rank_replay_deferred_for_backpressure": int(
                    replay_mem.get(
                        "memory_rank_replay_deferred_for_backpressure", 0
                    )
                ),
                "memory_rank_pump_loop_seconds_sum": float(
                    mem.get("memory_rank_pump_loop_seconds_sum", 0.0)
                ),
                "memory_rank_pump_loop_seconds_max": float(
                    mem.get("memory_rank_pump_loop_seconds_max", 0.0)
                ),
                "memory_rank_pump_score_calls": int(
                    mem.get("memory_rank_pump_score_calls", 0)
                ),
                "mailbox_write_seconds_max": max(
                    float(coord.get("mailbox_write_seconds_max", 0.0)),
                    float(mem.get("mailbox_write_seconds_max", 0.0)),
                ),
                "mailbox_read_seconds_max": max(
                    float(coord.get("mailbox_read_seconds_max", 0.0)),
                    float(mem.get("mailbox_read_seconds_max", 0.0)),
                ),
                "request_write_skipped_busy": int(
                    coord.get("request_write_skipped_busy", 0)
                ),
                "request_stage_seconds_max": float(
                    coord.get("request_stage_seconds_max", 0.0)
                ),
                "request_writer_cpu_copy_seconds_max": float(
                    coord.get("request_writer_cpu_copy_seconds_max", 0.0)
                ),
                "request_host_pinned": bool(coord.get("request_host_pinned", False)),
                "request_host_stage_bytes": int(
                    coord.get("request_host_stage_bytes", 0)
                ),
                "weight_snapshot_published": int(
                    coord.get("weight_snapshot_published", 0)
                ),
                "weight_snapshot_applied": int(
                    mem.get("weight_snapshot_applied", 0)
                ),
                "weight_snapshot_publish_skipped_busy": int(
                    coord.get("weight_snapshot_publish_skipped_busy", 0)
                ),
                "weight_snapshot_latest_overwrites": int(
                    coord.get("weight_snapshot_latest_overwrites", 0)
                ),
                "weight_snapshot_shm_writes": int(
                    coord.get("weight_snapshot_shm_writes", 0)
                ),
                "weight_snapshot_shm_reads": int(
                    mem.get("weight_snapshot_shm_reads", 0)
                ),
                "weight_snapshot_publish_errors": int(
                    coord.get("weight_snapshot_publish_errors", 0)
                ),
                "weight_snapshot_apply_errors": int(
                    mem.get("weight_snapshot_apply_errors", 0)
                ),
                "weight_snapshot_stat_skips": int(
                    mem.get("weight_snapshot_stat_skips", 0)
                ),
                "weight_snapshot_version_lag_steps": int(
                    mem.get("weight_snapshot_version_lag_steps", 0)
                ),
                "weight_snapshot_copy_seconds_max": float(
                    coord.get("weight_snapshot_copy_seconds_max", 0.0)
                ),
                "weight_snapshot_hotpath_cpu_copies": int(
                    coord.get("weight_snapshot_hotpath_cpu_copies", 0)
                ),
                "weight_snapshot_stage_enqueue_seconds_max": float(
                    coord.get("weight_snapshot_stage_enqueue_seconds_max", 0.0)
                ),
                "weight_snapshot_stage_gpu_seconds_max": float(
                    coord.get("weight_snapshot_stage_gpu_seconds_max", 0.0)
                ),
                "weight_snapshot_stage_wait_seconds_max": float(
                    coord.get("weight_snapshot_stage_wait_seconds_max", 0.0)
                ),
                "weight_snapshot_writer_cpu_copy_seconds_max": float(
                    coord.get("weight_snapshot_writer_cpu_copy_seconds_max", 0.0)
                ),
                "weight_snapshot_stage_bytes": int(
                    coord.get("weight_snapshot_stage_bytes", 0)
                ),
                "weight_snapshot_host_pinned_buffers": int(
                    coord.get("weight_snapshot_host_pinned_buffers", 0)
                ),
                "weight_snapshot_host_pinned_bytes": int(
                    coord.get("weight_snapshot_host_pinned_bytes", 0)
                ),
                "weight_snapshot_host_pageable_buffers": int(
                    coord.get("weight_snapshot_host_pageable_buffers", 0)
                ),
                "weight_snapshot_host_pageable_bytes": int(
                    coord.get("weight_snapshot_host_pageable_bytes", 0)
                ),
                "host_pin_memory_failures": int(
                    coord.get("host_pin_memory_failures", 0)
                ),
                "weight_snapshot_save_seconds_max": float(
                    coord.get("weight_snapshot_save_seconds_max", 0.0)
                ),
                "weight_snapshot_read_seconds_sum": float(
                    mem.get("weight_snapshot_read_seconds_sum", 0.0)
                ),
                "weight_snapshot_read_seconds_max": float(
                    mem.get("weight_snapshot_read_seconds_max", 0.0)
                ),
                "weight_snapshot_read_tensor_count": int(
                    mem.get("weight_snapshot_read_tensor_count", 0)
                ),
                "weight_snapshot_read_bytes": int(
                    mem.get("weight_snapshot_read_bytes", 0)
                ),
                "weight_snapshot_apply_seconds_max": float(
                    mem.get("weight_snapshot_apply_seconds_max", 0.0)
                ),
                "weight_snapshot_publisher_busy": bool(
                    coord.get("weight_snapshot_publisher_busy", False)
                ),
                "fast_slow_readiness_scores": int(
                    mem.get("fast_slow_readiness_scores", 0)
                ),
                "fast_slow_readiness_skips_gpu3_mirror": int(
                    mem.get("fast_slow_readiness_skips_gpu3_mirror", 0)
                ),
                "fast_slow_readiness_seconds_max": float(
                    mem.get("fast_slow_readiness_seconds_max", 0.0)
                ),
                "fast_slow_readiness_delta_last": float(
                    mem.get("fast_slow_readiness_delta_last", 0.0)
                ),
                "fast_slow_decisions_issued": int(
                    mem.get("fast_slow_decisions_issued", 0)
                ),
                "fast_slow_decisions_result_payloads": int(
                    mem.get("fast_slow_decisions_result_payloads", 0)
                ),
                "fast_slow_result_decisions_applied": int(
                    coord.get("fast_slow_result_decisions_applied", 0)
                ),
                "fast_slow_slow_mirror_version_lag_steps": int(
                    mem.get("fast_slow_slow_mirror_version_lag_steps", 0)
                ),
                "plasticity_packets_sent": int(
                    mem.get("plasticity_packets_sent", 0)
                ),
                "plasticity_packets_received": int(
                    coord.get("plasticity_packets_received", 0)
                ),
                "plasticity_budget_mean_received": float(
                    coord.get("plasticity_budget_mean_received", 0.0)
                ),
                "plasticity_budget_max_received": float(
                    coord.get("plasticity_budget_max_received", 0.0)
                ),
                "plasticity_confidence_mean_received": float(
                    coord.get("plasticity_confidence_mean_received", 0.0)
                ),
                "plasticity_lag_steps_max": int(
                    coord.get("plasticity_lag_steps_max", 0)
                ),
            }
    timing = summarize_train_timing(
        steps=steps,
        elapsed_s=elapsed_s,
        batch_size=batch_size,
        seq_len=seq_len,
        world_size=world_size_,
        episodic_enabled=bool(episodic_enabled)
        or (bool(crct_enabled) and world_size_ >= 4),
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
            "episodic_async_writes": {
                "enabled": bool(episodic_enabled)
                and bool(episodic_async_write_rings_enabled),
                "cuda_stream_enabled": bool(
                    getattr(
                        episodic_emit_handle,
                        "cuda_write_event_stream_enabled",
                        False,
                    )
                ),
                "publish_drops": int(
                    getattr(episodic_emit_handle, "write_ring_drops", 0)
                ),
                "publish_drop_batches": int(
                    getattr(
                        episodic_emit_handle,
                        "write_ring_drop_batches",
                        0,
                    )
                ),
                "submitted_batches": int(
                    getattr(
                        episodic_emit_handle,
                        "write_ring_submitted_batches",
                        0,
                    )
                ),
                "publisher_error": str(
                    getattr(
                        episodic_emit_handle,
                        "write_ring_publisher_error",
                        "",
                    )
                    or ""
                ),
                "cuda_unavailable_reason": str(
                    getattr(
                        episodic_emit_handle,
                        "cuda_write_event_unavailable_reason",
                        "",
                    )
                    or ""
                ),
                "pushed": int(
                    getattr(episodic_emit_handle, "write_ring_pushed", 0)
                ),
                "skipped": int(
                    getattr(episodic_emit_handle, "write_ring_skipped", 0)
                ),
                "drained": int(
                    getattr(episodic_consumer, "write_ring_events_drained", 0)
                ),
                "attach_misses": int(
                    getattr(episodic_consumer, "write_ring_attach_misses", 0)
                ),
                "max_event_age_steps": int(
                    getattr(episodic_consumer, "write_ring_event_age_max", 0)
                ),
                "drain_heartbeat": int(
                    episodic_write_drain_handle.heartbeat[0]
                    if episodic_write_drain_handle is not None
                    else 0
                ),
                "drain_errors": int(
                    getattr(episodic_consumer, "write_ring_drain_errors", 0)
                ),
                "artifact_impact": "artifact_training_only",
            },
            "crct": {
                "enabled": bool(crct_enabled),
                "memory_owner": str(crct_memory_owner)
                if bool(crct_enabled)
                else "disabled",
                "trunk_memory_mode": "packet" if bool(crct_enabled) else "disabled",
                "grad_sync_group": (
                    "all_ranks"
                    if bool(memory_rank_joins_grad)
                    else "train_ranks"
                ),
                "memory_rank_joins_grad": bool(memory_rank_joins_grad),
                "stop_sync_group": (
                    "all_ranks"
                    if bool(memory_rank_joins_grad)
                    else "train_ranks"
                )
                if bool(crct_enabled)
                else "disabled",
                "train_rank_slot_reads": 0 if bool(crct_enabled) else None,
                "train_rank_slot_writes": 0 if bool(crct_enabled) else None,
                "teacher_memory_slots": int(crct_teacher_memory_slots),
                "lm_weight_alpha_max": float(crct_lm_weight_alpha_max),
                "lm_weight_strength": float(crct_lm_weight_strength),
                "lm_weight_w_max": float(crct_lm_weight_w_max),
                "lm_weight_tau": float(crct_lm_weight_tau),
                "teacher_transport_mode": str(crct_teacher_transport_mode),
                "async_teacher_transport": bool(
                    crct_teacher_transport_mode
                    in {
                        "async_rank0_memory_broadcast",
                        "async_rank0_memory_mailbox",
                    }
                ),
                "teacher_coordinator_rank": 0,
                "teacher_memory_rank": int(crct_packet_rank),
                "teacher_maintenance_rank": int(crct_maintenance_rank),
                "memory_ranks": [int(r) for r in crct_memory_ranks],
                "train_ranks": [int(r) for r in crct_train_ranks],
                "teacher_bypass_steps": int(crct_teacher_bypass_steps),
                "async_teacher_pending_batches": int(
                    crct_async_teacher_pending_batches
                ),
                "async_teacher_max_lag_steps": int(
                    crct_async_teacher_max_lag_steps
                ),
                "async_teacher_payload_dtype": str(
                    crct_async_teacher_payload_dtype
                ),
                "teacher_score_interval_steps": int(
                    crct_teacher_score_interval_steps
                ),
                "teacher_param_sync_interval_steps": int(
                    crct_teacher_param_sync_interval
                ),
                "teacher_param_syncs": int(crct_teacher_param_syncs),
                "teacher_param_sync_seconds_sum": float(
                    crct_teacher_param_sync_seconds_sum
                ),
                "teacher_param_sync_seconds_max": float(
                    crct_teacher_param_sync_seconds_max
                ),
                "teacher_requests": int(crct_teacher_requests),
                "teacher_payloads": int(crct_teacher_payloads),
                "teacher_fail_open": int(crct_teacher_fail_open),
                "plasticity_budget_strength": float(
                    crct_plasticity_budget_strength
                ),
                "plasticity_budget_payloads_applied": int(
                    plasticity_budget_payloads_applied
                ),
                "plasticity_budget_payloads_missing": int(
                    plasticity_budget_payloads_missing
                ),
                "transport_summary": crct_transport_summary,
                "rank_diagnostics": crct_rank_diagnostics,
                "read_price": (
                    float(crct_scarcity.read_price)
                    if crct_scarcity is not None
                    else 0.0
                ),
                "read_rate_ema": (
                    float(crct_scarcity.read_rate_ema)
                    if crct_scarcity is not None
                    else 0.0
                ),
                "memory_write_tokens_per_step": int(
                    crct_memory_write_tokens_per_step
                ),
                "gradient_conflict": crct_gradient_conflict_summary,
                "artifact_impact": "artifact_changes_weights_only",
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
    if online_eval_state_payload:
        result["_online_eval_state"] = online_eval_state_payload
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
        crct_enabled=bool(config.get("crct_enabled", False)),
        crct_lm_weight_alpha_max=float(
            config.get("crct_lm_weight_alpha_max", 0.15)
        ),
        crct_lm_weight_strength=float(config.get("crct_lm_weight_strength", 0.10)),
        crct_lm_weight_w_max=float(config.get("crct_lm_weight_w_max", 1.20)),
        crct_lm_weight_tau=float(config.get("crct_lm_weight_tau", 0.10)),
        crct_target_read_rate=float(config.get("crct_target_read_rate", 0.25)),
        crct_target_write_rate=float(config.get("crct_target_write_rate", 0.10)),
        crct_dual_lr=float(config.get("crct_dual_lr", 0.01)),
        crct_ema_beta=float(config.get("crct_ema_beta", 0.95)),
        crct_max_price=float(config.get("crct_max_price", 0.50)),
        crct_plasticity_budget_strength=float(
            config.get("crct_plasticity_budget_strength", 0.25)
        ),
        crct_memory_write_tokens_per_step=int(
            config.get("crct_memory_write_tokens_per_step", 256)
        ),
        crct_async_teacher_transport=bool(
            config.get("crct_async_teacher_transport", True)
        ),
        crct_async_teacher_transport_backend=str(
            config.get("crct_async_teacher_transport_backend", "collective")
        ),
        crct_teacher_mailbox_dir=str(config.get("crct_teacher_mailbox_dir", "")),
        crct_async_teacher_pending_batches=int(
            config.get("crct_async_teacher_pending_batches", 64)
        ),
        crct_async_teacher_max_lag_steps=int(
            config.get("crct_async_teacher_max_lag_steps", 128)
        ),
        crct_async_teacher_payload_dtype=str(
            config.get("crct_async_teacher_payload_dtype", "auto")
        ),
        crct_teacher_score_interval_steps=int(
            config.get("crct_teacher_score_interval_steps", 1)
        ),
        crct_score_stage_timing_enabled=bool(
            config.get("crct_score_stage_timing_enabled", False)
        ),
        crct_teacher_param_sync_interval_steps=(
            None
            if config.get("crct_teacher_param_sync_interval_steps") is None
            else int(config.get("crct_teacher_param_sync_interval_steps", 1))
        ),
        crct_gradient_conflict_enabled=bool(
            config.get("crct_gradient_conflict_enabled", False)
        ),
        crct_gradient_conflict_ema_beta=float(
            config.get("crct_gradient_conflict_ema_beta", 0.95)
        ),
        crct_gradient_conflict_catastrophic_threshold=float(
            config.get("crct_gradient_conflict_catastrophic_threshold", -0.90)
        ),
        crct_gradient_conflict_soft_gate_strength=float(
            config.get("crct_gradient_conflict_soft_gate_strength", 0.0)
        ),
        crct_gradient_conflict_soft_gate_floor=float(
            config.get("crct_gradient_conflict_soft_gate_floor", 0.05)
        ),
        crct_gradient_conflict_trace_path=str(
            config.get("crct_gradient_conflict_trace_path", "")
        ),
        crct_gradient_conflict_trace_stride=int(
            config.get("crct_gradient_conflict_trace_stride", 1)
        ),
        crct_gradient_conflict_trace_max_rows=int(
            config.get("crct_gradient_conflict_trace_max_rows", 0)
        ),
        crct_gradient_conflict_trace_flush_rows=int(
            config.get("crct_gradient_conflict_trace_flush_rows", 256)
        ),
        replay_eviction_enabled=bool(config.get("replay_eviction_enabled", False)),
        replay_eviction_mode=str(config.get("replay_eviction_mode", "active")),
        replay_eviction_memory_streams=int(config.get("replay_eviction_memory_streams", 8)),
        replay_eviction_threshold=float(config.get("replay_eviction_threshold", 0.01)),
        replay_eviction_ema_beta=float(config.get("replay_eviction_ema_beta", 0.9)),
        replay_eviction_min_age_steps=int(config.get("replay_eviction_min_age_steps", 128)),
        replay_eviction_max_seconds=float(config.get("replay_eviction_max_seconds", 0.5)),
        replay_eviction_trace_path=str(config.get("replay_eviction_trace_path", "")),
        replay_eviction_trace_max_rows=int(config.get("replay_eviction_trace_max_rows", 0)),
        replay_eviction_trace_flush_rows=int(config.get("replay_eviction_trace_flush_rows", 256)),
        replay_eviction_probe_chunk_size=int(config.get("replay_eviction_probe_chunk_size", 16)),
        replay_eviction_scoring_mode=str(config.get("replay_eviction_scoring_mode", "proxy")),
        replay_eviction_oracle_confirm_top_k=int(
            config.get("replay_eviction_oracle_confirm_top_k", 32)
        ),
        replay_eviction_oracle_variant_chunk_size=int(
            config.get("replay_eviction_oracle_variant_chunk_size", 1)
        ),
        replay_eviction_drift_threshold=float(config.get("replay_eviction_drift_threshold", 0.3)),
        replay_eviction_repr_drift_threshold=float(config.get("replay_eviction_repr_drift_threshold", 0.2)),
        replay_eviction_refresh_lr=float(config.get("replay_eviction_refresh_lr", 0.1)),
        replay_eviction_refresh_candidate_count=int(
            config.get("replay_eviction_refresh_candidate_count", 16)
        ),
        replay_eviction_refresh_proposal_rank=int(
            config.get("replay_eviction_refresh_proposal_rank", 8)
        ),
        replay_eviction_refresh_proposal_noise_scale=float(
            config.get("replay_eviction_refresh_proposal_noise_scale", 0.04)
        ),
        replay_eviction_refresh_proposal_momentum=float(
            config.get("replay_eviction_refresh_proposal_momentum", 0.9)
        ),
        replay_eviction_refresh_candidate_variant_chunk_size=int(
            config.get("replay_eviction_refresh_candidate_variant_chunk_size", 16)
        ),
        replay_eviction_refresh_proposal_seed=int(
            config.get("replay_eviction_refresh_proposal_seed", 1729)
        ),
        replay_eviction_controller_state_dim=int(
            config.get("replay_eviction_controller_state_dim", 32)
        ),
        replay_eviction_controller_rank=int(
            config.get("replay_eviction_controller_rank", 8)
        ),
        replay_eviction_controller_dt=float(
            config.get("replay_eviction_controller_dt", 1.0)
        ),
        replay_eviction_controller_gamma=float(
            config.get("replay_eviction_controller_gamma", 0.08)
        ),
        replay_eviction_controller_target_log_sv=float(
            config.get("replay_eviction_controller_target_log_sv", -0.05)
        ),
        replay_eviction_controller_max_state_norm=float(
            config.get("replay_eviction_controller_max_state_norm", 8.0)
        ),
        replay_eviction_controller_perturbation_scale=float(
            config.get("replay_eviction_controller_perturbation_scale", 0.25)
        ),
        replay_eviction_controller_feedback_lr=float(
            config.get("replay_eviction_controller_feedback_lr", 0.05)
        ),
        replay_eviction_quarantine_threshold=float(config.get("replay_eviction_quarantine_threshold", -0.01)),
        replay_eviction_max_quarantined=int(config.get("replay_eviction_max_quarantined", 8)),
        replay_eviction_distill_peak_threshold=float(config.get("replay_eviction_distill_peak_threshold", 0.04)),
        replay_eviction_peak_preserve_utility_threshold=float(
            config.get("replay_eviction_peak_preserve_utility_threshold", 0.20)
        ),
        replay_eviction_peak_preserve_sharpness_threshold=float(
            config.get("replay_eviction_peak_preserve_sharpness_threshold", 0.20)
        ),
        replay_eviction_useful_threshold=float(config.get("replay_eviction_useful_threshold", 0.005)),
        replay_eviction_probe_buffer_size=int(config.get("replay_eviction_probe_buffer_size", 32)),
        replay_eviction_frame_ttl_steps=int(config.get("replay_eviction_frame_ttl_steps", 256)),
        replay_eviction_slot_work_chunk_size=int(config.get("replay_eviction_slot_work_chunk_size", 64)),
        replay_eviction_action_agreement_count=int(config.get("replay_eviction_action_agreement_count", 2)),
        replay_eviction_commit_policy=str(config.get("replay_eviction_commit_policy", "learned")),
        replay_eviction_commit_online_lr=float(config.get("replay_eviction_commit_online_lr", 0.05)),
        replay_eviction_commit_temperature=float(
            config.get("replay_eviction_commit_temperature", 0.75)
        ),
        replay_eviction_arm_runtime_enabled=bool(
            config.get("replay_eviction_arm_runtime_enabled", False)
        ),
        replay_eviction_arm_runtime_namespace=str(
            config.get("replay_eviction_arm_runtime_namespace", "")
        ),
        replay_eviction_evidence_engine_enabled=bool(
            config.get("replay_eviction_evidence_engine_enabled", False)
        ),
        replay_eviction_evidence_engine_d_model=int(
            config.get("replay_eviction_evidence_engine_d_model", 384)
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
    val_cache_dir: str | None = None,
) -> dict[str, Any]:
    _enable_runner_stack_dump_signal()
    rank, world_size, local_rank = _init_distributed(world_size_override)
    is_rank0 = rank == 0
    ddp_active = world_size > 1
    control_group = (
        dist.new_group(list(range(world_size)), backend="gloo")
        if ddp_active
        else None
    )
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
    eval_only = bool(config.get("eval_only", False))
    calc_types_requested = list(config.get("calc_types") or [])
    if eval_only:
        if bool(config.get("rare_bucket_ce_enabled", False)):
            raise ValueError(
                "eval_only=True cannot derive rare_bucket_ce token frequencies "
                "without loading the training stream"
            )
        train_tokens = torch.empty(0, dtype=torch.int16)
        if calc_types_requested:
            val_tokens = torch.empty(0, dtype=torch.int16)
        else:
            val_tokens = load_fineweb_val_tokens(data_path)
    else:
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
    _reject_unsupported_fast_step(
        model, crct_enabled=bool(config.get("crct_enabled", False))
    )
    model_params = sum(p.numel() for p in model.parameters())
    checkpoint_load_metadata: dict[str, Any] | None = None
    checkpoint_online_eval_state: dict[str, Any] = {}
    checkpoint_path_value = str(config.get("checkpoint_path", "") or "").strip()
    if checkpoint_path_value:
        checkpoint_load_metadata = _load_checkpoint_into_model_for_runner(
            model,
            checkpoint_path_value,
        )
        raw_online_state = checkpoint_load_metadata.get("online_eval_state")
        if isinstance(raw_online_state, dict):
            checkpoint_online_eval_state = raw_online_state
    elif eval_only:
        raise ValueError("eval_only=True requires config.checkpoint_path")

    if is_rank0:
        print(
            f"[rank 0/{world_size}] {config.get('name', '<unnamed>')} "
            f"vocab={config['vocab_size']} batch={batch_size} "
            f"chunk={config.get('chunk_size', 64)} "
            f"ckpt={bool(config.get('activation_checkpoint', False))} "
            f"params={model_params:,}",
            flush=True,
        )
        if checkpoint_load_metadata is not None:
            print(
                f"[rank 0/{world_size}] loaded checkpoint "
                f"{checkpoint_load_metadata['checkpoint_path']} "
                f"eval_only={eval_only}",
                flush=True,
            )

    saved_state = None
    if (
        not eval_only
        and bool(config.get("restore_after_warmup", False))
        and int(config.get("warmup_steps", 0)) > 0
    ):
        saved_state = _state_dict_clone(model)
    optimizer = None if eval_only else _build_optimizer(config, model)
    if not eval_only:
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

    crct_teacher_mailbox_dir = str(config.get("crct_teacher_mailbox_dir", "") or "")
    if (
        bool(config.get("crct_enabled", False))
        and str(config.get("crct_async_teacher_transport_backend", "")).strip().lower()
        in {"mailbox", "file", "file_mailbox"}
        and not crct_teacher_mailbox_dir
    ):
        stem = (
            Path(output_json).stem
            if output_json
            else str(config.get("name", "crct_run"))
        )
        crct_teacher_mailbox_dir = str(Path("/dev/shm") / "chaoscontrol_crct" / stem)

    train_fast_impl = (
        _train_fast_eval_only_result if eval_only else train_fast_for_budget
    )
    train_result = train_fast_impl(
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
        episodic_async_write_rings_enabled=bool(
            config.get("episodic_async_write_rings_enabled", True)
        ),
        episodic_cuda_write_event_stream_enabled=bool(
            config.get("episodic_cuda_write_event_stream_enabled", True)
        ),
        episodic_cuda_write_event_stage_depth=int(
            config.get("episodic_cuda_write_event_stage_depth", 4)
        ),
        episodic_event_ring_id=(
            str(config["episodic_event_ring_id"])
            if config.get("episodic_event_ring_id") is not None
            else None
        ),
        episodic_write_ring_max_drain_per_step=int(
            config.get("episodic_write_ring_max_drain_per_step", 4096)
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
        episodic_controller_simplex_trace_path=str(
            config.get("episodic_controller_simplex_trace_path", "") or ""
        ),
        episodic_controller_history_entries=int(
            config.get("episodic_controller_history_entries", 64)
        ),
        episodic_controller_action_space_enabled=bool(
            config.get("episodic_controller_action_space_enabled", False)
        ),
        episodic_controller_action_space_trace_only=bool(
            config.get("episodic_controller_action_space_trace_only", False)
        ),
        episodic_controller_selection_readiness=float(
            config.get("episodic_controller_selection_readiness", 0.0)
        ),
        episodic_controller_selection_max_delta=float(
            config.get("episodic_controller_selection_max_delta", 0.0)
        ),
        episodic_controller_max_tags_per_query=(
            int(config["episodic_controller_max_tags_per_query"])
            if config.get("episodic_controller_max_tags_per_query") is not None
            else None
        ),
        episodic_controller_action_trace_path=(
            str(config["episodic_controller_action_trace_path"])
            if config.get("episodic_controller_action_trace_path") is not None
            else None
        ),
        episodic_controller_shared_event_ssm_enabled=bool(
            config.get("episodic_controller_shared_event_ssm_enabled", True)
        ),
        episodic_controller_ssm_hidden_dim=int(
            config.get("episodic_controller_ssm_hidden_dim", 16)
        ),
        episodic_controller_ssm_seed=int(
            config.get("episodic_controller_ssm_seed", 0)
        ),
        episodic_controller_ssm_decay=float(
            config.get("episodic_controller_ssm_decay", 0.95)
        ),
        episodic_controller_ssm_input_scale=float(
            config.get("episodic_controller_ssm_input_scale", 0.05)
        ),
        episodic_controller_ssm_head_scale=float(
            config.get("episodic_controller_ssm_head_scale", 0.05)
        ),
        episodic_controller_head_readiness=_head_table_from_config(
            config, field="readiness"
        ),
        episodic_controller_head_max_delta=_head_table_from_config(
            config, field="max_delta"
        ),
        episodic_controller_action_learning_rate=float(
            config.get("episodic_controller_action_learning_rate", 0.0)
        ),
        episodic_controller_action_reward_clip=float(
            config.get("episodic_controller_action_reward_clip", 5.0)
        ),
        episodic_replay_max_replays_per_step=int(
            config.get("episodic_replay_max_replays_per_step", 0)
        ),
        crct_enabled=bool(config.get("crct_enabled", False)),
        crct_lm_weight_alpha_max=float(
            config.get("crct_lm_weight_alpha_max", 0.15)
        ),
        crct_lm_weight_strength=float(config.get("crct_lm_weight_strength", 0.10)),
        crct_lm_weight_w_max=float(config.get("crct_lm_weight_w_max", 1.20)),
        crct_lm_weight_tau=float(config.get("crct_lm_weight_tau", 0.10)),
        crct_target_read_rate=float(config.get("crct_target_read_rate", 0.25)),
        crct_target_write_rate=float(config.get("crct_target_write_rate", 0.10)),
        crct_dual_lr=float(config.get("crct_dual_lr", 0.01)),
        crct_ema_beta=float(config.get("crct_ema_beta", 0.95)),
        crct_max_price=float(config.get("crct_max_price", 0.50)),
        crct_plasticity_budget_strength=float(
            config.get("crct_plasticity_budget_strength", 0.25)
        ),
        crct_memory_write_tokens_per_step=int(
            config.get("crct_memory_write_tokens_per_step", 256)
        ),
        crct_async_teacher_transport=bool(
            config.get("crct_async_teacher_transport", True)
        ),
        crct_async_teacher_transport_backend=str(
            config.get("crct_async_teacher_transport_backend", "collective")
        ),
        crct_teacher_mailbox_dir=crct_teacher_mailbox_dir,
        crct_async_teacher_pending_batches=int(
            config.get("crct_async_teacher_pending_batches", 64)
        ),
        crct_async_teacher_max_lag_steps=int(
            config.get("crct_async_teacher_max_lag_steps", 128)
        ),
        crct_async_teacher_payload_dtype=str(
            config.get("crct_async_teacher_payload_dtype", "auto")
        ),
        crct_teacher_score_interval_steps=int(
            config.get("crct_teacher_score_interval_steps", 1)
        ),
        crct_score_stage_timing_enabled=bool(
            config.get("crct_score_stage_timing_enabled", False)
        ),
        crct_teacher_param_sync_interval_steps=(
            None
            if config.get("crct_teacher_param_sync_interval_steps") is None
            else int(config.get("crct_teacher_param_sync_interval_steps", 1))
        ),
        crct_gradient_conflict_enabled=bool(
            config.get("crct_gradient_conflict_enabled", False)
        ),
        crct_gradient_conflict_ema_beta=float(
            config.get("crct_gradient_conflict_ema_beta", 0.95)
        ),
        crct_gradient_conflict_catastrophic_threshold=float(
            config.get("crct_gradient_conflict_catastrophic_threshold", -0.90)
        ),
        crct_gradient_conflict_soft_gate_strength=float(
            config.get("crct_gradient_conflict_soft_gate_strength", 0.0)
        ),
        crct_gradient_conflict_soft_gate_floor=float(
            config.get("crct_gradient_conflict_soft_gate_floor", 0.05)
        ),
        crct_gradient_conflict_trace_path=str(
            config.get("crct_gradient_conflict_trace_path", "")
        ),
        crct_gradient_conflict_trace_stride=int(
            config.get("crct_gradient_conflict_trace_stride", 1)
        ),
        crct_gradient_conflict_trace_max_rows=int(
            config.get("crct_gradient_conflict_trace_max_rows", 0)
        ),
        crct_gradient_conflict_trace_flush_rows=int(
            config.get("crct_gradient_conflict_trace_flush_rows", 256)
        ),
        replay_eviction_enabled=bool(config.get("replay_eviction_enabled", False)),
        replay_eviction_mode=str(config.get("replay_eviction_mode", "active")),
        replay_eviction_memory_streams=int(config.get("replay_eviction_memory_streams", 8)),
        replay_eviction_threshold=float(config.get("replay_eviction_threshold", 0.01)),
        replay_eviction_ema_beta=float(config.get("replay_eviction_ema_beta", 0.9)),
        replay_eviction_min_age_steps=int(config.get("replay_eviction_min_age_steps", 128)),
        replay_eviction_max_seconds=float(config.get("replay_eviction_max_seconds", 0.5)),
        replay_eviction_trace_path=str(config.get("replay_eviction_trace_path", "")),
        replay_eviction_trace_max_rows=int(config.get("replay_eviction_trace_max_rows", 0)),
        replay_eviction_trace_flush_rows=int(config.get("replay_eviction_trace_flush_rows", 256)),
        replay_eviction_probe_chunk_size=int(config.get("replay_eviction_probe_chunk_size", 16)),
        replay_eviction_scoring_mode=str(config.get("replay_eviction_scoring_mode", "proxy")),
        replay_eviction_oracle_confirm_top_k=int(
            config.get("replay_eviction_oracle_confirm_top_k", 32)
        ),
        replay_eviction_oracle_variant_chunk_size=int(
            config.get("replay_eviction_oracle_variant_chunk_size", 1)
        ),
        replay_eviction_drift_threshold=float(config.get("replay_eviction_drift_threshold", 0.3)),
        replay_eviction_repr_drift_threshold=float(config.get("replay_eviction_repr_drift_threshold", 0.2)),
        replay_eviction_refresh_lr=float(config.get("replay_eviction_refresh_lr", 0.1)),
        replay_eviction_refresh_candidate_count=int(
            config.get("replay_eviction_refresh_candidate_count", 16)
        ),
        replay_eviction_refresh_proposal_rank=int(
            config.get("replay_eviction_refresh_proposal_rank", 8)
        ),
        replay_eviction_refresh_proposal_noise_scale=float(
            config.get("replay_eviction_refresh_proposal_noise_scale", 0.04)
        ),
        replay_eviction_refresh_proposal_momentum=float(
            config.get("replay_eviction_refresh_proposal_momentum", 0.9)
        ),
        replay_eviction_refresh_candidate_variant_chunk_size=int(
            config.get("replay_eviction_refresh_candidate_variant_chunk_size", 16)
        ),
        replay_eviction_refresh_proposal_seed=int(
            config.get("replay_eviction_refresh_proposal_seed", 1729)
        ),
        replay_eviction_controller_state_dim=int(
            config.get("replay_eviction_controller_state_dim", 32)
        ),
        replay_eviction_controller_rank=int(
            config.get("replay_eviction_controller_rank", 8)
        ),
        replay_eviction_controller_dt=float(
            config.get("replay_eviction_controller_dt", 1.0)
        ),
        replay_eviction_controller_gamma=float(
            config.get("replay_eviction_controller_gamma", 0.08)
        ),
        replay_eviction_controller_target_log_sv=float(
            config.get("replay_eviction_controller_target_log_sv", -0.05)
        ),
        replay_eviction_controller_max_state_norm=float(
            config.get("replay_eviction_controller_max_state_norm", 8.0)
        ),
        replay_eviction_controller_perturbation_scale=float(
            config.get("replay_eviction_controller_perturbation_scale", 0.25)
        ),
        replay_eviction_controller_feedback_lr=float(
            config.get("replay_eviction_controller_feedback_lr", 0.05)
        ),
        replay_eviction_quarantine_threshold=float(config.get("replay_eviction_quarantine_threshold", -0.01)),
        replay_eviction_max_quarantined=int(config.get("replay_eviction_max_quarantined", 8)),
        replay_eviction_distill_peak_threshold=float(config.get("replay_eviction_distill_peak_threshold", 0.04)),
        replay_eviction_peak_preserve_utility_threshold=float(
            config.get("replay_eviction_peak_preserve_utility_threshold", 0.20)
        ),
        replay_eviction_peak_preserve_sharpness_threshold=float(
            config.get("replay_eviction_peak_preserve_sharpness_threshold", 0.20)
        ),
        replay_eviction_useful_threshold=float(config.get("replay_eviction_useful_threshold", 0.005)),
        replay_eviction_probe_buffer_size=int(config.get("replay_eviction_probe_buffer_size", 32)),
        replay_eviction_frame_ttl_steps=int(config.get("replay_eviction_frame_ttl_steps", 256)),
        replay_eviction_slot_work_chunk_size=int(config.get("replay_eviction_slot_work_chunk_size", 64)),
        replay_eviction_action_agreement_count=int(config.get("replay_eviction_action_agreement_count", 2)),
        replay_eviction_commit_policy=str(config.get("replay_eviction_commit_policy", "learned")),
        replay_eviction_commit_online_lr=float(config.get("replay_eviction_commit_online_lr", 0.05)),
        replay_eviction_commit_temperature=float(
            config.get("replay_eviction_commit_temperature", 0.75)
        ),
        replay_eviction_arm_runtime_enabled=bool(
            config.get("replay_eviction_arm_runtime_enabled", False)
        ),
        replay_eviction_arm_runtime_namespace=str(
            config.get("replay_eviction_arm_runtime_namespace", "")
        ),
        replay_eviction_evidence_engine_enabled=bool(
            config.get("replay_eviction_evidence_engine_enabled", False)
        ),
        replay_eviction_evidence_engine_d_model=int(
            config.get("replay_eviction_evidence_engine_d_model", 384)
        ),
    )
    if checkpoint_load_metadata is not None:
        train_result["checkpoint_path"] = str(
            checkpoint_load_metadata["checkpoint_path"]
        )
        train_result["checkpoint_keys"] = list(
            checkpoint_load_metadata.get("checkpoint_keys", [])
        )
        if checkpoint_online_eval_state and "_online_eval_state" not in train_result:
            train_result["_online_eval_state"] = checkpoint_online_eval_state
    episodic_cache_payload = train_result.pop("_episodic_cache_payload", None)
    online_eval_state_payload = train_result.pop("_online_eval_state", None)
    if isinstance(online_eval_state_payload, dict) and online_eval_state_payload:
        model._online_eval_state = online_eval_state_payload
        applied_slots = _apply_crct_packet_cache_eval_state(
            model,
            online_eval_state_payload.get("packet_cache"),
        )
        if applied_slots > 0:
            train_result["online_eval_packet_cache_slots"] = int(applied_slots)
        train_result["online_eval_state_keys"] = sorted(
            str(k) for k in online_eval_state_payload.keys()
        )

    if ddp_active:
        _control_barrier(group=control_group, label="post_train_eval_state")

    eval_result: dict[str, Any] = {}
    if is_rank0 and (eval_batches > 0 or calc_types_requested):
        val_cache = None
        if calc_types_requested:
            if not val_cache_dir:
                raise ValueError(
                    "config sets calc_types but --val-cache-dir was not "
                    "provided to the runner"
                )
            val_cache = load_val_cache(Path(val_cache_dir))
        def _do_eval():
            return dispatch_eval_for_config(
                config,
                model=model,
                val_cache=val_cache,
                val_tokens=val_tokens,
                eval_starts=eval_starts,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
                legacy_evaluate_fn=evaluate_bpb_sp,
            )

        eval_result = eval_with_ema(
            model,
            getattr(optimizer, "_weight_ema", None) if optimizer is not None else None,
            _do_eval,
        )

    if ddp_active:
        _control_barrier(group=control_group, label="post_eval")

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
                online_eval_state=online_eval_state_payload,
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
    parser.add_argument(
        "--val-cache-dir",
        default=None,
        help="Path to a ValCache directory (tokens.npy/docs.npy/manifest.json). "
        "Required when the config sets calc_types.",
    )
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
        val_cache_dir=args.val_cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
