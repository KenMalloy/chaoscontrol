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
from chaoscontrol.precision import autocast_context  # noqa: E402
from chaoscontrol.train_ssm import (  # noqa: E402
    _compiled_step_fn,
    _reject_unsupported,
    chunked_lm_head_backward,
    fused_lm_head_backend_for_mode,
    fused_lm_head_backward,
    full_lm_head_backward,
)
from fast_path import (  # noqa: E402
    Exp23BatchPrefetcher,
    batch_from_start_tensor,
    choose_lm_starts_lazy,
    should_stop_training_loop,
    sample_sharded_lm_starts,
    steady_state_step_seconds,
    summarize_cuda_graph_gate,
    summarize_train_timing,
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
    params = list(model.parameters())
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
    raise ValueError(f"unknown optimizer {name!r}")


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
) -> torch.Tensor:
    _reject_unsupported(model)
    with autocast_context(precision, device_type=inputs.device.type):
        if compile_full_path:
            hidden = _compiled_step_fn()(model, inputs)
        else:
            hidden = model.encode(inputs)
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
        )
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
    losses: list[torch.Tensor] = []
    steps = 0
    start_time = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    prefetcher = None
    async_grad_reducer = (
        AsyncGradAllReducer(model, world_size_)
        if ddp_active and grad_allreduce_mode_ == "async_param"
        else None
    )
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
        )

    try:
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

            if prefetcher is not None:
                if steps == 0:
                    prefetcher.prime()
                inputs, targets = prefetcher.next()
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

            optimizer.zero_grad(set_to_none=True)
            if async_grad_reducer is not None:
                async_grad_reducer.reset()
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
            )
            if grad_clip_norm > 0.0:
                if fused_grad_clip:
                    clip_grad_norm_fused(model.parameters(), grad_clip_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            losses.append(loss.detach())
            steps += 1
    finally:
        if prefetcher is not None:
            prefetcher.close()
        if async_grad_reducer is not None:
            async_grad_reducer.close()

    if ddp_active:
        dist.barrier()

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
        "loss_delta": (
            float(loss_cpu[-1] - loss_cpu[0]) if loss_cpu.numel() >= 2 else float("nan")
        ),
        "peak_vram_mb": peak_vram_mb,
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

    result = {
        "config": config,
        "params": model_params,
        "train": train_result,
        "eval": eval_result,
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
