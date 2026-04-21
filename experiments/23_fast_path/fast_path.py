#!/usr/bin/env python3
"""Exp 23 fastest-path helpers.

This module holds the parts of the fast-path experiment that are useful
to test locally: matrix construction, vectorized batch assembly, and
token-accounting summaries. The CUDA/DDP runner imports these helpers
so the expensive path and the tests share one contract.
"""
from __future__ import annotations

import copy
import json
import random
import statistics
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import torch
import yaml


DEFAULT_STAGE_A_SEED = 1337
DEFAULT_STAGE_B_SEEDS = (1337, 2674, 4011, 5348)


def _base_fast_config(*, world_size: int, vocab_size: int) -> dict[str, Any]:
    return {
        "model_type": "ssm",
        "mode": "speed_sweep",
        "world_size": int(world_size),
        "vocab_size": int(vocab_size),
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "eval_batches": 0,
        "a_mode": "diag",
        "base_lr": 0.128,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "optimizer": "muon",
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
        "device": "auto",
        "dtype": "bf16",
        "precision": "bf16",
        "fused_grad_clip": True,
        "fused_muon": True,
        "compile_full_path": False,
        "cuda_graph_mode": "none",
        "cuda_graph_min_total_speedup": 0.05,
        "cuda_graph_max_capture_seconds": 30.0,
        "lm_head_backward_mode": "fused_streaming_v2",
        "lm_head_tile_size": 8192,
        "warmup_steps": 5,
        "restore_after_warmup": False,
        "stop_check_interval": 4,
        "stop_margin_seconds": 2.0,
        "prefetch_batches": True,
        "budget_seconds": 90.0,
    }


def build_stage_a_matrix(
    *,
    seeds: list[int] | tuple[int, ...] = (DEFAULT_STAGE_A_SEED,),
    vocab_sizes: list[int] | tuple[int, ...] = (16384,),
    batch_sizes: list[int] | tuple[int, ...] = (1024, 2048, 4096),
    chunk_sizes: list[int] | tuple[int, ...] = (64, 128, 256, 512),
    activation_checkpoints: list[bool] | tuple[bool, ...] = (True,),
    world_size: int = 8,
    budget_seconds: float = 90.0,
    base_lr: float = 0.128,
) -> list[dict[str, Any]]:
    """Build the Stage A speed-ceiling matrix.

    Stage A is intentionally quality-light: no eval, short budget, one
    seed by default. It exists to discover the fastest stable hot-loop
    envelope before we spend 600s runs on base selection.
    """
    entries: list[dict[str, Any]] = []
    for vocab_size in vocab_sizes:
        base = _base_fast_config(world_size=world_size, vocab_size=vocab_size)
        base["budget_seconds"] = float(budget_seconds)
        base["base_lr"] = float(base_lr)
        for batch_size in batch_sizes:
            for chunk_size in chunk_sizes:
                for activation_checkpoint in activation_checkpoints:
                    ckpt_label = "ckpt" if activation_checkpoint else "nockpt"
                    for seed in seeds:
                        entry = copy.deepcopy(base)
                        entry.update({
                            "name": (
                                f"stageA_v{vocab_size}_b{batch_size}_"
                                f"c{chunk_size}_{ckpt_label}_s{seed}"
                            ),
                            "seed": int(seed),
                            "batch_size": int(batch_size),
                            "chunk_size": int(chunk_size),
                            "activation_checkpoint": bool(activation_checkpoint),
                        })
                        entries.append(entry)
    return entries


def build_stage_b_matrix(
    *,
    speed_config: dict[str, Any],
    seeds: list[int] | tuple[int, ...] = DEFAULT_STAGE_B_SEEDS,
    vocab_sizes: list[int] | tuple[int, ...] = (8192, 16384),
    init_paths: dict[int, dict[str, str]] | None = None,
    world_size: int = 8,
    budget_seconds: float = 600.0,
) -> list[dict[str, Any]]:
    """Build the Stage B base-lock matrix from a Stage A winner config."""
    init_paths = init_paths or {}
    entries: list[dict[str, Any]] = []
    init_order = ("random", "meanstd", "fullcov")
    for vocab_size in vocab_sizes:
        base = _base_fast_config(world_size=world_size, vocab_size=vocab_size)
        base.update(copy.deepcopy(speed_config))
        base.update({
            "mode": "base_lock",
            "world_size": int(world_size),
            "vocab_size": int(vocab_size),
            "eval_batches": int(speed_config.get("stage_b_eval_batches", 16)),
            "budget_seconds": float(budget_seconds),
            "warmup_steps": int(speed_config.get("warmup_steps", 20)),
            "restore_after_warmup": bool(
                speed_config.get("restore_after_warmup", True)
            ),
        })
        for init_name in init_order:
            if init_name == "random":
                path = None
            else:
                vocab_paths = init_paths.get(int(vocab_size), {})
                if init_name not in vocab_paths:
                    continue
                path = vocab_paths[init_name]
            for seed in seeds:
                entry = copy.deepcopy(base)
                entry.update({
                    "name": f"stageB_v{vocab_size}_{init_name}_s{seed}",
                    "seed": int(seed),
                    "embed_init": init_name,
                    "embed_init_path": path,
                })
                entries.append(entry)
    return entries


def batch_from_start_tensor(
    *,
    tokens: torch.Tensor,
    starts: torch.Tensor,
    seq_len: int,
    device: torch.device,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized equivalent of ``chaoscontrol.data.batch_from_starts``.

    The old helper builds a Python list of ``B`` slices and stacks it.
    This creates the whole ``(B, T+1)`` gather index tensor at once,
    which removes Python per-example slicing from the hot loop.
    """
    if starts.ndim != 1:
        raise ValueError(f"starts must be 1D, got shape={tuple(starts.shape)}")
    start_idx = starts.to(device=tokens.device, dtype=torch.long, non_blocking=True)
    offsets = torch.arange(seq_len + 1, dtype=torch.long, device=tokens.device)
    positions = start_idx[:, None] + offsets[None, :]
    flat_positions = positions.reshape(-1)
    batch = torch.index_select(tokens, 0, flat_positions).view(
        starts.numel(), seq_len + 1,
    )
    if vocab_size is not None:
        batch.clamp_(0, int(vocab_size) - 1)
    inputs = batch[:, :-1].to(device=device, dtype=torch.int32, non_blocking=True)
    targets = batch[:, 1:].to(device=device, dtype=torch.long, non_blocking=True)
    return inputs, targets


def steady_state_step_seconds(step_times: list[float] | tuple[float, ...]) -> float:
    """Estimate steady-state per-step wall time from a sequence of measurements.

    The first step of any warmup loop is dominated by one-time overhead
    (cuBLAS algo selection, CUDA module JIT, allocator growth, Inductor
    compile) that the 600 s timed budget pays once, not once per step.
    Averaging the full warmup window and projecting that to 600 s
    double-counts the first-step overhead and makes downstream decisions
    (for example the CUDA graph gate) over-optimistic on per-step
    savings. Dropping the first sample and taking the median of the rest
    is robust to both the first-call spike and a single late allocator
    blip, while preserving every sample that represents honest
    steady-state work.

    Args:
        step_times: per-step elapsed wall times, in seconds, in the order
            they were measured. Must contain at least one sample.

    Returns:
        Float seconds-per-step suitable for projecting to a larger budget.
        With a single sample that sample is returned as-is (no trimming
        possible); with multiple samples the first is dropped and the
        median of the trailing samples is returned.
    """
    if not step_times:
        raise ValueError("step_times must be non-empty")
    if len(step_times) == 1:
        return float(step_times[0])
    trailing = [float(t) for t in step_times[1:]]
    return float(statistics.median(trailing))


def summarize_cuda_graph_gate(
    *,
    budget_seconds: float,
    capture_seconds: float,
    warmup_seconds: float,
    warmup_steps: int,
    eager_step_seconds: float,
    graph_step_seconds: float,
    min_total_speedup: float = 0.05,
    max_capture_seconds: float = 30.0,
) -> dict[str, Any]:
    """Summarize whether CUDA graph capture is worth enabling.

    Capture and warmup are counted inside the same budget as training.
    The gate is intentionally conservative: graph replay must improve total
    projected 600s throughput, not just steady-state step time.
    """
    budget = float(budget_seconds)
    capture = max(0.0, float(capture_seconds))
    warmup = max(0.0, float(warmup_seconds))
    overhead = capture + warmup
    eager_step = float(eager_step_seconds)
    graph_step = float(graph_step_seconds)
    reasons: list[str] = []

    if budget <= 0.0:
        reasons.append("budget_seconds_must_be_positive")
    if eager_step <= 0.0:
        reasons.append("eager_step_seconds_must_be_positive")
    if graph_step <= 0.0:
        reasons.append("graph_step_seconds_must_be_positive")

    projected_eager_steps = 0.0
    projected_graph_steps = 0.0
    projected_total_speedup = float("-inf")
    break_even_steps = float("inf")
    break_even_seconds = float("inf")

    if budget > 0.0 and eager_step > 0.0:
        projected_eager_steps = budget / eager_step
    if budget > overhead and graph_step > 0.0:
        projected_graph_steps = (budget - overhead) / graph_step
    elif overhead >= budget:
        reasons.append("capture_overhead_exhausts_budget")

    if projected_eager_steps > 0.0:
        projected_total_speedup = projected_graph_steps / projected_eager_steps - 1.0

    per_step_savings = eager_step - graph_step
    if per_step_savings <= 0.0:
        reasons.append("graph_step_not_faster")
    else:
        break_even_steps = overhead / per_step_savings
        break_even_seconds = break_even_steps * eager_step

    if capture > float(max_capture_seconds):
        reasons.append("capture_seconds_exceeds_limit")
    if projected_total_speedup < float(min_total_speedup):
        reasons.append("projected_speedup_below_minimum")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "budget_seconds": budget,
        "capture_seconds": capture,
        "warmup_seconds": warmup,
        "warmup_steps": int(warmup_steps),
        "overhead_seconds": overhead,
        "eager_step_seconds": eager_step,
        "graph_step_seconds": graph_step,
        "projected_eager_steps": projected_eager_steps,
        "projected_graph_steps": projected_graph_steps,
        "projected_total_speedup": projected_total_speedup,
        "min_total_speedup": float(min_total_speedup),
        "max_capture_seconds": float(max_capture_seconds),
        "break_even_steps": break_even_steps,
        "break_even_seconds": break_even_seconds,
    }


def _build_prefetched_lm_batch(
    *,
    tokens: torch.Tensor,
    seq_len: int,
    stride: int,
    batch_size: int,
    rank: int,
    world_size: int,
    generator: torch.Generator,
    device: torch.device,
    vocab_size: int | None,
    batch_sampler: Callable[..., torch.Tensor],
    batch_builder: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = batch_sampler(
        num_tokens=int(tokens.numel()),
        seq_len=int(seq_len),
        stride=int(stride),
        batch_size=int(batch_size),
        rank=int(rank),
        world_size=int(world_size),
        generator=generator,
    )
    return batch_builder(
        tokens=tokens,
        starts=starts,
        seq_len=int(seq_len),
        device=device,
        vocab_size=vocab_size,
    )


class Exp23BatchPrefetcher:
    """One-step-ahead LM batch prefetcher for Exp23.

    Call ``prime()`` once before the hot loop. Each ``next()`` returns the
    batch that was prefetched on the previous step and immediately starts
    preparing the following batch.
    """

    def __init__(
        self,
        *,
        tokens: torch.Tensor,
        seq_len: int,
        stride: int,
        batch_size: int,
        rank: int,
        world_size: int,
        device: torch.device,
        generator: torch.Generator,
        vocab_size: int | None = None,
        batch_sampler: Callable[..., torch.Tensor] | None = None,
        batch_builder: Callable[..., tuple[torch.Tensor, torch.Tensor]] = batch_from_start_tensor,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._tokens = tokens
        self._seq_len = int(seq_len)
        self._stride = int(stride)
        self._batch_size = int(batch_size)
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._device = torch.device(device)
        self._generator = generator
        self._vocab_size = None if vocab_size is None else int(vocab_size)
        self._batch_sampler = batch_sampler or sample_sharded_lm_starts
        self._batch_builder = batch_builder
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._owns_executor = executor is None
        self._pending: Future[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._closed = False
        self._lock = threading.Lock()

    def _submit_next(self) -> Future[tuple[torch.Tensor, torch.Tensor]]:
        return self._executor.submit(
            _build_prefetched_lm_batch,
            tokens=self._tokens,
            seq_len=self._seq_len,
            stride=self._stride,
            batch_size=self._batch_size,
            rank=self._rank,
            world_size=self._world_size,
            generator=self._generator,
            device=self._device,
            vocab_size=self._vocab_size,
            batch_sampler=self._batch_sampler,
            batch_builder=self._batch_builder,
        )

    def prime(self) -> "Exp23BatchPrefetcher":
        """Start prefetching the first batch."""
        with self._lock:
            if self._closed:
                raise RuntimeError("prefetcher is closed")
            if self._pending is None:
                self._pending = self._submit_next()
        return self

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current prefetched batch and queue the next one."""
        with self._lock:
            if self._closed:
                raise RuntimeError("prefetcher is closed")
            pending = self._pending
            if pending is None:
                batch = _build_prefetched_lm_batch(
                    tokens=self._tokens,
                    seq_len=self._seq_len,
                    stride=self._stride,
                    batch_size=self._batch_size,
                    rank=self._rank,
                    world_size=self._world_size,
                    generator=self._generator,
                    device=self._device,
                    vocab_size=self._vocab_size,
                    batch_sampler=self._batch_sampler,
                    batch_builder=self._batch_builder,
                )
            else:
                self._pending = None
                batch = pending.result()
            if self._closed:
                return batch
            self._pending = self._submit_next()
        return batch

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = self._pending
            self._pending = None
        if pending is not None:
            pending.cancel()
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "Exp23BatchPrefetcher":
        return self.prime()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def count_lm_starts(num_tokens: int, seq_len: int, stride: int) -> int:
    """Count ``range(0, num_tokens - seq_len - 1, stride)`` without listing it."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    stop = int(num_tokens) - int(seq_len) - 1
    if stop <= 0:
        return 0
    return ((stop - 1) // int(stride)) + 1


def count_sharded_lm_starts(
    *,
    total_starts: int,
    rank: int,
    world_size: int,
) -> int:
    """Count starts with global start-index ``i`` where ``i % world_size == rank``."""
    total_starts = int(total_starts)
    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if rank >= total_starts:
        return 0
    return ((total_starts - 1 - rank) // world_size) + 1


def sample_sharded_lm_starts(
    *,
    num_tokens: int,
    seq_len: int,
    stride: int,
    batch_size: int,
    rank: int,
    world_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample valid LM start offsets for one DDP rank without materializing all starts.

    The eager baseline uses ``build_lm_starts(...)[rank::world_size]`` and then
    samples from that list. For 10B tokens that list is tens of millions of
    Python integers. This helper samples the equivalent sharded start-index
    range directly and maps indices back to byte/token offsets.
    """
    total = count_lm_starts(num_tokens, seq_len, stride)
    sharded = count_sharded_lm_starts(
        total_starts=total,
        rank=rank,
        world_size=world_size,
    )
    if sharded <= 0:
        raise RuntimeError(
            f"rank {rank} has no train starts after world_size={world_size} sharding"
        )
    local_idx = torch.randint(
        low=0,
        high=sharded,
        size=(int(batch_size),),
        generator=generator,
        dtype=torch.long,
    )
    global_idx = int(rank) + local_idx * int(world_size)
    return global_idx * int(stride)


def choose_lm_starts_lazy(
    *,
    num_tokens: int,
    seq_len: int,
    stride: int,
    batch_size: int,
    eval_batches: int,
    seed: int,
) -> list[int]:
    """Choose eval starts without building the full validation-start list."""
    needed = int(batch_size) * int(eval_batches)
    if needed <= 0:
        return []
    total = count_lm_starts(num_tokens, seq_len, stride)
    if total <= 0:
        return []
    if total <= needed:
        return [idx * int(stride) for idx in range(total)]
    rng = random.Random(int(seed))
    return [idx * int(stride) for idx in rng.sample(range(total), needed)]


def summarize_train_timing(
    *,
    steps: int,
    elapsed_s: float,
    batch_size: int,
    seq_len: int,
    world_size: int,
) -> dict[str, float | int]:
    tokens_per_step = int(batch_size) * int(seq_len) * int(world_size)
    aggregate = (int(steps) * tokens_per_step) / max(float(elapsed_s), 1e-9)
    return {
        "tokens_per_step": tokens_per_step,
        "aggregate_tokens_per_sec": float(aggregate),
        "per_gpu_tokens_per_sec": float(aggregate / max(int(world_size), 1)),
    }


def should_stop_training_loop(
    *,
    steps: int,
    elapsed_s: float,
    budget_seconds: float,
    stop_margin_seconds: float,
    max_steps: int | None = None,
) -> bool:
    """Shared stop predicate for budgeted training and fixed-step warmup."""
    if int(steps) <= 0:
        return False
    if max_steps is not None and int(steps) >= int(max_steps):
        return True
    effective_budget = max(0.0, float(budget_seconds) - float(stop_margin_seconds))
    return float(elapsed_s) >= effective_budget


def write_matrix(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, default=str))


def read_speed_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if "config" in data:
        return dict(data["config"])
    return dict(data)
