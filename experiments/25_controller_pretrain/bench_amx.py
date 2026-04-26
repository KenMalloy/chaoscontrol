#!/usr/bin/env python3
"""Benchmark CPU SSM simplex controller kernels for Phase E4.

Two questions, two phases per run:

1. **Isolated kernel timings.** Per-call latency on tile-aligned synthetic
   inputs for ``avx512_diagonal_recurrence``, ``amx_bf16_matmul`` (single
   16x16x32 microkernel and a 64x64x32 tiled shape), and a generic fp32
   reference recurrence. Bounds the kernel-level perf claim independent
   of any particular call site.

2. **Per-query simplex policy end-to-end.** Drives ``simplex_forward``
   + ``SimplexOnlineLearner.record_simplex_decision`` +
   ``on_replay_outcome`` for each query event — the actual on-pod hot
   path during training. Three sub-modes:
     - ``forward_only`` — just the policy decision (the read path)
     - ``decision_record`` — forward + record_simplex_decision (write path)
     - ``full_replay_event`` — forward + record + replay outcome with
       REINFORCE backward + SGD (full hot path)

Safe to run on local development hosts. On non-x86 / default builds it
records unavailable AMX/AVX-512 kernels rather than fabricating fallback
numbers under accelerated labels. On the Sapphire Rapids pod, rebuild
with ``CHAOSCONTROL_CPU_SSM_X86_ACCEL=1`` before running to expose the
accelerated surfaces.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


N = 16        # simplex vertices
K_V = 16      # vertex feature dim
K_E = 1       # edge feature dim per pair
K_S = 4       # simplex feature dim
H = 32        # hidden dim


def _time_us(fn: Callable[[], None], iterations: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        samples.append((end - start) / 1_000.0)
    samples.sort()
    return {
        "mean_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "p99_us": samples[min(len(samples) - 1, int(0.99 * len(samples)))],
    }


def _reference_recurrence(decay: torch.Tensor, x: torch.Tensor, h: torch.Tensor) -> None:
    h.mul_(decay).add_(x)


def _make_simplex_weights(seed: int):
    g = torch.Generator().manual_seed(seed)
    w = _ext.SimplexWeights()
    w.K_v, w.K_e, w.K_s, w.H, w.N = K_V, K_E, K_S, H, N
    w.W_vp = (torch.randn(K_V, H, generator=g) * 0.1).flatten().tolist()
    w.b_vp = (torch.randn(H, generator=g) * 0.05).tolist()
    w.W_lh = (torch.randn(H, generator=g) * 0.1).tolist()
    w.b_lh = float(torch.randn((), generator=g) * 0.05)
    w.W_sb = (torch.randn(K_S, generator=g) * 0.05).tolist()
    w.alpha = float(torch.randn((), generator=g) * 0.1)
    w.temperature = 1.0
    w.bucket_embed = torch.zeros(8, 8).flatten().tolist()
    return w


def _make_simplex_inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(N, K_V, generator=g)
    raw = torch.randn(N, K_V, generator=g)
    raw_n = torch.nn.functional.normalize(raw, dim=1)
    E = (raw_n @ raw_n.T).clamp(-1.0, 1.0)
    sf = torch.randn(K_S, generator=g) * 0.5
    return (
        V.flatten().tolist(),
        E.flatten().tolist(),
        sf.tolist(),
    )


def _replay_outcome(slot_id: int, gpu_step: int, selection_step: int) -> dict:
    return {
        "event_type": 3,
        "selected_rank": 0,
        "outcome_status": 0,
        "replay_id": 1,
        "gpu_step": gpu_step,
        "query_event_id": 0,
        "source_write_id": 0,
        "slot_id": slot_id,
        "policy_version": 1,
        "selection_step": selection_step,
        "teacher_score": 0.0,
        "controller_logit": 0.0,
        "ce_before_replay": 0.0,
        "ce_after_replay": 0.0,
        "ce_delta_raw": 0.5,
        "bucket_baseline": 0.0,
        "reward_shaped": 0.0,
        "grad_cos_rare": math.nan,
        "grad_cos_total": math.nan,
        "flags": 0,
    }


def _bench_simplex_forward_only(*, iterations: int, warmup: int, seed: int) -> dict[str, float]:
    weights = _make_simplex_weights(seed)
    V, E, sf = _make_simplex_inputs(seed * 2)

    def one_query() -> None:
        _ext.simplex_forward(weights, V, E, sf)

    return _time_us(one_query, iterations, warmup)


def _bench_simplex_decision_record(*, iterations: int, warmup: int, seed: int) -> dict[str, float]:
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=0.995, learning_rate=1e-3,
        sgd_interval=1024, ema_interval=4096,
    )
    learner.initialize_simplex_weights(_make_simplex_weights(seed))
    V, E, sf = _make_simplex_inputs(seed * 2)
    step_counter = [0]
    num_slots = 8

    def one_query() -> None:
        slot = step_counter[0] % num_slots
        fwd = _ext.simplex_forward(learner.fast_weights(), V, E, sf)
        learner.record_simplex_decision(
            chosen_slot_id=slot, gpu_step=step_counter[0], policy_version=1,
            chosen_idx=step_counter[0] % N, p_chosen_decision=fwd.p[step_counter[0] % N],
            V=V, E=E, simplex_features=sf,
        )
        step_counter[0] += 1

    return _time_us(one_query, iterations, warmup)


def _bench_simplex_full_replay_event(*, iterations: int, warmup: int, seed: int) -> dict[str, float]:
    learner = _ext.SimplexOnlineLearner(
        num_slots=8, max_entries_per_slot=4,
        gamma=0.995, learning_rate=1e-3,
        # sgd_interval=1 so every event triggers an SGD apply — measure
        # the worst-case full-event latency, not just the amortized
        # forward+backward.
        sgd_interval=1, ema_interval=4096,
    )
    learner.initialize_simplex_weights(_make_simplex_weights(seed))
    V, E, sf = _make_simplex_inputs(seed * 2)
    step_counter = [0]
    num_slots = 8

    def one_event() -> None:
        slot = step_counter[0] % num_slots
        sel_step = step_counter[0] * 2
        out_step = sel_step + 1
        fwd = _ext.simplex_forward(learner.fast_weights(), V, E, sf)
        learner.record_simplex_decision(
            chosen_slot_id=slot, gpu_step=sel_step, policy_version=1,
            chosen_idx=step_counter[0] % N, p_chosen_decision=fwd.p[step_counter[0] % N],
            V=V, E=E, simplex_features=sf,
        )
        learner.on_replay_outcome(_replay_outcome(slot, out_step, sel_step))
        step_counter[0] += 1

    return _time_us(one_event, iterations, warmup)


def run_benchmark(
    iterations: int = 10_000,
    warmup: int = 1_000,
    recurrence_dim: int = 512,
    seed: int = 1337,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    decay = torch.rand(recurrence_dim, dtype=torch.float32) * 0.1 + 0.9
    x = torch.randn(recurrence_dim, dtype=torch.float32)
    h = torch.randn(recurrence_dim, dtype=torch.float32)
    a16 = torch.randn(16, 32, dtype=torch.float32).to(torch.bfloat16)
    b16 = torch.randn(32, 16, dtype=torch.float32).to(torch.bfloat16)
    a_tiled = torch.randn(64, 64, dtype=torch.float32).to(torch.bfloat16)
    b_tiled = torch.randn(64, 32, dtype=torch.float32).to(torch.bfloat16)

    features = _ext.cpu_features()
    avx512_kernel = bool(_ext.avx512_recurrence_kernel_available())
    avx512_runtime = bool(_ext.has_avx512f())
    avx512_matops_kernel = bool(_ext.avx512_matops_kernel_available())
    amx_kernel = bool(_ext.amx_bf16_kernel_available())
    amx_runtime = bool(_ext.has_amx_bf16())

    result: dict[str, Any] = {
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "cpu_features": features,
        "kernel_available": {
            "avx512_recurrence": avx512_kernel,
            "avx512_matops": avx512_matops_kernel,
            "amx_bf16_matmul": amx_kernel,
        },
        "iterations": int(iterations),
        "warmup": int(warmup),
        "simplex_dims": {
            "N": N, "K_v": K_V, "K_e": K_E, "K_s": K_S, "H": H,
        },
        "recurrence_dim": int(recurrence_dim),
        "isolated_kernel": {},
        "simplex_per_query": {},
    }

    # === Isolated kernel timings ===

    h_ref = h.clone()
    result["isolated_kernel"]["generic_fp32_recurrence"] = _time_us(
        lambda: _reference_recurrence(decay, x, h_ref), iterations, warmup
    )

    if avx512_kernel and avx512_runtime:
        h_avx = h.clone()
        result["isolated_kernel"]["avx512_recurrence"] = _time_us(
            lambda: _ext.avx512_diagonal_recurrence(decay, x, h_avx),
            iterations, warmup,
        )
    else:
        result["isolated_kernel"]["avx512_recurrence"] = {
            "available": False,
            "reason": "kernel or runtime AVX-512F unavailable",
        }

    if amx_kernel and amx_runtime:
        result["isolated_kernel"]["amx_bf16_matmul_16x32x16"] = _time_us(
            lambda: _ext.amx_bf16_matmul(a16, b16), iterations, warmup,
        )
        result["isolated_kernel"]["amx_bf16_matmul_64x64x32_tiled"] = _time_us(
            lambda: _ext.amx_bf16_matmul(a_tiled, b_tiled),
            iterations, warmup,
        )
    else:
        unavail = {
            "available": False,
            "reason": "kernel or runtime AMX BF16 unavailable",
        }
        result["isolated_kernel"]["amx_bf16_matmul_16x32x16"] = unavail
        result["isolated_kernel"]["amx_bf16_matmul_64x64x32_tiled"] = unavail

    # === Per-query simplex policy end-to-end ===

    result["simplex_per_query"]["forward_only"] = _bench_simplex_forward_only(
        iterations=iterations, warmup=warmup, seed=seed,
    )
    result["simplex_per_query"]["decision_record"] = _bench_simplex_decision_record(
        iterations=iterations, warmup=warmup, seed=seed,
    )
    result["simplex_per_query"]["full_replay_event"] = _bench_simplex_full_replay_event(
        iterations=iterations, warmup=warmup, seed=seed,
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--recurrence-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/25_controller_pretrain/bench_amx.json"),
    )
    args = parser.parse_args()

    result = run_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
        recurrence_dim=args.recurrence_dim,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
