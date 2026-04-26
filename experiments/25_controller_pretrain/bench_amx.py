#!/usr/bin/env python3
"""Benchmark CPU SSM controller kernels for Phase E4.

Two complementary phases per run:

1. **Isolated kernel timings.** Per-call latency on tile-aligned synthetic
   inputs for ``avx512_diagonal_recurrence``, ``amx_bf16_matmul`` (single
   16x16x32 tile and a tiled 64x32x64 shape mirroring controller pretrain
   in_proj for a 64-sample minibatch), and a generic fp32 reference.

2. **Per-event end-to-end hot path.** Drives ``OnlineLearningController``
   through ``record_replay_selection`` -> ``on_replay_outcome`` for each
   event, which is the actual on-pod online-learning hot path
   (forward matvec + diagonal recurrence + backward outer-product axpy +
   credit attribution + history append). The controller is run twice
   when AVX-512 is available: once with ``set_use_avx512_matops(False)``
   forcing the scalar dispatch, once with True forcing AVX-512. This
   side-by-side comparison validates the per-event speedup claim.

Safe to run on local development hosts. On non-x86 / default builds it
records unavailable AMX/AVX-512 kernels rather than fabricating fallback
numbers under accelerated labels. On the Sapphire Rapids pod, rebuild
with ``CHAOSCONTROL_CPU_SSM_X86_ACCEL=1`` before running to expose the
accelerated surfaces.
"""
from __future__ import annotations

import argparse
import json
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


def _replay_outcome_dict(replay_id: int, gpu_step: int, slot_id: int) -> dict:
    # Mirrors tests/test_online_learning_loop.py::_replay_outcome shape.
    # selection_step is set to gpu_step - 1 so the matching prior
    # record_replay_selection (at gpu_step - 1) becomes the credited entry.
    return {
        "event_type": 3,
        "selected_rank": 0,
        "outcome_status": 0,
        "replay_id": replay_id,
        "gpu_step": gpu_step,
        "query_event_id": 0,
        "source_write_id": 0,
        "slot_id": slot_id,
        "policy_version": 1,
        "selection_step": gpu_step - 1,
        "teacher_score": 0.5,
        "controller_logit": 0.5,
        "ce_before_replay": 4.0,
        "ce_after_replay": 3.5,
        "ce_delta_raw": 0.5,
        "bucket_baseline": 0.0,
        "reward_shaped": 0.5,
        "grad_cos_rare": float("nan"),
        "grad_cos_total": float("nan"),
        "flags": 0,
    }


def _build_seeded_controller(
    *,
    num_slots: int,
    fdim: int,
    gdim: int,
    sdim: int,
    seed: int,
):
    # SGD/EMA cadences set high so the bench measures the
    # forward+backward+history append without paying for an SGD step on
    # every replay (which would amortize differently across paths).
    # Real on-pod cadences are sgd_interval=256, ema_interval=64; the
    # bench is about per-event latency, not per-step.
    rng = torch.Generator().manual_seed(seed)
    controller = _ext.OnlineLearningController(
        num_slots=num_slots,
        max_entries_per_slot=8,
        gamma=0.995,
        gerber_c=0.5,
        learning_rate=1.0e-3,
        sgd_interval=1024,
        ema_alpha=0.25,
        ema_interval=4096,
    )
    w_global_in = (torch.randn(gdim * fdim, generator=rng) * 0.1).tolist()
    w_slot_in = (torch.randn(sdim * fdim, generator=rng) * 0.1).tolist()
    decay_global = (torch.rand(gdim, generator=rng) * 0.1 + 0.9).tolist()
    decay_slot = (torch.rand(sdim, generator=rng) * 0.1 + 0.9).tolist()
    w_global_out = (torch.randn(gdim, generator=rng) * 0.1).tolist()
    w_slot_out = (torch.randn(sdim, generator=rng) * 0.1).tolist()
    controller.initialize_weights(
        feature_dim=fdim,
        global_dim=gdim,
        slot_dim=sdim,
        w_global_in=w_global_in,
        w_slot_in=w_slot_in,
        decay_global=decay_global,
        decay_slot=decay_slot,
        w_global_out=w_global_out,
        w_slot_out=w_slot_out,
        bias=0.0,
    )
    return controller, rng


def _stage_replay_selection(controller, *, slot_id: int, gpu_step: int, fdim: int, gdim: int, sdim: int, rng) -> None:
    # Each replay outcome credits the most recent record_replay_selection
    # for the same slot_id whose selection_step == on_replay_outcome's
    # gpu_step - 1. Stage one selection per outcome.
    features = (torch.randn(fdim, generator=rng) * 0.1).tolist()
    global_state = (torch.randn(gdim, generator=rng) * 0.1).tolist()
    slot_state = (torch.randn(sdim, generator=rng) * 0.1).tolist()
    controller.record_replay_selection(
        slot_id=slot_id,
        gpu_step=gpu_step,
        policy_version=1,
        output_logit=0.5,
        selected_rank=0,
        features=features,
        global_state=global_state,
        slot_state=slot_state,
    )


def _bench_controller_per_event(
    *,
    iterations: int,
    warmup: int,
    fdim: int,
    gdim: int,
    sdim: int,
    use_avx512: bool,
    seed: int,
) -> dict[str, Any]:
    # Each iteration: (1) record selection at gpu_step, (2) emit replay
    # outcome at gpu_step+1 (selection_step = gpu_step). Step 2 triggers
    # accumulate_backward over the entry from step 1.
    num_slots = 8
    controller, rng = _build_seeded_controller(
        num_slots=num_slots, fdim=fdim, gdim=gdim, sdim=sdim, seed=seed,
    )
    controller.set_use_avx512_matops(use_avx512)

    step_counter = [0]

    def one_event() -> None:
        slot_id = step_counter[0] % num_slots
        sel_step = step_counter[0] * 2
        out_step = sel_step + 1
        _stage_replay_selection(
            controller,
            slot_id=slot_id,
            gpu_step=sel_step,
            fdim=fdim,
            gdim=gdim,
            sdim=sdim,
            rng=rng,
        )
        controller.on_replay_outcome(
            _replay_outcome_dict(
                replay_id=step_counter[0],
                gpu_step=out_step,
                slot_id=slot_id,
            )
        )
        step_counter[0] += 1

    timing = _time_us(one_event, iterations, warmup)
    timing["uses_avx512_matops"] = bool(controller.uses_avx512_matops)
    return timing


def run_benchmark(
    iterations: int = 10_000,
    warmup: int = 1_000,
    recurrence_dim: int = 512,
    fdim: int = 64,
    gdim: int = 32,
    sdim: int = 32,
    seed: int = 1337,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    decay = torch.rand(recurrence_dim, dtype=torch.float32) * 0.1 + 0.9
    x = torch.randn(recurrence_dim, dtype=torch.float32)
    h = torch.randn(recurrence_dim, dtype=torch.float32)
    a16 = torch.randn(16, 32, dtype=torch.float32).to(torch.bfloat16)
    b16 = torch.randn(32, 16, dtype=torch.float32).to(torch.bfloat16)
    # Tiled-shape AMX bench: mirrors controller pretrain in_proj for a
    # 64-sample mini-batch (B=64, fdim=64) -> d_global=32. K=64 needs two
    # 32-K tiles, M=64 needs four 16-row tiles, N=32 needs two 16-col
    # tiles -> 16 dst-tile blocks. Stresses the tiling logic, not just
    # the single-tile microkernel.
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
        "controller_dims": {"fdim": fdim, "gdim": gdim, "sdim": sdim},
        "recurrence_dim": int(recurrence_dim),
        "isolated_kernel": {},
        "controller_per_event": {},
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
            iterations,
            warmup,
        )
    else:
        result["isolated_kernel"]["avx512_recurrence"] = {
            "available": False,
            "reason": "kernel or runtime AVX-512F unavailable",
        }

    if amx_kernel and amx_runtime:
        result["isolated_kernel"]["amx_bf16_matmul_16x32x16"] = _time_us(
            lambda: _ext.amx_bf16_matmul(a16, b16), iterations, warmup
        )
        result["isolated_kernel"]["amx_bf16_matmul_64x64x32_tiled"] = _time_us(
            lambda: _ext.amx_bf16_matmul(a_tiled, b_tiled),
            iterations,
            warmup,
        )
    else:
        unavail = {
            "available": False,
            "reason": "kernel or runtime AMX BF16 unavailable",
        }
        result["isolated_kernel"]["amx_bf16_matmul_16x32x16"] = unavail
        result["isolated_kernel"]["amx_bf16_matmul_64x64x32_tiled"] = unavail

    # === Per-event end-to-end controller hot path ===

    result["controller_per_event"]["scalar"] = _bench_controller_per_event(
        iterations=iterations,
        warmup=warmup,
        fdim=fdim,
        gdim=gdim,
        sdim=sdim,
        use_avx512=False,
        seed=seed,
    )

    if avx512_matops_kernel and avx512_runtime:
        result["controller_per_event"]["avx512"] = _bench_controller_per_event(
            iterations=iterations,
            warmup=warmup,
            fdim=fdim,
            gdim=gdim,
            sdim=sdim,
            use_avx512=True,
            seed=seed,
        )
    else:
        result["controller_per_event"]["avx512"] = {
            "available": False,
            "reason": "kernel or runtime AVX-512F unavailable",
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--recurrence-dim", type=int, default=512)
    parser.add_argument("--fdim", type=int, default=64)
    parser.add_argument("--gdim", type=int, default=32)
    parser.add_argument("--sdim", type=int, default=32)
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
        fdim=args.fdim,
        gdim=args.gdim,
        sdim=args.sdim,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
