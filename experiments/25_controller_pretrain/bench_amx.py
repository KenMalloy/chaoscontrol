#!/usr/bin/env python3
"""Benchmark CPU SSM controller kernels for Phase E4.

This script is safe to run on local development hosts. On non-x86/default
builds it records unavailable AMX/AVX kernels rather than fabricating fallback
numbers under accelerated labels. On the Sapphire Rapids pod, rebuild with
``CHAOSCONTROL_CPU_SSM_X86_ACCEL=1`` before running to expose the accelerated
surfaces.
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


def run_benchmark(
    iterations: int = 10_000,
    dim: int = 512,
    warmup: int = 1_000,
) -> dict[str, Any]:
    torch.manual_seed(1337)
    decay = torch.rand(dim, dtype=torch.float32) * 0.1 + 0.9
    x = torch.randn(dim, dtype=torch.float32)
    h = torch.randn(dim, dtype=torch.float32)
    a = torch.randn(16, 32, dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn(32, 16, dtype=torch.float32).to(torch.bfloat16)

    features = _ext.cpu_features()
    result: dict[str, Any] = {
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "cpu_features": features,
        "kernel_available": {
            "avx512_recurrence": bool(_ext.avx512_recurrence_kernel_available()),
            "amx_bf16_matmul": bool(_ext.amx_bf16_kernel_available()),
        },
        "iterations": int(iterations),
        "warmup": int(warmup),
        "dim": int(dim),
        "modes": {},
    }

    h_ref = h.clone()
    result["modes"]["generic_fp32_recurrence"] = _time_us(
        lambda: _reference_recurrence(decay, x, h_ref), iterations, warmup
    )

    if _ext.avx512_recurrence_kernel_available() and _ext.has_avx512f():
        h_avx = h.clone()
        result["modes"]["avx512_recurrence"] = _time_us(
            lambda: _ext.avx512_diagonal_recurrence(decay, x, h_avx),
            iterations,
            warmup,
        )
    else:
        result["modes"]["avx512_recurrence"] = {
            "available": False,
            "reason": "kernel or runtime AVX-512F unavailable",
        }

    if _ext.amx_bf16_kernel_available() and _ext.has_amx_bf16():
        result["modes"]["amx_bf16_matmul_16x32x16"] = _time_us(
            lambda: _ext.amx_bf16_matmul(a, b), iterations, warmup
        )
    else:
        result["modes"]["amx_bf16_matmul_16x32x16"] = {
            "available": False,
            "reason": "kernel or runtime AMX BF16 unavailable",
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/25_controller_pretrain/bench_amx.json"),
    )
    args = parser.parse_args()

    result = run_benchmark(
        iterations=args.iterations,
        dim=args.dim,
        warmup=args.warmup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
