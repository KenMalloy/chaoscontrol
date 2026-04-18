#!/usr/bin/env python3
"""Benchmark the hand-written CUDA diag SSM scan kernel.

Phase 1B-4 follow-up to the -63% torch.compile regression. Compares
four backends at the submission regime:

  * ``python``   — sequential Python loop (``_diag_recurrence_inner``)
  * ``compile``  — ``torch.compile(_diag_recurrence_inner, dynamic=False)``
  * ``chunked``  — cumprod+cumsum chunked scan (``_diag_recurrence_chunked``)
  * ``ssm_scan`` — the CUDA kernel in this PR

Usage on a CUDA-capable pod:

    source /workspace/venv/bin/activate
    python benchmarks/bench_ssm_scan.py \\
        --B 1024 --T 512 --D 256 --dtype bf16 \\
        --warmup 10 --iters 50

Reports per backend:
  * ms/step (mean over timed iters after warmup)
  * tok/s   (B * T / mean_ms * 1e3)
  * speedup vs python baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.core import (  # noqa: E402
    _diag_recurrence_chunked,
    _diag_recurrence_inner,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=1024)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument(
        "--dtype", choices=["fp32", "bf16", "fp16"], default="bf16",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["python", "compile", "chunked", "ssm_scan"],
        help="Which backends to time. Useful to skip eager on small T.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to dump a JSON result summary.",
    )
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _make_backends(backend_names: list[str]):
    """Resolve each requested backend name to a callable.

    Skips backends whose extension / toolchain isn't available and
    reports which ones loaded.
    """
    backends: dict[str, callable] = {}

    if "python" in backend_names:
        backends["python"] = _diag_recurrence_inner

    if "compile" in backend_names:
        try:
            backends["compile"] = torch.compile(_diag_recurrence_inner, dynamic=False)
        except Exception as e:  # pragma: no cover
            print(f"[skip] compile backend: {e}")

    if "chunked" in backend_names:
        backends["chunked"] = lambda d, u: _diag_recurrence_chunked(d, u)

    if "ssm_scan" in backend_names:
        try:
            from chaoscontrol.kernels._ssm_scan import ssm_scan_forward
            backends["ssm_scan"] = ssm_scan_forward
        except ImportError as e:
            print(f"[skip] ssm_scan backend: {e}")

    return backends


def _time_backend(
    fn,
    decay: torch.Tensor,
    update: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict:
    """Run ``fn`` warmup+iters times; return timing stats.

    Uses CUDA events for wall-clock on the H100 stream. We don't call
    ``torch.cuda.synchronize`` between iters so fused pipelines overlap
    the next iter's dispatch with the previous iter's kernel — that's
    how the scan is called in training.
    """
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # Warmup.
    for _ in range(warmup):
        _ = fn(decay, update)
    torch.cuda.synchronize()

    # Timed.
    start_evt.record()
    for _ in range(iters):
        _ = fn(decay, update)
    end_evt.record()
    torch.cuda.synchronize()

    total_ms = start_evt.elapsed_time(end_evt)
    mean_ms = total_ms / iters
    return {"mean_ms": mean_ms, "total_ms": total_ms}


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA required")

    device = torch.device("cuda")
    dtype = _dtype_from_name(args.dtype)

    torch.manual_seed(1)
    decay = (torch.rand(args.B, args.T, args.D, device=device) * 0.3 + 0.65).to(dtype)
    update = (torch.randn(args.B, args.T, args.D, device=device) * 0.1).to(dtype)

    print(
        f"Shape: (B={args.B}, T={args.T}, D={args.D})  dtype={args.dtype}  "
        f"warmup={args.warmup}  iters={args.iters}"
    )

    backends = _make_backends(args.backends)
    results: dict[str, dict] = {}
    for name, fn in backends.items():
        print(f"  running {name}…", flush=True)
        try:
            stats = _time_backend(fn, decay, update, args.warmup, args.iters)
        except Exception as e:
            print(f"    [fail] {name}: {e}")
            continue
        tokens_per_iter = args.B * args.T
        tok_per_s = tokens_per_iter / (stats["mean_ms"] / 1e3)
        stats.update({"tok_per_s": tok_per_s})
        results[name] = stats

    # Report.
    print("\nResults:")
    baseline_name = None
    for candidate in ("python", "compile", "chunked", "ssm_scan"):
        if candidate in results:
            baseline_name = candidate
            break
    baseline = results.get(baseline_name, {}).get("mean_ms", None) if baseline_name else None

    for name, r in results.items():
        speedup_vs_baseline = (
            baseline / r["mean_ms"] if (baseline and baseline_name != name) else None
        )
        suffix = (
            f"  {speedup_vs_baseline:.2f}× vs {baseline_name}"
            if speedup_vs_baseline else ""
        )
        print(
            f"  {name:10s}  {r['mean_ms']:8.3f} ms  "
            f"{r['tok_per_s']:>12,.0f} tok/s{suffix}"
        )

    # Dump JSON.
    if args.json_out:
        payload = {
            "shape": {"B": args.B, "T": args.T, "D": args.D},
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "baseline": baseline_name,
            "results": results,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\n  wrote {args.json_out}")


if __name__ == "__main__":
    main()
