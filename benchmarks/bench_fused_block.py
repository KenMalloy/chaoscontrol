#!/usr/bin/env python3
"""Benchmark the fused vs unfused ChaosSSMBlock post-scan hot path.

Exp 18 Test 8: measure the tok/s delta from consolidating the block's
residual + RMSNorm + FF + residual chain into a single torch.compile
fusion region.

Usage:
    python benchmarks/bench_fused_block.py \
        --batch-sizes 32 128 512 --seq-len 512 --dim 512 --ff-mult 2 \
        --warmup 10 --iters 50 --dtype bf16

Do NOT run on a CPU-only dev machine — torch.compile's CPU codegen
is unstable in some venv paths (see chaoscontrol.core_fused for the
fallback behavior). This script is intended for the H100 pod.

Reports:
    - ms/step (mean over `iters` timed iterations after `warmup`)
    - tokens/s (batch * seq_len / step_time)
    - Relative speedup of fused over unfused
    - Backend info (compile vs eager fallback)

The benchmark runs a fresh model per configuration so the torch.compile
cache picks up a stable shape signature. Each configuration uses the
same weights on both variants (via `FusedChaosSSMBlock.from_unfused`)
to ensure the tok/s delta reflects kernel-launch overhead, not
coincidence of initialization.
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

from chaoscontrol.core_fused import FusedChaosSSMBlock, get_post_scan_backend  # noqa: E402
from chaoscontrol.model import ChaosSSMBlock  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--dtype", choices=["fp32", "bf16"], default="bf16",
        help="Autocast / param dtype for the benchmark run.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Override device (default: cuda if available).",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Optional JSON file to write the full results.",
    )
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench_block(
    block: torch.nn.Module,
    *,
    batch: int,
    seq: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    include_backward: bool,
) -> dict[str, float]:
    """Time forward (+ optional backward) passes through a block.

    Args:
        block: the module (ChaosSSMBlock or FusedChaosSSMBlock).
        include_backward: if True, time full forward+backward+step loop.

    Returns:
        dict with ms_per_step, tokens_per_s, peak_vram_gb.
    """
    block = block.to(device=device, dtype=dtype)
    block.train()
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-4)

    def make_input() -> torch.Tensor:
        return torch.randn(batch, seq, dim, device=device, dtype=dtype)

    # Warmup — triggers torch.compile tracing on the first call.
    for _ in range(warmup):
        x = make_input()
        if include_backward:
            optimizer.zero_grad(set_to_none=True)
            y = block(x)
            loss = (y ** 2).mean()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                block(x)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        x = make_input()
        if include_backward:
            optimizer.zero_grad(set_to_none=True)
            y = block(x)
            loss = (y ** 2).mean()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                block(x)
    _sync(device)
    elapsed = time.perf_counter() - t0

    step_s = elapsed / iters
    tokens_per_s = (batch * seq) / step_s
    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return {
        "ms_per_step": step_s * 1000.0,
        "tokens_per_s": tokens_per_s,
        "peak_vram_gb": peak_vram_gb,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    assert_cuda_or_warn = device.type != "cuda"
    if assert_cuda_or_warn:
        print(
            "[warn] running on non-CUDA device; kernel fusion gains will not "
            "appear. This benchmark is designed for the H100 pod.",
            file=sys.stderr,
        )

    print(f"device={device} dtype={dtype} dim={args.dim} seq={args.seq_len}")
    print(f"post_scan backend: {get_post_scan_backend()}")

    results = []
    for batch in args.batch_sizes:
        print(f"\n=== batch_size={batch} ===")

        torch.manual_seed(0)
        base_block = ChaosSSMBlock(dim=args.dim, ff_mult=args.ff_mult, a_mode="diag")
        fused_block = FusedChaosSSMBlock.from_unfused(base_block)

        for mode in ("forward", "train"):
            include_backward = mode == "train"

            base_metrics = _bench_block(
                base_block,
                batch=batch, seq=args.seq_len, dim=args.dim,
                device=device, dtype=dtype,
                warmup=args.warmup, iters=args.iters,
                include_backward=include_backward,
            )
            fused_metrics = _bench_block(
                fused_block,
                batch=batch, seq=args.seq_len, dim=args.dim,
                device=device, dtype=dtype,
                warmup=args.warmup, iters=args.iters,
                include_backward=include_backward,
            )

            tps_base = base_metrics["tokens_per_s"]
            tps_fused = fused_metrics["tokens_per_s"]
            speedup = tps_fused / tps_base if tps_base > 0 else float("nan")

            row = {
                "batch": batch,
                "seq": args.seq_len,
                "mode": mode,
                "base": base_metrics,
                "fused": fused_metrics,
                "speedup_x": speedup,
            }
            results.append(row)
            print(
                f"  {mode:8s}  base: {base_metrics['ms_per_step']:7.3f} ms/step  "
                f"{tps_base/1e3:7.1f} k tok/s  |  "
                f"fused: {fused_metrics['ms_per_step']:7.3f} ms/step  "
                f"{tps_fused/1e3:7.1f} k tok/s  |  "
                f"{speedup:5.3f}x"
            )

    summary = {
        "device": str(device),
        "dtype": args.dtype,
        "dim": args.dim,
        "seq_len": args.seq_len,
        "ff_mult": args.ff_mult,
        "warmup": args.warmup,
        "iters": args.iters,
        "backend": get_post_scan_backend(),
        "results": results,
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return summary


if __name__ == "__main__":
    run(_parse_args())
