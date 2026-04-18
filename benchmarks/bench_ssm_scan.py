#!/usr/bin/env python3
"""Benchmark the hand-written CUDA diag SSM scan kernel — forward + backward.

Phase 1B-4 + Phase 2 follow-up to the -63% torch.compile regression.
Compares four backends at the submission regime:

  * ``python``   — sequential Python loop (``_diag_recurrence_inner``)
  * ``compile``  — ``torch.compile(_diag_recurrence_inner, dynamic=False)``
  * ``chunked``  — cumprod+cumsum chunked scan (``_diag_recurrence_chunked``)
  * ``ssm_scan`` — the CUDA kernel pair in this PR

Measurement modes (reported per backend):

  * ``fwd``     — forward only, no requires_grad
  * ``bwd``     — backward only (autograd.grad given a saved forward graph)
  * ``fwd+bwd`` — full training step: fresh forward + autograd.grad

Training is dominated by ``fwd+bwd``; that row is the honest headline.
``fwd`` and ``bwd`` are there for diagnostic when a regression shows up
in the headline.

Note: ``fwd+bwd`` is typically slightly more than ``fwd + bwd`` at the
ssm_scan submission shape because the fwd+bwd loop builds a fresh
autograd graph each iter (fresh ``requires_grad_(True)`` leaves,
custom_op tracing). At bf16 B=1024/T=512/D=256 the overhead is ~0.27 ms.
The bwd row, which runs on a retained graph, is therefore a
lower bound on "pure reverse sweep" and the fwd+bwd row is what a
training step actually pays.

Usage on a CUDA-capable pod:

    source /workspace/venv/bin/activate
    python benchmarks/bench_ssm_scan.py \\
        --B 1024 --T 512 --D 256 --dtype bf16 \\
        --warmup 10 --iters 50

Reports per (backend, mode):
  * ms/step (mean over timed iters after warmup)
  * tok/s   (B * T / mean_ms * 1e3)
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
    """Resolve each requested backend to a dict with forward + autograd callables.

    Each backend entry has:
      ``fwd`` — returns state given (decay, update) with requires_grad=False
      ``autograd`` — returns state given (decay, update) with requires_grad=True;
                     for python/compile/chunked this is the same Python function
                     (autograd traces it); for ssm_scan it's the ``ssm_scan``
                     autograd.Function (routes through the kernel backward).

    Skips backends whose extension / toolchain isn't available.
    """
    backends: dict[str, dict] = {}

    if "python" in backend_names:
        backends["python"] = {
            "fwd": _diag_recurrence_inner,
            "autograd": _diag_recurrence_inner,
        }

    if "compile" in backend_names:
        try:
            compiled = torch.compile(_diag_recurrence_inner, dynamic=False)
        except Exception as e:  # pragma: no cover
            print(f"[skip] compile backend: {e}")
        else:
            backends["compile"] = {"fwd": compiled, "autograd": compiled}

    if "chunked" in backend_names:
        backends["chunked"] = {
            "fwd": (lambda d, u: _diag_recurrence_chunked(d, u)),
            "autograd": (lambda d, u: _diag_recurrence_chunked(d, u)),
        }

    if "ssm_scan" in backend_names:
        try:
            from chaoscontrol.kernels._ssm_scan import ssm_scan, ssm_scan_forward
        except ImportError as e:
            print(f"[skip] ssm_scan backend: {e}")
        else:
            backends["ssm_scan"] = {"fwd": ssm_scan_forward, "autograd": ssm_scan}

    return backends


def _time_loop(body, warmup: int, iters: int) -> dict:
    """Time ``body(step_idx)`` over warmup + iters.

    Uses CUDA events for wall-clock. ``body`` runs a single iteration's
    worth of work; caller is responsible for making that iteration
    self-contained (e.g. zeroing grads, detaching saved state).
    """
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for i in range(warmup):
        body(i)
    torch.cuda.synchronize()

    start_evt.record()
    for i in range(iters):
        body(i)
    end_evt.record()
    torch.cuda.synchronize()

    total_ms = start_evt.elapsed_time(end_evt)
    mean_ms = total_ms / iters
    return {"mean_ms": mean_ms, "total_ms": total_ms}


def _time_fwd(fwd_fn, decay: torch.Tensor, update: torch.Tensor,
              warmup: int, iters: int) -> dict:
    """Forward only, no grad."""
    def body(_i):
        with torch.no_grad():
            _ = fwd_fn(decay, update)
    return _time_loop(body, warmup, iters)


def _time_fwd_bwd(autograd_fn, decay: torch.Tensor, update: torch.Tensor,
                  warmup: int, iters: int) -> dict:
    """Fresh forward + backward each iter.

    Builds a new graph each iteration via fresh leaves (decay_leaf,
    update_leaf) to avoid gradient accumulation bias and graph reuse.
    Uses ``torch.autograd.grad`` not ``.backward()`` to skip the
    ``.grad`` accumulation step and keep it apples-to-apples with the
    kernel path — training-loop code that zeros grads every iter has
    the same shape.
    """
    def body(_i):
        d_leaf = decay.detach().requires_grad_(True)
        u_leaf = update.detach().requires_grad_(True)
        y = autograd_fn(d_leaf, u_leaf)
        loss = y.pow(2).sum()
        torch.autograd.grad(loss, [d_leaf, u_leaf])
    return _time_loop(body, warmup, iters)


def _time_bwd(autograd_fn, decay: torch.Tensor, update: torch.Tensor,
              warmup: int, iters: int) -> dict:
    """Backward only — isolate reverse sweep by retaining a single forward graph.

    Run one forward under requires_grad, keep it alive via
    ``retain_graph=True``, then time N calls to ``autograd.grad`` on
    that same graph. This is the cleanest PyTorch-level isolation of
    the reverse sweep — each iter skips the forward recomputation but
    still pays full autograd-machinery + saved-tensor load + reverse
    kernel cost.

    Caveat: for python/compile/chunked paths, autograd uses the
    traced-graph's saved intermediates. For ssm_scan, each iter calls
    into the kernel-backward custom_op. The comparison is still
    apples-to-apples at the "reverse-sweep kernel launch" level.
    """
    d_leaf = decay.detach().requires_grad_(True)
    u_leaf = update.detach().requires_grad_(True)
    y = autograd_fn(d_leaf, u_leaf)
    loss = y.pow(2).sum()

    def body(_i):
        torch.autograd.grad(loss, [d_leaf, u_leaf], retain_graph=True)
    return _time_loop(body, warmup, iters)


_MODE_TIMERS = {
    "fwd": _time_fwd,
    "bwd": _time_bwd,
    "fwd+bwd": _time_fwd_bwd,
}


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
    results: dict[str, dict[str, dict]] = {}
    for name, callables in backends.items():
        results[name] = {}
        for mode in ("fwd", "bwd", "fwd+bwd"):
            print(f"  {name:10s}  {mode:8s}…", flush=True)
            timer = _MODE_TIMERS[mode]
            fn = callables["fwd" if mode == "fwd" else "autograd"]
            try:
                stats = timer(fn, decay, update, args.warmup, args.iters)
            except Exception as e:
                print(f"    [fail] {name} {mode}: {e}")
                results[name][mode] = {"error": str(e)}
                continue
            tokens_per_iter = args.B * args.T
            tok_per_s = tokens_per_iter / (stats["mean_ms"] / 1e3)
            stats.update({"tok_per_s": tok_per_s})
            results[name][mode] = stats

    # Report: a compact (backend × mode) table.
    print("\nResults (ms/iter, tok/s):")
    header = f"  {'backend':10s}  {'fwd':>20s}  {'bwd':>20s}  {'fwd+bwd':>20s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, by_mode in results.items():
        row = f"  {name:10s}  "
        cells = []
        for mode in ("fwd", "bwd", "fwd+bwd"):
            r = by_mode.get(mode, {})
            if "error" in r:
                cells.append(f"{'[err]':>20s}")
            else:
                cells.append(
                    f"{r['mean_ms']:>7.3f} ms {r['tok_per_s']/1e6:>6.2f}M/s"
                )
        row += "  ".join(cells)
        print(row)

    # Honest headline: fwd+bwd is the training-loop shape.
    #
    # Report speedup against BOTH baselines. `compile` is the backend
    # whose -63% regression motivated this kernel — but `compile`'s
    # own backward is broken (OOMs / NaNs at our submission shape),
    # so the compile baseline is pessimistic for any honest compare.
    # `chunked` is the real baseline: it's what the codebase was
    # using before `compile` was tried, and it actually runs correctly
    # end-to-end. Headlines below show both so the reader can pick the
    # framing they trust.
    print("\nHeadline (fwd+bwd, training shape):")
    base_compile = results.get("compile", {}).get("fwd+bwd", {}).get("mean_ms")
    base_chunked = results.get("chunked", {}).get("fwd+bwd", {}).get("mean_ms")
    for name, by_mode in results.items():
        r = by_mode.get("fwd+bwd", {})
        if "error" in r or "mean_ms" not in r:
            continue
        parts: list[str] = []
        if base_compile and name not in ("compile",):
            parts.append(f"{base_compile / r['mean_ms']:.2f}× vs compile")
        if base_chunked and name not in ("chunked",):
            parts.append(f"{base_chunked / r['mean_ms']:.2f}× vs chunked")
        suffix = ("  " + " / ".join(parts)) if parts else ""
        print(
            f"  {name:10s}  {r['mean_ms']:8.3f} ms  "
            f"{r['tok_per_s']:>14,.0f} tok/s{suffix}"
        )
    print(
        "\n  Note: `compile` has a broken backward at submission shape "
        "(the reason this kernel exists); `chunked` is the honest baseline.\n"
        "  Prefer the `vs chunked` speedup for reporting."
    )

    # Dump JSON.
    if args.json_out:
        payload = {
            "shape": {"B": args.B, "T": args.T, "D": args.D},
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "results": results,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\n  wrote {args.json_out}")


if __name__ == "__main__":
    main()
