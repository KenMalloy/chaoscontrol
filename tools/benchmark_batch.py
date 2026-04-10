#!/usr/bin/env python3
"""Batch-size benchmark for ChaosStudentLM.

Sweeps batch sizes, measures step time (forward + backward), and recommends
the largest batch size before throughput starts degrading.

Usage:
    python tools/benchmark_batch.py
    python tools/benchmark_batch.py --batch-sizes 32,64,128 --measure-steps 50
    python tools/benchmark_batch.py --output-json results/batch_bench.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

from chaoscontrol.model import ChaosStudentLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-size benchmark for ChaosStudentLM")
    p.add_argument("--dim", type=int, default=128, help="Model dim (default: 128)")
    p.add_argument("--num-layers", type=int, default=4, help="Number of SSM layers (default: 4)")
    p.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    p.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size (default: 256)")
    p.add_argument("--wernicke-k-max", type=int, default=16, help="Wernicke k_max (default: 16)")
    p.add_argument("--batch-sizes", type=str, default="32,64,128,256,512",
                    help="Comma-separated batch sizes to try (default: 32,64,128,256,512)")
    p.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps per batch size (default: 5)")
    p.add_argument("--measure-steps", type=int, default=20, help="Measured steps per batch size (default: 20)")
    p.add_argument("--output-json", type=str, default=None, help="Path to save JSON results")
    return p.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[device] CUDA — {name}, {mem_gb:.1f} GB")
        return dev
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[device] MPS (Apple Silicon) — timings may differ from datacenter GPUs")
        return torch.device("mps")
    print("[WARNING] No GPU detected — running on CPU. Results are NOT representative "
          "of real training throughput. Use this only to verify the script runs.")
    return torch.device("cpu")


def build_model(
    device: torch.device,
    vocab_size: int,
    dim: int,
    num_layers: int,
    wernicke_k_max: int,
) -> ChaosStudentLM:
    model = ChaosStudentLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        wernicke_enabled=True,
        wernicke_k_max=wernicke_k_max,
        wernicke_router="moe",
        outer_model_dim=64,
        outer_model_type="multislot",
        outer_max_slots=0,
        buffer_mode="append_only",
        retrieval_mode="bucket_topk",
        retrieval_k=4,
    )
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    return model


def run_step(model: ChaosStudentLM, x: torch.Tensor) -> None:
    """Single forward + backward step."""
    out = model(x)
    logits = out["logits"]
    # Shift for next-token prediction loss
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        x[:, 1:].reshape(-1),
    )
    loss.backward()
    # Zero grads so buffers don't accumulate across steps
    model.zero_grad(set_to_none=True)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_batch_size(
    device: torch.device,
    vocab_size: int,
    dim: int,
    num_layers: int,
    seq_len: int,
    wernicke_k_max: int,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
) -> dict:
    """Benchmark a single batch size. Returns results dict or error info."""
    result: dict = {"batch_size": batch_size}
    model = None
    try:
        model = build_model(device, vocab_size, dim, num_layers, wernicke_k_max)
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Warmup
        for _ in range(warmup_steps):
            run_step(model, x)
        sync_device(device)

        # Measure
        step_times: list[float] = []
        for _ in range(measure_steps):
            sync_device(device)
            t0 = time.perf_counter()
            run_step(model, x)
            sync_device(device)
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)  # ms

        mean_ms = statistics.mean(step_times)
        std_ms = statistics.stdev(step_times) if len(step_times) > 1 else 0.0
        tokens_per_sec = (batch_size * seq_len) / (mean_ms / 1000.0)

        result["mean_ms"] = round(mean_ms, 2)
        result["std_ms"] = round(std_ms, 2)
        result["tokens_per_sec"] = round(tokens_per_sec, 0)
        result["oom"] = False

    except torch.cuda.OutOfMemoryError:
        result["mean_ms"] = None
        result["std_ms"] = None
        result["tokens_per_sec"] = None
        result["oom"] = True
        # Free whatever we can
        if device.type == "cuda":
            torch.cuda.empty_cache()

    except RuntimeError as e:
        # MPS and other backends may raise generic RuntimeError on OOM
        if "out of memory" in str(e).lower() or "MPS" in str(e):
            result["mean_ms"] = None
            result["std_ms"] = None
            result["tokens_per_sec"] = None
            result["oom"] = True
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            raise

    finally:
        # Ensure model is freed before next batch size
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return result


def recommend(results: list[dict]) -> int | None:
    """Pick the largest batch size whose step time is within 5% of the previous."""
    valid = [r for r in results if not r["oom"]]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]["batch_size"]
    recommended = valid[0]["batch_size"]
    for i in range(1, len(valid)):
        prev_ms = valid[i - 1]["mean_ms"]
        curr_ms = valid[i]["mean_ms"]
        # Check: did tokens/sec actually improve? And step time didn't blow up?
        if curr_ms <= prev_ms * 1.05:
            recommended = valid[i]["batch_size"]
        else:
            break
    return recommended


def print_table(results: list[dict], recommended_bs: int | None) -> None:
    hdr = f"{'batch':>8}  {'mean_ms':>10}  {'std_ms':>8}  {'tok/sec':>12}  {'status':>8}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        bs = r["batch_size"]
        if r["oom"]:
            print(f"{bs:>8}  {'—':>10}  {'—':>8}  {'—':>12}  {'OOM':>8}")
        else:
            marker = " *" if bs == recommended_bs else ""
            print(f"{bs:>8}  {r['mean_ms']:>10.2f}  {r['std_ms']:>8.2f}  {r['tokens_per_sec']:>12.0f}  {'OK' + marker:>8}")
    print()
    if recommended_bs is not None:
        print(f"Recommended batch size: {recommended_bs}")
    else:
        print("No valid batch size found (all OOM).")


def main() -> None:
    args = parse_args()
    batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]

    device = select_device()
    print(f"[config] dim={args.dim}, layers={args.num_layers}, seq={args.seq_len}, "
          f"vocab={args.vocab_size}, wernicke_k_max={args.wernicke_k_max}")
    print(f"[config] batch_sizes={batch_sizes}, warmup={args.warmup_steps}, measure={args.measure_steps}")
    print()

    results: list[dict] = []
    for bs in batch_sizes:
        print(f"  batch_size={bs} ...", end=" ", flush=True)
        r = benchmark_batch_size(
            device=device,
            vocab_size=args.vocab_size,
            dim=args.dim,
            num_layers=args.num_layers,
            seq_len=args.seq_len,
            wernicke_k_max=args.wernicke_k_max,
            batch_size=bs,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
        results.append(r)
        if r["oom"]:
            print("OOM")
        else:
            print(f"{r['mean_ms']:.2f} ms/step, {r['tokens_per_sec']:.0f} tok/s")

    recommended_bs = recommend(results)
    print_table(results, recommended_bs)

    if args.output_json:
        payload = {
            "config": {
                "dim": args.dim,
                "num_layers": args.num_layers,
                "seq_len": args.seq_len,
                "vocab_size": args.vocab_size,
                "wernicke_k_max": args.wernicke_k_max,
                "warmup_steps": args.warmup_steps,
                "measure_steps": args.measure_steps,
                "device": str(device),
            },
            "results": results,
            "recommended_batch_size": recommended_bs,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
