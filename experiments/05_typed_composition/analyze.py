#!/usr/bin/env python3
"""Analyze results for experiment 05 — Typed Composition."""
import json
from pathlib import Path


def main():
    results_dir = Path(__file__).parent / "results"
    results = {}
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            results[f.stem] = json.load(fh)

    if not results:
        print("No results found. Run the experiment first.")
        return

    # BPB summary table
    print(f"{'Config':<30} {'BPB':>8} {'Params':>10} {'Steps':>6} {'Time':>8}")
    print("-" * 66)
    for name, r in sorted(results.items(), key=lambda x: x[1].get("eval", {}).get("bpb", 999)):
        bpb = r.get("eval", {}).get("bpb", float("nan"))
        params = r.get("params", 0)
        steps = r.get("train", {}).get("steps", 0)
        elapsed = r.get("train", {}).get("elapsed_s", 0)
        print(f"{name:<30} {bpb:8.4f} {params:10,} {steps:6} {elapsed:7.1f}s")

    # Bucket utilization analysis
    print("\n" + "=" * 66)
    print("Bucket Utilization (Wernicke typed composition)")
    print("=" * 66)
    for name, r in sorted(results.items()):
        snapshots = r.get("train", {}).get("bucket_snapshots", [])
        if not snapshots:
            continue

        print(f"\n{name}:")
        # Final snapshot bucket distribution
        last = snapshots[-1]
        counts = last["bucket_counts"]
        active = last["active_buckets"]
        total_tokens = sum(counts)
        print(f"  active buckets: {active}/{len(counts)}")

        # Top buckets by usage
        indexed = sorted(enumerate(counts), key=lambda x: -x[1])
        top = indexed[:5]
        print(f"  top buckets: {', '.join(f'#{i}={c}/{total_tokens}' for i, c in top)}")

        # Utilization evenness (Gini-like): std of normalized counts
        if total_tokens > 0:
            fracs = [c / total_tokens for c in counts]
            mean_frac = 1.0 / len(counts)
            variance = sum((f - mean_frac) ** 2 for f in fracs) / len(fracs)
            print(f"  distribution std: {variance**0.5:.4f} (0=perfectly even)")

        # Evolution: active bucket count over time
        active_over_time = [s["active_buckets"] for s in snapshots]
        if len(active_over_time) > 1:
            print(f"  active buckets over time: {active_over_time[0]} -> {active_over_time[-1]}")


if __name__ == "__main__":
    main()
