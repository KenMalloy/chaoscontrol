#!/usr/bin/env python3
"""Analyze results for experiment 06 — Metabolic Gate."""
import json
import math
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

    # BPB summary table — uses bpb_gated (gate-aware eval) when available
    print(f"{'Config':<30} {'BPB':>8} {'BPB_gate':>8} {'Params':>10} {'Steps':>6} {'Forks':>6} {'Time':>8}")
    print("-" * 82)
    sort_key = lambda x: x[1].get("eval", {}).get("bpb_gated", x[1].get("eval", {}).get("bpb", 999))
    for name, r in sorted(results.items(), key=sort_key):
        ev = r.get("eval", {})
        bpb = ev.get("bpb", float("nan"))
        bpb_gated = ev.get("bpb_gated", float("nan"))
        params = r.get("params", 0)
        steps = r.get("train", {}).get("steps", 0)
        forks = r.get("train", {}).get("fork_count", 0)
        elapsed = r.get("train", {}).get("elapsed_s", 0)
        gated_str = f"{bpb_gated:8.4f}" if not math.isnan(bpb_gated) else "     N/A"
        print(f"{name:<30} {bpb:8.4f} {gated_str} {params:10,} {steps:6} {forks:6} {elapsed:7.1f}s")

    # Fork analysis
    print("\n" + "=" * 74)
    print("Fork Analysis")
    print("=" * 74)
    for name, r in sorted(results.items()):
        history = r.get("train", {}).get("history", [])
        if not history or "threshold" not in history[0]:
            continue

        forked_losses = [h["loss"] for h in history if h.get("forked")]
        normal_losses = [h["loss"] for h in history if not h.get("forked")]
        thresholds = [h["threshold"] for h in history if "threshold" in h]
        fork_count = sum(1 for h in history if h.get("forked"))
        total = len(history)

        print(f"\n{name}:")
        print(f"  fork rate: {fork_count}/{total} ({100*fork_count/max(total,1):.1f}%)")
        if forked_losses:
            print(f"  forked step mean loss:  {sum(forked_losses)/len(forked_losses):.4f}")
        if normal_losses:
            print(f"  normal step mean loss:  {sum(normal_losses)/len(normal_losses):.4f}")
        if thresholds:
            print(f"  threshold: start={thresholds[0]:.4f}  end={thresholds[-1]:.4f}"
                  f"  (delta={thresholds[-1]-thresholds[0]:+.4f})")


if __name__ == "__main__":
    main()
