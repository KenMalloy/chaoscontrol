#!/usr/bin/env python3
"""Analyze results for experiment 07 — Full System."""
import json
import sys
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

    print(f"{'Config':<30} {'BPB':>8} {'Params':>10} {'Steps':>6} {'Time':>8}")
    print("-" * 66)
    for name, r in sorted(results.items(), key=lambda x: x[1].get("eval", {}).get("bpb", 999)):
        bpb = r.get("eval", {}).get("bpb", float("nan"))
        params = r.get("params", 0)
        steps = r.get("train", {}).get("steps", 0)
        elapsed = r.get("train", {}).get("elapsed_s", 0)
        print(f"{name:<30} {bpb:8.4f} {params:10,} {steps:6} {elapsed:7.1f}s")

if __name__ == "__main__":
    main()
