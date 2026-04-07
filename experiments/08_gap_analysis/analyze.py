#!/usr/bin/env python3
"""Analyze results for experiment 08 — Gap Analysis."""
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

    # Per-layer criticality analysis (dynamic_crit_per_layer)
    for name, r in sorted(results.items()):
        snapshots = r.get("train", {}).get("spectral_snapshots", [])
        has_per_layer = any("per_layer_lambda_max" in s for s in snapshots)
        if has_per_layer:
            print(f"\n{'=' * 66}")
            print(f"Per-layer criticality: {name}")
            print(f"{'=' * 66}")
            first = next(s for s in snapshots if "per_layer_lambda_max" in s)
            last = next(s for s in reversed(snapshots) if "per_layer_lambda_max" in s)
            for li in range(len(first["per_layer_lambda_max"])):
                f_val = first["per_layer_lambda_max"][li]
                l_val = last["per_layer_lambda_max"][li]
                print(f"  layer {li}: lambda_max first={f_val:.4f} last={l_val:.4f} delta={l_val-f_val:+.4f}")

    # Semantic vs episodic divergence (semantic_emergence)
    for name, r in sorted(results.items()):
        snapshots = r.get("train", {}).get("spectral_snapshots", [])
        has_semantic = any("semantic_norm" in s for s in snapshots)
        if has_semantic:
            print(f"\n{'=' * 66}")
            print(f"Semantic/episodic divergence: {name}")
            print(f"{'=' * 66}")
            sem_norms = [s["semantic_norm"] for s in snapshots if "semantic_norm" in s]
            epi_norms = [s["episodic_norm"] for s in snapshots if "episodic_norm" in s]
            if sem_norms:
                print(f"  semantic_norm: first={sem_norms[0]:.4f} last={sem_norms[-1]:.4f}")
                print(f"  episodic_norm: first={epi_norms[0]:.4f} last={epi_norms[-1]:.4f}")
                ratio_first = sem_norms[0] / max(epi_norms[0], 1e-8)
                ratio_last = sem_norms[-1] / max(epi_norms[-1], 1e-8)
                print(f"  sem/epi ratio: first={ratio_first:.4f} last={ratio_last:.4f}")


if __name__ == "__main__":
    main()
