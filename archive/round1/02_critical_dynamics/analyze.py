#!/usr/bin/env python3
"""Analyze results for experiment 02 — Critical Dynamics."""
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

    # BPB summary table
    print(f"{'Config':<30} {'BPB':>8} {'Params':>10} {'Steps':>6} {'Time':>8}")
    print("-" * 66)
    for name, r in sorted(results.items(), key=lambda x: x[1].get("eval", {}).get("bpb", 999)):
        bpb = r.get("eval", {}).get("bpb", float("nan"))
        params = r.get("params", 0)
        steps = r.get("train", {}).get("steps", 0)
        elapsed = r.get("train", {}).get("elapsed_s", 0)
        print(f"{name:<30} {bpb:8.4f} {params:10,} {steps:6} {elapsed:7.1f}s")

    # Spectral analysis
    print("\n" + "=" * 66)
    print("Spectral Analysis (hidden state FFT)")
    print("=" * 66)
    for name, r in sorted(results.items()):
        snapshots = r.get("train", {}).get("spectral_snapshots", [])
        if not snapshots:
            print(f"\n{name}: no spectral data")
            continue

        # Report dominant frequency evolution and Jacobian stats
        print(f"\n{name}:")
        dom_freqs = [s["dominant_freq"] for s in snapshots]
        print(f"  dominant_freq: {dom_freqs[:5]}{'...' if len(dom_freqs) > 5 else ''}"
              f"  (mean={sum(dom_freqs)/len(dom_freqs):.1f})")

        # Jacobian stats evolution (lambda_max = top log singular value, criticality proxy)
        lam_vals = [s["lambda_max"] for s in snapshots if "lambda_max" in s]
        if lam_vals:
            print(f"  lambda_max (criticality proxy):"
                  f"  first={lam_vals[0]:.4f}  last={lam_vals[-1]:.4f}"
                  f"  mean={sum(lam_vals)/len(lam_vals):.4f}")
            sv_vars = [s["sv_log_var"] for s in snapshots if "sv_log_var" in s]
            if sv_vars:
                print(f"  sv_log_var:    first={sv_vars[0]:.4f}  last={sv_vars[-1]:.4f}"
                      f"  mean={sum(sv_vars)/len(sv_vars):.4f}")

        # Power spectrum shape: is it 1/f (pink noise = near-critical)?
        last_ps = snapshots[-1]["power_spectrum"]
        if len(last_ps) > 4:
            low_power = sum(last_ps[1:4]) / 3  # low freq (skip DC)
            high_power = sum(last_ps[-3:]) / 3  # high freq
            ratio = low_power / max(high_power, 1e-12)
            print(f"  power ratio (low/high): {ratio:.1f}"
                  f"  ({'1/f-like' if ratio > 10 else 'flat' if ratio < 3 else 'mixed'})")


if __name__ == "__main__":
    main()
