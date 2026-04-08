#!/usr/bin/env python3
"""Analyze scaling law results from experiment 10.

Reads results/scaling_results.json, fits power-law curves, generates plots,
and prints a summary table with kill criteria evaluation.

Core plots:
  1. scaling_curves.png  -- bpb vs params with power-law fits (log-log)
  2. isoflop_curves.png  -- bpb vs total training FLOPs
  3. component_delta.png -- component ROI (bare_ssm - full_ssm) vs size

Summary includes Welch t-test significance between conditions at each size.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Reference lines from competition
COMP_BASELINE_BPB = 1.2244
COMP_SOTA_BPB = 1.1147

# Import stats utilities from experiment 09 (pure Python, no scipy needed)
_STATS_DIR = Path(__file__).resolve().parents[1] / "09_revised_architecture"
sys.path.insert(0, str(_STATS_DIR))
try:
    from stats import welch_ttest, cohens_d, bootstrap_ci, sem as compute_sem
except ImportError:
    # Minimal fallbacks if stats module is unavailable
    def welch_ttest(a, b):  # type: ignore[misc]
        return (0.0, 1.0)

    def cohens_d(a, b):  # type: ignore[misc]
        return 0.0

    def bootstrap_ci(values, **_kw):  # type: ignore[misc]
        m = statistics.mean(values) if values else 0.0
        return (m, m)

    def compute_sem(values):  # type: ignore[misc]
        if len(values) < 2:
            return 0.0
        return statistics.stdev(values) / math.sqrt(len(values))


# ── Data loading ─────────────────────────────────────────────────────


def load_results() -> dict:
    """Load the combined scaling results JSON."""
    path = RESULTS_DIR / "scaling_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run run_scaling.py first.")
        sys.exit(1)
    return json.loads(path.read_text())


def extract_scaling_data(results: dict) -> dict[str, list[dict]]:
    """Extract (param_count, bpb, total_flops, steps) per condition.

    Returns {condition: [{"params": ..., "bpb": ..., "bpb_std": ...,
                          "bpb_values": [...], "total_flops": ...,
                          "steps": ..., "size": ...}, ...]}.
    """
    data: dict[str, list[dict]] = {}
    for size_name, size_data in results.items():
        if size_name == "comp_tfm":
            continue  # Handled separately
        if not isinstance(size_data, dict):
            continue
        for cond_name, seed_data in size_data.items():
            if not isinstance(seed_data, dict):
                continue
            if cond_name not in data:
                data[cond_name] = []

            bpbs = []
            params_list = []
            flops_list = []
            steps_list = []
            for seed_str, result in seed_data.items():
                if not isinstance(result, dict) or "eval" not in result:
                    continue
                ev = result["eval"]
                bpb = ev.get("bpb_gated", ev["bpb"])
                bpbs.append(bpb)
                params_list.append(result["params"])
                flops_list.append(result.get("total_flops", 0))
                steps_list.append(result["train"]["steps"])

            if not bpbs:
                continue

            data[cond_name].append({
                "size": size_name,
                "params": statistics.mean(params_list),
                "bpb": statistics.mean(bpbs),
                "bpb_std": statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0,
                "bpb_values": bpbs,
                "total_flops": statistics.mean(flops_list),
                "steps": statistics.mean(steps_list),
            })

    # Sort each condition by param count
    for cond in data:
        data[cond].sort(key=lambda x: x["params"])

    return data


# ── Power-law fitting ────────────────────────────────────────────────


def _power_law_residual(params_arr, bpbs_arr, A, alpha, bpb_irr):
    """Compute sum of squared residuals for power-law fit."""
    predicted = A * params_arr ** (-alpha) + bpb_irr
    return ((predicted - bpbs_arr) ** 2).sum()


def fit_power_law(params: list[float], bpbs: list[float]) -> dict:
    """Fit power-law to scaling data: bpb(N) = A * N^(-alpha) + bpb_irr.

    Uses scipy.optimize.curve_fit when available, otherwise falls back
    to a grid search over (A, alpha, bpb_irr).

    Returns {"A": float, "alpha": float, "bpb_irr": float, "r_squared": float}.
    """
    try:
        import numpy as np
    except ImportError:
        return {"A": 0, "alpha": 0, "bpb_irr": 0, "r_squared": 0}

    params_arr = np.array(params, dtype=np.float64)
    bpbs_arr = np.array(bpbs, dtype=np.float64)

    def power_law(N, A, alpha, bpb_irr):
        return A * np.power(N, -alpha) + bpb_irr

    # Try scipy first
    try:
        from scipy.optimize import curve_fit as scipy_curve_fit
        popt, _pcov = scipy_curve_fit(
            power_law, params_arr, bpbs_arr,
            p0=[100.0, 0.5, 1.0],
            bounds=([0, 0, 0], [1e6, 2.0, 5.0]),
            maxfev=10000,
        )
        A, alpha, bpb_irr = popt
    except (ImportError, RuntimeError) as e:
        # Fallback: grid search
        print(f"  [INFO] scipy unavailable or fit failed ({e}), using grid search")
        best_loss = float("inf")
        A, alpha, bpb_irr = 100.0, 0.5, 1.0
        for try_A in [1, 10, 50, 100, 500, 1000]:
            for try_alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                for try_irr in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
                    loss = float(
                        _power_law_residual(
                            params_arr, bpbs_arr, try_A, try_alpha, try_irr,
                        )
                    )
                    if loss < best_loss:
                        best_loss = loss
                        A, alpha, bpb_irr = try_A, try_alpha, try_irr

    # R-squared
    predicted = power_law(params_arr, A, alpha, bpb_irr)
    residuals = bpbs_arr - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((bpbs_arr - np.mean(bpbs_arr)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "A": float(A),
        "alpha": float(alpha),
        "bpb_irr": float(bpb_irr),
        "r_squared": r_squared,
    }


# ── Plotting ─────────────────────────────────────────────────────────


COLORS = {
    "bare_ssm": "#2196F3",
    "full_ssm": "#4CAF50",
    "our_tfm": "#FF9800",
    "mamba2_ssm": "#9C27B0",
}
MARKERS = {
    "bare_ssm": "o",
    "full_ssm": "s",
    "our_tfm": "^",
    "mamba2_ssm": "D",
}


def _import_matplotlib():
    """Import matplotlib with Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_scaling_curves(
    data: dict[str, list[dict]],
    fits: dict[str, dict],
    comp_bpb: float | None = None,
):
    """Plot bpb vs param count for all conditions (log-log)."""
    try:
        plt = _import_matplotlib()
        import numpy as np
    except ImportError:
        print("  [SKIP] scaling_curves.png: matplotlib/numpy not available")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for cond, points in data.items():
        params = [p["params"] for p in points]
        bpbs = [p["bpb"] for p in points]
        sems = [
            compute_sem(p.get("bpb_values", [])) if len(p.get("bpb_values", [])) >= 2 else 0.0
            for p in points
        ]
        color = COLORS.get(cond, "#999999")
        marker = MARKERS.get(cond, "D")

        ax.errorbar(
            params, bpbs, yerr=sems, fmt=marker, color=color,
            label=cond, markersize=8, capsize=4, linewidth=1.5,
        )

        # Plot fitted curve
        if cond in fits and fits[cond]["r_squared"] > 0.5:
            fit = fits[cond]
            x_fit = np.logspace(
                np.log10(min(params) * 0.8),
                np.log10(max(params) * 1.2),
                100,
            )
            y_fit = fit["A"] * np.power(x_fit, -fit["alpha"]) + fit["bpb_irr"]
            alpha_str = f"alpha={fit['alpha']:.3f}"
            ax.plot(
                x_fit, y_fit, "--", color=color, alpha=0.5,
                label=f"{cond} fit ({alpha_str})",
            )

    # Reference lines
    ax.axhline(
        y=COMP_BASELINE_BPB, color="red", linestyle=":", alpha=0.7,
        label=f"Competition baseline ({COMP_BASELINE_BPB})",
    )
    ax.axhline(
        y=COMP_SOTA_BPB, color="darkred", linestyle=":", alpha=0.7,
        label=f"Competition SOTA ({COMP_SOTA_BPB})",
    )
    if comp_bpb is not None:
        ax.axhline(
            y=comp_bpb, color="red", linestyle="-", alpha=0.5,
            label=f"Our comp_tfm run ({comp_bpb:.4f})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters", fontsize=12)
    ax.set_ylabel("BPB (bits per byte)", fontsize=12)
    ax.set_title("SESSM vs Transformer Scaling Laws", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scaling_curves.png", dpi=150)
    print(f"  Saved {PLOTS_DIR / 'scaling_curves.png'}")
    plt.close(fig)


def plot_isoflop(data: dict[str, list[dict]]):
    """Plot bpb vs total FLOPs for all conditions."""
    try:
        plt = _import_matplotlib()
    except ImportError:
        print("  [SKIP] isoflop_curves.png: matplotlib not available")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for cond, points in data.items():
        flops = [p["total_flops"] for p in points if p["total_flops"] > 0]
        bpbs = [p["bpb"] for p in points if p["total_flops"] > 0]
        if not flops:
            continue
        color = COLORS.get(cond, "#999999")
        marker = MARKERS.get(cond, "D")
        ax.scatter(flops, bpbs, c=color, marker=marker, s=80, label=cond, zorder=3)

    ax.axhline(
        y=COMP_BASELINE_BPB, color="red", linestyle=":", alpha=0.7,
        label=f"Competition baseline ({COMP_BASELINE_BPB})",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Total Training FLOPs", fontsize=12)
    ax.set_ylabel("BPB (bits per byte)", fontsize=12)
    ax.set_title("isoFLOP Comparison: BPB vs Compute", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "isoflop_curves.png", dpi=150)
    print(f"  Saved {PLOTS_DIR / 'isoflop_curves.png'}")
    plt.close(fig)


def plot_component_delta(data: dict[str, list[dict]]):
    """Plot component delta (bare_ssm - full_ssm) vs model size."""
    if "bare_ssm" not in data or "full_ssm" not in data:
        print("  [SKIP] Component delta plot: need both bare_ssm and full_ssm")
        return

    try:
        plt = _import_matplotlib()
    except ImportError:
        print("  [SKIP] component_delta.png: matplotlib not available")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    bare_map = {p["size"]: p for p in data["bare_ssm"]}
    full_map = {p["size"]: p for p in data["full_ssm"]}

    sizes = []
    deltas = []
    sig_labels = []
    for size in bare_map:
        if size in full_map:
            sizes.append(size)
            deltas.append(bare_map[size]["bpb"] - full_map[size]["bpb"])
            bare_vals = bare_map[size].get("bpb_values", [])
            full_vals = full_map[size].get("bpb_values", [])
            if len(bare_vals) >= 2 and len(full_vals) >= 2:
                _t, p_val = welch_ttest(bare_vals, full_vals)
                sig_labels.append("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            else:
                sig_labels.append("n<2")

    ax.bar(sizes, deltas, color="#4CAF50", alpha=0.7, edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel(
        "Component Delta (bare - full, positive = components help)", fontsize=12,
    )
    ax.set_title("Component ROI Across Scale", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate each bar with delta and significance
    for i, (size, delta, sig) in enumerate(zip(sizes, deltas, sig_labels)):
        ax.annotate(
            f"{delta:+.4f} {sig}", (i, delta), textcoords="offset points",
            xytext=(0, 10 if delta >= 0 else -15), ha="center", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "component_delta.png", dpi=150)
    print(f"  Saved {PLOTS_DIR / 'component_delta.png'}")
    plt.close(fig)


# ── Summary table with significance ─────────────────────────────────


def print_summary_table(data: dict[str, list[dict]], fits: dict[str, dict]):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 100)
    print("  SCALING LAW RESULTS")
    print("=" * 100)
    print(
        f"  {'Size':<6} {'Condition':<14} {'Params':>10} {'BPB':>8} "
        f"{'±SEM':>8} {'95% CI':>20} {'Seeds':>5} {'Steps':>8} {'FLOPs':>12}"
    )
    print(f"  {'-' * 95}")

    for cond, points in sorted(data.items()):
        for p in points:
            bpb_vals = p.get("bpb_values", [])
            n = len(bpb_vals)
            sem_val = compute_sem(bpb_vals) if n >= 2 else 0.0
            ci = bootstrap_ci(bpb_vals) if n >= 2 else (p["bpb"], p["bpb"])
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            print(
                f"  {p['size']:<6} {cond:<14} {p['params']:>10,.0f} "
                f"{p['bpb']:>8.4f} {sem_val:>8.4f} {ci_str:>20} "
                f"{n:>5} {p['steps']:>8,.0f} {p['total_flops']:>12.2e}"
            )
        print()

    # Power-law fits
    print("\n  POWER-LAW FITS: bpb(N) = A * N^(-alpha) + bpb_irr")
    print(f"  {'-' * 60}")
    print(f"  {'Condition':<14} {'A':>10} {'alpha':>10} {'bpb_irr':>10} {'R^2':>10}")
    for cond, fit in fits.items():
        print(
            f"  {cond:<14} {fit['A']:>10.4f} {fit['alpha']:>10.4f} "
            f"{fit['bpb_irr']:>10.4f} {fit['r_squared']:>10.4f}"
        )

    # Significance tests between conditions at each size
    print("\n  PAIRWISE SIGNIFICANCE (Welch t-test, per size)")
    print(f"  {'-' * 60}")

    # Build size -> condition -> bpb_values map
    size_cond_bpbs: dict[str, dict[str, list[float]]] = {}
    for cond, points in data.items():
        for p in points:
            size = p["size"]
            if size not in size_cond_bpbs:
                size_cond_bpbs[size] = {}
            size_cond_bpbs[size][cond] = p.get("bpb_values", [])

    for size in sorted(size_cond_bpbs.keys()):
        conds = size_cond_bpbs[size]
        cond_names = sorted(conds.keys())
        for i, c1 in enumerate(cond_names):
            for c2 in cond_names[i + 1:]:
                v1 = conds[c1]
                v2 = conds[c2]
                if len(v1) >= 2 and len(v2) >= 2:
                    _t, p = welch_ttest(v1, v2)
                    d = cohens_d(v1, v2)
                    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
                    m1 = statistics.mean(v1)
                    m2 = statistics.mean(v2)
                    print(
                        f"  {size:<6} {c1} vs {c2}: "
                        f"p={p:.4f} {sig} "
                        f"(d={d:.2f}, means={m1:.4f} vs {m2:.4f})"
                    )
                else:
                    print(
                        f"  {size:<6} {c1} vs {c2}: "
                        f"n<2, cannot test significance"
                    )

    # Kill criteria check
    print("\n  KILL CRITERIA CHECK")
    print(f"  {'-' * 60}")

    ssm_alpha = fits.get("bare_ssm", {}).get("alpha", 0)
    tfm_alpha = fits.get("our_tfm", {}).get("alpha", 0)
    if tfm_alpha > ssm_alpha + 0.1:
        print(
            f"  *** KILL: Transformer alpha ({tfm_alpha:.3f}) > "
            f"SSM alpha ({ssm_alpha:.3f}) + 0.1"
        )
    else:
        print(
            f"  OK: SSM alpha ({ssm_alpha:.3f}) vs "
            f"Transformer alpha ({tfm_alpha:.3f})"
        )

    # Check XL bare_ssm bpb
    for p in data.get("bare_ssm", []):
        if p["size"] == "XL" and p["bpb"] > 2.0:
            print(f"  *** KILL: bare_ssm at XL has bpb={p['bpb']:.4f} > 2.0")

    # Check component delta at XL
    bare_xl = next(
        (p for p in data.get("bare_ssm", []) if p["size"] == "XL"), None,
    )
    full_xl = next(
        (p for p in data.get("full_ssm", []) if p["size"] == "XL"), None,
    )
    if bare_xl and full_xl:
        delta = bare_xl["bpb"] - full_xl["bpb"]
        bare_vals = bare_xl.get("bpb_values", [])
        full_vals = full_xl.get("bpb_values", [])
        if len(bare_vals) >= 2 and len(full_vals) >= 2:
            _t, p_delta = welch_ttest(bare_vals, full_vals)
            d_delta = cohens_d(bare_vals, full_vals)
            sig_str = f"p={p_delta:.4f}, d={d_delta:.2f}"
        else:
            sig_str = "n<2, no test"
        if delta < 0:
            print(
                f"  *** KILL: component_delta at XL is {delta:.4f} < 0 "
                f"(full stack hurts) [{sig_str}]"
            )
        else:
            print(f"  OK: component_delta at XL is {delta:+.4f} (full stack helps) [{sig_str}]")

    # Seed variance comparison: is SSM more stable than transformer?
    print("\n  SEED VARIANCE COMPARISON")
    print(f"  {'-' * 60}")
    for size in sorted(size_cond_bpbs.keys()):
        conds = size_cond_bpbs[size]
        row = f"  {size:<6}"
        for cond in ["bare_ssm", "full_ssm", "our_tfm", "mamba2_ssm"]:
            vals = conds.get(cond, [])
            if len(vals) >= 2:
                row += f"  {cond}: std={statistics.stdev(vals):.4f} (n={len(vals)})"
            elif vals:
                row += f"  {cond}: n=1"
        print(row)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    results = load_results()
    data = extract_scaling_data(results)

    if not data:
        print("ERROR: No scaling data found in results.")
        sys.exit(1)

    # Fit power laws (need >= 3 points for 3-param fit)
    fits = {}
    for cond, points in data.items():
        if len(points) >= 3:
            params = [p["params"] for p in points]
            bpbs = [p["bpb"] for p in points]
            fits[cond] = fit_power_law(params, bpbs)

    # Load competition baseline if available
    comp_bpb = None
    comp_path = RESULTS_DIR / "comp_tfm_baseline.json"
    if comp_path.exists():
        comp_data = json.loads(comp_path.read_text())
        comp_bpb = comp_data.get("bpb")

    # Also check inline comp_tfm in results
    if "comp_tfm" in results and isinstance(results["comp_tfm"], dict):
        comp_bpb = comp_bpb or results["comp_tfm"].get("bpb")

    # Generate plots
    plot_scaling_curves(data, fits, comp_bpb)
    plot_isoflop(data)
    plot_component_delta(data)

    # Print summary
    print_summary_table(data, fits)

    # Save analysis summary as JSON
    analysis = {
        "fits": fits,
        "competition_baseline_bpb": comp_bpb,
        "data": {cond: points for cond, points in data.items()},
    }
    analysis_path = RESULTS_DIR / "scaling_analysis.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n  Analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
