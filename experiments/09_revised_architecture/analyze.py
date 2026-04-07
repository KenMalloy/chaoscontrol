#!/usr/bin/env python3
"""Analyze results for experiment 09 — revised architecture.

Reads per-layer summary JSONs and individual run results to produce:
1. Per-layer ranked tables with mean +/- std bpb
2. Step-normalized bpb comparison
3. Statistical significance flags (overlapping standard errors)
4. Cross-layer winner progression
5. Latent reactivation event counts per config
6. Fork rate per gate config
"""
import json
import statistics
import sys
from pathlib import Path

# Local statistical utilities (pure Python, no scipy dependency)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import bootstrap_ci, cohens_d, sem as compute_sem, welch_ttest

RESULTS = Path(__file__).resolve().parent / "results"


def load_layer_summary(layer_tag: str) -> dict | None:
    """Load a layer summary JSON if it exists."""
    path = RESULTS / f"{layer_tag}_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_individual_results() -> dict[str, dict]:
    """Load all individual seed-level results."""
    results = {}
    for f in sorted(RESULTS.glob("*_seed*.json")):
        with open(f) as fh:
            results[f.stem] = json.load(fh)
    return results


def print_ranked_table(layer_name: str, data: dict):
    """Print a ranked BPB table for a layer with SEM, CI, and significance."""
    layer_results = data.get("results", {})
    if not layer_results:
        return

    rows = []
    for name, seed_results in layer_results.items():
        # Prefer bpb_gated when available (matches pick_winner logic)
        bpbs = []
        for r in seed_results.values():
            if "eval" not in r:
                continue
            ev = r["eval"]
            bpbs.append(ev.get("bpb_gated", ev["bpb"]))
        steps_list = [r["train"]["steps"] for r in seed_results.values() if "train" in r]
        if not bpbs:
            continue
        mean_bpb = statistics.mean(bpbs)
        se = compute_sem(bpbs)
        ci = bootstrap_ci(bpbs)
        mean_steps = statistics.mean(steps_list) if steps_list else 0
        rows.append((name, mean_bpb, se, ci, len(bpbs), mean_steps, bpbs))
    rows.sort(key=lambda r: r[1])

    print(f"\n  {layer_name}")
    print(f"  {'=' * 88}")
    hdr = (
        f"  {'Config':<36} {'Sig':>4} {'Mean BPB':>10} {'SEM':>9}"
        f"  {'95% CI':>20} {'Seeds':>6} {'Steps':>8}"
    )
    print(hdr)
    print(f"  {'-' * 88}")

    winner_bpbs = rows[0][6] if rows else []
    for name, mean, se, ci, n, steps, bpbs in rows:
        # Significance label
        if n < 2:
            sig = "n=1"
        elif name == rows[0][0]:
            if len(rows) >= 2 and len(rows[1][6]) >= 2:
                _t, p = welch_ttest(winner_bpbs, rows[1][6])
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            else:
                sig = ""
        else:
            if len(winner_bpbs) >= 2:
                _t, p = welch_ttest(bpbs, winner_bpbs)
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            else:
                sig = ""

        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        marker = " <-- WINNER" if name == rows[0][0] else ""
        print(
            f"  {name:<36} {sig:>4} {mean:10.4f} +/-{se:.4f}"
            f"  {ci_str:>20} {n:6} {steps:8.0f}{marker}"
        )

    # Winner line with effect-size summary
    if len(rows) >= 2 and len(rows[0][6]) >= 2 and len(rows[1][6]) >= 2:
        _t, p = welch_ttest(rows[0][6], rows[1][6])
        d = cohens_d(rows[0][6], rows[1][6])
        print(
            f"\n  Winner: {rows[0][0]} "
            f"(p={p:.4f} vs runner-up, Cohen's d={d:.2f})"
        )
        if p >= 0.05:
            print(
                f"  *** WARNING: winner is NOT significantly better than "
                f"runner-up (p={p:.4f})"
            )
    elif rows and len(rows[0][6]) < 2:
        print(f"\n  Winner: {rows[0][0]} (NOTE: n=1, significance not testable)")

    # Step-normalized BPB
    print(f"\n  Step-normalized (bpb / steps * 1000):")
    for name, mean, _se, _ci, _n, steps, _bpbs in rows:
        if steps > 0:
            norm = mean / steps * 1000
            print(f"    {name:<40} {norm:10.6f}")


def analyze_fork_rates(individual_results: dict):
    """Analyze fork rates for gate configs."""
    # Group by config (strip _seedNNN suffix)
    configs: dict[str, list] = {}
    for name, result in individual_results.items():
        # Parse config name: everything before _seed
        parts = name.rsplit("_seed", 1)
        cfg_name = parts[0]
        if cfg_name not in configs:
            configs[cfg_name] = []
        configs[cfg_name].append(result)

    gate_configs = {k: v for k, v in configs.items() if k.startswith("L1_gate")}
    if not gate_configs:
        return

    print(f"\n{'=' * 80}")
    print(f"  Fork Rates (Layer 1 gate configs)")
    print(f"{'=' * 80}")
    print(f"  {'Config':<40} {'Fork Count':>12} {'Total Steps':>12} {'Fork Rate':>10}")
    print(f"  {'-' * 76}")
    for name, runs in sorted(gate_configs.items()):
        fork_counts = [r.get("train", {}).get("fork_count", 0) for r in runs]
        total_steps = [r.get("train", {}).get("steps", 0) for r in runs]
        mean_forks = statistics.mean(fork_counts) if fork_counts else 0
        mean_steps = statistics.mean(total_steps) if total_steps else 0
        rate = mean_forks / mean_steps if mean_steps > 0 else 0
        print(f"  {name:<40} {mean_forks:12.0f} {mean_steps:12.0f} {rate:10.4f}")


def analyze_latent_reactivation(individual_results: dict):
    """Count latent reactivation events per config."""
    configs: dict[str, list] = {}
    for name, result in individual_results.items():
        parts = name.rsplit("_seed", 1)
        cfg_name = parts[0]
        if cfg_name not in configs:
            configs[cfg_name] = []
        configs[cfg_name].append(result)

    print(f"\n{'=' * 80}")
    print(f"  Latent Reactivation Events")
    print(f"{'=' * 80}")
    print(f"  {'Config':<40} {'Mean Events':>14} {'Total Steps':>12}")
    print(f"  {'-' * 68}")
    found_any = False
    for name, runs in sorted(configs.items()):
        events = []
        for r in runs:
            history = r.get("train", {}).get("history", [])
            count = sum(1 for step in history if step.get("latent_reactivated", False))
            events.append(count)
        mean_events = statistics.mean(events) if events else 0
        if mean_events > 0:
            found_any = True
            total_steps = statistics.mean([r.get("train", {}).get("steps", 0) for r in runs])
            print(f"  {name:<40} {mean_events:14.1f} {total_steps:12.0f}")
    if not found_any:
        print(f"  (no latent reactivation events found)")


def print_cross_layer_progression():
    """Show how the winner evolves across layers."""
    print(f"\n{'=' * 80}")
    print(f"  Cross-Layer Winner Progression")
    print(f"{'=' * 80}")
    for tag in ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]:
        data = load_layer_summary(tag)
        if data and "winner" in data:
            winner = data["winner"]
            seed_results = data.get("results", {}).get(winner, {})
            bpbs = []
            for r in seed_results.values():
                if "eval" in r:
                    ev = r["eval"]
                    bpbs.append(ev.get("bpb_gated", ev["bpb"]))
            if bpbs:
                mean = statistics.mean(bpbs)
                se = compute_sem(bpbs)
                n = len(bpbs)
                n_note = "" if n >= 2 else " (n=1)"
                print(f"  {tag}: {winner:<40} bpb={mean:.4f} +/-{se:.4f} SEM (n={n}){n_note}")
            else:
                print(f"  {tag}: {winner}")

    # Dark horses
    l35 = load_layer_summary("L35")
    if l35:
        print(f"\n  Dark Horses (L3.5):")
        for name, seed_results in l35.get("results", {}).items():
            bpbs = []
            for r in seed_results.values():
                if "eval" in r:
                    ev = r["eval"]
                    bpbs.append(ev.get("bpb_gated", ev["bpb"]))
            if bpbs:
                se = compute_sem(bpbs)
                print(f"    {name:<38} bpb={statistics.mean(bpbs):.4f} +/-{se:.4f} SEM (n={len(bpbs)})")


def main():
    if not RESULTS.exists():
        print("No results directory found. Run the experiment first.")
        return

    individual = load_individual_results()
    if not individual:
        print("No individual results found. Run the experiment first.")
        return

    # Per-layer ranked tables
    for tag, label in [
        ("L1", "Layer 1: Gate Modes"),
        ("L2", "Layer 2: +Memory"),
        ("L3", "Layer 3: +Wernicke + Regret"),
        ("L35", "Layer 3.5: Dark Horses"),
        ("L4", "Layer 4: Scaling"),
        ("L5", "Layer 5: Full A-mode"),
    ]:
        data = load_layer_summary(tag)
        if data:
            print_ranked_table(label, data)

    # Fork rate analysis
    analyze_fork_rates(individual)

    # Latent reactivation analysis
    analyze_latent_reactivation(individual)

    # Cross-layer progression
    print_cross_layer_progression()

    # Full summary
    full_path = RESULTS / "full_summary.json"
    if full_path.exists():
        with open(full_path) as f:
            summary = json.load(f)
        print(f"\n{'=' * 80}")
        print(f"  Final Summary (values are mean +/- SEM)")
        print(f"{'=' * 80}")
        for layer, info in summary.items():
            # Recompute SEM from the layer summary if available
            layer_data = load_layer_summary(layer.replace(".", ""))
            if layer_data:
                winner = info["winner"]
                seed_results = layer_data.get("results", {}).get(winner, {})
                bpbs = []
                for r in seed_results.values():
                    if "eval" in r:
                        ev = r["eval"]
                        bpbs.append(ev.get("bpb_gated", ev["bpb"]))
                if bpbs:
                    mean = statistics.mean(bpbs)
                    se = compute_sem(bpbs)
                    print(f"  {layer}: {winner} — bpb={mean:.4f} +/-{se:.4f} SEM (n={len(bpbs)})")
                    continue
            # Fallback: use stored values (std_bpb may actually be std, not SEM)
            print(f"  {layer}: {info['winner']} — bpb={info['mean_bpb']:.4f} +/- {info['std_bpb']:.4f}")


if __name__ == "__main__":
    main()
