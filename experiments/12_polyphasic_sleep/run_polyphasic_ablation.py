#!/usr/bin/env python3
"""Polyphasic partitioned sleep ablation: does partition-scoped sleep match
or beat global offline consolidation?

Tests whether polyphasic sleep (partition-scoped interleaved micro-sleeps)
can compete with traditional full-offline sleep on the full ChaosControl
stack (SSM + memory + Wernicke MoE).

3 conditions x 7 seeds = 21 training runs

Conditions:
  1. no_sleep                    -- baseline, sleep disabled
  2. full_offline_sleep          -- global offline sleep (full_cycle)
  3. polyphasic_K3_N4_striped    -- polyphasic: 4 partitions, 3 awake,
                                   slot_striped topology, full_cycle stages

Confirmatory contrasts (paired Wilcoxon signed-rank, Holm-corrected, m=2):
  1. polyphasic vs no_sleep  (Does polyphasic sleep help?)
  2. polyphasic vs offline   (Is it competitive with global pause?)

Exploratory contrasts: none (populated later as conditions expand)

Decision criteria:
  - polyphasic < no_sleep (Holm-corrected p < 0.05)
    AND polyphasic >= full_offline (ns or better)  -> ADOPT POLYPHASIC
  - polyphasic < no_sleep but > full_offline       -> POLYPHASIC HELPS BUT
                                                      OFFLINE STILL BETTER
  - polyphasic >= no_sleep                         -> POLYPHASIC NOT HELPFUL YET
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / "results"
CHECKPOINTS = EXPERIMENT / "checkpoints"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
from stats import bootstrap_ci, cohens_d, sem

SEEDS = [1337, 2674, 4011, 5348, 6685, 8022, 9359]


# -- Config templates -------------------------------------------------------


def _base(**overrides) -> dict:
    """Full stack base config (SSM + memory + Wernicke MoE)."""
    base = {
        "model_type": "ssm",
        "vocab_size": 256,
        "model_dim": 128,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 256,
        "stride": 128,
        "batch_size": 32,
        "base_lr": 2e-3,
        # Memory
        "outer_model_type": "multislot",
        "outer_model_dim": 64,
        "outer_max_slots": 64,
        "consolidation_write": "full_sequence",
        "latent_persistence": True,
        # Wernicke
        "wernicke_enabled": True,
        "wernicke_router": "moe",
        "wernicke_k_max": 16,
        "typed_storage": True,
    }
    base.update(overrides)
    return base


SLEEP_COMMON = {
    "sleep_interval": 256,
    "sleep_budget": 128,
    "sleep_n2_budget": 64,
    "sleep_rem_budget": 64,
}

CONDITIONS = {
    "no_sleep": _base(sleep_enabled=False, polyphasic_enabled=False),
    "full_offline_sleep": _base(
        sleep_enabled=True, polyphasic_enabled=False,
        sleep_stages="full_cycle", **SLEEP_COMMON,
    ),
    "polyphasic_K3_N4_striped": _base(
        sleep_enabled=True, polyphasic_enabled=True,
        polyphasic_n_partitions=4, polyphasic_k_awake=3,
        polyphasic_topology="slot_striped",
        sleep_stages="full_cycle", **SLEEP_COMMON,
    ),
}

# Pre-specified contrasts: (name, condition_a, condition_b, description)
# Confirmatory family (Holm-corrected, 2 contrasts).
# With 7 seeds, min two-sided Wilcoxon p = 2/128 = 0.0156.
# After 2-way Holm: best corrected p = 0.0156 * 2 = 0.031 < 0.05.
CONFIRMATORY_CONTRASTS = [
    ("polyphasic_vs_none", "no_sleep", "polyphasic_K3_N4_striped", "Does polyphasic sleep help?"),
    ("polyphasic_vs_offline", "full_offline_sleep", "polyphasic_K3_N4_striped", "Is it competitive with global pause?"),
]

# Exploratory family (uncorrected, effect sizes and CIs reported).
EXPLORATORY_CONTRASTS = []  # populated later as conditions expand

CONTRASTS = CONFIRMATORY_CONTRASTS + EXPLORATORY_CONTRASTS


# -- Execution ---------------------------------------------------------------


def _launch(
    name: str, config: dict, seed: int, budget: float,
    data_path: str, gpu_id: int | None,
) -> tuple[subprocess.Popen, Path, Path, object]:
    config = dict(config, seed=seed)
    tag = f"{name}_s{seed}"
    tmp = Path(tempfile.mktemp(
        suffix=".yaml", prefix=f".tmp_{tag}_",
        dir=EXPERIMENT / "configs",
    ))
    tmp.write_text(yaml.dump(config, default_flow_style=False))

    out_path = RESULTS / f"{tag}.json"
    cmd = [
        sys.executable, "-m", "chaoscontrol.runner",
        "--config", str(tmp),
        "--data-path", data_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
        "--checkpoint-dir", str(CHECKPOINTS),
        "--checkpoint-name", tag,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = RESULTS / f"{tag}.log"
    log_fh = open(log_path, "w")
    if Path("/proc/1/fd/1").exists():
        shell_cmd = " ".join(shlex.quote(str(c)) for c in cmd)
        shell_cmd += f" 2>&1 | tee {shlex.quote(str(log_path))} /proc/1/fd/1"
        proc = subprocess.Popen(
            ["bash", "-c", shell_cmd], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, out_path, tmp, log_fh


def run_training_grid(data_path: str, budget: float, num_gpus: int):
    """Run all training configs, parallelizing across conditions and seeds.

    Builds a flat queue of (condition, seed) pairs, skips completed runs,
    and dispatches up to num_gpus concurrent jobs at a time.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    # Build flat queue of all pending runs
    queue: list[tuple[str, dict, int]] = []
    for cond_name, cond_config in CONDITIONS.items():
        for seed in SEEDS:
            tag = f"{cond_name}_s{seed}"
            if not (RESULTS / f"{tag}.json").exists():
                queue.append((cond_name, cond_config, seed))

    total = len(CONDITIONS) * len(SEEDS)
    completed = total - len(queue)
    print(f"\n  {len(queue)} runs pending, {completed} already done, {total} total")
    print(f"  {num_gpus} GPUs, ~{len(queue) // num_gpus + (1 if len(queue) % num_gpus else 0)} batches")

    for batch_start in range(0, len(queue), num_gpus):
        batch = queue[batch_start:batch_start + num_gpus]
        batch_num = batch_start // num_gpus + 1
        total_batches = len(queue) // num_gpus + (1 if len(queue) % num_gpus else 0)
        print(f"\n  --- Batch {batch_num}/{total_batches} ---")
        jobs = []
        for i, (cond_name, cond_config, seed) in enumerate(batch):
            gpu_id = i % num_gpus if num_gpus > 1 else None
            proc, out_path, tmp, log_fh = _launch(
                cond_name, cond_config, seed, budget, data_path, gpu_id,
            )
            jobs.append((proc, out_path, tmp, log_fh, cond_name, seed))
            print(f"    GPU {i}: {cond_name} seed={seed}")

        for proc, out_path, tmp, log_fh, cond_name, seed in jobs:
            proc.wait()
            log_fh.close()
            tmp.unlink(missing_ok=True)
            completed += 1
            if proc.returncode != 0:
                print(f"    FAILED: {cond_name} seed={seed} (exit {proc.returncode})")
                continue
            if out_path.exists():
                with open(out_path) as f:
                    result = json.load(f)
                bpb = result["eval"]["bpb"]
                steps = result["train"]["steps"]
                print(f"    {cond_name} seed={seed}: bpb={bpb:.4f}  steps={steps}")

        print(f"  [{completed}/{total}]")

    _print_summary()


def _load_results() -> dict[str, dict[int, float]]:
    """Load per-condition, per-seed bpb values from result JSON files.

    Returns {condition_name: {seed: bpb}} so paired analysis can match by seed.
    """
    results: dict[str, dict[int, float]] = {}
    for f in sorted(RESULTS.glob("*.json")):
        if f.name == "polyphasic_summary.json":
            continue
        with open(f) as fh:
            try:
                data = json.load(fh)
            except (json.JSONDecodeError, KeyError):
                continue
        stem = f.stem
        parts = stem.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        cond_name = parts[0]
        if cond_name not in CONDITIONS:
            continue
        try:
            seed = int(parts[1])
        except ValueError:
            continue
        bpb = data.get("eval", {}).get("bpb")
        if bpb is None:
            continue
        results.setdefault(cond_name, {})[seed] = bpb
    return results


def _paired_deltas(
    results: dict[str, dict[int, float]], cond_a: str, cond_b: str,
) -> list[float]:
    """Compute per-seed deltas (b - a) for matched seeds."""
    a_data = results.get(cond_a, {})
    b_data = results.get(cond_b, {})
    shared_seeds = sorted(set(a_data.keys()) & set(b_data.keys()))
    return [b_data[s] - a_data[s] for s in shared_seeds]


def _wilcoxon_signed_rank_p(deltas: list[float]) -> float:
    """Wilcoxon signed-rank test (two-sided) for small samples.

    Returns exact p-value. With n=7, minimum achievable two-sided p is
    2/128 = 0.015625 (all deltas same sign).
    """
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    # Rank by absolute value
    nonzero.sort(key=lambda x: x[0])
    w_plus = sum(rank + 1 for rank, (_, sign) in enumerate(nonzero) if sign > 0)
    w_minus = sum(rank + 1 for rank, (_, sign) in enumerate(nonzero) if sign < 0)
    w = min(w_plus, w_minus)
    # Exact p for small n: count permutations where W <= observed
    # Total rank sum = n*(n+1)/2
    count = 0
    for mask in range(1 << n):
        w_test = sum((rank + 1) for rank in range(n) if mask & (1 << rank))
        if w_test <= w:
            count += 1
    p = 2.0 * count / (1 << n)  # Two-sided
    return min(1.0, p)


def _holm_bonferroni(p_values: list[tuple[str, float]]) -> list[tuple[str, float, float, bool]]:
    """Apply Holm-Bonferroni correction with step-down monotonic max.

    Returns [(name, raw_p, corrected_p, significant)].
    """
    m = len(p_values)
    sorted_pv = sorted(p_values, key=lambda x: x[1])
    results = []
    prev_corrected = 0.0
    for i, (name, p) in enumerate(sorted_pv):
        corrected = min(1.0, p * (m - i))
        # Step-down monotonic max: corrected p can never decrease
        corrected = max(corrected, prev_corrected)
        prev_corrected = corrected
        results.append((name, p, corrected, corrected < 0.05))
    return results


def _print_summary():
    """Print summary table and statistical comparisons.

    Uses paired analysis (per-seed deltas), Wilcoxon signed-rank tests,
    Holm-Bonferroni correction, and bootstrap CIs.
    """
    results = _load_results()  # {condition: {seed: bpb}}
    if not results:
        print("\n  No results found.")
        return

    # -- Summary table --
    print(f"\n{'='*70}")
    print("  POLYPHASIC SLEEP ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Condition':<30} {'mean bpb':>10} {'SEM':>8} {'95% CI':>18} {'n':>3}")
    print(f"  {'-'*73}")

    summary: dict[str, dict] = {}
    for cond_name in CONDITIONS:
        seed_bpbs = results.get(cond_name, {})
        if not seed_bpbs:
            print(f"  {cond_name:<30} {'--':>10} {'--':>8} {'--':>18} {0:>3}")
            continue
        bpbs = [seed_bpbs[s] for s in sorted(seed_bpbs.keys())]
        mean_bpb = sum(bpbs) / len(bpbs)
        s = sem(bpbs)
        ci = bootstrap_ci(bpbs)
        tag = f"{cond_name}_s{SEEDS[0]}"
        steps_file = RESULTS / f"{tag}.json"
        steps = "?"
        if steps_file.exists():
            with open(steps_file) as fh:
                steps = json.load(fh)["train"]["steps"]
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"  {cond_name:<30} {mean_bpb:>10.4f} {s:>8.4f} {ci_str:>18} {len(bpbs):>3}  steps={steps}")
        summary[cond_name] = {"mean_bpb": mean_bpb, "sem": s, "ci": ci, "n": len(bpbs), "bpbs": bpbs}

    # -- Helper: compute contrast details --
    def _compute_contrast(contrast_name, cond_a, cond_b, desc):
        deltas = _paired_deltas(results, cond_a, cond_b)
        if len(deltas) < 2:
            return None
        p = _wilcoxon_signed_rank_p(deltas)
        a_bpbs = summary.get(cond_a, {}).get("bpbs", [])
        b_bpbs = summary.get(cond_b, {}).get("bpbs", [])
        return {
            "name": contrast_name, "desc": desc,
            "cond_a": cond_a, "cond_b": cond_b,
            "mean_delta": sum(deltas) / len(deltas),
            "p_raw": p,
            "d": cohens_d(a_bpbs, b_bpbs) if a_bpbs and b_bpbs else 0.0,
            "delta_ci": bootstrap_ci(deltas),
            "n_pairs": len(deltas),
        }

    def _print_contrast(det, corr_p=None, sig=None, label=""):
        winner = det["cond_b"] if det["mean_delta"] < 0 else det["cond_a"]
        a_mean = summary.get(det["cond_a"], {}).get("mean_bpb", 0)
        b_mean = summary.get(det["cond_b"], {}).get("mean_bpb", 0)
        print(f"    {label}{det['desc']}")
        print(f"      {det['cond_a']}={a_mean:.4f} vs {det['cond_b']}={b_mean:.4f}")
        print(f"      mean delta={det['mean_delta']:+.4f} ({winner} wins)")
        print(f"      95% CI of delta: [{det['delta_ci'][0]:+.4f}, {det['delta_ci'][1]:+.4f}]")
        if corr_p is not None:
            sig_str = "SIGNIFICANT" if sig else "ns"
            print(f"      Wilcoxon p={det['p_raw']:.4f} (corrected p={corr_p:.4f}) {sig_str}  d={det['d']:.2f}")
        else:
            print(f"      Wilcoxon p={det['p_raw']:.4f} (uncorrected)  d={det['d']:.2f}")

    # -- Confirmatory contrasts (Holm-corrected within family) --
    print(f"\n  Confirmatory contrasts (paired Wilcoxon, Holm-corrected, m={len(CONFIRMATORY_CONTRASTS)}):")
    print(f"  {'-'*70}")

    confirm_details = {}
    confirm_pvals = []
    for contrast_name, cond_a, cond_b, desc in CONFIRMATORY_CONTRASTS:
        det = _compute_contrast(contrast_name, cond_a, cond_b, desc)
        if det is None:
            print(f"    {desc}: insufficient paired data")
            continue
        confirm_details[contrast_name] = det
        confirm_pvals.append((contrast_name, det["p_raw"]))

    corrected = _holm_bonferroni(confirm_pvals)
    corrected_map = {name: (raw, corr, sig) for name, raw, corr, sig in corrected}

    for contrast_name, cond_a, cond_b, desc in CONFIRMATORY_CONTRASTS:
        det = confirm_details.get(contrast_name)
        if not det:
            continue
        raw_p, corr_p, sig = corrected_map.get(contrast_name, (1.0, 1.0, False))
        _print_contrast(det, corr_p=corr_p, sig=sig)

    # -- Exploratory contrasts (uncorrected, effect sizes and CIs) --
    if EXPLORATORY_CONTRASTS:
        print(f"\n  Exploratory contrasts (uncorrected, effect sizes only):")
        print(f"  {'-'*70}")

        for contrast_name, cond_a, cond_b, desc in EXPLORATORY_CONTRASTS:
            det = _compute_contrast(contrast_name, cond_a, cond_b, desc)
            if det is None:
                print(f"    {desc}: insufficient paired data")
                continue
            _print_contrast(det, label="[EXPLORATORY] ")

    # -- Decision recommendation --
    print(f"\n{'='*70}")
    print("  DECISION")
    print(f"{'='*70}")

    no_sleep = summary.get("no_sleep")
    offline = summary.get("full_offline_sleep")
    polyphasic = summary.get("polyphasic_K3_N4_striped")
    if not no_sleep or not polyphasic:
        print("  Insufficient data for decision.")
    else:
        poly_vs_none = corrected_map.get("polyphasic_vs_none")
        poly_vs_offline = corrected_map.get("polyphasic_vs_offline")

        delta_none = polyphasic["mean_bpb"] - no_sleep["mean_bpb"]
        _, corr_p_none, sig_none = poly_vs_none if poly_vs_none else (1.0, 1.0, False)

        print(f"\n  no_sleep:                  {no_sleep['mean_bpb']:.4f} bpb")
        if offline:
            print(f"  full_offline_sleep:        {offline['mean_bpb']:.4f} bpb")
        print(f"  polyphasic_K3_N4_striped:  {polyphasic['mean_bpb']:.4f} bpb")
        print(f"\n  polyphasic vs no_sleep: delta={delta_none:+.4f}  corrected p={corr_p_none:.4f} {'(significant)' if sig_none else '(not significant)'}")

        if offline and poly_vs_offline:
            delta_offline = polyphasic["mean_bpb"] - offline["mean_bpb"]
            _, corr_p_off, sig_off = poly_vs_offline
            print(f"  polyphasic vs offline:  delta={delta_offline:+.4f}  corrected p={corr_p_off:.4f} {'(significant)' if sig_off else '(not significant)'}")

        print(f"\n  Recommendation:")
        if delta_none < 0 and sig_none:
            if offline:
                delta_offline = polyphasic["mean_bpb"] - offline["mean_bpb"]
                if delta_offline <= 0:
                    print(f"    >>> ADOPT POLYPHASIC <<<")
                    print(f"    Polyphasic sleep significantly improves bpb over no-sleep")
                    print(f"    and matches or beats full offline sleep.")
                else:
                    print(f"    >>> POLYPHASIC HELPS BUT OFFLINE STILL BETTER <<<")
                    print(f"    Polyphasic improves over baseline but offline sleep")
                    print(f"    achieves lower bpb ({offline['mean_bpb']:.4f} vs {polyphasic['mean_bpb']:.4f}).")
            else:
                print(f"    >>> ADOPT POLYPHASIC <<<")
                print(f"    Polyphasic sleep significantly improves bpb (Holm-corrected).")
        elif delta_none < 0 and not sig_none:
            print(f"    >>> CAUTIOUS ADOPTION <<<")
            print(f"    Polyphasic trends better but not significant after Holm correction.")
        else:
            print(f"    >>> POLYPHASIC NOT HELPFUL YET <<<")
            print(f"    Polyphasic sleep does not outperform baseline after Holm correction.")

    with open(RESULTS / "polyphasic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Polyphasic partitioned sleep ablation: does partition-scoped sleep match or beat global offline consolidation?"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--budget", type=float, default=600)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    t0 = time.time()

    run_training_grid(args.data_path, args.budget, args.num_gpus)

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed/60:.1f} minutes")
    print(f"  Results: {RESULTS}")


if __name__ == "__main__":
    main()
