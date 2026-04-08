#!/usr/bin/env python3
"""Sleep cycle ablation: does structured offline consolidation improve bpb?

Tests whether the full sleep cycle (N1/N2/N3/REM) outperforms no-sleep
and partial-sleep configurations when added to the full ChaosControl stack
(SSM + memory + Wernicke MoE).

9 conditions x 7 seeds = 63 training runs

Conditions:
  1. no_sleep             -- baseline, sleep_enabled=False
  2. n3_only              -- sleep_enabled=True, stages="n3_only"
  3. n2_n3                -- stages="n2_n3"
  4. n2_n3_rem_base       -- dreams + scoring only (no mechanisms)
  5. n2_n3_rem_validate   -- stages="n2_n3_rem_validate"
  6. n2_n3_rem_cfr        -- stages="n2_n3_rem_cfr"
  7. n2_n3_rem_reactivate -- stages="n2_n3_rem_reactivate"
  8. n2_n3_rem_all        -- stages="n2_n3_rem_all"
  9. full_cycle           -- stages="full_cycle"

Confirmatory contrasts (paired Wilcoxon signed-rank, Holm-corrected, m=3):
  1. no_sleep vs full_cycle (does sleep help at all?)
  2. n3_only vs n2_n3 (does N2 tagging add value?)
  3. n2_n3 vs n2_n3_rem_all (does REM add value?)

Exploratory contrasts (uncorrected, effect sizes and CIs):
  4. n2_n3_rem_base vs n2_n3_rem_validate (merge validation vs base REM)
  5. n2_n3_rem_base vs n2_n3_rem_cfr (CFR vs base REM)
  6. n2_n3_rem_base vs n2_n3_rem_reactivate (reactivation vs base REM)
  7. n2_n3_rem_all vs full_cycle (does N1 transition help?)

Decision criteria:
  - full_cycle < no_sleep (Holm-corrected p < 0.05) -> ADOPT FULL CYCLE
  - full_cycle trends better but ns after correction -> CAUTIOUS ADOPTION
  - no_sleep = all sleep variants                    -> SLEEP NOT HELPFUL YET
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
        "batch_size": 64,
        "eval_batches": 32,
        "a_mode": "diag",
        "base_lr": 2e-3,
        "weight_decay": 1e-2,
        "grad_clip_norm": 1.0,
        "metabolic_gate": False,
        "cfr_enabled": False,
        # Memory
        "outer_model_dim": 64,
        "outer_model_type": "multislot",
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
    "no_sleep": _base(sleep_enabled=False),
    "n3_only": _base(sleep_enabled=True, sleep_stages="n3_only", **SLEEP_COMMON),
    "n2_n3": _base(sleep_enabled=True, sleep_stages="n2_n3", **SLEEP_COMMON),
    "n2_n3_rem_base": _base(
        sleep_enabled=True, sleep_stages="n2_n3_rem_base", **SLEEP_COMMON
    ),
    "n2_n3_rem_validate": _base(
        sleep_enabled=True, sleep_stages="n2_n3_rem_validate", **SLEEP_COMMON
    ),
    "n2_n3_rem_cfr": _base(
        sleep_enabled=True, sleep_stages="n2_n3_rem_cfr", **SLEEP_COMMON
    ),
    "n2_n3_rem_reactivate": _base(
        sleep_enabled=True, sleep_stages="n2_n3_rem_reactivate", **SLEEP_COMMON
    ),
    "n2_n3_rem_all": _base(
        sleep_enabled=True, sleep_stages="n2_n3_rem_all", **SLEEP_COMMON
    ),
    "full_cycle": _base(
        sleep_enabled=True, sleep_stages="full_cycle", **SLEEP_COMMON
    ),
}

# Pre-specified contrasts: (name, condition_a, condition_b, description)
# Confirmatory family (Holm-corrected, 3 contrasts).
# With 7 seeds, min two-sided Wilcoxon p = 2/128 = 0.0156.
# After 3-way Holm: best corrected p = 0.0156 * 3 = 0.047 < 0.05.
CONFIRMATORY_CONTRASTS = [
    ("sleep_vs_none", "no_sleep", "full_cycle", "Does sleep help at all?"),
    ("n2_value", "n3_only", "n2_n3", "Does N2 tagging add value?"),
    ("rem_value", "n2_n3", "n2_n3_rem_all", "Does REM add value?"),
]

# Exploratory family (uncorrected, effect sizes and CIs reported).
# REM mechanism contrasts compare against n2_n3_rem_base (dreams + scoring
# only) so each contrast isolates the named mechanism, not base REM.
EXPLORATORY_CONTRASTS = [
    ("validate_isolation", "n2_n3_rem_base", "n2_n3_rem_validate", "Base REM + validate vs base REM"),
    ("cfr_isolation", "n2_n3_rem_base", "n2_n3_rem_cfr", "Base REM + CFR vs base REM"),
    ("reactivate_isolation", "n2_n3_rem_base", "n2_n3_rem_reactivate", "Base REM + reactivate vs base REM"),
    ("n1_value", "n2_n3_rem_all", "full_cycle", "Does N1 transition help?"),
]

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
        if f.name == "sleep_summary.json":
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
    total = n * (n + 1) // 2
    # Enumerate all 2^n sign assignments
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
    print("  SLEEP ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Condition':<25} {'mean bpb':>10} {'SEM':>8} {'95% CI':>18} {'n':>3}")
    print(f"  {'-'*68}")

    summary: dict[str, dict] = {}
    for cond_name in CONDITIONS:
        seed_bpbs = results.get(cond_name, {})
        if not seed_bpbs:
            print(f"  {cond_name:<25} {'--':>10} {'--':>8} {'--':>18} {0:>3}")
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
        print(f"  {cond_name:<25} {mean_bpb:>10.4f} {s:>8.4f} {ci_str:>18} {len(bpbs):>3}  steps={steps}")
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
    print(f"\n  Exploratory contrasts (uncorrected, effect sizes only):")
    print(f"  {'-'*70}")

    explore_details = {}
    for contrast_name, cond_a, cond_b, desc in EXPLORATORY_CONTRASTS:
        det = _compute_contrast(contrast_name, cond_a, cond_b, desc)
        if det is None:
            print(f"    {desc}: insufficient paired data")
            continue
        explore_details[contrast_name] = det
        _print_contrast(det, label="[EXPLORATORY] ")

    # -- Decision recommendation --
    print(f"\n{'='*70}")
    print("  DECISION")
    print(f"{'='*70}")

    no_sleep = summary.get("no_sleep")
    full_cycle = summary.get("full_cycle")
    if not no_sleep or not full_cycle:
        print("  Insufficient data for decision.")
    else:
        primary = corrected_map.get("sleep_vs_none")
        delta = full_cycle["mean_bpb"] - no_sleep["mean_bpb"]
        _, corr_p, sig = primary if primary else (1.0, 1.0, False)
        print(f"\n  no_sleep:    {no_sleep['mean_bpb']:.4f} bpb")
        print(f"  full_cycle:  {full_cycle['mean_bpb']:.4f} bpb")
        print(f"  delta: {delta:+.4f}  corrected p={corr_p:.4f} {'(significant)' if sig else '(not significant)'}")

        print(f"\n  Recommendation:")
        if delta < 0 and sig:
            print(f"    >>> ADOPT FULL CYCLE <<<")
            print(f"    Full sleep cycle significantly improves bpb (Holm-corrected).")
        elif delta < 0 and not sig:
            print(f"    >>> CAUTIOUS ADOPTION <<<")
            print(f"    Full cycle trends better but not significant after correction.")
            # Note best partial as EXPLORATORY only
            best_partial = None
            best_partial_bpb = float("inf")
            for cond in ["n3_only", "n2_n3", "n2_n3_rem_base", "n2_n3_rem_validate", "n2_n3_rem_cfr", "n2_n3_rem_reactivate", "n2_n3_rem_all"]:
                s = summary.get(cond)
                if s and s["mean_bpb"] < best_partial_bpb:
                    best_partial = cond
                    best_partial_bpb = s["mean_bpb"]
            if best_partial and best_partial_bpb < no_sleep["mean_bpb"]:
                print(f"    [EXPLORATORY] Best partial: {best_partial} ({best_partial_bpb:.4f})")
                print(f"    NOTE: this is post-hoc selection, not a confirmatory test.")
        else:
            print(f"    >>> SLEEP NOT HELPFUL YET <<<")
            print(f"    No sleep variant significantly outperforms baseline after Holm correction.")

    with open(RESULTS / "sleep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Sleep cycle ablation: does structured offline consolidation improve bpb?"
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
