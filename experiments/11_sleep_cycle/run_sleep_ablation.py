#!/usr/bin/env python3
"""Sleep cycle ablation: does structured offline consolidation improve bpb?

Tests whether the full sleep cycle (N1/N2/N3/REM) outperforms no-sleep
and partial-sleep configurations when added to the full ChaosControl stack
(SSM + memory + Wernicke MoE).

7 conditions x 5 seeds = 35 training runs

Conditions:
  1. no_sleep        -- baseline, sleep_enabled=False
  2. n3_only         -- sleep_enabled=True, stages="n3_only"
  3. n2_n3           -- stages="n2_n3"
  4. n2_n3_rem_validate -- stages="n2_n3_rem_validate"
  5. n2_n3_rem_cfr   -- stages="n2_n3_rem_cfr"
  6. n2_n3_rem_full  -- stages="n2_n3_rem_full"
  7. full_cycle      -- stages="full_cycle"

Pre-specified contrasts (Welch t-test):
  1. no_sleep vs full_cycle (does sleep help at all?)
  2. n3_only vs n2_n3 (does N2 tagging add value?)
  3. n2_n3 vs n2_n3_rem_full (does REM add value?)
  4. n2_n3_rem_validate vs n2_n3_rem_cfr (validate vs CFR)
  5. n2_n3_rem_full vs full_cycle (does N1 transition help?)

Decision criteria:
  - full_cycle < no_sleep by >0.05 bpb (sig) -> ADOPT FULL CYCLE
  - partial < no_sleep, full_cycle ~ partial  -> ADOPT PARTIAL
  - no_sleep = all sleep variants             -> SLEEP NOT HELPFUL YET
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
from stats import welch_ttest, bootstrap_ci, cohens_d, sem

SEEDS = [1337, 2674, 4011, 5348, 6685]


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
    "n2_n3_rem_validate": _base(
        sleep_enabled=True, sleep_stages="n2_n3_rem_validate", **SLEEP_COMMON
    ),
    "n2_n3_rem_cfr": _base(
        sleep_enabled=True, sleep_stages="n2_n3_rem_cfr", **SLEEP_COMMON
    ),
    "n2_n3_rem_full": _base(
        sleep_enabled=True, sleep_stages="n2_n3_rem_full", **SLEEP_COMMON
    ),
    "full_cycle": _base(
        sleep_enabled=True, sleep_stages="full_cycle", **SLEEP_COMMON
    ),
}

# Pre-specified contrasts: (name, condition_a, condition_b, description)
CONTRASTS = [
    ("sleep_vs_none", "no_sleep", "full_cycle", "Does sleep help at all?"),
    ("n2_value", "n3_only", "n2_n3", "Does N2 tagging add value?"),
    ("rem_value", "n2_n3", "n2_n3_rem_full", "Does REM add value?"),
    ("validate_vs_cfr", "n2_n3_rem_validate", "n2_n3_rem_cfr", "Validate vs CFR"),
    ("n1_value", "n2_n3_rem_full", "full_cycle", "Does N1 transition help?"),
]


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
    """Run all 35 training configs, parallelizing seeds across GPUs."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    total = len(CONDITIONS) * len(SEEDS)
    completed = 0

    for cond_name, cond_config in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"  {cond_name}  ({len(SEEDS)} seeds, budget={budget}s)")
        print(f"{'='*60}")

        # Check for existing results (resume)
        existing = []
        for seed in SEEDS:
            tag = f"{cond_name}_s{seed}"
            path = RESULTS / f"{tag}.json"
            if path.exists():
                existing.append(seed)
        if len(existing) == len(SEEDS):
            print(f"  Already done, skipping")
            completed += len(SEEDS)
            continue

        # Launch seeds in batches of num_gpus
        seed_queue = [s for s in SEEDS if not (RESULTS / f"{cond_name}_s{s}.json").exists()]
        for batch_start in range(0, len(seed_queue), num_gpus):
            batch_seeds = seed_queue[batch_start:batch_start + num_gpus]
            jobs = []
            for i, seed in enumerate(batch_seeds):
                gpu_id = i % num_gpus if num_gpus > 1 else None
                proc, out_path, tmp, log_fh = _launch(
                    cond_name, cond_config, seed, budget, data_path, gpu_id,
                )
                jobs.append((proc, out_path, tmp, log_fh, seed))

            for proc, out_path, tmp, log_fh, seed in jobs:
                proc.wait()
                log_fh.close()
                tmp.unlink(missing_ok=True)
                completed += 1
                if proc.returncode != 0:
                    print(f"  FAILED: seed={seed} (exit {proc.returncode})")
                    continue
                if out_path.exists():
                    with open(out_path) as f:
                        result = json.load(f)
                    bpb = result["eval"]["bpb"]
                    steps = result["train"]["steps"]
                    print(f"  seed={seed}: bpb={bpb:.4f}  steps={steps}")

        print(f"  [{completed}/{total}]")

    _print_summary()


def _load_results() -> dict[str, list[float]]:
    """Load per-condition bpb lists from result JSON files."""
    results: dict[str, list[float]] = {}
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
        bpb = data.get("eval", {}).get("bpb")
        if bpb is None:
            continue
        results.setdefault(cond_name, []).append(bpb)
    return results


def _print_summary():
    """Print summary table and statistical comparisons."""
    results = _load_results()
    if not results:
        print("\n  No results found.")
        return

    # -- Summary table --
    print(f"\n{'='*70}")
    print("  SLEEP ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Condition':<25} {'mean bpb':>10} {'SEM':>8} {'n':>3}")
    print(f"  {'-'*50}")

    summary: dict[str, dict] = {}
    for cond_name in CONDITIONS:
        bpbs = results.get(cond_name, [])
        if not bpbs:
            print(f"  {cond_name:<25} {'--':>10} {'--':>8} {0:>3}")
            continue
        mean_bpb = sum(bpbs) / len(bpbs)
        s = sem(bpbs)
        # Get steps from first result
        tag = f"{cond_name}_s{SEEDS[0]}"
        steps_file = RESULTS / f"{tag}.json"
        steps = "?"
        if steps_file.exists():
            with open(steps_file) as fh:
                steps = json.load(fh)["train"]["steps"]
        print(f"  {cond_name:<25} {mean_bpb:>10.4f} {s:>8.4f} {len(bpbs):>3}  steps={steps}")
        summary[cond_name] = {"mean_bpb": mean_bpb, "sem": s, "n": len(bpbs), "bpbs": bpbs}

    # -- Pre-specified contrasts --
    print(f"\n  Pre-specified contrasts (Welch t-test):")
    print(f"  {'-'*70}")
    for contrast_name, cond_a, cond_b, desc in CONTRASTS:
        a = summary.get(cond_a)
        b = summary.get(cond_b)
        if not a or not b:
            print(f"    {desc}: insufficient data ({cond_a} or {cond_b} missing)")
            continue
        t, p = welch_ttest(a["bpbs"], b["bpbs"])
        d = cohens_d(a["bpbs"], b["bpbs"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        delta = b["mean_bpb"] - a["mean_bpb"]
        winner = cond_b if delta < 0 else cond_a
        print(f"    {desc}")
        print(f"      {cond_a}={a['mean_bpb']:.4f} vs {cond_b}={b['mean_bpb']:.4f}")
        print(f"      delta={delta:+.4f} ({winner} wins) p={p:.4f} {sig} d={d:.2f}")

    # -- Decision recommendation --
    print(f"\n{'='*70}")
    print("  DECISION")
    print(f"{'='*70}")

    no_sleep = summary.get("no_sleep")
    full_cycle = summary.get("full_cycle")
    if not no_sleep or not full_cycle:
        print("  Insufficient data for decision.")
    else:
        delta = full_cycle["mean_bpb"] - no_sleep["mean_bpb"]
        t, p = welch_ttest(no_sleep["bpbs"], full_cycle["bpbs"])
        sig = p < 0.05
        print(f"\n  no_sleep:    {no_sleep['mean_bpb']:.4f} bpb")
        print(f"  full_cycle:  {full_cycle['mean_bpb']:.4f} bpb")
        print(f"  delta: {delta:+.4f}  p={p:.4f} {'(significant)' if sig else '(not significant)'}")

        # Find best partial condition
        best_partial = None
        best_partial_bpb = float("inf")
        for cond in ["n3_only", "n2_n3", "n2_n3_rem_validate", "n2_n3_rem_cfr", "n2_n3_rem_full"]:
            s = summary.get(cond)
            if s and s["mean_bpb"] < best_partial_bpb:
                best_partial = cond
                best_partial_bpb = s["mean_bpb"]

        print(f"\n  Recommendation:")
        if delta < -0.05 and sig:
            print(f"    >>> ADOPT FULL CYCLE <<<")
            print(f"    Full sleep cycle significantly improves bpb.")
        elif delta < 0 and not sig:
            print(f"    >>> CAUTIOUS ADOPTION <<<")
            print(f"    Full cycle trends better but not significant with 5 seeds.")
            if best_partial and best_partial_bpb < no_sleep["mean_bpb"]:
                print(f"    Best partial: {best_partial} ({best_partial_bpb:.4f})")
        elif best_partial and best_partial_bpb < no_sleep["mean_bpb"] - 0.05:
            bp = summary[best_partial]
            _, bp_p = welch_ttest(no_sleep["bpbs"], bp["bpbs"])
            if bp_p < 0.05:
                print(f"    >>> ADOPT PARTIAL ({best_partial}) <<<")
                print(f"    Partial sleep helps ({best_partial_bpb:.4f}) but full cycle does not.")
            else:
                print(f"    >>> HOLD <<<")
                print(f"    Partial trends better but not significant.")
        else:
            print(f"    >>> SLEEP NOT HELPFUL YET <<<")
            print(f"    No sleep variant significantly outperforms the baseline.")
            print(f"    Consider longer training budgets or architecture changes.")

    # Save summary
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
