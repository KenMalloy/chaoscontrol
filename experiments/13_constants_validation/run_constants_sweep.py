#!/usr/bin/env python3
"""Experiment 13: Constants validation — exploratory sweeps.

Two independent 1-D sweeps on the highest-priority unvalidated constants
identified in docs/plans/constants-audit.md. Both sweeps are EXPLORATORY.
Confirmation requires a second-stage rerun on 8+ fresh paired seeds.

Sweep 1 — Criticality target (crit_target_coupling)
  Bare SSM (no memory, no Wernicke) to isolate trunk dynamics.
  5 values x 7 seeds = 35 runs.
  NOTE: If winner lands on edge (0.80 or 0.96), extend range before locking.
  NOTE: A small confirmatory check on the locked full stack is needed before
  setting a repo-wide default, since Wernicke/memory may shift the optimum.

Sweep 2 — Memory slot dimension (outer_model_dim)
  Full stack without sleep. Isolates memory embedding capacity.
  3 values x 7 seeds = 21 runs.

Sweep 3 — Max slots (outer_max_slots)
  Full stack without sleep. Isolates memory capacity.
  3 values x 7 seeds = 21 runs.

Sweep 4 — Merge similarity threshold (sleep_merge_sim_threshold)
  Full stack with full_cycle sleep (PROVISIONAL — may need re-run if exp 11
  picks a non-full_cycle payload or k_max sweep changes from 16).
  5 values x 7 seeds = 35 runs.
  NOTE: Threshold filters candidates by raw similarity, then affinity reorders
  them via merge_score = sim * affinity. So this sweep tests "how many
  candidates pass the gate", not the affinity ranking itself.

wernicke_k_max sweep is covered by experiments/baselines/run_mamba2_baseline.py,
not duplicated here.

Total: 16 conditions x 7 seeds = 112 runs.
At 4 GPUs: ~28 batches x 10 min = ~5 hours.

Statistical discipline: 7 seeds, paired Wilcoxon, bootstrap CIs.
All contrasts are EXPLORATORY (post-selection from 5-point grid).
Confirmation: rerun winner vs default on 8+ fresh seeds after locks.
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
CONFIGS = EXPERIMENT / "configs"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
from stats import bootstrap_ci, cohens_d, sem

SEEDS = [1337, 2674, 4011, 5348, 6685, 8022, 9359]


# -- Config templates -------------------------------------------------------

def _bare_ssm(**overrides) -> dict:
    """Bare SSM — no memory, no Wernicke. Isolates trunk dynamics."""
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
    }
    base.update(overrides)
    return base


def _full_stack(**overrides) -> dict:
    """Full stack without sleep. Tests memory parameters in isolation."""
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


def _full_stack_sleep(**overrides) -> dict:
    """Full stack + full_cycle sleep. Tests consolidation parameters."""
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
        # Sleep
        "sleep_enabled": True,
        "sleep_stages": "full_cycle",
        "sleep_interval": 256,
        "sleep_budget": 128,
        "sleep_n2_budget": 64,
        "sleep_rem_budget": 64,
    }
    base.update(overrides)
    return base


# -- Conditions --------------------------------------------------------------

CONDITIONS = {}

# Sweep 1: Criticality target (bare SSM)
for coupling in [0.80, 0.85, 0.88, 0.92, 0.96]:
    name = f"crit_{int(coupling*100):03d}"
    CONDITIONS[name] = _bare_ssm(crit_target_coupling=coupling)

# Sweep 2: Memory slot dimension (full stack, no sleep)
for dim in [32, 64, 128]:
    name = f"memdim_{dim:03d}"
    CONDITIONS[name] = _full_stack(outer_model_dim=dim)

# Sweep 3: Max slots (full stack, no sleep)
for slots in [32, 64, 128]:
    name = f"slots_{slots:03d}"
    CONDITIONS[name] = _full_stack(outer_max_slots=slots)

# Sweep 4: Merge similarity threshold (full stack + sleep)
for threshold in [0.75, 0.80, 0.85, 0.90, 0.95]:
    name = f"merge_{int(threshold*100):03d}"
    CONDITIONS[name] = _full_stack_sleep(sleep_merge_sim_threshold=threshold)


# -- Statistical helpers (same as exp 11) ------------------------------------

def _paired_deltas(results, cond_a, cond_b):
    a_data = results.get(cond_a, {})
    b_data = results.get(cond_b, {})
    shared = sorted(set(a_data.keys()) & set(b_data.keys()))
    return [b_data[s] - a_data[s] for s in shared]


def _wilcoxon_signed_rank_p(deltas):
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    nonzero.sort(key=lambda x: x[0])
    w_plus = sum(rank + 1 for rank, (_, sign) in enumerate(nonzero) if sign > 0)
    w_minus = sum(rank + 1 for rank, (_, sign) in enumerate(nonzero) if sign < 0)
    w = min(w_plus, w_minus)
    count = 0
    for mask in range(1 << n):
        w_test = sum((rank + 1) for rank in range(n) if mask & (1 << rank))
        if w_test <= w:
            count += 1
    return min(1.0, 2.0 * count / (1 << n))


# -- Execution ---------------------------------------------------------------

def _launch(name, config, seed, budget, data_path, gpu_id):
    config = dict(config, seed=seed)
    tag = f"{name}_s{seed}"
    out_path = RESULTS / f"{tag}.json"
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        dir=CONFIGS, prefix=f".tmp_{tag}_",
    )
    yaml.dump(config, tmp)
    tmp.close()
    cmd = [
        sys.executable, "-m", "chaoscontrol.runner",
        "--config", tmp.name,
        "--data-path", data_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
        "--checkpoint-dir", str(EXPERIMENT / "checkpoints"),
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
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
        )
    return proc, out_path, Path(tmp.name), log_fh


def run_grid(data_path, budget, num_gpus):
    RESULTS.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "checkpoints").mkdir(parents=True, exist_ok=True)

    queue = []
    for cond_name, cond_config in CONDITIONS.items():
        for seed in SEEDS:
            tag = f"{cond_name}_s{seed}"
            if not (RESULTS / f"{tag}.json").exists():
                queue.append((cond_name, cond_config, seed))

    total = len(CONDITIONS) * len(SEEDS)
    succeeded = total - len(queue)
    n_failed = len(list(RESULTS.glob("*.failed")))
    print(f"\n  {len(queue)} runs pending, {succeeded} succeeded, {n_failed} previously failed, {total} total")
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
            if proc.returncode != 0:
                tag = f"{cond_name}_s{seed}"
                failed_path = RESULTS / f"{tag}.failed"
                log_path = RESULTS / f"{tag}.log"
                error_tail = ""
                if log_path.exists():
                    error_tail = log_path.read_text()[-500:]
                failed_path.write_text(json.dumps({
                    "condition": cond_name, "seed": seed,
                    "exit_code": proc.returncode, "error_tail": error_tail,
                }))
                print(f"    FAILED: {cond_name} seed={seed} (exit {proc.returncode})")
                continue
            if out_path.exists():
                with open(out_path) as f:
                    result = json.load(f)
                bpb = result["eval"]["bpb"]
                steps = result["train"]["steps"]
                print(f"    {cond_name} seed={seed}: bpb={bpb:.4f}  steps={steps}")

        n_json = len([f for f in RESULTS.glob("*.json") if f.name != "constants_summary.json"])
        n_fail = len(list(RESULTS.glob("*.failed")))
        print(f"  [{n_json} succeeded, {n_fail} failed / {total} total]")

    _print_summary()


def _load_results():
    results = {}
    for f in sorted(RESULTS.glob("*.json")):
        if f.name == "constants_summary.json":
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
        if bpb is not None:
            results.setdefault(cond_name, {})[seed] = bpb
    return results


def _print_summary():
    results = _load_results()
    if not results:
        print("\n  No results found.")
        return

    print(f"\n{'='*70}")
    print("  CONSTANTS VALIDATION RESULTS")
    print(f"{'='*70}")

    # -- Per-sweep summaries --
    for sweep_name, prefix, default_name in [
        ("Criticality target (bare SSM)", "crit_", "crit_088"),
        ("Memory slot dimension (full stack)", "memdim_", "memdim_064"),
        ("Max slots (full stack)", "slots_", "slots_064"),
        ("Merge threshold (full stack + sleep)", "merge_", "merge_085"),
    ]:
        print(f"\n  {sweep_name}:")
        print(f"  {'-'*60}")
        print(f"  {'Condition':<20} {'mean bpb':>10} {'SEM':>8} {'95% CI':>18} {'n':>3}")

        sweep_conds = sorted([c for c in CONDITIONS if c.startswith(prefix)])
        best_name = None
        best_bpb = float("inf")

        for cond_name in sweep_conds:
            seed_bpbs = results.get(cond_name, {})
            if not seed_bpbs:
                print(f"  {cond_name:<20} {'--':>10}")
                continue
            bpbs = list(seed_bpbs.values())
            mean_bpb = sum(bpbs) / len(bpbs)
            s = sem(bpbs)
            ci = bootstrap_ci(bpbs)
            marker = " <-- default" if cond_name == default_name else ""
            print(f"  {cond_name:<20} {mean_bpb:>10.4f} {s:>8.4f} [{ci[0]:.4f}, {ci[1]:.4f}] {len(bpbs):>3}{marker}")
            if mean_bpb < best_bpb:
                best_bpb = mean_bpb
                best_name = cond_name

        # Exploratory contrast: best vs default (post-selected, NOT confirmatory)
        if best_name and best_name != default_name and default_name in results:
            deltas = _paired_deltas(results, default_name, best_name)
            if len(deltas) >= 2:
                p = _wilcoxon_signed_rank_p(deltas)
                mean_d = sum(deltas) / len(deltas)
                d_bpbs = list(results[default_name].values())
                b_bpbs = list(results[best_name].values())
                d = cohens_d(d_bpbs, b_bpbs) if d_bpbs and b_bpbs else 0.0
                delta_ci = bootstrap_ci(deltas)
                print(f"\n  [EXPLORATORY] Best ({best_name}) vs default ({default_name}):")
                print(f"    delta={mean_d:+.4f}  Wilcoxon p={p:.4f} (uncorrected, post-selected)  d={d:.2f}")
                print(f"    95% CI of delta: [{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}]")
                if best_name.endswith("080") or best_name.endswith("096") or best_name.endswith("075") or best_name.endswith("095"):
                    print(f"    WARNING: winner is at sweep edge — extend range before locking")
                if mean_d < 0:
                    print(f"    CANDIDATE: {best_name} for confirmatory rerun on 8+ fresh seeds")
                else:
                    print(f"    Default {default_name} appears optimal in this range")

    with open(RESULTS / "constants_summary.json", "w") as f:
        json.dump({k: dict(v) for k, v in results.items()}, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 13: Constants validation sweeps"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--budget", type=float, default=600)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    t0 = time.time()
    run_grid(args.data_path, args.budget, args.num_gpus)
    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed/60:.1f} minutes")
    print(f"  Results: {RESULTS}")


if __name__ == "__main__":
    main()
