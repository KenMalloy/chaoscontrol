#!/usr/bin/env python3
"""Architecture baseline ablation: Mamba-2 vs ChaosControl SSM vs full stack.

Param-matched comparison for the paper. Three models at the same budget,
dimensions, and layer count. Answers: "how much does each architectural
layer contribute?"

8 conditions x 7 seeds = 56 runs

Conditions:
  Architecture baselines:
    1. mamba2             -- external SSM baseline (requires mamba-ssm)
    2. bare_ssm           -- our SSM core, no memory, no Wernicke
    3. ssm_wernicke_k16   -- SSM + Wernicke MoE (k_max=16)
    4. full_stack_k16     -- SSM + Wernicke + episodic memory (k_max=16)
  k_max sweep:
    5. ssm_wernicke_k32   -- Wernicke k_max=32
    6. ssm_wernicke_k64   -- Wernicke k_max=64
    7. full_stack_k32     -- full stack k_max=32
    8. full_stack_k64     -- full stack k_max=64

Confirmatory contrasts (Holm-corrected, m=2):
  1. full_stack_k16 vs mamba2   (does our full architecture beat Mamba-2?)
  2. bare_ssm vs mamba2         (is our SSM core competitive with Mamba-2?)

Exploratory:
  3. ssm_wernicke_k16 vs bare_ssm       (Wernicke contribution)
  4. full_stack_k16 vs ssm_wernicke_k16  (memory contribution)
  5-8. k_max sweep: 16→32, 16→64 for both Wernicke-only and full stack

Requires: pip install mamba-ssm>=2.3.0

Usage:
    python experiments/baselines/run_mamba2_baseline.py \
        --data-path /workspace/fineweb_data --budget 600 --num-gpus 4
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

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / "results"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
from stats import bootstrap_ci, sem

SEEDS = [1337, 2674, 4011, 5348, 6685, 8022, 9359]


def _base(**overrides) -> dict:
    """Shared base config — minimal, no memory, no Wernicke."""
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


# Expert param budget: k_max=16 at dim=128 uses 16*128*128 = 262,144 expert params.
# Bottleneck experts have 2 matrices (down + up): k_max * 2 * expert_dim * dim.
# To hold total ≈ 262K: expert_dim = 128*128*16 / (2 * k_max * 128) = 128*16/(2*k_max).
# k_max=16 → expert_dim=128 (full rank, no bottleneck needed)
# k_max=32 → expert_dim=32  (32*2*32*128 = 262,144)
# k_max=64 → expert_dim=16  (64*2*16*128 = 262,144)
EXPERT_DIM_FOR_K = {16: 0, 32: 32, 64: 16}  # 0 = full rank (default)

def _wernicke(k_max=16, **extra):
    return _base(
        wernicke_enabled=True, wernicke_router="moe",
        wernicke_k_max=k_max, typed_storage=True,
        wernicke_expert_dim=EXPERT_DIM_FOR_K.get(k_max, 0),
        **extra,
    )

def _full(k_max=16, **extra):
    return _wernicke(
        k_max=k_max,
        outer_model_type="multislot", outer_model_dim=64,
        outer_max_slots=64, consolidation_write="full_sequence",
        latent_persistence=True, **extra,
    )

CONDITIONS = {
    # Architecture baselines
    "mamba2": _base(model_type="mamba2"),
    "bare_ssm": _base(),
    "ssm_wernicke_k16": _wernicke(k_max=16),
    "full_stack_k16": _full(k_max=16),
    # k_max sweep — param-controlled (expert_dim shrinks as k_max grows)
    "ssm_wernicke_k32": _wernicke(k_max=32),
    "ssm_wernicke_k64": _wernicke(k_max=64),
    "full_stack_k32": _full(k_max=32),
    "full_stack_k64": _full(k_max=64),
}

# Wilcoxon signed-rank helpers (same as experiment 11)
def _paired_deltas(results, cond_a, cond_b):
    a_data = results.get(cond_a, {})
    b_data = results.get(cond_b, {})
    shared_seeds = sorted(set(a_data.keys()) & set(b_data.keys()))
    return [b_data[s] - a_data[s] for s in shared_seeds]


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


def _holm_bonferroni(p_values):
    m = len(p_values)
    sorted_pv = sorted(p_values, key=lambda x: x[1])
    results = []
    prev_corrected = 0.0
    for i, (name, p) in enumerate(sorted_pv):
        corrected = min(1.0, p * (m - i))
        corrected = max(corrected, prev_corrected)
        prev_corrected = corrected
        results.append((name, p, corrected, corrected < 0.05))
    return results


CONFIRMATORY_CONTRASTS = [
    ("full_vs_mamba2", "mamba2", "full_stack_k16", "Full stack vs Mamba-2"),
    ("bare_vs_mamba2", "mamba2", "bare_ssm", "Our SSM vs Mamba-2"),
]

EXPLORATORY_CONTRASTS = [
    ("wernicke_value", "bare_ssm", "ssm_wernicke_k16", "Wernicke MoE contribution"),
    ("memory_value", "ssm_wernicke_k16", "full_stack_k16", "Episodic memory contribution"),
    # k_max sweep (Wernicke only)
    ("kmax_w_16v32", "ssm_wernicke_k16", "ssm_wernicke_k32", "Wernicke k_max 16→32"),
    ("kmax_w_16v64", "ssm_wernicke_k16", "ssm_wernicke_k64", "Wernicke k_max 16→64"),
    # k_max sweep (full stack)
    ("kmax_f_16v32", "full_stack_k16", "full_stack_k32", "Full stack k_max 16→32"),
    ("kmax_f_16v64", "full_stack_k16", "full_stack_k64", "Full stack k_max 16→64"),
]


def _launch(name, config, seed, budget, data_path, gpu_id):
    config = dict(config, seed=seed)
    tag = f"{name}_s{seed}"
    out_path = RESULTS / f"{tag}.json"
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        dir=EXPERIMENT / "configs" if (EXPERIMENT / "configs").exists() else EXPERIMENT,
        prefix=f".tmp_{tag}_",
    )
    import yaml
    yaml.dump(config, tmp)
    tmp.close()
    cmd = [
        sys.executable, "-m", "chaoscontrol.runner",
        "--config", tmp.name,
        "--data-path", data_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path = RESULTS / f"{tag}.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", "-c", " ".join(shlex.quote(c) for c in cmd)
         + f" 2>&1 | tee {shlex.quote(str(log_path))}"
         + (" /proc/1/fd/1" if os.path.exists("/proc/1/fd/1") else "")],
        env=env, stdout=log_fh, stderr=subprocess.STDOUT,
    )
    return proc, out_path, Path(tmp.name), log_fh


def run_grid(data_path, budget, num_gpus):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "configs").mkdir(parents=True, exist_ok=True)

    queue = []
    for cond_name, cond_config in CONDITIONS.items():
        for seed in SEEDS:
            tag = f"{cond_name}_s{seed}"
            if not (RESULTS / f"{tag}.json").exists():
                queue.append((cond_name, cond_config, seed))

    total = len(CONDITIONS) * len(SEEDS)
    completed = total - len(queue)
    print(f"\n  {len(queue)} runs pending, {completed} done, {total} total")

    for batch_start in range(0, len(queue), num_gpus):
        batch = queue[batch_start:batch_start + num_gpus]
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

    # Summary
    print(f"\n{'='*60}")
    print("  ARCHITECTURE BASELINE RESULTS")
    print(f"{'='*60}")

    results = {}  # {cond: {seed: bpb}}
    for cond_name in CONDITIONS:
        results[cond_name] = {}
        for seed in SEEDS:
            path = RESULTS / f"{cond_name}_s{seed}.json"
            if path.exists():
                with open(path) as f:
                    results[cond_name][seed] = json.load(f)["eval"]["bpb"]

    print(f"\n  {'Condition':<20} {'mean bpb':>10} {'SEM':>8} {'95% CI':>18} {'n':>3}")
    print(f"  {'-'*62}")
    from stats import cohens_d
    for cond_name in CONDITIONS:
        bpbs = list(results.get(cond_name, {}).values())
        if bpbs:
            mean = sum(bpbs) / len(bpbs)
            s = sem(bpbs)
            ci = bootstrap_ci(bpbs)
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            print(f"  {cond_name:<20} {mean:>10.4f} {s:>8.4f} {ci_str:>18} {len(bpbs):>3}")
        else:
            print(f"  {cond_name:<20} {'--':>10}")

    # Confirmatory contrasts
    print(f"\n  Confirmatory contrasts (Holm-corrected, m={len(CONFIRMATORY_CONTRASTS)}):")
    print(f"  {'-'*62}")
    raw_pvals = []
    for name, cond_a, cond_b, desc in CONFIRMATORY_CONTRASTS:
        deltas = _paired_deltas(results, cond_a, cond_b)
        if len(deltas) < 2:
            print(f"    {desc}: insufficient data")
            continue
        p = _wilcoxon_signed_rank_p(deltas)
        raw_pvals.append((name, p))
    corrected = _holm_bonferroni(raw_pvals)
    corr_map = {n: (r, c, s) for n, r, c, s in corrected}
    for name, cond_a, cond_b, desc in CONFIRMATORY_CONTRASTS:
        deltas = _paired_deltas(results, cond_a, cond_b)
        if len(deltas) < 2:
            continue
        mean_d = sum(deltas) / len(deltas)
        winner = cond_b if mean_d < 0 else cond_a
        a_bpbs = list(results.get(cond_a, {}).values())
        b_bpbs = list(results.get(cond_b, {}).values())
        d = cohens_d(a_bpbs, b_bpbs) if a_bpbs and b_bpbs else 0.0
        raw_p, corr_p, sig = corr_map.get(name, (1.0, 1.0, False))
        sig_str = "SIGNIFICANT" if sig else "ns"
        print(f"    {desc}")
        print(f"      delta={mean_d:+.4f} ({winner} wins) p={raw_p:.4f} corrected={corr_p:.4f} {sig_str} d={d:.2f}")

    # Exploratory contrasts
    print(f"\n  Exploratory contrasts (uncorrected):")
    print(f"  {'-'*62}")
    for name, cond_a, cond_b, desc in EXPLORATORY_CONTRASTS:
        deltas = _paired_deltas(results, cond_a, cond_b)
        if len(deltas) < 2:
            print(f"    {desc}: insufficient data")
            continue
        mean_d = sum(deltas) / len(deltas)
        winner = cond_b if mean_d < 0 else cond_a
        p = _wilcoxon_signed_rank_p(deltas)
        a_bpbs = list(results.get(cond_a, {}).values())
        b_bpbs = list(results.get(cond_b, {}).values())
        d = cohens_d(a_bpbs, b_bpbs) if a_bpbs and b_bpbs else 0.0
        print(f"    [EXPLORATORY] {desc}")
        print(f"      delta={mean_d:+.4f} ({winner} wins) p={p:.4f} d={d:.2f}")

    # Save summary
    with open(RESULTS / "baseline_summary.json", "w") as f:
        json.dump({k: dict(v) for k, v in results.items()}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Mamba-2 baseline: param-matched comparison"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--budget", type=float, default=600)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    t0 = time.time()
    run_grid(args.data_path, args.budget, args.num_gpus)
    elapsed = time.time() - t0
    print(f"\n  Wall time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
