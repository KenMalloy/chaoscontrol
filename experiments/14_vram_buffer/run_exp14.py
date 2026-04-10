#!/usr/bin/env python3
"""VRAM-resident typed KV buffer ablation sweep.

Tests whether storing per-token key-value pairs in VRAM, organized by
Wernicke bucket type, improves language model bpb compared to the
consolidation-based episodic memory and bare baselines (SSM, Transformer,
Mamba2). The sweep is structured in three phases:

Phase A -- Core ablation (T2 + T3):
  T2: Retrieval mode x capacity — how the model selects from stored entries
      (softmax-all, bucket-mean, bucket-recent, bucket-topk) across buffer
      sizes (capped vs uncapped). 8 conditions x 7 seeds = 56 runs.
  T3: Wernicke routing structure — flat vs hierarchical codebooks at matched
      parameter budgets. 5 conditions x 7 seeds = 35 runs.

Phase B -- Developmental fast weights (T5, contingent on Phase A):
  T5: Fast weight freeze schedule. 5 conditions x 7 seeds = 35 runs.

Phase C -- Composition + confirmation (T6 + T7):
  T6: Combines Phase A/B winners with baselines. 5+ conditions x 7 seeds.
  T7: Locked confirmation of T6 winner vs strongest baseline with 8 fresh
      seeds and a single pre-registered statistical test. 2 x 8 = 16 runs.

Run with:
  python experiments/14_vram_buffer/run_exp14.py \\
      --data-path /workspace/fineweb_data/datasets/fineweb10B_byte260 \\
      --budget 600 --num-gpus 8 --phase A
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

# -- Seeds --
SWEEP_SEEDS = [1337, 2674, 4011, 5348, 6685, 8022, 9359]
T7_SEEDS = [10696, 12033, 13370, 14707, 16044, 17381, 18718, 20055]  # fresh


# -- Base config --

def _base(**overrides) -> dict:
    """Full stack base config for typed buffer experiments."""
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
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
        # Memory: append-only typed buffer
        "outer_model_type": "multislot",
        "outer_model_dim": 64,
        "outer_max_slots": 0,  # unlimited
        "buffer_mode": "append_only",
        # Wernicke: MoE router
        "wernicke_enabled": True,
        "wernicke_router": "moe",
        "wernicke_k_max": 16,
    }
    base.update(overrides)
    return base


# -- T2: Retrieval mode x capacity --
# Condition naming: {retrieval_mode}_{capacity}[_{k}]
#   e.g. "topk_uncapped_8" = bucket_topk retrieval, no slot cap, k=8 neighbors

T2_CONDITIONS = {
    # softmax_all_32 uses legacy buffer_mode intentionally as the
    # current-system baseline — this is the exp 9/11/13 architecture
    # with softmax over all slots, NOT the new typed-bucket retrieval.
    "softmax_all_32": _base(
        outer_max_slots=32, retrieval_mode="softmax_all",
        buffer_mode="legacy",
    ),
    "softmax_all_uncapped": _base(
        outer_max_slots=0, retrieval_mode="softmax_all",
    ),
    "mean_uncapped": _base(
        outer_max_slots=0, retrieval_mode="bucket_mean",
    ),
    "recent_uncapped": _base(
        outer_max_slots=0, retrieval_mode="bucket_recent", retrieval_k=8,
    ),
    "topk_32_8": _base(
        outer_max_slots=32, retrieval_mode="bucket_topk", retrieval_k=8,
    ),
    "topk_uncapped_4": _base(
        outer_max_slots=0, retrieval_mode="bucket_topk", retrieval_k=4,
    ),
    "topk_uncapped_8": _base(
        outer_max_slots=0, retrieval_mode="bucket_topk", retrieval_k=8,
    ),
    "topk_uncapped_16": _base(
        outer_max_slots=0, retrieval_mode="bucket_topk", retrieval_k=16,
    ),
}

# -- T3: Wernicke structure (param-matched) --

def _expert_dim_for_k(k_max: int, target_params: int = 16 * 128) -> int:
    """Scale expert_dim inversely with k_max to hold total expert budget ~constant."""
    return max(8, target_params // k_max)


def _expert_dim_for_hier(k_coarse: int, k_fine: int, target_params: int = 16 * 128) -> int:
    """Scale expert_dim for hierarchical configs based on total expert count.

    For hierarchical routing, the total number of experts is k_coarse + k_fine
    (coarse-layer experts plus fine-layer experts). Scale expert_dim inversely
    with that total to keep the parameter budget approximately constant.
    """
    total_experts = k_coarse + k_fine
    return max(8, target_params // total_experts)


T3_CONDITIONS = {
    "flat_8": _base(
        wernicke_k_max=8,
        wernicke_expert_dim=_expert_dim_for_k(8),
        retrieval_mode="bucket_topk", retrieval_k=8,
    ),
    "flat_16": _base(
        wernicke_k_max=16,
        wernicke_expert_dim=_expert_dim_for_k(16),
        retrieval_mode="bucket_topk", retrieval_k=8,
    ),
    "flat_64": _base(
        wernicke_k_max=64,
        wernicke_expert_dim=_expert_dim_for_k(64),
        retrieval_mode="bucket_topk", retrieval_k=8,
    ),
    "hier_8x8": _base(
        wernicke_layers=2,
        wernicke_k_max=8,
        wernicke_k_max_fine=8,
        wernicke_expert_dim=_expert_dim_for_hier(8, 8),
        retrieval_mode="bucket_topk", retrieval_k=8,
    ),
    "hier_8x32": _base(
        wernicke_layers=2,
        wernicke_k_max=8,
        wernicke_k_max_fine=32,
        wernicke_expert_dim=_expert_dim_for_hier(8, 32),
        retrieval_mode="bucket_topk", retrieval_k=8,
    ),
}

# -- T6: Composition (exploratory) --

T6_CONDITIONS = {
    "bare_ssm": {
        "model_type": "ssm",
        "vocab_size": 256, "model_dim": 128, "num_layers": 4,
        "ff_mult": 2, "seq_len": 256, "stride": 128,
        "batch_size": 32, "base_lr": 2e-3,
        "a_mode": "diag", "crit_target_coupling": 0.92,
        "outer_model_dim": 0,
        "wernicke_enabled": False,
    },
    "transformer": {
        "model_type": "transformer",
        "vocab_size": 256, "model_dim": 128, "num_layers": 4,
        "ff_mult": 2, "seq_len": 256, "stride": 128,
        "batch_size": 32, "base_lr": 2e-3,
    },
    "mamba2": {
        "model_type": "mamba2",
        "vocab_size": 256, "model_dim": 128, "num_layers": 4,
        "seq_len": 256, "stride": 128,
        "batch_size": 32, "base_lr": 2e-3,
    },
}
# Dynamic T6 conditions injected by inject_dynamic_t6_conditions():
#   claim1_winner — combines best T2 retrieval mode + best T3 Wernicke structure
#   current_best  — hardcoded winner from prior experiments (consolidation-based memory)
#   full_winner   — claim1_winner + best T5 fast-weight schedule (if Phase B ran)


# -- Dynamic T6 condition injection from Phase A results --

def inject_dynamic_t6_conditions() -> None:
    """Load Phase A results and inject dynamic T6 conditions.

    1. Load T2 and T3 summary JSONs to identify winners.
    2. Build claim1_winner from combining T2 retrieval winner + T3 structure winner.
    3. Build current_best from exp 9/11/13 hardcoded winners.
    4. If Phase B results exist, build full_winner from claim1 + T5 winner.
    5. Inject all into T6_CONDITIONS.
    """
    t2_summary_path = RESULTS / "t2_retrieval_mode_x_capacity_summary.json"
    t3_summary_path = RESULTS / "t3_wernicke_structure_summary.json"

    t2_winner_name = None
    t3_winner_name = None

    if t2_summary_path.exists():
        with open(t2_summary_path) as f:
            t2_summary = json.load(f)
        t2_winner_name = min(t2_summary, key=lambda c: t2_summary[c]["mean_bpb"])
        print(f"  T2 winner: {t2_winner_name} ({t2_summary[t2_winner_name]['mean_bpb']:.4f} bpb)")

    if t3_summary_path.exists():
        with open(t3_summary_path) as f:
            t3_summary = json.load(f)
        t3_winner_name = min(t3_summary, key=lambda c: t3_summary[c]["mean_bpb"])
        print(f"  T3 winner: {t3_winner_name} ({t3_summary[t3_winner_name]['mean_bpb']:.4f} bpb)")

    # Build claim1_winner: combine T2 retrieval mode + T3 Wernicke structure
    if t2_winner_name and t3_winner_name:
        t2_config = T2_CONDITIONS.get(t2_winner_name, {})
        t3_config = T3_CONDITIONS.get(t3_winner_name, {})
        claim1 = dict(_base())
        # Take retrieval settings from T2 winner
        for key in ("retrieval_mode", "retrieval_k", "outer_max_slots"):
            if key in t2_config:
                claim1[key] = t2_config[key]
        # Take Wernicke structure from T3 winner
        for key in ("wernicke_k_max", "wernicke_k_max_fine", "wernicke_layers",
                     "wernicke_expert_dim"):
            if key in t3_config:
                claim1[key] = t3_config[key]
        T6_CONDITIONS["claim1_winner"] = claim1

    # Build current_best: hardcoded from exp 9/11/13 winners
    T6_CONDITIONS["current_best"] = _base(
        outer_max_slots=32,
        retrieval_mode="softmax_all",
        buffer_mode="legacy",
        wernicke_k_max=16,
        semantic_tier_bases=8,
    )

    # Build full_winner if Phase B (T5) results exist
    t5_summary_path = RESULTS / "t5_developmental_fast_weights_summary.json"
    if t5_summary_path.exists() and "claim1_winner" in T6_CONDITIONS:
        with open(t5_summary_path) as f:
            t5_summary = json.load(f)
        t5_winner_name = min(t5_summary, key=lambda c: t5_summary[c]["mean_bpb"])
        print(f"  T5 winner: {t5_winner_name} ({t5_summary[t5_winner_name]['mean_bpb']:.4f} bpb)")
        full = dict(T6_CONDITIONS["claim1_winner"])
        # Add T5 fast weight settings if applicable
        full["_t5_winner"] = t5_winner_name
        T6_CONDITIONS["full_winner"] = full


# -- Launch and grid helpers --

def _launch(
    name: str, config: dict, seed: int, budget: float,
    data_path: str, gpu_id: int | None,
) -> tuple[subprocess.Popen, Path, Path, object]:
    config = dict(config, seed=seed)
    tag = f"{name}_s{seed}"
    (EXPERIMENT / "configs").mkdir(parents=True, exist_ok=True)
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


def run_grid(conditions: dict, seeds: list[int], data_path: str, budget: float, num_gpus: int, phase_label: str):
    """Run a grid of conditions x seeds, parallelizing across GPUs."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    queue: list[tuple[str, dict, int]] = []
    for cond_name, cond_config in conditions.items():
        for seed in seeds:
            tag = f"{cond_name}_s{seed}"
            if not (RESULTS / f"{tag}.json").exists():
                queue.append((cond_name, cond_config, seed))

    total = len(conditions) * len(seeds)
    completed = total - len(queue)
    print(f"\n  Phase {phase_label}: {len(queue)} pending, {completed} done, {total} total")
    print(f"  {num_gpus} GPUs, ~{len(queue) // max(num_gpus, 1) + (1 if len(queue) % max(num_gpus, 1) else 0)} batches")

    for batch_start in range(0, len(queue), max(num_gpus, 1)):
        batch = queue[batch_start:batch_start + max(num_gpus, 1)]
        batch_num = batch_start // max(num_gpus, 1) + 1
        total_batches = len(queue) // max(num_gpus, 1) + (1 if len(queue) % max(num_gpus, 1) else 0)
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
                bpb = result.get("eval", {}).get("bpb", "?")
                steps = result.get("train", {}).get("steps", "?")
                params = result.get("params", "?")
                warming = result.get("warming_curve", {})
                warming_str = ""
                if warming:
                    warming_str = "  warming=[" + ", ".join(
                        f"{n}:{b:.3f}" for n, b in sorted(warming.items(), key=lambda x: int(x[0]))
                    ) + "]"
                print(f"    {cond_name} seed={seed}: bpb={bpb}  steps={steps}  params={params}{warming_str}")

        print(f"  [{completed}/{total}]")


# -- Analysis --

def _load_results() -> dict[str, dict[int, dict]]:
    """Load per-condition, per-seed results from JSON files.
    Returns {condition_name: {seed: result_dict}}.
    """
    results: dict[str, dict[int, dict]] = {}
    for f in sorted(RESULTS.glob("*.json")):
        if f.name.endswith("_summary.json"):
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
        try:
            seed = int(parts[1])
        except ValueError:
            continue
        bpb = data.get("eval", {}).get("bpb")
        if bpb is None:
            continue
        results.setdefault(cond_name, {})[seed] = data
    return results


def _paired_deltas(results: dict[str, dict[int, dict]], cond_a: str, cond_b: str) -> list[float]:
    a_data = results.get(cond_a, {})
    b_data = results.get(cond_b, {})
    shared_seeds = sorted(set(a_data.keys()) & set(b_data.keys()))
    return [b_data[s]["eval"]["bpb"] - a_data[s]["eval"]["bpb"] for s in shared_seeds]


def _wilcoxon_signed_rank_p(deltas: list[float]) -> float:
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
    p = 2.0 * count / (1 << n)
    return min(1.0, p)


def _holm_bonferroni(p_values: list[tuple[str, float]]) -> list[tuple[str, float, float, bool]]:
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


def print_summary(conditions: dict, label: str):
    """Print summary table and statistical comparisons."""
    results = _load_results()
    if not results:
        print(f"\n  No results found for {label}.")
        return

    print(f"\n{'='*80}")
    print(f"  {label} RESULTS")
    print(f"{'='*80}")
    print(f"\n  {'Condition':<30} {'mean bpb':>10} {'SEM':>8} {'95% CI':>18} {'n':>3} {'params':>10}")
    print(f"  {'-'*83}")

    cond_bpbs: dict[str, list[float]] = {}
    for cond_name in conditions:
        seed_data = results.get(cond_name, {})
        if not seed_data:
            print(f"  {cond_name:<30} {'--':>10}")
            continue
        bpbs = [seed_data[s]["eval"]["bpb"] for s in sorted(seed_data.keys())]
        cond_bpbs[cond_name] = bpbs
        mean_bpb = sum(bpbs) / len(bpbs)
        s = sem(bpbs)
        ci = bootstrap_ci(bpbs)
        # Get params from first seed
        first_seed = sorted(seed_data.keys())[0]
        params = seed_data[first_seed].get("params", "?")
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"  {cond_name:<30} {mean_bpb:>10.4f} {s:>8.4f} {ci_str:>18} {len(bpbs):>3} {params:>10}")

    # Find best condition
    if cond_bpbs:
        best_cond = min(cond_bpbs, key=lambda c: sum(cond_bpbs[c]) / len(cond_bpbs[c]))
        best_bpb = sum(cond_bpbs[best_cond]) / len(cond_bpbs[best_cond])
        print(f"\n  Best: {best_cond} ({best_bpb:.4f} bpb)")

    # Save summary
    summary_path = RESULTS / f"{label.lower().replace(' ', '_')}_summary.json"
    summary = {}
    for cond_name, bpbs in cond_bpbs.items():
        summary[cond_name] = {
            "mean_bpb": sum(bpbs) / len(bpbs),
            "sem": sem(bpbs),
            "ci": list(bootstrap_ci(bpbs)),
            "n": len(bpbs),
            "bpbs": bpbs,
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")


def _identify_t6_winner_and_baseline() -> tuple[str | None, dict | None, str | None, dict | None]:
    """Find the T6 winner (lowest mean bpb) and strongest baseline.

    Returns (winner_name, winner_config, baseline_name, baseline_config).
    Baselines are bare_ssm and transformer.
    """
    results = _load_results()
    baseline_names = {"bare_ssm", "transformer", "mamba2"}

    cond_means: dict[str, float] = {}
    for cond_name in T6_CONDITIONS:
        seed_data = results.get(cond_name, {})
        if seed_data:
            bpbs = [seed_data[s]["eval"]["bpb"] for s in sorted(seed_data.keys())]
            cond_means[cond_name] = sum(bpbs) / len(bpbs)

    if not cond_means:
        return None, None, None, None

    # Winner: lowest bpb across ALL conditions
    winner_name = min(cond_means, key=cond_means.get)

    # Strongest baseline: lowest bpb among baselines
    baseline_means = {c: m for c, m in cond_means.items() if c in baseline_names}
    baseline_name = min(baseline_means, key=baseline_means.get) if baseline_means else None

    return (
        winner_name, T6_CONDITIONS.get(winner_name),
        baseline_name, T6_CONDITIONS.get(baseline_name) if baseline_name else None,
    )


def print_t7_confirmation(conditions: dict):
    """Print T7 confirmation results with locked statistical test.

    T7 uses strict confirmation discipline: single pre-specified contrast,
    8 fresh seeds, locked held-out validation, no extra pairwise tests.
    """
    results = _load_results()
    cond_names = list(conditions.keys())
    if len(cond_names) != 2:
        print("  T7 requires exactly 2 conditions.")
        return

    cond_a, cond_b = cond_names[0], cond_names[1]
    deltas = _paired_deltas(results, cond_a, cond_b)
    if len(deltas) < 2:
        print("  Insufficient paired data for T7.")
        return

    p = _wilcoxon_signed_rank_p(deltas)
    a_bpbs = [results[cond_a][s]["eval"]["bpb"] for s in sorted(set(results.get(cond_a, {}).keys()) & set(results.get(cond_b, {}).keys()))]
    b_bpbs = [results[cond_b][s]["eval"]["bpb"] for s in sorted(set(results.get(cond_a, {}).keys()) & set(results.get(cond_b, {}).keys()))]

    print(f"\n{'='*80}")
    print(f"  T7 CONFIRMATION (locked test, 8 fresh seeds)")
    print(f"{'='*80}")
    print(f"  {cond_a}: {sum(a_bpbs)/len(a_bpbs):.4f} bpb")
    print(f"  {cond_b}: {sum(b_bpbs)/len(b_bpbs):.4f} bpb")
    print(f"  mean delta: {sum(deltas)/len(deltas):+.4f}")
    print(f"  Wilcoxon p: {p:.4f}  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    print(f"  Cohen's d: {cohens_d(a_bpbs, b_bpbs):.2f}")
    delta_ci = bootstrap_ci(deltas)
    print(f"  95% CI of delta: [{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="VRAM-resident typed KV buffer ablation sweep"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--budget", type=float, default=600)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--phase", choices=["A", "B", "C", "all"], default="A")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print condition list and exit")
    args = parser.parse_args()

    if args.phase in ("A", "all"):
        conditions_a = {**T2_CONDITIONS, **T3_CONDITIONS}
        if args.dry_run:
            print(f"\n  Phase A: {len(conditions_a)} conditions x {len(SWEEP_SEEDS)} seeds = {len(conditions_a) * len(SWEEP_SEEDS)} runs")
            for name in conditions_a:
                print(f"    {name}")
            if args.phase == "A":
                return
        else:
            t0 = time.time()
            run_grid(T2_CONDITIONS, SWEEP_SEEDS, args.data_path, args.budget, args.num_gpus, "T2")
            print_summary(T2_CONDITIONS, "T2 Retrieval Mode x Capacity")
            run_grid(T3_CONDITIONS, SWEEP_SEEDS, args.data_path, args.budget, args.num_gpus, "T3")
            print_summary(T3_CONDITIONS, "T3 Wernicke Structure")
            print(f"\n  Phase A wall time: {(time.time() - t0)/60:.1f} min")

    if args.phase == "B":
        print("\n  Phase B (developmental fast weights) is deferred. Run after Phase A results.")
        return

    if args.phase in ("C", "all"):
        if args.dry_run:
            print(f"\n  Phase C (T6): {len(T6_CONDITIONS)} base conditions x {len(SWEEP_SEEDS)} seeds")
            print(f"  Phase C (T7): 2 conditions x {len(T7_SEEDS)} fresh seeds = {2 * len(T7_SEEDS)} runs")
            for name in T6_CONDITIONS:
                print(f"    T6: {name}")
            return

        t0 = time.time()

        # Inject dynamic T6 conditions from Phase A results
        print("\n  Injecting dynamic T6 conditions from Phase A results...")
        inject_dynamic_t6_conditions()

        run_grid(T6_CONDITIONS, SWEEP_SEEDS, args.data_path, args.budget, args.num_gpus, "T6")
        print_summary(T6_CONDITIONS, "T6 Composition")

        # T7: Identify T6 winner and strongest baseline, run confirmation
        winner_name, winner_config, baseline_name, baseline_config = _identify_t6_winner_and_baseline()
        if winner_name and baseline_name and winner_config and baseline_config:
            t7_conditions = {
                f"t7_{winner_name}": winner_config,
                f"t7_{baseline_name}": baseline_config,
            }
            print(f"\n  T7 contrast: {winner_name} vs {baseline_name}")
            run_grid(t7_conditions, T7_SEEDS, args.data_path, args.budget, args.num_gpus, "T7")
            print_t7_confirmation(t7_conditions)
        else:
            print("\n  T7 skipped: insufficient T6 results to identify winner and baseline.")

        print(f"\n  Phase C wall time: {(time.time() - t0)/60:.1f} min")

    total_elapsed = 0
    print(f"\n  Results directory: {RESULTS}")


if __name__ == "__main__":
    main()
