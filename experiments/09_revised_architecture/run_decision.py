#!/usr/bin/env python3
"""EXPLORATORY-ONLY — not paper-grade.

Uses unpaired Welch-style summaries via ``stats.welch_ttest`` and an older
exploratory decision workflow rather than the paired confirmatory framing
the paper requires. Per ``docs/plans/2026-04-17-paper-status.md``
§"Claims that are not safe yet" (item 4), numbers from this script MUST
NOT appear in any confirmatory paper table without being re-run under the
paired repeated-measures framing called out in §"Immediate next steps".

Decision experiment: go/no-go on H100 scale-up.

Tests whether the full ChaosControl stack (SSM + memory + Wernicke MoE)
outperforms bare SSM and transformer at increasing training budgets.

Grid: 4 configs x 3 budgets x 3 seeds = 36 training runs
Then Phase 2 eval ablation on full_stack checkpoints only.

Decision criteria:
  - full_stack beats bare_ssm at 600s + gate helps at eval → SCALE
  - full_stack beats bare_ssm at 600s + gate neutral → SCALE (rethink gate)
  - full_stack = bare_ssm at all budgets → DON'T SCALE
  - transformer beats all at all budgets → REWORK ARCHITECTURE
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
RESULTS = EXPERIMENT / "results_decision"
CHECKPOINTS = EXPERIMENT / "checkpoints_decision"

sys.path.insert(0, str(EXPERIMENT))
from stats import welch_ttest, bootstrap_ci, cohens_d, sem

SEEDS = [1337, 2674, 4011]
BUDGETS = [150, 300, 600]


# ── Config templates ────────────────────────────────────────────────


def _base(**overrides) -> dict:
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
    }
    base.update(overrides)
    return base


CONFIGS = {
    "bare_ssm": _base(),
    "full_stack": _base(
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=64, outer_compress_ratio=2,
        consolidation_write="full_sequence",
        latent_persistence=True,
        wernicke_enabled=True, wernicke_router="moe",
        wernicke_k_max=16, wernicke_window=8,
        typed_storage=True,
    ),
    "full_stack_sem": _base(
        outer_model_dim=64, outer_model_type="multislot",
        outer_max_slots=64, outer_compress_ratio=2,
        consolidation_write="full_sequence",
        latent_persistence=True,
        semantic_tier_bases=8,
        wernicke_enabled=True, wernicke_router="moe",
        wernicke_k_max=16, wernicke_window=8,
        typed_storage=True,
    ),
    "transformer": _base(model_type="transformer"),
}


# ── Execution ───────────────────────────────────────────────────────


def _launch(
    name: str, config: dict, seed: int, budget: float,
    data_path: str, gpu_id: int | None,
) -> tuple[subprocess.Popen, Path, Path, object]:
    config = dict(config, seed=seed)
    tag = f"{name}_b{int(budget)}_s{seed}"
    tmp = Path(tempfile.mktemp(suffix=".yaml", prefix=f".tmp_{tag}_",
                               dir=EXPERIMENT / "configs"))
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


def run_training_grid(data_path: str, num_gpus: int):
    """Run all 36 training configs, parallelizing seeds across GPUs."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    total = len(CONFIGS) * len(BUDGETS) * len(SEEDS)
    completed = 0

    for budget in BUDGETS:
        for config_name, config_dict in CONFIGS.items():
            label = f"{config_name}_b{budget}"
            print(f"\n{'='*60}")
            print(f"  {label}  ({len(SEEDS)} seeds, budget={budget}s)")
            print(f"{'='*60}")

            # Check for existing results (resume)
            existing = []
            for seed in SEEDS:
                tag = f"{config_name}_b{int(budget)}_s{seed}"
                path = RESULTS / f"{tag}.json"
                if path.exists():
                    existing.append(seed)
            if len(existing) == len(SEEDS):
                print(f"  Already done, skipping")
                completed += len(SEEDS)
                continue

            jobs = []
            for i, seed in enumerate(SEEDS):
                tag = f"{config_name}_b{int(budget)}_s{seed}"
                if (RESULTS / f"{tag}.json").exists():
                    completed += 1
                    continue
                gpu_id = i % num_gpus if num_gpus > 1 else None
                proc, out_path, tmp, log_fh = _launch(
                    config_name, config_dict, seed, budget, data_path, gpu_id,
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

    # Checkpoint results
    _checkpoint_summary()


def _checkpoint_summary():
    """Print summary table from completed results."""
    results = {}
    for f in RESULTS.glob("*.json"):
        if f.name == "decision_summary.json":
            continue
        with open(f) as fh:
            data = json.load(fh)
        # Parse tag from filename: {config}_b{budget}_s{seed}.json
        stem = f.stem
        parts = stem.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        prefix, seed_str = parts
        config_budget = prefix.rsplit("_b", 1)
        if len(config_budget) != 2:
            continue
        config_name, budget_str = config_budget
        key = (config_name, int(budget_str))
        results.setdefault(key, []).append(data["eval"]["bpb"])

    print(f"\n{'='*70}")
    print("  TRAINING RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Config':<20} {'Budget':>8} {'mean bpb':>10} {'SEM':>8} {'steps':>8} {'n':>3}")
    print(f"  {'-'*62}")

    summary = {}
    for (config_name, budget), bpbs in sorted(results.items()):
        mean_bpb = sum(bpbs) / len(bpbs)
        s = sem(bpbs)
        # Get steps from first result
        tag = f"{config_name}_b{budget}_s{SEEDS[0]}"
        steps_file = RESULTS / f"{tag}.json"
        steps = "?"
        if steps_file.exists():
            with open(steps_file) as fh:
                steps = json.load(fh)["train"]["steps"]
        print(f"  {config_name:<20} {budget:>7}s {mean_bpb:>10.4f} {s:>8.4f} {steps:>8} {len(bpbs):>3}")
        summary[(config_name, budget)] = {"mean_bpb": mean_bpb, "sem": s, "n": len(bpbs), "bpbs": bpbs}

    # Statistical comparisons at each budget
    print(f"\n  Statistical comparisons (Welch t-test):")
    for budget in BUDGETS:
        bare = summary.get(("bare_ssm", budget))
        full = summary.get(("full_stack", budget))
        tfm = summary.get(("transformer", budget))
        if bare and full:
            t, p = welch_ttest(bare["bpbs"], full["bpbs"])
            d = cohens_d(bare["bpbs"], full["bpbs"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            delta = full["mean_bpb"] - bare["mean_bpb"]
            direction = "full_stack wins" if delta < 0 else "bare_ssm wins"
            print(f"    {budget}s: bare vs full_stack: delta={delta:+.4f} ({direction}) p={p:.4f} {sig} d={d:.2f}")
        if bare and tfm:
            t, p = welch_ttest(bare["bpbs"], tfm["bpbs"])
            delta = tfm["mean_bpb"] - bare["mean_bpb"]
            direction = "transformer wins" if delta < 0 else "ssm wins"
            print(f"    {budget}s: bare_ssm vs transformer: delta={delta:+.4f} ({direction}) p={p:.4f}")

    # Save summary
    with open(RESULTS / "decision_summary.json", "w") as f:
        json.dump({f"{k[0]}_b{k[1]}": v for k, v in summary.items()}, f, indent=2, default=str)


def run_eval_ablation(data_path: str, num_gpus: int):
    """Run Phase 2 eval grid on full_stack checkpoints at 600s budget."""
    # Find full_stack checkpoints at highest budget
    best_budget = max(BUDGETS)
    ckpts = sorted(CHECKPOINTS.glob(f"full_stack_b{best_budget}_s*.pt"))
    if not ckpts:
        # Fall back to any full_stack checkpoint
        ckpts = sorted(CHECKPOINTS.glob("full_stack_*.pt"))
    if not ckpts:
        print("No full_stack checkpoints found. Skipping eval ablation.")
        return

    print(f"\n{'='*70}")
    print(f"  PHASE 2: Eval ablation on {len(ckpts)} full_stack checkpoints")
    print(f"{'='*70}")

    # Launch run_eval_ablation.py with --checkpoint-dir pointing to our checkpoints
    # But filter to only full_stack checkpoints by creating a temp dir with symlinks
    eval_ckpt_dir = CHECKPOINTS / "_eval_subset"
    eval_ckpt_dir.mkdir(exist_ok=True)
    for c in ckpts:
        link = eval_ckpt_dir / c.name
        if not link.exists():
            link.symlink_to(c)

    cmd = [
        sys.executable, str(EXPERIMENT / "run_eval_ablation.py"),
        "--checkpoint-dir", str(eval_ckpt_dir),
        "--data-path", data_path,
        "--num-gpus", str(num_gpus),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    eval_log = RESULTS / "eval_ablation.log"
    print(f"  Logging to {eval_log}")
    with open(eval_log, "w") as log_fh:
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        proc.wait()

    if proc.returncode != 0:
        print(f"  Eval ablation failed (exit {proc.returncode})")
        print(f"  Check {eval_log}")
        return

    # Read and summarize eval results
    eval_results_file = EXPERIMENT / "results_phase2" / "eval_results.json"
    if eval_results_file.exists():
        with open(eval_results_file) as f:
            eval_results = json.load(f)
        # Filter to full_stack checkpoints only
        full_stack_results = [r for r in eval_results if "full_stack" in r["checkpoint"]]
        if full_stack_results:
            by_gate = {}
            for r in full_stack_results:
                g = r["gate"]
                bpb = r.get("bpb_gated") or r["bpb"]
                by_gate.setdefault(g, []).append(bpb)
            print(f"\n  Eval ablation on full_stack checkpoints ({len(full_stack_results)} evals):")
            print(f"  {'Gate':<15} {'mean bpb':>10} {'SEM':>8} {'n':>5}")
            print(f"  {'-'*42}")
            for g, bpbs in sorted(by_gate.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
                print(f"  {g:<15} {sum(bpbs)/len(bpbs):>10.4f} {sem(bpbs):>8.4f} {len(bpbs):>5}")

            # Gate vs no-gate comparison
            none_bpbs = by_gate.get("none", [])
            for gate_name in ["fork_k4", "mcts_k4", "mcts_k8"]:
                gate_bpbs = by_gate.get(gate_name, [])
                if none_bpbs and gate_bpbs:
                    t, p = welch_ttest(none_bpbs, gate_bpbs)
                    delta = sum(gate_bpbs) / len(gate_bpbs) - sum(none_bpbs) / len(none_bpbs)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"    gate={gate_name} vs none: delta={delta:+.4f} p={p:.4f} {sig}")


def print_decision(data_path: str):
    """Print the go/no-go recommendation."""
    summary_file = RESULTS / "decision_summary.json"
    if not summary_file.exists():
        print("No summary file found.")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    print(f"\n{'='*70}")
    print("  DECISION")
    print(f"{'='*70}")

    best_budget = max(BUDGETS)
    bare = summary.get(f"bare_ssm_b{best_budget}")
    full = summary.get(f"full_stack_b{best_budget}")
    tfm = summary.get(f"transformer_b{best_budget}")

    if not bare or not full:
        print("  Insufficient data for decision.")
        return

    bare_bpb = bare["mean_bpb"]
    full_bpb = full["mean_bpb"]
    tfm_bpb = tfm["mean_bpb"] if tfm else None

    delta = full_bpb - bare_bpb
    t, p = welch_ttest(bare["bpbs"], full["bpbs"])
    sig = p < 0.05

    print(f"\n  At {best_budget}s budget:")
    print(f"    bare_ssm:   {bare_bpb:.4f} bpb")
    print(f"    full_stack: {full_bpb:.4f} bpb  (delta: {delta:+.4f})")
    if tfm_bpb:
        print(f"    transformer: {tfm_bpb:.4f} bpb")
    print(f"    Welch t-test: p={p:.4f} {'(significant)' if sig else '(not significant)'}")

    # Scaling trend: does full_stack improve more with budget?
    bare_150 = summary.get("bare_ssm_b150", {}).get("mean_bpb")
    full_150 = summary.get("full_stack_b150", {}).get("mean_bpb")
    if bare_150 and full_150:
        bare_improvement = bare_150 - bare_bpb
        full_improvement = full_150 - full_bpb
        print(f"\n  Scaling trend (150s → {best_budget}s):")
        print(f"    bare_ssm improvement:   {bare_improvement:+.4f}")
        print(f"    full_stack improvement: {full_improvement:+.4f}")
        if full_improvement > bare_improvement:
            print(f"    full_stack scales BETTER (+{full_improvement - bare_improvement:.4f} more improvement)")
        else:
            print(f"    bare_ssm scales better")

    print(f"\n  Recommendation:")
    if delta < -0.05 and sig:
        print(f"    >>> SCALE TO H100s <<<")
        print(f"    Full stack significantly outperforms bare SSM. Memory+Wernicke adds value.")
    elif delta < 0 and not sig:
        print(f"    >>> CAUTIOUS SCALE <<<")
        print(f"    Full stack trends better but not significant with 3 seeds.")
        print(f"    Consider 5 seeds at 600s to confirm, or scale with awareness.")
    elif abs(delta) < 0.05:
        print(f"    >>> HOLD <<<")
        print(f"    No meaningful difference. Wernicke may carry alone.")
        print(f"    Test Wernicke-only (no memory) at longer budgets before scaling.")
    else:
        print(f"    >>> DO NOT SCALE <<<")
        print(f"    Full stack is worse. Simplify architecture.")
    if tfm_bpb and tfm_bpb < bare_bpb - 0.1:
        print(f"    WARNING: Transformer baseline ({tfm_bpb:.4f}) beats SSM ({bare_bpb:.4f}).")
        print(f"    SSM throughput bottleneck may be fundamental at this scale.")


def main():
    print(
        "\n  EXPLORATORY-ONLY — not paper-grade.\n"
        "  Numbers from this script MUST NOT appear in confirmatory paper tables.\n"
        "  See docs/plans/2026-04-17-paper-status.md \"Claims that are not safe yet\" (item 4).\n",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(description="Decision experiment: go/no-go on H100 scale-up")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--skip-eval", action="store_true", help="Skip Phase 2 eval ablation")
    args = parser.parse_args()

    t0 = time.time()

    # Phase 1: Training grid
    run_training_grid(args.data_path, args.num_gpus)

    # Phase 2: Eval ablation on full_stack checkpoints
    if not args.skip_eval:
        run_eval_ablation(args.data_path, args.num_gpus)

    # Decision
    print_decision(args.data_path)

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed/60:.1f} minutes")
    print(f"  Results: {RESULTS}")


if __name__ == "__main__":
    main()
