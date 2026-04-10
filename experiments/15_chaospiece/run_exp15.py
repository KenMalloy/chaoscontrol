#!/usr/bin/env python3
"""ChaosPiece Phase A experiment matrix.

Tests whether SP8192 tokenization rescues the SSM and how far the
tokenized SSM is from a tokenized transformer control.

Two-stage design:
  Stage 1: 5 SP-SSM configs + 1 byte control = 42 runs (6 x 7 seeds)
  Stage 2: 1 GPT-matched transformer = 7 runs (param-matched to Stage 1 winner)

Run with:
    python experiments/15_chaospiece/run_exp15.py \
        --data-path /workspace/fineweb_data/datasets/fineweb10B_sp8192 \
        --byte-data-path /workspace/fineweb_data/datasets/fineweb10B_byte260 \
        --sp-model-path /workspace/fineweb_data/tokenizers/fineweb_8192_bpe.model \
        --budget 600 --num-gpus 8 --stage 1

    # After Stage 1 completes:
    python experiments/15_chaospiece/run_exp15.py \
        --data-path /workspace/fineweb_data/datasets/fineweb10B_sp8192 \
        --sp-model-path /workspace/fineweb_data/tokenizers/fineweb_8192_bpe.model \
        --budget 600 --num-gpus 8 --stage 2
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
RUNNER = EXPERIMENT / "runner_exp15.py"

sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
from stats import bootstrap_ci, cohens_d, sem, welch_ttest

# -- Seeds --
SEEDS = [1337, 2674, 4011, 5348, 6685, 8022, 9359]

# -- SP8192 SSM conditions --
SP_SSM_CONDITIONS = {
    "sp_d128_L4": {
        "model_type": "ssm",
        "vocab_size": 8192,
        "model_dim": 128,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 32,
        "base_lr": 2e-3,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
    },
    "sp_d192_L4": {
        "model_type": "ssm",
        "vocab_size": 8192,
        "model_dim": 192,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 32,
        "base_lr": 2e-3,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
    },
    "sp_d128_L6": {
        "model_type": "ssm",
        "vocab_size": 8192,
        "model_dim": 128,
        "num_layers": 6,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 32,
        "base_lr": 2e-3,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
    },
    "sp_d192_L6": {
        "model_type": "ssm",
        "vocab_size": 8192,
        "model_dim": 192,
        "num_layers": 6,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 32,
        "base_lr": 2e-3,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
    },
    "sp_d256_L4": {
        "model_type": "ssm",
        "vocab_size": 8192,
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 32,
        "base_lr": 2e-3,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
    },
}

# -- Byte-level control (Exp 14 bare_ssm winner) --
BYTE_CONTROL = {
    "bare_ssm_byte256": {
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
    },
}


# -- Launch helpers --

def _launch(
    name: str,
    config: dict,
    seed: int,
    budget: float,
    data_path: str,
    gpu_id: int | None,
    sp_model_path: str | None,
) -> tuple[subprocess.Popen, Path, Path, object]:
    """Launch a single runner_exp15.py subprocess."""
    config = dict(config, seed=seed)
    tag = f"{name}_s{seed}"
    CONFIGS.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(
        suffix=".yaml", prefix=f".tmp_{tag}_", dir=CONFIGS,
    ))
    tmp.write_text(yaml.dump(config, default_flow_style=False))

    out_path = RESULTS / f"{tag}.json"
    cmd = [
        sys.executable, str(RUNNER),
        "--config", str(tmp),
        "--data-path", data_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
    ]
    if sp_model_path:
        cmd += ["--sp-model-path", sp_model_path]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = RESULTS / f"{tag}.log"
    log_fh = open(log_path, "w")
    # On pods with /proc, tee to container stdout for live monitoring
    if Path("/proc/1/fd/1").exists():
        shell_cmd = " ".join(shlex.quote(str(c)) for c in cmd)
        shell_cmd += f" 2>&1 | tee {shlex.quote(str(log_path))} /proc/1/fd/1"
        proc = subprocess.Popen(
            ["bash", "-o", "pipefail", "-c", shell_cmd], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, out_path, tmp, log_fh


def run_grid(
    conditions: dict,
    seeds: list[int],
    data_path: str,
    budget: float,
    num_gpus: int,
    sp_model_path: str | None,
    stage_label: str,
    timeout_multiplier: float = 2.5,
):
    """Run a grid of conditions x seeds across GPUs."""
    RESULTS.mkdir(parents=True, exist_ok=True)

    queue: list[tuple[str, dict, int]] = []
    for cond_name, cond_config in conditions.items():
        for seed in seeds:
            tag = f"{cond_name}_s{seed}"
            if not (RESULTS / f"{tag}.json").exists():
                queue.append((cond_name, cond_config, seed))

    total = len(conditions) * len(seeds)
    completed = total - len(queue)
    n_gpus = max(num_gpus, 1)
    n_batches = (len(queue) + n_gpus - 1) // n_gpus
    print(f"\n  Stage {stage_label}: {len(queue)} pending, {completed} done, {total} total")
    print(f"  {num_gpus} GPUs, ~{n_batches} batches")

    for batch_start in range(0, len(queue), n_gpus):
        batch = queue[batch_start : batch_start + n_gpus]
        batch_num = batch_start // n_gpus + 1
        print(f"\n  --- Batch {batch_num}/{n_batches} ---")

        jobs = []
        for i, (cond_name, cond_config, seed) in enumerate(batch):
            gpu_id = i % num_gpus if num_gpus > 1 else None
            proc, out_path, tmp, log_fh = _launch(
                cond_name, cond_config, seed, budget,
                data_path, gpu_id, sp_model_path,
            )
            jobs.append((proc, out_path, tmp, log_fh, cond_name, seed))
            print(f"    GPU {i}: {cond_name} seed={seed}")

        run_timeout = budget * timeout_multiplier
        for proc, out_path, tmp, log_fh, cond_name, seed in jobs:
            try:
                proc.wait(timeout=run_timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                log_fh.close()
                tmp.unlink(missing_ok=True)
                completed += 1
                tag = f"{cond_name}_s{seed}"
                log_path = RESULTS / f"{tag}.log"
                error_tail = log_path.read_text()[-2000:] if log_path.exists() else ""
                (RESULTS / f"{tag}.failed").write_text(json.dumps({
                    "condition": cond_name, "seed": seed, "exit_code": -9,
                    "reason": f"TIMEOUT after {run_timeout:.0f}s",
                    "log_tail": error_tail,
                }))
                print(f"    TIMEOUT: {cond_name} seed={seed}")
                continue

            log_fh.close()
            tmp.unlink(missing_ok=True)
            completed += 1

            if proc.returncode != 0:
                tag = f"{cond_name}_s{seed}"
                log_path = RESULTS / f"{tag}.log"
                error_tail = log_path.read_text()[-2000:] if log_path.exists() else ""
                (RESULTS / f"{tag}.failed").write_text(json.dumps({
                    "condition": cond_name, "seed": seed, "exit_code": proc.returncode,
                    "reason": f"non-zero exit code {proc.returncode}",
                    "log_tail": error_tail,
                }))
                print(f"    FAILED: {cond_name} seed={seed} (exit {proc.returncode})")
                continue

            if out_path.exists():
                try:
                    with open(out_path) as f:
                        result = json.load(f)
                    bpb = result.get("eval", {}).get("bpb", "?")
                    steps = result.get("train", {}).get("steps", "?")
                    params = result.get("params", "?")
                    print(f"    OK: {cond_name} seed={seed} bpb={bpb} steps={steps} params={params:,}")
                except (json.JSONDecodeError, KeyError, ValueError):
                    tag = f"{cond_name}_s{seed}"
                    log_path = RESULTS / f"{tag}.log"
                    error_tail = log_path.read_text()[-2000:] if log_path.exists() else ""
                    (RESULTS / f"{tag}.failed").write_text(json.dumps({
                        "condition": cond_name, "seed": seed, "exit_code": 0,
                        "reason": "corrupt output JSON",
                        "log_tail": error_tail,
                    }))
                    out_path.unlink()
                    print(f"    CORRUPT: {cond_name} seed={seed}")


# -- Summary and go/no-go --

def summarize_stage1() -> dict[str, dict]:
    """Load Stage 1 results, compute stats, print ranked table."""
    all_conditions = dict(SP_SSM_CONDITIONS)
    all_conditions.update(BYTE_CONTROL)

    summary = {}
    for cond_name in all_conditions:
        bpbs = []
        for seed in SEEDS:
            path = RESULTS / f"{cond_name}_s{seed}.json"
            if path.exists():
                with open(path) as f:
                    result = json.load(f)
                bpbs.append(result["eval"]["bpb"])
        if bpbs:
            summary[cond_name] = {
                "mean_bpb": sum(bpbs) / len(bpbs),
                "sem": sem(bpbs),
                "ci_95": list(bootstrap_ci(bpbs)),
                "n_seeds": len(bpbs),
                "bpbs": bpbs,
            }

    # Ranked table
    ranked = sorted(summary.items(), key=lambda x: x[1]["mean_bpb"])
    print("\n" + "=" * 72)
    print("  PHASE A — STAGE 1 RESULTS")
    print("=" * 72)
    print(f"  {'Condition':<24} {'Mean bpb':>10} {'SEM':>8} {'95% CI':>20} {'N':>4}")
    print("-" * 72)
    for name, stats in ranked:
        ci = stats["ci_95"]
        warn = " !!!" if stats["n_seeds"] < len(SEEDS) else ""
        print(f"  {name:<24} {stats['mean_bpb']:>10.4f} {stats['sem']:>8.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]{stats['n_seeds']:>4}{warn}")

    # Warn about incomplete conditions
    incomplete = {k: v for k, v in summary.items() if v["n_seeds"] < len(SEEDS)}
    if incomplete:
        print(f"\n  WARNING: {len(incomplete)} condition(s) have fewer than {len(SEEDS)} seeds:")
        for name, stats in incomplete.items():
            print(f"    {name}: {stats['n_seeds']}/{len(SEEDS)} seeds")

    # Identify SSM winner (require >= 5 seeds for eligibility)
    MIN_SEEDS_FOR_GATE = 5
    sp_conditions = {k: v for k, v in summary.items()
                     if k.startswith("sp_") and v["n_seeds"] >= MIN_SEEDS_FOR_GATE}
    sp_excluded = {k: v for k, v in summary.items()
                   if k.startswith("sp_") and v["n_seeds"] < MIN_SEEDS_FOR_GATE}
    if sp_excluded:
        print(f"\n  Excluded from winner selection (< {MIN_SEEDS_FOR_GATE} seeds):")
        for name, stats in sp_excluded.items():
            print(f"    {name}: {stats['n_seeds']} seeds")
    if sp_conditions:
        winner_name = min(sp_conditions, key=lambda k: sp_conditions[k]["mean_bpb"])
        winner = sp_conditions[winner_name]
        print(f"\n  SSM winner: {winner_name} ({winner['mean_bpb']:.4f} bpb)")

        # Go/no-go: tokenizer helps?
        byte_control = summary.get("bare_ssm_byte256")
        if byte_control and byte_control["n_seeds"] < MIN_SEEDS_FOR_GATE:
            print(f"\n  WARNING: byte control has only {byte_control['n_seeds']} seeds — Gate 1 unreliable")
        if byte_control:
            improvement = byte_control["mean_bpb"] - winner["mean_bpb"]
            t_stat, p_val = welch_ttest(byte_control["bpbs"], winner["bpbs"])
            d = cohens_d(byte_control["bpbs"], winner["bpbs"])
            print(f"\n  Tokenizer effect: {improvement:+.4f} bpb (p={p_val:.4f}, d={d:.2f})")
            if improvement >= 0.1 and p_val < 0.05:
                print("  GATE 1 PASS: SP-SSM beats byte baseline by >= 0.1 bpb (p < 0.05)")
            else:
                print("  GATE 1 FAIL: SP-SSM does not beat byte baseline by >= 0.1 bpb")

    # Save summary
    summary_path = RESULTS / "stage1_summary.json"
    # Strip raw bpb lists for cleaner JSON
    summary_clean = {
        k: {sk: sv for sk, sv in v.items() if sk != "bpbs"}
        for k, v in summary.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary_clean, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")
    return summary


def build_gpt_matched(stage1_summary: dict) -> dict:
    """Build GPT condition param-matched to Stage 1 SSM winner."""
    sp_conditions = {k: v for k, v in stage1_summary.items() if k.startswith("sp_")}
    if not sp_conditions:
        raise RuntimeError("No SP-SSM results found — run Stage 1 first")

    winner_name = min(sp_conditions, key=lambda k: sp_conditions[k]["mean_bpb"])
    print(f"\n  Matching GPT to SSM winner: {winner_name}")

    # Get winner's param count from any seed's result
    for seed in SEEDS:
        path = RESULTS / f"{winner_name}_s{seed}.json"
        if path.exists():
            with open(path) as f:
                result = json.load(f)
            target_params = result["params"]
            break
    else:
        raise RuntimeError(f"No result file found for {winner_name}")

    sys.path.insert(0, str(REPO / "src"))
    from runner_exp15 import match_transformer_params
    matched = match_transformer_params(target_params, vocab_size=8192)
    print(f"  Target: {target_params:,} params")
    print(f"  Matched GPT: dim={matched['model_dim']}, layers={matched['num_layers']}, "
          f"params={matched['total_params']:,} (gap: {abs(matched['total_params'] - target_params):,})")

    return {
        "gpt_matched": {
            "model_type": "transformer",
            "vocab_size": 8192,
            "model_dim": matched["model_dim"],
            "num_layers": matched["num_layers"],
            "ff_mult": 2,
            "seq_len": 512,
            "stride": 256,
            "batch_size": 32,
            "base_lr": 2e-3,
            # Disable SSM-specific criticality regularization
            "crit_reg_alpha": 0.0,
            "crit_reg_beta": 0.0,
        },
    }


def summarize_final(stage1_summary: dict) -> None:
    """Final summary including GPT control. Print go/no-go decision."""
    # Load GPT results
    gpt_bpbs = []
    for seed in SEEDS:
        path = RESULTS / f"gpt_matched_s{seed}.json"
        if path.exists():
            with open(path) as f:
                result = json.load(f)
            gpt_bpbs.append(result["eval"]["bpb"])

    if not gpt_bpbs:
        print("  No GPT control results found — run Stage 2 first")
        return

    gpt_stats = {
        "mean_bpb": sum(gpt_bpbs) / len(gpt_bpbs),
        "sem": sem(gpt_bpbs),
        "ci_95": list(bootstrap_ci(gpt_bpbs)),
        "n_seeds": len(gpt_bpbs),
        "bpbs": gpt_bpbs,
    }

    # Find SSM winner
    sp_conditions = {k: v for k, v in stage1_summary.items() if k.startswith("sp_")}
    winner_name = min(sp_conditions, key=lambda k: sp_conditions[k]["mean_bpb"])
    winner = sp_conditions[winner_name]

    print("\n" + "=" * 72)
    print("  PHASE A — FINAL RESULTS")
    print("=" * 72)

    # Full ranked table
    all_results = dict(stage1_summary)
    all_results["gpt_matched"] = gpt_stats
    ranked = sorted(all_results.items(), key=lambda x: x[1]["mean_bpb"])
    print(f"  {'Condition':<24} {'Mean bpb':>10} {'SEM':>8} {'95% CI':>20} {'N':>4}")
    print("-" * 72)
    for name, stats in ranked:
        ci = stats["ci_95"]
        marker = " <-- SSM winner" if name == winner_name else (" <-- GPT control" if name == "gpt_matched" else "")
        print(f"  {name:<24} {stats['mean_bpb']:>10.4f} {stats['sem']:>8.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]{stats['n_seeds']:>4}{marker}")

    # Gate 2: SSM competitive with transformer?
    # gap > 0 means SSM is worse; gap <= 0 means SSM is equal or better
    gap = winner["mean_bpb"] - gpt_stats["mean_bpb"]
    t_stat, p_val = welch_ttest(winner["bpbs"], gpt_bpbs)
    d = cohens_d(winner["bpbs"], gpt_bpbs)
    print(f"\n  SSM vs GPT gap: {gap:+.4f} bpb (p={p_val:.4f}, d={d:.2f})")
    if gap <= 0.15:
        if gap <= 0:
            print("  GATE 2 PASS: SSM matches or beats matched transformer")
        else:
            print(f"  GATE 2 PASS: SSM within {gap:.3f} bpb of matched transformer (tolerance: 0.15)")
    else:
        print(f"  GATE 2 FAIL: SSM is {gap:.3f} bpb worse than matched transformer (tolerance: 0.15)")

    # Overall go/no-go
    byte_control = stage1_summary.get("bare_ssm_byte256")
    gate1 = False
    if byte_control:
        improvement = byte_control["mean_bpb"] - winner["mean_bpb"]
        _, p1 = welch_ttest(byte_control["bpbs"], winner["bpbs"])
        gate1 = improvement >= 0.1 and p1 < 0.05
    gate2 = gap <= 0.15

    print(f"\n  {'='*40}")
    if gate1 and gate2:
        print("  DECISION: GO — proceed to Phase B")
        print(f"  Carry-forward config: {winner_name}")
    elif gate1 and not gate2:
        print("  DECISION: NO-GO — SSM backbone is the bottleneck")
        print("  Recommended pivot: depth recurrence / attention hybrid (Exp 16)")
    elif not gate1:
        print("  DECISION: NO-GO — tokenizer alone does not help enough")
        print("  Recommended pivot: revisit backbone architecture")
    print(f"  {'='*40}")

    # Save final summary
    final = dict(stage1_summary)
    final["gpt_matched"] = {k: v for k, v in gpt_stats.items() if k != "bpbs"}
    final["_decision"] = {
        "gate1_pass": gate1,
        "gate2_pass": gate2,
        "ssm_winner": winner_name,
        "ssm_gpt_gap": gap,
    }
    final_path = RESULTS / "phase_a_summary.json"
    # Strip raw bpb lists
    final_clean = {}
    for k, v in final.items():
        if isinstance(v, dict) and "bpbs" in v:
            final_clean[k] = {sk: sv for sk, sv in v.items() if sk != "bpbs"}
        else:
            final_clean[k] = v
    with open(final_path, "w") as f:
        json.dump(final_clean, f, indent=2)
    print(f"\n  Final summary saved to {final_path}")


# -- Main --

def main():
    p = argparse.ArgumentParser(description="Exp 15 Phase A matrix launcher")
    p.add_argument("--data-path", required=True, dest="data_path",
                   help="Path to SP8192 tokenized data directory")
    p.add_argument("--byte-data-path", default=None, dest="byte_data_path",
                   help="Path to raw byte data directory (for byte control)")
    p.add_argument("--sp-model-path", required=True, dest="sp_model_path",
                   help="Path to SentencePiece .model file")
    p.add_argument("--budget", type=float, default=600)
    p.add_argument("--num-gpus", type=int, default=8, dest="num_gpus")
    p.add_argument("--stage", type=int, default=0,
                   help="1=SP-SSM+byte, 2=GPT control, 0=both+summary")

    args = p.parse_args()
    byte_data_path = args.byte_data_path
    if byte_data_path is None:
        # Default: look for byte260 sibling of SP data dir
        sp_parent = Path(args.data_path).parent
        byte_candidate = sp_parent / "fineweb10B_byte260"
        if byte_candidate.exists():
            byte_data_path = str(byte_candidate)
        else:
            print(f"WARNING: --byte-data-path not set and {byte_candidate} not found.")
            print(f"  Byte control will fail unless raw byte data exists at the SP data path.")
            byte_data_path = args.data_path

    print(f"Experiment 15: ChaosPiece Phase A")
    print(f"  SP data: {args.data_path}")
    print(f"  Byte data: {byte_data_path}")
    print(f"  SP model: {args.sp_model_path}")
    print(f"  Budget: {args.budget}s | GPUs: {args.num_gpus}")

    if args.stage in (0, 1):
        # Stage 1: SP-SSM conditions
        run_grid(
            SP_SSM_CONDITIONS, SEEDS,
            args.data_path, args.budget, args.num_gpus,
            sp_model_path=args.sp_model_path,
            stage_label="1 (SP-SSM)",
        )
        # Byte control
        run_grid(
            BYTE_CONTROL, SEEDS,
            byte_data_path, args.budget, args.num_gpus,
            sp_model_path=None,  # byte mode
            stage_label="1 (byte control)",
        )
        stage1_summary = summarize_stage1()

    if args.stage in (0, 2):
        # Load Stage 1 summary if running Stage 2 standalone
        if args.stage == 2:
            summary_path = RESULTS / "stage1_summary.json"
            if not summary_path.exists():
                print("ERROR: Run Stage 1 first")
                sys.exit(1)
            with open(summary_path) as f:
                stage1_summary_clean = json.load(f)
            # Reload full data with bpb lists for statistical tests
            stage1_summary = {}
            for cond_name in list(SP_SSM_CONDITIONS) + list(BYTE_CONTROL):
                bpbs = []
                for seed in SEEDS:
                    path = RESULTS / f"{cond_name}_s{seed}.json"
                    if path.exists():
                        with open(path) as f:
                            result = json.load(f)
                        bpbs.append(result["eval"]["bpb"])
                if bpbs:
                    stage1_summary[cond_name] = {
                        "mean_bpb": sum(bpbs) / len(bpbs),
                        "sem": sem(bpbs),
                        "ci_95": list(bootstrap_ci(bpbs)),
                        "n_seeds": len(bpbs),
                        "bpbs": bpbs,
                    }

        # Stage 2: GPT matched control
        gpt_condition = build_gpt_matched(stage1_summary)
        run_grid(
            gpt_condition, SEEDS,
            args.data_path, args.budget, args.num_gpus,
            sp_model_path=args.sp_model_path,
            stage_label="2 (GPT control)",
        )
        summarize_final(stage1_summary)


if __name__ == "__main__":
    main()
