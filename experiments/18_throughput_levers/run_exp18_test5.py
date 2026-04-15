#!/usr/bin/env python3
"""Experiment 18 Test 5 launcher: LR stability screen at DDP global batch.

Once Test 4 establishes DDP scaling, the global batch jumps from ws=1's
bs=1024 to ws=2's 2048 (or higher). Linear LR scaling from the single-GPU
2e-3 baseline implies LR_target = 2e-3 * (global_batch / single_batch). At
gb=2048 that's LR=4e-3. Whether the model trains stably at that LR, or
needs sqrt-scaling / capped-scaling / warmup extension, is exactly the
question Test 5 answers.

Three LR conditions over the ws=2 DDP stack (matches Tests 6/7's regime):

    linear   LR = 2e-3 * (2048 / 32)  = 0.128   (aggressive)
    linear/2 LR = 0.064                          (middle)
    linear/4 LR = 0.032                          (conservative)

Scientific gate (stability + bpb):
    At least one LR trains stably (no NaN, no divergence) AND produces
    bpb-at-600s that beats the ws=1 reference at matched wall-clock. A
    stable-but-worse LR is not a usable submission config; a
    faster-but-diverging LR tells us linear scaling is too aggressive.

Launch: parallel 2-slot DDP at ws=2 (slot 0 -> GPUs [0,1], slot 1 ->
GPUs [2,3]). Four seeds per condition = 12 runs in 6 waves = ~60 min.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / "results_test5"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
sys.path.insert(0, str(EXPERIMENT))
from stats import bootstrap_ci, paired_ttest, sem  # noqa: E402
from _harness import (  # noqa: E402
    run_parallel_ddp_matrix,
    validate_data_paths,
)

# Cross-test reference: if Test 4 has already run, we can compare the
# winning Test 5 LR condition against Test 4's ws=1 baseline on matched
# seeds. Location is a sibling results directory.
TEST4_RESULTS = EXPERIMENT / "results_test4"
TEST4_WS1_CONDITION = "ws1"  # matches run_exp18_test4.CONDITIONS key

SWEEP_SEEDS = [1337, 2674, 4011, 5348]

# LR values derived from linear scaling off single-GPU 2e-3 baseline at
# bs=32: the global_batch ratio for ws=2/bs=1024 is 2048/32 = 64, giving
# linear LR = 0.128. Halving and quartering handle the common failure
# modes where linear is too aggressive at scale.
LR_LINEAR = 0.128
LR_HALF = 0.064
LR_QUARTER = 0.032


def _base(base_lr: float, **overrides: Any) -> dict[str, Any]:
    cfg = {
        "model_type": "ssm",
        "vocab_size": 16384,
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 1024,
        "eval_batches": 16,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
        "base_lr": base_lr,
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
    }
    cfg.update(overrides)
    return cfg


CONDITIONS: dict[str, dict[str, Any]] = {
    "lr_linear":  _base(base_lr=LR_LINEAR),
    "lr_half":    _base(base_lr=LR_HALF),
    "lr_quarter": _base(base_lr=LR_QUARTER),
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _loss_is_stable(final_loss: float, initial_loss: float = 9.7) -> bool:
    """Stability check: a run that diverged, NaN'd, or stalled near uniform
    fails this gate. Vocab=16384 means uniform-prediction loss is
    log(16384) ~= 9.7; a run that barely moved from uniform is not
    "stable" in any useful sense — it's a dead LR. Require final loss
    at least 1.0 nat below initial, catching both divergence and
    near-uniform stagnation.
    """
    import math
    if not math.isfinite(final_loss):
        return False
    return final_loss < initial_loss - 1.0


def _load_ws1_seed_bpbs() -> dict[int, float]:
    """Load Test 4's ws=1 seed -> bpb mapping if the results exist.

    Used by the Test 5 gate to verify the winning LR at ws=2 actually
    beats the ws=1 single-GPU baseline at matched wall-clock — i.e.,
    DDP scaling is buying us something. Returns an empty dict if
    Test 4 hasn't been run yet, which the caller interprets as
    "skip the ws=1 comparison" rather than treating it as a failure.
    """
    result: dict[int, float] = {}
    if not TEST4_RESULTS.exists():
        return result
    pattern = re.compile(rf"^{re.escape(TEST4_WS1_CONDITION)}_s(\d+)\.json$")
    for file in TEST4_RESULTS.iterdir():
        m = pattern.match(file.name)
        if not m:
            continue
        try:
            data = json.loads(file.read_text())
            result[int(m.group(1))] = float(data["eval"]["bpb"])
        except Exception:
            continue
    return result


def summarize_results(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for condition_name in conditions:
        pattern = re.compile(rf"^{re.escape(condition_name)}_s(\d+)\.json$")
        matches: list[tuple[int, Path]] = []
        if RESULTS.exists():
            for file in RESULTS.iterdir():
                m = pattern.match(file.name)
                if m:
                    matches.append((int(m.group(1)), file))
        if not matches:
            continue
        matches.sort(key=lambda pair: pair[0])
        bpb_by_seed: dict[int, float] = {}
        loss_by_seed: dict[int, float] = {}
        for seed, file in matches:
            data = json.loads(file.read_text())
            bpb_by_seed[seed] = float(data["eval"]["bpb"])
            loss_by_seed[seed] = float(data["train"]["final_loss"])
        bpb_values = [bpb_by_seed[s] for s in sorted(bpb_by_seed)]
        stable_count = sum(1 for v in loss_by_seed.values() if _loss_is_stable(v))
        rows.append({
            "name": condition_name,
            "base_lr": conditions[condition_name]["base_lr"],
            "bpb_by_seed": bpb_by_seed,
            "bpb_values": bpb_values,
            "mean_bpb": _mean(bpb_values),
            "se_bpb": sem(bpb_values),
            "ci_bpb": bootstrap_ci(bpb_values),
            "stable_seed_count": stable_count,
            "total_seed_count": len(bpb_values),
        })

    rows.sort(key=lambda row: row["mean_bpb"])
    if not rows:
        return summary

    print("\nTest 5 results — LR stability screen at ws=2 DDP")
    print(
        f"  {'condition':<14} {'base_lr':>10} {'stable':>8} {'mean_bpb':>9} {'sem':>7}"
    )
    for row in rows:
        print(
            f"  {row['name']:<14} {row['base_lr']:10.4f} "
            f"{row['stable_seed_count']}/{row['total_seed_count']:<6} "
            f"{row['mean_bpb']:9.4f} {row['se_bpb']:7.4f}"
        )
        summary[row["name"]] = {
            "base_lr": row["base_lr"],
            "mean_bpb": row["mean_bpb"],
            "sem_bpb": row["se_bpb"],
            "ci_95_bpb": row["ci_bpb"],
            "stable_seed_count": row["stable_seed_count"],
            "total_seed_count": row["total_seed_count"],
        }

    # Gate (two-stage):
    #
    #   1. Pick a candidate: among conditions where ALL seeds trained
    #      stably, the one with the lowest mean bpb. If the mean-bpb gap
    #      between the top two stable conditions is <= 0.006 bpb AND
    #      the paired t-test between them is p >= 0.05, the choice is
    #      inside noise and we fall back to the most conservative
    #      (lowest-LR) stable condition to avoid picking a noisy outlier
    #      that contaminates Tests 6/7 downstream.
    #
    #   2. Cross-validate the candidate against Test 4's ws=1 baseline
    #      if those results exist: the candidate's mean bpb must beat
    #      ws=1 at matched 600s via paired t-test (p<0.05) on shared
    #      seeds, otherwise DDP at ws=2 is not buying anything over
    #      single-GPU and the LR screen's choice is moot.
    stable_rows = [row for row in rows if row["stable_seed_count"] == row["total_seed_count"]]
    winner = None
    gap_threshold = 0.006
    if len(stable_rows) >= 2:
        top = stable_rows[0]
        second = stable_rows[1]
        shared = sorted(set(top["bpb_by_seed"]) & set(second["bpb_by_seed"]))
        if len(shared) >= 2:
            top_paired = [top["bpb_by_seed"][s] for s in shared]
            second_paired = [second["bpb_by_seed"][s] for s in shared]
            _, p_top_vs_second = paired_ttest(second_paired, top_paired)
            gap = second["mean_bpb"] - top["mean_bpb"]
            if gap > gap_threshold or p_top_vs_second < 0.05:
                winner = top["name"]
            else:
                # Inside noise: prefer the more conservative LR.
                winner = min(
                    (top, second), key=lambda r: r["base_lr"]
                )["name"]
        else:
            winner = top["name"]
    elif len(stable_rows) == 1:
        winner = stable_rows[0]["name"]

    # Cross-test gate vs Test 4 ws=1 baseline (if available).
    ws1_by_seed = _load_ws1_seed_bpbs()
    ws1_comparison: dict[str, Any] | None = None
    if winner is not None and ws1_by_seed:
        winner_row = next(row for row in rows if row["name"] == winner)
        shared = sorted(set(winner_row["bpb_by_seed"]) & set(ws1_by_seed))
        if len(shared) >= 2:
            a = [ws1_by_seed[s] for s in shared]
            b = [winner_row["bpb_by_seed"][s] for s in shared]
            t, p = paired_ttest(a, b)
            delta = sum(a) / len(a) - sum(b) / len(b)
            passes = delta > 0 and p < 0.05
            ws1_comparison = {
                "winner_condition": winner,
                "delta_ws1_minus_winner_bpb": delta,
                "paired_t": t,
                "paired_p": p,
                "n_paired_seeds": len(shared),
                "winner_beats_ws1_p_lt_0.05": bool(passes),
            }
            print(
                f"\nvs Test 4 ws1 (n={len(shared)}): {winner} "
                f"delta={delta:+.4f} bpb  p_paired={p:.4g}  "
                f"beats_ws1={passes}"
            )

    summary["_decision"] = {
        "winner_lr_condition": winner,
        "winner_base_lr": (
            next(row["base_lr"] for row in rows if row["name"] == winner) if winner else None
        ),
        "stable_conditions": [row["name"] for row in stable_rows],
        "winner_vs_ws1_baseline": ws1_comparison,
    }
    print(
        f"\nGate: winner (2-stage: top-stable w/ threshold, then cross-check vs ws1) "
        f"= {winner}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 5 launcher — LR stability screen at ws=2 DDP"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--num-slots", type=int, default=2,
                        help="Number of parallel DDP groups (each uses 2 GPUs)")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if not args.summarize_only:
        validate_data_paths(args.data_path, args.sp_model_path)
        run_parallel_ddp_matrix(
            conditions=CONDITIONS,
            seeds=SWEEP_SEEDS,
            ws_per_slot=2,
            num_slots=args.num_slots,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS,
        )

    summary = summarize_results(CONDITIONS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "test5_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
