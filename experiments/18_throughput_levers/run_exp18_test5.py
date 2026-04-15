#!/usr/bin/env python3
"""Experiment 18 Test 5 launcher: LR stability screen at DDP global batch.

Once Test 4 establishes DDP scaling at ws=2 (global batch = bs_per_rank *
2 = 2048), we need a stable LR at that global batch that also beats
single-GPU at matched wall-clock. Test 5 screens three LR values around
the linearly-scaled target and picks the winner.

**Anchor.** Every LR in the Exp 18 stack is derived from the same
calibrated reference: Exp 17 / Exp 18 phase0 established ``(bs=32,
lr=2e-3)`` as a stable per-example learning rate. The linear scaling
rule is ``LR = 2e-3 * (global_batch / 32)``. At ws=2 with bs_per_rank =
1024, global_batch = 2048 and the linear-scaled target is ``2e-3 * 64 =
0.128``. This matches the LR Test 4 uses for its ws=2 condition.

Three conditions around that target:

    linear    LR = 0.128   (aggressive, may diverge at this global batch)
    linear/2  LR = 0.064   (middle, phase0 bs=1024 single-GPU winner)
    linear/4  LR = 0.032   (conservative)

Scientific gate — **two stages, both required**:

  Stage 1 (intra-test): pick a candidate among stable conditions
    (all seeds trained without divergence or near-uniform stall). If
    the mean-bpb gap between the top two stable conditions is inside
    paired noise (< 0.006 bpb AND p_paired >= 0.05), fall back to the
    more conservative LR to avoid picking a noise outlier that
    contaminates Tests 6/7 downstream.

  Stage 2 (cross-test vs Test 4): the candidate must beat Test 4's
    ws=1 bpb-at-600s at matched seeds via paired t-test (p<0.05).
    This is what "usable ws=2 submission regime" actually means —
    DDP at ws=2 has to buy a real bpb improvement over single-GPU,
    otherwise the whole ws=2 regime is pointless.

If Test 4 results are absent, the Stage 2 check can't run and the
winner is emitted as provisional (marked ``winner_is_provisional=True``
in the summary). Tests 6 and 7 reading the Test 5 summary will refuse
to inherit a provisional winner — they require Stage 2 to have passed.

Launch: parallel 2-slot DDP at ws=2 (slot 0 -> GPUs [0,1], slot 1 ->
GPUs [2,3]). Four seeds per condition = 12 runs in 6 waves = ~60 min.

**Statistical power caveat:** at n=4 and per-condition noise ~0.004 bpb,
Stage 1's paired test is only powered for differences larger than
~0.008 bpb. A "no clear winner" outcome from Stage 1 may reflect low
power as much as no effect; the Stage 1 fallback to the conservative
LR is the safety measure when the screen can't discriminate.
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

    # Gate stage 1: pick a candidate from stable conditions with a
    # noise-aware tiebreaker. Both the gap AND the p-value are computed
    # from the same paired-seed cohort to avoid the sample-set drift bug
    # where a partial rerun would make the two statistics disagree.
    stable_rows = [row for row in rows if row["stable_seed_count"] == row["total_seed_count"]]
    candidate: str | None = None
    gap_threshold = 0.006
    if len(stable_rows) >= 2:
        top = stable_rows[0]
        second = stable_rows[1]
        shared = sorted(set(top["bpb_by_seed"]) & set(second["bpb_by_seed"]))
        if len(shared) >= 2:
            top_paired = [top["bpb_by_seed"][s] for s in shared]
            second_paired = [second["bpb_by_seed"][s] for s in shared]
            _, p_top_vs_second = paired_ttest(second_paired, top_paired)
            top_paired_mean = sum(top_paired) / len(top_paired)
            second_paired_mean = sum(second_paired) / len(second_paired)
            gap = second_paired_mean - top_paired_mean  # paired-cohort gap
            if gap > gap_threshold or p_top_vs_second < 0.05:
                candidate = top["name"]
            else:
                # Inside noise: fall back to the more conservative LR.
                candidate = min(
                    (top, second), key=lambda r: r["base_lr"]
                )["name"]
        else:
            candidate = top["name"]
    elif len(stable_rows) == 1:
        candidate = stable_rows[0]["name"]

    # Gate stage 2: candidate must beat Test 4 ws=1 baseline at matched
    # seeds via paired t-test. If Test 4 results are absent the winner
    # is marked provisional; if Test 4 results exist but the cross-check
    # fails, winner is None and Tests 6/7 must not inherit it.
    ws1_by_seed = _load_ws1_seed_bpbs()
    ws1_comparison: dict[str, Any] | None = None
    winner: str | None = None
    winner_is_provisional = False
    if candidate is None:
        print("\nStage 1: no stable candidate LR — no winner.")
    elif not ws1_by_seed:
        # Test 4 hasn't been run yet. Emit the Stage 1 candidate as
        # provisional so a standalone Test 5 run still produces something
        # usable, but flag it so Tests 6/7 can refuse to inherit it.
        winner = candidate
        winner_is_provisional = True
        print(
            f"\nStage 2: Test 4 ws=1 results not found; emitting "
            f"{candidate} as PROVISIONAL (not cross-validated)."
        )
    else:
        candidate_row = next(row for row in rows if row["name"] == candidate)
        shared = sorted(set(candidate_row["bpb_by_seed"]) & set(ws1_by_seed))
        if len(shared) < 2:
            print(
                f"\nStage 2: Test 4 ws=1 data found but shared-seed "
                f"overlap with {candidate} is {len(shared)} < 2; "
                f"cross-check skipped, emitting as PROVISIONAL."
            )
            winner = candidate
            winner_is_provisional = True
        else:
            a = [ws1_by_seed[s] for s in shared]
            b = [candidate_row["bpb_by_seed"][s] for s in shared]
            t, p = paired_ttest(a, b)
            delta = sum(a) / len(a) - sum(b) / len(b)
            passes = delta > 0 and p < 0.05
            ws1_comparison = {
                "candidate_condition": candidate,
                "delta_ws1_minus_candidate_bpb": delta,
                "paired_t": t,
                "paired_p": p,
                "n_paired_seeds": len(shared),
                "candidate_beats_ws1_p_lt_0.05": bool(passes),
            }
            print(
                f"\nStage 2 vs Test 4 ws1 (n={len(shared)}): {candidate} "
                f"delta={delta:+.4f} bpb  p_paired={p:.4g}  "
                f"beats_ws1={passes}"
            )
            if passes:
                winner = candidate
            else:
                print(
                    f"Gate FAILED: {candidate} does not beat Test 4 ws=1 "
                    f"at p<0.05; no winner emitted. Tests 6/7 must not "
                    f"inherit this result."
                )
                winner = None

    summary["_decision"] = {
        "winner_lr_condition": winner,
        "winner_is_provisional": winner_is_provisional,
        "winner_base_lr": (
            next(row["base_lr"] for row in rows if row["name"] == winner)
            if winner else None
        ),
        "stage1_candidate": candidate,
        "stable_conditions": [row["name"] for row in stable_rows],
        "winner_vs_ws1_baseline": ws1_comparison,
    }
    print(
        f"\nGate: winner = {winner}"
        f"{' (PROVISIONAL)' if winner_is_provisional else ''}"
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
