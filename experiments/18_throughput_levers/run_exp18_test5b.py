#!/usr/bin/env python3
"""Experiment 18 Test 5b launcher: LR re-screen at the post-chunked-backward regime.

Test 5 (original) ran at ws=2 × bs_per_rank=512 → global_batch=1024.
The chunked-LM-head backward path landed on 2026-04-16 lifts the
logits.grad bottleneck that capped bs_per_rank at 512; bs_per_rank=1024
now fits on an 80 GiB H100 with ~43 GiB headroom (peak 36.78 GiB
measured). At the new global batch 2048, Test 5's LR winner (0.032)
may no longer be optimal — the linear-scaling predictor doubles to
0.128 and the screen walks the same three-point grid around it.

**Anchor.** Same reference point as Test 5: Exp 17 / Exp 18 phase0
established ``(bs=32, lr=2e-3)`` as a stable per-example LR. Linear
scaling to global_batch=2048 gives ``2e-3 × (2048/32) = 0.128``. We
also carry the Test 5 empirical finding forward — linear was too
aggressive at global_batch=1024 (LR=0.064 lost to half-linear 0.032),
so the screen is centered on the same shape: linear / half / quarter
off the predicted value.

Three conditions:

    linear    LR = 0.128   (linear prediction, may diverge)
    linear/2  LR = 0.064   (empirical winner at the old global_batch)
    linear/4  LR = 0.032   (old Test 5 winner — carries forward if
                            LR is scale-invariant past global_batch=1024)

**Locked axes:**
    - Muon optimizer (Test 7 winner, +0.2 bpb over AdamW at p=0.00016).
    - SP16384 tokenizer (Test 2 winner).
    - seq_len = 512 (Test 6 ceiling — Test 8 will re-open this).
    - 4L × 256d SSM (Test 5 architecture, same as prior tests).
    - chunk_size = 64 for the LM-head chunked backward.

**Gate.** Stage 1 only — pick a stable-condition winner by paired-seed
mean-bpb with a 0.006-bpb / p<0.05 noise tiebreak. Stage 2 (cross-test
vs Test 4 ws=1) is dropped because Test 4 ran at a different batch
regime; the comparison would be confounded.

No cross-screen comparison against Test 5 is emitted. Test 5b differs
from Test 5 on three axes simultaneously — batch size (intended
change), optimizer (AdamW → Muon, locked from Test 7), and the
presence of criticality regularization (train_ssm drops it) — so
any "Test 5b bpb vs Test 5 bpb" delta would be a three-axis
confound, not the single-lever signal it would need to be to
support a causal claim. Interpretation lives at the paper level,
not in this launcher's summary.

Launch: same 2-slot parallel DDP pattern as Test 5, each slot runs
ws=2. On a 4-GPU pod this schedules 3 conditions × 4 seeds = 12 runs
in 6 waves of 600s each ≈ 1 hour wall-clock.
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
RESULTS = EXPERIMENT / "results_test5b"
RUNNER_SSM = EXPERIMENT / "runner_exp18_ssm.py"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
sys.path.insert(0, str(EXPERIMENT))
from stats import bootstrap_ci, paired_ttest, sem  # noqa: E402
from _harness import (  # noqa: E402
    result_is_finite,
    run_parallel_ddp_matrix,
    validate_data_paths,
)

SWEEP_SEEDS = [1337, 2674, 4011, 5348]

# LR grid. Anchored to (bs=32, lr=2e-3). At ws=2 × bs_per_rank=1024,
# global_batch = 2048, linear LR = 2e-3 × (2048/32) = 0.128. Half/quarter
# cover the post-Test-5 empirical pattern where linear was too hot.
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
        # The axis we're changing. Chunked backward makes bs=1024/rank
        # fit; old Test 5 ran 512/rank because of the logits.grad OOM
        # (see memory/project_logits_grad_bottleneck_2026-04-16.md).
        "batch_size": 1024,
        "eval_batches": 16,
        "a_mode": "diag",
        "base_lr": base_lr,
        # Locked axes — the Muon × SP16384 × 4L×256d regime is what
        # Tests 2 / 5 / 7 established. Test 5b is only re-screening LR.
        "optimizer": "muon",
        "chunk_size": 64,
        # Carry forward the Test 5 local_attn defaults explicitly so any
        # config comparison script sees the identical shape.
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
        # crit_target_coupling is intentionally omitted — the bare-SSM
        # train_ssm path doesn't support criticality regularization.
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
    """Stability check — identical to Test 5's helper. Vocab=16384 means
    uniform-prediction CE is log(16384) ≈ 9.7 nats; a run that barely
    moved from uniform is a dead LR. Require final loss at least 1.0
    nat below initial to count as stable.
    """
    import math
    if not math.isfinite(final_loss):
        return False
    return final_loss < initial_loss - 1.0


def summarize_results(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    invalid_results: list[str] = []

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
            if not result_is_finite(data):
                invalid_results.append(f"{condition_name}_s{seed}")
                continue
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

    print("\nTest 5b results — LR re-screen at ws=2 × bs_per_rank=1024")
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

    # Stage 1 winner pick — paired-seed mean-bpb with noise tiebreak,
    # identical to Test 5's Stage 1 logic so the two screens produce
    # comparably-gated winners.
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
            gap = second_paired_mean - top_paired_mean
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

    winner = candidate  # Stage 1 is the only gate in 5b.

    # No cross-screen comparison is emitted — see module docstring.
    # Test 5 and Test 5b differ on batch size, optimizer, AND
    # criticality-regularization presence, so any vs-Test-5 number
    # would be a three-axis confound.

    summary["_decision"] = {
        "winner_lr_condition": winner,
        "winner_base_lr": (
            next(row["base_lr"] for row in rows if row["name"] == winner)
            if winner else None
        ),
        "stage1_candidate": candidate,
        "stable_conditions": [row["name"] for row in stable_rows],
        "invalid_result_files": invalid_results,
    }
    if invalid_results:
        print(
            f"\nDropped {len(invalid_results)} non-finite result file(s): "
            f"{invalid_results}"
        )
    print(f"\nGate: winner = {winner}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 5b launcher — LR re-screen at bs_per_rank=1024"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument(
        "--num-slots", type=int, default=2,
        help="Number of parallel DDP groups (each uses 2 GPUs)",
    )
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
            runner_script=RUNNER_SSM,
        )

    summary = summarize_results(CONDITIONS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "test5b_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
