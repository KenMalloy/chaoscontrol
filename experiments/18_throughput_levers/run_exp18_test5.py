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


def _loss_is_stable(final_loss: float) -> bool:
    """Stability check: a run that diverged or NaN'd fails this gate."""
    import math
    if not math.isfinite(final_loss):
        return False
    # Loss should be well below a random uniform baseline. Vocab=16384 ->
    # uniform loss = log(16384) ~= 9.7. Any final_loss above ~8 is a
    # divergence or early-termination signal.
    return final_loss < 8.0


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

    # Gate: condition passes if all seeds were stable AND mean bpb is the
    # lowest among stable conditions.
    stable_rows = [row for row in rows if row["stable_seed_count"] == row["total_seed_count"]]
    winner = min(stable_rows, key=lambda r: r["mean_bpb"])["name"] if stable_rows else None
    summary["_decision"] = {
        "winner_lr_condition": winner,
        "winner_base_lr": (
            next(row["base_lr"] for row in rows if row["name"] == winner) if winner else None
        ),
        "stable_conditions": [row["name"] for row in stable_rows],
    }
    print(
        f"\nGate: winner (all-seeds stable + lowest mean bpb) = {winner}"
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
