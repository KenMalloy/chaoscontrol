#!/usr/bin/env python3
"""Experiment 18 Test 6 launcher: sequence length sweep.

At the winning ws=2 DDP config, vary ``seq_len ∈ {512, 1024, 2048}`` to
see whether longer sequences buy better bpb per wall-clock. The chunked
scan is O(N) theoretically but kernel launch overhead, fp64 cumprod
allocation, and HBM traffic can make the real curve sub-linear or
super-linear depending on where we land on the compute/memory tradeoff.

Three conditions at fixed (batch=1024 per rank, LR from Test 5 winner):

    seq512   baseline (matches Tests 4/5 regime)
    seq1024  double length, half compute-per-position density
    seq2048  quadruple length, may hit VRAM ceiling on ws=2

Gate (bpb at matched 600s):
    Winner is the seq_len with the lowest mean_bpb. No statistical
    preference for smaller seq_len — longer sequences are fine as long
    as bpb drops.

Launch: parallel 2-slot DDP at ws=2 (same as Test 5). Four seeds per
condition = 12 runs, 6 waves, ~60 min. Skips any condition that the
runner reports OOMs on — VRAM ceiling at seq=2048 is the most likely
failure and worth detecting rather than crashing the matrix.
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
RESULTS = EXPERIMENT / "results_test6"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
sys.path.insert(0, str(EXPERIMENT))
from stats import bootstrap_ci, paired_ttest, sem  # noqa: E402
from _harness import (  # noqa: E402
    run_parallel_ddp_matrix,
    validate_data_paths,
)

SWEEP_SEEDS = [1337, 2674, 4011, 5348]


def _base(seq_len: int, **overrides: Any) -> dict[str, Any]:
    cfg = {
        "model_type": "ssm",
        "vocab_size": 16384,
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": seq_len,
        "stride": seq_len // 2,
        "batch_size": 1024,
        "eval_batches": 16,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
        "base_lr": 0.064,
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
    }
    cfg.update(overrides)
    return cfg


CONDITIONS: dict[str, dict[str, Any]] = {
    "seq512":  _base(seq_len=512),
    "seq1024": _base(seq_len=1024),
    "seq2048": _base(seq_len=2048),
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
        tok_per_s_values: list[float] = []
        for seed, file in matches:
            data = json.loads(file.read_text())
            bpb_by_seed[seed] = float(data["eval"]["bpb"])
            train = data["train"]
            cfg = conditions[condition_name]
            tok_per_step = int(cfg["batch_size"]) * int(cfg["seq_len"]) * 2  # ws=2
            tok_per_s_values.append(
                float(train["steps"]) * tok_per_step / max(float(train["elapsed_s"]), 1e-9)
            )
        bpb_values = [bpb_by_seed[s] for s in sorted(bpb_by_seed)]
        rows.append({
            "name": condition_name,
            "seq_len": conditions[condition_name]["seq_len"],
            "bpb_by_seed": bpb_by_seed,
            "bpb_values": bpb_values,
            "mean_bpb": _mean(bpb_values),
            "se_bpb": sem(bpb_values),
            "ci_bpb": bootstrap_ci(bpb_values),
            "mean_tok_per_s": _mean(tok_per_s_values),
        })

    rows.sort(key=lambda row: row["mean_bpb"])
    if not rows:
        return summary

    print("\nTest 6 results — sequence length sweep at ws=2 DDP")
    print(
        f"  {'condition':<10} {'seq_len':>8} {'mean_bpb':>9} {'sem':>7} "
        f"{'tok/s':>12}"
    )
    for row in rows:
        print(
            f"  {row['name']:<10} {row['seq_len']:>8} {row['mean_bpb']:9.4f} "
            f"{row['se_bpb']:7.4f} {row['mean_tok_per_s']:12.0f}"
        )
        summary[row["name"]] = {
            "seq_len": row["seq_len"],
            "mean_bpb": row["mean_bpb"],
            "sem_bpb": row["se_bpb"],
            "ci_95_bpb": row["ci_bpb"],
            "mean_tok_per_s": row["mean_tok_per_s"],
            "n_seeds": len(row["bpb_values"]),
        }

    # Gate: winner by lowest mean_bpb. Additionally report the paired
    # test vs seq_len=512 for each longer sequence so we can see whether
    # the gap is statistically meaningful or within noise.
    baseline = next((row for row in rows if row["name"] == "seq512"), None)
    pair_results: list[dict[str, Any]] = []
    if baseline is not None:
        for row in rows:
            if row["name"] == "seq512":
                continue
            shared = sorted(set(baseline["bpb_by_seed"]) & set(row["bpb_by_seed"]))
            if len(shared) < 2:
                continue
            a = [baseline["bpb_by_seed"][s] for s in shared]
            b = [row["bpb_by_seed"][s] for s in shared]
            t, p = paired_ttest(a, b)
            delta = sum(a) / len(a) - sum(b) / len(b)
            pair_results.append({
                "condition": row["name"],
                "seq_len": row["seq_len"],
                "delta_bpb_vs_seq512": delta,
                "paired_t": t,
                "paired_p": p,
                "n_paired_seeds": len(shared),
            })
            print(
                f"\nvs seq512 n={len(shared)}: {row['name']} "
                f"delta_bpb={delta:+.4f} p_paired={p:.4g}"
            )

    winner = rows[0]["name"]  # already sorted by mean_bpb ascending
    summary["_decision"] = {
        "winner_condition": winner,
        "winner_seq_len": rows[0]["seq_len"],
        "pair_results": pair_results,
    }
    print(f"\nGate: winner (lowest mean bpb) = {winner} (seq_len={rows[0]['seq_len']})")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 6 launcher — seq_len sweep at ws=2 DDP"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--num-slots", type=int, default=2)
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
    (RESULTS / "test6_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
