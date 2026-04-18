#!/usr/bin/env python3
"""Exp 21 Phase 0 — transformer LR sweep.

One-seed sweep over three Muon LRs on the transformer_nanogpt_lean arm
with random init. The winning LR is carried into the 4-cell ablation for
both transformer cells (A, B). The SSM arm's LR stays at 0.064 (Exp 18
Test 5b winner at ws=2 × bs_per_rank=1024).

Matches the launch pattern of ``run_exp18_test5b.py`` — ``_harness.run_parallel_ddp_matrix``
over (condition × seed) pairs, one JSON per run, post-hoc summarization
picks the winner. Unlike Test 5b this is a single-seed screen; the
4-cell ablation is the statistically-powered stage, not this LR pick.
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
RESULTS = EXPERIMENT / "results" / "phase0"
RUNNER = EXPERIMENT / "runner_exp21.py"

# Reuse the Exp 18 test-orchestration harness — same runner-script contract
# (YAML in, JSON out), same slot/parallelism/timeout semantics.
sys.path.insert(0, str(REPO / "experiments" / "18_throughput_levers"))
from _harness import run_parallel_ddp_matrix, validate_data_paths  # noqa: E402


SEEDS = [1337]
LRS = (0.016, 0.032, 0.064)


def _base_transformer(base_lr: float) -> dict[str, Any]:
    return {
        "model_type": "transformer_nanogpt_lean",
        "vocab_size": 8192,
        "model_dim": 256,
        "num_layers": 8,
        "n_head": 4,
        "ff_mult": 4,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 1024,
        "eval_batches": 16,
        "optimizer": "muon",
        "base_lr": base_lr,
        "weight_decay": 1e-2,
        "grad_clip_norm": 1.0,
        "chunk_size": 64,
        "dtype": "bf16",
        "precision": "bf16",
        "activation_checkpoint": True,
    }


CONDITIONS: dict[str, dict[str, Any]] = {
    f"lr_{lr:.3f}".replace(".", "p"): _base_transformer(lr) for lr in LRS
}


def summarize_results() -> dict[str, Any]:
    """Read per-condition JSONs, pick lowest-bpb LR, write phase0 summary."""
    rows: list[dict[str, Any]] = []
    for name in CONDITIONS:
        pattern = re.compile(rf"^{re.escape(name)}_s(\d+)\.json$")
        if not RESULTS.exists():
            continue
        for file in RESULTS.iterdir():
            m = pattern.match(file.name)
            if not m:
                continue
            data = json.loads(file.read_text())
            bpb = float(data["eval"].get("bpb", float("nan"))) if data.get("eval") else float("nan")
            rows.append({
                "name": name,
                "seed": int(m.group(1)),
                "base_lr": CONDITIONS[name]["base_lr"],
                "bpb": bpb,
                "final_loss": float(data["train"]["final_loss"]),
            })

    rows.sort(key=lambda r: (r["bpb"] if r["bpb"] == r["bpb"] else float("inf")))
    print("\nPhase 0 transformer LR sweep:")
    print(f"  {'condition':<12} {'base_lr':>8} {'seed':>5} {'bpb':>8} {'final_loss':>11}")
    for r in rows:
        print(
            f"  {r['name']:<12} {r['base_lr']:8.4f} {r['seed']:>5} "
            f"{r['bpb']:8.4f} {r['final_loss']:11.4f}"
        )

    winner = rows[0]["name"] if rows else None
    winner_lr = rows[0]["base_lr"] if rows else None
    summary = {
        "rows": rows,
        "winner_condition": winner,
        "winner_lr": winner_lr,
    }
    if winner:
        print(f"\nWinner: {winner} (base_lr={winner_lr})")
    else:
        print("\nWinner: (none — no complete results)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 21 Phase 0 — transformer LR sweep (3 LRs × 1 seed)"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument(
        "--num-slots", type=int, default=2,
        help="Parallel DDP groups (each uses ws_per_slot=2 GPUs)",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if not args.summarize_only:
        validate_data_paths(args.data_path, args.sp_model_path)
        run_parallel_ddp_matrix(
            conditions=CONDITIONS,
            seeds=SEEDS,
            ws_per_slot=2,
            num_slots=args.num_slots,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS,
            runner_script=RUNNER,
        )

    summary = summarize_results()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
