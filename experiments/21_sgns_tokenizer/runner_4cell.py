#!/usr/bin/env python3
"""Exp 21 4-cell main ablation — SSM × modded-NanoGPT × {random, SGNS init}.

20 runs (4 cells × 5 seeds) under ``_harness.run_parallel_ddp_matrix``.
Per-run JSONs land in ``results/four_cell/`` keyed by ``{cell}_s{seed}.json``.
Statistical analysis lives in ``scripts/exp21_analyze.py``; this orchestrator
only launches, aggregates, and writes a simple per-cell summary.

Transformer arm's ``base_lr`` is read from the Phase 0 summary
(``results/phase0/phase0_summary.json``) unless overridden via
``--transformer-lr``. SSM arm's ``base_lr`` is fixed at 0.064 — the Exp 18
Test 5b winner at ws=2 × bs_per_rank=1024.
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
RESULTS = EXPERIMENT / "results" / "four_cell"
PHASE0_SUMMARY = EXPERIMENT / "results" / "phase0" / "phase0_summary.json"
RUNNER = EXPERIMENT / "runner_exp21.py"

sys.path.insert(0, str(REPO / "experiments" / "18_throughput_levers"))
from _harness import run_parallel_ddp_matrix, validate_data_paths  # noqa: E402


SEEDS = [1337, 42, 123, 7, 8]
SSM_LR = 0.064
SGNS_INIT = "artifacts/sgns_init_meanstd.pt"


def _shared_axes() -> dict[str, Any]:
    return {
        "vocab_size": 8192,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 1024,
        "eval_batches": 16,
        "optimizer": "muon",
        "weight_decay": 1e-2,
        "grad_clip_norm": 1.0,
        "chunk_size": 64,
        "dtype": "bf16",
        "precision": "bf16",
        "activation_checkpoint": True,
    }


def _transformer_cell(*, base_lr: float, embed_init_path: str | None) -> dict[str, Any]:
    cfg = _shared_axes()
    cfg.update({
        "model_type": "transformer_nanogpt_lean",
        "model_dim": 256,
        "num_layers": 8,
        "n_head": 4,
        "ff_mult": 4,
        "base_lr": base_lr,
        "embed_init_path": embed_init_path,
    })
    return cfg


def _ssm_cell(*, base_lr: float, embed_init_path: str | None) -> dict[str, Any]:
    cfg = _shared_axes()
    cfg.update({
        "model_type": "ssm",
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "a_mode": "diag",
        "a_full_rank": 8,
        "a_full_gamma": 0.05,
        "base_lr": base_lr,
        "embed_init_path": embed_init_path,
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
    })
    return cfg


def build_conditions(transformer_lr: float) -> dict[str, dict[str, Any]]:
    return {
        "A_transformer_random": _transformer_cell(
            base_lr=transformer_lr, embed_init_path=None
        ),
        "B_transformer_sgns": _transformer_cell(
            base_lr=transformer_lr, embed_init_path=SGNS_INIT
        ),
        "C_ssm_random": _ssm_cell(
            base_lr=SSM_LR, embed_init_path=None
        ),
        "D_ssm_sgns": _ssm_cell(
            base_lr=SSM_LR, embed_init_path=SGNS_INIT
        ),
    }


def _load_transformer_lr(override: float | None) -> float:
    if override is not None:
        return override
    if not PHASE0_SUMMARY.exists():
        raise FileNotFoundError(
            f"Phase 0 summary not found at {PHASE0_SUMMARY}. "
            "Run runner_phase0.py first, or pass --transformer-lr explicitly."
        )
    data = json.loads(PHASE0_SUMMARY.read_text())
    winner = data.get("winner_lr")
    if winner is None:
        raise ValueError(
            f"{PHASE0_SUMMARY} has no winner_lr; rerun Phase 0 or pass "
            f"--transformer-lr."
        )
    return float(winner)


def summarize_results(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_cell: dict[str, dict[int, float]] = {}
    for name in conditions:
        per_cell[name] = {}
        pattern = re.compile(rf"^{re.escape(name)}_s(\d+)\.json$")
        if not RESULTS.exists():
            continue
        for file in RESULTS.iterdir():
            m = pattern.match(file.name)
            if not m:
                continue
            data = json.loads(file.read_text())
            bpb = float(data["eval"].get("bpb", float("nan"))) if data.get("eval") else float("nan")
            per_cell[name][int(m.group(1))] = bpb

    print("\nExp 21 4-cell results (per-seed bpb):")
    for name, by_seed in per_cell.items():
        seeds = sorted(by_seed.keys())
        if not seeds:
            print(f"  {name}: (no results)")
            continue
        bpbs = [by_seed[s] for s in seeds]
        mean = sum(bpbs) / len(bpbs) if bpbs else float("nan")
        print(f"  {name:<24} n={len(seeds)} mean_bpb={mean:.4f}  seeds={seeds}")

    return {"per_cell": per_cell}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 21 4-cell ablation (20 runs)"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--num-slots", type=int, default=2)
    parser.add_argument(
        "--transformer-lr", type=float, default=None,
        help="Override Phase 0 winner LR (else read from phase0_summary.json)",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    transformer_lr = _load_transformer_lr(args.transformer_lr)
    conditions = build_conditions(transformer_lr)
    print(f"Transformer LR: {transformer_lr}  |  SSM LR: {SSM_LR}")

    if not args.summarize_only:
        validate_data_paths(args.data_path, args.sp_model_path)
        run_parallel_ddp_matrix(
            conditions=conditions,
            seeds=SEEDS,
            ws_per_slot=2,
            num_slots=args.num_slots,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS,
            runner_script=RUNNER,
        )

    summary = summarize_results(conditions)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "four_cell_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
