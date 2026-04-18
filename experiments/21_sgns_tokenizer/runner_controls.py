#!/usr/bin/env python3
"""Exp 21 controls — full-cov moment-match + shuffled-row sanity on the SSM arm.

Two conditions run only on the SSM arm (where the main signal is expected):

  - ``ssm_fullcov``: SGNS embeddings matched to random init's full row
    covariance (Cholesky whiten+recolor). Stronger null than the
    ``meanstd`` default — covariance parity leaves only directional
    (semantic) signal. 5 seeds.

  - ``ssm_shuffled``: meanstd SGNS tensor with rows randomly permuted.
    Preserves marginal distribution, destroys ID→vector mapping.
    Any "SGNS helps" measurement must vanish here, else the effect is
    distributional rather than semantic. 1 seed.

Reuses the 4-cell SSM cell config; varies only the ``embed_init_path``.
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
RESULTS_FULLCOV = EXPERIMENT / "results" / "fullcov"
RESULTS_SHUFFLED = EXPERIMENT / "results" / "shuffled"
RUNNER = EXPERIMENT / "runner_exp21.py"

sys.path.insert(0, str(REPO / "experiments" / "18_throughput_levers"))
from _harness import run_parallel_ddp_matrix, validate_data_paths  # noqa: E402

# Reuse the 4-cell SSM cell builder so control and main share an identical
# architecture/optimizer regime — the only axis that differs is the init.
sys.path.insert(0, str(EXPERIMENT))
from runner_4cell import SSM_LR, _ssm_cell  # noqa: E402


FULLCOV_SEEDS = [1337, 42, 123, 7, 8]
SHUFFLED_SEEDS = [1337]

FULLCOV_CONDITIONS = {
    "ssm_fullcov": _ssm_cell(
        base_lr=SSM_LR, embed_init_path="artifacts/sgns_init_fullcov.pt"
    ),
}

SHUFFLED_CONDITIONS = {
    "ssm_shuffled": _ssm_cell(
        base_lr=SSM_LR, embed_init_path="artifacts/sgns_init_shuffled.pt"
    ),
}


def _collect(conditions: dict[str, dict[str, Any]], results_dir: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for name in conditions:
        out[name] = {}
        pattern = re.compile(rf"^{re.escape(name)}_s(\d+)\.json$")
        if not results_dir.exists():
            continue
        for file in results_dir.iterdir():
            m = pattern.match(file.name)
            if not m:
                continue
            data = json.loads(file.read_text())
            bpb = float(data["eval"].get("bpb", float("nan"))) if data.get("eval") else float("nan")
            out[name][int(m.group(1))] = bpb
    return out


def summarize_results() -> dict[str, Any]:
    fullcov_per = _collect(FULLCOV_CONDITIONS, RESULTS_FULLCOV)
    shuffled_per = _collect(SHUFFLED_CONDITIONS, RESULTS_SHUFFLED)

    def _print_block(label: str, per: dict[str, dict[int, float]]) -> None:
        print(f"\n{label}:")
        for name, by_seed in per.items():
            seeds = sorted(by_seed.keys())
            if not seeds:
                print(f"  {name}: (no results)")
                continue
            bpbs = [by_seed[s] for s in seeds]
            mean = sum(bpbs) / len(bpbs)
            print(f"  {name:<16} n={len(seeds)} mean_bpb={mean:.4f}  seeds={seeds}")

    _print_block("Full-cov moment-match", fullcov_per)
    _print_block("Shuffled-row control", shuffled_per)

    return {
        "fullcov": fullcov_per,
        "shuffled": shuffled_per,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 21 controls — full-cov + shuffled-row"
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
            conditions=FULLCOV_CONDITIONS,
            seeds=FULLCOV_SEEDS,
            ws_per_slot=2,
            num_slots=args.num_slots,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS_FULLCOV,
            runner_script=RUNNER,
        )
        run_parallel_ddp_matrix(
            conditions=SHUFFLED_CONDITIONS,
            seeds=SHUFFLED_SEEDS,
            ws_per_slot=2,
            num_slots=args.num_slots,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS_SHUFFLED,
            runner_script=RUNNER,
        )

    summary = summarize_results()
    RESULTS_FULLCOV.mkdir(parents=True, exist_ok=True)
    (RESULTS_FULLCOV.parent / "controls_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
