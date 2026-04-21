#!/usr/bin/env python3
"""Run Exp 23 Stage B base-lock matrix from a Stage A winner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT))

from fast_path import build_stage_b_matrix, read_speed_config, write_matrix  # noqa: E402
from launch import run_matrix_entries  # noqa: E402


def _init_paths(args: argparse.Namespace) -> dict[int, dict[str, str]]:
    paths: dict[int, dict[str, str]] = {}
    if args.sgns_8192_meanstd and args.sgns_8192_fullcov:
        paths[8192] = {
            "meanstd": args.sgns_8192_meanstd,
            "fullcov": args.sgns_8192_fullcov,
        }
    if args.sgns_16384_meanstd and args.sgns_16384_fullcov:
        paths[16384] = {
            "meanstd": args.sgns_16384_meanstd,
            "fullcov": args.sgns_16384_fullcov,
        }
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path-8192", required=True)
    parser.add_argument("--sp-model-path-16384", required=True)
    parser.add_argument("--speed-config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT / "results_stage_b")
    parser.add_argument("--checkpoint-dir", type=Path, default=EXPERIMENT / "checkpoints_stage_b")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2674, 4011, 5348])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--sgns-8192-meanstd", default=None)
    parser.add_argument("--sgns-8192-fullcov", default=None)
    parser.add_argument("--sgns-16384-meanstd", default=None)
    parser.add_argument("--sgns-16384-fullcov", default=None)
    args = parser.parse_args(argv)

    speed_config = read_speed_config(args.speed_config)
    entries = build_stage_b_matrix(
        speed_config=speed_config,
        seeds=args.seeds,
        vocab_sizes=[8192, 16384],
        init_paths=_init_paths(args),
        world_size=args.world_size,
        budget_seconds=args.budget,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(args.results_dir / "matrix.json", entries)
    summary = run_matrix_entries(
        entries=entries,
        runner_path=EXPERIMENT / "runner_fast_path.py",
        data_path=args.data_path,
        sp_model_paths={
            8192: args.sp_model_path_8192,
            16384: args.sp_model_path_16384,
        },
        results_dir=args.results_dir,
        world_size=args.world_size,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
