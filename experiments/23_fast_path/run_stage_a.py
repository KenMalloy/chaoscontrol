#!/usr/bin/env python3
"""Run Exp 23 Stage A speed-ceiling sweep."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.data import load_fineweb_tokens  # noqa: E402
from fast_path import build_stage_a_matrix, write_matrix  # noqa: E402
from launch import run_matrix_entries  # noqa: E402


def curated_stage_a_matrix(*, world_size: int, budget_seconds: float) -> list[dict]:
    entries: list[dict] = []
    entries.extend(build_stage_a_matrix(
        vocab_sizes=[16384],
        batch_sizes=[1024],
        chunk_sizes=[64, 128, 256, 512],
        activation_checkpoints=[True],
        world_size=world_size,
        budget_seconds=budget_seconds,
    ))
    entries.extend(build_stage_a_matrix(
        vocab_sizes=[16384],
        batch_sizes=[2048],
        chunk_sizes=[128, 256, 512],
        activation_checkpoints=[True],
        world_size=world_size,
        budget_seconds=budget_seconds,
    ))
    entries.extend(build_stage_a_matrix(
        vocab_sizes=[16384],
        batch_sizes=[4096],
        chunk_sizes=[256, 512],
        activation_checkpoints=[True],
        world_size=world_size,
        budget_seconds=budget_seconds,
    ))
    # Aggressive hot-loop variants: remove activation checkpointing where
    # VRAM might still fit. These are speed probes, not quality claims.
    entries.extend(build_stage_a_matrix(
        vocab_sizes=[16384],
        batch_sizes=[1024, 2048],
        chunk_sizes=[256],
        activation_checkpoints=[False],
        world_size=world_size,
        budget_seconds=budget_seconds,
    ))
    return entries


def prebuild_sp_cache(data_path: str) -> None:
    """Create mmap cache files once before torchrun fans out to DDP ranks."""
    train_tokens, val_tokens = load_fineweb_tokens(data_path)
    print(
        "[stage-a] data cache ready "
        f"train={int(train_tokens.numel()):,} val={int(val_tokens.numel()):,}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path-16384", required=True)
    parser.add_argument("--sp-model-path-8192", default=None)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT / "results_stage_a")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--budget", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-cache-prep", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--include-v8192-baseline", action="store_true")
    args = parser.parse_args(argv)

    entries = curated_stage_a_matrix(
        world_size=args.world_size,
        budget_seconds=args.budget,
    )
    if args.include_v8192_baseline:
        entries.extend(build_stage_a_matrix(
            vocab_sizes=[8192],
            batch_sizes=[1024, 2048],
            chunk_sizes=[256, 512],
            activation_checkpoints=[True],
            world_size=args.world_size,
            budget_seconds=args.budget,
        ))

    if not args.dry_run and not args.skip_cache_prep:
        prebuild_sp_cache(args.data_path)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(args.results_dir / "matrix.json", entries)
    sp_paths = {16384: args.sp_model_path_16384}
    if args.sp_model_path_8192:
        sp_paths[8192] = args.sp_model_path_8192
    summary = run_matrix_entries(
        entries=entries,
        runner_path=EXPERIMENT / "runner_fast_path.py",
        data_path=args.data_path,
        sp_model_paths=sp_paths,
        results_dir=args.results_dir,
        world_size=args.world_size,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
