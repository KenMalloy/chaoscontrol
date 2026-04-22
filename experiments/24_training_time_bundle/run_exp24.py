#!/usr/bin/env python3
"""Run Exp24 training-time bundle matrices."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
EXP23 = REPO / "experiments" / "23_fast_path"
EXP24 = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXP23 / "configs" / "base_seq_epoch_lr0064_full_corpus.yaml"
DEFAULT_DATA_PATH = REPO / "baselines" / "parameter_golf" / "datasets" / "fineweb10B_sp16384"
DEFAULT_SP_MODEL_16384 = (
    REPO / "baselines" / "parameter_golf" / "tokenizers" / "fineweb_16384_bpe.model"
)

sys.path.insert(0, str(EXP23))
sys.path.insert(0, str(EXP24))
sys.path.insert(0, str(REPO / "src"))

from exp24 import (  # noqa: E402
    DEFAULT_CONTROL_SEEDS,
    build_first_wave_mechanism_matrix,
    build_phase_a_sampling_matrix,
    build_ring0_control_matrix,
    build_semantic_overhead_gate_matrix,
)
from fast_path import read_speed_config, write_matrix  # noqa: E402
from launch import run_matrix_entries  # noqa: E402


def _prebuild_cache(data_path: str) -> None:
    from chaoscontrol.data import load_fineweb_tokens  # noqa: E402

    train_tokens, val_tokens = load_fineweb_tokens(data_path)
    print(
        "[exp24] data cache ready "
        f"train={int(train_tokens.numel()):,} val={int(val_tokens.numel()):,}",
        flush=True,
    )


def _print_entries(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        print(f"[exp24] {entry['name']}")
        print(json.dumps(entry, indent=2, sort_keys=True, default=str))


def _build_entries(
    *,
    matrix: str,
    speed_config: dict[str, Any],
    world_size: int,
    budget_seconds: float,
    seeds: list[int],
) -> list[dict[str, Any]]:
    if matrix == "ring0_control":
        return build_ring0_control_matrix(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seeds=seeds,
        )
    if matrix == "phase_a_sampling":
        return build_phase_a_sampling_matrix(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seeds=seeds,
        )
    if matrix == "semantic_overhead_gate":
        entries: list[dict[str, Any]] = []
        for seed in seeds:
            entries.extend(
                build_semantic_overhead_gate_matrix(
                    speed_config=speed_config,
                    seed=seed,
                    world_size=world_size,
                    budget_seconds=budget_seconds,
                )
            )
        return entries
    if matrix == "first_wave":
        return build_first_wave_mechanism_matrix(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seed_values=seeds,
        )
    if matrix == "all":
        entries: list[dict[str, Any]] = []
        entries.extend(
            build_ring0_control_matrix(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
                seeds=seeds,
            )
        )
        entries.extend(
            build_phase_a_sampling_matrix(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
                seeds=seeds,
            )
        )
        for seed in seeds:
            entries.extend(
                build_semantic_overhead_gate_matrix(
                    speed_config=speed_config,
                    seed=seed,
                    world_size=world_size,
                    budget_seconds=budget_seconds,
                )
            )
        entries.extend(
            build_first_wave_mechanism_matrix(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
                seed_values=seeds,
            )
        )
        return entries
    raise ValueError(f"unsupported matrix: {matrix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        choices=[
            "ring0_control",
            "phase_a_sampling",
            "semantic_overhead_gate",
            "first_wave",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=EXP24 / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_CONTROL_SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--sp-model-path-16384",
        type=Path,
        default=DEFAULT_SP_MODEL_16384,
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args(argv)

    default_world_size = 1 if args.matrix == "semantic_overhead_gate" else 8
    default_budget = 90.0 if args.matrix == "semantic_overhead_gate" else 600.0
    world_size = int(args.world_size) if args.world_size is not None else default_world_size
    budget = float(args.budget) if args.budget is not None else default_budget

    speed_config = read_speed_config(args.config)
    entries = _build_entries(
        matrix=args.matrix,
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget,
        seeds=[int(seed) for seed in args.seeds],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(args.output_dir / "matrix.json", entries)

    effective_dry_run = bool(args.dry_run or args.show)
    selected = entries[: args.limit] if args.limit is not None else entries
    if effective_dry_run:
        print(
            f"[exp24] matrix={args.matrix} entries={len(selected)} "
            f"world_size={world_size} dry_run={effective_dry_run}",
            flush=True,
        )
        _print_entries(selected)

    if not effective_dry_run:
        _prebuild_cache(str(args.data_path))

    summary = run_matrix_entries(
        entries=entries,
        runner_path=EXP23 / "runner_fast_path.py",
        data_path=str(args.data_path),
        sp_model_paths={16384: str(args.sp_model_path_16384)},
        results_dir=args.output_dir,
        world_size=world_size,
        limit=args.limit,
        dry_run=effective_dry_run,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
