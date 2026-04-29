#!/usr/bin/env python3
"""Run the Exp26 Adaptive Residual Memory validation canary.

This launcher deliberately has no ablation switches. It always runs the fixed
two-cell validation set:

  1. locked fast/slow control
  2. full Adaptive Residual Memory: CRCT evidence + learned maintenance

CRCT-only, shadow-mode, calibration, and headline matrix switches were removed
because they made the wrong thing easy to run. CRCT is the evidence substrate
inside ARM here, not a standalone architecture.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
EXP23 = REPO / "experiments" / "23_fast_path"
EXP24 = REPO / "experiments" / "24_training_time_bundle"
EXP26 = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXP23 / "configs" / "base_seq_epoch_lr0064_full_corpus.yaml"
DEFAULT_DATA_PATH = (
    REPO / "baselines" / "parameter_golf" / "datasets" / "fineweb10B_sp16384"
)
DEFAULT_SP_MODEL_16384 = (
    REPO / "baselines" / "parameter_golf" / "tokenizers" / "fineweb_16384_bpe.model"
)

sys.path.insert(0, str(EXP23))
sys.path.insert(0, str(EXP24))
sys.path.insert(0, str(EXP26))
sys.path.insert(0, str(REPO / "src"))

from exp26 import (  # noqa: E402
    DEFAULT_VALIDATION_DIR,
    DEFAULT_VALIDATION_TRACE_DIR,
    build_validation_matrix,
)
from fast_path import read_speed_config, write_matrix  # noqa: E402
from launch import run_matrix_entries  # noqa: E402


def _print_entries(entries: list[dict[str, Any]]) -> None:
    import json

    for entry in entries:
        print(f"[exp26] {entry['name']}")
        print(json.dumps(entry, indent=2, sort_keys=True, default=str))


def _prebuild_cache(data_path: str) -> None:
    from chaoscontrol.data import load_fineweb_tokens

    train_tokens, val_tokens = load_fineweb_tokens(data_path)
    print(
        "[exp26] data cache ready "
        f"train={int(train_tokens.numel()):,} val={int(val_tokens.numel()):,}",
        flush=True,
    )


def _run_validation(
    *,
    speed_config: dict[str, Any],
    world_size: int,
    seed: int,
    budget_seconds: float,
    data_path: Path,
    sp_model_path_16384: Path,
    dry_run: bool,
) -> None:
    entries = build_validation_matrix(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
        seed=seed,
    )
    print(
        f"[exp26] validation entries={len(entries)} world_size={world_size} "
        f"budget={budget_seconds}s dry_run={dry_run}",
        flush=True,
    )
    if dry_run:
        _print_entries(entries)
        return
    DEFAULT_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_VALIDATION_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    write_matrix(DEFAULT_VALIDATION_DIR / "matrix.json", entries)
    _prebuild_cache(str(data_path))
    run_matrix_entries(
        entries=entries,
        runner_path=EXP23 / "runner_fast_path.py",
        data_path=str(data_path),
        sp_model_paths={16384: str(sp_model_path_16384)},
        results_dir=DEFAULT_VALIDATION_DIR,
        world_size=world_size,
        limit=None,
        dry_run=False,
        skip_existing=False,
        checkpoint_dir=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--sp-model-path-16384", type=Path, default=DEFAULT_SP_MODEL_16384
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--budget", type=float, default=45.0,
        help="Wall budget per validation cell (seconds)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    speed_config = read_speed_config(args.config)
    _run_validation(
        speed_config=speed_config,
        world_size=int(args.world_size),
        seed=int(args.seed),
        budget_seconds=float(args.budget),
        data_path=args.data_path,
        sp_model_path_16384=args.sp_model_path_16384,
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
