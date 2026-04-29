#!/usr/bin/env python3
"""Run Exp26 Adaptive Residual Memory matrices.

Three-stage discipline plus a phase-0 runtime smoke:

  Phase 0 (smoke)       : 2 cells, one short control and one short active
                          CRCT+ARM cell, isolated under smoke/.
  Stage 1 (calibrate)   : 1 cell, 1 seed, ~180s, shadow mode, full ARM pipeline.
                          Trace at calibration/trace.ndjson.
  Stage 2 (analyze)     : Read trace, percentile-anchor threshold
                          counterfactuals, write calibration/manifest.json.
  Stage 3 (headline)    : 4 arms x 3 seeds, 600s each. The active ARM cell
                          uses learned Full-A commit authority; manifest
                          thresholds are rule-prior telemetry only.

Usage:

  # one-shot run smoke + all three experiment stages on 4xH100
  python experiments/26_arm/run_exp26.py --stage all

  # smoke-only: launches briefly, writes only to smoke/
  python experiments/26_arm/run_exp26.py --stage smoke

  # calibrate-only (writes manifest then stops)
  python experiments/26_arm/run_exp26.py --stage calibrate

  # headline only (uses manifest when present, but does not require it)
  python experiments/26_arm/run_exp26.py --stage headline

  # dry-run any stage to print entries without launching
  python experiments/26_arm/run_exp26.py --stage all --dry-run
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
    DEFAULT_CALIBRATION_TRACE,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SMOKE_DIR,
    DEFAULT_SMOKE_TRACE_DIR,
    build_arm_v1_matrix,
    build_calibration_matrix,
    build_smoke_matrix,
)
from calibrate import analyze as analyze_calibration  # noqa: E402
from fast_path import read_speed_config, write_matrix  # noqa: E402
from launch import run_matrix_entries  # noqa: E402


CALIBRATION_RESULTS_DIR = EXP26 / "calibration"
HEADLINE_RESULTS_DIR = EXP26 / "results"
SMOKE_RESULTS_DIR = DEFAULT_SMOKE_DIR


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


def _run_smoke(
    *,
    speed_config: dict[str, Any],
    world_size: int,
    seed: int,
    budget_seconds: float,
    data_path: Path,
    sp_model_path_16384: Path,
    dry_run: bool,
) -> None:
    entries = build_smoke_matrix(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
        seed=seed,
    )
    print(
        f"[exp26] stage=smoke entries={len(entries)} world_size={world_size} "
        f"budget={budget_seconds}s dry_run={dry_run}",
        flush=True,
    )
    if dry_run:
        _print_entries(entries)
        return
    SMOKE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_SMOKE_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    write_matrix(SMOKE_RESULTS_DIR / "matrix.json", entries)
    _prebuild_cache(str(data_path))
    run_matrix_entries(
        entries=entries,
        runner_path=EXP23 / "runner_fast_path.py",
        data_path=str(data_path),
        sp_model_paths={16384: str(sp_model_path_16384)},
        results_dir=SMOKE_RESULTS_DIR,
        world_size=world_size,
        limit=None,
        dry_run=False,
        skip_existing=False,
        checkpoint_dir=None,
    )


def _run_calibration(
    *,
    speed_config: dict[str, Any],
    world_size: int,
    seed: int,
    budget_seconds: float,
    data_path: Path,
    sp_model_path_16384: Path,
    dry_run: bool,
) -> None:
    entries = build_calibration_matrix(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
        seed=seed,
    )
    print(
        f"[exp26] stage=calibrate entries={len(entries)} world_size={world_size} "
        f"budget={budget_seconds}s dry_run={dry_run}",
        flush=True,
    )
    if dry_run:
        _print_entries(entries)
        return
    CALIBRATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CALIBRATION_TRACE.parent.mkdir(parents=True, exist_ok=True)
    write_matrix(CALIBRATION_RESULTS_DIR / "matrix.json", entries)
    _prebuild_cache(str(data_path))
    run_matrix_entries(
        entries=entries,
        runner_path=EXP23 / "runner_fast_path.py",
        data_path=str(data_path),
        sp_model_paths={16384: str(sp_model_path_16384)},
        results_dir=CALIBRATION_RESULTS_DIR,
        world_size=world_size,
        limit=None,
        dry_run=False,
        skip_existing=False,
        checkpoint_dir=None,
    )


def _run_analysis(
    *,
    trace_path: Path,
    manifest_path: Path,
) -> None:
    manifest = analyze_calibration(
        trace_path=trace_path,
        manifest_path=manifest_path,
    )
    summary = manifest.get("signal_summary", {})
    n_decisions = manifest.get("n_decisions_observed", 0)
    print(
        f"[exp26] stage=analyze trace={trace_path} "
        f"decisions={n_decisions} manifest={manifest_path}",
        flush=True,
    )
    for signal_name in ("utility_ema", "peak_utility", "peak_sharpness", "max_drift"):
        s = summary.get(signal_name, {})
        if not s:
            continue
        print(
            f"[exp26]   {signal_name:>20s}  n={s['n']:>6d}  "
            f"p25={s['p25']:.4g}  p50={s['p50']:.4g}  "
            f"p75={s['p75']:.4g}  p90={s['p90']:.4g}",
            flush=True,
        )
    bal = manifest.get("thresholds_balanced", {})
    agg = manifest.get("thresholds_aggressive", {})
    print(
        f"[exp26]   balanced eviction_threshold={bal.get('threshold'):.4g}  "
        f"aggressive eviction_threshold={agg.get('threshold'):.4g}",
        flush=True,
    )


def _run_headline(
    *,
    speed_config: dict[str, Any],
    world_size: int,
    seeds: list[int],
    budget_seconds: float,
    data_path: Path,
    sp_model_path_16384: Path,
    output_dir: Path,
    manifest_path: Path,
    arms: list[str] | None,
    limit: int | None,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    entries = build_arm_v1_matrix(
        speed_config=speed_config,
        calibration_manifest_path=manifest_path,
        world_size=world_size,
        budget_seconds=budget_seconds,
        seed_values=seeds,
        arms=arms,
    )
    print(
        f"[exp26] stage=headline entries={len(entries)} world_size={world_size} "
        f"budget={budget_seconds}s dry_run={dry_run}",
        flush=True,
    )
    if dry_run:
        _print_entries(entries)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(output_dir / "matrix.json", entries)
    _prebuild_cache(str(data_path))
    run_matrix_entries(
        entries=entries,
        runner_path=EXP23 / "runner_fast_path.py",
        data_path=str(data_path),
        sp_model_paths={16384: str(sp_model_path_16384)},
        results_dir=output_dir,
        world_size=world_size,
        limit=limit,
        dry_run=False,
        skip_existing=skip_existing,
        checkpoint_dir=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["smoke", "calibrate", "analyze", "headline", "all"],
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--sp-model-path-16384", type=Path, default=DEFAULT_SP_MODEL_16384
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--smoke-budget", type=float, default=30.0,
        help="Wall budget per smoke cell (seconds)",
    )
    parser.add_argument(
        "--smoke-seed", type=int, default=1337,
    )
    parser.add_argument(
        "--calibration-budget", type=float, default=180.0,
        help="Wall budget for the calibration cell (seconds)",
    )
    parser.add_argument(
        "--calibration-seed", type=int, default=1337,
    )
    parser.add_argument(
        "--headline-budget", type=float, default=600.0,
        help="Wall budget per headline cell (seconds)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1337, 2674, 4011],
    )
    parser.add_argument(
        "--arms", nargs="+", default=None,
        help="Restrict headline matrix to a subset of arms",
    )
    parser.add_argument(
        "--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--trace-path", type=Path, default=DEFAULT_CALIBRATION_TRACE,
    )
    parser.add_argument(
        "--headline-output-dir", type=Path, default=HEADLINE_RESULTS_DIR,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    speed_config = read_speed_config(args.config)

    if args.stage in ("smoke", "all"):
        _run_smoke(
            speed_config=speed_config,
            world_size=int(args.world_size),
            seed=int(args.smoke_seed),
            budget_seconds=float(args.smoke_budget),
            data_path=args.data_path,
            sp_model_path_16384=args.sp_model_path_16384,
            dry_run=bool(args.dry_run),
        )

    if args.stage in ("calibrate", "all"):
        _run_calibration(
            speed_config=speed_config,
            world_size=int(args.world_size),
            seed=int(args.calibration_seed),
            budget_seconds=float(args.calibration_budget),
            data_path=args.data_path,
            sp_model_path_16384=args.sp_model_path_16384,
            dry_run=bool(args.dry_run),
        )

    if args.stage in ("analyze", "all") and not args.dry_run:
        _run_analysis(
            trace_path=args.trace_path,
            manifest_path=args.manifest_path,
        )

    if args.stage in ("headline", "all"):
        _run_headline(
            speed_config=speed_config,
            world_size=int(args.world_size),
            seeds=[int(s) for s in args.seeds],
            budget_seconds=float(args.headline_budget),
            data_path=args.data_path,
            sp_model_path_16384=args.sp_model_path_16384,
            output_dir=args.headline_output_dir,
            manifest_path=args.manifest_path,
            arms=args.arms,
            limit=args.limit,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
