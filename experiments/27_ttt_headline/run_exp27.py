#!/usr/bin/env python3
"""Run Exp27 TTT headline matrix.

Three-stage discipline mirroring exp26:

  Stage 1 (calibrate)  : Stub — would probe each calc_type on the winning
                          trunk to fit 600s. The current stub skips the
                          probe and emits default hyperparams.
  Stage 2 (analyze)    : Read the probe trace (when one exists) and write
                          ``calibration/manifest.json``. Today the analyzer
                          writes the stub defaults.
  Stage 3 (headline)   : Winning trunk x N seeds. Each entry trains one
                          trunk and runs the selected calc_types as serial
                          eval passes. The default selector is the floor
                          plus packet-clean adaptive carry.

``--dry-run`` is strictly side-effect-free: no ``mkdir``, no ``write_matrix``,
no manifest write, no run. Each stage's dry-run path prints entries (or the
manifest dict for analyze) and returns before any disk write.

Usage:

  # one-shot calibrate -> analyze -> headline on 4xH100
  python experiments/27_ttt_headline/run_exp27.py --stage all

  # calibrate-only (stub)
  python experiments/27_ttt_headline/run_exp27.py --stage calibrate

  # analyze-only (writes the stub manifest)
  python experiments/27_ttt_headline/run_exp27.py --stage analyze

  # headline only (requires manifest from a prior analyze)
  python experiments/27_ttt_headline/run_exp27.py --stage headline

  # dry-run any stage; prints entries and returns without touching disk
  python experiments/27_ttt_headline/run_exp27.py --stage all --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
EXP23 = REPO / "experiments" / "23_fast_path"
EXP24 = REPO / "experiments" / "24_training_time_bundle"
EXP27 = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXP23 / "configs" / "base_seq_epoch_lr0064_full_corpus.yaml"
DEFAULT_DATA_PATH = (
    REPO / "baselines" / "parameter_golf" / "datasets" / "fineweb10B_sp16384"
)
DEFAULT_SP_MODEL_16384 = (
    REPO / "baselines" / "parameter_golf" / "tokenizers" / "fineweb_16384_bpe.model"
)
DEFAULT_VAL_CACHE_DIR = EXP27 / "val_cache"

sys.path.insert(0, str(EXP23))
sys.path.insert(0, str(EXP24))
sys.path.insert(0, str(EXP27))
sys.path.insert(0, str(REPO / "src"))

from exp27 import (  # noqa: E402
    CALC_TYPES_DEFAULT,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RESULTS_DIR,
    build_ttt_headline_matrix,
)
from calibrate import (  # noqa: E402
    _build_stub_manifest_dict,
    analyze as analyze_calibration,
)


def _print_entries(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        print(f"[exp27] {entry['name']}")
        print(json.dumps(entry, indent=2, sort_keys=True, default=str))


def _print_manifest(manifest: dict[str, Any]) -> None:
    print("[exp27] manifest preview")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


def _run_calibration(*, dry_run: bool) -> None:
    """Stage 1. Stub today; would launch a probe cell when a real one exists."""
    print(
        f"[exp27] stage=calibrate stub=True dry_run={dry_run}",
        flush=True,
    )
    if dry_run:
        # Stub probe is a no-op; surface what the stub would record.
        _print_manifest(_build_stub_manifest_dict())
        return
    # Real probe cell will land here once a winning trunk is fixed; for now
    # the stub does no probing of its own and the manifest is written by the
    # analyze stage.


def _run_analysis(
    *,
    manifest_path: Path,
    trace_path: Path | None,
    dry_run: bool,
) -> None:
    """Stage 2. Writes the manifest (stub defaults today)."""
    if dry_run:
        manifest = _build_stub_manifest_dict(trace_path=trace_path)
        print(
            f"[exp27] stage=analyze manifest={manifest_path} dry_run=True",
            flush=True,
        )
        _print_manifest(manifest)
        return
    manifest = analyze_calibration(
        trace_path=trace_path,
        manifest_path=manifest_path,
    )
    n = len(manifest.get("calc_type_hyperparams", {}))
    print(
        f"[exp27] stage=analyze manifest={manifest_path} "
        f"calc_types={n} (stub={manifest.get('source_trace') == 'stub'})",
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
    val_cache_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    calc_types: list[str] | None,
    checkpoint_path: Path | None,
    limit: int | None,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    """Stage 3. Build entries, then either preview or launch."""
    if dry_run and not Path(manifest_path).exists():
        # Strict dry-run cannot fabricate a manifest on disk. Use the stub
        # dict in-memory for the preview and skip the matrix build, since
        # build_ttt_headline_matrix loads from disk by contract.
        print(
            f"[exp27] stage=headline dry_run=True world_size={world_size} "
            f"budget={budget_seconds}s seeds={seeds} manifest_absent=True",
            flush=True,
        )
        print(
            "[exp27] manifest at "
            f"{manifest_path} does not exist; previewing stub defaults"
        )
        _print_manifest(_build_stub_manifest_dict())
        return
    entries = build_ttt_headline_matrix(
        speed_config=speed_config,
        calibration_manifest_path=Path(manifest_path),
        checkpoint_path=checkpoint_path,
        world_size=world_size,
        budget_seconds=budget_seconds,
        seed_values=list(seeds),
        calc_types=list(calc_types) if calc_types is not None else None,
    )
    print(
        f"[exp27] stage=headline entries={len(entries)} world_size={world_size} "
        f"budget={budget_seconds}s dry_run={dry_run}",
        flush=True,
    )
    if dry_run:
        _print_entries(entries)
        return
    if entries and not val_cache_dir.is_dir():
        raise FileNotFoundError(
            "Exp27 calc_types require a ValCache directory; build it with "
            "scripts/build_exp20_val_cache.py and pass --val-cache-dir. "
            f"Missing: {val_cache_dir}"
        )

    from fast_path import write_matrix  # noqa: E402
    from launch import run_matrix_entries  # noqa: E402
    from chaoscontrol.data import load_fineweb_tokens  # noqa: E402

    output_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(output_dir / "matrix.json", entries)
    train_tokens, val_tokens = load_fineweb_tokens(str(data_path))
    print(
        "[exp27] data cache ready "
        f"train={int(train_tokens.numel()):,} val={int(val_tokens.numel()):,}",
        flush=True,
    )
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
        val_cache_dir=val_cache_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["calibrate", "analyze", "headline", "all"],
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--sp-model-path-16384", type=Path, default=DEFAULT_SP_MODEL_16384
    )
    parser.add_argument(
        "--val-cache-dir",
        type=Path,
        default=DEFAULT_VAL_CACHE_DIR,
        help=(
            "ValCache directory consumed by runner_fast_path when calc_types "
            "are enabled."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Reserved for a future explicit checkpoint-load path. Currently "
            "unsupported because runner_fast_path does not load this field."
        ),
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1337, 2674, 4011]
    )
    parser.add_argument(
        "--calc-types",
        nargs="+",
        default=list(CALC_TYPES_DEFAULT),
        help="Restrict headline calc_types to a subset of the registered names",
    )
    parser.add_argument(
        "--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Forward-compat probe trace path; the stub analyzer ignores it",
    )
    parser.add_argument(
        "--headline-output-dir", type=Path, default=DEFAULT_RESULTS_DIR
    )
    parser.add_argument(
        "--headline-budget",
        type=float,
        default=600.0,
        help="Wall budget per headline cell (seconds)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from fast_path import read_speed_config  # noqa: E402

    speed_config = read_speed_config(args.config)

    if args.stage in ("calibrate", "all"):
        _run_calibration(dry_run=bool(args.dry_run))

    if args.stage in ("analyze", "all"):
        _run_analysis(
            manifest_path=args.manifest_path,
            trace_path=args.trace_path,
            dry_run=bool(args.dry_run),
        )

    if args.stage in ("headline", "all"):
        _run_headline(
            speed_config=speed_config,
            world_size=int(args.world_size),
            seeds=[int(s) for s in args.seeds],
            budget_seconds=float(args.headline_budget),
            data_path=args.data_path,
            sp_model_path_16384=args.sp_model_path_16384,
            val_cache_dir=args.val_cache_dir,
            output_dir=args.headline_output_dir,
            manifest_path=args.manifest_path,
            calc_types=list(args.calc_types) if args.calc_types else None,
            checkpoint_path=args.checkpoint_path,
            limit=args.limit,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
