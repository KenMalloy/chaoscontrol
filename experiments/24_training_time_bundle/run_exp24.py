#!/usr/bin/env python3
"""Run Exp24 training-time bundle matrices."""
from __future__ import annotations

import argparse
import json
import subprocess
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
    build_criticality_distillation_first_smoke_matrix,
    build_criticality_distillation_multiseed_matrix,
    build_fastslow_dreamworld_matrix,
    build_first_wave_mechanism_matrix,
    build_phase_a_sampling_matrix,
    build_phase0_confirm,
    build_phase0_dreamworld_sweep,
    build_phase0_fastslow_only_control,
    build_phase0_fastslow_sweep,
    build_ring0_control_matrix,
    build_scopt_calibration_sweep_matrix,
    build_scopt_overhead_gate_matrix,
    build_semantic_overhead_gate_matrix,
)
from fast_path import read_speed_config, write_matrix  # noqa: E402
from launch import DRY_RUN_RDZV_PORT, pick_free_port, run_matrix_entries  # noqa: E402


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


def _score_full_val(
    *,
    entries: list[dict[str, Any]],
    checkpoint_dir: Path,
    results_dir: Path,
    world_size: int,
    cache_dir: Path,
    budget_seconds: float,
    dry_run: bool = False,
) -> list[list[str]]:
    full_val_dir = results_dir / "full_val"
    full_val_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    scorer = REPO / "scripts" / "run_exp20_fast_score.py"
    commands: list[list[str]] = []
    for entry in entries:
        name = str(entry["name"])
        ckpt = checkpoint_dir / f"{name}.pt"
        if not dry_run and not ckpt.exists():
            continue
        summary_path = full_val_dir / f"{name}.summary.json"
        if summary_path.exists():
            continue
        jsonl_path = full_val_dir / f"{name}.jsonl"
        log_path = logs_dir / f"{name}.full_val.log"
        rdzv_port = DRY_RUN_RDZV_PORT if dry_run else pick_free_port()
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={int(world_size)}",
            f"--rdzv-endpoint=localhost:{rdzv_port}",
            "--rdzv-backend=c10d",
            f"--rdzv-id=score_{name}_{rdzv_port}",
            str(scorer),
            "--cache-dir",
            str(cache_dir),
            "--checkpoint-path",
            str(ckpt),
            "--output-path",
            str(jsonl_path),
            "--summary-path",
            str(summary_path),
            "--chunk-size",
            "256",
            "--budget-seconds",
            str(float(budget_seconds)),
            "--doc-batch-size",
            "4096",
            "--max-forward-tokens",
            "auto",
            "--score-boundary-targets",
            "--doc-packing",
            "chunk_count_tail",
        ]
        commands.append(cmd)
        if dry_run:
            continue
        with log_path.open("w") as log:
            log.write("+ " + " ".join(cmd) + "\n")
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            log.write(f"\n[returncode] {proc.returncode}\n")
    return commands


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
    if matrix == "scopt_overhead_gate":
        entries: list[dict[str, Any]] = []
        for seed in seeds:
            entries.extend(
                build_scopt_overhead_gate_matrix(
                    speed_config=speed_config,
                    seed=seed,
                    world_size=world_size,
                    budget_seconds=budget_seconds,
                )
            )
        return entries
    if matrix == "scopt_calibration_sweep":
        entries: list[dict[str, Any]] = []
        for seed in seeds:
            entries.extend(
                build_scopt_calibration_sweep_matrix(
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
    if matrix == "fastslow_dreamworld":
        return build_fastslow_dreamworld_matrix(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seed_values=seeds,
        )
    if matrix == "phase0_dreamworld_sweep":
        return build_phase0_dreamworld_sweep(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seed_values=seeds,
        )
    if matrix == "phase0_fastslow_sweep":
        return build_phase0_fastslow_sweep(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seed_values=seeds,
        )
    if matrix == "phase0_confirm":
        return build_phase0_confirm(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seed_values=seeds,
        )
    if matrix == "phase0_fastslow_only_control":
        return build_phase0_fastslow_only_control(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seed_values=seeds,
        )
    if matrix == "cd_first_smoke":
        return build_criticality_distillation_first_smoke_matrix(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seed_values=seeds,
        )
    if matrix == "cd_multiseed":
        return build_criticality_distillation_multiseed_matrix(
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


def _default_world_size_for_matrix(matrix: str) -> int:
    if matrix in {"semantic_overhead_gate", "scopt_overhead_gate", "scopt_calibration_sweep"}:
        return 1
    if matrix == "cd_first_smoke":
        return 1
    if matrix == "cd_multiseed":
        return 1
    if matrix in {
        "phase0_dreamworld_sweep",
        "phase0_fastslow_sweep",
        "phase0_confirm",
        "phase0_fastslow_only_control",
    }:
        return 4
    return 8


def _default_budget_for_matrix(matrix: str) -> float:
    """Smoke matrices need a budget that clears the optimizer's warmup.

    - ``semantic_overhead_gate``: SemanticOptimizer has no warmup; 90s is
      enough to read a β distribution post-saturation.
    - ``scopt_overhead_gate``: ScOpt's default ``warmup_steps=200`` gates
      pressure/rare-EMA writes; 180s at bs=512/~11 steps-per-second gives
      ~1800 steps, ~1600 post-warmup, enough for Tier 0 probes.
    """
    if matrix == "semantic_overhead_gate":
        return 90.0
    if matrix == "scopt_overhead_gate":
        return 180.0
    if matrix == "scopt_calibration_sweep":
        return 1200.0
    return 600.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        choices=[
            "ring0_control",
            "phase_a_sampling",
            "semantic_overhead_gate",
            "scopt_overhead_gate",
            "scopt_calibration_sweep",
            "first_wave",
            "fastslow_dreamworld",
            "phase0_dreamworld_sweep",
            "phase0_fastslow_sweep",
            "phase0_confirm",
            "phase0_fastslow_only_control",
            "cd_first_smoke",
            "cd_multiseed",
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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="If set, save per-entry training checkpoints here via runner --output-ckpt.",
    )
    parser.add_argument(
        "--full-val-score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After training each entry, run scripts/run_exp20_fast_score.py on the saved checkpoint.",
    )
    parser.add_argument(
        "--val-cache-dir",
        type=Path,
        default=Path("/workspace/cache/exp23_val_16384"),
        help="Tokenized val cache dir for the fast scorer.",
    )
    parser.add_argument(
        "--val-budget-seconds",
        type=float,
        default=600.0,
        help="Eval budget passed to run_exp20_fast_score.py.",
    )
    args = parser.parse_args(argv)

    default_world_size = _default_world_size_for_matrix(args.matrix)
    default_budget = _default_budget_for_matrix(args.matrix)
    world_size = int(args.world_size) if args.world_size is not None else default_world_size
    budget = float(args.budget) if args.budget is not None else default_budget
    checkpoint_dir = args.checkpoint_dir
    if args.full_val_score and checkpoint_dir is None:
        checkpoint_dir = args.output_dir / "checkpoints"

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
        if args.full_val_score:
            print(
                "[exp24] full-val-score enabled "
                f"checkpoint_dir={checkpoint_dir} val_cache_dir={args.val_cache_dir} "
                f"val_budget_seconds={float(args.val_budget_seconds)}",
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
        checkpoint_dir=checkpoint_dir,
    )
    if args.full_val_score:
        full_val_commands = _score_full_val(
            entries=selected,
            checkpoint_dir=checkpoint_dir,
            results_dir=args.output_dir,
            world_size=world_size,
            cache_dir=args.val_cache_dir,
            budget_seconds=float(args.val_budget_seconds),
            dry_run=effective_dry_run,
        )
        if effective_dry_run:
            summary["full_val_commands"] = full_val_commands
        else:
            summary = run_matrix_entries(
                entries=entries,
                runner_path=EXP23 / "runner_fast_path.py",
                data_path=str(args.data_path),
                sp_model_paths={16384: str(args.sp_model_path_16384)},
                results_dir=args.output_dir,
                world_size=world_size,
                limit=args.limit,
                dry_run=True,
                skip_existing=True,
                checkpoint_dir=checkpoint_dir,
            )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
