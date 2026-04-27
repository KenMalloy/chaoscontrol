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
    build_episodic_controller_v1_matrix,
    build_episodic_dw_curation_v1_matrix,
    build_episodic_ttt_v1_matrix,
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


# Smoke-first policy: before a real matrix launch, run ONE cell with a
# tight budget against the most-instrumented arm and assert that the
# load-bearing telemetry actually fires. Catches the silent-fallback
# failure mode where episodic-on-CUDA runs land on the legacy CPU path
# and look fine in logs but never exercise the new producer pipeline.
# See docs/reports/2026-04-27-step3-results.md for the incident the
# policy is meant to prevent.
SMOKE_BUDGET_SECONDS = 90.0
SMOKE_ARM_PREFIX_BY_MATRIX = {
    "episodic_controller_v1": "arm_e_simplex_warm_online",
}


def smoke_check_result(
    *,
    result_path: Path,
    entry: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate a smoke cell's result JSON. Returns (ok, failures).

    The contract: when ``episodic_enabled`` is on and the run had a CUDA
    device, every load-bearing producer counter must have fired. Silent
    fallback to the legacy CPU path is the failure mode this catches.
    """
    failures: list[str] = []
    if not result_path.exists():
        return False, [f"result JSON not produced at {result_path}"]
    try:
        d = json.loads(result_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return False, [f"result JSON malformed: {exc!r}"]
    if "error" in d:
        return False, [f"runner reported error: {d.get('error')!r}"]
    train = d.get("train", {})
    if not isinstance(train, dict) or "steps" not in train:
        return False, ["result JSON missing train block / train.steps"]
    aw = train.get("mechanisms", {}).get("episodic_async_writes", {})
    if entry.get("episodic_enabled") and aw.get("enabled"):
        # CUDA-side gates that catch the 2026-04-27 silent-fallback mode.
        if not aw.get("cuda_stream_enabled"):
            failures.append(
                "cuda_stream_enabled=false on episodic-enabled cell — "
                "GPU-pack producer pipeline did not activate; runs would "
                "silently fall back to the legacy CPU producer"
            )
        if int(aw.get("submitted_batches", 0)) == 0:
            failures.append("submitted_batches=0 — no producer batches were submitted")
        if int(aw.get("pushed", 0)) == 0:
            failures.append("pushed=0 — no events made it onto the shm ring")
        if int(aw.get("drain_errors", 0)) != 0:
            failures.append(f"drain_errors={aw['drain_errors']}")
        if aw.get("publisher_error"):
            failures.append(f"publisher_error={aw['publisher_error']!r}")
    if "simplex" in str(entry.get("name", "")):
        trace_path = entry.get("episodic_controller_simplex_trace_path")
        if trace_path:
            tp = Path(trace_path)
            if not tp.exists():
                failures.append(f"simplex trace not produced at {tp}")
            else:
                n_decision = n_credit = 0
                min_entropy = float("inf")
                with tp.open("r") as fh:
                    for i, line in enumerate(fh):
                        if i >= 2000:
                            break
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        rt = row.get("row_type")
                        if rt == "decision":
                            n_decision += 1
                            ent = row.get("entropy")
                            if isinstance(ent, (int, float)) and ent < min_entropy:
                                min_entropy = float(ent)
                        elif rt == "credit":
                            n_credit += 1
                if n_decision == 0:
                    failures.append("simplex trace has no decision rows")
                if n_credit == 0:
                    failures.append("simplex trace has no credit rows (no replay reached the learner)")
                # ln(16) ≈ 2.7726. If every sampled decision is at near-max
                # entropy the policy is degenerate (all-zero weights, empty
                # candidates, etc.). One non-uniform decision proves the
                # path is at least live; full-matrix tells us whether it
                # learns over time.
                if min_entropy > 2.5 and n_decision > 0:
                    failures.append(
                        f"all {n_decision} sampled decision rows have entropy > 2.5 "
                        f"(min={min_entropy:.4f}); policy stuck at near-uniform"
                    )
    return len(failures) == 0, failures


def _run_smoke(
    *,
    matrix_name: str,
    entries: list[dict[str, Any]],
    runner_path: Path,
    data_path: str,
    sp_model_paths: dict[int, str],
    full_results_dir: Path,
    world_size: int,
) -> tuple[bool, list[str]]:
    """Pick the most-instrumented arm, run it with a 90s budget, assert."""
    prefix = SMOKE_ARM_PREFIX_BY_MATRIX.get(matrix_name)
    if prefix is None:
        return True, [f"smoke skipped: no smoke arm registered for {matrix_name}"]
    candidates = [e for e in entries if prefix in str(e.get("name", ""))]
    if not candidates:
        return False, [f"smoke arm prefix {prefix!r} matched no entries in matrix {matrix_name}"]
    smoke_entry = dict(candidates[0])
    smoke_entry["budget_seconds"] = SMOKE_BUDGET_SECONDS
    smoke_entry["name"] = f"smoke_{smoke_entry['name']}"
    smoke_dir = full_results_dir / ".smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    # Re-route the trace path so the smoke run doesn't clobber any
    # existing trace file from a real matrix launch.
    if smoke_entry.get("episodic_controller_simplex_trace_path"):
        smoke_entry["episodic_controller_simplex_trace_path"] = str(
            smoke_dir / "smoke_simplex_trace.ndjson"
        )
    print(
        f"[smoke] running 1 cell ({smoke_entry['name']}) for "
        f"{SMOKE_BUDGET_SECONDS:.0f}s before full matrix",
        flush=True,
    )
    run_matrix_entries(
        entries=[smoke_entry],
        runner_path=runner_path,
        data_path=data_path,
        sp_model_paths=sp_model_paths,
        results_dir=smoke_dir,
        world_size=world_size,
        limit=None,
        dry_run=False,
        skip_existing=False,
        checkpoint_dir=None,
    )
    result_path = smoke_dir / f"{smoke_entry['name']}.json"
    return smoke_check_result(result_path=result_path, entry=smoke_entry)


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


def _episodic_eval_args_from_entry(entry: dict[str, Any]) -> list[str]:
    # Per-entry eval-side episodic flags. F1 controller_v1 arms set these
    # to differentiate cold/warm cache and frozen/online controller TTT;
    # other matrices leave the keys absent and the scorer's defaults
    # (cache off, source=auto) take over.
    args: list[str] = []
    cache_enabled = entry.get("eval_episodic_cache_enabled")
    if cache_enabled is True:
        args.append("--episodic-cache-enabled")
    elif cache_enabled is False:
        args.append("--no-episodic-cache-enabled")
    mode = entry.get("eval_episodic_cache_mode")
    if mode == "cold":
        args.extend(["--episodic-cache-source", "fresh"])
    elif mode == "warm":
        args.extend(["--episodic-cache-source", "checkpoint"])
    capacity = entry.get("eval_episodic_cache_capacity")
    if capacity is not None:
        args.extend(["--episodic-cache-capacity", str(int(capacity))])
    span = entry.get("eval_episodic_span_length")
    if span is not None:
        args.extend(["--episodic-span-length", str(int(span))])
    key_dim = entry.get("eval_episodic_key_rep_dim")
    if key_dim is not None:
        args.extend(["--episodic-key-rep-dim", str(int(key_dim))])
    grace = entry.get("eval_episodic_grace_steps")
    if grace is not None:
        args.extend(["--episodic-grace-steps", str(int(grace))])
    fp_window = entry.get("eval_episodic_fingerprint_window")
    if fp_window is not None:
        args.extend(["--episodic-fingerprint-window", str(int(fp_window))])
    reset = entry.get("eval_episodic_cache_reset_per_doc")
    if reset is True:
        args.append("--episodic-cache-reset-per-doc")
    elif reset is False:
        args.append("--no-episodic-cache-reset-per-doc")
    online = entry.get("controller_train_online")
    if online is True:
        args.append("--controller-train-online")
    return args


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
    scorer = REPO / "scripts" / "run_exp20_full_val_score.py"
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
        cmd.extend(_episodic_eval_args_from_entry(entry))
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
    if matrix == "episodic_dw_curation_v1":
        return build_episodic_dw_curation_v1_matrix(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
            seed_values=seeds,
        )
    if matrix == "episodic_ttt_v1":
        return build_episodic_ttt_v1_matrix(
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
    if matrix == "episodic_controller_v1":
        return build_episodic_controller_v1_matrix(
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
        "episodic_dw_curation_v1",
        "episodic_ttt_v1",
        "episodic_controller_v1",
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
            "episodic_dw_curation_v1",
            "episodic_ttt_v1",
            "cd_first_smoke",
            "cd_multiseed",
            "episodic_controller_v1",
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
        help="After training each entry, run scripts/run_exp20_full_val_score.py on the saved checkpoint.",
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
        help="Eval budget passed to run_exp20_full_val_score.py.",
    )
    parser.add_argument(
        "--smoke",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before launching the full matrix, run ONE cell with a "
            f"{SMOKE_BUDGET_SECONDS:.0f}s budget against the most-instrumented arm "
            "and assert that load-bearing telemetry fired. Catches silent "
            "fallback to the legacy CPU producer. Auto-skipped for matrices "
            "without a registered smoke arm."
        ),
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
    # Pre-create the trace dir so the C++ writer can open files in
    # append mode without checking. Cells with no trace path skip this.
    trace_targets = {
        Path(entry["episodic_controller_simplex_trace_path"]).parent
        for entry in entries
        if entry.get("episodic_controller_simplex_trace_path")
    }
    for trace_dir in trace_targets:
        trace_dir.mkdir(parents=True, exist_ok=True)
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

    if not effective_dry_run and args.smoke:
        smoke_ok, smoke_msgs = _run_smoke(
            matrix_name=args.matrix,
            entries=entries,
            runner_path=EXP23 / "runner_fast_path.py",
            data_path=str(args.data_path),
            sp_model_paths={16384: str(args.sp_model_path_16384)},
            full_results_dir=args.output_dir,
            world_size=world_size,
        )
        if not smoke_ok:
            print("[smoke] FAILED — full matrix NOT launched. Failures:", flush=True)
            for msg in smoke_msgs:
                print(f"  - {msg}", flush=True)
            return 2
        if smoke_msgs:
            for msg in smoke_msgs:
                print(f"[smoke] {msg}", flush=True)
        else:
            print("[smoke] passed; launching full matrix", flush=True)

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
