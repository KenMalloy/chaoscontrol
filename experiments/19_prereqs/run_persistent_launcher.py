#!/usr/bin/env python3
"""Top-level launcher for the persistent-DDP multi-seed worker.

Replaces the per-seed ``torchrun`` spawn pattern from
``experiments/18_throughput_levers/run_exp18_test10.py``. Builds a
matrix of ``(condition, seed, config)`` entries, drops fp8 entries if
transformer_engine is unavailable on the pod, writes the matrix to
``/tmp/`` as JSON, and shells out to a single ``torchrun`` invocation
that runs every entry in order inside one persistent process.

Usage:

    python experiments/19_prereqs/run_persistent_launcher.py \\
        --data-path baselines/parameter_golf/datasets/fineweb10B_sp16384 \\
        --sp-model-path baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \\
        --output-dir experiments/19_prereqs/results_smoke/ \\
        --world-size 4 --base-lr 0.128 --budget 600

Exits with non-zero if the torchrun subprocess fails OR if any entry
wrote an error marker without a corresponding success JSON. Does NOT
run a summarizer — the persistent runner's job is to produce per-entry
JSONs with the same schema as ``runner_exp18_ssm``. Downstream
analysis (paired t-test, throughput ratios, decision gates) lives in
the experiment-specific summarizer the caller wires up on top of this.

Design notes vs run_exp18_test10.py:

    - One torchrun per matrix, not one per seed. A four-seed × two-
      precision matrix goes from 8 torchrun launches (≈80 min overhead)
      to 1 torchrun launch (≈10 min overhead amortized over 8 entries).
    - No YAML config per seed. The matrix is a single JSON list of
      fully-resolved config dicts, written once.
    - fp8 skip-on-missing-TE happens in the launcher (before matrix
      write), not in the runner. The runner sees only entries whose
      precision can actually run; error-marker JSONs for the skipped
      fp8 entries are pre-written so the downstream summarizer treats
      them as ``errored``, not silently ``absent``.
    - No OOM auto-skip at launch level. The persistent runner writes
      per-entry error markers on failure and continues; a condition
      that OOMs on every seed shows up as four error-marker JSONs
      which the summarizer can interpret the same way the frozen
      harness's ``skip_oom_conditions`` logic did.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
RUNNER = EXPERIMENT / "runner_persistent_ddp.py"

# The canonical submission regime matching Exp 18 Test 4b's winner —
# used when the launcher is invoked as a smoke test or as the
# fp8-vs-bf16 matrix the old run_exp18_test10.py covered. Callers
# that want a different matrix pass their own via --matrix-json.
DEFAULT_SEEDS = [1337, 2674, 4011, 5348]

# Error-string patterns that mean "not a real failure." The launcher
# pre-writes skip markers for entries it intentionally drops (e.g. fp8
# entries on a pod without transformer_engine); those JSONs have 'error'
# set but shouldn't cause a non-zero exit. Anything else with an 'error'
# key is a real failure and fails the run closed.
_BENIGN_ERROR_PATTERNS: tuple[str, ...] = (
    "skipped: transformer_engine unavailable on pod",
)

# Subset of _BENIGN_ERROR_PATTERNS that specifically indicate
# "transformer_engine was missing, so we never actually tried this entry."
# reject_stale_skip_markers only treats these as stale on a TE-capable
# pod — adding a non-TE benign pattern (e.g., OOM-skip) to the main
# tuple must not cause those markers to be wrongly flagged as stale.
_TE_UNAVAILABLE_PATTERNS: tuple[str, ...] = (
    "skipped: transformer_engine unavailable on pod",
)


def _check_output_integrity(
    output_dir: Path,
    entries: list[dict[str, Any]],
    benign_error_patterns: tuple[str, ...] = _BENIGN_ERROR_PATTERNS,
) -> tuple[int, int, list[str]]:
    """Classify every expected output JSON into success / benign-skip / real-error.

    Returns ``(success_count, benign_skip_count, real_errors)`` where
    ``real_errors`` is a list of human-readable strings, one per problem.
    The caller returns non-zero iff ``real_errors`` is non-empty. This
    replaces the previous 'just count JSON files' check, which returned
    zero even when every output was an error marker.
    """
    success = 0
    benign = 0
    real_errors: list[str] = []
    for entry in entries:
        name = str(entry.get("name", "?"))
        seed = int(entry.get("seed", 0))
        out_path = output_dir / f"{name}_s{seed}.json"
        if not out_path.exists():
            real_errors.append(f"{name}_s{seed}: missing output file")
            continue
        try:
            data = json.loads(out_path.read_text())
        except Exception as exc:  # noqa: BLE001 — surface the parse error
            real_errors.append(f"{name}_s{seed}: malformed JSON ({exc})")
            continue
        if "error" in data:
            err_str = str(data["error"])
            if any(pat in err_str for pat in benign_error_patterns):
                benign += 1
            else:
                real_errors.append(f"{name}_s{seed}: error={err_str}")
        elif "eval" in data and "train" in data:
            success += 1
        else:
            real_errors.append(
                f"{name}_s{seed}: neither 'error' nor ('eval'+'train') keys present"
            )
    return success, benign, real_errors


def _validate_matrix_world_size(
    entries: list[dict[str, Any]],
    cli_world_size: int,
) -> None:
    """Raise if any entry's ``world_size`` differs from the CLI --world-size.

    The runner uses the torchrun-provided process-group size for sharding
    and training, not the entry's field. Accepting a matrix whose entries
    disagree silently runs one regime (the CLI world size) while writing
    JSONs tagged with another (the entry's field), breaking downstream
    analysis. Pre-flight check refuses the mismatch before any subprocess
    launches.
    """
    mismatches = [
        (str(entry.get("name", "?")), int(entry.get("seed", 0)),
         entry.get("world_size"))
        for entry in entries
        if entry.get("world_size") != cli_world_size
    ]
    if mismatches:
        preview = mismatches[:5]
        more = f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        raise ValueError(
            f"matrix entries' world_size disagrees with --world-size={cli_world_size}: "
            f"{preview}{more}"
        )


def _base_config(
    *,
    world_size: int,
    base_lr: float,
    precision: str,
    seed: int,
    name: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Build one matrix entry matching the Test 4b / Test 10 regime.

    Mirrors ``run_exp18_test10._base`` and adds the ``name`` / ``seed``
    fields the persistent runner uses to key output JSONs. Overridable
    via kwargs for alternate matrices.
    """
    cfg = {
        "name": name,
        "seed": seed,
        "precision": precision,
        "model_type": "ssm",
        "vocab_size": 16384,
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 1024,
        "eval_batches": 16,
        "a_mode": "diag",
        "base_lr": base_lr,
        "activation_checkpoint": True,
        "optimizer": "muon",
        "chunk_size": 64,
        "world_size": world_size,
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
        "grad_clip_norm": 1.0,
        "weight_decay": 0.01,
        "device": "auto",
        "dtype": "bf16",
    }
    cfg.update(overrides)
    return cfg


def build_default_matrix(
    world_size: int,
    base_lr: float,
    seeds: list[int],
    conditions: list[str],
) -> list[dict[str, Any]]:
    """Cartesian product of conditions × seeds, returned as flat list.

    Ordering: all seeds of condition 0, then all seeds of condition 1,
    and so on. Groups by condition so the per-entry JSON filenames are
    created in a predictable order and partial-failure triage is easier.
    """
    entries: list[dict[str, Any]] = []
    for precision in conditions:
        for seed in seeds:
            entries.append(_base_config(
                world_size=world_size,
                base_lr=base_lr,
                precision=precision,
                seed=seed,
                name=precision,
            ))
    return entries


def _te_is_available() -> bool:
    """Probe transformer_engine importability without side effects.

    Isolated from ``filter_matrix_for_te`` so ``reject_stale_skip_markers``
    can share the same probe — keeps the TE-availability definition
    single-sourced.
    """
    sys.path.insert(0, str(REPO / "src"))
    try:
        from chaoscontrol.precision import _check_te_available
    except Exception:
        return False
    return _check_te_available()


def reject_stale_skip_markers(
    entries: list[dict[str, Any]],
    output_dir: Path,
    te_unavailable_patterns: tuple[str, ...] = _TE_UNAVAILABLE_PATTERNS,
    te_probe: "callable[[], bool]" = _te_is_available,
) -> None:
    """Fail-closed if fp8 entries have stale TE-unavailable skip markers.

    The failure mode: a prior run on a TE-less pod pre-wrote skip markers
    (``{"error": "skipped: transformer_engine unavailable on pod"}``) for
    fp8 entries. On a subsequent run *on a TE-capable pod*, the idempotent
    existing-output check in the runner honors those markers and the
    launcher's ``_check_output_integrity`` classifies them as ``benign``
    skips — so the launcher exits rc=0 having produced zero fp8 results,
    despite the user explicitly requesting fp8. Silent successes on
    reproducibility pods are the exact contamination pattern the 2026-04-17
    Exp 19 planning locked in guards against.

    This check runs before ``filter_matrix_for_te``: if TE is now available
    AND any fp8 entry in the current matrix has a matching stale marker on
    disk, raise with an actionable message. The caller is expected to
    delete the markers (or --skip-fp8) and re-run.

    ``te_probe`` is injected so tests can drive both branches without
    importing ``transformer_engine``.
    """
    if not te_probe():
        return
    stale: list[Path] = []
    for entry in entries:
        if entry.get("precision") != "fp8":
            continue
        name = str(entry.get("name", "?"))
        seed = int(entry.get("seed", 0))
        out_path = output_dir / f"{name}_s{seed}.json"
        if not out_path.exists():
            continue
        try:
            data = json.loads(out_path.read_text())
        except Exception:
            # Malformed markers are caught downstream by _check_output_integrity.
            continue
        err_str = str(data.get("error", ""))
        if any(pat in err_str for pat in te_unavailable_patterns):
            stale.append(out_path)
    if stale:
        preview = "\n  ".join(str(p) for p in stale[:5])
        more = f"\n  (+{len(stale) - 5} more)" if len(stale) > 5 else ""
        raise RuntimeError(
            f"transformer_engine is available on this host, but "
            f"{len(stale)} fp8 entries in the matrix have stale skip markers "
            f"from a prior TE-less run. Honoring them would silently succeed "
            f"with rc=0 and zero fp8 results. Delete them to re-run fp8:\n  "
            f"{preview}{more}"
        )


def filter_matrix_for_te(
    entries: list[dict[str, Any]],
    output_dir: Path,
    te_probe: "callable[[], bool]" = _te_is_available,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop fp8 entries if transformer_engine isn't importable.

    Returns ``(runnable_entries, skipped_entries)``. Callers pre-write
    error-marker JSONs for every skipped entry so the downstream
    summarizer sees ``errored`` rather than ``absent`` — an absent JSON
    is ambiguous (could be "never launched" or "not yet finished"), an
    error marker is decisive.

    ``te_probe`` is injected so tests can drive both branches without
    importing ``transformer_engine``.
    """
    has_fp8 = any(e.get("precision") == "fp8" for e in entries)
    if not has_fp8:
        return entries, []

    if te_probe():
        return entries, []

    runnable = [e for e in entries if e.get("precision") != "fp8"]
    skipped = [e for e in entries if e.get("precision") == "fp8"]
    print(
        f"[launcher] transformer_engine unavailable on this host; "
        f"dropping {len(skipped)} fp8 entries from the matrix.",
        flush=True,
    )
    return runnable, skipped


def write_skip_markers(
    skipped: list[dict[str, Any]],
    output_dir: Path,
    reason: str,
) -> None:
    """Pre-write one error-marker JSON per skipped entry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in skipped:
        name = str(entry["name"])
        seed = int(entry["seed"])
        out_path = output_dir / f"{name}_s{seed}.json"
        if out_path.exists():
            continue
        payload = {"config": entry, "error": reason}
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.rename(out_path)
        print(f"[launcher] wrote skip marker {out_path}", flush=True)


def build_torchrun_cmd(
    *,
    nproc_per_node: int,
    matrix_path: Path,
    data_path: str,
    sp_model_path: str,
    output_dir: Path,
    budget: float,
    rdzv_port: int = 29500,
) -> list[str]:
    """Construct the single torchrun argv for the persistent worker.

    Uses ``python -m torch.distributed.run`` for version-resilience
    (same rationale as ``_harness.build_launch_cmd``).
    """
    return [
        sys.executable,
        "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        f"--rdzv-endpoint=localhost:{rdzv_port}",
        "--rdzv-backend=c10d",
        f"--rdzv-id=cc_exp19_persistent_{rdzv_port}",
        str(RUNNER),
        "--data-path", data_path,
        "--sp-model-path", sp_model_path,
        "--config-matrix", str(matrix_path),
        "--output-dir", str(output_dir),
        "--budget", str(budget),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Persistent-DDP multi-seed launcher for Exp 19+"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument(
        "--world-size", type=int, default=4,
        help="DDP world size (== nproc_per_node for this launcher).",
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.128,
        help="Base LR; default 0.128 matches Test 4b ws=4 winner.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Seeds to run per condition. Default: 4 seeds matching Exp 18.",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=["bf16", "fp8"],
        help=(
            "Precision conditions to sweep. Default: bf16+fp8 (the Test 10 "
            "matrix). Pass 'bf16' alone for a bf16-only smoke."
        ),
    )
    parser.add_argument(
        "--matrix-json", default=None,
        help=(
            "Override the auto-built matrix with a pre-written JSON file. "
            "When set, --conditions/--seeds/--world-size/--base-lr are "
            "ignored at matrix level (still used for the torchrun world "
            "size, which must match the matrix's entries)."
        ),
    )
    parser.add_argument(
        "--rdzv-port", type=int, default=29500,
        help="torchrun rendezvous port. Change if another run has it bound.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the matrix + torchrun cmd without launching.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Build the matrix -----
    if args.matrix_json:
        entries = json.loads(Path(args.matrix_json).read_text())
        if not isinstance(entries, list):
            raise ValueError(
                f"--matrix-json file must be a JSON list, got {type(entries).__name__}"
            )
        print(f"[launcher] using external matrix: {args.matrix_json} "
              f"({len(entries)} entries)", flush=True)
    else:
        entries = build_default_matrix(
            world_size=args.world_size,
            base_lr=args.base_lr,
            seeds=args.seeds,
            conditions=args.conditions,
        )
        print(
            f"[launcher] built matrix: conditions={args.conditions} "
            f"seeds={args.seeds} ws={args.world_size} lr={args.base_lr} "
            f"({len(entries)} entries)",
            flush=True,
        )

    # Pre-flight: refuse a matrix whose entries disagree with --world-size.
    # The runner uses the torchrun process-group size, not the entry field,
    # so a mismatch silently runs one regime and writes JSONs tagged with
    # another. Fail loud before any subprocess work.
    _validate_matrix_world_size(entries, args.world_size)

    # Pre-flight: refuse stale fp8 skip markers if TE is now available.
    # Stale markers from a prior TE-less run would be honored by the runner
    # and classified as benign by _check_output_integrity — silent rc=0
    # with zero fp8 results despite the user requesting fp8.
    reject_stale_skip_markers(entries, output_dir)

    # ----- fp8 pre-flight + skip markers -----
    runnable, skipped = filter_matrix_for_te(entries, output_dir)
    if skipped:
        write_skip_markers(
            skipped, output_dir,
            reason="skipped: transformer_engine unavailable on pod",
        )

    if not runnable:
        print(
            "[launcher] nothing to run (all entries filtered by TE pre-flight). "
            "Skip markers written; exiting.",
            flush=True,
        )
        return 0

    # ----- Write the matrix for the runner -----
    matrix_fd, matrix_path_str = tempfile.mkstemp(
        prefix="persistent_matrix_", suffix=".json"
    )
    os.close(matrix_fd)
    matrix_path = Path(matrix_path_str)
    matrix_path.write_text(json.dumps(runnable, indent=2, default=str))
    print(f"[launcher] wrote matrix to {matrix_path}", flush=True)

    cmd = build_torchrun_cmd(
        nproc_per_node=args.world_size,
        matrix_path=matrix_path,
        data_path=args.data_path,
        sp_model_path=args.sp_model_path,
        output_dir=output_dir,
        budget=args.budget,
        rdzv_port=args.rdzv_port,
    )

    if args.dry_run:
        print("[launcher] dry-run; would execute:")
        print("  " + " ".join(cmd))
        print(f"[launcher] first runnable entry:")
        print("  " + json.dumps(runnable[0], indent=2, default=str))
        # Leave the matrix file in place so the user can inspect it.
        print(f"[launcher] matrix left at {matrix_path} for inspection")
        return 0

    print(f"[launcher] launching: {' '.join(cmd)}", flush=True)
    try:
        ret = subprocess.call(cmd)
    finally:
        # Clean up the /tmp matrix file regardless of exit status. It's
        # already been consumed by the runner at this point.
        matrix_path.unlink(missing_ok=True)

    if ret != 0:
        print(f"[launcher] torchrun exited non-zero: {ret}", flush=True)
        return ret

    # ----- Post-run: fail closed on any real error among expected outputs. -----
    success, benign, real_errors = _check_output_integrity(output_dir, entries)
    expected = len(entries)
    print(
        f"[launcher] outputs: success={success} benign_skip={benign} "
        f"errors={len(real_errors)} total_expected={expected}",
        flush=True,
    )
    if real_errors:
        print("[launcher] real errors in outputs:", flush=True)
        for err in real_errors:
            print(f"  - {err}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
