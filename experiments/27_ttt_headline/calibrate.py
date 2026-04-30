#!/usr/bin/env python3
"""Exp27 calc_type calibration analyzer.

Stub for now — emits a manifest of sensible default hyperparameters per
calc_type without actually probing. The real probe routine times each
calc_type on a frozen winning trunk, fits N/K/R/L/steps inside the
600s eval budget minus the baseline forward pass, and writes the
per-calc_type hyperparameter set picked from the probe sweep. That
follow-up lands once exp26 settles on a winning trunk.

Until then, this stub keeps the orchestrator end-to-end runnable: the
defaults below are the same values exp22's design notes carried.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_STUB_NOTE = (
    "stub manifest — replace with measured probe values when a winning "
    "trunk is fixed"
)


def _default_calc_type_hyperparams() -> dict[str, dict[str, Any]]:
    """Sensible defaults that fit the 600s eval budget on the locked trunk."""
    return {
        "score_only_reset": {},
        "carry_state": {"decay": 1.0},
        "dreamworld_eval": {
            "K": 8,
            "L": 64,
            "lr": 0.001,
            "steps": 1,
            "per_doc_reset": True,
            "dream_target_mode": "argmax",
            "dream_temperature": 1.0,
            "prefix_len": 16,
        },
    }


def _build_stub_manifest_dict(*, trace_path: Path | None = None) -> dict[str, Any]:
    """Pure helper. Returns the stub manifest dict without writing to disk.

    Kept pure so the orchestrator can preview the manifest in dry-run mode
    without violating the no-side-effects contract.
    """
    return {
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "source_trace": str(trace_path) if trace_path is not None else "stub",
        "calc_type_hyperparams": _default_calc_type_hyperparams(),
        "note": _STUB_NOTE,
    }


def analyze(
    *,
    trace_path: Path | None = None,
    manifest_path: Path,
) -> dict[str, Any]:
    """Stub calibration — writes a manifest of default hyperparams.

    A real probe routine that times each calc_type on a winning trunk
    and picks N/K/R/L to fit 600s is a follow-up. For now we emit
    sensible defaults so the orchestrator can proceed end-to-end.

    ``trace_path`` is accepted for forward compatibility (the future
    probe writes a trace and feeds it back in here); the stub records it
    in ``source_trace`` and otherwise ignores it.
    """
    manifest = _build_stub_manifest_dict(trace_path=trace_path)
    out_path = Path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Output path for the calc_type calibration manifest JSON",
    )
    parser.add_argument(
        "--trace",
        required=False,
        type=Path,
        default=None,
        help=(
            "Forward-compat path to a probe trace; ignored by the stub but "
            "recorded in source_trace when supplied"
        ),
    )
    args = parser.parse_args(argv)
    manifest = analyze(trace_path=args.trace, manifest_path=args.manifest)
    n = len(manifest["calc_type_hyperparams"])
    print(
        f"[calibrate] wrote {args.manifest} (stub) with {n} calc_type "
        f"hyperparam sets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
