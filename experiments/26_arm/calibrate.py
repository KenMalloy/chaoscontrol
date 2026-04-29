#!/usr/bin/env python3
"""Exp26 calibration analyzer.

Reads a shadow-mode replay-maintenance trace (NDJSON), pulls per-decision
EMAs out of the action rows, and writes a manifest of percentile-anchored
threshold counterfactuals. These values support post-hoc rule replay; they
do not feed active arms or own the commit decision.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPLAY_DECISION_ROW_TYPES = {
    "replay_preserve",
    "replay_decay",
    "replay_evict",
    "replay_refresh",
    "replay_quarantine",
    "replay_distill",
}


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile. q in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (q / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _collect_signals(trace_path: Path) -> dict[str, list[float]]:
    """Walk the trace once, collect per-decision EMAs into per-signal lists."""
    signals: dict[str, list[float]] = {
        "utility_ema": [],
        "marginal_gain_ema": [],
        "sharpness_ema": [],
        "peak_utility": [],
        "peak_sharpness": [],
        "contradiction_ema": [],
        "activation_drift": [],
        "representation_drift": [],
        "semantic_drift": [],
        "max_drift": [],
        "retrieval_mass": [],
    }
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_type = row.get("row_type", "")
            if row_type not in REPLAY_DECISION_ROW_TYPES:
                continue
            if "marginal_gain" in row:
                signals["marginal_gain_ema"].append(float(row["marginal_gain"]))
                signals["utility_ema"].append(float(row["marginal_gain"]))
            if "sharpness" in row:
                signals["sharpness_ema"].append(float(row["sharpness"]))
            if "peak_utility" in row:
                signals["peak_utility"].append(float(row["peak_utility"]))
            if "peak_sharpness" in row:
                signals["peak_sharpness"].append(float(row["peak_sharpness"]))
            if "contradiction" in row:
                signals["contradiction_ema"].append(float(row["contradiction"]))
            if "retrieval_mass" in row:
                signals["retrieval_mass"].append(float(row["retrieval_mass"]))
            ad = float(row.get("activation_drift", 0.0))
            rd = float(row.get("representation_drift", 0.0))
            sd = float(row.get("semantic_drift", 0.0))
            signals["activation_drift"].append(ad)
            signals["representation_drift"].append(rd)
            signals["semantic_drift"].append(sd)
            signals["max_drift"].append(max(ad, rd, sd))
    return signals


def _summarize(signals: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Per-signal min/median/max/mean/p25/p50/p75/p90/p95/p99."""
    out: dict[str, dict[str, float]] = {}
    for name, values in signals.items():
        if not values:
            out[name] = {
                "n": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
            continue
        out[name] = {
            "n": len(values),
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(statistics.fmean(values)),
            "p25": _percentile(values, 25.0),
            "p50": _percentile(values, 50.0),
            "p75": _percentile(values, 75.0),
            "p90": _percentile(values, 90.0),
            "p95": _percentile(values, 95.0),
            "p99": _percentile(values, 99.0),
        }
    return out


def _balanced_thresholds(summary: dict[str, dict[str, float]]) -> dict[str, float]:
    """Balanced threshold counterfactual for post-hoc rule replay."""
    return {
        "threshold": summary["utility_ema"]["p50"],
        "useful_threshold": summary["utility_ema"]["p25"],
        "drift_threshold": summary["max_drift"]["p75"],
        "repr_drift_threshold": summary["representation_drift"]["p75"],
        "quarantine_threshold": -abs(summary["contradiction_ema"]["p75"]),
        "distill_peak_threshold": summary["peak_utility"]["p90"],
        "peak_preserve_utility_threshold": summary["peak_utility"]["p75"],
        "peak_preserve_sharpness_threshold": summary["peak_sharpness"]["p75"],
        "min_age_steps": 128,
    }


def analyze(
    *,
    trace_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Read trace, summarize, write manifest. Returns manifest dict."""
    if not trace_path.exists():
        raise FileNotFoundError(f"calibration trace missing at {trace_path}")
    signals = _collect_signals(trace_path)
    n_decisions = len(signals["utility_ema"])
    if n_decisions == 0:
        raise ValueError(
            f"trace at {trace_path} has no replay-decision rows; "
            f"calibration cell must have produced shadow-policy decisions"
        )
    summary = _summarize(signals)
    manifest: dict[str, Any] = {
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "source_trace": str(trace_path),
        "n_decisions_observed": n_decisions,
        "signal_summary": summary,
        "thresholds_balanced": _balanced_thresholds(summary),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        required=True,
        type=Path,
        help="Path to shadow-mode replay_eviction trace NDJSON",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Output path for the calibration manifest JSON",
    )
    args = parser.parse_args(argv)
    manifest = analyze(
        trace_path=args.trace,
        manifest_path=args.manifest,
    )
    print(
        f"[calibrate] wrote {args.manifest} from "
        f"{manifest['n_decisions_observed']} decisions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
