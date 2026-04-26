"""Analyze the Phase 3 falsifier matrix `episodic_dw_curation_v1`.

Reads the 12-cell (4 arms × 3 seeds) result JSONs and applies the
decision gate from the plan:

  - Pass + mechanism-specific: Arm B beats both A AND B' on rare-bucket
    val CE by ≥ 0.005 bpb. Memory-persistence + similarity-recall
    thesis is supported. Phase 4 unlocks.
  - Pass + mechanism-agnostic: Arm B ties B' (within stderr) but both
    beat A. Thesis collapses to "any rare-grad-aligned curation works."
    Phase 4 unlocks but Phase 5.3 (refresh) likely never pays.
  - Mixed/null: Arm B ties Arm A. Curation didn't help.
  - Regression: Treatment worse than control.

Pre-committed σ-escalation (Decision 0.5): if σ(rare-bucket δ_bpb)
across 3 seeds on Arm B > 0.008 bpb, run 3 additional seeds on Arms A,
B, AND B' before declaring pass/fail. This script reports the σ and
flags whether escalation is required; it does NOT auto-launch the
extra cells.

Usage::

    python analyze_episodic_dw_curation_v1.py [results_dir]

If `results_dir` is omitted, defaults to the most recent
`phase3_episodic_dw_curation_v1_*` directory under
`experiments/24_training_time_bundle/`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


_ARM_NAMES = (
    "arm_a_uncurated",
    "arm_b_cosine_utility",
    "arm_bp_pressure_only",
    "arm_c_no_dw",
)
_ARM_LABELS = {
    "arm_a_uncurated": "A — uncurated DW",
    "arm_b_cosine_utility": "B — curated (cosine × utility)",
    "arm_bp_pressure_only": "B' — curated (pressure-only)",
    "arm_c_no_dw": "C — no DW reference",
}
_ARM_DELTA_MIN = 0.005  # bpb improvement threshold for "pass"
_SIGMA_ESCALATION_BPB = 0.008  # σ threshold for the escalation rule
_RARE_BUCKET_INDEX = 0  # smallest token-frequency bucket


def _load_cells(results_dir: Path) -> list[dict[str, Any]]:
    cells = []
    for path in sorted(results_dir.glob("exp24_phase3_episodic_dw_curation_v1_*.json")):
        with path.open() as f:
            cells.append(json.load(f))
    if not cells:
        raise SystemExit(f"no cells found under {results_dir}")
    return cells


def _arm_of(cell: dict[str, Any]) -> str:
    name = cell["name"]
    for arm in _ARM_NAMES:
        if arm in name:
            return arm
    raise ValueError(f"could not identify arm from name: {name!r}")


def _val_per_bucket_bpb(cell: dict[str, Any]) -> list[float]:
    """The per-bucket val BPB list. Schema produced by the runner's
    per-bucket val CE diagnostics commit `b6765fb`. Falls back to the
    full-val summary path if the per-bucket field isn't on the top
    level."""
    if "val_per_bucket_bpb" in cell:
        return [float(x) for x in cell["val_per_bucket_bpb"]]
    if "val_summary_path" in cell:
        with Path(cell["val_summary_path"]).open() as f:
            summary = json.load(f)
        return [float(x) for x in summary.get("val_per_bucket_bpb", [])]
    raise SystemExit(
        f"cell {cell.get('name')!r} has no per-bucket val BPB — "
        "did the runner emit the diagnostic schema?"
    )


def _mean_stderr(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (float(values[0]) if values else float("nan"), float("nan"))
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def _summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in _ARM_NAMES}
    for cell in cells:
        by_arm[_arm_of(cell)].append(cell)
    rare_bucket: dict[str, list[float]] = {}
    overall_bpb: dict[str, list[float]] = {}
    for arm, arm_cells in by_arm.items():
        rare_bucket[arm] = [
            _val_per_bucket_bpb(c)[_RARE_BUCKET_INDEX] for c in arm_cells
        ]
        overall_bpb[arm] = [float(c["val_bpb"]) for c in arm_cells]
    return {
        "by_arm_count": {arm: len(by_arm[arm]) for arm in _ARM_NAMES},
        "rare_bucket": rare_bucket,
        "overall_bpb": overall_bpb,
    }


def _decision_gate(summary: dict[str, Any]) -> dict[str, Any]:
    rare = summary["rare_bucket"]
    a_mean, _ = _mean_stderr(rare["arm_a_uncurated"])
    b_values = rare["arm_b_cosine_utility"]
    b_mean, b_se = _mean_stderr(b_values)
    bp_mean, bp_se = _mean_stderr(rare["arm_bp_pressure_only"])
    c_mean, _ = _mean_stderr(rare["arm_c_no_dw"])

    delta_b_vs_a = b_mean - a_mean        # negative = B better than A
    delta_b_vs_bp = b_mean - bp_mean      # negative = B better than B'
    sigma_b = _std(b_values)
    needs_escalation = (
        not math.isnan(sigma_b)
        and sigma_b > _SIGMA_ESCALATION_BPB
    )

    # Stderr-band overlap heuristic for "tie": |Δ| < 2 × pooled stderr
    pooled_se_b_bp = math.sqrt((b_se or 0.0) ** 2 + (bp_se or 0.0) ** 2)
    b_ties_bp = abs(delta_b_vs_bp) < 2 * pooled_se_b_bp

    # Decision
    if delta_b_vs_a >= 0:  # B not better than A on rare bucket (positive Δ = worse)
        if delta_b_vs_a > 2 * (b_se or 0.0):
            verdict = "regression"
        else:
            verdict = "mixed_null"
    elif abs(delta_b_vs_a) < _ARM_DELTA_MIN:
        verdict = "mixed_null"
    elif b_ties_bp:
        # B beats A but ties B' → mechanism-agnostic
        verdict = "pass_mechanism_agnostic"
    else:
        # B beats A AND beats B' → mechanism-specific
        verdict = "pass_mechanism_specific"

    return {
        "verdict": verdict,
        "delta_b_vs_a": delta_b_vs_a,
        "delta_b_vs_bp": delta_b_vs_bp,
        "sigma_b": sigma_b,
        "needs_escalation": needs_escalation,
        "rare_means": {
            "A": a_mean, "B": b_mean, "Bp": bp_mean, "C": c_mean,
        },
    }


def _format_table(summary: dict[str, Any], gate: dict[str, Any]) -> str:
    lines = []
    lines.append(
        f"{'arm':<32}  {'n':>3}  {'rare_mean':>10}  {'rare_stderr':>11}  {'overall_mean':>12}"
    )
    lines.append("-" * 76)
    for arm in _ARM_NAMES:
        n = summary["by_arm_count"][arm]
        rare_mean, rare_se = _mean_stderr(summary["rare_bucket"][arm])
        overall_mean, _ = _mean_stderr(summary["overall_bpb"][arm])
        label = _ARM_LABELS[arm]
        lines.append(
            f"{label:<32}  {n:>3}  {rare_mean:>10.4f}  {rare_se:>11.4f}  {overall_mean:>12.4f}"
        )
    lines.append("")
    lines.append(f"Δ rare-bucket (B − A): {gate['delta_b_vs_a']:+.4f} bpb (negative = B better)")
    lines.append(f"Δ rare-bucket (B − B'): {gate['delta_b_vs_bp']:+.4f} bpb (negative = mechanism-specific)")
    lines.append(f"σ(rare-bucket B): {gate['sigma_b']:.4f} bpb"
                 + (f"  ⚠️ > {_SIGMA_ESCALATION_BPB} — escalate to 6 seeds" if gate["needs_escalation"] else ""))
    lines.append("")
    verdict_label = {
        "pass_mechanism_specific":
            "✅ PASS — mechanism-specific (memory persistence + similarity recall). Phase 4 unlocks.",
        "pass_mechanism_agnostic":
            "🟡 PASS — mechanism-agnostic (rare-grad-aligned curation, not memory persistence). "
            "Phase 4 unlocks; Phase 5.3 (refresh) likely doesn't pay.",
        "mixed_null":
            "🟡 MIXED/NULL — curation didn't beat uncurated DW. Diagnose via per-replay logs.",
        "regression":
            "❌ REGRESSION — curation hurt vs uncurated. Disable, root-cause.",
    }
    lines.append(f"Verdict: {verdict_label[gate['verdict']]}")
    return "\n".join(lines)


def _default_results_dir() -> Path:
    base = (
        Path(__file__).resolve().parent
        / "results"
    )
    if not base.exists():
        base = Path(__file__).resolve().parent
    candidates = sorted(base.glob("phase3_episodic_dw_curation_v1_*"))
    if not candidates:
        raise SystemExit(
            f"no phase3_episodic_dw_curation_v1_* directory under {base}"
        )
    return candidates[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir", nargs="?", default=None,
        help="Directory containing exp24_phase3_episodic_dw_curation_v1_*.json cells",
    )
    args = parser.parse_args(argv)
    results_dir = Path(args.results_dir) if args.results_dir else _default_results_dir()
    cells = _load_cells(results_dir)
    summary = _summarize(cells)
    gate = _decision_gate(summary)
    print(f"Results: {results_dir}")
    print(f"Cells: {len(cells)} ({sum(summary['by_arm_count'].values())} expected)")
    print()
    print(_format_table(summary, gate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
