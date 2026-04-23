"""Tier 0 sanity probes for the scarcity-aware optimizer.

The optimizer emits a snapshot dict via ``ScarcityAwareOptimizer.scarcity_trace()``
every N steps. This module consumes a chronologically ordered list of those
snapshots captured across a short training run and evaluates the design's
pre-quality gates.

Design reference: ``docs/plans/2026-04-22-scarcity-optimizer-design.md``,
section "Tier 0 — Pre-run sanity probes" (lines 219-230). The four probes:

* **0.1 Signal distribution** — per-channel scarcity signal varies across the
  run. Flat signal means pressure isn't differentiating channels.
* **0.2 NS convergence** — Newton-Schulz residual on scarcity-scaled inputs
  stays comparable to the baseline. Not currently instrumented; stubbed.
* **0.3 Rare/common alignment** — median |cos(rare, common)| falls in a
  safe band. Too-parallel means the mechanism is inert; too-orthogonal
  means it will destabilize training.
* **0.4 Pressure sparsity** — ``fraction_positive`` in the calibrated band.
  Outside means the baseline is mis-calibrated.

Gates are run against post-warmup traces only.
"""
from __future__ import annotations

from typing import Any


def evaluate_tier0_gates(
    traces: list[dict[str, Any]],
    *,
    min_late_steps: int = 200,
    sparsity_target: tuple[float, float] = (0.05, 0.25),
    alignment_safe_range: tuple[float, float] = (0.1, 0.9),
    distribution_degenerate_ratio: float = 1.05,
) -> dict[str, dict[str, Any]]:
    """Evaluate the four Tier 0 probes over a sequence of scarcity traces.

    Args:
        traces: list of dicts returned by
            :meth:`ScarcityAwareOptimizer.scarcity_trace`, captured at
            regular step intervals. Must be chronologically ordered.
        min_late_steps: minimum ``step`` value for a trace to count as
            "late" training — gates only inspect late traces to let
            warm-up finish.
        sparsity_target: ``(low, high)`` band for ``fraction_positive``.
        alignment_safe_range: ``(low, high)`` band for mean absolute
            ``cos(rare, common)`` median.
        distribution_degenerate_ratio: required ratio of max/min
            ``out_scarcity`` median across traces to count as non-flat.

    Returns:
        dict keyed by probe name. Each value is
        ``{"status": "pass"|"fail"|"skip", "metric": ..., "reason": str|None}``.
        Probe 0.2 is always ``"skip"`` until NS residual is added to the
        optimizer's telemetry.
    """
    late = [
        t for t in traces
        if int(t.get("step", 0)) >= min_late_steps
        and t.get("scarcity_enabled", False)
    ]

    return {
        "0.1_signal_distribution": _probe_signal_distribution(
            late, ratio_threshold=distribution_degenerate_ratio,
        ),
        "0.2_ns_convergence": {
            "status": "skip",
            "metric": None,
            "reason": "NS residual not currently captured in scarcity_trace",
        },
        "0.3_rare_common_alignment": _probe_alignment(
            late, safe_range=alignment_safe_range,
        ),
        "0.4_pressure_sparsity": _probe_sparsity(
            late, target=sparsity_target,
        ),
    }


def _collect_medians(
    traces: list[dict[str, Any]],
    key: str,
) -> list[float]:
    medians: list[float] = []
    for trace in traces:
        block = trace.get(key)
        if isinstance(block, dict) and "median" in block:
            medians.append(float(block["median"]))
    return medians


def _probe_signal_distribution(
    late: list[dict[str, Any]],
    *,
    ratio_threshold: float,
) -> dict[str, Any]:
    medians = _collect_medians(late, "out_scarcity")
    if len(medians) < 2:
        return {
            "status": "skip",
            "metric": None,
            "reason": "need >=2 post-warmup traces with out_scarcity telemetry",
        }

    lo = min(medians)
    hi = max(medians)
    ratio = hi / max(lo, 1e-9)

    if ratio < ratio_threshold:
        return {
            "status": "fail",
            "metric": {"min": lo, "max": hi, "ratio": ratio},
            "reason": (
                f"out_scarcity median range too narrow ({lo:.4f}-{hi:.4f}); "
                "channel pressure likely degenerate"
            ),
        }
    return {
        "status": "pass",
        "metric": {"min": lo, "max": hi, "ratio": ratio},
        "reason": None,
    }


def _probe_alignment(
    late: list[dict[str, Any]],
    *,
    safe_range: tuple[float, float],
) -> dict[str, Any]:
    low, high = safe_range
    medians = _collect_medians(late, "cos_rare_common")
    if not medians:
        return {
            "status": "skip",
            "metric": None,
            "reason": "no cos_rare_common telemetry in late traces",
        }

    abs_medians = [abs(m) for m in medians]
    mean_abs = sum(abs_medians) / len(abs_medians)

    if mean_abs < low:
        return {
            "status": "fail",
            "metric": {"mean_abs_cos": mean_abs},
            "reason": (
                f"|cos(rare, common)| mean = {mean_abs:.3f} below {low}; "
                "rare direction nearly orthogonal everywhere — destabilization risk"
            ),
        }
    if mean_abs > high:
        return {
            "status": "fail",
            "metric": {"mean_abs_cos": mean_abs},
            "reason": (
                f"|cos(rare, common)| mean = {mean_abs:.3f} above {high}; "
                "rare direction nearly parallel to common — mechanism inert"
            ),
        }
    return {
        "status": "pass",
        "metric": {"mean_abs_cos": mean_abs},
        "reason": None,
    }


def _probe_sparsity(
    late: list[dict[str, Any]],
    *,
    target: tuple[float, float],
) -> dict[str, Any]:
    low, high = target
    fractions: list[float] = []
    for trace in late:
        ps = trace.get("pressure_stats")
        if isinstance(ps, dict) and "fraction_positive" in ps:
            fractions.append(float(ps["fraction_positive"]))

    if not fractions:
        return {
            "status": "skip",
            "metric": None,
            "reason": "no pressure_stats.fraction_positive in late traces",
        }

    mean_fp = sum(fractions) / len(fractions)

    if mean_fp < low:
        return {
            "status": "fail",
            "metric": {"mean_fraction_positive": mean_fp},
            "reason": (
                f"mean fraction_positive = {mean_fp:.3f} below {low}; "
                "pressure too sparse — baseline too strong or calibration wrong"
            ),
        }
    if mean_fp > high:
        return {
            "status": "fail",
            "metric": {"mean_fraction_positive": mean_fp},
            "reason": (
                f"mean fraction_positive = {mean_fp:.3f} above {high}; "
                "pressure too dense — degenerates to weighted ordinary CE"
            ),
        }
    return {
        "status": "pass",
        "metric": {"mean_fraction_positive": mean_fp},
        "reason": None,
    }


def summarize_gates(results: dict[str, dict[str, Any]]) -> str:
    """Format gate results as a compact human-readable summary."""
    lines: list[str] = []
    for name, body in results.items():
        status = body["status"].upper()
        reason = body.get("reason") or ""
        lines.append(f"{name}: {status}  {reason}")
    return "\n".join(lines)


__all__ = ["evaluate_tier0_gates", "summarize_gates"]
