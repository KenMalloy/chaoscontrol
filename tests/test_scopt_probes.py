"""Tests for the Tier 0 probe analyzer.

Pure-Python (no torch); covers gate pass/fail/skip semantics for each of
the design's four Tier 0 probes against synthetic ``scarcity_trace``
snapshots.
"""
from __future__ import annotations

from chaoscontrol.scopt_probes import evaluate_tier0_gates, summarize_gates


def _trace(
    step: int,
    *,
    enabled: bool = True,
    out_median: float | None = None,
    cos_median: float | None = None,
    fraction_positive: float | None = None,
) -> dict:
    """Build a minimal trace dict with only the fields the probes read."""
    trace: dict = {"step": step, "scarcity_enabled": enabled}
    if out_median is not None:
        trace["out_scarcity"] = {"min": out_median, "median": out_median, "max": out_median, "count": 1}
    if cos_median is not None:
        trace["cos_rare_common"] = {"min": cos_median, "median": cos_median, "max": cos_median, "count": 1}
    if fraction_positive is not None:
        trace["pressure_stats"] = {
            "min": 0.0,
            "median": 0.0,
            "p95": 1.0,
            "max": 1.0,
            "fraction_positive": fraction_positive,
        }
    return trace


def test_all_probes_pass_on_healthy_traces() -> None:
    traces = [
        _trace(250, out_median=1.05, cos_median=0.3, fraction_positive=0.10),
        _trace(500, out_median=1.15, cos_median=0.5, fraction_positive=0.15),
        _trace(750, out_median=1.25, cos_median=0.4, fraction_positive=0.12),
    ]
    results = evaluate_tier0_gates(traces)
    assert results["0.1_signal_distribution"]["status"] == "pass"
    assert results["0.3_rare_common_alignment"]["status"] == "pass"
    assert results["0.4_pressure_sparsity"]["status"] == "pass"
    # 0.2 always skips for now.
    assert results["0.2_ns_convergence"]["status"] == "skip"


def test_warmup_traces_filtered_out() -> None:
    """Traces before ``min_late_steps`` or with ``scarcity_enabled=False``
    are dropped before gating, matching the design's warm-start policy.
    """
    traces = [
        _trace(50, out_median=1.00, cos_median=0.0, fraction_positive=0.0),  # pre-warmup
        _trace(150, enabled=False, out_median=1.00, cos_median=0.0, fraction_positive=0.0),  # warmup
        _trace(300, out_median=1.20, cos_median=0.4, fraction_positive=0.10),  # late
        _trace(500, out_median=1.30, cos_median=0.5, fraction_positive=0.12),  # late
    ]
    results = evaluate_tier0_gates(traces)
    assert results["0.1_signal_distribution"]["status"] == "pass"
    assert results["0.3_rare_common_alignment"]["status"] == "pass"
    assert results["0.4_pressure_sparsity"]["status"] == "pass"


def test_signal_distribution_fails_when_flat() -> None:
    """out_scarcity medians that barely move across training indicate the
    channel-pressure signal is degenerate (all channels look equally rare).
    """
    traces = [
        _trace(300, out_median=1.001, cos_median=0.4, fraction_positive=0.10),
        _trace(500, out_median=1.002, cos_median=0.4, fraction_positive=0.10),
        _trace(700, out_median=1.001, cos_median=0.4, fraction_positive=0.10),
    ]
    results = evaluate_tier0_gates(traces)
    probe = results["0.1_signal_distribution"]
    assert probe["status"] == "fail"
    assert "degenerate" in probe["reason"]


def test_alignment_fails_when_too_parallel() -> None:
    """cos(rare, common) saturated near ±1 means r_orth is tiny — mechanism inert."""
    traces = [
        _trace(300, out_median=1.20, cos_median=0.95, fraction_positive=0.10),
        _trace(500, out_median=1.25, cos_median=0.98, fraction_positive=0.10),
    ]
    results = evaluate_tier0_gates(traces)
    probe = results["0.3_rare_common_alignment"]
    assert probe["status"] == "fail"
    assert "inert" in probe["reason"]


def test_alignment_fails_when_too_orthogonal() -> None:
    """cos(rare, common) near 0 means r_orth dominates — destabilization risk."""
    traces = [
        _trace(300, out_median=1.20, cos_median=0.02, fraction_positive=0.10),
        _trace(500, out_median=1.25, cos_median=-0.03, fraction_positive=0.10),
    ]
    results = evaluate_tier0_gates(traces)
    probe = results["0.3_rare_common_alignment"]
    assert probe["status"] == "fail"
    assert "destabilization" in probe["reason"]


def test_sparsity_fails_below_band() -> None:
    """fraction_positive consistently < 5% means the baseline is too strong."""
    traces = [
        _trace(300, out_median=1.20, cos_median=0.4, fraction_positive=0.01),
        _trace(500, out_median=1.25, cos_median=0.4, fraction_positive=0.02),
    ]
    results = evaluate_tier0_gates(traces)
    probe = results["0.4_pressure_sparsity"]
    assert probe["status"] == "fail"
    assert "too sparse" in probe["reason"]


def test_sparsity_fails_above_band() -> None:
    """fraction_positive consistently > 25% means the baseline is too weak
    and pressure collapses to a re-weighted CE.
    """
    traces = [
        _trace(300, out_median=1.20, cos_median=0.4, fraction_positive=0.55),
        _trace(500, out_median=1.25, cos_median=0.4, fraction_positive=0.60),
    ]
    results = evaluate_tier0_gates(traces)
    probe = results["0.4_pressure_sparsity"]
    assert probe["status"] == "fail"
    assert "too dense" in probe["reason"]


def test_missing_telemetry_skips_not_fails() -> None:
    """Telemetry absent entirely should skip the gate, not fail it —
    otherwise a capture bug reads as a thesis bug.
    """
    traces = [
        _trace(300),  # no out/cos/fraction fields at all
        _trace(500),
    ]
    results = evaluate_tier0_gates(traces)
    assert results["0.1_signal_distribution"]["status"] == "skip"
    assert results["0.3_rare_common_alignment"]["status"] == "skip"
    assert results["0.4_pressure_sparsity"]["status"] == "skip"


def test_summarize_gates_includes_all_probe_names() -> None:
    traces = [
        _trace(300, out_median=1.20, cos_median=0.4, fraction_positive=0.10),
        _trace(500, out_median=1.25, cos_median=0.5, fraction_positive=0.12),
    ]
    results = evaluate_tier0_gates(traces)
    summary = summarize_gates(results)
    for probe_name in ("0.1", "0.2", "0.3", "0.4"):
        assert probe_name in summary


def test_custom_thresholds_propagate() -> None:
    """Callers should be able to tighten/loosen the bands without edits."""
    traces = [
        _trace(300, out_median=1.20, cos_median=0.4, fraction_positive=0.30),
        _trace(500, out_median=1.25, cos_median=0.5, fraction_positive=0.32),
    ]
    # Default band rejects 0.30+; loosened band accepts it.
    default = evaluate_tier0_gates(traces)
    assert default["0.4_pressure_sparsity"]["status"] == "fail"

    loose = evaluate_tier0_gates(traces, sparsity_target=(0.05, 0.40))
    assert loose["0.4_pressure_sparsity"]["status"] == "pass"
