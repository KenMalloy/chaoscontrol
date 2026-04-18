#!/usr/bin/env python3
"""Exp 21 statistical analysis + verdict report.

Reads per-run JSON results produced by the Phase 5 orchestrators
(runner_4cell.py, runner_controls.py) and prints a verdict against
the thesis gates from the design doc:

    Primary:   Δ_SSM = bpb(C_ssm_random) - bpb(D_ssm_sgns) > 0 at p<0.01
    Secondary: Δ_SSM > Δ_Trans                              at p<0.01

Both tests are one-sided paired t-tests on seeds present in both arms.
The `alternative="greater"` direction encodes "random should be worse than
SGNS" (higher bpb = worse).

Expected results-directory layout (matches Phase 5 orchestrators):

    results/
      four_cell/
        A_transformer_random_s{seed}.json
        B_transformer_sgns_s{seed}.json
        C_ssm_random_s{seed}.json
        D_ssm_sgns_s{seed}.json
      fullcov/
        ssm_fullcov_s{seed}.json
      shuffled/
        ssm_shuffled_s{seed}.json
      zero/
        ssm_zero_s{seed}.json

The zero-init control tests whether random init itself is informative
structure: zero vs random paired t (H1: zero bpb > random bpb). If
training diverges on zero init, the runner writes a NaN result with
``nonfinite.flag=True``; those seeds are dropped from the paired test
by ``load_cell_bpbs``'s finite-value filter.

Each per-run JSON is produced by runner_exp21.py and contains at least
``{"eval": {"bpb": <float>}}``.

No scipy dependency — reuses the pure-Python paired_ttest implementation
from ``experiments/09_revised_architecture/stats.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))

from stats import paired_ttest  # noqa: E402


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def paired_t_one_sided(
    x: list[float],
    y: list[float],
    alternative: str = "greater",
) -> float:
    """Paired one-sided t-test p-value.

    ``alternative="greater"``: H1 mean(x) > mean(y).
    ``alternative="less"``:    H1 mean(x) < mean(y).

    Seeds must be paired in the same order across ``x`` and ``y``.
    Reduces the existing two-tailed ``paired_ttest`` by sign of the
    t-statistic:

        greater:  t > 0 → p_two/2;  t <= 0 → 1 - p_two/2
        less:     t < 0 → p_two/2;  t >= 0 → 1 - p_two/2

    The two-tailed implementation matches scipy.stats.ttest_rel to
    floating-point precision (see stats.py docstring), so this
    one-sided wrapper matches scipy.stats.ttest_rel(alternative=...)
    likewise.
    """
    if alternative not in ("greater", "less"):
        raise ValueError(
            f"alternative must be 'greater' or 'less', got {alternative!r}"
        )
    if len(x) != len(y):
        raise ValueError(
            f"paired inputs must have equal length; got len(x)={len(x)}, len(y)={len(y)}"
        )
    if len(x) < 2:
        raise ValueError(
            f"need at least 2 paired observations for a paired t-test, got n={len(x)}"
        )

    t_stat, p_two = paired_ttest(list(x), list(y))
    if math.isnan(t_stat) or math.isnan(p_two):
        # stats.paired_ttest returns (nan, nan) when diffs have zero variance
        # and zero mean (fully null sample). Two-sided-null → one-sided p = 0.5.
        return 0.5

    if alternative == "greater":
        return p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
    # alternative == "less"
    return p_two / 2.0 if t_stat < 0 else 1.0 - p_two / 2.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


_RUN_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _run_pattern(cell_name: str) -> re.Pattern[str]:
    """Compile (once) the ``{cell_name}_s{seed}.json`` matcher."""
    if cell_name not in _RUN_PATTERN_CACHE:
        _RUN_PATTERN_CACHE[cell_name] = re.compile(
            rf"^{re.escape(cell_name)}_s(\d+)\.json$"
        )
    return _RUN_PATTERN_CACHE[cell_name]


def load_cell_bpbs(results_dir: Path, cell_name: str) -> dict[int, float]:
    """Load ``{seed: bpb}`` for ``cell_name`` from ``{results_dir}/``.

    Silently returns ``{}`` when the directory is missing — partial
    experimental state is expected (e.g., controls not yet run).
    Non-finite bpb values are dropped with a warning printed to stdout
    so the caller can see which runs were excluded.
    """
    out: dict[int, float] = {}
    if not results_dir.exists():
        return out
    pattern = _run_pattern(cell_name)
    for path in sorted(results_dir.iterdir()):
        m = pattern.match(path.name)
        if not m:
            continue
        seed = int(m.group(1))
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  warn: could not read {path.name}: {exc}")
            continue
        eval_block = data.get("eval") or {}
        bpb_raw = eval_block.get("bpb")
        if bpb_raw is None:
            print(f"  warn: {path.name} has no eval.bpb; skipping")
            continue
        bpb = float(bpb_raw)
        if not math.isfinite(bpb):
            print(f"  warn: {path.name} bpb={bpb} is not finite; skipping")
            continue
        out[seed] = bpb
    return out


def pair_seeds(
    x_by_seed: dict[int, float],
    y_by_seed: dict[int, float],
) -> tuple[list[float], list[float]]:
    """Return two parallel lists for seeds present in both arms, sorted by seed."""
    shared = sorted(set(x_by_seed) & set(y_by_seed))
    return (
        [x_by_seed[s] for s in shared],
        [y_by_seed[s] for s in shared],
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    mean = sum(values) / n
    if n < 2:
        std = 0.0
    else:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
    return {"n": n, "mean": mean, "std": std}


def build_report(results_dir: Path) -> dict[str, Any]:
    """Build the Exp 21 verdict report dict from a results directory.

    Read four_cell/ (A, B, C, D), fullcov/ (ssm_fullcov), shuffled/
    (ssm_shuffled). Returns a structured dict with per-cell stats,
    paired p-values, and the thesis verdict booleans.

    Partial runs (missing cells, missing seeds) are handled by reporting
    n and leaving p-values as ``None`` when there are fewer than 2
    paired observations. Callers decide how to surface that.
    """
    four_cell_dir = results_dir / "four_cell"
    fullcov_dir = results_dir / "fullcov"
    shuffled_dir = results_dir / "shuffled"
    zero_dir = results_dir / "zero"

    A_raw = load_cell_bpbs(four_cell_dir, "A_transformer_random")
    B_raw = load_cell_bpbs(four_cell_dir, "B_transformer_sgns")
    C_raw = load_cell_bpbs(four_cell_dir, "C_ssm_random")
    D_raw = load_cell_bpbs(four_cell_dir, "D_ssm_sgns")

    # Use one common seed set across ALL four cells for BOTH tests so the
    # reported Δ means and the p-values are computed on the same data.
    # If any cell drops a seed (pod flake, non-finite eval), that seed is
    # dropped from both primary and secondary tests. The design doc targets
    # 5 matched seeds; internal consistency matters more than squeezing out
    # an extra C∩D pair when A or B lost a seed.
    common_seeds = sorted(
        set(A_raw) & set(B_raw) & set(C_raw) & set(D_raw)
    )
    A_c = [A_raw[s] for s in common_seeds]
    B_c = [B_raw[s] for s in common_seeds]
    C_c = [C_raw[s] for s in common_seeds]
    D_c = [D_raw[s] for s in common_seeds]
    delta_ssm = [c - d for c, d in zip(C_c, D_c)]
    delta_trans = [a - b for a, b in zip(A_c, B_c)]

    def _p_paired_or_none(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) < 2:
            return None
        return paired_t_one_sided(xs, ys, alternative="greater")

    p_primary = _p_paired_or_none(C_c, D_c)
    p_secondary = _p_paired_or_none(delta_ssm, delta_trans)

    # Controls (optional; may be absent in partial state).
    fullcov_raw = load_cell_bpbs(fullcov_dir, "ssm_fullcov")
    shuffled_raw = load_cell_bpbs(shuffled_dir, "ssm_shuffled")

    # Paired test on fullcov control: does the meanstd arm (D) beat full-cov
    # matched init at matched seeds? This is the strongest null — same
    # covariance geometry, different direction — so a delta here is the
    # cleanest evidence of semantic directionality beyond distributional parity.
    fullcov_common = sorted(set(D_raw) & set(fullcov_raw))
    D_fc = [D_raw[s] for s in fullcov_common]
    F_fc = [fullcov_raw[s] for s in fullcov_common]
    p_fullcov_vs_meanstd = (
        paired_t_one_sided(F_fc, D_fc, alternative="greater")
        if len(fullcov_common) >= 2
        else None
    )

    shuffled_common = sorted(set(D_raw) & set(shuffled_raw))
    D_sh = [D_raw[s] for s in shuffled_common]
    S_sh = [shuffled_raw[s] for s in shuffled_common]
    p_shuffled_vs_meanstd = (
        paired_t_one_sided(S_sh, D_sh, alternative="greater")
        if len(shuffled_common) >= 2
        else None
    )

    # Zero-init floor vs random init (H1: zero bpb > random bpb).
    # A significant delta here means random init is itself informative
    # structure; insignificant means random is near-zero noise and any
    # non-noise init (meanstd, shuffled, fullcov) is doing real work.
    # load_cell_bpbs drops non-finite seeds automatically, so divergent
    # zero runs simply shrink the paired N rather than poisoning the t-stat.
    zero_raw = load_cell_bpbs(zero_dir, "ssm_zero")
    zero_common = sorted(set(C_raw) & set(zero_raw))
    C_zc = [C_raw[s] for s in zero_common]
    Z_zc = [zero_raw[s] for s in zero_common]
    p_zero_vs_random = (
        paired_t_one_sided(Z_zc, C_zc, alternative="greater")
        if len(zero_common) >= 2
        else None
    )

    controls_complete = (
        len(fullcov_common) >= 2
        and len(shuffled_common) >= 2
    )
    controls_support_semantics = (
        p_fullcov_vs_meanstd is not None
        and p_fullcov_vs_meanstd < 0.01
        and p_shuffled_vs_meanstd is not None
        and p_shuffled_vs_meanstd < 0.01
    )

    thesis_validating = (
        p_primary is not None
        and p_secondary is not None
        and p_primary < 0.01
        and p_secondary < 0.01
        and controls_complete
        and controls_support_semantics
    )
    thesis_weak = (
        p_primary is not None
        and p_primary < 0.01
        and not thesis_validating
    )

    report: dict[str, Any] = {
        "cells": {
            "A_transformer_random": _stats(list(A_raw.values())),
            "B_transformer_sgns": _stats(list(B_raw.values())),
            "C_ssm_random": _stats(list(C_raw.values())),
            "D_ssm_sgns": _stats(list(D_raw.values())),
        },
        "n_common_seeds": len(common_seeds),
        "n_ssm_pairs": len(common_seeds),
        "n_trans_pairs": len(common_seeds),
        "n_common_seeds_for_secondary": len(common_seeds),
        "delta_ssm_mean": (sum(delta_ssm) / len(delta_ssm)) if delta_ssm else float("nan"),
        "delta_trans_mean": (sum(delta_trans) / len(delta_trans)) if delta_trans else float("nan"),
        "p_primary": p_primary,
        "p_secondary": p_secondary,
        "thesis_validating": thesis_validating,
        "thesis_weak": thesis_weak,
        "fullcov": _stats(list(fullcov_raw.values())),
        "shuffled": _stats(list(shuffled_raw.values())),
        "zero": _stats(list(zero_raw.values())),
        "p_fullcov_vs_meanstd": p_fullcov_vs_meanstd,
        "n_fullcov_pairs": len(fullcov_common),
        "p_shuffled_vs_meanstd": p_shuffled_vs_meanstd,
        "n_shuffled_pairs": len(shuffled_common),
        "p_zero_vs_random": p_zero_vs_random,
        "n_zero_pairs": len(zero_common),
        "controls_complete": controls_complete,
        "controls_support_semantics": controls_support_semantics,
    }
    return report


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a (insufficient data)"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _fmt_cell(label: str, stats: dict[str, float | int]) -> str:
    n = int(stats["n"])
    if n == 0:
        return f"  {label:<28} n=0  (no runs found)"
    return (
        f"  {label:<28} n={n}  mean={float(stats['mean']):.4f}  "
        f"std={float(stats['std']):.4f}"
    )


def print_report(report: dict[str, Any]) -> None:
    print("== Exp 21 results ==")
    print(_fmt_cell("A transformer × random", report["cells"]["A_transformer_random"]))
    print(_fmt_cell("B transformer × SGNS  ", report["cells"]["B_transformer_sgns"]))
    print(_fmt_cell("C SSM × random        ", report["cells"]["C_ssm_random"]))
    print(_fmt_cell("D SSM × SGNS          ", report["cells"]["D_ssm_sgns"]))
    print()
    print(
        f"  Δ_SSM   = C - D (mean over {report['n_common_seeds_for_secondary']} common seeds): "
        f"{report['delta_ssm_mean']:+.4f}"
    )
    print(
        f"  Δ_Trans = A - B (mean over {report['n_common_seeds_for_secondary']} common seeds): "
        f"{report['delta_trans_mean']:+.4f}"
    )
    print()
    print(f"  Primary   (Δ_SSM > 0)          p = {_fmt_p(report['p_primary'])}")
    print(f"  Secondary (Δ_SSM > Δ_Trans)    p = {_fmt_p(report['p_secondary'])}")
    print()
    print(f"  thesis-validating: {report['thesis_validating']}")
    print(f"  thesis-weak:       {report['thesis_weak']}")

    if report["fullcov"]["n"] or report["shuffled"]["n"] or report["zero"]["n"]:
        print("\n== Controls ==")
        if report["fullcov"]["n"]:
            print(_fmt_cell("SSM × SGNS (full-cov)", report["fullcov"]))
            print(
                f"  full-cov vs meanstd (paired on {report['n_fullcov_pairs']} seeds, "
                f"H1: full-cov bpb > meanstd bpb): p = {_fmt_p(report['p_fullcov_vs_meanstd'])}"
            )
        if report["shuffled"]["n"]:
            print(_fmt_cell("SSM × SGNS (shuffled)", report["shuffled"]))
            c_mean = report["cells"]["C_ssm_random"]["mean"]
            print(
                f"  (thesis predicts shuffled bpb ≈ C_ssm_random mean = {c_mean:.4f})"
            )
            print(
                f"  shuffled vs meanstd (paired on {report['n_shuffled_pairs']} seeds, "
                f"H1: shuffled bpb > meanstd bpb): p = {_fmt_p(report['p_shuffled_vs_meanstd'])}"
            )
        if report["zero"]["n"]:
            print(_fmt_cell("SSM × zero-init floor", report["zero"]))
            print(
                f"  zero vs random (paired on {report['n_zero_pairs']} seeds, "
                f"H1: zero bpb > random bpb): p = {_fmt_p(report['p_zero_vs_random'])}"
            )
        print(f"  controls-complete:         {report['controls_complete']}")
        print(f"  controls-support-semantics:{report['controls_support_semantics']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exp 21 statistical analysis + verdict report"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO / "experiments" / "21_sgns_tokenizer" / "results",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the report as JSON",
    )
    args = parser.parse_args(argv)

    report = build_report(args.results_dir)
    print_report(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote machine-readable report to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
