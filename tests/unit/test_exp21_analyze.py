"""Unit tests for scripts/exp21_analyze.py.

Covers the paired-t helper (spec requirement), directional correctness,
JSON loader robustness, seed-pairing, and the end-to-end verdict gate
against synthetic 4-cell results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.exp21_analyze import (  # noqa: E402
    build_report,
    load_cell_bpbs,
    pair_seeds,
    paired_t_one_sided,
)


# ---------------------------------------------------------------------------
# paired_t_one_sided (spec + directional + edge cases)
# ---------------------------------------------------------------------------


def test_paired_t_one_sided_obvious_effect():
    """Clear effect: x consistently > y → p < 0.01."""
    x = [1.0, 1.1, 1.05, 1.08, 1.03]
    y = [0.5, 0.6, 0.55, 0.58, 0.53]
    p = paired_t_one_sided(x, y, alternative="greater")
    assert p < 0.01


def test_paired_t_one_sided_no_effect():
    """No effect: x ≈ y → p > 0.1."""
    x = [1.0, 1.1, 0.9, 1.0, 1.05]
    y = [1.01, 1.09, 0.91, 1.02, 1.04]
    p = paired_t_one_sided(x, y, alternative="greater")
    assert p > 0.1


def test_paired_t_one_sided_wrong_direction_is_high():
    """x < y but H1: x > y → p near 1 (not near 0)."""
    x = [0.5, 0.6, 0.55, 0.58, 0.53]
    y = [1.0, 1.1, 1.05, 1.08, 1.03]
    p = paired_t_one_sided(x, y, alternative="greater")
    assert p > 0.95


def test_paired_t_one_sided_less_alternative():
    """H1: x < y when x is indeed consistently lower → p < 0.01."""
    x = [0.5, 0.6, 0.55, 0.58, 0.53]
    y = [1.0, 1.1, 1.05, 1.08, 1.03]
    p = paired_t_one_sided(x, y, alternative="less")
    assert p < 0.01


def test_paired_t_one_sided_rejects_bad_alternative():
    with pytest.raises(ValueError, match="alternative"):
        paired_t_one_sided([1.0, 2.0], [0.5, 1.5], alternative="two-sided")


def test_paired_t_one_sided_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="paired"):
        paired_t_one_sided([1.0, 2.0, 3.0], [0.5, 1.5], alternative="greater")


def test_paired_t_one_sided_rejects_too_few_pairs():
    """Need >= 2 pairs; 1 pair has no degrees of freedom for variance."""
    with pytest.raises(ValueError, match="at least 2"):
        paired_t_one_sided([1.0], [0.5], alternative="greater")


def test_paired_t_one_sided_zero_variance_perfect_separation():
    """All diffs identical and positive → perfect separation, p_one_sided = 0.

    This exercises paired_ttest's ``(+inf, 0.0)`` return branch (zero variance,
    positive mean). The one-sided wrapper must convert correctly:
    greater direction sees ``t=+inf > 0`` → p_two/2 = 0.
    """
    x = [1.6, 1.6, 1.6, 1.6, 1.6]
    y = [1.4, 1.4, 1.4, 1.4, 1.4]
    p = paired_t_one_sided(x, y, alternative="greater")
    assert p == 0.0


def test_paired_t_one_sided_zero_variance_zero_mean():
    """Fully identical samples → paired_ttest returns (nan, nan); wrapper
    must map that to p = 0.5 (no evidence either way), not propagate nan."""
    x = [1.5, 1.5, 1.5, 1.5, 1.5]
    y = [1.5, 1.5, 1.5, 1.5, 1.5]
    p = paired_t_one_sided(x, y, alternative="greater")
    assert p == 0.5


# ---------------------------------------------------------------------------
# load_cell_bpbs — robust JSON loader
# ---------------------------------------------------------------------------


def _write_run_json(path: Path, bpb: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"eval": {"bpb": bpb}, "train": {"final_loss": 1.0}}))


def test_load_cell_bpbs_reads_per_run_jsons(tmp_path):
    """Loader must find {cell}_s{seed}.json files and return {seed: bpb}."""
    cell_dir = tmp_path / "four_cell"
    _write_run_json(cell_dir / "A_transformer_random_s1337.json", 1.55)
    _write_run_json(cell_dir / "A_transformer_random_s42.json", 1.54)
    # Another cell in the same dir — must not leak into A.
    _write_run_json(cell_dir / "C_ssm_random_s1337.json", 1.60)

    out = load_cell_bpbs(cell_dir, "A_transformer_random")
    assert out == {1337: 1.55, 42: 1.54}


def test_load_cell_bpbs_skips_non_finite_bpb(tmp_path, capsys):
    """NaN/inf bpb → dropped with a printed warning."""
    cell_dir = tmp_path / "four_cell"
    _write_run_json(cell_dir / "A_transformer_random_s1337.json", 1.55)
    _write_run_json(cell_dir / "A_transformer_random_s42.json", float("nan"))

    out = load_cell_bpbs(cell_dir, "A_transformer_random")
    assert out == {1337: 1.55}
    captured = capsys.readouterr().out
    assert "A_transformer_random_s42" in captured


def test_load_cell_bpbs_missing_dir_returns_empty(tmp_path):
    """A missing cell directory is not an error — just zero runs."""
    out = load_cell_bpbs(tmp_path / "does_not_exist", "A")
    assert out == {}


# ---------------------------------------------------------------------------
# pair_seeds — keep only seeds present in both arms
# ---------------------------------------------------------------------------


def test_pair_seeds_keeps_intersection_sorted():
    x = {1337: 1.5, 42: 1.6, 7: 1.7}
    y = {42: 1.4, 1337: 1.3, 123: 1.8}  # missing 7, extra 123
    xl, yl = pair_seeds(x, y)
    assert xl == [1.6, 1.5]  # sorted by seed: 42, 1337
    assert yl == [1.4, 1.3]


def test_pair_seeds_empty_intersection_returns_empty():
    xl, yl = pair_seeds({1: 1.0}, {2: 2.0})
    assert xl == [] and yl == []


# ---------------------------------------------------------------------------
# build_report — end-to-end verdict against synthetic results
# ---------------------------------------------------------------------------


def _seed_synthetic_four_cell(
    results_dir: Path,
    *,
    A: list[tuple[int, float]],
    B: list[tuple[int, float]],
    C: list[tuple[int, float]],
    D: list[tuple[int, float]],
) -> None:
    for cell_name, seeds in [
        ("A_transformer_random", A),
        ("B_transformer_sgns", B),
        ("C_ssm_random", C),
        ("D_ssm_sgns", D),
    ]:
        for seed, bpb in seeds:
            _write_run_json(
                results_dir / "four_cell" / f"{cell_name}_s{seed}.json", bpb
            )


def test_build_report_thesis_validating(tmp_path):
    """Large Δ_SSM, much smaller Δ_Trans → thesis-validating on both gates."""
    # SSM benefits a lot from SGNS (C high, D much lower).
    # Transformer benefits barely (A ≈ B).
    seeds = [1337, 42, 123, 7, 8]
    _seed_synthetic_four_cell(
        tmp_path,
        A=[(s, 1.50 + i * 0.001) for i, s in enumerate(seeds)],
        B=[(s, 1.498 + i * 0.001) for i, s in enumerate(seeds)],
        C=[(s, 1.60 + i * 0.001) for i, s in enumerate(seeds)],
        D=[(s, 1.40 + i * 0.001) for i, s in enumerate(seeds)],
    )

    report = build_report(tmp_path)
    assert report["p_primary"] < 0.01
    assert report["p_secondary"] < 0.01
    assert report["thesis_validating"] is True


def test_build_report_null_effect(tmp_path):
    """Paired diffs with similar mean and realistic noise → gate doesn't fire.

    Using independent jitter on C and D so the paired-diff distribution
    has non-zero variance — the per-seed diff is roughly ``±0.005``,
    centered near zero.
    """
    seeds = [1337, 42, 123, 7, 8]
    c_jitter = [0.005, -0.003, 0.002, 0.004, -0.001]
    d_jitter = [-0.002, 0.004, -0.005, 0.001, 0.003]
    _seed_synthetic_four_cell(
        tmp_path,
        A=[(s, 1.50) for s in seeds],
        B=[(s, 1.50) for s in seeds],
        C=[(s, 1.60 + c_jitter[i]) for i, s in enumerate(seeds)],
        D=[(s, 1.60 + d_jitter[i]) for i, s in enumerate(seeds)],
    )

    report = build_report(tmp_path)
    assert report["p_primary"] > 0.01
    assert report["thesis_validating"] is False


def test_build_report_missing_cells_reports_gracefully(tmp_path):
    """Partial results (one cell missing) → report returns with n=0 for that cell."""
    seeds = [1337, 42, 123, 7, 8]
    _seed_synthetic_four_cell(
        tmp_path,
        A=[(s, 1.50) for s in seeds],
        B=[(s, 1.49) for s in seeds],
        C=[(s, 1.60) for s in seeds],
        D=[],  # D never ran
    )

    report = build_report(tmp_path)
    # With zero D seeds, pairing C×D is empty; primary p is None (insufficient data).
    assert report["n_ssm_pairs"] == 0
    assert report["p_primary"] is None
    assert report["thesis_validating"] is False


def test_build_report_includes_control_results(tmp_path):
    """fullcov + shuffled dirs are picked up when present, with paired test."""
    seeds = [1337, 42, 123, 7, 8]
    _seed_synthetic_four_cell(
        tmp_path,
        A=[(s, 1.50) for s in seeds],
        B=[(s, 1.49) for s in seeds],
        C=[(s, 1.60) for s in seeds],
        D=[(s, 1.40) for s in seeds],
    )
    for s in seeds:
        _write_run_json(
            tmp_path / "fullcov" / f"ssm_fullcov_s{s}.json", 1.45
        )
    _write_run_json(tmp_path / "shuffled" / "ssm_shuffled_s1337.json", 1.59)

    report = build_report(tmp_path)
    assert report["fullcov"]["n"] == 5
    assert report["fullcov"]["mean"] == pytest.approx(1.45)
    assert report["shuffled"]["mean"] == pytest.approx(1.59)
    # Paired test on fullcov vs meanstd-D: H1 fullcov bpb > D (i.e., full-cov
    # is worse → meanstd arm's directional signal is still needed).
    assert report["n_fullcov_pairs"] == 5
    assert report["p_fullcov_vs_meanstd"] is not None
    assert report["p_fullcov_vs_meanstd"] < 0.01  # 1.45 >> 1.40 at every seed


def test_build_report_consistency_of_delta_and_p(tmp_path):
    """Reported Δ_SSM must be computed on the same seed set as p_primary.

    Regression: earlier design used different seed-set strategies for
    Δ (4-way intersection) vs p_primary (C∩D only), which could display
    a mean that didn't match the test when A or B lost a seed.
    """
    seeds = [1337, 42, 123, 7, 8]
    # A is missing seed 8 (transformer flake), C and D are complete.
    _seed_synthetic_four_cell(
        tmp_path,
        A=[(s, 1.50) for s in seeds if s != 8],
        B=[(s, 1.49) for s in seeds if s != 8],
        C=[(s, 1.60) for s in seeds],
        D=[(s, 1.40) for s in seeds],
    )

    report = build_report(tmp_path)
    # Both primary and secondary operate on the 4-way common set = 4 seeds.
    assert report["n_common_seeds"] == 4
    assert report["n_ssm_pairs"] == 4
    assert report["delta_ssm_mean"] == pytest.approx(0.20)


def test_load_cell_bpbs_duplicate_seeds_last_wins(tmp_path):
    """Two JSONs with the same {cell, seed} → the {seed: bpb} dict stores one.

    This locks in graceful degradation: in practice orchestrators don't
    write duplicates, but if a rerun produces an overwrite, the loader
    must not crash or double-count."""
    cell_dir = tmp_path / "four_cell"
    _write_run_json(cell_dir / "A_transformer_random_s1337.json", 1.55)
    # Overwrite with different content — same filename, simulating a rerun.
    _write_run_json(cell_dir / "A_transformer_random_s1337.json", 1.60)

    out = load_cell_bpbs(cell_dir, "A_transformer_random")
    assert out == {1337: 1.60}
