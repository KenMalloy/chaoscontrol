"""Tests for experiments/19_phase1/summarize_phase1.py.

Spec source: docs/plans/2026-04-17-experiment-19-phase1-impl.md Task
1C-3 and "Ablation matrix gates". Synthetic JSON fixtures — no GPU,
no real runner output. All tests must complete in under 2 seconds.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "19_phase1" / "summarize_phase1.py"


def _load_module() -> Any:
    """Load summarize_phase1.py as a module without touching sys.path
    globally — the experiments/19_phase1/ directory has no
    ``__init__.py`` (repo convention), so we go through importlib.
    """
    spec = importlib.util.spec_from_file_location(
        "summarize_phase1", MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None, (
        f"could not load {MODULE_PATH}"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LEVER_KEYS = ("fused_grad_clip", "fused_muon", "compile_full_path")


def _make_entry(
    *,
    precision: str,
    seed: int,
    label: str,
    fused_grad_clip: bool,
    fused_muon: bool,
    compile_full_path: bool,
) -> dict[str, Any]:
    """Build a matrix-entry dict in the exact shape build_matrix_phase1
    emits — name, seed, precision, lever flags, and the minimum config
    fields the summarizer reads (batch_size, seq_len, world_size).
    """
    return {
        "name": f"{precision}_{label}_seed{seed}",
        "seed": seed,
        "precision": precision,
        "fused_grad_clip": fused_grad_clip,
        "fused_muon": fused_muon,
        "compile_full_path": compile_full_path,
        "batch_size": 1024,
        "seq_len": 512,
        "world_size": 4,
    }


def _stock_entry(precision: str, seed: int) -> dict[str, Any]:
    return _make_entry(
        precision=precision, seed=seed, label="stock",
        fused_grad_clip=False, fused_muon=False, compile_full_path=False,
    )


def _all_on_entry(precision: str, seed: int) -> dict[str, Any]:
    return _make_entry(
        precision=precision, seed=seed, label="all",
        fused_grad_clip=True, fused_muon=True, compile_full_path=True,
    )


def _lever_off_entry(
    precision: str, seed: int, lever: str,
) -> dict[str, Any]:
    label = {
        "fused_grad_clip": "no_fused_clip",
        "fused_muon": "no_fused_muon",
        "compile_full_path": "no_compile",
    }[lever]
    flags = {k: (k != lever) for k in LEVER_KEYS}
    return _make_entry(
        precision=precision, seed=seed, label=label, **flags,
    )


def _build_full_matrix(
    seeds: list[int], precisions: tuple[str, ...] = ("bf16",),
) -> list[dict[str, Any]]:
    """Emit a complete Phase 1 matrix: stock + all + one-off-per-lever
    per (precision, seed)."""
    entries: list[dict[str, Any]] = []
    for precision in precisions:
        for seed in seeds:
            entries.append(_stock_entry(precision, seed))
            entries.append(_all_on_entry(precision, seed))
            for lever in LEVER_KEYS:
                entries.append(_lever_off_entry(precision, seed, lever))
    return entries


def _write_result(
    runs_dir: Path,
    entry: dict[str, Any],
    *,
    steps_per_second: float,
    final_loss: float = 4.0,
    peak_vram_mb: float = 30_000.0,
    bpb: float | None = 1.5,
    error: str | None = None,
) -> None:
    """Write a runner-shaped result JSON for this matrix entry."""
    name = entry["name"]
    seed = int(entry["seed"])
    path = runs_dir / f"{name}_s{seed}.json"
    config = {
        "batch_size": entry["batch_size"],
        "seq_len": entry["seq_len"],
        "world_size": entry["world_size"],
        "precision": entry["precision"],
        "base_lr": 0.128,
    }
    if error is not None:
        payload = {"config": config, "error": error}
    else:
        eval_block: dict[str, Any] = {}
        if bpb is not None:
            eval_block = {"bpb": bpb, "loss": 4.0}
        payload = {
            "config": config,
            "params": 10_000_000,
            "train": {
                "steps": 1000,
                "elapsed_s": 600.0,
                "steps_per_second": steps_per_second,
                "final_loss": final_loss,
                "peak_vram_mb": peak_vram_mb,
            },
            "eval": eval_block,
        }
    path.write_text(json.dumps(payload))


def _write_matrix_json(tmp_path: Path, matrix: list[dict[str, Any]]) -> Path:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    return matrix_path


# --- coherent-pair resolver ------------------------------------------


def test_coherent_pair_resolver_happy_path(tmp_path: Path) -> None:
    mod = _load_module()
    matrix = _build_full_matrix(seeds=[1337, 2674])
    pairs = mod.resolve_coherent_pairs(matrix)

    # Every (precision, lever) pairing exists with one pair per seed.
    assert set(pairs.keys()) == {("bf16", k) for k in LEVER_KEYS}
    for (precision, lever), pair_list in pairs.items():
        assert len(pair_list) == 2, (
            f"expected 2 pairs for {precision}/{lever}, got {len(pair_list)}"
        )
        for all_on, lever_off in pair_list:
            assert all_on["seed"] == lever_off["seed"]
            assert all_on["precision"] == precision
            assert lever_off["precision"] == precision
            assert lever_off[lever] is False
            # Other two levers must be ON in the "lever-off" entry.
            for other in LEVER_KEYS:
                if other == lever:
                    continue
                assert lever_off[other] is True
            # all-on entry has every lever ON.
            for k in LEVER_KEYS:
                assert all_on[k] is True


def test_coherent_pair_resolver_ambiguous_raises(tmp_path: Path) -> None:
    mod = _load_module()
    matrix = _build_full_matrix(seeds=[1337])
    # Duplicate the all-on entry at seed=1337 with a distinct name so
    # two entries match the "all-on" shape at the same (precision, seed).
    dup = _all_on_entry("bf16", 1337)
    dup["name"] = dup["name"] + "_dup"
    matrix.append(dup)

    with pytest.raises(RuntimeError, match="coherent-pair resolver"):
        mod.resolve_coherent_pairs(matrix)


# --- full-coverage gate -----------------------------------------------


def test_full_coverage_gate_catches_missing(tmp_path: Path) -> None:
    mod = _load_module()
    matrix = _build_full_matrix(seeds=[1337, 2674])
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # Write results for ALL entries except one (bf16_all_seed2674).
    skipped_name = "bf16_all_seed2674"
    for entry in matrix:
        if entry["name"] == skipped_name:
            continue
        _write_result(runs_dir, entry, steps_per_second=5.0)

    with pytest.raises(RuntimeError, match="full-coverage gate"):
        mod.full_coverage_gate(matrix, runs_dir, lenient=False)


def test_full_coverage_gate_lenient_passes(tmp_path: Path) -> None:
    mod = _load_module()
    matrix = _build_full_matrix(seeds=[1337, 2674])
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for entry in matrix:
        if entry["name"] == "bf16_all_seed2674":
            continue
        _write_result(runs_dir, entry, steps_per_second=5.0)

    missing = mod.full_coverage_gate(matrix, runs_dir, lenient=True)
    assert len(missing) == 1
    assert missing[0]["name"] == "bf16_all_seed2674"

    # And the full summarize() path surfaces the missing entry in the
    # markdown rather than raising.
    matrix_path = _write_matrix_json(tmp_path, matrix)
    exit_code, markdown = mod.summarize(
        matrix_path=matrix_path, runs_dir=runs_dir, lenient=True,
    )
    # Exit code is 1 iff quality gate fails — missing file is a real
    # error in integrity terms, so lenient coverage + strict quality
    # still flags it.
    assert "bf16_all_seed2674" in markdown
    assert "missing" in markdown


# --- quality gate -----------------------------------------------------


def test_quality_gate_real_errors_refuses(tmp_path: Path) -> None:
    mod = _load_module()
    matrix = _build_full_matrix(seeds=[1337, 2674])
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # One entry writes an error marker with a non-benign error string.
    error_name = "bf16_all_seed1337"
    for entry in matrix:
        if entry["name"] == error_name:
            _write_result(
                runs_dir, entry, steps_per_second=0.0,
                error="RuntimeError: CUDA out of memory",
            )
        else:
            _write_result(runs_dir, entry, steps_per_second=5.0)

    matrix_path = _write_matrix_json(tmp_path, matrix)
    exit_code, markdown = mod.summarize(
        matrix_path=matrix_path, runs_dir=runs_dir, lenient=False,
    )
    assert exit_code == 1, "quality gate should return non-zero on real errors"
    assert "Quality gate failed" in markdown
    assert "CUDA out of memory" in markdown
    # Verdict tables must NOT appear — verdicts are suppressed.
    assert "paired-t p" not in markdown


# --- verdict logic ----------------------------------------------------


def test_verdict_ship(tmp_path: Path) -> None:
    """Lever clearly wins — verdict is SHIP.

    Seeds give the paired-t its dof. All-on beats lever-off by ~+100
    tok/step with very low within-pair variance; loss and VRAM match.
    """
    mod = _load_module()
    seeds = [1337, 2674, 4011, 5348]
    matrix = _build_full_matrix(seeds=seeds)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for entry in matrix:
        # Mark the all-on + every-other-entry as "baseline 5.0 sps";
        # toggle off fused_grad_clip pushes sps down to 4.8.
        # Stock and other leave-out entries share the baseline to keep
        # coverage simple.
        if entry["fused_grad_clip"] and entry["fused_muon"] and entry["compile_full_path"]:
            sps = 5.2
        elif not entry["fused_grad_clip"] and entry["fused_muon"] and entry["compile_full_path"]:
            sps = 4.8
        else:
            sps = 5.0
        _write_result(runs_dir, entry, steps_per_second=sps)

    matrix_path = _write_matrix_json(tmp_path, matrix)
    exit_code, markdown = mod.summarize(
        matrix_path=matrix_path, runs_dir=runs_dir, lenient=False,
    )
    assert exit_code == 0, markdown
    # Find the fused_grad_clip row.
    assert "fused_grad_clip" in markdown
    # Locate the SHIP verdict in fused_grad_clip row.
    rows = [
        line for line in markdown.splitlines()
        if line.startswith("| fused_grad_clip")
    ]
    assert len(rows) == 1
    assert "SHIP" in rows[0], f"expected SHIP verdict in row: {rows[0]}"


def test_verdict_inconclusive_low_n(tmp_path: Path) -> None:
    """Positive mean Δ tok/s but p ≥ 0.05 with only 2 seeds."""
    mod = _load_module()
    seeds = [1337, 2674]
    matrix = _build_full_matrix(seeds=seeds)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # Tiny positive delta with large noise — paired-t can't resolve it.
    # seed 1337: all_on=5.0, lever_off=4.99 (delta=+0.01)
    # seed 2674: all_on=5.0, lever_off=5.1  (delta=-0.1)
    # Mean delta is negative → actually PARK, not INCONCLUSIVE.
    # Use a slightly different setup: mean positive but variance huge.
    # seed 1337: all_on=5.1, lever_off=5.0 (delta=+0.1)
    # seed 2674: all_on=5.0, lever_off=4.95 (delta=+0.05)
    # Mean = +0.075, p is high because only n=2.
    for entry in matrix:
        if entry["fused_grad_clip"] and entry["fused_muon"] and entry["compile_full_path"]:
            sps = 5.1 if entry["seed"] == 1337 else 5.0
        elif (
            not entry["fused_grad_clip"]
            and entry["fused_muon"]
            and entry["compile_full_path"]
        ):
            sps = 5.0 if entry["seed"] == 1337 else 4.95
        else:
            sps = 5.0
        _write_result(runs_dir, entry, steps_per_second=sps)

    matrix_path = _write_matrix_json(tmp_path, matrix)
    exit_code, markdown = mod.summarize(
        matrix_path=matrix_path, runs_dir=runs_dir, lenient=False,
    )
    assert exit_code == 0, markdown
    rows = [
        line for line in markdown.splitlines()
        if line.startswith("| fused_grad_clip")
    ]
    assert len(rows) == 1
    assert "INCONCLUSIVE" in rows[0], (
        f"expected INCONCLUSIVE verdict in row: {rows[0]}"
    )


def test_verdict_park_on_regression(tmp_path: Path) -> None:
    """Negative mean Δ tok/s — verdict is PARK."""
    mod = _load_module()
    seeds = [1337, 2674, 4011, 5348]
    matrix = _build_full_matrix(seeds=seeds)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for entry in matrix:
        # all-on is SLOWER than lever-off — the lever hurts throughput.
        if entry["fused_grad_clip"] and entry["fused_muon"] and entry["compile_full_path"]:
            sps = 4.5
        elif (
            not entry["fused_grad_clip"]
            and entry["fused_muon"]
            and entry["compile_full_path"]
        ):
            sps = 5.0
        else:
            sps = 5.0
        _write_result(runs_dir, entry, steps_per_second=sps)

    matrix_path = _write_matrix_json(tmp_path, matrix)
    exit_code, markdown = mod.summarize(
        matrix_path=matrix_path, runs_dir=runs_dir, lenient=False,
    )
    assert exit_code == 0, markdown
    rows = [
        line for line in markdown.splitlines()
        if line.startswith("| fused_grad_clip")
    ]
    assert len(rows) == 1
    assert "PARK" in rows[0], f"expected PARK verdict in row: {rows[0]}"


# --- markdown shape ---------------------------------------------------


def test_output_markdown_shape(tmp_path: Path) -> None:
    """One section per precision, one row per lever in each table."""
    mod = _load_module()
    seeds = [1337, 2674, 4011, 5348]
    matrix = _build_full_matrix(seeds=seeds, precisions=("bf16", "fp8_fused"))

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for entry in matrix:
        _write_result(runs_dir, entry, steps_per_second=5.0)

    matrix_path = _write_matrix_json(tmp_path, matrix)
    exit_code, markdown = mod.summarize(
        matrix_path=matrix_path, runs_dir=runs_dir, lenient=False,
    )
    assert exit_code == 0, markdown

    # Two precision sections.
    assert "## precision = bf16" in markdown
    assert "## precision = fp8_fused" in markdown

    # Each precision section has one row per lever.
    for precision in ("bf16", "fp8_fused"):
        section_start = markdown.index(f"## precision = {precision}")
        next_section = markdown.find("## precision = ", section_start + 1)
        section_body = markdown[
            section_start: next_section if next_section >= 0 else None
        ]
        for lever in LEVER_KEYS:
            pattern = re.compile(
                rf"^\|\s*{re.escape(lever)}\s*\|", re.MULTILINE,
            )
            assert pattern.search(section_body), (
                f"missing row for lever {lever!r} in {precision} section"
            )
