"""Tests for experiments/19_phase1/build_matrix_phase1.py.

Spec source: docs/plans/2026-04-17-experiment-19-phase1-impl.md, Task 1C-1.

The matrix is a one-at-a-time lever-leave-out: for each precision, there
are exactly 5 lever combinations — stock (all three levers off), all-on,
and one "leave-one-out" condition per lever (the other two on). Each
combination × N seeds gives the seeds their paired-t leverage.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "19_phase1" / "build_matrix_phase1.py"
RUNNER_PATH = REPO / "experiments" / "19_prereqs" / "runner_persistent_ddp.py"


def _load_module():
    """Load build_matrix_phase1.py as a module without touching sys.path
    globally — the experiments/19_phase1/ directory has no __init__.py
    and the spec forbids adding one.
    """
    spec = importlib.util.spec_from_file_location(
        "build_matrix_phase1", MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None, (
        f"could not load {MODULE_PATH}"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LEVER_KEYS = ("fused_grad_clip", "fused_muon", "compile_full_path")


def test_bf16_only_size():
    mod = _load_module()
    entries = mod.build_matrix_phase1(
        seeds=[1, 2, 3, 4], include_fp8=False,
    )
    # 4 seeds × 5 lever-combos (stock + all + 3 leave-one-outs) = 20.
    assert len(entries) == 20


def test_with_fp8_size():
    mod = _load_module()
    entries = mod.build_matrix_phase1(
        seeds=[1, 2, 3, 4], include_fp8=True,
    )
    # Doubles to 40 — 5 combos × 2 precisions × 4 seeds.
    assert len(entries) == 40


def test_each_lever_off_in_exactly_one_nonbaseline_condition():
    """For each lever L and each precision, across one seed's slice there
    must be exactly ONE lever-combo that has L=False with the other two
    levers True (the "leave-L-out" row). Plus the all-off stock row and
    the all-on row, which don't satisfy that pattern.
    """
    mod = _load_module()
    entries = mod.build_matrix_phase1(
        seeds=[7], include_fp8=True,
    )
    # 5 combos × 2 precisions × 1 seed = 10 entries.
    assert len(entries) == 10

    for precision in ("bf16", "fp8_fused"):
        slice_ = [e for e in entries if e["precision"] == precision]
        assert len(slice_) == 5, (
            f"expected 5 entries for precision={precision}, got {len(slice_)}"
        )

        # Exactly one "all-on" row.
        all_on = [e for e in slice_ if all(e[k] for k in LEVER_KEYS)]
        assert len(all_on) == 1, f"{precision}: all-on count = {len(all_on)}"

        # Exactly one "stock" (all-off) row.
        stock = [e for e in slice_ if not any(e[k] for k in LEVER_KEYS)]
        assert len(stock) == 1, f"{precision}: stock count = {len(stock)}"

        # Each lever must be OFF in exactly one entry that has the other
        # two ON (the leave-one-out row).
        for lever in LEVER_KEYS:
            others = [k for k in LEVER_KEYS if k != lever]
            matches = [
                e for e in slice_
                if e[lever] is False and all(e[o] is True for o in others)
            ]
            assert len(matches) == 1, (
                f"{precision}: expected exactly 1 leave-{lever}-out row, "
                f"found {len(matches)}"
            )


def test_seed_x_condition_is_unique():
    mod = _load_module()
    entries = mod.build_matrix_phase1(
        seeds=[1337, 2674, 4011, 5348], include_fp8=True,
    )
    keys = [
        (e["seed"], e["precision"], e["fused_grad_clip"],
         e["fused_muon"], e["compile_full_path"])
        for e in entries
    ]
    assert len(set(keys)) == len(keys), (
        f"duplicate (seed, precision, levers) tuples found: "
        f"{len(keys) - len(set(keys))} duplicates"
    )


def test_names_unique_and_readable():
    mod = _load_module()
    entries = mod.build_matrix_phase1(
        seeds=[1337, 2674], include_fp8=True,
    )
    names = [e["name"] for e in entries]
    assert len(set(names)) == len(names), "names are not unique"

    pattern = re.compile(
        r"^(bf16|fp8_fused)_(stock|all|no_fused_clip|no_fused_muon|no_compile)_seed\d+$"
    )
    for name in names:
        assert pattern.match(name), f"name {name!r} does not match expected pattern"


def test_base_config_merged():
    mod = _load_module()
    base = {"model_dim": 256, "num_layers": 4, "custom_field": "sentinel"}
    entries = mod.build_matrix_phase1(
        seeds=[1], include_fp8=False, base_config=base,
    )
    assert len(entries) == 5
    for entry in entries:
        assert entry["model_dim"] == 256
        assert entry["num_layers"] == 4
        assert entry["custom_field"] == "sentinel"
        # Lever toggles and precision must be present layered on top.
        assert "fused_grad_clip" in entry
        assert "fused_muon" in entry
        assert "compile_full_path" in entry
        assert entry["precision"] == "bf16"


def test_cli_stdout_roundtrip(tmp_path):
    """`python build_matrix_phase1.py --seeds 1337` prints JSON to stdout."""
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--seeds", "1337"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"CLI exited non-zero: rc={result.returncode} stderr={result.stderr}"
    )
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list), "CLI stdout must be a JSON list"
    assert len(parsed) == 5, f"expected 5 entries (bf16-only, 1 seed), got {len(parsed)}"
    for entry in parsed:
        assert entry["seed"] == 1337
        assert entry["precision"] == "bf16"


def test_duplicate_seeds_rejected():
    """Duplicate seeds must raise — otherwise name collisions would
    silently halve the paired-t sample size via per-entry JSON overwrites.
    """
    mod = _load_module()
    with pytest.raises(ValueError, match="duplicates"):
        mod.build_matrix_phase1(seeds=[1337, 1337])
    with pytest.raises(ValueError, match="duplicates"):
        mod.build_matrix_phase1(seeds=[1, 2, 1, 3])


def test_cli_output_file_roundtrip(tmp_path):
    """`--output PATH` writes the matrix JSON to PATH."""
    out = tmp_path / "matrix.json"
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH),
         "--seeds", "1337", "2674", "--output", str(out)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"CLI exited non-zero: rc={result.returncode} stderr={result.stderr}"
    )
    assert out.exists(), f"--output path was not written: {out}"
    parsed = json.loads(out.read_text())
    assert isinstance(parsed, list) and len(parsed) == 10  # 2 seeds × 5 bf16 combos
    seeds = {e["seed"] for e in parsed}
    assert seeds == {1337, 2674}


def test_key_names_match_runner_contract():
    """The runner must consume the exact keys this module emits.

    We read the runner source as text and check that every lever/precision
    key name we emit appears verbatim in it — a substring match is
    sufficient because the runner uses ``config.get("fused_muon", ...)``
    style lookups, so the key string is literally in the source.
    """
    assert RUNNER_PATH.exists(), f"runner file missing: {RUNNER_PATH}"
    src = RUNNER_PATH.read_text()
    for key in ("fused_grad_clip", "fused_muon", "compile_full_path", "precision"):
        assert key in src, (
            f"runner {RUNNER_PATH} does not reference key {key!r}; "
            f"matrix would emit unconsumed fields"
        )
