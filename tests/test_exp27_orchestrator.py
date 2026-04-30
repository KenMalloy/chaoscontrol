"""Tests for exp27 TTT headline orchestrator + matrix builder + stub analyzer."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP23 = REPO / "experiments" / "23_fast_path"
EXP24 = REPO / "experiments" / "24_training_time_bundle"
EXP27 = REPO / "experiments" / "27_ttt_headline"
SRC = REPO / "src"


def _load_module(name: str, path: Path):
    for p in (EXP23, EXP24, EXP27, SRC):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exp27():
    return _load_module("exp27", EXP27 / "exp27.py")


@pytest.fixture
def calibrate():
    return _load_module("calibrate", EXP27 / "calibrate.py")


@pytest.fixture
def run_exp27_module():
    return _load_module("run_exp27", EXP27 / "run_exp27.py")


@pytest.fixture
def speed_config() -> dict:
    return {
        "lr": 0.064,
        "world_size": 4,
        "model_dim": 256,
        "seq_len": 512,
        "batch_size": 64,
        "warmup_steps": 10,
        "submit_valid": True,
    }


def _write_stub_manifest(path: Path, calibrate_module) -> dict:
    manifest = calibrate_module.analyze(manifest_path=path)
    return manifest


# ---- analyzer (stub) -------------------------------------------------------


def test_analyze_writes_manifest_with_required_keys(tmp_path, calibrate):
    manifest_path = tmp_path / "manifest.json"
    out = calibrate.analyze(manifest_path=manifest_path)
    assert manifest_path.exists()
    assert set(out.keys()) >= {
        "calibrated_at",
        "source_trace",
        "calc_type_hyperparams",
        "note",
    }
    loaded = json.loads(manifest_path.read_text())
    assert set(loaded["calc_type_hyperparams"].keys()) == {
        "score_only_reset",
        "adaptive_carry",
        "carry_state",
        "dreamworld_eval",
    }
    assert loaded["source_trace"] == "stub"
    assert "stub" in loaded["note"].lower()


def test_analyze_requires_manifest_path_kwarg(calibrate):
    # `manifest_path` is keyword-only and required.
    with pytest.raises(TypeError):
        calibrate.analyze()  # type: ignore[call-arg]


# ---- load_manifest ---------------------------------------------------------


def test_load_manifest_missing_raises(tmp_path, exp27):
    with pytest.raises(FileNotFoundError, match="calibration manifest missing"):
        exp27.load_manifest(tmp_path / "nope.json")


def test_load_manifest_malformed_raises(tmp_path, exp27):
    p = tmp_path / "manifest.json"
    p.write_text("{not valid json")
    with pytest.raises(ValueError, match="malformed"):
        exp27.load_manifest(p)


def test_load_manifest_missing_required_keys_raises(tmp_path, exp27):
    p = tmp_path / "manifest.json"
    # Has calibrated_at but is missing calc_type_hyperparams.
    p.write_text(json.dumps({"calibrated_at": "2026-04-28T00:00:00Z"}))
    with pytest.raises(ValueError, match="missing required keys"):
        exp27.load_manifest(p)


# ---- build_ttt_headline_matrix ---------------------------------------------


def test_matrix_default_args_returns_one_entry_per_seed(
    tmp_path, exp27, calibrate, speed_config
):
    manifest_path = tmp_path / "manifest.json"
    _write_stub_manifest(manifest_path, calibrate)
    entries = exp27.build_ttt_headline_matrix(
        speed_config=speed_config,
        calibration_manifest_path=manifest_path,
    )
    assert len(entries) == 3
    seeds = {entry["seed"] for entry in entries}
    assert seeds == {1337, 2674, 4011}
    names = {entry["name"] for entry in entries}
    assert names == {
        "exp27_ttt_headline_s1337",
        "exp27_ttt_headline_s2674",
        "exp27_ttt_headline_s4011",
    }


def test_matrix_entries_carry_default_calc_types_and_hyperparams(
    tmp_path, exp27, calibrate, speed_config
):
    manifest_path = tmp_path / "manifest.json"
    _write_stub_manifest(manifest_path, calibrate)
    entries = exp27.build_ttt_headline_matrix(
        speed_config=speed_config,
        calibration_manifest_path=manifest_path,
    )
    for entry in entries:
        assert tuple(entry["calc_types"]) == exp27.CALC_TYPES_DEFAULT
        configs = entry["calc_type_configs"]
        assert set(configs.keys()) == set(exp27.CALC_TYPES_DEFAULT)
        # Stub defaults survive the round-trip into the entry.
        assert configs["adaptive_carry"]["decay"] == 1.0
        assert configs["adaptive_carry"]["horizon_shifts"] == [-0.5, 0.0, 0.5]
        assert configs["adaptive_carry"]["online_eta"] == 1.0
        # Default checkpoint_path is None until the orchestrator sets it.
        assert entry["checkpoint_path"] is None
        # World size + budget land in the entry.
        assert entry["world_size"] == 4
        assert entry["budget_seconds"] == 600.0


def test_matrix_calc_type_subset_is_respected(
    tmp_path, exp27, calibrate, speed_config
):
    manifest_path = tmp_path / "manifest.json"
    _write_stub_manifest(manifest_path, calibrate)
    entries = exp27.build_ttt_headline_matrix(
        speed_config=speed_config,
        calibration_manifest_path=manifest_path,
        calc_types=["score_only_reset"],
        seed_values=[1337],
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry["calc_types"] == ["score_only_reset"]
    assert set(entry["calc_type_configs"].keys()) == {"score_only_reset"}


def test_matrix_unknown_calc_type_raises(
    tmp_path, exp27, calibrate, speed_config
):
    manifest_path = tmp_path / "manifest.json"
    _write_stub_manifest(manifest_path, calibrate)
    with pytest.raises(ValueError, match="unknown calc_type"):
        exp27.build_ttt_headline_matrix(
            speed_config=speed_config,
            calibration_manifest_path=manifest_path,
            calc_types=["does_not_exist"],
        )


def test_matrix_missing_manifest_raises(tmp_path, exp27, speed_config):
    with pytest.raises(FileNotFoundError, match="calibration manifest missing"):
        exp27.build_ttt_headline_matrix(
            speed_config=speed_config,
            calibration_manifest_path=tmp_path / "missing.json",
        )


def test_matrix_checkpoint_path_fails_loud_until_runner_load_is_wired(
    tmp_path, exp27, calibrate, speed_config
):
    manifest_path = tmp_path / "manifest.json"
    _write_stub_manifest(manifest_path, calibrate)
    ckpt = tmp_path / "winner.pt"
    with pytest.raises(NotImplementedError, match="checkpoint_path"):
        exp27.build_ttt_headline_matrix(
            speed_config=speed_config,
            calibration_manifest_path=manifest_path,
            checkpoint_path=ckpt,
            seed_values=[1337],
        )


# ---- orchestrator dry-run isolation ----------------------------------------


def test_dry_run_creates_no_files(tmp_path, run_exp27_module):
    """``--dry-run`` must not mkdir or write anywhere we point it.

    Test fixture pre-creates only the output dir's parent (tmp_path) and
    asserts no manifest, no matrix.json, and no calibration/results
    subdirs appeared after main() returns.
    """
    headline_dir = tmp_path / "results"
    manifest_path = tmp_path / "calibration" / "manifest.json"
    # Sanity: tmp_path starts effectively empty.
    assert list(tmp_path.iterdir()) == []

    rc = run_exp27_module.main(
        [
            "--stage", "all",
            "--dry-run",
            "--headline-output-dir", str(headline_dir),
            "--manifest-path", str(manifest_path),
        ]
    )
    assert rc == 0

    # Strict isolation: nothing under tmp_path was created.
    assert list(tmp_path.iterdir()) == [], (
        f"dry-run leaked files: {list(tmp_path.iterdir())}"
    )
    assert not headline_dir.exists()
    assert not manifest_path.parent.exists()
    assert not manifest_path.exists()


def test_launcher_pythonpath_includes_runner_helper_dirs(
    monkeypatch, run_exp27_module
):
    monkeypatch.setenv("PYTHONPATH", "/already/here")

    run_exp27_module._ensure_child_pythonpath()

    parts = run_exp27_module.os.environ["PYTHONPATH"].split(
        run_exp27_module.os.pathsep
    )
    assert str(run_exp27_module.REPO / "src") in parts
    assert str(run_exp27_module.REPO / "experiments") in parts
    assert str(run_exp27_module.EXP17) in parts
    assert str(run_exp27_module.EXP21) in parts
    assert "/already/here" in parts
