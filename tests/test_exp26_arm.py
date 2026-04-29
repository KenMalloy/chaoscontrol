"""Tests for exp26 Adaptive Residual Memory matrix builders + analyzer."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP24 = REPO / "experiments" / "24_training_time_bundle"
EXP26 = REPO / "experiments" / "26_arm"


def _load_module(name: str, path: Path):
    if str(EXP24) not in sys.path:
        sys.path.insert(0, str(EXP24))
    if str(EXP26) not in sys.path:
        sys.path.insert(0, str(EXP26))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exp26():
    return _load_module("exp26", EXP26 / "exp26.py")


@pytest.fixture
def calibrate():
    return _load_module("calibrate", EXP26 / "calibrate.py")


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


def _write_synthetic_trace(path: Path, n_decisions: int = 200) -> None:
    """Write a synthetic shadow-mode trace with realistic signal scales."""
    rng_state = 1
    rows = []
    for i in range(n_decisions):
        # cheap deterministic pseudo-random generator
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        u = (rng_state / 0x7FFFFFFF)
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        v = (rng_state / 0x7FFFFFFF)
        rows.append(
            {
                "row_type": "replay_preserve" if u < 0.7 else "replay_evict",
                "step": i,
                "tick": i // 4,
                "slot_id": i % 50,
                "action": "PRESERVE" if u < 0.7 else "EVICT",
                "marginal_gain": 0.001 + 0.05 * v,
                "sharpness": 0.002 * v,
                "activation_drift": 0.01 * u,
                "representation_drift": 0.02 * u,
                "semantic_drift": 0.005 * u,
                "contradiction": 0.001 * (1.0 - v),
                "retrieval_mass": 0.1 * v,
                "peak_utility": 0.005 + 0.1 * v,
                "peak_sharpness": 0.003 + 0.05 * v,
                "score_count": 5 + int(10 * u),
                "state": "ACTIVE",
            }
        )
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# ---- build_calibration_matrix ----------------------------------------------


def test_smoke_matrix_two_entries_and_isolated_outputs(exp26, speed_config):
    entries = exp26.build_smoke_matrix(
        speed_config=speed_config,
        world_size=4,
        budget_seconds=30.0,
        seed=42,
    )
    assert len(entries) == 2
    by_arm = {entry["arm"]: entry for entry in entries}
    assert set(by_arm) == {
        "smoke_fastslow_control",
        "smoke_crct_replay_active",
    }

    control = by_arm["smoke_fastslow_control"]
    active = by_arm["smoke_crct_replay_active"]
    assert control["budget_seconds"] == 30.0
    assert control["model_dim"] == exp26.EXP26_MODEL_DIM == 384
    assert control.get("crct_enabled") is not True
    assert control.get("replay_eviction_enabled") is not True
    assert active["budget_seconds"] == 30.0
    assert active["model_dim"] == 384
    assert active["crct_enabled"] is True
    assert active["replay_eviction_enabled"] is True
    assert active["replay_eviction_mode"] == "active"
    assert active["replay_eviction_scoring_mode"] == "oracle"
    assert active["replay_eviction_arm_runtime_enabled"] is True
    assert (
        active["replay_eviction_arm_runtime_namespace"]
        == "exp26_smoke_smoke_crct_replay_active_s42"
    )
    assert "replay_eviction_cpu_scorer_backend" not in active
    assert "replay_eviction_cpu_scorer_lanes" not in active
    assert active["replay_eviction_refresh_candidate_count"] == 16
    assert active["replay_eviction_refresh_proposal_rank"] == 8
    assert active["replay_eviction_controller_state_dim"] == 32
    assert active["replay_eviction_controller_rank"] == 8
    assert active["replay_eviction_controller_target_log_sv"] == -0.05
    assert active["replay_eviction_refresh_candidate_variant_chunk_size"] == 16
    assert active["replay_eviction_refresh_proposal_weight_sync_interval_steps"] == 64
    assert active["bucket_prototypes"] is True
    assert active["prototype_dim"] == 64
    assert active["replay_eviction_action_agreement_count"] == 1
    assert active["replay_eviction_commit_policy"] == "learned"
    assert active["replay_eviction_commit_online_lr"] == 0.05
    assert "smoke" in active["replay_eviction_trace_path"]
    assert "calibration" not in active["replay_eviction_trace_path"]
    assert "results" not in active["replay_eviction_trace_path"]


def test_calibration_matrix_single_entry(exp26, speed_config):
    entries = exp26.build_calibration_matrix(
        speed_config=speed_config,
        world_size=4,
        budget_seconds=180.0,
        seed=42,
    )
    assert len(entries) == 1
    e = entries[0]
    assert e["seed"] == 42
    assert e["replay_eviction_mode"] == "shadow"
    assert e["replay_eviction_enabled"] is True
    assert e["crct_enabled"] is True
    assert e["fast_slow_enabled"] is True
    assert "exp26_calibration_shadow_s42" in e["name"]
    assert e["replay_eviction_action_agreement_count"] == 1
    assert e["replay_eviction_commit_policy"] == "learned"
    assert e["replay_eviction_max_seconds"] == 8.0
    # Trace path lives in the calibration directory.
    assert "calibration" in str(e["replay_eviction_trace_path"])


def test_calibration_matrix_uses_full_arm_pipeline(exp26, speed_config):
    """Calibration should run the SAME pipeline knobs as headline arms.

    The signal distributions we observe must reflect what arm_d
    will actually see, not a stripped-down telemetry-only path.
    """
    entries = exp26.build_calibration_matrix(speed_config=speed_config)
    assert len(entries) == 1
    e = entries[0]
    # Pipeline knobs identical to headline lock.
    assert e["model_dim"] == 384
    assert e["replay_eviction_memory_streams"] == 8
    assert e["replay_eviction_arm_runtime_enabled"] is True
    assert (
        e["replay_eviction_arm_runtime_namespace"]
        == "exp26_calibration_shadow_s1337"
    )
    assert e["replay_eviction_scoring_mode"] == "oracle"
    assert "replay_eviction_cpu_scorer_backend" not in e
    assert "replay_eviction_cpu_scorer_lanes" not in e
    assert "replay_eviction_cpu_scorer_weight_sync_interval_steps" not in e
    assert e["replay_eviction_oracle_confirm_top_k"] == 32
    assert e["replay_eviction_oracle_variant_chunk_size"] == 1
    assert e["replay_eviction_refresh_candidate_count"] == 16
    assert e["replay_eviction_refresh_proposal_rank"] == 8
    assert e["replay_eviction_controller_state_dim"] == 32
    assert e["replay_eviction_controller_rank"] == 8
    assert e["replay_eviction_controller_target_log_sv"] == -0.05
    assert e["replay_eviction_refresh_candidate_variant_chunk_size"] == 16
    assert e["replay_eviction_refresh_proposal_weight_sync_interval_steps"] == 64
    assert e["replay_eviction_probe_buffer_size"] == 32
    assert e["replay_eviction_frame_ttl_steps"] == 256
    assert e["replay_eviction_max_seconds"] == 8.0
    assert e["replay_eviction_commit_policy"] == "learned"
    assert e["replay_eviction_trace_flush_rows"] == 256
    # Mode is shadow, not active.
    assert e["replay_eviction_mode"] == "shadow"


# ---- analyzer ---------------------------------------------------------------


def test_analyzer_writes_manifest(tmp_path, calibrate):
    trace = tmp_path / "trace.ndjson"
    manifest = tmp_path / "manifest.json"
    _write_synthetic_trace(trace, n_decisions=300)
    out = calibrate.analyze(trace_path=trace, manifest_path=manifest)
    assert manifest.exists()
    loaded = json.loads(manifest.read_text())
    assert loaded["n_decisions_observed"] == 300
    assert "thresholds_balanced" in loaded
    assert "signal_summary" in loaded
    bal = loaded["thresholds_balanced"]
    assert bal["threshold"] > 0.0
    assert bal["min_age_steps"] == 128


def test_analyzer_raises_on_empty_trace(tmp_path, calibrate):
    trace = tmp_path / "empty.ndjson"
    trace.write_text("")
    with pytest.raises(ValueError, match="no replay-decision rows"):
        calibrate.analyze(
            trace_path=trace,
            manifest_path=tmp_path / "manifest.json",
        )


def test_analyzer_raises_on_missing_trace(tmp_path, calibrate):
    with pytest.raises(FileNotFoundError):
        calibrate.analyze(
            trace_path=tmp_path / "nope.ndjson",
            manifest_path=tmp_path / "manifest.json",
        )


def test_analyzer_balanced_uses_p50_thresholds(tmp_path, calibrate):
    """Balanced thresholds should match the percentile-anchor contract.

    eviction threshold = p50 of utility, peak_preserve = p75 of peak.
    """
    trace = tmp_path / "trace.ndjson"
    _write_synthetic_trace(trace, n_decisions=400)
    manifest = calibrate.analyze(
        trace_path=trace,
        manifest_path=tmp_path / "manifest.json",
    )
    summary = manifest["signal_summary"]
    bal = manifest["thresholds_balanced"]
    assert bal["threshold"] == pytest.approx(summary["utility_ema"]["p50"])
    assert bal["useful_threshold"] == pytest.approx(summary["utility_ema"]["p25"])
    assert bal["peak_preserve_utility_threshold"] == pytest.approx(
        summary["peak_utility"]["p75"]
    )
    assert bal["peak_preserve_sharpness_threshold"] == pytest.approx(
        summary["peak_sharpness"]["p75"]
    )


# ---- build_arm_v1_matrix ----------------------------------------------------


def _write_realistic_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "calibrated_at": "2026-04-28T22:00:00Z",
        "source_trace": str(path.parent / "trace.ndjson"),
        "n_decisions_observed": 400,
        "signal_summary": {
            "utility_ema": {"p25": 0.001, "p50": 0.01, "p75": 0.04},
            "peak_utility": {"p25": 0.005, "p50": 0.02, "p75": 0.05, "p90": 0.08},
            "peak_sharpness": {"p25": 0.003, "p50": 0.01, "p75": 0.03},
            "max_drift": {"p25": 0.005, "p50": 0.02, "p75": 0.05},
            "representation_drift": {"p25": 0.005, "p50": 0.02, "p75": 0.04},
            "contradiction_ema": {"p50": 0.001, "p75": 0.005},
        },
        "thresholds_balanced": {
            "threshold": 0.01,
            "useful_threshold": 0.001,
            "drift_threshold": 0.05,
            "repr_drift_threshold": 0.04,
            "quarantine_threshold": -0.005,
            "distill_peak_threshold": 0.08,
            "peak_preserve_utility_threshold": 0.05,
            "peak_preserve_sharpness_threshold": 0.03,
            "min_age_steps": 128,
            "min_score_count": 2,
        },
    }
    path.write_text(json.dumps(manifest, indent=2))


def test_arm_v1_matrix_does_not_require_threshold_manifest(tmp_path, exp26, speed_config):
    entries = exp26.build_arm_v1_matrix(
        speed_config=speed_config,
        calibration_manifest_path=tmp_path / "missing.json",
        seed_values=[1337],
    )
    assert {entry["arm"] for entry in entries} == set(exp26.ARM_V1_ARMS)


def test_arm_v1_matrix_4_arms_x_3_seeds(tmp_path, exp26, speed_config):
    manifest_path = tmp_path / "manifest.json"
    _write_realistic_manifest(manifest_path)
    entries = exp26.build_arm_v1_matrix(
        speed_config=speed_config,
        calibration_manifest_path=manifest_path,
        seed_values=[1337, 2674, 4011],
    )
    assert len(entries) == 4 * 3
    arms_seen = {e["arm"] for e in entries}
    assert arms_seen == set(exp26.ARM_V1_ARMS)
    assert {e["model_dim"] for e in entries} == {384}


def test_arm_v1_active_uses_learned_commit_authority(tmp_path, exp26, speed_config):
    manifest_path = tmp_path / "manifest.json"
    _write_realistic_manifest(manifest_path)
    entries = exp26.build_arm_v1_matrix(
        speed_config=speed_config,
        calibration_manifest_path=manifest_path,
        seed_values=[1337],
        arms=["arm_d_crct_replay_active_learned"],
    )
    assert len(entries) == 1
    e = entries[0]
    # Manifest thresholds are post-hoc counterfactuals only. The active
    # owner is the learned Full-A action simplex plus GPU3 physics.
    assert "replay_eviction_threshold" not in e
    assert "replay_eviction_peak_preserve_utility_threshold" not in e
    assert "replay_eviction_peak_preserve_sharpness_threshold" not in e
    assert e["replay_eviction_action_agreement_count"] == 1
    assert e["replay_eviction_commit_policy"] == "learned"
    assert e["replay_eviction_commit_online_lr"] == 0.05
    assert "replay_eviction_commit_rule_prior_scale" not in e
    assert e["replay_eviction_mode"] == "active"


def test_arm_v1_control_arm_has_no_crct_or_maintenance(tmp_path, exp26, speed_config):
    manifest_path = tmp_path / "manifest.json"
    _write_realistic_manifest(manifest_path)
    entries = exp26.build_arm_v1_matrix(
        speed_config=speed_config,
        calibration_manifest_path=manifest_path,
        seed_values=[1337],
        arms=["arm_a_fastslow_control"],
    )
    assert len(entries) == 1
    e = entries[0]
    assert e.get("crct_enabled") is not True
    assert e.get("replay_eviction_enabled") is not True
    assert e["fast_slow_enabled"] is True


def test_arm_v1_per_arm_per_seed_trace_paths(tmp_path, exp26, speed_config):
    manifest_path = tmp_path / "manifest.json"
    _write_realistic_manifest(manifest_path)
    entries = exp26.build_arm_v1_matrix(
        speed_config=speed_config,
        calibration_manifest_path=manifest_path,
        seed_values=[1337, 2674],
        arms=["arm_d_crct_replay_active_learned"],
    )
    paths = {e["replay_eviction_trace_path"] for e in entries}
    assert len(paths) == 2  # two seeds, two trace paths
    for p in paths:
        assert "arm_d_crct_replay_active_learned" in p


def test_arm_v1_unknown_arm_raises(tmp_path, exp26, speed_config):
    manifest_path = tmp_path / "manifest.json"
    _write_realistic_manifest(manifest_path)
    with pytest.raises(ValueError, match="unknown arm"):
        exp26.build_arm_v1_matrix(
            speed_config=speed_config,
            calibration_manifest_path=manifest_path,
            arms=["arm_z_does_not_exist"],
        )


def test_arm_v1_headline_ignores_malformed_manifest(tmp_path, exp26, speed_config):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{not valid json")
    entries = exp26.build_arm_v1_matrix(
        speed_config=speed_config,
        calibration_manifest_path=manifest_path,
        seed_values=[1337],
    )
    assert {entry["arm"] for entry in entries} == set(exp26.ARM_V1_ARMS)


def test_load_manifest_missing_required_keys_raises(tmp_path, exp26):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"calibrated_at": "2026-04-28"}))
    with pytest.raises(ValueError, match="missing required keys"):
        exp26.load_manifest(manifest_path)
