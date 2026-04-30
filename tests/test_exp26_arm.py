"""Tests for the fixed Exp26 Adaptive Residual Memory validation canary."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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


def _speed_config() -> dict:
    return {
        "lr": 0.064,
        "world_size": 4,
        "model_dim": 256,
        "seq_len": 512,
        "batch_size": 64,
        "warmup_steps": 10,
        "submit_valid": True,
    }


def test_validation_matrix_is_fixed_two_cell_canary():
    exp26 = _load_module("exp26_validation", EXP26 / "exp26.py")
    entries = exp26.build_validation_matrix(
        speed_config=_speed_config(),
        world_size=4,
        budget_seconds=45.0,
        seed=42,
    )
    assert [entry["arm"] for entry in entries] == list(exp26.VALIDATION_ARMS)
    assert len(entries) == 2
    assert {entry["seed"] for entry in entries} == {42}
    assert {entry["budget_seconds"] for entry in entries} == {45.0}
    assert {entry["model_dim"] for entry in entries} == {384}


def test_validation_control_has_no_memory_sidecar():
    exp26 = _load_module("exp26_validation_control", EXP26 / "exp26.py")
    control = exp26.build_validation_matrix(
        speed_config=_speed_config(),
        seed=42,
    )[0]
    assert control["arm"] == "validation_fastslow_control"
    assert control.get("crct_enabled") is not True
    assert control.get("replay_eviction_enabled") is not True
    assert control["fast_slow_enabled"] is True
    assert "fast_slow_interval" not in control
    assert control["episodic_controller_action_space_enabled"] is True
    assert control["episodic_controller_shared_event_ssm_enabled"] is True
    assert control["episodic_controller_head_readiness"]["consolidation"] > 0.0
    assert control["episodic_controller_head_readiness"]["ema_alpha"] > 0.0
    assert "fast_slow_action_space" in control["episodic_controller_action_trace_path"]


def test_validation_active_is_full_arm_not_crct_only_or_shadow():
    exp26 = _load_module("exp26_validation_active", EXP26 / "exp26.py")
    active = exp26.build_validation_matrix(
        speed_config=_speed_config(),
        seed=42,
    )[1]
    assert active["arm"] == "validation_adaptive_residual_memory"
    assert active["crct_enabled"] is True
    assert active["replay_eviction_enabled"] is True
    assert active["replay_eviction_mode"] == "active"
    assert active["replay_eviction_commit_policy"] == "learned"
    assert active["replay_eviction_scoring_mode"] == "oracle"
    assert active["crct_teacher_score_interval_steps"] == 1
    assert active["compile_full_path"] is True
    assert active["cuda_graph_mode"] == "probe"
    assert active["optimizer_log_a_beta_coupling"] is True
    assert active["optimizer_log_a_beta_ema"] == 0.99
    assert active["optimizer_log_a_beta_min"] == 0.5
    assert active["crct_plasticity_budget_strength"] == 0.25
    assert "enable_controller" not in active
    assert "crct_lambda_controller" not in active
    assert "controller_hidden_dim" not in active
    assert "crct_teacher_param_sync_interval_steps" not in active
    assert "replay_eviction_refresh_proposal_weight_sync_interval_steps" not in active
    assert active["replay_eviction_arm_runtime_enabled"] is True
    assert active["replay_eviction_action_agreement_count"] == 1
    assert active["bucket_prototypes"] is True
    assert active["prototype_dim"] == 64
    assert "shadow" not in active["name"]
    assert "crct_controller" not in active["name"]


def test_validation_active_uses_full_runtime_pipeline_and_isolated_traces():
    exp26 = _load_module("exp26_validation_trace", EXP26 / "exp26.py")
    active = exp26.build_validation_matrix(
        speed_config=_speed_config(),
        seed=42,
    )[1]
    assert (
        active["replay_eviction_arm_runtime_namespace"]
        == "exp26_validation_validation_adaptive_residual_memory_s42"
    )
    assert active["replay_eviction_memory_streams"] == 8
    assert active["replay_eviction_oracle_confirm_top_k"] == 32
    assert active["replay_eviction_oracle_variant_chunk_size"] == 1
    assert active["replay_eviction_refresh_candidate_count"] == 16
    assert active["replay_eviction_refresh_candidate_variant_chunk_size"] == 16
    assert active["replay_eviction_controller_state_dim"] == 32
    assert active["replay_eviction_controller_rank"] == 8
    assert "validation" in active["replay_eviction_trace_path"]
    assert "calibration" not in active["replay_eviction_trace_path"]
    assert "results" not in active["replay_eviction_trace_path"]
    assert "validation" in active["crct_gradient_conflict_trace_path"]


def test_exp26_module_no_longer_exposes_ablation_builders():
    exp26 = _load_module("exp26_validation_no_switches", EXP26 / "exp26.py")
    assert not hasattr(exp26, "build_arm_v1_matrix")
    assert not hasattr(exp26, "build_calibration_matrix")
    assert not hasattr(exp26, "ARM_V1_ARMS")
    assert not hasattr(exp26, "load_manifest")
