"""Tests for the fixed Exp26 Adaptive Residual Memory validation canary."""
from __future__ import annotations

import importlib.util
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
    assert {entry["ssm_delta_rank"] for entry in entries} == {32}


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
    assert active["crct_memory_write_tokens_per_step"] == 32
    assert "crct_async_teacher_pending_batches" not in active
    assert "crct_async_teacher_max_lag_steps" not in active
    assert active["compile_full_path"] is False
    assert active["cuda_graph_mode"] == "none"
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
    assert active["replay_eviction_max_seconds"] == 0.0
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


def test_profile_launcher_can_dry_run_single_arm(tmp_path):
    profile = _load_module("exp26_profile_dry_run", EXP26 / "profile_exp26.py")

    rc = profile.main(
        [
            "--dry-run",
            "--arm",
            "adaptive",
            "--score-stage-timing",
            "--budget",
            "3",
            "--results-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    matrix = (tmp_path / "matrix.json").read_text()
    assert "validation_adaptive_residual_memory" in matrix
    assert "validation_fastslow_control" not in matrix
    assert "crct_score_stage_timing_enabled" in matrix


def test_profile_summary_extracts_transport_and_maintenance_health(tmp_path):
    profile = _load_module("exp26_profile_summary", EXP26 / "profile_exp26.py")
    result_path = tmp_path / "arm.json"
    result_path.write_text(
        """
{
  "config": {"name": "arm", "arm": "validation_adaptive_residual_memory"},
  "train": {
    "steps": 2,
    "elapsed_s": 1.5,
    "aggregate_tokens_per_sec": 123.0,
    "per_gpu_tokens_per_sec": 41.0,
    "final_loss": 3.14,
    "optimizer": {"plasticity_budget": {"lr_multiplier_max": 1.25}},
    "mechanisms": {
      "crct": {
	        "teacher_fail_open": 1,
	        "transport_summary": {
	          "health": {
	            "payloads_used": 2,
	            "payloads_scored": 3,
	            "payloads_served": 4,
	            "payloads_served_approximate": 4,
	            "packet_service_seconds_mean": 0.125,
	            "packet_service_seconds_max": 0.25,
	            "packet_service_source_count_mean": 1.5,
	            "crct_loss_reweight_samples": 2,
	            "crct_loss_reweight_plain_nll_mean": 1.5,
	            "crct_loss_reweight_weighted_nll_mean": 1.45,
	            "crct_loss_reweight_delta_mean": -0.05,
	            "crct_loss_weight_abs_dev_mean": 0.02,
	            "score_stage_timing_enabled": true,
	            "score_stage_samples": 2,
	            "score_stage_encode_off_seconds_sum": 0.5,
	            "score_stage_encode_force_on_seconds_sum": 0.7,
	            "score_stage_nll_off_seconds_sum": 0.2,
	            "score_stage_nll_mem_seconds_sum": 0.3,
	            "score_stage_plasticity_seconds_sum": 0.1,
	            "score_stage_append_memory_seconds_sum": 0.05,
	            "score_stage_peak_allocated_mb_max": 1234.0,
	            "weight_snapshot_published": 4,
	            "weight_snapshot_applied": 5,
            "weight_snapshot_shm_writes": 6,
            "weight_snapshot_shm_reads": 7,
            "weight_snapshot_read_seconds_sum": 0.125,
            "weight_snapshot_read_seconds_max": 0.05,
            "weight_snapshot_read_tensor_count": 51,
            "weight_snapshot_read_bytes": 34957058,
            "memory_rank_request_events_superseded": 13,
            "memory_rank_outer_loop_seconds_sum": 2.0,
            "memory_rank_outer_loop_seconds_max": 0.75,
            "memory_rank_pre_pump_seconds_sum": 0.5,
            "memory_rank_pre_pump_seconds_max": 0.1,
            "memory_rank_replay_seconds_sum": 0.25,
            "memory_rank_replay_seconds_max": 0.05,
            "memory_rank_replay_ticks": 14,
            "memory_rank_replay_probes_ingested": 15,
            "memory_rank_replay_deferred_for_packet_work": 16,
            "memory_rank_replay_deferred_for_backpressure": 17,
            "memory_rank_pump_loop_seconds_sum": 1.25,
            "memory_rank_pump_loop_seconds_max": 0.5,
            "memory_rank_pump_idle_yields": 18,
            "plasticity_packets_received": 8
          },
          "coordinator": {"teacher_shm_request_ring_full_drops": 9},
          "memory": {"teacher_shm_result_ring_full_drops": 10}
        },
        "replay_eviction": {
          "gpu3_starvation_reason": "ok",
          "memory_streams_active": true,
          "arm_runtime": {"jobs_pushed": 11, "jobs_popped": 12}
        }
      }
    }
  }
}
        """.strip()
    )

    summary = profile.summarize_profile(tmp_path)
    row = summary["rows"][0]

    assert row["payloads_used"] == 2
    assert row["payloads_served"] == 4
    assert row["payloads_served_approximate"] == 4
    assert row["packet_service_seconds_mean"] == pytest.approx(0.125)
    assert row["packet_service_seconds_max"] == pytest.approx(0.25)
    assert row["packet_service_source_count_mean"] == pytest.approx(1.5)
    assert row["crct_loss_reweight_samples"] == 2
    assert row["crct_loss_reweight_plain_nll_mean"] == pytest.approx(1.5)
    assert row["crct_loss_reweight_weighted_nll_mean"] == pytest.approx(1.45)
    assert row["crct_loss_reweight_delta_mean"] == pytest.approx(-0.05)
    assert row["crct_loss_weight_abs_dev_mean"] == pytest.approx(0.02)
    assert row["weight_snapshot_shm_writes"] == 6
    assert row["weight_snapshot_shm_reads"] == 7
    assert row["weight_snapshot_read_seconds_sum"] == pytest.approx(0.125)
    assert row["weight_snapshot_read_seconds_max"] == pytest.approx(0.05)
    assert row["weight_snapshot_read_tensor_count"] == 51
    assert row["weight_snapshot_read_bytes"] == 34957058
    assert row["memory_rank_request_events_superseded"] == 13
    assert row["memory_rank_outer_loop_seconds_sum"] == pytest.approx(2.0)
    assert row["memory_rank_outer_loop_seconds_max"] == pytest.approx(0.75)
    assert row["memory_rank_pre_pump_seconds_sum"] == pytest.approx(0.5)
    assert row["memory_rank_pre_pump_seconds_max"] == pytest.approx(0.1)
    assert row["memory_rank_replay_seconds_sum"] == pytest.approx(0.25)
    assert row["memory_rank_replay_seconds_max"] == pytest.approx(0.05)
    assert row["memory_rank_replay_ticks"] == 14
    assert row["memory_rank_replay_probes_ingested"] == 15
    assert row["memory_rank_replay_deferred_for_packet_work"] == 16
    assert row["memory_rank_replay_deferred_for_backpressure"] == 17
    assert row["memory_rank_pump_loop_seconds_sum"] == pytest.approx(1.25)
    assert row["memory_rank_pump_loop_seconds_max"] == pytest.approx(0.5)
    assert row["memory_rank_pump_idle_yields"] == 18
    assert row["plasticity_packets_received"] == 8
    assert row["score_stage_timing_enabled"] is True
    assert row["score_stage_samples"] == 2
    assert row["score_stage_encode_seconds_sum"] == pytest.approx(1.2)
    assert row["score_stage_nll_seconds_sum"] == pytest.approx(0.5)
    assert row["score_stage_peak_allocated_mb_max"] == pytest.approx(1234.0)
    assert row["request_ring_full_drops"] == 9
    assert row["result_ring_full_drops"] == 10
    assert row["maintenance_jobs_pushed"] == 11
    assert row["maintenance_jobs_popped"] == 12
    assert (tmp_path / "profile_summary.json").exists()
