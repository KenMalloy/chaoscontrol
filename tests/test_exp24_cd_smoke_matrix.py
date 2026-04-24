"""Assertions for the cd_first_smoke matrix produced by exp24."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SPEED_CONFIG = {
    "model_size": "10M",
    "seq_len": 512,
    "batch_size_per_rank": 128,
    "stride": 64,
    "chunk_size": 16,
    "precision": "bf16",
    "grad_clip_norm": 1.0,
    "fused_grad_clip": True,
    "artifact_impact": "training_only",
    "train_sampling_mode": "random",
}

SEEDS = [1337]


def _load_exp24():
    path = Path(__file__).resolve().parent.parent / "experiments" / "24_training_time_bundle" / "exp24.py"
    spec = importlib.util.spec_from_file_location("exp24", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_run_exp24():
    path = Path(__file__).resolve().parent.parent / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    spec = importlib.util.spec_from_file_location("run_exp24", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cd_first_smoke_matrix_has_eight_entries_per_seed():
    mod = _load_exp24()
    entries = mod.build_criticality_distillation_first_smoke_matrix(
        speed_config=SPEED_CONFIG, world_size=1, budget_seconds=600.0,
        seed_values=SEEDS,
    )
    assert len(entries) == 8


def test_cd_first_smoke_cells_have_required_arm_names():
    mod = _load_exp24()
    entries = mod.build_criticality_distillation_first_smoke_matrix(
        speed_config=SPEED_CONFIG, world_size=1, budget_seconds=600.0,
        seed_values=SEEDS,
    )
    # Arm name is the middle token of the generated name:
    # exp24_cd_first_smoke_{arm}_s{seed}
    arms = set()
    for entry in entries:
        name = entry["name"]
        arms.add(name.replace("exp24_cd_first_smoke_", "").rsplit("_s", 1)[0])
    expected = {"treatment", "telemetry", "shuffled", "budget_only",
                "hl_short", "hl_long", "H_short", "H_long"}
    assert arms == expected, f"got {arms}"


def test_cd_first_smoke_cells_all_ride_locked_fast_slow_base():
    mod = _load_exp24()
    entries = mod.build_criticality_distillation_first_smoke_matrix(
        speed_config=SPEED_CONFIG, world_size=1, budget_seconds=600.0,
        seed_values=SEEDS,
    )
    for e in entries:
        assert e.get("fast_slow_enabled") is True
        assert e.get("fast_slow_interval") == 64
        assert e.get("fast_slow_alpha") == 0.25


def test_cd_first_smoke_cells_emit_entropy_fused_streaming_cached():
    mod = _load_exp24()
    entries = mod.build_criticality_distillation_first_smoke_matrix(
        speed_config=SPEED_CONFIG, world_size=1, budget_seconds=600.0,
        seed_values=SEEDS,
    )
    for e in entries:
        assert e.get("lm_head_backward_mode") == "fused_streaming_cached"
        assert e.get("lm_head_emit_entropy") is True


def test_cd_first_smoke_cells_all_have_cd_enabled():
    mod = _load_exp24()
    entries = mod.build_criticality_distillation_first_smoke_matrix(
        speed_config=SPEED_CONFIG, world_size=1, budget_seconds=600.0,
        seed_values=SEEDS,
    )
    for e in entries:
        assert e.get("criticality_distill_enabled") is True


def test_cd_first_smoke_cells_emit_rare_bucket_val_ce():
    """All cells must enable rare_bucket_ce so per-bucket val CE is
    emitted — it is the primary falsifier axis for CD vs controls."""
    mod = _load_exp24()
    entries = mod.build_criticality_distillation_first_smoke_matrix(
        speed_config=SPEED_CONFIG, world_size=1, budget_seconds=600.0,
        seed_values=SEEDS,
    )
    for e in entries:
        assert e.get("rare_bucket_ce_enabled") is True
        assert int(e.get("rare_bucket_ce_num_buckets", 0)) == 4


def test_cd_first_smoke_falsifier_cell_flags():
    mod = _load_exp24()
    entries = mod.build_criticality_distillation_first_smoke_matrix(
        speed_config=SPEED_CONFIG, world_size=1, budget_seconds=600.0,
        seed_values=SEEDS,
    )
    by_arm = {
        e["name"].replace("exp24_cd_first_smoke_", "").rsplit("_s", 1)[0]: e
        for e in entries
    }
    assert by_arm["telemetry"]["criticality_distill_weight"] == 0.0
    assert by_arm["shuffled"]["criticality_distill_score_permute_before_topk"] is True
    assert by_arm["budget_only"]["criticality_distill_fixed_random_seats"] is True
    for sens in ("hl_short", "hl_long", "H_short", "H_long"):
        assert by_arm[sens]["criticality_distill_score_permute_before_topk"] is False
        assert by_arm[sens]["criticality_distill_fixed_random_seats"] is False


def test_cd_first_smoke_sensitivity_cells_have_distinct_knob():
    mod = _load_exp24()
    entries = mod.build_criticality_distillation_first_smoke_matrix(
        speed_config=SPEED_CONFIG, world_size=1, budget_seconds=600.0,
        seed_values=SEEDS,
    )
    by_arm = {
        e["name"].replace("exp24_cd_first_smoke_", "").rsplit("_s", 1)[0]: e
        for e in entries
    }
    assert by_arm["hl_short"]["criticality_distill_trace_half_life_steps"] == 128.0
    assert by_arm["hl_long"]["criticality_distill_trace_half_life_steps"] == 512.0
    assert by_arm["H_short"]["criticality_distill_horizon_H"] == 8
    assert by_arm["H_long"]["criticality_distill_horizon_H"] == 32


def test_run_exp24_defaults_cd_first_smoke_to_world_size_1():
    mod = _load_run_exp24()
    assert mod._default_world_size_for_matrix("cd_first_smoke") == 1
