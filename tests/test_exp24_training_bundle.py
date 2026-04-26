"""Tests for Exp24 training-time bundle matrix helpers."""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
EXP24_PATH = REPO / "experiments" / "24_training_time_bundle" / "exp24.py"
RUN_EXP24_PATH = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
LAUNCH_PATH = REPO / "experiments" / "23_fast_path" / "launch.py"
EXP24_BASE_CONFIG_PATH = (
    REPO / "experiments" / "24_training_time_bundle" / "configs" / "exp24_base.yaml"
)


def _load_exp24():
    spec = importlib.util.spec_from_file_location("exp24_training_bundle", EXP24_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_run_exp24():
    spec = importlib.util.spec_from_file_location("run_exp24_for_tests", RUN_EXP24_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_launch():
    spec = importlib.util.spec_from_file_location("exp23_launch_for_tests", LAUNCH_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ring0_control_matrix_uses_seed_ladder_and_600s_budget():
    mod = _load_exp24()

    entries = mod.build_ring0_control_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=8,
    )

    assert [entry["seed"] for entry in entries] == [1337, 2674, 4011]
    assert [entry["name"] for entry in entries] == [
        "exp24_ring0_control_s1337",
        "exp24_ring0_control_s2674",
        "exp24_ring0_control_s4011",
    ]
    for entry in entries:
        assert entry["exp24_phase"] == "ring0"
        assert entry["exp24_mechanism"] == "control"
        assert entry["budget_seconds"] == 600.0
        assert entry["world_size"] == 8
        assert entry["train_sampling_mode"] == "random"
        assert entry["artifact_impact"] == "artifact_changes_weights_only"
        assert entry["submit_valid"] is True


def test_phase_a_sampling_matrix_is_mechanism_free():
    mod = _load_exp24()

    entries = mod.build_phase_a_sampling_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        seeds=[1337],
        world_size=8,
    )

    assert [entry["train_sampling_mode"] for entry in entries] == [
        "random",
        "sequential_epoch",
        "shuffled_epoch",
    ]
    assert [entry["name"] for entry in entries] == [
        "exp24_phaseA_random_s1337",
        "exp24_phaseA_sequential_epoch_s1337",
        "exp24_phaseA_shuffled_epoch_s1337",
    ]
    assert all(entry["exp24_mechanism"] == "sampling_policy" for entry in entries)
    assert all(entry["fast_slow_enabled"] is False for entry in entries)
    assert all(entry["spectral_reg_lambda_dead"] == 0.0 for entry in entries)
    assert all(entry["predictive_aux_weight"] == 0.0 for entry in entries)


def test_control_noise_summary_uses_sample_std_and_min_max():
    mod = _load_exp24()
    results = [
        {
            "config": {"seed": 1337},
            "eval": {"bpb": 1.05},
            "train": {"elapsed_s": 599.0, "aggregate_tokens_per_sec": 41.0},
        },
        {
            "config": {"seed": 2674},
            "eval": {"bpb": 1.07},
            "train": {"elapsed_s": 598.0, "aggregate_tokens_per_sec": 43.0},
        },
        {
            "config": {"seed": 4011},
            "eval": {"bpb": 1.06},
            "train": {"elapsed_s": 597.0, "aggregate_tokens_per_sec": 42.0},
        },
    ]

    summary = mod.summarize_control_noise(results)

    assert summary["seeds"] == [1337, 2674, 4011]
    assert summary["count"] == 3
    assert math.isclose(summary["bpb_mean"], 1.06)
    assert math.isclose(summary["bpb_sample_std"], 0.01)
    assert summary["bpb_min"] == 1.05
    assert summary["bpb_max"] == 1.07
    assert summary["tokens_per_sec_mean"] == 42.0


def test_build_semantic_overhead_gate_matrix_has_muon_and_semantic_rows():
    mod = _load_exp24()

    entries = mod.build_semantic_overhead_gate_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        seed=1337,
        world_size=1,
        budget_seconds=90.0,
    )

    assert len(entries) == 2
    assert [entry["name"] for entry in entries] == [
        "exp24_smoke_semantic_gate_muon_s1337",
        "exp24_smoke_semantic_gate_semantic_s1337",
    ]
    assert [entry["optimizer"] for entry in entries] == ["muon", "semantic"]
    for entry in entries:
        assert entry["world_size"] == 1
        assert entry["budget_seconds"] == 90.0
        assert entry["semantic_overhead_gate"] == 0.08
        assert entry["exp24_phase"] == "smoke"
        assert entry["exp24_mechanism"] == "semantic_optimizer_gate"
        assert entry["artifact_impact"] == "artifact_changes_weights_only"


def test_build_scopt_overhead_gate_matrix_has_muon_and_scopt_rows():
    mod = _load_exp24()

    entries = mod.build_scopt_overhead_gate_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        seed=1337,
        world_size=1,
        budget_seconds=180.0,
    )

    assert len(entries) == 2
    assert [entry["name"] for entry in entries] == [
        "exp24_smoke_scopt_gate_muon_s1337",
        "exp24_smoke_scopt_gate_scopt_s1337",
    ]
    assert [entry["optimizer"] for entry in entries] == ["muon", "scopt"]
    for entry in entries:
        assert entry["world_size"] == 1
        assert entry["budget_seconds"] == 180.0
        assert entry["batch_size"] == 512, (
            "ScOpt's unfused LM-head path OOMs at bs=1024 / V=16384; "
            "smoke must use the VRAM-safe bs=512."
        )
        assert entry["exp24_phase"] == "smoke"
        assert entry["exp24_mechanism"] == "scopt_overhead_gate"
        assert entry["artifact_impact"] == "artifact_changes_weights_only"
        assert entry["optimizer_param_grouping"] == "ssm_three_group", (
            "ScOpt overhead gate must run against the SSM-aware baseline; "
            "flat grouping leaves log_a fighting wd=0.01 while ScOpt's "
            "recurrence scarcity tries to move it."
        )
        assert entry["optimizer_dynamics_lr_mul"] == 0.1

    scopt_entry = entries[1]
    assert scopt_entry["scopt_warmup_steps"] == 200
    assert scopt_entry["scopt_split_interval"] == 4
    assert scopt_entry["scopt_trace_interval_steps"] == 64
    assert scopt_entry["scopt_layer_index"] == 0
    # Muon entry must NOT carry ScOpt-only knobs.
    muon_entry = entries[0]
    assert "scopt_warmup_steps" not in muon_entry


def test_build_scopt_overhead_gate_matrix_preserves_chunk_size():
    """speed_config knobs (batch_size override, chunk_size) flow through
    _base_entry. Only batch_size gets overridden by the gate; chunk_size
    should stay whatever the user's Exp23 base config says.
    """
    mod = _load_exp24()

    entries = mod.build_scopt_overhead_gate_matrix(
        speed_config={
            "batch_size": 1024,
            "chunk_size": 128,
            "seq_len": 512,
            "model_dim": 256,
            "num_layers": 4,
            "vocab_size": 16384,
        },
        seed=1337,
    )
    for entry in entries:
        assert entry["chunk_size"] == 128
        assert entry["seq_len"] == 512
        assert entry["vocab_size"] == 16384


def test_first_wave_mechanism_matrix_names_and_tags():
    mod = _load_exp24()

    entries = mod.build_first_wave_mechanism_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        seed_values=[1337],
        world_size=8,
    )

    names = [entry["name"] for entry in entries]
    assert names == [
        "exp24_fastslow_i32_a050_s1337",
        "exp24_spectral_dead1e-04_sticky1e-04_s1337",
        "exp24_predictive_h4_w010_s1337",
        "exp24_dreamworld_c4_i4_w025_s1337",
        "exp24_fastslow_i32_a050_dreamworld_c8_i8_w025_sub128_s1337",
        "exp24_fastslow_i32_a050_dreamworld_eventsleep_r110_p005_sub128_s1337",
    ]

    mechanisms = {entry["exp24_mechanism"] for entry in entries}
    assert mechanisms == {
        "fast_slow",
        "spectral",
        "predictive_aux",
        "dreamworld",
        "fast_slow_dreamworld",
        "fast_slow_dreamworld_event_sleep",
    }

    fastslow = next(
        entry for entry in entries if entry["name"] == "exp24_fastslow_i32_a050_s1337"
    )
    assert fastslow["fast_slow_enabled"] is True
    assert fastslow["fast_slow_interval"] == 32
    assert fastslow["fast_slow_alpha"] == 0.5
    assert fastslow["fast_slow_eval_copy"] == "slow"
    assert fastslow["artifact_impact"] == "artifact_training_only"

    spectral = next(
        entry
        for entry in entries
        if entry["name"] == "exp24_spectral_dead1e-04_sticky1e-04_s1337"
    )
    assert spectral["spectral_reg_lambda_dead"] == 0.0001
    assert spectral["spectral_reg_lambda_sticky"] == 0.0001
    assert spectral["artifact_impact"] == "artifact_changes_weights_only"

    predictive = next(
        entry for entry in entries if entry["name"] == "exp24_predictive_h4_w010_s1337"
    )
    assert predictive["predictive_aux_horizon"] == 4
    assert predictive["predictive_aux_weight"] == 0.1
    assert predictive["artifact_impact"] == "artifact_training_only"

    dream = next(
        entry for entry in entries if entry["name"] == "exp24_dreamworld_c4_i4_w025_s1337"
    )
    assert dream["dreamworld_enabled"] is True
    assert dream["dreamworld_cache_interval"] == 4
    assert dream["dreamworld_interval"] == 4
    assert dream["dreamworld_weight"] == 0.25
    assert dream["artifact_impact"] == "artifact_training_only"

    stack = next(
        entry
        for entry in entries
        if entry["name"]
        == "exp24_fastslow_i32_a050_dreamworld_c8_i8_w025_sub128_s1337"
    )
    assert stack["exp24_mechanism"] == "fast_slow_dreamworld"
    assert stack["fast_slow_enabled"] is True
    assert stack["fast_slow_interval"] == 32
    assert stack["fast_slow_alpha"] == 0.5
    assert stack["fast_slow_eval_copy"] == "slow"
    assert stack["dreamworld_enabled"] is True
    assert stack["dreamworld_cache_interval"] == 8
    assert stack["dreamworld_interval"] == 8
    assert stack["dreamworld_weight"] == 0.25
    assert stack["dreamworld_replay_batch_size"] == 128
    assert stack["artifact_impact"] == "artifact_training_only"

    event_stack = next(
        entry
        for entry in entries
        if entry["name"]
        == "exp24_fastslow_i32_a050_dreamworld_eventsleep_r110_p005_sub128_s1337"
    )
    assert event_stack["exp24_mechanism"] == "fast_slow_dreamworld_event_sleep"
    assert event_stack["fast_slow_enabled"] is True
    assert event_stack["dreamworld_enabled"] is True
    assert event_stack["event_sleep_enabled"] is True
    assert event_stack["event_sleep_loss_ratio"] == 1.10
    assert event_stack["event_sleep_pressure_threshold"] == 0.05
    assert event_stack["event_sleep_min_interval"] == 8
    assert event_stack["dreamworld_replay_batch_size"] == 128
    assert event_stack["artifact_impact"] == "artifact_training_only"


def test_fastslow_dreamworld_matrix_is_stack_only():
    mod = _load_exp24()

    entries = mod.build_fastslow_dreamworld_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        seed_values=[1337, 2674],
        world_size=8,
    )

    assert [entry["name"] for entry in entries] == [
        "exp24_fastslow_i32_a050_dreamworld_c8_i8_w025_sub128_s1337",
        "exp24_fastslow_i32_a050_dreamworld_c8_i8_w025_sub128_s2674",
    ]
    assert all(entry["exp24_mechanism"] == "fast_slow_dreamworld" for entry in entries)
    assert all(entry["fast_slow_enabled"] is True for entry in entries)
    assert all(entry["dreamworld_enabled"] is True for entry in entries)


def test_build_phase0_dreamworld_sweep_shape_and_knobs():
    mod = _load_exp24()

    entries = mod.build_phase0_dreamworld_sweep(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
        seed_values=(1337,),
    )

    assert len(entries) == 9
    expected = {
        f"exp24_phase0_fs_i32a050_dw_c{i}i{i}_w{int(w * 100):03d}_s1337"
        for i in (4, 8, 16)
        for w in (0.10, 0.25, 0.50)
    }
    assert {entry["name"] for entry in entries} == expected
    assert {entry["world_size"] for entry in entries} == {4}
    assert {entry["budget_seconds"] for entry in entries} == {600.0}
    for entry in entries:
        assert entry["exp24_phase"] == "phase0"
        assert entry["exp24_mechanism"] == "fast_slow_dreamworld"
        assert entry["artifact_impact"] == "artifact_training_only"
        assert entry["fast_slow_enabled"] is True
        assert entry["fast_slow_interval"] == 32
        assert entry["fast_slow_alpha"] == 0.50
        assert entry["fast_slow_eval_copy"] == "slow"
        assert entry["dreamworld_enabled"] is True
        assert entry["dreamworld_cache_interval"] == entry["dreamworld_interval"]
        assert entry["dreamworld_replay_batch_size"] == 128
        assert entry["dreamworld_prefix_tokens"] == 128
        assert entry["dreamworld_replay_tokens"] == 64
        assert entry["dreamworld_buffer_size"] == 16
        assert entry["dreamworld_min_size"] == 2
        assert entry["dreamworld_max_age_steps"] == 256


def test_build_phase0_fastslow_sweep_shape_and_knobs():
    mod = _load_exp24()

    entries = mod.build_phase0_fastslow_sweep(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
        seed_values=(1337,),
    )

    assert len(entries) == 18
    assert {entry["seed"] for entry in entries} == {1337}
    assert {entry["world_size"] for entry in entries} == {4}
    assert {
        (entry["fast_slow_interval"], entry["fast_slow_alpha"])
        for entry in entries
    } == {
        (16, 0.25),
        (16, 0.50),
        (32, 0.25),
        (32, 0.50),
        (64, 0.25),
        (64, 0.50),
    }
    dw_settings = {
        (
            entry["dreamworld_cache_interval"],
            entry["dreamworld_interval"],
            entry["dreamworld_weight"],
        )
        for entry in entries
    }
    assert dw_settings == {
        (16, 16, 0.10),
        (16, 16, 0.25),
        (8, 8, 0.10),
    }
    expected_names = {
        (
            f"exp24_phase0_fs_i{fs_interval}a{int(fs_alpha * 100):03d}_"
            f"dw_c{dw_cache}i{dw_interval}_w{int(dw_weight * 100):03d}_s1337"
        )
        for dw_cache, dw_interval, dw_weight in dw_settings
        for fs_interval in (16, 32, 64)
        for fs_alpha in (0.25, 0.50)
    }
    assert {entry["name"] for entry in entries} == expected_names
    for entry in entries:
        assert entry["name"].startswith("exp24_phase0_fs_i")
        assert entry["exp24_phase"] == "phase0"
        assert entry["exp24_mechanism"] == "fast_slow_dreamworld"
        assert entry["artifact_impact"] == "artifact_training_only"
        assert entry["fast_slow_enabled"] is True
        assert entry["fast_slow_eval_copy"] == "slow"
        assert entry["dreamworld_enabled"] is True
        assert entry["dreamworld_replay_batch_size"] == 128


def test_build_phase0_confirm_shape_and_knobs():
    mod = _load_exp24()

    entries = mod.build_phase0_confirm(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
    )

    assert len(entries) == 6
    assert {entry["seed"] for entry in entries} == {1337, 2674, 4011}
    assert {entry["world_size"] for entry in entries} == {4}
    assert {entry["budget_seconds"] for entry in entries} == {600.0}
    confirm_labels = {
        entry["name"].rsplit("_s", 1)[0]
        for entry in entries
    }
    assert confirm_labels == {
        "exp24_phase0_confirm_A_fs_i32a025_dw_c16i16_w010",
        "exp24_phase0_confirm_B_fs_i64a025_dw_c16i16_w010",
    }
    assert {
        (
            entry["fast_slow_interval"],
            entry["fast_slow_alpha"],
            entry["dreamworld_cache_interval"],
            entry["dreamworld_interval"],
            entry["dreamworld_weight"],
        )
        for entry in entries
    } == {
        (32, 0.25, 16, 16, 0.10),
        (64, 0.25, 16, 16, 0.10),
    }
    for entry in entries:
        assert entry["name"].startswith("exp24_phase0_confirm_")
        assert entry["exp24_phase"] == "phase0"
        assert entry["exp24_mechanism"] == "fast_slow_dreamworld"
        assert entry["artifact_impact"] == "artifact_training_only"
        assert entry["fast_slow_enabled"] is True
        assert entry["fast_slow_eval_copy"] == "slow"
        assert entry["dreamworld_enabled"] is True
        assert entry["dreamworld_replay_batch_size"] == 128


def test_build_phase0_fastslow_only_control_matches_locked_base_without_dreamworld():
    mod = _load_exp24()

    entries = mod.build_phase0_fastslow_only_control(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
    )

    assert len(entries) == 3
    assert [entry["seed"] for entry in entries] == [1337, 2674, 4011]
    assert [entry["name"] for entry in entries] == [
        "exp24_phase0_control_fastslow_only_i64a025_s1337",
        "exp24_phase0_control_fastslow_only_i64a025_s2674",
        "exp24_phase0_control_fastslow_only_i64a025_s4011",
    ]
    assert {entry["world_size"] for entry in entries} == {4}
    assert {entry["budget_seconds"] for entry in entries} == {600.0}
    for entry in entries:
        assert entry["exp24_phase"] == "phase0"
        assert entry["exp24_mechanism"] == "fast_slow"
        assert entry["artifact_impact"] == "artifact_training_only"
        assert entry["fast_slow_enabled"] is True
        assert entry["fast_slow_interval"] == 64
        assert entry["fast_slow_alpha"] == 0.25
        assert entry["fast_slow_eval_copy"] == "slow"
        assert entry["dreamworld_enabled"] is False
        assert entry["dreamworld_cache_interval"] == 0
        assert entry["dreamworld_interval"] == 0
        assert entry["dreamworld_weight"] == 0.0
        assert entry["dreamworld_replay_batch_size"] == 0


def test_episodic_dw_curation_v1_matrix_shape():
    """Phase 3 falsifier matrix: 4 arms x 3 seeds, all topologically matched.

    Per ``docs/plans/2026-04-25-memory-aware-optimizer-plan.md`` Task 3.3:
    the only difference between arms is the replay-candidate-selection
    mechanism. World size, budget, batch size, model dims, fast/slow
    recipe, and the Dreamworld replay knobs (cache_interval, interval,
    replay_batch_size, prefix/replay tokens, buffer/min/max-age) are
    identical across all four arms.

    Per-arm differences allowed:
      - Arm A:   episodic_enabled=False (replay reads online buffer)
      - Arm B:   episodic_enabled=True,  controller_query_mode="cosine_utility_weighted"
      - Arm B':  episodic_enabled=True,  controller_query_mode="pressure_only"
      - Arm C:   episodic_enabled=True,  dreamworld_weight=0.0 (topology-only)
    """
    mod = _load_exp24()

    entries = mod.build_episodic_dw_curation_v1_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
    )

    assert len(entries) == 12
    assert {entry["seed"] for entry in entries} == {1337, 2674, 4011}

    expected_arms = (
        "arm_a_uncurated",
        "arm_b_cosine_utility",
        "arm_bp_pressure_only",
        "arm_c_no_dw",
    )
    expected_names = {
        f"exp24_phase3_episodic_dw_curation_v1_{arm}_s{seed}"
        for arm in expected_arms
        for seed in (1337, 2674, 4011)
    }
    assert {entry["name"] for entry in entries} == expected_names

    for entry in entries:
        assert entry["exp24_phase"] == "phase3"
        assert entry["exp24_mechanism"] == "episodic_dw_curation_v1"
        assert entry["world_size"] == 4
        assert entry["budget_seconds"] == 600.0
        assert entry["artifact_impact"] == "artifact_training_only"
        # Locked fast/slow recipe (shared with phase0_fastslow_only_control).
        assert entry["fast_slow_enabled"] is True
        assert entry["fast_slow_interval"] == 64
        assert entry["fast_slow_alpha"] == 0.25
        assert entry["fast_slow_eval_copy"] == "slow"
        # Locked Dreamworld replay topology — identical knobs in all 4 arms
        # so the only difference is candidate selection / replay weight.
        assert entry["dreamworld_enabled"] is True, (
            "Topology-equivalence: replay backward must fire in all arms; "
            "Arm C zeroes dreamworld_weight instead of disabling DW."
        )
        assert entry["dreamworld_cache_interval"] == 16
        assert entry["dreamworld_interval"] == 16
        assert entry["dreamworld_replay_batch_size"] == 128
        assert entry["dreamworld_prefix_tokens"] == 128
        assert entry["dreamworld_replay_tokens"] == 64
        assert entry["dreamworld_buffer_size"] == 16
        assert entry["dreamworld_min_size"] == 2
        assert entry["dreamworld_max_age_steps"] == 256

    by_arm: dict[str, list[dict]] = {arm: [] for arm in expected_arms}
    for entry in entries:
        # Recover arm tag from the name suffix (between v1_ and _s<seed>).
        suffix = entry["name"].split("episodic_dw_curation_v1_", 1)[1]
        arm_tag = suffix.rsplit("_s", 1)[0]
        by_arm[arm_tag].append(entry)
    assert all(len(rows) == 3 for rows in by_arm.values())

    for entry in by_arm["arm_a_uncurated"]:
        assert entry["episodic_enabled"] is False
        assert entry["dreamworld_weight"] == 0.10
        # Arm A's replay reads the online buffer; controller_query_mode is
        # not load-bearing here. Omitted to keep the config surface honest.
        assert "controller_query_mode" not in entry
        assert "controller_query_enabled" not in entry

    for entry in by_arm["arm_b_cosine_utility"]:
        assert entry["episodic_enabled"] is True
        assert entry["dreamworld_weight"] == 0.10
        assert entry["controller_query_mode"] == "cosine_utility_weighted"
        # The controller-query gate must be on so the queue actually fills
        # with retrieval candidates the future controller will drain.
        assert entry["controller_query_enabled"] is True

    for entry in by_arm["arm_bp_pressure_only"]:
        assert entry["episodic_enabled"] is True
        assert entry["dreamworld_weight"] == 0.10
        assert entry["controller_query_mode"] == "pressure_only"
        assert entry["controller_query_enabled"] is True

    for entry in by_arm["arm_c_no_dw"]:
        assert entry["episodic_enabled"] is True
        assert entry["dreamworld_weight"] == 0.0, (
            "Arm C zeroes the replay weight to establish a 3+1-topology "
            "baseline without any DW signal at all."
        )
        # Arm C's controller mode is irrelevant — replay grad is zeroed
        # before it lands in any param.grad. Omit to avoid a silent claim.
        assert "controller_query_mode" not in entry
        assert "controller_query_enabled" not in entry


def test_episodic_dw_curation_v1_matrix_supports_extra_seeds_for_sigma_escalation():
    """Decision 0.5: if sigma(rare-bucket delta) on Arm B > 0.008 bpb across
    the 3 default seeds, add 3 more seeds on Arms A/B/B' (NOT Arm C — the
    topology baseline doesn't need extra seeds). The matrix builder must
    accept both a custom seed list AND an arm filter so the escalation is a
    one-line follow-up call.
    """
    mod = _load_exp24()

    # Default behavior — all four arms, custom seed set.
    extra_all = mod.build_episodic_dw_curation_v1_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
        seed_values=[5012, 7331, 9183],
    )
    assert len(extra_all) == 12
    assert {entry["seed"] for entry in extra_all} == {5012, 7331, 9183}

    # Decision 0.5 escalation shape: A/B/B' only, 3 seeds = 9 cells.
    escalated = mod.build_episodic_dw_curation_v1_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
        seed_values=mod.EPISODIC_DW_CURATION_V1_ESCALATION_SEEDS,
        arms=mod.EPISODIC_DW_CURATION_V1_ESCALATION_ARMS,
    )
    assert len(escalated) == 9
    arms_seen = {
        next(
            arm for arm in mod.EPISODIC_DW_CURATION_V1_ARMS if arm in entry["name"]
        )
        for entry in escalated
    }
    assert arms_seen == {
        "arm_a_uncurated",
        "arm_b_cosine_utility",
        "arm_bp_pressure_only",
    }
    # No Arm C cells in the escalation matrix.
    assert all("arm_c_no_dw" not in entry["name"] for entry in escalated)


def test_episodic_dw_curation_v1_matrix_rejects_unknown_arm_name():
    """Typo defense: passing an arm name that isn't in the canonical set
    raises ValueError with the allowed names listed.
    """
    mod = _load_exp24()

    try:
        mod.build_episodic_dw_curation_v1_matrix(
            speed_config={"batch_size": 1024, "chunk_size": 64},
            world_size=4,
            budget_seconds=600.0,
            arms=("arm_b_cosine_utility", "arm_typo"),
        )
    except ValueError as exc:
        assert "arm_typo" in str(exc)
        assert "arm_a_uncurated" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown arm name")


def test_run_exp24_cli_episodic_dw_curation_v1_dry_run(tmp_path):
    script = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    output_dir = tmp_path / "exp24-episodic-dw-curation-v1-dryrun"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            "episodic_dw_curation_v1",
            "--dry-run",
            "--limit",
            "4",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "matrix=episodic_dw_curation_v1" in stdout
    assert "world_size=4" in stdout
    assert "exp24_phase3_episodic_dw_curation_v1_arm_a_uncurated_s1337" in stdout
    assert "exp24_phase3_episodic_dw_curation_v1_arm_b_cosine_utility_s1337" in stdout
    assert '"exp24_mechanism": "episodic_dw_curation_v1"' in stdout
    assert '"--nproc_per_node=4"' in stdout


def test_exp24_base_config_matches_fastslow_only_lock():
    cfg = yaml.safe_load(EXP24_BASE_CONFIG_PATH.read_text())

    assert cfg["name"] == "exp24_base"
    assert cfg["exp24_mechanism"] == "fast_slow"
    assert cfg["artifact_impact"] == "artifact_training_only"
    assert cfg["world_size"] == 4
    assert cfg["budget_seconds"] == 600.0
    assert cfg["fast_slow_enabled"] is True
    assert cfg["fast_slow_interval"] == 64
    assert cfg["fast_slow_alpha"] == 0.25
    assert cfg["fast_slow_eval_copy"] == "slow"
    assert cfg["dreamworld_enabled"] is False
    assert cfg["dreamworld_cache_interval"] == 0
    assert cfg["dreamworld_interval"] == 0
    assert cfg["dreamworld_weight"] == 0.0
    assert cfg["dreamworld_replay_batch_size"] == 0
    assert cfg["event_sleep_enabled"] is False
    assert cfg["event_sleep_weight"] == 0.0


def test_run_exp24_cli_dry_run_prints_first_wave_plan(tmp_path):
    script = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    output_dir = tmp_path / "exp24-dryrun"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            "first_wave",
            "--seeds",
            "1337",
            "--dry-run",
            "--limit",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "matrix=first_wave" in stdout
    assert "exp24_fastslow_i32_a050_s1337" in stdout
    assert "exp24_spectral_dead1e-04_sticky1e-04_s1337" in stdout
    assert '"exp24_mechanism": "fast_slow"' in stdout
    assert '"exp24_mechanism": "spectral"' in stdout


def test_run_exp24_cli_ring0_defaults_to_control_seed_ladder(tmp_path):
    script = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    output_dir = tmp_path / "exp24-ring0-dryrun"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            "ring0_control",
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "entries=3" in stdout
    assert "exp24_ring0_control_s1337" in stdout
    assert "exp24_ring0_control_s2674" in stdout
    assert "exp24_ring0_control_s4011" in stdout


def test_run_exp24_cli_phase0_fastslow_only_control_accepts_locked_base_config(tmp_path):
    script = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    output_dir = tmp_path / "exp24-phase0-fastslow-only-dryrun"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            "phase0_fastslow_only_control",
            "--config",
            str(EXP24_BASE_CONFIG_PATH),
            "--dry-run",
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "matrix=phase0_fastslow_only_control" in stdout
    assert "world_size=4" in stdout
    assert "exp24_phase0_control_fastslow_only_i64a025_s1337" in stdout
    assert '"exp24_mechanism": "fast_slow"' in stdout
    assert '"fast_slow_interval": 64' in stdout
    assert '"dreamworld_enabled": false' in stdout
    assert '"event_sleep_enabled": false' in stdout
    assert '"--nproc_per_node=4"' in stdout


def test_run_exp24_cli_semantic_gate_defaults_to_cheap_smoke(tmp_path):
    script = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    output_dir = tmp_path / "exp24-semantic-gate-dryrun"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            "semantic_overhead_gate",
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "matrix=semantic_overhead_gate" in stdout
    assert "world_size=1" in stdout
    assert '"budget_seconds": 90.0' in stdout
    assert "--nproc_per_node=1" in stdout
    assert '"--budget",\n      "90.0"' in stdout


def test_run_exp24_cli_scopt_gate_defaults_to_180s_bs512():
    """``--matrix scopt_overhead_gate`` without explicit budget/batch
    should pick 180s / bs=512 / world_size=1, enough steps to clear
    ScOpt's warmup_steps=200 and read a Tier 0 probe trace.
    """
    import tempfile

    script = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "exp24-scopt-gate-dryrun"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--matrix",
                "scopt_overhead_gate",
                "--dry-run",
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "matrix=scopt_overhead_gate" in stdout
    assert "world_size=1" in stdout
    assert '"budget_seconds": 180.0' in stdout
    assert '"batch_size": 512' in stdout
    assert '"optimizer_param_grouping": "ssm_three_group"' in stdout
    # Guard: dynamics_lr_mul must propagate through _base_entry or the
    # Muon/ScOpt constructors silently fall back to base_lr everywhere.
    assert '"optimizer_dynamics_lr_mul": 0.1' in stdout
    # Verify both optimizer rows present.
    assert '"optimizer": "muon"' in stdout
    assert '"optimizer": "scopt"' in stdout


def test_run_exp24_cli_fastslow_dreamworld_matrix_is_stack_only(tmp_path):
    script = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    output_dir = tmp_path / "exp24-fastslow-dreamworld-dryrun"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            "fastslow_dreamworld",
            "--seeds",
            "1337",
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "matrix=fastslow_dreamworld" in stdout
    assert "entries=1" in stdout
    assert "exp24_fastslow_i32_a050_dreamworld_c8_i8_w025_sub128_s1337" in stdout
    assert '"exp24_mechanism": "fast_slow_dreamworld"' in stdout


def test_run_exp24_full_val_defaults_checkpoint_dir(tmp_path):
    script = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    output_dir = tmp_path / "exp24-full-val-dryrun"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            "fastslow_dreamworld",
            "--seeds",
            "1337",
            "--dry-run",
            "--full-val-score",
            "--world-size",
            "4",
            "--output-dir",
            str(output_dir),
            "--val-cache-dir",
            str(tmp_path / "val-cache"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "[exp24] full-val-score enabled" in stdout
    assert str(output_dir / "checkpoints") in stdout
    assert "run_exp20_full_val_score.py" in stdout
    assert "--nproc_per_node=4" in stdout


def test_score_full_val_builds_expected_cmd(tmp_path):
    mod = _load_run_exp24()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    name = "exp24_fastslow_i32_a050_s1337"
    (checkpoint_dir / f"{name}.pt").write_bytes(b"fake checkpoint")

    commands = mod._score_full_val(
        entries=[{"name": name}],
        checkpoint_dir=checkpoint_dir,
        results_dir=tmp_path,
        world_size=4,
        cache_dir=tmp_path / "val-cache",
        budget_seconds=600.0,
        dry_run=True,
    )

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=4" in cmd
    rdzv_endpoint = next(arg for arg in cmd if arg.startswith("--rdzv-endpoint="))
    assert rdzv_endpoint.startswith("--rdzv-endpoint=localhost:")
    assert "--rdzv-backend=c10d" in cmd
    rdzv_id = next(arg for arg in cmd if arg.startswith("--rdzv-id="))
    assert rdzv_id.startswith(f"--rdzv-id=score_{name}_")
    assert str(REPO / "scripts" / "run_exp20_full_val_score.py") in cmd
    assert cmd[cmd.index("--cache-dir") + 1] == str(tmp_path / "val-cache")
    assert cmd[cmd.index("--checkpoint-path") + 1] == str(checkpoint_dir / f"{name}.pt")
    assert cmd[cmd.index("--output-path") + 1] == str(tmp_path / "full_val" / f"{name}.jsonl")
    assert cmd[cmd.index("--summary-path") + 1] == str(
        tmp_path / "full_val" / f"{name}.summary.json"
    )
    assert cmd[cmd.index("--chunk-size") + 1] == "256"
    assert cmd[cmd.index("--budget-seconds") + 1] == "600.0"
    assert cmd[cmd.index("--doc-batch-size") + 1] == "4096"
    assert cmd[cmd.index("--max-forward-tokens") + 1] == "auto"
    assert "--score-boundary-targets" in cmd
    assert cmd[cmd.index("--doc-packing") + 1] == "chunk_count_tail"


def test_score_full_val_dry_run_does_not_bind_port(tmp_path, monkeypatch):
    """Sandboxed CI / offline dev machines can't bind localhost sockets.
    Dry-run must render a command without calling pick_free_port, which
    otherwise opens a socket to discover a free port.
    """
    mod = _load_run_exp24()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    name = "exp24_dry_run_no_port_s1337"
    (checkpoint_dir / f"{name}.pt").write_bytes(b"fake checkpoint")

    def _raising_pick_free_port():
        raise AssertionError(
            "pick_free_port called during dry-run — would bind a socket"
        )

    monkeypatch.setattr(mod, "pick_free_port", _raising_pick_free_port)

    commands = mod._score_full_val(
        entries=[{"name": name}],
        checkpoint_dir=checkpoint_dir,
        results_dir=tmp_path,
        world_size=4,
        cache_dir=tmp_path / "val-cache",
        budget_seconds=600.0,
        dry_run=True,
    )
    assert len(commands) == 1
    # Sentinel shows up in the rendered command, no real port bound.
    rdzv_endpoint = next(
        arg for arg in commands[0] if arg.startswith("--rdzv-endpoint=")
    )
    assert rdzv_endpoint == f"--rdzv-endpoint=localhost:{mod.DRY_RUN_RDZV_PORT}"


def test_run_matrix_entries_dry_run_does_not_bind_port(tmp_path, monkeypatch):
    """``launch.run_matrix_entries`` must not call ``pick_free_port``
    under ``dry_run=True``. The function is called by run_exp24's main
    dispatch for every training-matrix dry-run.
    """
    launch = _load_launch()

    def _raising_pick_free_port():
        raise AssertionError(
            "pick_free_port called during dry-run — would bind a socket"
        )

    monkeypatch.setattr(launch, "pick_free_port", _raising_pick_free_port)

    runner_path = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"
    results_dir = tmp_path / "matrix_dry_run"
    summary = launch.run_matrix_entries(
        entries=[
            {
                "name": "exp24_dry_run_no_port_s1337",
                "vocab_size": 16384,
                "budget_seconds": 90.0,
            }
        ],
        runner_path=runner_path,
        data_path=str(tmp_path / "data"),
        sp_model_paths={16384: str(tmp_path / "tokenizer.model")},
        results_dir=results_dir,
        world_size=1,
        dry_run=True,
        skip_existing=False,
    )
    commands = summary.get("commands") or []
    assert commands, "dry-run should populate summary['commands']"
    for cmd in commands:
        rdzv_endpoint = next(
            arg for arg in cmd if arg.startswith("--rdzv-endpoint=")
        )
        assert rdzv_endpoint == (
            f"--rdzv-endpoint=localhost:{launch.DRY_RUN_RDZV_PORT}"
        )


def test_summarize_result_dir_merges_val_bpb(tmp_path):
    launch = _load_launch()
    for name, tokens_per_sec, val_bpb in [
        ("slow_best", 10.0, 1.10),
        ("fast_worse", 100.0, 1.20),
    ]:
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "config": {"name": name},
                    "train": {
                        "aggregate_tokens_per_sec": tokens_per_sec,
                        "per_gpu_tokens_per_sec": tokens_per_sec / 4,
                        "steps": 7,
                        "final_loss": 3.0,
                        "peak_vram_mb": 123.0,
                    },
                    "artifact": {
                        "artifact_impact": "artifact_training_only",
                        "submit_valid": True,
                    },
                    "exp24": {
                        "phase": "phase0",
                        "mechanism": "fast_slow_dreamworld",
                    },
                }
            )
        )
        full_val = tmp_path / "full_val"
        full_val.mkdir(exist_ok=True)
        (full_val / f"{name}.summary.json").write_text(
            json.dumps({"aggregate_bpb": val_bpb, "docs_scored": 50000})
        )

    summary = launch.summarize_result_dir(tmp_path)

    assert [row["name"] for row in summary["ranked"]] == ["slow_best", "fast_worse"]
    assert [row["val_bpb"] for row in summary["ranked"]] == [1.10, 1.20]
    assert [row["val_docs_scored"] for row in summary["ranked"]] == [50000, 50000]


def test_episodic_ttt_v1_matrix_shape():
    """Phase 3 TTT-shaped falsifier matrix: 4 arms x 3 seeds.

    Supersedes the training-only ``build_episodic_dw_curation_v1_matrix``
    after the architecture pivot — the cache must be live at eval too,
    not just during training. Per the W-task spec the four arms are::

      Arm | Train side                | Eval side
      ----|---------------------------|------------------------------
      A   | Standard (no cache)       | No TTT, no cache (SOTA shape)
      B   | Cache-curated DW          | TTT with loaded cache (the bet)
      C   | Cache-curated DW          | No TTT (cache vs TTT control)
      D   | Standard (no cache)       | TTT, fresh empty cache

    Topology-locking: the train-side knobs (fast/slow, dreamworld replay
    knobs, world_size, budget) are identical within each train-side
    half (A=D, B=C). The eval-side knobs (eval_episodic_cache_enabled,
    eval_steps_per_chunk, eval_adapt_set) similarly share the same
    eval shape within each eval-side half (A=C, B=D).

    NOTE: the eval-side fields are RECORDED ON THE MATRIX ENTRY but
    cannot today be plumbed into ``run_exp20_full_val_score.py`` (the path
    used by run_exp24's ``--full-val-score``). Wiring them through is a
    separate task; this matrix encodes the intent so downstream analysis
    can attribute outcomes correctly.
    """
    mod = _load_exp24()

    entries = mod.build_episodic_ttt_v1_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
    )

    assert len(entries) == 12
    assert {entry["seed"] for entry in entries} == {1337, 2674, 4011}

    expected_arms = (
        "arm_a_no_cache_no_ttt",
        "arm_b_cache_train_ttt_with_cache",
        "arm_c_cache_train_no_ttt",
        "arm_d_no_cache_train_ttt_only",
    )
    expected_names = {
        f"exp24_phase3_episodic_ttt_v1_{arm}_s{seed}"
        for arm in expected_arms
        for seed in (1337, 2674, 4011)
    }
    assert {entry["name"] for entry in entries} == expected_names

    by_arm: dict[str, list[dict]] = {arm: [] for arm in expected_arms}
    for entry in entries:
        suffix = entry["name"].split("episodic_ttt_v1_", 1)[1]
        arm_tag = suffix.rsplit("_s", 1)[0]
        by_arm[arm_tag].append(entry)
    assert all(len(rows) == 3 for rows in by_arm.values())

    # ---- shared topology across ALL arms ----
    for entry in entries:
        assert entry["exp24_phase"] == "phase3"
        assert entry["exp24_mechanism"] == "episodic_ttt_v1"
        assert entry["world_size"] == 4
        assert entry["budget_seconds"] == 600.0
        assert entry["artifact_impact"] == "artifact_training_only"
        # Locked fast/slow recipe (shared with phase0_fastslow_only_control
        # — Arm A is the SOTA control, must mirror this).
        assert entry["fast_slow_enabled"] is True
        assert entry["fast_slow_interval"] == 64
        assert entry["fast_slow_alpha"] == 0.25
        assert entry["fast_slow_eval_copy"] == "slow"

    # ---- train-side topology halves: A==D (no cache), B==C (cache) ----
    for entry in by_arm["arm_a_no_cache_no_ttt"]:
        # Standard training: dreamworld OFF, episodic OFF. Mirrors
        # phase0_fastslow_only_control exactly so Arm A is a true SOTA
        # control vs Arm B's cache-train treatment.
        assert entry["dreamworld_enabled"] is False
        assert entry["dreamworld_weight"] == 0.0
        assert entry.get("episodic_enabled", False) is False

    for entry in by_arm["arm_d_no_cache_train_ttt_only"]:
        # Same train-side as Arm A — only eval differs.
        assert entry["dreamworld_enabled"] is False
        assert entry["dreamworld_weight"] == 0.0
        assert entry.get("episodic_enabled", False) is False

    for entry in by_arm["arm_b_cache_train_ttt_with_cache"]:
        # Cache-curated DW training (mirrors arm_b_cosine_utility from
        # the deprecated v1 matrix).
        assert entry["dreamworld_enabled"] is True
        assert entry["episodic_enabled"] is True
        assert entry["controller_query_enabled"] is True
        assert entry["controller_query_mode"] == "cosine_utility_weighted"
        assert entry["dreamworld_weight"] == 0.10
        assert entry["dreamworld_cache_interval"] == 16
        assert entry["dreamworld_interval"] == 16
        assert entry["dreamworld_replay_batch_size"] == 128
        assert entry["dreamworld_prefix_tokens"] == 128
        assert entry["dreamworld_replay_tokens"] == 64
        assert entry["dreamworld_buffer_size"] == 16
        assert entry["dreamworld_min_size"] == 2
        assert entry["dreamworld_max_age_steps"] == 256

    for entry in by_arm["arm_c_cache_train_no_ttt"]:
        # Same train-side as Arm B — only eval differs.
        assert entry["dreamworld_enabled"] is True
        assert entry["episodic_enabled"] is True
        assert entry["controller_query_enabled"] is True
        assert entry["controller_query_mode"] == "cosine_utility_weighted"
        assert entry["dreamworld_weight"] == 0.10
        assert entry["dreamworld_cache_interval"] == 16
        assert entry["dreamworld_interval"] == 16
        assert entry["dreamworld_replay_batch_size"] == 128
        assert entry["dreamworld_prefix_tokens"] == 128
        assert entry["dreamworld_replay_tokens"] == 64
        assert entry["dreamworld_buffer_size"] == 16
        assert entry["dreamworld_min_size"] == 2
        assert entry["dreamworld_max_age_steps"] == 256

    # ---- eval-side topology halves: A==C (no TTT), B==D (TTT) ----
    for entry in by_arm["arm_a_no_cache_no_ttt"]:
        # No TTT, no cache.
        assert entry["eval_episodic_cache_enabled"] is False
        assert entry["eval_steps_per_chunk"] == 0
        assert entry["eval_adapt_set"] == "none"
        assert entry["eval_episodic_cache_reset_per_doc"] is False

    for entry in by_arm["arm_c_cache_train_no_ttt"]:
        # Same eval shape as Arm A — score-only, no cache load at eval.
        assert entry["eval_episodic_cache_enabled"] is False
        assert entry["eval_steps_per_chunk"] == 0
        assert entry["eval_adapt_set"] == "none"
        assert entry["eval_episodic_cache_reset_per_doc"] is False

    for entry in by_arm["arm_b_cache_train_ttt_with_cache"]:
        # TTT with the trained cache loaded.
        assert entry["eval_episodic_cache_enabled"] is True
        assert entry["eval_steps_per_chunk"] == 1
        assert entry["eval_adapt_set"] == "lm_head"
        assert entry["eval_episodic_cache_reset_per_doc"] is False

    for entry in by_arm["arm_d_no_cache_train_ttt_only"]:
        # Same eval shape as Arm B (TTT enabled), but the checkpoint
        # has no cache so the driver constructs a fresh empty one.
        assert entry["eval_episodic_cache_enabled"] is True
        assert entry["eval_steps_per_chunk"] == 1
        assert entry["eval_adapt_set"] == "lm_head"
        assert entry["eval_episodic_cache_reset_per_doc"] is False


def test_episodic_ttt_v1_matrix_arm_a_matches_fastslow_only_control():
    """Arm A is the SOTA control — train-side fields must reproduce
    ``phase0_fastslow_only_control`` exactly so Arm A vs Arm B isolates the
    cache mechanism from any other train-side change.
    """
    mod = _load_exp24()

    ttt = mod.build_episodic_ttt_v1_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
    )
    fastslow = mod.build_phase0_fastslow_only_control(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
    )
    arm_a = [e for e in ttt if "arm_a_no_cache_no_ttt" in e["name"]]
    # Must have all 3 seeds.
    assert len(arm_a) == len(fastslow) == 3

    # Pin the train-side topology fields one-by-one. Eval-side fields and
    # naming differ — exclude those when matching against the control.
    train_side_fields = (
        "world_size", "budget_seconds", "artifact_impact",
        "fast_slow_enabled", "fast_slow_interval", "fast_slow_alpha",
        "fast_slow_eval_copy",
        "dreamworld_enabled", "dreamworld_cache_interval",
        "dreamworld_interval", "dreamworld_weight",
        "dreamworld_replay_batch_size",
        "dreamworld_prefix_tokens", "dreamworld_replay_tokens",
        "dreamworld_buffer_size", "dreamworld_min_size",
        "dreamworld_max_age_steps",
    )
    arm_a_by_seed = {entry["seed"]: entry for entry in arm_a}
    fastslow_by_seed = {entry["seed"]: entry for entry in fastslow}
    assert set(arm_a_by_seed.keys()) == set(fastslow_by_seed.keys())
    for seed, entry in arm_a_by_seed.items():
        ref = fastslow_by_seed[seed]
        for field_name in train_side_fields:
            assert entry[field_name] == ref[field_name], (
                f"Arm A field {field_name!r} (seed={seed}) diverges from "
                f"phase0_fastslow_only_control: {entry[field_name]!r} != "
                f"{ref[field_name]!r}"
            )


def test_episodic_ttt_v1_matrix_supports_arms_filter_for_sigma_escalation():
    """Decision 0.5 escalation pattern: re-invoke the builder with a custom
    seed list and an arm filter to escalate signal-to-noise on the cache
    treatment without paying for re-running the SOTA controls.
    """
    mod = _load_exp24()

    extra_all = mod.build_episodic_ttt_v1_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
        seed_values=[5012, 7331, 9183],
    )
    assert len(extra_all) == 12
    assert {entry["seed"] for entry in extra_all} == {5012, 7331, 9183}

    # Escalate the cache-vs-no-cache contrast: Arms A, B, D only.
    escalated = mod.build_episodic_ttt_v1_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
        seed_values=[5012, 7331, 9183],
        arms=("arm_a_no_cache_no_ttt", "arm_b_cache_train_ttt_with_cache",
              "arm_d_no_cache_train_ttt_only"),
    )
    assert len(escalated) == 9
    assert all("arm_c_cache_train_no_ttt" not in e["name"] for e in escalated)


def test_episodic_ttt_v1_matrix_rejects_unknown_arm_name():
    """Typo defense: passing an arm name not in the canonical four raises
    ValueError with the allowed names listed.
    """
    mod = _load_exp24()

    try:
        mod.build_episodic_ttt_v1_matrix(
            speed_config={"batch_size": 1024, "chunk_size": 64},
            world_size=4,
            budget_seconds=600.0,
            arms=("arm_b_cache_train_ttt_with_cache", "arm_typo"),
        )
    except ValueError as exc:
        assert "arm_typo" in str(exc)
        assert "arm_a_no_cache_no_ttt" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown arm name")


def test_run_exp24_cli_episodic_ttt_v1_dry_run(tmp_path):
    """The CLI must accept ``--matrix episodic_ttt_v1`` and dispatch to
    the new builder with the right default world_size (4)."""
    script = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    output_dir = tmp_path / "exp24-episodic-ttt-v1-dryrun"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            "episodic_ttt_v1",
            "--dry-run",
            "--limit",
            "4",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "matrix=episodic_ttt_v1" in stdout
    assert "world_size=4" in stdout
    assert "exp24_phase3_episodic_ttt_v1_arm_a_no_cache_no_ttt_s1337" in stdout
    assert "exp24_phase3_episodic_ttt_v1_arm_b_cache_train_ttt_with_cache_s1337" in stdout
    assert '"exp24_mechanism": "episodic_ttt_v1"' in stdout


def test_episodic_ttt_v1_matrix_pins_eval_cache_schema_on_all_arms():
    """All four arms must carry the same cache schema fields (capacity,
    span_length, key_rep_dim sentinel, grace_steps). Differing shape across
    arms would let a cache-shape difference look like a cache-content
    difference and silently corrupt the falsifier matrix's Arm B vs Arm D
    contrast — defense-in-depth on top of the RunConfig defaults.
    """
    mod = _load_exp24()

    entries = mod.build_episodic_ttt_v1_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
    )

    for entry in entries:
        # Defaults that mirror runner_fast_path.py:_construct_episodic_cache.
        assert entry["eval_episodic_cache_capacity"] == 4096
        assert entry["eval_episodic_span_length"] == 4
        assert entry["eval_episodic_key_rep_dim"] == -1, (
            "-1 sentinel resolves to model_dim at cache construction; "
            "matches trainer's episodic_key_rep_dim=model_dim default"
        )
        assert entry["eval_episodic_grace_steps"] == 1000
        assert entry["eval_episodic_fingerprint_window"] == 8, (
            "W must match runner_fast_path.py's episodic_fingerprint_window "
            "default; mismatch silently zeros the cache hit rate"
        )

    # Cross-arm: every arm has identical cache shape — Arm B vs Arm D
    # cannot diverge here.
    schema_fields = (
        "eval_episodic_cache_capacity",
        "eval_episodic_span_length",
        "eval_episodic_key_rep_dim",
        "eval_episodic_grace_steps",
        "eval_episodic_fingerprint_window",
    )
    shapes = {
        tuple(entry[f] for f in schema_fields)
        for entry in entries
    }
    assert len(shapes) == 1, (
        f"all arms must share an identical cache shape; got {shapes!r}"
    )


def test_episodic_controller_v1_matrix_has_five_arms_three_seeds():
    """Simplex controller V1 falsifier matrix: 5 arms x 3 seeds.

    Three pairwise contrasts the matrix exposes:
      - heuristic vs simplex: arm_b vs arm_c (frozen-trained policy is
        the only difference; same retrieval semantics).
      - frozen vs online: arm_c vs arm_d (online REINFORCE is the
        only difference).
      - cold vs warm cache: arm_d vs arm_e (online controller held
        constant; cache init is the only difference).
    """
    mod = _load_exp24()

    entries = mod.build_episodic_controller_v1_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
    )

    assert len(entries) == 15
    assert len({entry["name"] for entry in entries}) == 15
    assert {entry["seed"] for entry in entries} == {1337, 2674, 4011}

    expected_arms = (
        "arm_a_control",
        "arm_b_heuristic",
        "arm_c_simplex_frozen",
        "arm_d_simplex_online",
        "arm_e_simplex_warm_online",
    )
    assert {entry["arm"] for entry in entries} == set(expected_arms)
    assert {entry["exp24_mechanism"] for entry in entries} == {
        "episodic_controller_v1"
    }
    assert {entry["exp24_phase"] for entry in entries} == {"phase3"}
    for arm in expected_arms:
        assert {entry["seed"] for entry in entries if entry["arm"] == arm} == {
            1337,
            2674,
            4011,
        }

    by_arm: dict[str, list[dict]] = {arm: [] for arm in expected_arms}
    for entry in entries:
        by_arm[entry["arm"]].append(entry)
        assert entry["name"] == (
            f"exp24_phase3_episodic_controller_v1_{entry['arm']}_"
            f"s{entry['seed']}"
        )
        assert entry["world_size"] == 4
        assert entry["budget_seconds"] == 600.0
        assert entry["artifact_impact"] == "artifact_training_only"
        assert entry["fast_slow_enabled"] is True
        assert entry["fast_slow_interval"] == 64
        assert entry["fast_slow_alpha"] == 0.25
        assert entry["fast_slow_eval_copy"] == "slow"

    assert all(len(rows) == 3 for rows in by_arm.values())

    for entry in by_arm["arm_a_control"]:
        assert entry["episodic_enabled"] is False
        assert entry["eval_episodic_cache_enabled"] is False
        assert entry["episodic_event_log_enabled"] is False
        assert entry["episodic_controller_runtime"] == "heuristic"
        assert entry["controller_train_online"] is False

    for entry in by_arm["arm_b_heuristic"]:
        assert entry["episodic_enabled"] is True
        assert entry["eval_episodic_cache_enabled"] is True
        assert entry["eval_episodic_cache_mode"] == "cold"
        assert entry["episodic_controller_runtime"] == "heuristic"
        assert entry["controller_train_online"] is False
        assert "episodic_controller_weights_path" not in entry

    simplex_arms = (
        "arm_c_simplex_frozen",
        "arm_d_simplex_online",
        "arm_e_simplex_warm_online",
    )
    for arm in simplex_arms:
        for entry in by_arm[arm]:
            assert entry["episodic_enabled"] is True
            assert entry["eval_episodic_cache_enabled"] is True
            assert entry["episodic_event_log_enabled"] is True
            assert entry["episodic_controller_runtime"] == "simplex_v1"
            assert entry["episodic_controller_selection_mode"] == "sample"
            assert entry["episodic_controller_weights_path"] == (
                "TO_BE_FILLED/episodic_controller_v1_weights.pt"
            )

    for entry in by_arm["arm_c_simplex_frozen"]:
        assert entry["eval_episodic_cache_mode"] == "cold"
        assert entry["controller_train_online"] is False

    for entry in by_arm["arm_d_simplex_online"]:
        assert entry["eval_episodic_cache_mode"] == "cold"
        assert entry["controller_train_online"] is True

    for entry in by_arm["arm_e_simplex_warm_online"]:
        assert entry["eval_episodic_cache_mode"] == "warm"
        assert entry["eval_episodic_cache_source"] == "checkpoint"
        assert entry["controller_train_online"] is True

    for arm in simplex_arms:
        for entry in by_arm[arm]:
            assert entry["episodic_compute_replay_ce_pair"] is True

    for arm in ("arm_a_control", "arm_b_heuristic"):
        for entry in by_arm[arm]:
            assert "episodic_compute_replay_ce_pair" not in entry


def test_episodic_controller_v1_weights_path_honors_env_override(monkeypatch):
    """Pod runbook substitutes the real weights via env var; matrix builder
    must read the env at build time so the test default + pod override
    can both work without editing this file.
    """
    mod = _load_exp24()
    monkeypatch.setenv(
        mod.EPISODIC_CONTROLLER_V1_WEIGHTS_PATH_ENV,
        "/workspace/episodic_controller_v1_real.pt",
    )
    entries = mod.build_episodic_controller_v1_matrix(
        speed_config={"batch_size": 1024, "chunk_size": 64},
        world_size=4,
        budget_seconds=600.0,
    )
    simplex = [
        entry for entry in entries
        if entry["arm"] in {
            "arm_c_simplex_frozen",
            "arm_d_simplex_online",
            "arm_e_simplex_warm_online",
        }
    ]
    assert simplex
    for entry in simplex:
        assert entry["episodic_controller_weights_path"] == (
            "/workspace/episodic_controller_v1_real.pt"
        )


def test_episodic_eval_args_from_entry_cold_cache():
    mod = _load_run_exp24()
    args = mod._episodic_eval_args_from_entry(
        {
            "eval_episodic_cache_enabled": True,
            "eval_episodic_cache_mode": "cold",
            "eval_episodic_cache_capacity": 4096,
            "eval_episodic_span_length": 4,
            "eval_episodic_key_rep_dim": -1,
            "eval_episodic_grace_steps": 1000,
            "eval_episodic_fingerprint_window": 8,
            "eval_episodic_cache_reset_per_doc": False,
            "controller_train_online": False,
        }
    )
    assert "--episodic-cache-enabled" in args
    assert "--episodic-cache-source" in args
    assert args[args.index("--episodic-cache-source") + 1] == "fresh"
    assert "--no-episodic-cache-reset-per-doc" in args
    assert "--controller-train-online" not in args


def test_episodic_eval_args_from_entry_warm_online():
    mod = _load_run_exp24()
    args = mod._episodic_eval_args_from_entry(
        {
            "eval_episodic_cache_enabled": True,
            "eval_episodic_cache_mode": "warm",
            "eval_episodic_cache_source": "checkpoint",
            "controller_train_online": True,
        }
    )
    assert "--episodic-cache-enabled" in args
    assert args[args.index("--episodic-cache-source") + 1] == "checkpoint"
    assert "--controller-train-online" in args


def test_episodic_eval_args_from_entry_disabled():
    mod = _load_run_exp24()
    args = mod._episodic_eval_args_from_entry(
        {"eval_episodic_cache_enabled": False, "eval_episodic_cache_mode": "none"}
    )
    assert "--no-episodic-cache-enabled" in args
    assert "--episodic-cache-source" not in args
    assert "--controller-train-online" not in args


def test_run_exp24_cli_episodic_controller_v1_dry_run(tmp_path):
    script = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
    output_dir = tmp_path / "exp24-controller-v1-dryrun"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--matrix",
            "episodic_controller_v1",
            "--dry-run",
            "--limit",
            "15",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "matrix=episodic_controller_v1" in stdout
    assert "world_size=4" in stdout
    assert "entries=15" in stdout
    assert "exp24_phase3_episodic_controller_v1_arm_a_control_s1337" in stdout
    assert "exp24_phase3_episodic_controller_v1_arm_d_simplex_online_s1337" in stdout
    assert "exp24_phase3_episodic_controller_v1_arm_e_simplex_warm_online_s4011" in stdout
    assert '"exp24_mechanism": "episodic_controller_v1"' in stdout
