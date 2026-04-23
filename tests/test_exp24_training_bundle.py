"""Tests for Exp24 training-time bundle matrix helpers."""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXP24_PATH = REPO / "experiments" / "24_training_time_bundle" / "exp24.py"
RUN_EXP24_PATH = REPO / "experiments" / "24_training_time_bundle" / "run_exp24.py"
LAUNCH_PATH = REPO / "experiments" / "23_fast_path" / "launch.py"


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
    assert "run_exp20_fast_score.py" in stdout
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
    assert str(REPO / "scripts" / "run_exp20_fast_score.py") in cmd
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
