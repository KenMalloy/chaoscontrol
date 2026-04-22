"""Tests for Exp24 training-time bundle matrix helpers."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXP24_PATH = REPO / "experiments" / "24_training_time_bundle" / "exp24.py"


def _load_exp24():
    spec = importlib.util.spec_from_file_location("exp24_training_bundle", EXP24_PATH)
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
