"""Tests for Experiment 18 scaffold and preflight logic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXPERIMENT))

import bench_throughput as bench  # noqa: E402
from bench_throughput import orchestrate_phase0  # noqa: E402
from run_exp18 import run_phase_a, summarize_results  # noqa: E402
from runner_exp18 import (  # noqa: E402
    LOW_COVERAGE_FLOOR,
    PhaseBudget,
    assert_runtime_compatibility,
    build_child_env,
    build_coverage_plan,
    build_phase_a_conditions,
    choose_tokenizer,
    generate_sweep_starts,
    make_smoke_summary,
    resolve_visible_cuda_devices,
    run_condition,
    select_subset,
)


def test_generate_sweep_starts_is_non_overlapping():
    starts = generate_sweep_starts(100, 10)
    assert starts[:5] == [0, 10, 20, 30, 40]
    assert all((b - a) == 10 for a, b in zip(starts, starts[1:], strict=False))


def test_build_coverage_plan_marks_low_coverage():
    plan = build_coverage_plan(10_000, seq_len=100, projected_windows=10)
    assert plan.planned_windows == 10
    assert plan.coverage_frac < LOW_COVERAGE_FLOOR
    assert plan.low_coverage_regime is True


def test_phase_budget_tracks_selection_overhead():
    budget = PhaseBudget(total_s=600.0, sweep_s=400.0, rescore_s=45.0, subset_s=5.0, retarget_s=120.0)
    assert budget.remaining_s == pytest.approx(30.0)
    assert budget.selection_overhead_s == pytest.approx(50.0)


def test_choose_tokenizer_prefers_coverage_when_material():
    chosen = choose_tokenizer(
        [
            {
                "tokenizer": "sp8192",
                "stable": True,
                "prior_mean_bpb": 1.967,
                "projected_coverage_frac": 0.60,
                "tokens_per_s": 10_000.0,
                "low_coverage_regime": False,
            },
            {
                "tokenizer": "sp16384",
                "stable": True,
                "prior_mean_bpb": 1.959,
                "projected_coverage_frac": 0.40,
                "tokens_per_s": 8_000.0,
                "low_coverage_regime": False,
            },
        ]
    )
    assert chosen["tokenizer"] == "sp8192"


def test_choose_tokenizer_prefers_sp16384_on_small_gap():
    chosen = choose_tokenizer(
        [
            {
                "tokenizer": "sp8192",
                "stable": True,
                "prior_mean_bpb": 1.967,
                "projected_coverage_frac": 0.52,
                "tokens_per_s": 10_000.0,
                "low_coverage_regime": False,
            },
            {
                "tokenizer": "sp16384",
                "stable": True,
                "prior_mean_bpb": 1.959,
                "projected_coverage_frac": 0.50,
                "tokens_per_s": 9_900.0,
                "low_coverage_regime": False,
            },
        ]
    )
    assert chosen["tokenizer"] == "sp16384"


def test_select_subset_top_and_random_are_matched_size():
    scored = [{"start": float(i), "loss": float(i)} for i in range(20)]
    top = select_subset(scored, fraction=0.10, mode="top", seed=1)
    rand = select_subset(scored, fraction=0.10, mode="random", seed=1)
    assert len(top) == 2
    assert len(rand) == 2
    assert set(top) == {19, 18}
    assert set(rand) != set(top)


def test_build_phase_a_conditions_contains_core_matrix():
    summary = make_smoke_summary()
    conditions = build_phase_a_conditions(summary, total_budget_s=600.0)
    assert set(conditions) == {
        "baseline_b32",
        "sweep_only",
        "sweep_target_top10",
        "sweep_random_retrain",
    }


def test_build_child_env_respects_parent_mask():
    env = build_child_env(
        gpu_slot=1,
        smoke=False,
        base_env={"CUDA_VISIBLE_DEVICES": "3,5,7"},
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert resolve_visible_cuda_devices({"CUDA_VISIBLE_DEVICES": "3,5,7"}) == ["3", "5", "7"]


def test_runtime_compatibility_smoke_reports_backend():
    info = assert_runtime_compatibility(smoke=True)
    assert "python_version" in info
    assert "diag_recurrence" in info


def test_orchestrate_phase0_smoke_outputs_selected_tokenizer():
    summary = orchestrate_phase0(
        data_root=None,
        num_gpus=1,
        batch_sizes=[2, 4],
        throughput_steps=1,
        lr_steps=2,
        output_json=None,
        smoke=True,
    )
    assert summary["phase"] == "phase0"
    assert summary["selected"]["tokenizer"] in {"sp8192", "sp16384"}
    assert "low_coverage_reconsideration" in summary


def test_benchmark_tokenizer_stops_at_first_oom(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bench, "build_model", lambda cfg, device, param_dtype: object())
    monkeypatch.setattr(
        bench,
        "_throughput_steps",
        lambda model, **kwargs: (
            (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
            if kwargs["batch_size"] >= 8
            else {
                "steps": 1.0,
                "elapsed_s": 0.1,
                "step_time_s": 0.1,
                "tokens_per_s": 1000.0,
                "peak_vram_gb": 1.0,
            }
        ),
    )
    monkeypatch.setattr(
        bench,
        "_lr_screen",
        lambda *args, **kwargs: [{"label": "fixed", "lr": 0.002, "stable": True, "failed": False, "oom": False}],
    )
    result = bench.benchmark_tokenizer(
        vocab_size=8192,
        data_path=None,
        sp_model_path=None,
        device_name="cpu",
        smoke=True,
        batch_sizes=[2, 4, 8, 16],
        throughput_steps=1,
        lr_steps=1,
        sweep_budget_s=10.0,
    )
    assert result["selected_candidate"]["batch_size"] == 4
    assert result["oom_batches"] == [8]


def test_run_condition_smoke_exercises_full_chain(tmp_path: Path):
    summary = make_smoke_summary(total_budget_s=4.0)
    conditions = build_phase_a_conditions(summary, total_budget_s=4.0)
    output_path = tmp_path / "smoke_result.json"
    result = run_condition(
        conditions["sweep_target_top10"],
        phase0_summary=summary,
        output_json=str(output_path),
        smoke=True,
    )
    assert result["phase"] == "preflight"
    assert result["condition"] == "sweep_target_top10"
    assert result["timings"]["rescore_s"] >= 0.0
    assert result["timings"]["selection_overhead_s"] >= result["timings"]["rescore_s"]
    assert result["selection"]["selected_windows"] > 0
    assert "rescore" in result["train_phases"]
    assert "retarget" in result["train_phases"]
    disk = json.loads(output_path.read_text())
    assert disk["condition"] == result["condition"]


def test_run_phase_a_smoke_reports_core_matrix(tmp_path: Path):
    summary_path = tmp_path / "phase0_smoke.json"
    summary_path.write_text(json.dumps(make_smoke_summary(total_budget_s=2.0)))
    summary = run_phase_a(
        phase0_summary_path=str(summary_path),
        budget_s=2.0,
        num_gpus=4,
        smoke=True,
    )
    assert set(summary) >= {
        "baseline_b32",
        "sweep_only",
        "sweep_target_top10",
        "sweep_random_retrain",
        "_paired",
    }


def test_summarize_results_reports_rescore_tax():
    summary = summarize_results(
        [
            {
                "condition": "sweep_target_top10",
                "eval": {"bpb": 1.9},
                "timings": {"rescore_frac_of_budget": 0.08, "selection_overhead_s": 50.0},
            },
            {
                "condition": "sweep_target_top10",
                "eval": {"bpb": 1.8},
                "timings": {"rescore_frac_of_budget": 0.09, "selection_overhead_s": 51.0},
            },
            {
                "condition": "sweep_only",
                "eval": {"bpb": 2.0},
                "timings": {"rescore_frac_of_budget": 0.0, "selection_overhead_s": 0.0},
            },
            {
                "condition": "baseline_b32",
                "eval": {"bpb": 2.1},
                "timings": {"rescore_frac_of_budget": 0.0, "selection_overhead_s": 0.0},
            },
        ]
    )
    assert summary["sweep_target_top10"]["mean_rescore_frac_of_budget"] == pytest.approx(0.085)
    assert summary["sweep_target_top10"]["mean_selection_overhead_s"] == pytest.approx(50.5)
    assert "_paired" in summary
