"""Assertions for the cd_first_smoke matrix produced by exp24."""
from __future__ import annotations

import importlib.util
from pathlib import Path


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


def test_cd_first_smoke_matrix_has_eight_cells():
    mod = _load_exp24()
    cells = mod.build_criticality_distillation_first_smoke_matrix()
    assert len(cells) == 8, f"expected 8 cells; got {len(cells)}"


def test_cd_first_smoke_matrix_cells_have_required_names():
    mod = _load_exp24()
    cells = mod.build_criticality_distillation_first_smoke_matrix()
    names = {c["name"] for c in cells}
    expected = {
        "treatment", "telemetry", "shuffled", "budget_only",
        "hl_short", "hl_long", "H_short", "H_long",
    }
    assert names == expected, f"got {names}"


def test_cd_first_smoke_cells_all_ride_locked_fast_slow_base():
    mod = _load_exp24()
    cells = mod.build_criticality_distillation_first_smoke_matrix()
    for c in cells:
        cfg = c["config"]
        assert cfg.get("fast_slow_enabled") is True, f"{c['name']} missing fast_slow_enabled=True"
        assert cfg.get("fast_slow_interval") == 64, f"{c['name']} fast_slow_interval != 64"
        assert cfg.get("fast_slow_alpha") == 0.25, f"{c['name']} fast_slow_alpha != 0.25"


def test_cd_first_smoke_cells_all_emit_entropy_fused_streaming_cached():
    mod = _load_exp24()
    cells = mod.build_criticality_distillation_first_smoke_matrix()
    for c in cells:
        cfg = c["config"]
        assert cfg.get("lm_head_backward_mode") == "fused_streaming_cached", (
            f"{c['name']} wrong lm_head_backward_mode"
        )
        assert cfg.get("lm_head_emit_entropy") is True, f"{c['name']} missing lm_head_emit_entropy"


def test_cd_first_smoke_cells_all_have_cd_enabled():
    mod = _load_exp24()
    cells = mod.build_criticality_distillation_first_smoke_matrix()
    for c in cells:
        cfg = c["config"]
        assert cfg.get("criticality_distill_enabled") is True, (
            f"{c['name']} missing criticality_distill_enabled=True"
        )


def test_cd_first_smoke_falsifier_cell_flags():
    mod = _load_exp24()
    cells = {c["name"]: c for c in mod.build_criticality_distillation_first_smoke_matrix()}
    tel = cells["telemetry"]["config"]
    assert tel["criticality_distill_weight"] == 0.0, "telemetry must zero the loss weight"
    sh = cells["shuffled"]["config"]
    assert sh["criticality_distill_score_permute_before_topk"] is True, "shuffled flag"
    bo = cells["budget_only"]["config"]
    assert bo["criticality_distill_fixed_random_seats"] is True, "budget_only flag"
    # Sensitivity cells leave falsifier flags False.
    for sens_name in ("hl_short", "hl_long", "H_short", "H_long"):
        cfg = cells[sens_name]["config"]
        assert cfg["criticality_distill_score_permute_before_topk"] is False
        assert cfg["criticality_distill_fixed_random_seats"] is False


def test_cd_first_smoke_sensitivity_cells_have_distinct_knob():
    mod = _load_exp24()
    cells = {c["name"]: c for c in mod.build_criticality_distillation_first_smoke_matrix()}
    assert cells["hl_short"]["config"]["criticality_distill_trace_half_life_steps"] == 128.0
    assert cells["hl_long"]["config"]["criticality_distill_trace_half_life_steps"] == 512.0
    assert cells["H_short"]["config"]["criticality_distill_horizon_H"] == 8
    assert cells["H_long"]["config"]["criticality_distill_horizon_H"] == 32


def test_run_exp24_defaults_cd_first_smoke_to_world_size_1():
    mod = _load_run_exp24()
    assert mod._default_world_size_for_matrix("cd_first_smoke") == 1
