import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("experiments/20_ssm_native_ttt/build_matrix.py")


def _run_build_matrix(tmp_path, *extra_args):
    cfg_dir = tmp_path / "configs"
    out_root = tmp_path / "results"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--matrix",
        "first_wave",
        "--config-dir",
        str(cfg_dir),
        "--output-root",
        str(out_root),
        "--checkpoint-path",
        "/workspace/results/final.pt",
        "--sp-model-path",
        "/workspace/tokenizers/fineweb_8192.model",
        "--jsonl-path",
        "/workspace/data/eval.jsonl",
        "--seeds",
        "0",
        "1",
        "2",
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((cfg_dir / "manifest.json").read_text())
    configs = {
        path.name: json.loads(path.read_text())
        for path in sorted(cfg_dir.glob("*.json"))
        if path.name != "manifest.json"
    }
    return manifest, configs


def test_first_wave_matrix_generates_real_exp20_screens(tmp_path):
    manifest, configs = _run_build_matrix(
        tmp_path,
        "--score-floor-seconds",
        "596.7",
        "--safety-margin-seconds",
        "20",
    )

    assert manifest["matrix"] == "first_wave"
    assert manifest["total_configs"] == 57
    assert len(configs) == 57

    names = {cfg["name"] for cfg in configs.values()}
    assert "floor_reset_s0" in names
    assert "floor_carry_state_s2" in names
    assert "axis1_log_a_lr0p016_carry_state_s0" in names
    assert "axis1_lm_head_lr0p064_carry_state_s2" in names
    assert "axis3_delta_scale_0p5_carry_state_s1" in names
    assert "axis3_log_a_shift_m0p5_carry_state_s1" in names

    adapt_sets = {cfg["adapt_set"] for cfg in configs.values()}
    assert "all" not in adapt_sets
    assert {
        "none",
        "log_a",
        "delta_proj",
        "log_a+delta_proj",
        "B_side",
        "C_side",
        "lm_head",
    } <= adapt_sets

    for cfg in configs.values():
        assert cfg["checkpoint_path"] == "/workspace/results/final.pt"
        assert cfg["sp_model_path"] == "/workspace/tokenizers/fineweb_8192.model"
        assert cfg["jsonl_paths"] == ["/workspace/data/eval.jsonl"]
        assert cfg["output_path"].endswith(".jsonl")
        assert cfg["summary_path"].endswith("_summary.json")
        assert cfg["budget_seconds"] == 600.0
        assert cfg["max_docs"] == 50000
        assert cfg["safety_margin_seconds"] == 20.0

    floor_configs = [cfg for cfg in configs.values() if cfg["phase"] == "floor"]
    assert len(floor_configs) == 6
    assert {cfg["persistence_mode"] for cfg in floor_configs} == {"reset", "carry_state"}
    assert all(cfg["adapt_set"] == "none" for cfg in floor_configs)
    assert all(cfg["steps_per_chunk"] == 0 for cfg in floor_configs)
    assert all(cfg["score_floor_seconds"] == 0.0 for cfg in floor_configs)

    non_floor = [cfg for cfg in configs.values() if cfg["phase"] != "floor"]
    assert len(non_floor) == 51
    assert all(cfg["persistence_mode"] == "carry_state" for cfg in non_floor)
    assert all(cfg["score_floor_seconds"] == 596.7 for cfg in non_floor)


def test_phase_filter_generates_only_requested_phase(tmp_path):
    manifest, configs = _run_build_matrix(tmp_path, "--phase", "axis3")

    assert manifest["total_configs"] == 15
    assert {cfg["phase"] for cfg in configs.values()} == {"axis3"}

    knob_pairs = {
        (cfg["delta_scale"], cfg["log_a_shift"])
        for cfg in configs.values()
        if cfg["seed"] == 0
    }
    assert knob_pairs == {
        (0.5, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (1.0, -0.5),
        (1.0, 0.5),
    }
