"""Tests for CRCT artifact headroom validator."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "24_crct" / "validate_dim384_artifact.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("validate_dim384_artifact", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_estimate_headroom_reports_controller_params_and_margin():
    mod = _load_script()

    result = mod.estimate_artifact_headroom(
        dim=32,
        vocab_size=128,
        num_layers=1,
        ff_mult=2,
        overhead_bytes=0,
    )

    assert result.raw_bf16_bytes > 0
    assert result.baseline_dim256_raw_bf16_bytes > result.raw_bf16_bytes
    assert result.controller_params > 0
    assert result.margin_bytes > 0
    assert result.under_budget is True


def test_main_writes_json_output(tmp_path):
    mod = _load_script()
    out = tmp_path / "artifact.json"

    code = mod.main([
        "--dim", "32",
        "--vocab-size", "128",
        "--num-layers", "1",
        "--output", str(out),
    ])

    assert code == 0
    text = out.read_text()
    assert '"dim": 32' in text
    assert '"under_budget": true' in text
