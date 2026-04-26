"""Smoke test for the clearer Exp20 full-val score launcher."""
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_exp20_full_val_score.py"


def test_full_val_score_launcher_imports_and_exposes_main():
    spec = importlib.util.spec_from_file_location("run_exp20_full_val_score", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
