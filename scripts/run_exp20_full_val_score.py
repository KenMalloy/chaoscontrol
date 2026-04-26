"""Clear launcher for Exp20/Exp24 full-validation scoring.

This is intentionally a thin wrapper around ``run_exp20_fast_score.py``.
The old module name describes an implementation detail (the optimized scorer);
this launcher names the job the matrix actually wants: score the validation
set from a checkpoint, optionally with the eval-side episodic cache fields.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_fast_score_module():
    path = Path(__file__).resolve().with_name("run_exp20_fast_score.py")
    spec = importlib.util.spec_from_file_location("run_exp20_fast_score", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _load_fast_score_module().main()


if __name__ == "__main__":
    main()
