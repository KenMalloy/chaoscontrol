#!/usr/bin/env python3
"""Bulk-register Exp 21 run results into the paper_results registry.

Reads per-run result JSONs under experiments/21_sgns_tokenizer/results/
and appends one confirmatory RunRecord per (cell, seed) to the registry.

Config-hash convention: sha256 of the JSON-serialized ``config`` field
in each result JSON, with keys sorted. Deterministic and self-contained
(no need to keep the ``/tmp/*.yaml`` files the harness generated).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from chaoscontrol.paper_results import register

REPO = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO / "experiments" / "21_sgns_tokenizer" / "results"

# Result-JSON directories by cell family → result subdir.
CELL_DIRS = {
    "four_cell": RESULTS_ROOT / "four_cell",
    "fullcov": RESULTS_ROOT / "fullcov",
    "shuffled": RESULTS_ROOT / "shuffled",
    "zero": RESULTS_ROOT / "zero",
}

# Cell-name aliases between filename prefix and the paper-results condition
# label. ``ssm_fullcov`` / ``ssm_shuffled`` / ``ssm_zero`` files live in the
# control dirs; ``A_transformer_random`` etc. live in the four_cell dir.
# They map 1:1 but this table is explicit so a reader sees the full
# vocabulary at a glance.
CONDITION_NAMES = {
    "A_transformer_random",
    "B_transformer_sgns",
    "C_ssm_random",
    "D_ssm_sgns",
    "ssm_fullcov",
    "ssm_shuffled",
    "ssm_zero",
}

FILENAME_RE = re.compile(r"^(?P<cell>[A-Za-z0-9_]+?)_s(?P<seed>\d+)\.json$")


def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _discover_results() -> list[tuple[str, int, Path]]:
    """Yield (cell_name, seed, result_json_path) for every discoverable run."""
    found: list[tuple[str, int, Path]] = []
    for _label, dirpath in CELL_DIRS.items():
        if not dirpath.is_dir():
            continue
        for p in dirpath.iterdir():
            m = FILENAME_RE.match(p.name)
            if not m:
                continue
            cell = m.group("cell")
            if cell not in CONDITION_NAMES:
                continue
            found.append((cell, int(m.group("seed")), p))
    return found


def main() -> int:
    records = _discover_results()
    if not records:
        print(f"no result JSONs found under {RESULTS_ROOT}")
        return 1

    print(f"discovered {len(records)} result JSONs; registering as confirmatory")
    for cell, seed, jpath in sorted(records):
        data = json.loads(jpath.read_text())
        config = data.get("config", {})
        eval_block = data.get("eval") or {}
        train_block = data.get("train") or {}
        bpb = float(eval_block.get("bpb", float("nan")))
        metrics = {
            "bpb": bpb,
            "wall_clock_s": float(train_block.get("elapsed_s", float("nan"))),
            "final_loss": float(train_block.get("final_loss", float("nan"))),
        }
        # Divergent runs (e.g. zero-init floor) are preserved as
        # datapoints by runner_exp21 with a ``nonfinite`` section and
        # bpb=nan. Surface the flag as a first-class metric so registry
        # consumers can filter without re-reading the per-run JSON.
        nonfinite = data.get("nonfinite") or {}
        if nonfinite.get("flag"):
            metrics["nonfinite_flag"] = 1.0
        rec = register(
            experiment="exp21",
            condition=cell,
            seed=seed,
            status="confirmatory",
            metrics=metrics,
            config_hash=_config_hash(config),
            artifacts=[str(jpath.relative_to(REPO))],
        )
        bpb_str = f"{bpb:.4f}" if bpb == bpb else "nan"  # nan != nan
        flag = " [NONFINITE]" if nonfinite.get("flag") else ""
        print(
            f"  {rec.experiment}/{rec.condition} seed={rec.seed} "
            f"bpb={bpb_str}{flag} dirty={rec.git_dirty}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
