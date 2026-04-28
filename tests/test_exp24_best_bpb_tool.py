from __future__ import annotations

import json
from pathlib import Path

from tools.exp24_best_bpb import collect_best_rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_collect_best_rows_ranks_by_val_bpb(tmp_path: Path) -> None:
    root = tmp_path / "exp24"
    _write_json(
        root / "run_a" / "summary.json",
        {
            "ranked": [
                {"name": "a_bad", "val_bpb": 1.6, "steps": 100, "per_gpu_tokens_per_sec": 10.0},
                {"name": "a_good", "val_bpb": 1.4, "steps": 120, "per_gpu_tokens_per_sec": 12.0},
            ]
        },
    )
    _write_json(
        root / "run_b" / "summary.json",
        {
            "ranked": [
                {"name": "b", "val_bpb": 1.5, "steps": 90, "per_gpu_tokens_per_sec": 9.0},
            ]
        },
    )

    rows = collect_best_rows(root)
    assert [row.run_name for row in rows] == ["a_good", "b"]
    assert rows[0].val_bpb == 1.4
    assert rows[0].steps == 120


def test_collect_best_rows_ignores_unranked_or_invalid(tmp_path: Path) -> None:
    root = tmp_path / "exp24"
    _write_json(root / "run_a" / "summary.json", {"rows": []})
    (root / "run_b").mkdir(parents=True)
    (root / "run_b" / "summary.json").write_text("not json")

    rows = collect_best_rows(root)
    assert rows == []
