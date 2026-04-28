#!/usr/bin/env python3
"""Summarize best BPB candidates from Exp24 summary artifacts.

Usage:
  python tools/exp24_best_bpb.py
  python tools/exp24_best_bpb.py --root experiments/24_training_time_bundle --top 10 --json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BestRow:
    summary_path: str
    run_name: str
    val_bpb: float
    steps: int
    per_gpu_tokens_per_sec: float


def _iter_summary_paths(root: Path) -> Iterable[Path]:
    return root.glob("**/summary.json")


def _extract_best_row(summary_path: Path) -> BestRow | None:
    try:
        payload = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    ranked = payload.get("ranked") if isinstance(payload, dict) else None
    if not isinstance(ranked, list) or not ranked:
        return None

    candidates = [
        row for row in ranked
        if isinstance(row, dict) and isinstance(row.get("val_bpb"), (int, float))
    ]
    if not candidates:
        return None

    best = min(candidates, key=lambda row: float(row["val_bpb"]))
    return BestRow(
        summary_path=str(summary_path),
        run_name=str(best.get("name", "unknown")),
        val_bpb=float(best["val_bpb"]),
        steps=int(best.get("steps", 0)),
        per_gpu_tokens_per_sec=float(best.get("per_gpu_tokens_per_sec", 0.0)),
    )


def collect_best_rows(root: Path) -> list[BestRow]:
    rows: list[BestRow] = []
    for summary_path in _iter_summary_paths(root):
        row = _extract_best_row(summary_path)
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda row: row.val_bpb)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/24_training_time_bundle"))
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    rows = collect_best_rows(args.root)
    if args.top > 0:
        rows = rows[: args.top]

    if args.as_json:
        print(json.dumps([row.__dict__ for row in rows], indent=2))
        return 0

    if not rows:
        print("No ranked summary.json files with val_bpb found.")
        return 0

    print("val_bpb | steps | per_gpu_tok/s | run_name | summary_path")
    for row in rows:
        print(
            f"{row.val_bpb:.12f} | {row.steps} | {row.per_gpu_tokens_per_sec:.0f} | "
            f"{row.run_name} | {row.summary_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
