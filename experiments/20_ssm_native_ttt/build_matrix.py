#!/usr/bin/env python3
"""Build runnable Exp20 config JSONs for ``scripts/run_exp20_eval.py``.

The first-wave matrix turns the design doc's real SSM-native ablation into
concrete configs:

* floor: score-only reset/carry_state timing
* axis1: small/native adapt-set screen at a chosen persistence mode
* axis3: no-grad delta/log-a modulation screen

It deliberately excludes ``adapt_set=all``. That condition is an envelope
sanity check, not the Exp20 hypothesis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_AXIS1_ADAPT_SETS = [
    "log_a",
    "delta_proj",
    "log_a+delta_proj",
    "B_side",
    "C_side",
    "lm_head",
]
DEFAULT_AXIS1_EVAL_LRS = [0.016, 0.064]
DEFAULT_DELTA_SCALES = [0.5, 1.0, 2.0]
DEFAULT_LOG_A_SHIFTS = [-0.5, 0.0, 0.5]


def _float_tag(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def _config_path(config_dir: Path, name: str) -> Path:
    return config_dir / f"{name}.json"


def _result_paths(output_root: Path, phase: str, name: str) -> tuple[str, str]:
    phase_dir = output_root / phase
    return (
        str(phase_dir / f"{name}.jsonl"),
        str(phase_dir / f"{name}_summary.json"),
    )


def _base_config(args: argparse.Namespace, *, name: str, phase: str, seed: int) -> dict[str, Any]:
    output_path, summary_path = _result_paths(args.output_root, phase, name)
    return {
        "name": name,
        "phase": phase,
        "adapt_set": "none",
        "persistence_mode": "reset",
        "delta_scale": 1.0,
        "log_a_shift": 0.0,
        "chunk_size": args.chunk_size,
        "steps_per_chunk": 1,
        "eval_lr": args.default_eval_lr,
        "persistent_muon_moments": False,
        "warmup_steps": args.warmup_steps,
        "seed": seed,
        "max_docs": args.max_docs,
        "budget_seconds": args.budget_seconds,
        "score_floor_seconds": args.score_floor_seconds,
        "safety_margin_seconds": args.safety_margin_seconds,
        "checkpoint_path": str(args.checkpoint_path),
        "output_path": output_path,
        "summary_path": summary_path,
        "jsonl_paths": [str(p) for p in args.jsonl_path],
        "sp_model_path": str(args.sp_model_path),
    }


def _floor_configs(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    for persistence in ("reset", "carry_state"):
        for seed in args.seeds:
            name = f"floor_{persistence}_s{seed}"
            cfg = _base_config(args, name=name, phase="floor", seed=seed)
            cfg.update(
                adapt_set="none",
                persistence_mode=persistence,
                steps_per_chunk=0,
                score_floor_seconds=0.0,
            )
            yield cfg


def _axis1_configs(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    for adapt_set in args.axis1_adapt_set:
        if adapt_set == "all":
            raise ValueError("first-wave axis1 screen must not include adapt_set='all'")
        for eval_lr in args.axis1_eval_lr:
            for seed in args.seeds:
                name = (
                    f"axis1_{adapt_set}_lr{_float_tag(eval_lr)}_"
                    f"{args.persistence_winner}_s{seed}"
                )
                cfg = _base_config(args, name=name, phase="axis1", seed=seed)
                cfg.update(
                    adapt_set=adapt_set,
                    persistence_mode=args.persistence_winner,
                    steps_per_chunk=args.steps_per_chunk,
                    eval_lr=eval_lr,
                )
                yield cfg


def _axis3_configs(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    seen: set[tuple[float, float]] = set()
    knobs: list[tuple[str, float, float]] = []
    for scale in args.delta_scale:
        pair = (scale, 0.0)
        if pair not in seen:
            seen.add(pair)
            knobs.append((f"delta_scale_{_float_tag(scale)}", scale, 0.0))
    for shift in args.log_a_shift:
        pair = (1.0, shift)
        if pair not in seen:
            seen.add(pair)
            knobs.append((f"log_a_shift_{_float_tag(shift)}", 1.0, shift))

    for label, scale, shift in knobs:
        for seed in args.seeds:
            name = f"axis3_{label}_{args.persistence_winner}_s{seed}"
            cfg = _base_config(args, name=name, phase="axis3", seed=seed)
            cfg.update(
                adapt_set="none",
                persistence_mode=args.persistence_winner,
                steps_per_chunk=0,
                delta_scale=scale,
                log_a_shift=shift,
            )
            yield cfg


def build_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    phases = ["floor", "axis1", "axis3"] if args.phase == "all" else [args.phase]
    configs: list[dict[str, Any]] = []
    for phase in phases:
        if phase == "floor":
            configs.extend(_floor_configs(args))
        elif phase == "axis1":
            configs.extend(_axis1_configs(args))
        elif phase == "axis3":
            configs.extend(_axis3_configs(args))
        else:
            raise ValueError(f"unknown phase: {phase}")
    return configs


def write_configs(args: argparse.Namespace, configs: list[dict[str, Any]]) -> dict[str, Any]:
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        Path(cfg["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        _config_path(args.config_dir, cfg["name"]).write_text(
            json.dumps(cfg, indent=2, sort_keys=True) + "\n"
        )

    counts = Counter(str(cfg["phase"]) for cfg in configs)
    manifest = {
        "matrix": args.matrix,
        "phase": args.phase,
        "total_configs": len(configs),
        "counts_by_phase": dict(sorted(counts.items())),
        "seeds": args.seeds,
        "checkpoint_path": str(args.checkpoint_path),
        "sp_model_path": str(args.sp_model_path),
        "jsonl_paths": [str(p) for p in args.jsonl_path],
        "output_root": str(args.output_root),
    }
    (args.config_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=["first_wave"], default="first_wave")
    parser.add_argument("--phase", choices=["all", "floor", "axis1", "axis3"], default="all")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--sp-model-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--persistence-winner", default="carry_state")
    parser.add_argument("--axis1-adapt-set", nargs="+", default=DEFAULT_AXIS1_ADAPT_SETS)
    parser.add_argument("--axis1-eval-lr", type=float, nargs="+", default=DEFAULT_AXIS1_EVAL_LRS)
    parser.add_argument("--delta-scale", type=float, nargs="+", default=DEFAULT_DELTA_SCALES)
    parser.add_argument("--log-a-shift", type=float, nargs="+", default=DEFAULT_LOG_A_SHIFTS)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--steps-per-chunk", type=int, default=1)
    parser.add_argument("--default-eval-lr", type=float, default=0.064)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-docs", type=int, default=50_000)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument("--score-floor-seconds", type=float, default=0.0)
    parser.add_argument("--safety-margin-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_configs(args)
    manifest = write_configs(args, configs)
    print(
        f"wrote {manifest['total_configs']} configs to {args.config_dir} "
        f"({manifest['counts_by_phase']})"
    )


if __name__ == "__main__":
    main()
