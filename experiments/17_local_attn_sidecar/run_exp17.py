#!/usr/bin/env python3
"""Experiment 17 Phase A matrix launcher."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / "results"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
from stats import bootstrap_ci, sem, welch_ttest


SWEEP_SEEDS = [1337, 2674, 4011, 5348, 6685, 8022, 9359]
ARTIFACT_LIMIT_BYTES = 16 * 1024 * 1024


def _base(**overrides: Any) -> dict[str, Any]:
    cfg = {
        "model_type": "ssm",
        "vocab_size": 8192,
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 32,
        "eval_batches": 16,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
        "base_lr": 2e-3,
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
    }
    cfg.update(overrides)
    return cfg


CONDITIONS = {
    "bare_fast_ssm": _base(local_attn_window=0),
    "local_w16": _base(local_attn_window=16, local_attn_heads=1, local_attn_dim=64),
    "local_w32": _base(local_attn_window=32, local_attn_heads=1, local_attn_dim=64),
    "local_w64": _base(local_attn_window=64, local_attn_heads=1, local_attn_dim=64),
}


def launch_matrix(
    *,
    data_path: str,
    sp_model_path: str,
    budget: float,
    num_gpus: int,
    conditions: dict[str, dict[str, Any]],
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    queue: list[tuple[str, int, Path]] = []
    for condition_name, cfg in conditions.items():
        for seed in SWEEP_SEEDS:
            out_path = RESULTS / f"{condition_name}_s{seed}.json"
            if out_path.exists():
                continue
            seed_cfg = dict(cfg, seed=seed)
            tmp = Path(tempfile.mkstemp(prefix=f"{condition_name}_s{seed}_", suffix=".yaml")[1])
            tmp.write_text(yaml.safe_dump(seed_cfg, sort_keys=False))
            queue.append((condition_name, seed, tmp))

    active: list[tuple[subprocess.Popen[str], Path, str, int]] = []
    gpu_cursor = 0

    while queue or active:
        while queue and len(active) < max(num_gpus, 1):
            condition_name, seed, cfg_path = queue.pop(0)
            out_path = RESULTS / f"{condition_name}_s{seed}.json"
            env = os.environ.copy()
            if num_gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_cursor % num_gpus)
            gpu_cursor += 1
            cmd = [
                sys.executable,
                str(EXPERIMENT / "runner_exp17.py"),
                "--config",
                str(cfg_path),
                "--data-path",
                data_path,
                "--sp-model-path",
                sp_model_path,
                "--budget",
                str(budget),
                "--output-json",
                str(out_path),
            ]
            print(f"Launching {condition_name} seed={seed}")
            proc = subprocess.Popen(cmd, env=env, text=True)
            active.append((proc, cfg_path, condition_name, seed))

        next_active: list[tuple[subprocess.Popen[str], Path, str, int]] = []
        for proc, cfg_path, condition_name, seed in active:
            ret = proc.poll()
            if ret is None:
                next_active.append((proc, cfg_path, condition_name, seed))
                continue
            cfg_path.unlink(missing_ok=True)
            if ret != 0:
                raise RuntimeError(f"{condition_name} seed={seed} failed with exit code {ret}")
        active = next_active
        if active:
            time.sleep(2.0)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def summarize_results(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for condition_name in conditions:
        pattern = re.compile(rf"^{re.escape(condition_name)}_s\d+\.json$")
        files = sorted(file for file in RESULTS.iterdir() if pattern.match(file.name))
        if not files:
            continue
        bpb_values: list[float] = []
        steps_values: list[float] = []
        artifact_values: list[int] = []
        for file in files:
            data = json.loads(file.read_text())
            bpb_values.append(float(data["eval"]["bpb"]))
            steps_values.append(float(data["train"]["steps_per_second"]))
            artifact_values.append(int(data["artifact_bytes"]))
        rows.append({
            "name": condition_name,
            "bpb_values": bpb_values,
            "steps_values": steps_values,
            "artifact_values": artifact_values,
            "mean_bpb": _mean(bpb_values),
            "mean_steps_per_second": _mean(steps_values),
            "mean_artifact_bytes": _mean([float(x) for x in artifact_values]),
            "se_bpb": sem(bpb_values),
            "ci_bpb": bootstrap_ci(bpb_values),
        })

    rows.sort(key=lambda row: row["mean_bpb"])
    if not rows:
        return summary

    bare = next((row for row in rows if row["name"] == "bare_fast_ssm"), None)
    bare_bpb_values = bare["bpb_values"] if bare is not None else []
    bare_mean_bpb = bare["mean_bpb"] if bare is not None else float("nan")
    bare_mean_steps = bare["mean_steps_per_second"] if bare is not None else float("nan")

    print("\nPer-condition results (ranked by mean bpb)")
    print(
        f"  {'condition':<16} {'mean_bpb':>9} {'sem':>7} {'95% CI':>21} "
        f"{'steps/s':>9} {'delta_vs_bare':>14} {'p':>9} {'artifact_mb':>12}"
    )
    for row in rows:
        delta_vs_bare = float("nan")
        p_value = float("nan")
        if bare is not None and row["name"] != "bare_fast_ssm":
            delta_vs_bare = bare_mean_bpb - row["mean_bpb"]
            _, p_value = welch_ttest(bare_bpb_values, row["bpb_values"])
        print(
            f"  {row['name']:<16} {row['mean_bpb']:9.4f} {row['se_bpb']:7.4f} "
            f"[{row['ci_bpb'][0]:.4f}, {row['ci_bpb'][1]:.4f}] {row['mean_steps_per_second']:9.2f} "
            f"{delta_vs_bare:14.4f} {p_value:9.4g} {row['mean_artifact_bytes'] / 1e6:12.2f}"
        )
        summary[row["name"]] = {
            "mean_bpb": row["mean_bpb"],
            "sem_bpb": row["se_bpb"],
            "ci_95_bpb": row["ci_bpb"],
            "mean_steps_per_second": row["mean_steps_per_second"],
            "mean_artifact_bytes": row["mean_artifact_bytes"],
            "delta_bpb_vs_bare": delta_vs_bare,
            "p_vs_bare": p_value,
            "n_seeds": len(row["bpb_values"]),
        }

    local_rows = [row for row in rows if row["name"] != "bare_fast_ssm"]
    passing_rows: list[dict[str, Any]] = []
    for row in local_rows:
        delta_vs_bare = bare_mean_bpb - row["mean_bpb"]
        _, p_value = welch_ttest(bare_bpb_values, row["bpb_values"])
        gate_gain = delta_vs_bare >= 0.02 and p_value < 0.05
        gate_speed = row["mean_steps_per_second"] >= 0.5 * bare_mean_steps
        gate_artifact = row["mean_artifact_bytes"] < ARTIFACT_LIMIT_BYTES
        if gate_gain and gate_speed and gate_artifact:
            passing_rows.append(row)

    best = passing_rows[0] if passing_rows else (local_rows[0] if local_rows else rows[0])
    best_delta_vs_bare = bare_mean_bpb - best["mean_bpb"] if bare is not None and best["name"] != "bare_fast_ssm" else 0.0
    best_p = float("nan")
    if bare is not None and best["name"] != "bare_fast_ssm":
        _, best_p = welch_ttest(bare_bpb_values, best["bpb_values"])
    summary["_decision"] = {
        "best_condition": best["name"],
        "gate_gain_ge_0.02_bpb": best_delta_vs_bare >= 0.02 and best_p < 0.05,
        "gate_steps_per_second": best["mean_steps_per_second"] >= 0.5 * bare_mean_steps if bare is not None else False,
        "gate_artifact_bytes": best["mean_artifact_bytes"] < ARTIFACT_LIMIT_BYTES,
        "n_passing_conditions": len(passing_rows),
        "all_gates_pass": best in passing_rows,
    }
    print(
        f"\nGo/no-go: gain>=0.02+p<0.05: {summary['_decision']['gate_gain_ge_0.02_bpb']} | "
        f"speed>=50%: {summary['_decision']['gate_steps_per_second']} | "
        f"artifact<16MB: {summary['_decision']['gate_artifact_bytes']}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 17 Phase A matrix launcher")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if not args.summarize_only:
        launch_matrix(
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            num_gpus=args.num_gpus,
            conditions=CONDITIONS,
        )

    summary = summarize_results(CONDITIONS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase_a_summary.json").write_text(json.dumps(summary, indent=2, default=str))
