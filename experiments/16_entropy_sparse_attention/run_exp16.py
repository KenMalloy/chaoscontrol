#!/usr/bin/env python3
"""Experiment 16 Phase A matrix launcher.

Scaffolds the four planned oracle-probe conditions:
  - oracle_buf64_k4
  - oracle_buf64_k8
  - oracle_buf128_k4
  - oracle_buf128_k8

The launcher reuses the Exp 15 winner backbone and summarizes probe metrics
across seeds. The scheduler is intentionally simple: it launches up to
`num_gpus` subprocesses at a time and assigns GPUs round-robin.
"""
from __future__ import annotations

import argparse
import json
import os
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
        "oracle_eval_batches": 12,
        "oracle_max_examples": 4096,
        "oracle_layer_index": 3,
        "oracle_query_source": "x_state",
        "oracle_write_source": "x_state",
        "oracle_selector_epochs": 10,
        "oracle_selector_batch_size": 128,
        "oracle_selector_lr": 1e-3,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
        "base_lr": 2e-3,
        "sparse_attn_selector_dim": 0,
    }
    cfg.update(overrides)
    return cfg


CONDITIONS = {
    "oracle_buf64_k4": _base(sparse_attn_buffer_size=64, sparse_attn_k=4),
    "oracle_buf64_k8": _base(sparse_attn_buffer_size=64, sparse_attn_k=8),
    "oracle_buf128_k4": _base(sparse_attn_buffer_size=128, sparse_attn_k=4),
    "oracle_buf128_k8": _base(sparse_attn_buffer_size=128, sparse_attn_k=8),
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
                str(EXPERIMENT / "runner_exp16.py"),
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


def summarize_results(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for condition_name in conditions:
        files = sorted(RESULTS.glob(f"{condition_name}_s*.json"))
        if not files:
            continue
        mass_values: list[float] = []
        recall_values: list[float] = []
        eff_values: list[float] = []
        tk_mass_values: list[float] = []
        tk_recall_values: list[float] = []
        for file in files:
            data = json.loads(file.read_text())
            probe = data["oracle_probe"]
            mass_values.append(float(probe["selector_mass_capture_at_k"]))
            recall_values.append(float(probe["selector_recall_at_k"]))
            eff_values.append(float(probe["effective_connections"]))
            tk_mass_values.append(float(probe.get("token_keyed_mass_capture_at_k", 0.0)))
            tk_recall_values.append(float(probe.get("token_keyed_recall_at_k", 0.0)))
        mean_mass = sum(mass_values) / len(mass_values)
        ci = bootstrap_ci(mass_values)
        rows.append({
            "name": condition_name,
            "mean_mass": mean_mass,
            "se_mass": sem(mass_values),
            "ci": ci,
            "mass_values": mass_values,
            "recall_values": recall_values,
            "eff_values": eff_values,
            "tk_mass_values": tk_mass_values,
            "tk_recall_values": tk_recall_values,
        })

    rows.sort(key=lambda row: row["mean_mass"], reverse=True)
    if not rows:
        return summary

    print("\nRanked by selector mass capture@k")
    print("  condition              mean_mass    sem               95% CI        mean_recall   mean_eff_conn  tk_mass@k")
    for row in rows:
        mean_recall = sum(row["recall_values"]) / len(row["recall_values"])
        mean_eff = sum(row["eff_values"]) / len(row["eff_values"])
        mean_tk_mass = sum(row["tk_mass_values"]) / len(row["tk_mass_values"])
        print(
            f"  {row['name']:<20} {row['mean_mass']:9.4f} {row['se_mass']:7.4f} "
            f"[{row['ci'][0]:.4f}, {row['ci'][1]:.4f}] {mean_recall:12.4f} "
            f"{mean_eff:14.4f} {mean_tk_mass:10.4f}"
        )
        summary[row["name"]] = {
            "mean_mass_capture_at_k": row["mean_mass"],
            "sem_mass_capture_at_k": row["se_mass"],
            "ci_95_mass_capture_at_k": row["ci"],
            "mean_recall_at_k": mean_recall,
            "mean_effective_connections": mean_eff,
            "mean_token_keyed_mass_capture_at_k": mean_tk_mass,
            "mean_token_keyed_recall_at_k": sum(row["tk_recall_values"]) / len(row["tk_recall_values"]),
            "n_seeds": len(row["mass_values"]),
        }

    if len(rows) > 1:
        _, p = welch_ttest(rows[0]["mass_values"], rows[1]["mass_values"])
        summary["_decision"] = {
            "winner": rows[0]["name"],
            "runner_up": rows[1]["name"],
            "winner_vs_runner_up_p": p,
        }
        print(f"\nWinner vs runner-up p-value: {p:.4g}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 16 Phase A matrix launcher")
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
