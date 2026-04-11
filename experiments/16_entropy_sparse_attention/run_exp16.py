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
    # Buffer size × k sweep (all x_state)
    "oracle_buf64_k4": _base(sparse_attn_buffer_size=64, sparse_attn_k=4),
    "oracle_buf64_k8": _base(sparse_attn_buffer_size=64, sparse_attn_k=8),
    "oracle_buf128_k4": _base(sparse_attn_buffer_size=128, sparse_attn_k=4),
    "oracle_buf128_k8": _base(sparse_attn_buffer_size=128, sparse_attn_k=8),
    # Feature-source ablations at buf128_k8: answers the central question
    # "does recurrent state add selector value beyond non-SSM features?"
    "oracle_buf128_k8_xonly": _base(
        sparse_attn_buffer_size=128, sparse_attn_k=8,
        oracle_query_source="x", oracle_write_source="x",
    ),
    "oracle_buf128_k8_stateonly": _base(
        sparse_attn_buffer_size=128, sparse_attn_k=8,
        oracle_query_source="state", oracle_write_source="state",
    ),
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
        mass_values: list[float] = []
        recall_values: list[float] = []
        eff_values: list[float] = []
        recent_mass_values: list[float] = []
        tk_mass_values: list[float] = []
        # Per-seed deltas: selector minus baseline
        delta_recent: list[float] = []
        delta_tk: list[float] = []
        for file in files:
            data = json.loads(file.read_text())
            probe = data["oracle_probe"]
            sel_mass = float(probe["selector_mass_capture_at_k"])
            rec_mass = float(probe.get("recent_mass_capture_at_k", 0.0))
            tk_mass = float(probe.get("token_keyed_mass_capture_at_k", 0.0))
            mass_values.append(sel_mass)
            recall_values.append(float(probe["selector_recall_at_k"]))
            eff_values.append(float(probe["effective_connections"]))
            recent_mass_values.append(rec_mass)
            tk_mass_values.append(tk_mass)
            delta_recent.append(sel_mass - rec_mass)
            delta_tk.append(sel_mass - tk_mass)
        ci = bootstrap_ci(mass_values)
        rows.append({
            "name": condition_name,
            "mean_mass": _mean(mass_values),
            "se_mass": sem(mass_values),
            "ci": ci,
            "mass_values": mass_values,
            "recall_values": recall_values,
            "eff_values": eff_values,
            "recent_mass_values": recent_mass_values,
            "tk_mass_values": tk_mass_values,
            "delta_recent": delta_recent,
            "delta_tk": delta_tk,
        })

    rows.sort(key=lambda row: row["mean_mass"], reverse=True)
    if not rows:
        return summary

    # --- Table 1: per-condition selector mass and baseline deltas ---
    print("\nPer-condition results (ranked by selector mass capture@k)")
    print(f"  {'condition':<30} {'sel_mass':>9} {'recent':>9} {'tk':>9} "
          f"{'sel-recent':>11} {'sel-tk':>11} {'eff_conn':>9}")
    for row in rows:
        mean_sel = row["mean_mass"]
        mean_rec = _mean(row["recent_mass_values"])
        mean_tk = _mean(row["tk_mass_values"])
        mean_dr = _mean(row["delta_recent"])
        mean_dt = _mean(row["delta_tk"])
        mean_eff = _mean(row["eff_values"])
        print(
            f"  {row['name']:<30} {mean_sel:9.4f} {mean_rec:9.4f} {mean_tk:9.4f} "
            f"{mean_dr:+11.4f} {mean_dt:+11.4f} {mean_eff:9.2f}"
        )
        summary[row["name"]] = {
            "mean_mass_capture_at_k": mean_sel,
            "sem_mass_capture_at_k": row["se_mass"],
            "ci_95_mass_capture_at_k": row["ci"],
            "mean_recall_at_k": _mean(row["recall_values"]),
            "mean_effective_connections": mean_eff,
            "mean_recent_mass_capture_at_k": mean_rec,
            "mean_token_keyed_mass_capture_at_k": mean_tk,
            "mean_delta_vs_recent": mean_dr,
            "mean_delta_vs_token_keyed": mean_dt,
            "n_seeds": len(row["mass_values"]),
        }

    # --- Selector vs baseline significance tests ---
    print("\nSelector vs baseline tests (paired per-seed deltas)")
    for row in rows:
        name = row["name"]
        if not row["delta_recent"]:
            continue
        dr = row["delta_recent"]
        dt = row["delta_tk"]
        # One-sample test: is the per-seed delta > 0?
        if len(dr) >= 3:
            _, p_rec = welch_ttest(dr, [0.0] * len(dr))
            print(f"  {name:<30} sel-recent: {_mean(dr):+.4f} (p={p_rec:.4g})")
        if len(dt) >= 3:
            _, p_tk = welch_ttest(dt, [0.0] * len(dt))
            print(f"  {'':<30} sel-tk:     {_mean(dt):+.4f} (p={p_tk:.4g})")

    # --- Feature-source comparison (if ablations ran) ---
    xstate_rows = [r for r in rows if "xonly" not in r["name"] and "stateonly" not in r["name"]]
    xonly_rows = [r for r in rows if "xonly" in r["name"]]
    stateonly_rows = [r for r in rows if "stateonly" in r["name"]]
    if xstate_rows and (xonly_rows or stateonly_rows):
        print("\nFeature-source ablation (buf128_k8)")
        ref = next((r for r in xstate_rows if "buf128_k8" in r["name"]), xstate_rows[0])
        for ablation in xonly_rows + stateonly_rows:
            if len(ref["mass_values"]) >= 3 and len(ablation["mass_values"]) >= 3:
                _, p = welch_ttest(ref["mass_values"], ablation["mass_values"])
                diff = ref["mean_mass"] - ablation["mean_mass"]
                print(f"  x_state vs {ablation['name'].split('_')[-1]}: "
                      f"delta={diff:+.4f} (p={p:.4g})")

    # --- Go/no-go decision ---
    gated_rows: list[dict[str, Any]] = []
    for row in rows:
        k = conditions[row["name"]].get("sparse_attn_k", 8)
        gate_mass = row["mean_mass"] >= 0.60
        gate_beats_recent = _mean(row["delta_recent"]) > 0
        gate_beats_tk = _mean(row["delta_tk"]) > 0
        gate_eff_conn = _mean(row["eff_values"]) <= 2 * k
        if gate_mass and gate_beats_recent and gate_beats_tk and gate_eff_conn:
            gated_rows.append(row)

    best = gated_rows[0] if gated_rows else rows[0]
    gate_mass = best["mean_mass"] >= 0.60
    gate_beats_recent = _mean(best["delta_recent"]) > 0
    gate_beats_tk = _mean(best["delta_tk"]) > 0
    gate_eff_conn = _mean(best["eff_values"]) <= 2 * conditions[best["name"]].get("sparse_attn_k", 8)
    summary["_decision"] = {
        "best_condition": best["name"],
        "gate_mass_capture_ge_0.60": gate_mass,
        "gate_beats_recent_k": gate_beats_recent,
        "gate_beats_token_keyed": gate_beats_tk,
        "gate_effective_connections": gate_eff_conn,
        "n_passing_conditions": len(gated_rows),
        "all_gates_pass": gate_mass and gate_beats_recent and gate_beats_tk and gate_eff_conn,
    }
    print(f"\nGo/no-go: mass>={0.60}: {gate_mass} | beats recent: {gate_beats_recent} | "
          f"beats token_keyed: {gate_beats_tk} | eff_conn: {gate_eff_conn}")
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
