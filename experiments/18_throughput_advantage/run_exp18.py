#!/usr/bin/env python3
"""Experiment 18 Phase A launcher."""
from __future__ import annotations

import argparse
import json
import os
import statistics
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
RUNNER = EXPERIMENT / "runner_exp18.py"

sys.path.insert(0, str(EXPERIMENT))

from runner_exp18 import (  # noqa: E402
    SMOKE_TOTAL_BUDGET_S,
    build_child_env,
    build_phase_a_conditions,
    make_smoke_summary,
    validate_gpu_concurrency,
)


SEEDS = [1337, 2674, 4011, 5348, 6685, 8022, 9359]
TIMEOUT_MULTIPLIER = 2.5


def _launch_job(
    *,
    condition_name: str,
    config: dict[str, Any],
    phase0_summary_path: str,
    gpu_id: int | None,
    output_json: Path,
    smoke: bool,
    log_fh,
) -> tuple[subprocess.Popen[str], Path]:
    tmp_cfg = Path(tempfile.mkstemp(prefix=f"{condition_name}_", suffix=".yaml")[1])
    tmp_cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    cmd = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--config",
        str(tmp_cfg),
        "--phase0-summary",
        phase0_summary_path,
        "--output-json",
        str(output_json),
    ]
    if smoke:
        cmd.append("--smoke")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env = build_child_env(gpu_slot=gpu_id, smoke=smoke, base_env=env)
    proc = subprocess.Popen(cmd, env=env, text=True, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, tmp_cfg


def _cleanup_active(active: list) -> None:
    for entry in active:
        proc, cfg_path = entry[0], entry[1]
        if proc.poll() is None:
            proc.terminate()
        cfg_path.unlink(missing_ok=True)
    for entry in active:
        proc = entry[0]
        if proc.poll() is None:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        if len(entry) > 6:
            entry[6].close()


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_condition.setdefault(row["condition"], []).append(row)

    summary: dict[str, Any] = {}
    for condition, rows in by_condition.items():
        bpbs = [float(row["eval"]["bpb"]) for row in rows]
        rescore_fracs = [float(row["timings"]["rescore_frac_of_budget"]) for row in rows]
        selection_overheads = [float(row["timings"]["selection_overhead_s"]) for row in rows]
        summary[condition] = {
            "mean_bpb": statistics.mean(bpbs),
            "n_seeds": len(rows),
            "mean_rescore_frac_of_budget": statistics.mean(rescore_fracs),
            "mean_selection_overhead_s": statistics.mean(selection_overheads),
        }

    if "baseline_b32" in summary and "sweep_only" in summary:
        summary["_paired"] = {
            "sweep_only_minus_baseline_b32": summary["sweep_only"]["mean_bpb"] - summary["baseline_b32"]["mean_bpb"],
        }
        if "sweep_target_top10" in summary:
            summary["_paired"]["sweep_target_top10_minus_sweep_only"] = (
                summary["sweep_target_top10"]["mean_bpb"] - summary["sweep_only"]["mean_bpb"]
            )
        if "sweep_random_retrain" in summary and "sweep_target_top10" in summary:
            summary["_paired"]["sweep_target_top10_minus_sweep_random_retrain"] = (
                summary["sweep_target_top10"]["mean_bpb"] - summary["sweep_random_retrain"]["mean_bpb"]
            )
    return summary


def run_phase_a(
    *,
    phase0_summary_path: str,
    budget_s: float,
    num_gpus: int,
    smoke: bool,
) -> dict[str, Any]:
    summary = json.loads(Path(phase0_summary_path).read_text())
    conditions = build_phase_a_conditions(summary, total_budget_s=budget_s)
    if smoke:
        # Keep smoke fast but exercise every condition once.
        phase0_summary_path = str(Path(tempfile.mkstemp(prefix="exp18_phase0_smoke_", suffix=".json")[1]))
        Path(phase0_summary_path).write_text(json.dumps(make_smoke_summary(total_budget_s=budget_s), indent=2))

    RESULTS.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, dict[str, Any], int]] = []
    # Each active entry: (proc, tmp_cfg, condition_name, seed, out_path, t0, log_fh)
    active: list[tuple] = []
    seeds = [1337] if smoke else SEEDS
    if not smoke:
        validate_gpu_concurrency(max(num_gpus, 1))

    for condition_name, cfg in conditions.items():
        for seed in seeds:
            out_path = RESULTS / f"{condition_name}_s{seed}.json"
            if out_path.exists() and not smoke:
                continue
            seed_cfg = dict(cfg)
            seed_cfg["model_config"] = dict(cfg["model_config"], seed=seed)
            pending.append((condition_name, seed_cfg, seed))

    run_timeout = budget_s * TIMEOUT_MULTIPLIER
    gpu_cursor = 0
    while pending or active:
        while pending and len(active) < max(num_gpus, 1):
            condition_name, cfg, seed = pending.pop(0)
            out_path = RESULTS / f"{condition_name}_s{seed}.json"
            log_path = RESULTS / f"{condition_name}_s{seed}.log"
            log_fh = open(log_path, "w")
            proc, tmp_cfg = _launch_job(
                condition_name=condition_name,
                config=cfg,
                phase0_summary_path=phase0_summary_path,
                gpu_id=(gpu_cursor % num_gpus) if num_gpus > 0 else None,
                output_json=out_path,
                smoke=smoke,
                log_fh=log_fh,
            )
            print(f"Launching {condition_name} seed={seed}", flush=True)
            active.append((proc, tmp_cfg, condition_name, seed, out_path, time.monotonic(), log_fh))
            gpu_cursor += 1

        next_active: list[tuple] = []
        for proc, tmp_cfg, condition_name, seed, out_path, t0, log_fh in active:
            ret = proc.poll()
            elapsed = time.monotonic() - t0
            if ret is None and elapsed < run_timeout:
                next_active.append((proc, tmp_cfg, condition_name, seed, out_path, t0, log_fh))
                continue
            log_fh.close()
            if ret is None:
                print(f"TIMEOUT: {condition_name} seed={seed} after {elapsed:.0f}s (limit {run_timeout:.0f}s)", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                tmp_cfg.unlink(missing_ok=True)
                _cleanup_active(next_active)
                raise RuntimeError(
                    f"{condition_name} seed={seed} TIMEOUT after {elapsed:.0f}s "
                    f"(budget={budget_s}s, limit={run_timeout:.0f}s)"
                )
            tmp_cfg.unlink(missing_ok=True)
            if ret != 0:
                log_path = RESULTS / f"{condition_name}_s{seed}.log"
                tail = ""
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    tail = "\n".join(lines[-20:])
                _cleanup_active(next_active)
                raise RuntimeError(
                    f"{condition_name} seed={seed} failed with exit code {ret}\n"
                    f"--- last 20 lines of {log_path} ---\n{tail}"
                )
        active = next_active
        if active:
            time.sleep(1.0)

    result_rows = []
    for condition_name in conditions:
        for seed in seeds:
            out_path = RESULTS / f"{condition_name}_s{seed}.json"
            result_rows.append(json.loads(out_path.read_text()))
    summary_out = summarize_results(result_rows)
    path = RESULTS / ("phase_a_summary.smoke.json" if smoke else "phase_a_summary.json")
    path.write_text(json.dumps(summary_out, indent=2))
    return summary_out


def main() -> None:
    p = argparse.ArgumentParser(description="Exp 18 Phase A launcher")
    p.add_argument("--phase0-summary", help="Phase 0 summary JSON path")
    p.add_argument("--budget", type=float, default=600.0)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if not args.phase0_summary and not args.smoke:
        raise SystemExit("--phase0-summary is required unless --smoke is used")
    phase0_summary = args.phase0_summary
    if args.smoke:
        tmp = Path(tempfile.mkstemp(prefix="exp18_phase0_", suffix=".json")[1])
        smoke_budget = min(args.budget, SMOKE_TOTAL_BUDGET_S)
        tmp.write_text(json.dumps(make_smoke_summary(total_budget_s=smoke_budget), indent=2))
        phase0_summary = str(tmp)

    assert phase0_summary is not None
    summary = run_phase_a(
        phase0_summary_path=phase0_summary,
        budget_s=min(args.budget, SMOKE_TOTAL_BUDGET_S) if args.smoke else args.budget,
        num_gpus=args.num_gpus,
        smoke=args.smoke,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
