#!/usr/bin/env python3
"""Experiment 18 Test 2 launcher: SP8192 vs SP16384 on the chunked scan backend.

Test 2 is the tokenizer revisit after Test 1 shipped the chunked vectorized
diag scan backend (7.77x over torch.compile, now the default diag path). SP16384
won Exp 15 SSM configs by 0.008 bpb then was shelved on a single OOM under the
old torch.compile backend; the chunked backend's VRAM profile is different, and
this test rediscovers the Exp 15 effect if it's real.

Launch convention (single spot H100):

    python experiments/18_throughput_levers/run_exp18.py \
        --sp8192-data-path /data/fineweb_sp8192 \
        --sp8192-model-path /models/sp8192.model \
        --sp16384-data-path /data/fineweb_sp16384 \
        --sp16384-model-path /models/sp16384.model \
        --budget 600 \
        --num-gpus 1

Both conditions share dim=256, layers=4, seq_len=512, batch=32, LR=2e-3,
7 matched seeds. The training data MUST be pre-tokenized with the matching
SentencePiece model: load_fineweb_tokens reads raw uint16 token-id shards and
the runner clamps to (vocab_size - 1), so feeding SP8192-tokenized shards to
the SP16384 model would silently leave embedding rows 8192-16383 unused and
bias the test.

Gate (from 2026-04-12-experiment-18-throughput-levers-design.md Test 2):
    SP16384 beats SP8192 at paired p<0.05 -> adopt SP16384 for Exp 19.
    Otherwise SP8192 stays. No goalpost moving at p<0.10.

Budget: ~4 GPU-hours on a single spot H100, ~$8.
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
sys.path.insert(0, str(REPO / "experiments" / "17_local_attn_sidecar"))
from stats import bootstrap_ci, paired_ttest, sem, welch_ttest  # noqa: E402
from runner_exp17 import build_child_env, validate_gpu_concurrency  # noqa: E402


# Exp 17 seed set — reused deliberately so Test 2 inherits the Exp 17 variance
# calibration (per-condition std ~0.004 bpb, 7 seeds detect >=0.008 bpb effect
# at 80% power, which matches the Exp 15 SP16384-vs-SP8192 gap exactly).
SWEEP_SEEDS = [1337, 2674, 4011, 5348, 6685, 8022, 9359]
ARTIFACT_LIMIT_BYTES = 16 * 1024 * 1024
TIMEOUT_MULTIPLIER = 2.5

RUNNER_SCRIPT = REPO / "experiments" / "17_local_attn_sidecar" / "runner_exp17.py"


def _base(**overrides: Any) -> dict[str, Any]:
    """Matched training envelope for both tokenizer conditions.

    Identical to Exp 17's `bare_fast_ssm` config (dim=256, L=4, bs=32, seq=512,
    LR=2e-3, diag a_mode) so the only varying axis is vocab_size / SP model /
    pre-tokenized data dir. The chunked scan backend is now the default diag
    path after Test 1 — no extra config flag needed to select it.
    """
    cfg = {
        "model_type": "ssm",
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


CONDITIONS: dict[str, dict[str, Any]] = {
    "bare_fast_ssm_sp8192": _base(vocab_size=8192),
    "bare_fast_ssm_sp16384": _base(vocab_size=16384),
}


# condition_name -> (data_path_key, sp_model_key) in the caller-supplied paths
# dict. Keeps the per-tokenizer path plumbing in one declarative place so new
# conditions only touch CONDITIONS and this table.
CONDITION_PATHS: dict[str, tuple[str, str]] = {
    "bare_fast_ssm_sp8192": ("sp8192_data_path", "sp8192_model_path"),
    "bare_fast_ssm_sp16384": ("sp16384_data_path", "sp16384_model_path"),
}


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
        if len(entry) > 5:
            entry[5].close()


def launch_matrix(
    *,
    paths: dict[str, str],
    budget: float,
    num_gpus: int,
    conditions: dict[str, dict[str, Any]],
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if num_gpus > 0:
        validate_gpu_concurrency(num_gpus)

    # Fail fast on missing paths and on tokenizer/data files that don't exist,
    # BEFORE spending queue-processing time on subprocess launches that would
    # only die inside the runner with a less specific error.
    for condition_name in conditions:
        data_key, model_key = CONDITION_PATHS[condition_name]
        for key in (data_key, model_key):
            if key not in paths or not paths[key]:
                raise ValueError(
                    f"Missing path for {condition_name}: --{key.replace('_', '-')} is required"
                )
        data_dir = Path(paths[data_key])
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"{condition_name}: data dir {data_dir} does not exist"
            )
        model_file = Path(paths[model_key])
        if not model_file.is_file():
            raise FileNotFoundError(
                f"{condition_name}: SP model file {model_file} does not exist"
            )

    queue: list[tuple[str, int, Path, str, str]] = []
    for condition_name, cfg in conditions.items():
        data_key, model_key = CONDITION_PATHS[condition_name]
        data_path = paths[data_key]
        sp_model_path = paths[model_key]
        for seed in SWEEP_SEEDS:
            out_path = RESULTS / f"{condition_name}_s{seed}.json"
            if out_path.exists():
                continue
            seed_cfg = dict(cfg, seed=seed)
            tmp = Path(tempfile.mkstemp(prefix=f"{condition_name}_s{seed}_", suffix=".yaml")[1])
            tmp.write_text(yaml.safe_dump(seed_cfg, sort_keys=False))
            queue.append((condition_name, seed, tmp, data_path, sp_model_path))

    run_timeout = max(budget * TIMEOUT_MULTIPLIER, 60.0)
    active: list = []
    gpu_cursor = 0

    while queue or active:
        while queue and len(active) < max(num_gpus, 1):
            condition_name, seed, cfg_path, data_path, sp_model_path = queue.pop(0)
            out_path = RESULTS / f"{condition_name}_s{seed}.json"
            env = os.environ.copy()
            env = build_child_env(
                gpu_slot=(gpu_cursor % num_gpus) if num_gpus > 0 else None,
                base_env=env,
            )
            gpu_cursor += 1
            log_path = RESULTS / f"{condition_name}_s{seed}.log"
            log_fh = open(log_path, "w")
            cmd = [
                sys.executable,
                "-u",
                str(RUNNER_SCRIPT),
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
            print(f"Launching {condition_name} seed={seed}", flush=True)
            proc = subprocess.Popen(cmd, env=env, text=True, stdout=log_fh, stderr=subprocess.STDOUT)
            active.append((proc, cfg_path, condition_name, seed, time.monotonic(), log_fh))

        next_active: list = []
        for i, (proc, cfg_path, condition_name, seed, t0, log_fh) in enumerate(active):
            ret = proc.poll()
            elapsed = time.monotonic() - t0
            if ret is None and elapsed < run_timeout:
                next_active.append((proc, cfg_path, condition_name, seed, t0, log_fh))
                continue
            log_fh.close()
            if ret is None:
                print(
                    f"TIMEOUT: {condition_name} seed={seed} after {elapsed:.0f}s "
                    f"(limit {run_timeout:.0f}s)",
                    flush=True,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                cfg_path.unlink(missing_ok=True)
                _cleanup_active(next_active + list(active[i + 1:]))
                raise RuntimeError(
                    f"{condition_name} seed={seed} TIMEOUT after {elapsed:.0f}s "
                    f"(budget={budget}s, limit={run_timeout:.0f}s)"
                )
            cfg_path.unlink(missing_ok=True)
            if ret != 0:
                log_path = RESULTS / f"{condition_name}_s{seed}.log"
                tail = ""
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    tail = "\n".join(lines[-20:])
                _cleanup_active(next_active + list(active[i + 1:]))
                raise RuntimeError(
                    f"{condition_name} seed={seed} failed with exit code {ret}\n"
                    f"--- last 20 lines of {log_path} ---\n{tail}"
                )
        active = next_active
        if active:
            time.sleep(2.0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_results(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for condition_name in conditions:
        pattern = re.compile(rf"^{re.escape(condition_name)}_s(\d+)\.json$")
        matches: list[tuple[int, Path]] = []
        if RESULTS.exists():
            for file in RESULTS.iterdir():
                m = pattern.match(file.name)
                if m:
                    matches.append((int(m.group(1)), file))
        matches.sort(key=lambda pair: pair[0])
        if not matches:
            continue
        bpb_by_seed: dict[int, float] = {}
        steps_values: list[float] = []
        artifact_values: list[int] = []
        for seed, file in matches:
            data = json.loads(file.read_text())
            bpb_by_seed[seed] = float(data["eval"]["bpb"])
            steps_values.append(float(data["train"]["steps_per_second"]))
            artifact_values.append(int(data.get("artifact_bytes", 0)))
        bpb_values = [bpb_by_seed[s] for s in sorted(bpb_by_seed)]
        rows.append({
            "name": condition_name,
            "bpb_by_seed": bpb_by_seed,
            "bpb_values": bpb_values,
            "mean_bpb": _mean(bpb_values),
            "se_bpb": sem(bpb_values),
            "ci_bpb": bootstrap_ci(bpb_values),
            "mean_steps_per_second": _mean(steps_values),
            "mean_artifact_bytes": _mean([float(x) for x in artifact_values]),
        })

    rows.sort(key=lambda row: row["mean_bpb"])
    if not rows:
        return summary

    print("\nTest 2 results — SP8192 vs SP16384 on chunked scan backend")
    print(
        f"  {'condition':<26} {'mean_bpb':>9} {'sem':>7} {'95% CI':>21} "
        f"{'steps/s':>9} {'artifact_mb':>12}"
    )
    for row in rows:
        print(
            f"  {row['name']:<26} {row['mean_bpb']:9.4f} {row['se_bpb']:7.4f} "
            f"[{row['ci_bpb'][0]:.4f}, {row['ci_bpb'][1]:.4f}] "
            f"{row['mean_steps_per_second']:9.2f} "
            f"{row['mean_artifact_bytes'] / 1e6:12.2f}"
        )
        summary[row["name"]] = {
            "mean_bpb": row["mean_bpb"],
            "sem_bpb": row["se_bpb"],
            "ci_95_bpb": row["ci_bpb"],
            "mean_steps_per_second": row["mean_steps_per_second"],
            "mean_artifact_bytes": row["mean_artifact_bytes"],
            "n_seeds": len(row["bpb_values"]),
        }

    # Paired t-test across matched seeds — the design doc's named stat.
    # SP8192 is the incumbent; SP16384 must beat it at paired p<0.05 to win.
    by_name = {row["name"]: row for row in rows}
    sp16k = by_name.get("bare_fast_ssm_sp16384")
    sp8k = by_name.get("bare_fast_ssm_sp8192")
    if sp16k is not None and sp8k is not None:
        shared_seeds = sorted(set(sp16k["bpb_by_seed"]) & set(sp8k["bpb_by_seed"]))
        if len(shared_seeds) >= 2:
            sp16k_paired = [sp16k["bpb_by_seed"][s] for s in shared_seeds]
            sp8k_paired = [sp8k["bpb_by_seed"][s] for s in shared_seeds]
            # Order (sp8k, sp16k): positive t-stat => sp8k > sp16k => SP16384 wins.
            t_paired, p_paired = paired_ttest(sp8k_paired, sp16k_paired)
            _, p_welch = welch_ttest(sp8k_paired, sp16k_paired)
            delta = sp8k["mean_bpb"] - sp16k["mean_bpb"]
            gate_pass = (delta > 0) and (p_paired < 0.05)
            summary["_decision"] = {
                "delta_sp8192_minus_sp16384_bpb": delta,
                "paired_t": t_paired,
                "paired_p": p_paired,
                "welch_p": p_welch,
                "n_paired_seeds": len(shared_seeds),
                "gate_sp16384_wins_p_lt_0.05": bool(gate_pass),
                "submission_tokenizer": "SP16384" if gate_pass else "SP8192",
            }
            print(
                f"\nPaired (n={len(shared_seeds)}): "
                f"SP8192 - SP16384 = {delta:+.4f} bpb  "
                f"t={t_paired:.3f}  p_paired={p_paired:.4g}  p_welch={p_welch:.4g}"
            )
            print(
                f"Gate: SP16384 adopted iff delta>0 and p_paired<0.05 -> "
                f"{summary['_decision']['submission_tokenizer']}"
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 2 launcher — SP8192 vs SP16384 on chunked scan"
    )
    parser.add_argument("--sp8192-data-path", required=True,
                        help="Directory with fineweb_*_*.bin shards tokenized by SP8192")
    parser.add_argument("--sp8192-model-path", required=True,
                        help="Path to the SP8192 SentencePiece model file")
    parser.add_argument("--sp16384-data-path", required=True,
                        help="Directory with fineweb_*_*.bin shards tokenized by SP16384")
    parser.add_argument("--sp16384-model-path", required=True,
                        help="Path to the SP16384 SentencePiece model file")
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    paths = {
        "sp8192_data_path": args.sp8192_data_path,
        "sp8192_model_path": args.sp8192_model_path,
        "sp16384_data_path": args.sp16384_data_path,
        "sp16384_model_path": args.sp16384_model_path,
    }

    if not args.summarize_only:
        launch_matrix(
            paths=paths,
            budget=args.budget,
            num_gpus=args.num_gpus,
            conditions=CONDITIONS,
        )

    summary = summarize_results(CONDITIONS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "test2_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
