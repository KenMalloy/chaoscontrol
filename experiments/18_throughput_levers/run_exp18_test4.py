#!/usr/bin/env python3
"""Experiment 18 Test 4 launcher: DDP scaling efficiency.

Test 4 measures whether DDP actually translates extra hardware into
better bpb-at-wall-clock. Three world-size conditions over the same
per-rank batch size:

    ws=1  single-device baseline  (global batch = bs * 1 = 1024)
    ws=2  2-rank DDP on 2 GPUs     (global batch = bs * 2 = 2048)
    ws=4  4-rank DDP on 4 GPUs     (global batch = bs * 4 = 4096)

**LR is linearly scaled per condition** from the Exp 17 / Exp 18 phase0
anchor ``(bs=32, lr=2e-3)``. That means ``LR = 2e-3 * (global_batch / 32)``,
so ws=1 runs at 0.064, ws=2 at 0.128, ws=4 at 0.256. Using a single flat
LR across conditions would make the bpb-half of the gate measure "LR
appropriateness" instead of "DDP scaling translates to learning". The
per-ws LR is aggressive at ws=4 and may diverge — if it does, that is
information about linear-scaling breakdown, not a Test 4 failure.

A DDP configuration earns the gate when BOTH:
    (a) per-GPU tok/s at ws=N is at least 85% of ws=1 per-GPU tok/s, AND
    (b) mean bpb-at-600s at ws=N is statistically lower than ws=1 on
        matched seeds (paired t-test, p<0.05).

The gate is not a "proportional learning gain" claim; it is simply
"did DDP produce statistically significant bpb improvement at matched
wall-clock". Proportionality of learning gain to token gain is a
separate, noisier measurement we do not attempt here.

**OOM policy.** Unlike Tests 3 and 6 which deliberately push the VRAM
ceiling, Test 4's conditions use the same bs=1024 per rank at every
world size. An OOM at any ws is not a ceiling-push signal — it is a
misconfiguration (typically DDP buffer allocation or gradient bucket
sizing). Test 4 therefore **hard-fails on any non-zero exit**: the
run is aborted and the whole matrix exits. We want to know about
that loudly, not continue with garbage results.

**Seed parallelism differs by world size** because each DDP group
claims all its GPUs for the duration of the run. On a 4-GPU pod:
    - ws=1: 4 seeds fit in 1 wave (4 concurrent single-GPU runs)
    - ws=2: 4 seeds fit in 2 waves (2 concurrent DDP groups x 2 GPUs)
    - ws=4: 4 seeds run serially (1 DDP group per wave)

Conditions run in ascending world-size order so the parallel cases
finish first and only the serial ws=4 runs drag wall-clock at the end.

Budget: ~70 min wall-clock on a 4-GPU pod, ~$10.
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
RESULTS = EXPERIMENT / "results_test4"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
sys.path.insert(0, str(EXPERIMENT))
from stats import bootstrap_ci, paired_ttest, sem  # noqa: E402
from _harness import (  # noqa: E402
    build_env_with_gpu_mask,
    build_launch_cmd,
    pick_free_port,
    validate_data_paths,
)

SWEEP_SEEDS = [1337, 2674, 4011, 5348]
TIMEOUT_MULTIPLIER = 2.5

# Linear LR scaling anchor: Exp 17 / Exp 18 phase0 established
# (bs=32, lr=2e-3) as the stable per-example learning rate. Every LR
# in this test is derived as 2e-3 * (global_batch / 32) so that each
# world-size condition is at its linearly-scaled target, not at a
# single flat LR that would under-train the higher-ws conditions and
# make the bpb gate measure LR appropriateness instead of DDP scaling.
LR_ANCHOR_BASE = 2e-3
LR_ANCHOR_BATCH = 32


def _linear_scaled_lr(global_batch: int) -> float:
    return LR_ANCHOR_BASE * (global_batch / LR_ANCHOR_BATCH)


BATCH_PER_RANK = 1024


def _base(world_size: int, **overrides: Any) -> dict[str, Any]:
    """Per-rank config with world-size-aware linear LR scaling.

    ``world_size`` is used both as a sidecar the orchestrator reads to
    decide the launch pattern AND to compute the condition's LR via
    linear scaling from the phase0 anchor. torchrun's env vars still
    carry WORLD_SIZE/RANK at runtime; this argument exists so the
    orchestrator can set the right LR per condition at yaml-write time.
    """
    global_batch = BATCH_PER_RANK * world_size
    cfg = {
        "model_type": "ssm",
        "vocab_size": 16384,
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": BATCH_PER_RANK,
        "eval_batches": 16,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
        "base_lr": _linear_scaled_lr(global_batch),
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
    }
    cfg.update(overrides)
    return cfg


# The (condition_name -> (world_size, config)) mapping.
# Per-condition LRs (bs=1024 per rank, linearly scaled from bs=32):
#   ws=1: global 1024 -> LR 0.064
#   ws=2: global 2048 -> LR 0.128
#   ws=4: global 4096 -> LR 0.256
CONDITIONS: dict[str, tuple[int, dict[str, Any]]] = {
    "ws1":       (1, _base(world_size=1)),
    "ws2_ddp":   (2, _base(world_size=2)),
    "ws4_ddp":   (4, _base(world_size=4)),
}


def _cleanup_active(active: list) -> None:
    """Terminate any still-running procs and close their log file handles.

    Entries here are 6-tuples ``(proc, cfg_path, seed, t0, log_fh, slot_id)``.
    The ``log_fh`` is at index 4 and is always closed unconditionally —
    the previous ``len(entry) > 5`` guard was dead code because tuples
    from this caller have exactly 6 elements with log_fh at position 4,
    so it never fired.
    """
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
        log_fh = entry[4]
        if not log_fh.closed:
            log_fh.close()


def _run_condition_group(
    *,
    condition_name: str,
    world_size: int,
    cfg: dict[str, Any],
    data_path: str,
    sp_model_path: str,
    budget: float,
    num_gpus_total: int,
) -> None:
    """Run all seeds of one condition with slot parallelism appropriate to
    its ``world_size``. Blocks until every seed for this condition has
    either produced a result JSON or failed hard.

    Slot accounting uses an explicit free-slot set, not a ``slot_cursor
    % max_slots`` round-robin. The cursor scheme silently aliases two
    concurrent runs to the same GPU mask whenever runs finish out-of-
    order (a near-certainty for Test 5/6/7 where per-step costs differ
    across conditions). Test 4's ws=4 case has max_slots=1 so the bug
    wouldn't trigger there, but ws=1 (max_slots=4) and ws=2 (max_slots=2)
    are both exposed; fixing all three code paths uniformly is cheaper
    than reasoning about which ones are safe.
    """
    max_slots = max(num_gpus_total // world_size, 1)
    run_timeout = max(budget * TIMEOUT_MULTIPLIER, 60.0)

    queue: list[tuple[int, Path]] = []
    for seed in SWEEP_SEEDS:
        out_path = RESULTS / f"{condition_name}_s{seed}.json"
        if out_path.exists():
            continue
        seed_cfg = dict(cfg, seed=seed)
        tmp = Path(tempfile.mkstemp(prefix=f"{condition_name}_s{seed}_", suffix=".yaml")[1])
        tmp.write_text(yaml.safe_dump(seed_cfg, sort_keys=False))
        queue.append((seed, tmp))

    print(
        f"[{condition_name}] ws={world_size} max_slots={max_slots} "
        f"queued={len(queue)}",
        flush=True,
    )

    free_slots: list[int] = list(range(max_slots))
    active: list = []  # (proc, cfg_path, seed, t0, log_fh, slot_id)

    while queue or active:
        while queue and free_slots:
            seed, cfg_path = queue.pop(0)
            out_path = RESULTS / f"{condition_name}_s{seed}.json"
            log_path = RESULTS / f"{condition_name}_s{seed}.log"
            log_fh = open(log_path, "w")
            slot_id = free_slots.pop(0)  # smallest free
            gpu_ids = [slot_id * world_size + i for i in range(world_size)]
            env = build_env_with_gpu_mask(gpu_ids)
            if world_size == 1:
                cmd = build_launch_cmd(
                    num_gpus=1,
                    cfg_path=cfg_path,
                    data_path=data_path,
                    sp_model_path=sp_model_path,
                    budget=budget,
                    out_path=out_path,
                )
            else:
                rdzv = pick_free_port()
                cmd = build_launch_cmd(
                    num_gpus=world_size,
                    cfg_path=cfg_path,
                    data_path=data_path,
                    sp_model_path=sp_model_path,
                    budget=budget,
                    out_path=out_path,
                    rdzv_port=rdzv,
                )
            print(
                f"Launching {condition_name} seed={seed} slot={slot_id} "
                f"gpus={gpu_ids}",
                flush=True,
            )
            proc = subprocess.Popen(
                cmd, env=env, text=True, stdout=log_fh, stderr=subprocess.STDOUT,
            )
            active.append(
                (proc, cfg_path, seed, time.monotonic(), log_fh, slot_id)
            )

        next_active: list = []
        for i, entry in enumerate(active):
            proc, cfg_path, seed, t0, log_fh, slot_id = entry
            ret = proc.poll()
            elapsed = time.monotonic() - t0
            if ret is None and elapsed < run_timeout:
                next_active.append(entry)
                continue
            log_fh.close()
            free_slots.append(slot_id)
            free_slots.sort()
            if ret is None:
                print(
                    f"TIMEOUT: {condition_name} seed={seed} after {elapsed:.0f}s",
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
                    f"{condition_name} seed={seed} TIMEOUT after {elapsed:.0f}s"
                )
            cfg_path.unlink(missing_ok=True)
            if ret != 0:
                log_path = RESULTS / f"{condition_name}_s{seed}.log"
                tail = ""
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    tail = "\n".join(lines[-20:])
                _cleanup_active(next_active + list(active[i + 1:]))
                # Test 4 hard-fails on any non-zero exit, including OOM.
                # Unlike Tests 3 and 6 which deliberately push the VRAM
                # ceiling, all Test 4 conditions run at bs=1024 per rank;
                # an OOM here is a real DDP misconfiguration signal we
                # want to know about loudly rather than silently skip.
                raise RuntimeError(
                    f"{condition_name} seed={seed} failed with exit code {ret}\n"
                    f"--- last 20 lines of {log_path} ---\n{tail}"
                )
        active = next_active
        if active:
            time.sleep(2.0)


def launch_matrix(
    *,
    data_path: str,
    sp_model_path: str,
    budget: float,
    num_gpus: int,
    conditions: dict[str, tuple[int, dict[str, Any]]],
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    validate_data_paths(data_path, sp_model_path)
    # Process in ascending world_size order: ws=1 (parallel) -> ws=2
    # (partially parallel) -> ws=4 (serial). Lets the parallel cases
    # finish while we're still paying for the pod, and only drags into
    # serial time at the end.
    for condition_name, (ws, cfg) in sorted(conditions.items(), key=lambda item: item[1][0]):
        if ws > num_gpus:
            print(
                f"[{condition_name}] SKIPPED: ws={ws} > num_gpus={num_gpus}",
                flush=True,
            )
            continue
        _run_condition_group(
            condition_name=condition_name,
            world_size=ws,
            cfg=cfg,
            data_path=data_path,
            sp_model_path=sp_model_path,
            budget=budget,
            num_gpus_total=num_gpus,
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_results(
    conditions: dict[str, tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for condition_name, (ws, cfg) in conditions.items():
        pattern = re.compile(rf"^{re.escape(condition_name)}_s(\d+)\.json$")
        matches: list[tuple[int, Path]] = []
        if RESULTS.exists():
            for file in RESULTS.iterdir():
                m = pattern.match(file.name)
                if m:
                    matches.append((int(m.group(1)), file))
        if not matches:
            continue
        matches.sort(key=lambda pair: pair[0])
        bpb_by_seed: dict[int, float] = {}
        tok_per_s_values: list[float] = []
        for seed, file in matches:
            data = json.loads(file.read_text())
            bpb_by_seed[seed] = float(data["eval"]["bpb"])
            train = data["train"]
            # Aggregate tok/s: (steps * per-rank batch * seq * world_size) / elapsed_s
            # Divide by world_size again to report per-GPU tok/s for the scaling gate.
            tok_per_step_global = int(cfg["batch_size"]) * int(cfg["seq_len"]) * ws
            total_tok_per_s = (
                float(train["steps"]) * tok_per_step_global
                / max(float(train["elapsed_s"]), 1e-9)
            )
            tok_per_s_values.append(total_tok_per_s / ws)  # per-GPU
        bpb_values = [bpb_by_seed[s] for s in sorted(bpb_by_seed)]
        rows.append({
            "name": condition_name,
            "world_size": ws,
            "bpb_by_seed": bpb_by_seed,
            "bpb_values": bpb_values,
            "mean_bpb": _mean(bpb_values),
            "se_bpb": sem(bpb_values),
            "ci_bpb": bootstrap_ci(bpb_values),
            "mean_tok_per_gpu_per_s": _mean(tok_per_s_values),
        })

    rows.sort(key=lambda row: row["world_size"])
    if not rows:
        return summary

    print("\nTest 4 results — DDP scaling efficiency")
    print(
        f"  {'condition':<12} {'ws':>3} {'mean_bpb':>9} {'sem':>7} "
        f"{'tok/gpu/s':>11} {'scaling_eff':>13}"
    )
    ws1_tok = next((row["mean_tok_per_gpu_per_s"] for row in rows if row["world_size"] == 1), None)
    for row in rows:
        eff = (row["mean_tok_per_gpu_per_s"] / ws1_tok) if ws1_tok else float("nan")
        print(
            f"  {row['name']:<12} {row['world_size']:>3} {row['mean_bpb']:9.4f} "
            f"{row['se_bpb']:7.4f} {row['mean_tok_per_gpu_per_s']:11.0f} "
            f"{eff:13.3f}"
        )
        summary[row["name"]] = {
            "world_size": row["world_size"],
            "mean_bpb": row["mean_bpb"],
            "sem_bpb": row["se_bpb"],
            "mean_tok_per_gpu_per_s": row["mean_tok_per_gpu_per_s"],
            "scaling_efficiency_vs_ws1": eff,
            "n_seeds": len(row["bpb_values"]),
        }

    # Gate: each DDP condition must achieve (a) per-GPU tok/s >= 0.85 * ws1
    # AND (b) mean bpb lower than ws1 at matched 600s (via paired t-test).
    ws1_row = next((row for row in rows if row["world_size"] == 1), None)
    passing: list[str] = []
    pair_results: list[dict[str, Any]] = []
    if ws1_row is not None:
        for row in rows:
            if row["world_size"] == 1:
                continue
            shared = sorted(set(ws1_row["bpb_by_seed"]) & set(row["bpb_by_seed"]))
            if len(shared) < 2:
                continue
            a = [ws1_row["bpb_by_seed"][s] for s in shared]
            b = [row["bpb_by_seed"][s] for s in shared]
            t, p = paired_ttest(a, b)
            delta = sum(a) / len(a) - sum(b) / len(b)
            eff = row["mean_tok_per_gpu_per_s"] / ws1_row["mean_tok_per_gpu_per_s"]
            gate_tok = eff >= 0.85
            gate_bpb = delta > 0 and p < 0.05
            passes = bool(gate_tok and gate_bpb)
            pair_results.append({
                "condition": row["name"],
                "world_size": row["world_size"],
                "delta_bpb_vs_ws1": delta,
                "paired_t": t,
                "paired_p": p,
                "per_gpu_scaling_efficiency": eff,
                "passes_tok_gate_0.85": gate_tok,
                "passes_bpb_gate_p_lt_0.05": gate_bpb,
                "passes_both": passes,
            })
            if passes:
                passing.append(row["name"])
            print(
                f"\nvs ws1 n={len(shared)}: {row['name']}  "
                f"delta_bpb={delta:+.4f}  p_paired={p:.4g}  eff={eff:.3f}  "
                f"tok_gate={gate_tok}  bpb_gate={gate_bpb}"
            )
    summary["_decision"] = {
        "passing_conditions": passing,
        "pair_results": pair_results,
        "best_passing_world_size": (
            max((row["world_size"] for row in rows if row["name"] in passing), default=1)
        ),
    }
    print(f"\nGate: passing conditions = {passing}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 4 launcher — DDP scaling efficiency"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--num-gpus", type=int, default=4)
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
    (RESULTS / "test4_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
