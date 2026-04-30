#!/usr/bin/env python3
"""Experiment 18 Test 3 launcher: activation checkpointing pushes batch ceiling.

Test 3 asks whether wrapping the block stack in ``torch.utils.checkpoint``
lets us trade recompute cost for VRAM headroom, pushing the single-GPU batch
ceiling past the chunked scan's bs=512 wall (bound by fp64 cumprod
intermediates) up to bs=1024 or bs=2048. The ``activation_checkpoint`` flag
on ``CareStudentLM`` landed in the merged subagent work; this orchestrator
exercises it under real bpb measurement.

Three conditions:
    - no-ckpt, bs=512   (baseline at the chunked-backend ceiling)
    - ckpt, bs=1024     (first batch step past the ceiling)
    - ckpt, bs=2048     (further push, only feasible with checkpointing)

Scientific gate (bpb, not just tok/s):
    The ckpt+larger-batch conditions must achieve lower bpb at matched
    600s wall-clock than the no-ckpt/bs=512 baseline. A lever that pushes
    batch size higher but spends the extra VRAM for no per-wall-clock
    learning gain is not worth the recompute overhead.

Launch pattern: parallel-single-GPU, up to ``--num-gpus`` concurrent runs.
Each run takes one GPU. With a 4-GPU pod and 3 conditions x 4 seeds =
12 runs, 4 parallel slots produce 3 waves of 4 = ~30 min end-to-end.

Budget: ~30 min on a 4-GPU pod, ~$5.
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
RESULTS = EXPERIMENT / "results_test3"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
sys.path.insert(0, str(EXPERIMENT))
from stats import bootstrap_ci, paired_ttest, sem  # noqa: E402
from _harness import _is_oom_failure, build_launch_cmd, result_is_finite, validate_data_paths  # noqa: E402

# Four seeds per condition fill a 4-slot parallel-single-GPU matrix cleanly
# (3 conditions x 4 seeds = 12 runs -> 3 waves of 4). Exp 17 per-condition
# std is ~0.004 bpb, so n=4 detects ~0.01 bpb effects at paired t-test
# 80% power — enough to resolve "did the larger batch actually help".
SWEEP_SEEDS = [1337, 2674, 4011, 5348]
TIMEOUT_MULTIPLIER = 2.5

# Populated by main() from launch_matrix's return value so the summary
# can annotate which VRAM-ceiling conditions were OOM-skipped.
_OOM_SKIPPED: list[str] = []


def _base(**overrides: Any) -> dict[str, Any]:
    """Matched config envelope. Only ``batch_size`` and
    ``activation_checkpoint`` vary between conditions; everything else is
    the Test 2 SP16384 winner config (dim=256, L=4, seq=512, LR=2e-3).
    """
    cfg = {
        "model_type": "ssm",
        "vocab_size": 16384,
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 512,
        "eval_batches": 16,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
        "base_lr": 2e-3,
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
        "activation_checkpoint": False,
    }
    cfg.update(overrides)
    return cfg


CONDITIONS: dict[str, dict[str, Any]] = {
    "noctk_bs512":  _base(batch_size=512,  activation_checkpoint=False),
    "ckpt_bs1024":  _base(batch_size=1024, activation_checkpoint=True),
    "ckpt_bs2048":  _base(batch_size=2048, activation_checkpoint=True),
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
    data_path: str,
    sp_model_path: str,
    budget: float,
    num_gpus: int,
    conditions: dict[str, dict[str, Any]],
) -> list[str]:
    """Parallel-single-GPU launch. Each run pinned to one GPU via
    ``CUDA_VISIBLE_DEVICES``. Up to ``num_gpus`` concurrent children.

    Slot accounting uses an explicit free-GPU set, not a monotonic
    cursor — runs can finish out-of-order (different conditions have
    different VRAM footprints and therefore different eval times) and
    a ``gpu_cursor % num_gpus`` scheme would silently alias a new run
    to a GPU still held by an earlier one.

    Returns the list of condition names that were OOM-skipped so the
    caller can surface them in the summary.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    validate_data_paths(data_path, sp_model_path)

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

    run_timeout = max(budget * TIMEOUT_MULTIPLIER, 60.0)
    free_gpus: list[int] = list(range(max(num_gpus, 1)))
    active: list = []  # (proc, cfg_path, condition_name, seed, t0, log_fh, gpu_id)
    skipped_conditions: list[str] = []

    def _drop_condition_from_queue(name: str) -> int:
        dropped = 0
        remaining: list[tuple[str, int, Path]] = []
        for entry in queue:
            if entry[0] == name:
                entry[2].unlink(missing_ok=True)
                dropped += 1
            else:
                remaining.append(entry)
        queue[:] = remaining
        return dropped

    while queue or active:
        while queue and free_gpus:
            condition_name, seed, cfg_path = queue.pop(0)
            if condition_name in skipped_conditions:
                cfg_path.unlink(missing_ok=True)
                continue
            out_path = RESULTS / f"{condition_name}_s{seed}.json"
            gpu_id = free_gpus.pop(0)  # smallest free
            env = os.environ.copy()
            if num_gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            log_path = RESULTS / f"{condition_name}_s{seed}.log"
            log_fh = open(log_path, "w")
            cmd = build_launch_cmd(
                num_gpus=1,
                cfg_path=cfg_path,
                data_path=data_path,
                sp_model_path=sp_model_path,
                budget=budget,
                out_path=out_path,
            )
            print(
                f"Launching {condition_name} seed={seed} gpu={gpu_id}",
                flush=True,
            )
            proc = subprocess.Popen(
                cmd, env=env, text=True, stdout=log_fh, stderr=subprocess.STDOUT,
            )
            active.append(
                (proc, cfg_path, condition_name, seed, time.monotonic(), log_fh, gpu_id)
            )

        next_active: list = []
        for i, entry in enumerate(active):
            proc, cfg_path, condition_name, seed, t0, log_fh, gpu_id = entry
            ret = proc.poll()
            elapsed = time.monotonic() - t0
            if ret is None and elapsed < run_timeout:
                next_active.append(entry)
                continue
            log_fh.close()
            free_gpus.append(gpu_id)
            free_gpus.sort()
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
                if _is_oom_failure(log_path):
                    # bs=2048 is deliberately a VRAM ceiling push; a
                    # legitimate OOM on that condition just means the
                    # lever didn't earn its place, not that the whole
                    # matrix should die. Drop its remaining seeds and
                    # continue.
                    dropped = _drop_condition_from_queue(condition_name)
                    if condition_name not in skipped_conditions:
                        skipped_conditions.append(condition_name)
                    print(
                        f"OOM_SKIP: {condition_name} seed={seed} OOM'd; "
                        f"dropped {dropped} remaining seeds of this condition",
                        flush=True,
                    )
                    continue
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

    return skipped_conditions


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_results(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    invalid_results: list[str] = []  # {condition}_s{seed} of files dropped as non-finite

    for condition_name in conditions:
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
            if not result_is_finite(data):
                invalid_results.append(f"{condition_name}_s{seed}")
                continue
            bpb_by_seed[seed] = float(data["eval"]["bpb"])
            train = data["train"]
            # tok/s = (steps * batch * seq) / elapsed_s
            cfg = conditions[condition_name]
            tok_per_step = int(cfg["batch_size"]) * int(cfg["seq_len"])
            tok_per_s_values.append(
                float(train["steps"]) * tok_per_step / max(float(train["elapsed_s"]), 1e-9)
            )
        bpb_values = [bpb_by_seed[s] for s in sorted(bpb_by_seed)]
        rows.append({
            "name": condition_name,
            "bpb_by_seed": bpb_by_seed,
            "bpb_values": bpb_values,
            "mean_bpb": _mean(bpb_values),
            "se_bpb": sem(bpb_values),
            "ci_bpb": bootstrap_ci(bpb_values),
            "mean_tok_per_s": _mean(tok_per_s_values),
        })

    rows.sort(key=lambda row: row["mean_bpb"])
    if not rows:
        return summary

    print("\nTest 3 results — activation checkpointing batch ceiling push")
    print(
        f"  {'condition':<16} {'mean_bpb':>9} {'sem':>7} {'95% CI':>21} "
        f"{'tok/s':>12}"
    )
    for row in rows:
        print(
            f"  {row['name']:<16} {row['mean_bpb']:9.4f} {row['se_bpb']:7.4f} "
            f"[{row['ci_bpb'][0]:.4f}, {row['ci_bpb'][1]:.4f}] "
            f"{row['mean_tok_per_s']:12.0f}"
        )
        summary[row["name"]] = {
            "mean_bpb": row["mean_bpb"],
            "sem_bpb": row["se_bpb"],
            "ci_95_bpb": row["ci_bpb"],
            "mean_tok_per_s": row["mean_tok_per_s"],
            "n_seeds": len(row["bpb_values"]),
        }

    # Gate: each ckpt condition must beat the noctk_bs512 baseline in bpb
    # at matched 600s wall-clock, via paired t-test across shared seeds.
    by_name = {row["name"]: row for row in rows}
    baseline = by_name.get("noctk_bs512")
    if baseline is None:
        return summary

    winners: list[str] = []
    pair_results: list[dict[str, Any]] = []
    for candidate_name in ("ckpt_bs1024", "ckpt_bs2048"):
        cand = by_name.get(candidate_name)
        if cand is None:
            continue
        shared = sorted(set(baseline["bpb_by_seed"]) & set(cand["bpb_by_seed"]))
        if len(shared) < 2:
            continue
        base_paired = [baseline["bpb_by_seed"][s] for s in shared]
        cand_paired = [cand["bpb_by_seed"][s] for s in shared]
        # Order (baseline, cand): positive t => baseline > cand => candidate wins.
        t, p = paired_ttest(base_paired, cand_paired)
        delta = sum(base_paired) / len(base_paired) - sum(cand_paired) / len(cand_paired)
        gate_pass = delta > 0 and p < 0.05
        pair_results.append({
            "candidate": candidate_name,
            "delta_vs_baseline_bpb": delta,
            "paired_t": t,
            "paired_p": p,
            "n_paired_seeds": len(shared),
            "beats_baseline": bool(gate_pass),
        })
        if gate_pass:
            winners.append(candidate_name)
        print(
            f"\nPaired (n={len(shared)}): "
            f"noctk_bs512 - {candidate_name} = {delta:+.4f} bpb  "
            f"t={t:.3f}  p_paired={p:.4g}"
        )

    # Winner rule: among passing candidates, the one with the lowest mean_bpb wins.
    if winners:
        winner = min(winners, key=lambda name: by_name[name]["mean_bpb"])
    else:
        winner = "noctk_bs512"
    summary["_decision"] = {
        "winner": winner,
        "pair_results": pair_results,
        "oom_skipped_conditions": list(_OOM_SKIPPED),
        "invalid_result_files": invalid_results,
    }
    if invalid_results:
        print(
            f"\nDropped {len(invalid_results)} non-finite result file(s): "
            f"{invalid_results}"
        )
    if _OOM_SKIPPED:
        print(
            f"\nOOM-skipped conditions (near-ceiling pushes that failed): "
            f"{_OOM_SKIPPED}"
        )
    print(
        f"\nGate: ckpt condition adopted iff bpb beats baseline at paired p<0.05 -> "
        f"winner={winner}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 3 launcher — activation checkpointing batch ceiling"
    )
    parser.add_argument("--data-path", required=True,
                        help="Directory with pre-tokenized fineweb shards")
    parser.add_argument("--sp-model-path", required=True,
                        help="Path to the SentencePiece model file")
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if not args.summarize_only:
        skipped = launch_matrix(
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            num_gpus=args.num_gpus,
            conditions=CONDITIONS,
        )
        _OOM_SKIPPED.extend(skipped)

    summary = summarize_results(CONDITIONS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "test3_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
