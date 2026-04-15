#!/usr/bin/env python3
"""Experiment 18 Test 7 launcher: AdamW vs Muon vs LAMB under DDP.

Test 7 is the optimizer ablation called out in the Exp 18 design doc
(docs/plans/2026-04-12-experiment-18-throughput-levers-design.md). Three
conditions run against the same bare fast SSM config: the AdamW baseline,
Muon (Newton-Schulz matrix orthogonalization with inline AdamW fallback for
non-matrix params), and LAMB (per-tensor trust ratio, first-principles
large-batch choice). Any alternative must beat AdamW on bpb_at_elapsed_s_600
with paired p<0.05 across matched seeds to earn inclusion in the Test 9
integration run; otherwise AdamW stays.

**DDP path, not single-process.** Test 7's hypothesis is about optimizer
behavior at the large-global-batch regime produced by 8xH100 DDP — Tests
4-6 establish that regime, Test 7 measures optimizer choice inside it. So
this launcher spawns each (condition, seed) via
``python -m torch.distributed.run --standalone --nproc_per_node=N``
against ``runner_exp18.py``, the DDP-capable entry point that reads the
optimizer kwarg from the config yaml and passes it through
``train_chaoscontrol_for_budget``. Unlike Test 2's orchestrator (which
runs 14 independent single-process jobs in parallel across all visible
GPUs), Test 7 is strictly sequential: each run seizes all N GPUs under
torchrun's rendezvous, and the next run can only start after the previous
one tears down.

Seed choice: the design doc budgets ~1h on 8xH100 (3 optimizers x 20 min
each). Three seeds per condition is low statistical power — a true
~0.004 bpb effect will often miss p<0.05 — but still resolves a ~0.008 bpb
or larger winner at n=3 (that is the Exp 15 tokenizer effect size and
would pass the paired t-test cleanly). The three-seed scientific contract
is that we'll either see a strong winner or conclude the alternatives are
indistinguishable from AdamW at this sample size; we do NOT get to declare
a marginal winner at p<0.10. Note the total wall-clock at 3 seeds is 3
seeds x 3 optimizers x 20 min = ~3h, not the design doc's 1h line — that
line assumed 1 seed, which doesn't permit any statistical test. The ~3h
figure is the real cost of actually running paired comparisons.

Launch convention (8xH100 grant pod):

    python experiments/18_throughput_levers/run_exp18_test7.py \
        --data-path /data/fineweb_sp8192 \
        --sp-model-path /models/sp8192.model \
        --budget 600 \
        --num-gpus 8

For local smoke / validation on fewer GPUs:

    python experiments/18_throughput_levers/run_exp18_test7.py \
        --data-path ... --sp-model-path ... --budget 600 --num-gpus 4

When ``--num-gpus 1`` the launcher skips torchrun entirely and invokes
``runner_exp18.py`` directly (runner_exp18 degrades cleanly to world_size=1
when WORLD_SIZE is unset), so this script works as a single-device smoke
without needing a distributed backend.

All three conditions share dim=256, layers=4, seq_len=512, batch=32, LR=2e-3,
the same Exp 17 bare fast SSM config used for Test 2's SP8192 baseline. The
only varying axis is the ``optimizer`` config key, which
``runner_exp18.py`` reads and threads through to the training loop.

Gate (from 2026-04-12-experiment-18-throughput-levers-design.md Test 7):
    Any alternative beats AdamW at paired p<0.05 on mean bpb across seeds
    -> that alternative is adopted for Exp 19. Otherwise AdamW stays. Per
    the design doc "the scientifically honest version of 'does optimizer
    matter here?'", we print the per-pair comparison for all three pairs
    (muon vs adamw, lamb vs adamw, muon vs lamb) so the reviewer sees the
    full ablation, not just the winning pair.

Budget: ~3h on 8xH100 grant pod, ~$60.
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
RESULTS = EXPERIMENT / "results_test7"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
from stats import bootstrap_ci, paired_ttest, sem  # noqa: E402


# Three-seed subset of the Exp 17 sweep (1337, 2674, 4011). Power argument in
# the module docstring.
SWEEP_SEEDS = [1337, 2674, 4011]
ARTIFACT_LIMIT_BYTES = 16 * 1024 * 1024
TIMEOUT_MULTIPLIER = 2.5

# Test 7 goes through the DDP runner, NOT the Exp 17 single-process runner.
# This is the whole point of the rewrite — optimizer behavior under DDP is
# the question, and single-process can't answer it.
RUNNER_SCRIPT = EXPERIMENT / "runner_exp18.py"


def _base(**overrides: Any) -> dict[str, Any]:
    """Matched bare fast SSM envelope for all three optimizer conditions.

    Identical to Test 2's ``bare_fast_ssm_sp8192`` config (dim=256, L=4,
    bs=32, seq=512, LR=2e-3, SP8192 vocab, diag a_mode) so the only varying
    axis is ``optimizer``. The chunked scan is the default diag backend
    after Test 1 — no extra flag needed.
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
        "vocab_size": 8192,
    }
    cfg.update(overrides)
    return cfg


CONDITIONS: dict[str, dict[str, Any]] = {
    "adamw_baseline": _base(optimizer="adamw"),
    "muon": _base(optimizer="muon"),
    "lamb": _base(optimizer="lamb"),
}


def build_launch_cmd(
    *,
    num_gpus: int,
    cfg_path: Path,
    data_path: str,
    sp_model_path: str,
    budget: float,
    out_path: Path,
) -> list[str]:
    """Construct the subprocess argv for one (condition, seed) run.

    Single-device (num_gpus=1): invoke ``runner_exp18.py`` directly under
    the same Python interpreter. ``runner_exp18.py`` detects the absence of
    ``WORLD_SIZE`` and takes the bit-identical single-device code path, so
    this works for local CPU smokes and 1-GPU pods alike.

    Multi-device (num_gpus>1): invoke ``python -m torch.distributed.run
    --standalone --nproc_per_node=N`` against ``runner_exp18.py``. Each
    rank binds to its own GPU via torchrun's LOCAL_RANK, runs the same
    training step, and synchronizes gradients via DDP inside
    ``train_chaoscontrol_for_budget``. Output JSON is written by rank 0
    only (runner_exp18 already guards this).
    """
    common_tail = [
        str(RUNNER_SCRIPT),
        "--config", str(cfg_path),
        "--data-path", data_path,
        "--sp-model-path", sp_model_path,
        "--budget", str(budget),
        "--output-json", str(out_path),
    ]
    if num_gpus <= 1:
        return [sys.executable, "-u"] + common_tail
    return [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
    ] + common_tail


def launch_matrix(
    *,
    data_path: str,
    sp_model_path: str,
    budget: float,
    num_gpus: int,
    conditions: dict[str, dict[str, Any]],
) -> None:
    """Launch every (condition, seed) sequentially under torchrun.

    Each invocation seizes all ``num_gpus`` GPUs under torchrun, so runs are
    strictly sequential — there's no per-GPU parallelism to exploit the way
    Test 2 does, because the point of Test 7 is measuring optimizer
    behavior at the DDP global batch, not at single-GPU small batch.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)

    if not Path(data_path).is_dir():
        raise FileNotFoundError(f"data dir {data_path} does not exist")
    if not Path(sp_model_path).is_file():
        raise FileNotFoundError(f"SP model file {sp_model_path} does not exist")

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

    for condition_name, seed, cfg_path in queue:
        out_path = RESULTS / f"{condition_name}_s{seed}.json"
        log_path = RESULTS / f"{condition_name}_s{seed}.log"
        cmd = build_launch_cmd(
            num_gpus=num_gpus,
            cfg_path=cfg_path,
            data_path=data_path,
            sp_model_path=sp_model_path,
            budget=budget,
            out_path=out_path,
        )
        print(
            f"Launching {condition_name} seed={seed} nproc={num_gpus}",
            flush=True,
        )
        t0 = time.monotonic()
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                cmd,
                env=os.environ.copy(),
                text=True,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            try:
                ret = proc.wait(timeout=run_timeout)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                cfg_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{condition_name} seed={seed} TIMEOUT after {elapsed:.0f}s "
                    f"(budget={budget}s, limit={run_timeout:.0f}s)"
                )

        cfg_path.unlink(missing_ok=True)
        if ret != 0:
            tail = ""
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                tail = "\n".join(lines[-20:])
            raise RuntimeError(
                f"{condition_name} seed={seed} failed with exit code {ret}\n"
                f"--- last 20 lines of {log_path} ---\n{tail}"
            )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _collect_rows(conditions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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
    return rows


def _pairwise_paired(
    rows: list[dict[str, Any]],
    a_name: str,
    b_name: str,
) -> dict[str, Any] | None:
    by_name = {row["name"]: row for row in rows}
    a = by_name.get(a_name)
    b = by_name.get(b_name)
    if a is None or b is None:
        return None
    shared_seeds = sorted(set(a["bpb_by_seed"]) & set(b["bpb_by_seed"]))
    if len(shared_seeds) < 2:
        return None
    a_paired = [a["bpb_by_seed"][s] for s in shared_seeds]
    b_paired = [b["bpb_by_seed"][s] for s in shared_seeds]
    # (a, b): positive t-stat means a > b in bpb, so b is the better optimizer.
    # Delta must use the same paired sample set as the p-value, not all-seed
    # means, otherwise under partial reruns the printed delta and the b_beats_a
    # gate would come from a different cohort than the t-test they're reported
    # alongside.
    t, p = paired_ttest(a_paired, b_paired)
    delta = sum(a_paired) / len(a_paired) - sum(b_paired) / len(b_paired)
    return {
        "a_name": a_name,
        "b_name": b_name,
        "n_paired_seeds": len(shared_seeds),
        "delta_a_minus_b_bpb": delta,
        "paired_t": t,
        "paired_p": p,
        "b_beats_a": bool(delta > 0 and p < 0.05),
    }


def summarize_results(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows = _collect_rows(conditions)
    if not rows:
        return summary

    print("\nTest 7 results — AdamW vs Muon vs LAMB on bare fast SSM SP8192")
    print(
        f"  {'condition':<20} {'mean_bpb':>9} {'sem':>7} {'95% CI':>21} "
        f"{'steps/s':>9} {'artifact_mb':>12}"
    )
    for row in rows:
        print(
            f"  {row['name']:<20} {row['mean_bpb']:9.4f} {row['se_bpb']:7.4f} "
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

    # Three paired comparisons laid out the way the design doc asks for.
    # Print raw numbers first, then apply the gate rule to pick a winner.
    pair_results = []
    for a, b in (
        ("adamw_baseline", "muon"),
        ("adamw_baseline", "lamb"),
        ("muon", "lamb"),
    ):
        pr = _pairwise_paired(rows, a, b)
        if pr is None:
            continue
        pair_results.append(pr)
        print(
            f"\nPaired (n={pr['n_paired_seeds']}): "
            f"{pr['a_name']} - {pr['b_name']} = {pr['delta_a_minus_b_bpb']:+.4f} bpb  "
            f"t={pr['paired_t']:.3f}  p_paired={pr['paired_p']:.4g}"
        )

    muon_vs_adamw = next(
        (pr for pr in pair_results if pr["a_name"] == "adamw_baseline" and pr["b_name"] == "muon"),
        None,
    )
    lamb_vs_adamw = next(
        (pr for pr in pair_results if pr["a_name"] == "adamw_baseline" and pr["b_name"] == "lamb"),
        None,
    )
    muon_passes = muon_vs_adamw is not None and muon_vs_adamw["b_beats_a"]
    lamb_passes = lamb_vs_adamw is not None and lamb_vs_adamw["b_beats_a"]

    # Winner rule: the passing alternative with the lowest mean_bpb wins; if
    # none pass, AdamW stays. When both pass, prefer the one with the larger
    # |delta| so we adopt the bigger-win optimizer rather than the noisier tie.
    by_name = {row["name"]: row for row in rows}
    candidates: list[tuple[str, float]] = []
    if muon_passes:
        candidates.append(("muon", by_name["muon"]["mean_bpb"]))
    if lamb_passes:
        candidates.append(("lamb", by_name["lamb"]["mean_bpb"]))
    if candidates:
        winner_name = min(candidates, key=lambda item: item[1])[0]
    else:
        winner_name = "adamw_baseline"

    summary["_decision"] = {
        "muon_beats_adamw_p_lt_0.05": muon_passes,
        "lamb_beats_adamw_p_lt_0.05": lamb_passes,
        "winner": winner_name,
        "pair_results": pair_results,
    }
    print(
        f"\nGate: alternative adopted iff it beats AdamW at paired p<0.05 -> "
        f"winner={winner_name}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 7 launcher — AdamW vs Muon vs LAMB"
    )
    parser.add_argument("--data-path", required=True,
                        help="Directory with SP8192-tokenized fineweb shards")
    parser.add_argument("--sp-model-path", required=True,
                        help="Path to the SP8192 SentencePiece model file")
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--num-gpus", type=int, default=8)
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
    (RESULTS / "test7_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
