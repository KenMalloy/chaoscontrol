#!/usr/bin/env python3
"""Experiment 18 Test 4b — ws scaling screen at submission regime.

Exp 18's throughput thesis: maximize tokens seen in 10 min on 8 GPUs.
Test 4 (original) measured ws=1 vs ws=2 at the old batch regime
(bs=512/rank, AdamW, criticality on, train_chaoscontrol path) and
never got ws=4 data because the DDP bucket-hook × chunked_cross_entropy
deadlock killed the run. That bug is fixed (manual all-reduce in
``training.py`` / ``train_ssm.py``), so ws=4 is runnable now — and the
regime has shifted entirely (bs=1024/rank, Muon, train_ssm path) so
Test 4's ws=1/2 data isn't comparable anyway.

Test 4b re-screens ws at the submission regime. On a 4-GPU pod we can
measure ws=2 and ws=4 directly; ws=8 is extrapolated from the
ws=2→ws=4 scaling efficiency. ws=8 stability itself gets checked for
the first time during Exp 19's 8-GPU submission run (the first time
we're actually on 8-GPU hardware).

**Conditions:**

    ws2: ws=2, bs=1024/rank → global_batch=2048, LR=0.064
    ws4: ws=4, bs=1024/rank → global_batch=4096, LR=0.128

**LR scaling rationale.** Test 5→5b established that the optimum LR
scales LINEARLY with global_batch, while the literature linear-from-
bs=32 rule is ~2× too hot at our scale. Test 5b winner was LR=0.064
at global_batch=2048. At global_batch=4096 we project LR=0.128 by
the same scaling. If Test 4b finds ws=4 unstable at 0.128, the
follow-up is a targeted LR screen at ws=4, not a failure of the
scaling rule.

**Locked axes (inherited from Tests 5b/7/8):**
    - Muon optimizer (Test 7 winner)
    - seq_len = 512 (Test 8 winner at matched wall-clock)
    - chunk_size = 64 (Test 5b default)
    - activation_checkpoint = True (Test 8 regime; keeps peak VRAM
      predictable across ws scaling)
    - SP16384 tokenizer
    - 4L × 256d SSM

**Expected memory budget:**
Peak VRAM per rank is independent of ws at fixed bs/rank + seq + model.
Test 8 measured 16.1 GiB/rank at bs=1024/seq=512 with ckpt on —
plenty of headroom on 80 GiB H100 regardless of ws.

**Pod-layout notes:**

    ws=2: 2 slots × 2 GPUs on a 4-GPU pod → 2 runs in parallel per wave,
          4 seeds = 2 waves of 600s ≈ 25 min wall-clock.
    ws=4: 1 slot × 4 GPUs → 1 run at a time, 4 seeds = 4 waves of 600s
          ≈ 45 min wall-clock.
    Total: ~70 min pod time.

**Throughput reporting.** The summary computes aggregate tokens/sec
per condition, the ws=4/ws=2 scaling efficiency, and an extrapolation
to ws=8. The extrapolation assumes efficiency carries (4→8 tends to
be at worst slightly lower than 2→4 on NVLink-connected H100s); if
the 2→4 number is unexpectedly poor we flag that as evidence the
submission regime should cap at ws=4.

**Gate.** For the "most training data in 10 min on 8 GPUs" framing,
any aggregate scaling > 1.0x keeps ws=8 strictly preferable — more
tokens always win unless ws=8 simply refuses to train. The launcher
emits a recommendation reflecting this:

    - If either condition has an unstable seed → "UNSTABLE at scaling
      endpoint; submission should use the stable ws".
    - If aggregate_scaling_2_to_4 < 1.2x → "scaling collapsing; ws=8
      throughput gains likely marginal; default to ws=4 for
      submission unless Exp 19 evidence says otherwise".
    - Else → "submit at ws=8 pending first-step NCCL + loss-descent
      check during Exp 19".

The bpb column at ws=4 carries a caveat: LR=0.128 at ws=4 is the
half-linear prediction from Test 5→5b; if ws=4 is stable but bpb is
worse than ws=2, the result confounds "ws=4 is worse at learning"
with "LR=0.128 is mis-calibrated at ws=4". Follow-up is a targeted
LR tiebreak at ws=4, not a scaling failure.

Projection band: the summary reports three ws=8 throughput
projections (optimistic / realistic / pessimistic) at 100% / 85% /
75% of the measured 2→4 per-GPU efficiency, so one number isn't
load-bearing for a recommendation.

Launch:

    python experiments/18_throughput_levers/run_exp18_test4b.py \\
        --data-path baselines/parameter_golf/datasets/fineweb10B_sp16384 \\
        --sp-model-path baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \\
        --budget 600
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
RESULTS = EXPERIMENT / "results_test4b"
RUNNER_SSM = EXPERIMENT / "runner_exp18_ssm.py"
TEST5B_SUMMARY = EXPERIMENT / "results_test5b" / "test5b_summary.json"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
sys.path.insert(0, str(EXPERIMENT))
from stats import bootstrap_ci, sem  # noqa: E402
from _harness import (  # noqa: E402
    result_is_finite,
    run_parallel_ddp_matrix,
    validate_data_paths,
)

SWEEP_SEEDS = [1337, 2674, 4011, 5348]

# OOM-skipped conditions, populated from run_parallel_ddp_matrix return.
_OOM_SKIPPED: list[str] = []


def _read_test5b_winning_lr() -> tuple[float | None, str]:
    """Read Test 5b's validated winner. Same pattern as Test 8."""
    if not TEST5B_SUMMARY.exists():
        return None, "missing"
    try:
        data = json.loads(TEST5B_SUMMARY.read_text())
    except Exception:
        return None, "missing"
    decision = data.get("_decision", {}) or {}
    lr = decision.get("winner_base_lr")
    winner = decision.get("winner_lr_condition")
    if winner is None or lr is None:
        return None, "no_winner"
    return float(lr), "ok"


def _resolve_base_lr_at_ws2(cli_override: float | None) -> float:
    """Pick the base LR for ws=2 (global_batch=2048).

    Priority:
      1. ``--base-lr`` CLI override.
      2. Test 5b winner from results_test5b/test5b_summary.json.
      3. Fail fast if neither is available.

    The ws=4 condition uses 2× this LR per the linear-scaling-with-
    global-batch rule Test 5→5b established.
    """
    if cli_override is not None:
        print(f"Test 4b: using --base-lr override {cli_override} (ws=2 anchor)", flush=True)
        return float(cli_override)
    lr, status = _read_test5b_winning_lr()
    if status == "ok":
        print(
            f"Test 4b: inherited ws=2 anchor base_lr={lr} from Test 5b",
            flush=True,
        )
        return lr
    raise RuntimeError(
        f"Test 4b could not resolve base_lr at ws=2 (status={status!r}). "
        f"Either run Test 5b first so its summary exists at {TEST5B_SUMMARY}, "
        f"or pass --base-lr <value> to set the ws=2 anchor explicitly."
    )


def _base(world_size: int, base_lr: float, **overrides: Any) -> dict[str, Any]:
    cfg = {
        "model_type": "ssm",
        "vocab_size": 16384,
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": 512,
        "stride": 256,
        "batch_size": 1024,
        "eval_batches": 16,
        "a_mode": "diag",
        "base_lr": base_lr,
        "activation_checkpoint": True,
        "optimizer": "muon",
        "chunk_size": 64,
        "world_size": world_size,
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
    }
    cfg.update(overrides)
    return cfg


def build_conditions(base_lr_ws2: float) -> dict[str, dict[str, Any]]:
    """Return {name: config}. LR at ws=4 is 2× the ws=2 anchor per the
    linear-scaling-with-global-batch rule Test 5→5b established.
    """
    return {
        "ws2": _base(world_size=2, base_lr=base_lr_ws2),
        "ws4": _base(world_size=4, base_lr=2 * base_lr_ws2),
    }


# Populated by main() so importers going through the canonical entry point
# see the resolved conditions.
CONDITIONS: dict[str, dict[str, Any]] = {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _loss_is_stable(final_loss: float, initial_loss: float = 9.7) -> bool:
    import math
    if not math.isfinite(final_loss):
        return False
    return final_loss < initial_loss - 1.0


def summarize_results(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    invalid_results: list[str] = []

    for condition_name, cfg in conditions.items():
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
        loss_by_seed: dict[int, float] = {}
        steps_by_seed: dict[int, int] = {}
        sps_by_seed: dict[int, float] = {}  # steps_per_second
        for seed, file in matches:
            data = json.loads(file.read_text())
            if not result_is_finite(data):
                invalid_results.append(f"{condition_name}_s{seed}")
                continue
            bpb_by_seed[seed] = float(data["eval"]["bpb"])
            loss_by_seed[seed] = float(data["train"]["final_loss"])
            steps_by_seed[seed] = int(data["train"]["steps"])
            sps_by_seed[seed] = float(data["train"]["steps_per_second"])

        bpb_values = [bpb_by_seed[s] for s in sorted(bpb_by_seed)]
        stable_count = sum(1 for v in loss_by_seed.values() if _loss_is_stable(v))

        # Tokens per optimizer step = world_size × bs_per_rank × seq_len.
        ws = int(cfg["world_size"])
        tokens_per_step = ws * int(cfg["batch_size"]) * int(cfg["seq_len"])
        mean_sps = _mean(list(sps_by_seed.values()))
        aggregate_tokens_per_sec = mean_sps * tokens_per_step

        rows.append({
            "name": condition_name,
            "world_size": ws,
            "base_lr": float(cfg["base_lr"]),
            "bpb_by_seed": bpb_by_seed,
            "bpb_values": bpb_values,
            "mean_bpb": _mean(bpb_values),
            "se_bpb": sem(bpb_values),
            "ci_bpb": bootstrap_ci(bpb_values) if bpb_values else (0.0, 0.0),
            "stable_seed_count": stable_count,
            "total_seed_count": len(bpb_values),
            "mean_steps_per_second": mean_sps,
            "tokens_per_step": tokens_per_step,
            "aggregate_tokens_per_sec": aggregate_tokens_per_sec,
        })

    # Sort rows by world_size so ws=2 comes before ws=4 for readability.
    rows.sort(key=lambda row: row["world_size"])
    if not rows:
        return summary

    print("\nTest 4b results — ws scaling at bs_per_rank=1024 (ckpt on uniformly)")
    hdr = (
        f"  {'condition':<8} {'ws':>3} {'lr':>7} {'stable':>8} "
        f"{'mean_bpb':>9} {'sem':>7} {'steps/s':>9} {'tok/s':>11}"
    )
    print(hdr)
    for row in rows:
        print(
            f"  {row['name']:<8} {row['world_size']:3d} {row['base_lr']:7.4f} "
            f"{row['stable_seed_count']}/{row['total_seed_count']:<6} "
            f"{row['mean_bpb']:9.4f} {row['se_bpb']:7.4f} "
            f"{row['mean_steps_per_second']:9.3f} {row['aggregate_tokens_per_sec']:11,.0f}"
        )
        summary[row["name"]] = {
            "world_size": row["world_size"],
            "base_lr": row["base_lr"],
            "mean_bpb": row["mean_bpb"],
            "sem_bpb": row["se_bpb"],
            "ci_95_bpb": row["ci_bpb"],
            "stable_seed_count": row["stable_seed_count"],
            "total_seed_count": row["total_seed_count"],
            "mean_steps_per_second": row["mean_steps_per_second"],
            "tokens_per_step": row["tokens_per_step"],
            "aggregate_tokens_per_sec": row["aggregate_tokens_per_sec"],
        }

    # Scaling efficiency ws=4 vs ws=2, and extrapolated ws=8 throughput.
    decision: dict[str, Any] = {
        "oom_skipped_conditions": list(_OOM_SKIPPED),
        "invalid_result_files": invalid_results,
    }
    ws2 = next((r for r in rows if r["name"] == "ws2"), None)
    ws4 = next((r for r in rows if r["name"] == "ws4"), None)
    if ws2 and ws4 and ws2["stable_seed_count"] == ws2["total_seed_count"] \
            and ws4["stable_seed_count"] == ws4["total_seed_count"]:
        # Per-GPU throughput (aggregate / ws) gives a fair scaling-efficiency
        # metric: 1.0 means perfect scaling. A 2× GPU count should deliver
        # 2× aggregate throughput → 1.0 per-GPU efficiency.
        tps_per_gpu_ws2 = ws2["aggregate_tokens_per_sec"] / 2.0
        tps_per_gpu_ws4 = ws4["aggregate_tokens_per_sec"] / 4.0
        efficiency_2_to_4 = tps_per_gpu_ws4 / tps_per_gpu_ws2 if tps_per_gpu_ws2 > 0 else 0.0

        # Extrapolate ws=8 with a three-point band. Optimistic assumes 4→8
        # efficiency matches 2→4 (geometric decay — same relative drop).
        # Realistic applies an 85% discount to that per-GPU number (modest
        # extra NVLink degradation across the 4→8 step). Pessimistic 75%.
        # The band is better than a single-point projection because we
        # only have ONE measurement of per-doubling efficiency, not two.
        band_factors = {"optimistic": 1.00, "realistic": 0.85, "pessimistic": 0.75}
        projections: dict[str, dict[str, float]] = {}
        for label, factor in band_factors.items():
            tps_per_gpu_ws8 = tps_per_gpu_ws4 * efficiency_2_to_4 * factor
            aggregate_ws8 = tps_per_gpu_ws8 * 8.0
            projections[label] = {
                "per_gpu_efficiency_factor": factor,
                "aggregate_tokens_per_sec_ws8": aggregate_ws8,
                "tokens_seen_10min_ws8": aggregate_ws8 * 600.0,
            }

        # Aggregate-throughput scaling ratio: 2.0 = perfect, <1.0 = degraded.
        aggregate_scaling_2_to_4 = (
            ws4["aggregate_tokens_per_sec"] / ws2["aggregate_tokens_per_sec"]
            if ws2["aggregate_tokens_per_sec"] > 0 else 0.0
        )

        # Recommendation reflects the task framing ("most training data in
        # 10 min on 8 GPUs"): more tokens always win unless ws=8 refuses
        # to train. The gate flags scaling collapse (<1.2×) as a "prefer
        # ws=4 unless Exp 19 evidence differs" — not a hard switch. Under
        # 1.0× is the only truly degenerate case and we don't expect that
        # on NVLink H100s.
        all_stable = (
            ws2["stable_seed_count"] == ws2["total_seed_count"]
            and ws4["stable_seed_count"] == ws4["total_seed_count"]
        )
        if not all_stable:
            recommendation = (
                "UNSTABLE at scaling endpoint; submission should use the "
                "stable ws"
            )
        elif aggregate_scaling_2_to_4 < 1.2:
            recommendation = (
                "scaling collapsing (aggregate 2→4 < 1.2×); ws=8 gains "
                "likely marginal; default to ws=4 for submission unless "
                "Exp 19 evidence says otherwise"
            )
        else:
            recommendation = (
                "submit at ws=8 pending first-step NCCL + loss-descent "
                "check during Exp 19"
            )

        decision.update({
            "aggregate_scaling_2_to_4": aggregate_scaling_2_to_4,
            "per_gpu_efficiency_2_to_4": efficiency_2_to_4,
            "projections_ws8": projections,
            "recommendation": recommendation,
            "bpb_at_ws4_confound_caveat": (
                "LR=0.128 at ws=4 is the half-linear prediction from Test "
                "5→5b; if ws=4 is stable but bpb is worse than ws=2, "
                "rerun a targeted LR tiebreak at ws=4 before concluding "
                "ws=4 is worse at learning."
            ),
        })

        realistic = projections["realistic"]
        print(
            f"\nScaling efficiency (ws=4 vs ws=2):"
            f"\n  aggregate tok/s ratio: {aggregate_scaling_2_to_4:.3f} "
            f"(2.0 = perfect)"
            f"\n  per-GPU efficiency:    {efficiency_2_to_4:.3f} "
            f"(1.0 = perfect)"
            f"\n  projected ws=8 tok/s (realistic 85%): "
            f"{realistic['aggregate_tokens_per_sec_ws8']:,.0f}"
            f"\n  projected 10min ws=8 tokens (realistic 85%): "
            f"{realistic['tokens_seen_10min_ws8']:,.0f}"
            f"\n  band: optimistic={projections['optimistic']['tokens_seen_10min_ws8']:,.0f} "
            f"realistic={realistic['tokens_seen_10min_ws8']:,.0f} "
            f"pessimistic={projections['pessimistic']['tokens_seen_10min_ws8']:,.0f}"
            f"\n  recommendation: {recommendation}"
        )

    summary["_decision"] = decision
    if invalid_results:
        print(
            f"\nDropped {len(invalid_results)} non-finite result file(s): "
            f"{invalid_results}"
        )
    if _OOM_SKIPPED:
        print(
            f"\nOOM-skipped conditions (excluded from winner selection): "
            f"{_OOM_SKIPPED}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 4b launcher — ws scaling at bs_per_rank=1024"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument(
        "--base-lr", type=float, default=None,
        help=(
            "Override the ws=2 anchor base LR. Defaults to Test 5b's winner. "
            "ws=4's LR is set to 2× this value per the linear-with-global-batch "
            "scaling rule established by Test 5→5b."
        ),
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    base_lr_ws2 = _resolve_base_lr_at_ws2(args.base_lr)
    global CONDITIONS
    CONDITIONS = build_conditions(base_lr_ws2)

    if not args.summarize_only:
        validate_data_paths(args.data_path, args.sp_model_path)

        # Two separate matrix calls because run_parallel_ddp_matrix takes a
        # single ws_per_slot. ws=2 runs 2 slots × 2 GPUs in parallel on a
        # 4-GPU pod; ws=4 runs 1 slot × 4 GPUs serially.
        ws2_skipped = run_parallel_ddp_matrix(
            conditions={"ws2": CONDITIONS["ws2"]},
            seeds=SWEEP_SEEDS,
            ws_per_slot=2,
            num_slots=2,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS,
            runner_script=RUNNER_SSM,
            skip_oom_conditions=True,
        )
        _OOM_SKIPPED.extend(ws2_skipped)

        ws4_skipped = run_parallel_ddp_matrix(
            conditions={"ws4": CONDITIONS["ws4"]},
            seeds=SWEEP_SEEDS,
            ws_per_slot=4,
            num_slots=1,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS,
            runner_script=RUNNER_SSM,
            skip_oom_conditions=True,
        )
        _OOM_SKIPPED.extend(ws4_skipped)

    summary = summarize_results(CONDITIONS)
    summary["_base_lr_ws2_anchor"] = base_lr_ws2
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "test4b_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
