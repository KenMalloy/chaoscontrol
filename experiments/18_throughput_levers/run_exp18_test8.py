#!/usr/bin/env python3
"""Experiment 18 Test 8 launcher: sequence-length re-screen at the
chunked-backward memory ceiling.

Test 6 (original) capped at seq=512 because seq≥1024 OOMed in
``loss.backward()`` on the frozen path — the logits.grad allocation
was the binding constraint. Test 6b extended the launcher with
``activation_checkpoint=True`` but was never run on a pod. Test 8
supersedes both: now that chunked LM-head backward has removed the
logits.grad bottleneck (``project_chunked_lm_backward_empirical_2026-04-16.md``),
encoder activations are the dominant memory term and linearly
scale with seq. At bs=1024/seq=512 peak is 36.78 GiB; linear
projection gives ~73 GiB at seq=1024 and ~147 GiB at seq=2048 on an
80 GiB H100. seq=1024 fits tight without checkpointing; seq=2048
requires activation checkpointing.

**Clean-axis design choice.** Mixing ``activation_checkpoint`` state
across conditions would introduce a two-axis confound (seq × ckpt) in
the same mistake the Test 5b review caught. Instead we run
activation_checkpoint=True **uniformly** across all three conditions.
Cost: ~30% per-step compute overhead at seq=512 (where ckpt isn't
memory-necessary). Benefit: the seq=512 vs seq=1024 vs seq=2048
comparison is a clean single-axis screen.

Three conditions:

    seq512     budget=600s, activation_checkpoint=True (baseline)
    seq1024    budget=600s, activation_checkpoint=True
    seq2048    budget=600s, activation_checkpoint=True (may OOM if the
                                     rough ~80 GiB projection is too
                                     optimistic; OOMs are handled by
                                     the harness's skip_oom_conditions)

**Locked axes (inherited from Test 5b):**
    - Muon optimizer @ LR=0.064 (Test 5b winner at bs=1024/rank/ws=2)
    - SP16384 tokenizer (Test 2 winner)
    - 4L × 256d SSM, bs=1024/rank, ws=2
    - chunk_size=64 (Test 5b default; holds logit memory at ~2.15 GiB
      per chunk regardless of seq)

**Gate.** Stage 1 only — pick stable winner by paired-seed mean bpb
with the same 0.006-bpb / p<0.05 noise tiebreak as Test 5/5b. No
cross-screen comparison is emitted. Test 8 differs from Test 5b on
the activation_checkpoint axis (Test 5b ran ckpt-off), so a paired
delta would be two-axis-confounded (seq × ckpt) — the same mistake
Test 5b's review already caught and dropped.

**Seq=512 is NOT a direct Test 5b control.** The Test 8 seq=512 row
is a ckpt-on control for this screen only. Compare Test 8 rows to
each other, not to Test 5b rows directly. Any "did bigger seq help"
claim needs to come out of this screen's own rows.

**This is a matched-wall-clock comparison, not matched-tokens-seen.**
All three conditions run on a 600s budget. At fixed bs=1024/rank a
2× longer seq is roughly 2× more compute per step, so seq=2048 sees
~1/4 the optimizer steps of seq=512 within the budget. That's the
intended framing for the Exp 19 question — "given a fixed wall-clock
budget, which seq wins" — but it means the screen does NOT answer
"which seq wins at matched tokens-seen" or "which seq is best-in-
class with per-condition budget tuning." Both are separately
legitimate questions; they would require separate screens.

``stride`` is locked at 256 across all conditions (Test 5b's value at
seq=512). Holding stride constant keeps the per-window token
exposure comparable across seq values, and seq=2048 still has
enough starting positions for 4 ranks × 600s of sampling. If
stride scaled with seq_len, seq=2048 would have 4× fewer distinct
starts than seq=512 and the comparison would be biased by data
diversity, not just the seq lever.

Launch: same 2-slot parallel DDP pattern as Test 5b. On a 4-GPU pod
this schedules 3 conditions × 4 seeds = 12 runs in 6 waves of 600s
each ≈ 1 hour wall-clock, plus activation-checkpoint overhead
(~30%) → realistically ~70-80 min. If seq=2048 OOMs the harness
drops its remaining seeds and continues with the other conditions.
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
RESULTS = EXPERIMENT / "results_test8"
RUNNER_SSM = EXPERIMENT / "runner_exp18_ssm.py"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
sys.path.insert(0, str(EXPERIMENT))
from stats import bootstrap_ci, paired_ttest, sem  # noqa: E402
from _harness import (  # noqa: E402
    result_is_finite,
    run_parallel_ddp_matrix,
    validate_data_paths,
)

SWEEP_SEEDS = [1337, 2674, 4011, 5348]

# Locked LR from Test 5b — see
# memory/project_exp18_test5b_results_2026-04-16.md. Re-screening LR
# inside a seq sweep would turn a single-axis test into a 2D screen
# (seq × LR). We inherit instead and accept that the optimum LR may
# shift slightly with seq; any surprise in Test 8 that suggests
# LR/seq interaction is a signal to run a targeted follow-up, not a
# reason to nest the screens here.
LOCKED_LR = 0.064

SEQ_VALUES = (512, 1024, 2048)


def _base(seq_len: int, **overrides: Any) -> dict[str, Any]:
    cfg = {
        "model_type": "ssm",
        "vocab_size": 16384,
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": seq_len,
        # stride=256 uniform (matches Test 5b at seq=512). Holding stride
        # constant across conditions keeps the per-window token exposure
        # comparable and gives seq=2048 enough distinct train starts to
        # avoid "fewer unique windows" being a hidden axis.
        "stride": 256,
        "batch_size": 1024,  # inherited from Test 5b — bs/rank at ws=2
        "eval_batches": 16,
        "a_mode": "diag",
        "base_lr": LOCKED_LR,
        # Uniform checkpointing across all conditions so the seq
        # comparison isn't confounded by ckpt state. Costs ~30% compute
        # per step at seq=512 where ckpt isn't memory-necessary, but
        # buys a clean single-axis signal.
        "activation_checkpoint": True,
        "optimizer": "muon",  # Test 7 winner
        "chunk_size": 64,  # Test 5b default; per-chunk logit mem ~2.15 GiB
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
    }
    cfg.update(overrides)
    return cfg


CONDITIONS: dict[str, dict[str, Any]] = {
    f"seq{seq}": _base(seq_len=seq) for seq in SEQ_VALUES
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _loss_is_stable(final_loss: float, initial_loss: float = 9.7) -> bool:
    """Stability check — same shape as Test 5/5b. Vocab=16384 gives a
    uniform-prediction CE of log(16384) ≈ 9.7 nats; require at least
    1.0 nat of progress to count as stable.
    """
    import math
    if not math.isfinite(final_loss):
        return False
    return final_loss < initial_loss - 1.0


def summarize_results(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    invalid_results: list[str] = []

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
        loss_by_seed: dict[int, float] = {}
        for seed, file in matches:
            data = json.loads(file.read_text())
            if not result_is_finite(data):
                invalid_results.append(f"{condition_name}_s{seed}")
                continue
            bpb_by_seed[seed] = float(data["eval"]["bpb"])
            loss_by_seed[seed] = float(data["train"]["final_loss"])
        bpb_values = [bpb_by_seed[s] for s in sorted(bpb_by_seed)]
        stable_count = sum(1 for v in loss_by_seed.values() if _loss_is_stable(v))
        rows.append({
            "name": condition_name,
            "seq_len": conditions[condition_name]["seq_len"],
            "bpb_by_seed": bpb_by_seed,
            "bpb_values": bpb_values,
            "mean_bpb": _mean(bpb_values),
            "se_bpb": sem(bpb_values),
            "ci_bpb": bootstrap_ci(bpb_values),
            "stable_seed_count": stable_count,
            "total_seed_count": len(bpb_values),
        })

    rows.sort(key=lambda row: row["mean_bpb"])
    if not rows:
        return summary

    print("\nTest 8 results — seq re-screen at bs_per_rank=1024 (ckpt on uniformly)")
    print(
        f"  {'condition':<10} {'seq':>6} {'stable':>8} {'mean_bpb':>9} {'sem':>7}"
    )
    for row in rows:
        print(
            f"  {row['name']:<10} {row['seq_len']:6d} "
            f"{row['stable_seed_count']}/{row['total_seed_count']:<6} "
            f"{row['mean_bpb']:9.4f} {row['se_bpb']:7.4f}"
        )
        summary[row["name"]] = {
            "seq_len": row["seq_len"],
            "base_lr": conditions[row["name"]]["base_lr"],
            "mean_bpb": row["mean_bpb"],
            "sem_bpb": row["se_bpb"],
            "ci_95_bpb": row["ci_bpb"],
            "stable_seed_count": row["stable_seed_count"],
            "total_seed_count": row["total_seed_count"],
        }

    # Stage 1 winner pick — paired-seed mean-bpb with noise tiebreak,
    # identical logic to Test 5/5b so the gate is comparable.
    stable_rows = [row for row in rows if row["stable_seed_count"] == row["total_seed_count"]]
    candidate: str | None = None
    gap_threshold = 0.006
    if len(stable_rows) >= 2:
        top = stable_rows[0]
        second = stable_rows[1]
        shared = sorted(set(top["bpb_by_seed"]) & set(second["bpb_by_seed"]))
        if len(shared) >= 2:
            top_paired = [top["bpb_by_seed"][s] for s in shared]
            second_paired = [second["bpb_by_seed"][s] for s in shared]
            _, p_top_vs_second = paired_ttest(second_paired, top_paired)
            top_paired_mean = sum(top_paired) / len(top_paired)
            second_paired_mean = sum(second_paired) / len(second_paired)
            gap = second_paired_mean - top_paired_mean
            if gap > gap_threshold or p_top_vs_second < 0.05:
                candidate = top["name"]
            else:
                # Inside noise: fall back to shorter seq (cheaper to train,
                # same signal). This mirrors Test 5's "conservative-LR
                # fallback" — the conservative direction for seq is
                # smaller, not larger.
                candidate = min(
                    (top, second), key=lambda r: r["seq_len"]
                )["name"]
        else:
            candidate = top["name"]
    elif len(stable_rows) == 1:
        candidate = stable_rows[0]["name"]

    winner = candidate

    summary["_decision"] = {
        "winner_condition": winner,
        "winner_seq_len": (
            next(row["seq_len"] for row in rows if row["name"] == winner)
            if winner else None
        ),
        "stage1_candidate": candidate,
        "stable_conditions": [row["name"] for row in stable_rows],
        "invalid_result_files": invalid_results,
    }
    if invalid_results:
        print(
            f"\nDropped {len(invalid_results)} non-finite result file(s): "
            f"{invalid_results}"
        )
    print(f"\nGate: winner = {winner}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 8 launcher — seq re-screen at bs_per_rank=1024"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument(
        "--num-slots", type=int, default=2,
        help="Number of parallel DDP groups (each uses 2 GPUs)",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if not args.summarize_only:
        validate_data_paths(args.data_path, args.sp_model_path)
        # skip_oom_conditions=True so seq=2048 (most memory-hungry
        # condition) can OOM without wiping the results from seq=512
        # and seq=1024. The harness drops the remaining seeds of an
        # OOMing condition and continues with the others.
        run_parallel_ddp_matrix(
            conditions=CONDITIONS,
            seeds=SWEEP_SEEDS,
            ws_per_slot=2,
            num_slots=args.num_slots,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS,
            runner_script=RUNNER_SSM,
            skip_oom_conditions=True,
        )

    summary = summarize_results(CONDITIONS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "test8_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
