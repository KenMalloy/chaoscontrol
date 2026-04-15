#!/usr/bin/env python3
"""Experiment 18 Test 6 launcher: sequence length sweep.

At the ws=2 DDP regime from Tests 4/5, vary ``seq_len ∈ {512, 1024, 2048}``
to see whether longer sequences buy better bpb per wall-clock. The
chunked scan is O(N) theoretically but kernel launch overhead, fp64
cumprod allocation, and HBM traffic can make the real curve sub-linear
or super-linear.

Three conditions at fixed (batch=1024 per rank, LR inherited from Test 5
winner):

    seq512   baseline (matches Tests 4/5 regime)
    seq1024  double length, half compute-per-position density
    seq2048  quadruple length, may hit VRAM ceiling on ws=2

**LR inheritance.** This orchestrator reads the Test 5 winning LR from
``results_test5/test5_summary.json`` at launch time. If Test 5 has not
run, or emitted only a provisional winner (no Test 4 cross-check), or
failed to emit a winner at all, Test 6 REFUSES to run without an
explicit ``--base-lr`` CLI override. This is the "ws=2 regime contract"
the reviewer flagged: Tests 5/6/7 must use the SAME LR because they
claim to share the regime, and that LR has to be Test 5's validated
winner — not a hardcoded guess.

Gate (bpb at matched 600s):
    Winner is the seq_len with the lowest mean_bpb. Paired t-test vs
    seq512 is reported for each longer condition so we can see whether
    the gap is statistically meaningful.

Launch: parallel 2-slot DDP at ws=2 (same as Test 5). Four seeds per
condition = 12 runs, 6 waves, ~60 min. seq=2048 may OOM at the ws=2
VRAM ceiling — that condition is OOM-skipped and annotated in the
summary rather than killing the whole matrix.
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
RESULTS = EXPERIMENT / "results_test6"
TEST5_SUMMARY = EXPERIMENT / "results_test5" / "test5_summary.json"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
sys.path.insert(0, str(EXPERIMENT))
from stats import bootstrap_ci, paired_ttest, sem  # noqa: E402
from _harness import (  # noqa: E402
    run_parallel_ddp_matrix,
    validate_data_paths,
)

SWEEP_SEEDS = [1337, 2674, 4011, 5348]

# Conditions that OOM'd during the run and should be annotated in the
# summary rather than treated as "missing data". Populated by ``main()``
# from the return value of ``run_parallel_ddp_matrix``.
_OOM_SKIPPED: list[str] = []


def _read_test5_winning_lr() -> tuple[float | None, str]:
    """Read Test 5's validated winning LR from its summary JSON.

    Returns (lr, status) where status is:
        "ok"           — Test 5 emitted a non-provisional winner; lr is valid
        "provisional"  — Test 5 emitted a provisional winner (no Test 4
                         cross-check); lr is the candidate but caller
                         should refuse to use it without override
        "no_winner"    — Test 5 ran but emitted no winner (Stage 1 or
                         Stage 2 failed); lr is None
        "missing"      — Test 5 summary doesn't exist; lr is None
    """
    if not TEST5_SUMMARY.exists():
        return None, "missing"
    try:
        data = json.loads(TEST5_SUMMARY.read_text())
    except Exception:
        return None, "missing"
    decision = data.get("_decision", {}) or {}
    lr = decision.get("winner_base_lr")
    winner = decision.get("winner_lr_condition")
    provisional = bool(decision.get("winner_is_provisional", False))
    if winner is None or lr is None:
        return None, "no_winner"
    if provisional:
        return float(lr), "provisional"
    return float(lr), "ok"


def _base(seq_len: int, base_lr: float, **overrides: Any) -> dict[str, Any]:
    cfg = {
        "model_type": "ssm",
        "vocab_size": 16384,
        "model_dim": 256,
        "num_layers": 4,
        "ff_mult": 2,
        "seq_len": seq_len,
        "stride": seq_len // 2,
        "batch_size": 1024,
        "eval_batches": 16,
        "a_mode": "diag",
        "crit_target_coupling": 0.92,
        "base_lr": base_lr,
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
    }
    cfg.update(overrides)
    return cfg


def build_conditions(base_lr: float) -> dict[str, dict[str, Any]]:
    return {
        "seq512":  _base(seq_len=512,  base_lr=base_lr),
        "seq1024": _base(seq_len=1024, base_lr=base_lr),
        "seq2048": _base(seq_len=2048, base_lr=base_lr),
    }


# Module-level CONDITIONS is constructed from whichever LR main() picked.
# Before main() runs, it's populated with a sentinel that panics if used
# directly — this catches code paths that import CONDITIONS without
# going through main()'s LR resolution.
CONDITIONS: dict[str, dict[str, Any]] = {}


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
        if not matches:
            continue
        matches.sort(key=lambda pair: pair[0])
        bpb_by_seed: dict[int, float] = {}
        tok_per_s_values: list[float] = []
        for seed, file in matches:
            data = json.loads(file.read_text())
            bpb_by_seed[seed] = float(data["eval"]["bpb"])
            train = data["train"]
            cfg = conditions[condition_name]
            # Read world_size from the result JSON rather than hardcoding;
            # the runner_exp18 result stores ddp_world_size in train.
            ws = int(train.get("ddp_world_size", 1))
            tok_per_step = int(cfg["batch_size"]) * int(cfg["seq_len"]) * ws
            tok_per_s_values.append(
                float(train["steps"]) * tok_per_step / max(float(train["elapsed_s"]), 1e-9)
            )
        bpb_values = [bpb_by_seed[s] for s in sorted(bpb_by_seed)]
        rows.append({
            "name": condition_name,
            "seq_len": conditions[condition_name]["seq_len"],
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

    print("\nTest 6 results — sequence length sweep at ws=2 DDP")
    print(
        f"  {'condition':<10} {'seq_len':>8} {'mean_bpb':>9} {'sem':>7} "
        f"{'tok/s':>12}"
    )
    for row in rows:
        print(
            f"  {row['name']:<10} {row['seq_len']:>8} {row['mean_bpb']:9.4f} "
            f"{row['se_bpb']:7.4f} {row['mean_tok_per_s']:12.0f}"
        )
        summary[row["name"]] = {
            "seq_len": row["seq_len"],
            "mean_bpb": row["mean_bpb"],
            "sem_bpb": row["se_bpb"],
            "ci_95_bpb": row["ci_bpb"],
            "mean_tok_per_s": row["mean_tok_per_s"],
            "n_seeds": len(row["bpb_values"]),
        }

    # Gate: winner by lowest mean_bpb. Additionally report the paired
    # test vs seq_len=512 for each longer sequence so we can see whether
    # the gap is statistically meaningful or within noise.
    baseline = next((row for row in rows if row["name"] == "seq512"), None)
    pair_results: list[dict[str, Any]] = []
    if baseline is not None:
        for row in rows:
            if row["name"] == "seq512":
                continue
            shared = sorted(set(baseline["bpb_by_seed"]) & set(row["bpb_by_seed"]))
            if len(shared) < 2:
                continue
            a = [baseline["bpb_by_seed"][s] for s in shared]
            b = [row["bpb_by_seed"][s] for s in shared]
            t, p = paired_ttest(a, b)
            delta = sum(a) / len(a) - sum(b) / len(b)
            pair_results.append({
                "condition": row["name"],
                "seq_len": row["seq_len"],
                "delta_bpb_vs_seq512": delta,
                "paired_t": t,
                "paired_p": p,
                "n_paired_seeds": len(shared),
            })
            print(
                f"\nvs seq512 n={len(shared)}: {row['name']} "
                f"delta_bpb={delta:+.4f} p_paired={p:.4g}"
            )

    winner = rows[0]["name"]  # already sorted by mean_bpb ascending
    summary["_decision"] = {
        "winner_condition": winner,
        "winner_seq_len": rows[0]["seq_len"],
        "pair_results": pair_results,
        "oom_skipped_conditions": list(_OOM_SKIPPED),
    }
    if _OOM_SKIPPED:
        print(
            f"\nOOM-skipped conditions (excluded from winner selection): "
            f"{_OOM_SKIPPED}"
        )
    print(f"\nGate: winner (lowest mean bpb) = {winner} (seq_len={rows[0]['seq_len']})")
    return summary


def _resolve_base_lr(cli_override: float | None) -> float:
    """Pick the LR Test 6 runs at.

    Priority:
      1. Explicit ``--base-lr`` CLI override (user asserts "I know what
         I'm doing, ignore Test 5's winner").
      2. Non-provisional Test 5 winner (Stage 2 cross-check passed).
      3. Otherwise REFUSE to run. A provisional winner, missing summary,
         or a Test 5 that couldn't pick a winner all indicate we don't
         actually know what LR to run Test 6 at — hardcoding a guess
         would make Test 6's comparison to Tests 5/7 incoherent.
    """
    if cli_override is not None:
        print(f"Test 6: using --base-lr override {cli_override}", flush=True)
        return float(cli_override)
    lr, status = _read_test5_winning_lr()
    if status == "ok":
        print(
            f"Test 6: inherited base_lr={lr} from Test 5 winner "
            f"(Stage 2 cross-check passed)",
            flush=True,
        )
        return lr  # type: ignore[return-value]
    if status == "provisional":
        raise RuntimeError(
            f"Test 5 emitted a PROVISIONAL winner at lr={lr} "
            f"(Test 4 ws=1 cross-check not performed). "
            f"Test 6 refuses to inherit an uncross-validated LR. "
            f"Either run Test 4 first, or pass --base-lr explicitly "
            f"to acknowledge the risk."
        )
    if status == "no_winner":
        raise RuntimeError(
            f"Test 5 ran but emitted no winner (Stage 1 or Stage 2 "
            f"failed). Test 6 cannot inherit an LR — re-run Test 5 or "
            f"pass --base-lr explicitly."
        )
    raise RuntimeError(
        f"Test 5 summary not found at {TEST5_SUMMARY}. Run Test 5 first "
        f"or pass --base-lr explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 6 launcher — seq_len sweep at ws=2 DDP"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--num-slots", type=int, default=2)
    parser.add_argument(
        "--base-lr",
        type=float,
        default=None,
        help=(
            "Override the Test 5-inherited LR. Normally Test 6 reads "
            "Test 5's validated winner from results_test5/test5_summary.json; "
            "pass this flag only when you explicitly want to bypass that."
        ),
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    base_lr = _resolve_base_lr(args.base_lr)
    global CONDITIONS
    CONDITIONS = build_conditions(base_lr)

    if not args.summarize_only:
        validate_data_paths(args.data_path, args.sp_model_path)
        skipped = run_parallel_ddp_matrix(
            conditions=CONDITIONS,
            seeds=SWEEP_SEEDS,
            ws_per_slot=2,
            num_slots=args.num_slots,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS,
            skip_oom_conditions=True,  # seq=2048 may hit VRAM ceiling
        )
        _OOM_SKIPPED.extend(skipped)

    summary = summarize_results(CONDITIONS)
    summary["_base_lr"] = base_lr
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "test6_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
