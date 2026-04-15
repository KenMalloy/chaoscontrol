#!/usr/bin/env python3
"""Experiment 18 Test 7 launcher: AdamW vs Muon vs LAMB under ws=2 DDP.

Test 7 is the optimizer ablation called out in the Exp 18 design doc. It
runs at the same ws=2 DDP regime as Tests 5 and 6 so that the three
tests share a common large-batch baseline. Three optimizer conditions
against the same bare-SSM config (bs=1024 per rank, seq=512, LR
inherited from the Test 5 winner):

    adamw_baseline  current default
    muon            Newton-Schulz matrix orthogonalization + AdamW fallback
    lamb            per-tensor trust ratio, architecture-agnostic large-batch

**LR inheritance.** Test 7 reads Test 5's validated winning LR from
``results_test5/test5_summary.json`` and refuses to run on a provisional
or missing winner without an explicit ``--base-lr`` override. This
matches Test 6's contract — all three tests (5/6/7) that claim to share
the "ws=2 regime" must actually use the same LR, and that LR must be
Test 5's Stage-2 cross-validated winner.

Gate:
    Any alternative beats AdamW at paired p<0.05 on mean bpb across the
    matched seeds -> that alternative becomes the submission optimizer.
    Otherwise AdamW stays. The per-pair comparison (muon vs adamw,
    lamb vs adamw, muon vs lamb) is printed for the full three-way
    picture, not just the winning pair.

**Statistical power caveat:** at n=4 and per-condition noise ~0.004 bpb,
the paired test is only powered for differences larger than ~0.008 bpb.
Optimizer deltas at matched budget may be smaller than that. "No
winner, AdamW stays" in this test could reflect low power as much as
no real effect.

Launch: parallel 2-slot DDP at ws=2 via ``_harness.run_parallel_ddp_matrix``.
Four seeds per condition x three optimizers = 12 runs, 6 waves, ~60 min.
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
RESULTS = EXPERIMENT / "results_test7"
TEST5_SUMMARY = EXPERIMENT / "results_test5" / "test5_summary.json"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
sys.path.insert(0, str(EXPERIMENT))
from stats import bootstrap_ci, paired_ttest, sem  # noqa: E402
from _harness import (  # noqa: E402
    run_parallel_ddp_matrix,
    validate_data_paths,
)

SWEEP_SEEDS = [1337, 2674, 4011, 5348]


def _read_test5_winning_lr() -> tuple[float | None, str]:
    """Same contract as Test 6's helper. Returns (lr, status) where
    status in {"ok", "provisional", "no_winner", "missing"}.
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


def _base(optimizer: str, base_lr: float, **overrides: Any) -> dict[str, Any]:
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
        "crit_target_coupling": 0.92,
        "base_lr": base_lr,
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
        "optimizer": optimizer,
    }
    cfg.update(overrides)
    return cfg


def build_conditions(base_lr: float) -> dict[str, dict[str, Any]]:
    return {
        "adamw_baseline": _base(optimizer="adamw", base_lr=base_lr),
        "muon":           _base(optimizer="muon",  base_lr=base_lr),
        "lamb":           _base(optimizer="lamb",  base_lr=base_lr),
    }


CONDITIONS: dict[str, dict[str, Any]] = {}


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
        if not matches:
            continue
        matches.sort(key=lambda pair: pair[0])
        bpb_by_seed: dict[int, float] = {}
        for seed, file in matches:
            data = json.loads(file.read_text())
            bpb_by_seed[seed] = float(data["eval"]["bpb"])
        bpb_values = [bpb_by_seed[s] for s in sorted(bpb_by_seed)]
        rows.append({
            "name": condition_name,
            "bpb_by_seed": bpb_by_seed,
            "bpb_values": bpb_values,
            "mean_bpb": _mean(bpb_values),
            "se_bpb": sem(bpb_values),
            "ci_bpb": bootstrap_ci(bpb_values),
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
    # (a, b): positive t => a > b in bpb => b is the better optimizer.
    # Delta uses the same paired sample set as the p-value so partial
    # reruns don't split the two statistics across different cohorts.
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

    print("\nTest 7 results — AdamW vs Muon vs LAMB at ws=2 DDP")
    print(
        f"  {'condition':<20} {'mean_bpb':>9} {'sem':>7} {'95% CI':>21}"
    )
    for row in rows:
        print(
            f"  {row['name']:<20} {row['mean_bpb']:9.4f} {row['se_bpb']:7.4f} "
            f"[{row['ci_bpb'][0]:.4f}, {row['ci_bpb'][1]:.4f}]"
        )
        summary[row["name"]] = {
            "mean_bpb": row["mean_bpb"],
            "sem_bpb": row["se_bpb"],
            "ci_95_bpb": row["ci_bpb"],
            "n_seeds": len(row["bpb_values"]),
        }

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


def _resolve_base_lr(cli_override: float | None) -> float:
    """Pick the LR Test 7 runs at. Matches Test 6's contract: explicit
    override > non-provisional Test 5 winner > hard refuse."""
    if cli_override is not None:
        print(f"Test 7: using --base-lr override {cli_override}", flush=True)
        return float(cli_override)
    lr, status = _read_test5_winning_lr()
    if status == "ok":
        print(
            f"Test 7: inherited base_lr={lr} from Test 5 winner "
            f"(Stage 2 cross-check passed)",
            flush=True,
        )
        return lr  # type: ignore[return-value]
    if status == "provisional":
        raise RuntimeError(
            f"Test 5 emitted a PROVISIONAL winner at lr={lr}. "
            f"Test 7 refuses to inherit an uncross-validated LR. "
            f"Run Test 4 first or pass --base-lr explicitly."
        )
    if status == "no_winner":
        raise RuntimeError(
            "Test 5 ran but emitted no winner. Test 7 cannot inherit "
            "an LR — re-run Test 5 or pass --base-lr explicitly."
        )
    raise RuntimeError(
        f"Test 5 summary not found at {TEST5_SUMMARY}. Run Test 5 first "
        f"or pass --base-lr explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 18 Test 7 launcher — AdamW vs Muon vs LAMB at ws=2 DDP"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--num-slots", type=int, default=2,
                        help="Number of parallel DDP groups (each uses 2 GPUs)")
    parser.add_argument(
        "--base-lr",
        type=float,
        default=None,
        help=(
            "Override the Test 5-inherited LR. Normally Test 7 reads "
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
        run_parallel_ddp_matrix(
            conditions=CONDITIONS,
            seeds=SWEEP_SEEDS,
            ws_per_slot=2,
            num_slots=args.num_slots,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS,
        )

    summary = summarize_results(CONDITIONS)
    summary["_base_lr"] = base_lr
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "test7_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
