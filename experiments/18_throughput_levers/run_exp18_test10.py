#!/usr/bin/env python3
"""Experiment 18 Test 10 — fp8 vs bf16 at the submission regime.

The final Exp 18 lever. Exp 18's throughput thesis is "maximize tokens
seen in 10 min on 8 GPUs"; every prior test narrowed one axis of the
training recipe. Test 10 is the precision axis: does switching bf16 ->
fp8 through transformer_engine buy enough throughput to justify the
dtype risk at submission time?

**Conditions:**

    bf16: precision="bf16"  (current submission regime, Test 4b winner)
    fp8:  precision="fp8"   (te.Linear + te.fp8_autocast, same regime)

Both conditions share every other axis — batch, seq_len, stride,
optimizer, chunk_size, activation checkpoint, tokenizer, model shape,
and world_size — so the delta is purely attributable to precision.

**Inherited axes (from prior Exp 18 tests):**
    - world_size: read from Test 4b winner (4b picks among ws=2, ws=4)
    - base_lr: pinned to Test 4b's winning condition (ws=4 -> 0.128,
      ws=2 -> 0.064 by the linear-with-global-batch scaling rule
      Test 5b established)
    - seq_len=512, stride=256 (Test 8 winner)
    - Muon optimizer (Test 7 winner)
    - chunk_size=64 (Test 5b default)
    - activation_checkpoint=True (Test 8 regime)
    - SP16384 tokenizer
    - 4L x 256d SSM

**Pod-layout notes:**
    ws=4: 1 slot x 4 GPUs on a 4-GPU pod, one condition at a time,
          4 seeds per condition => 4 waves of 600s per condition.
          2 conditions * 4 waves = ~80 min wall-clock.
    ws=2: 2 slots x 2 GPUs, 4 seeds per condition = 2 waves of 600s
          per condition. 2 conditions * 2 waves = ~40 min wall-clock.

**Decision gate.** Two-axis: (1) fp8 stable on every seed (final
loss < 8.7; numerical collapse is the primary fp8 risk), and (2) fp8
buys meaningful throughput. "Meaningful" = 1.3x aggregate tokens/sec
over bf16 at matched wall-clock; below that the dtype risk at
submission isn't paid off by the speedup. If fp8's bpb degrades by
more than 0.02 at p<0.05 paired, that's an automatic no regardless
of throughput.

Launch:

    python experiments/18_throughput_levers/run_exp18_test10.py \\
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
RESULTS = EXPERIMENT / "results_test10"
RUNNER_SSM = EXPERIMENT / "runner_exp18_ssm.py"
TEST4B_SUMMARY = EXPERIMENT / "results_test4b" / "test4b_summary.json"

sys.path.insert(0, str(REPO / "experiments" / "09_revised_architecture"))
sys.path.insert(0, str(EXPERIMENT))
from stats import bootstrap_ci, paired_ttest, sem  # noqa: E402
from _harness import (  # noqa: E402
    result_is_finite,
    run_parallel_ddp_matrix,
    validate_data_paths,
)

SWEEP_SEEDS = [1337, 2674, 4011, 5348]

# OOM-skipped conditions, populated from run_parallel_ddp_matrix return.
_OOM_SKIPPED: list[str] = []


def _read_test4b_winning_ws() -> tuple[tuple[int, float] | None, str]:
    """Read Test 4b's winning (world_size, base_lr) tuple.

    Test 4b's summary doesn't have a ``winner_ws`` key — it has a
    ``recommendation`` string plus per-condition rows. Reconstruct the
    winner by filtering to conditions with 100% stable seeds, then
    picking the one with the highest ``aggregate_tokens_per_sec``.

    Returns ((ws, base_lr), status). Status is "ok" on success,
    "missing" if the summary file doesn't exist, or "no_winner" if no
    fully-stable condition exists in the summary.
    """
    if not TEST4B_SUMMARY.exists():
        return None, "missing"
    try:
        data = json.loads(TEST4B_SUMMARY.read_text())
    except Exception:
        return None, "missing"

    candidates: list[tuple[float, int, float]] = []
    for name, row in data.items():
        # Skip sidecars — ``_decision``, ``_base_lr_ws2_anchor``, etc.
        if not isinstance(row, dict) or name.startswith("_"):
            continue
        stable = int(row.get("stable_seed_count", 0))
        total = int(row.get("total_seed_count", 0))
        if total == 0 or stable != total:
            continue
        tps = float(row.get("aggregate_tokens_per_sec", 0.0))
        ws = int(row.get("world_size", 0))
        lr = float(row.get("base_lr", 0.0))
        if ws <= 0:
            continue
        candidates.append((tps, ws, lr))

    if not candidates:
        return None, "no_winner"
    # Highest aggregate tokens/sec wins — matches Test 4b's throughput
    # thesis (the metric its recommendation block optimizes for).
    candidates.sort(key=lambda row: row[0], reverse=True)
    _, ws, lr = candidates[0]
    return (ws, lr), "ok"


def _resolve_ws_and_lr(
    cli_ws_override: int | None,
    cli_lr_override: float | None,
) -> tuple[int, float]:
    """Pick the (world_size, base_lr) for Test 10.

    Priority:
      1. ``--world-size`` + ``--base-lr`` CLI overrides (smoke test path).
      2. Test 4b winner from results_test4b/test4b_summary.json.
      3. Fail fast if neither is available.

    Partial override (one of ws/lr but not the other) is also allowed:
    CLI-provided values win per-field, with the summary filling in the
    rest. Fails fast if a needed field isn't resolvable.
    """
    # ws and base_lr are LINKED via the linear-with-global-batch rule
    # (Test 5→5b evidence). Partial override is a scientifically-invalid
    # regime — a bare ``--world-size 2`` would inherit Test 4b's LR
    # calibrated for ws=4, producing a mis-calibrated fp8-vs-bf16
    # comparison. Reject partial overrides up front.
    partial_override = (cli_ws_override is None) != (cli_lr_override is None)
    if partial_override:
        raise RuntimeError(
            "--world-size and --base-lr must be set TOGETHER or both "
            "left unset. They are linked by the linear-with-global-batch "
            "scaling rule (Test 5→5b); overriding only one produces a "
            "mis-calibrated regime that invalidates the fp8-vs-bf16 "
            "comparison. Unset both to inherit the coherent pair from "
            "Test 4b, or set both to a known-good coherent pair."
        )

    if cli_ws_override is not None and cli_lr_override is not None:
        print(
            f"Test 10: using explicit CLI pair ws={cli_ws_override} "
            f"base_lr={cli_lr_override} (user takes responsibility for "
            f"coherence)",
            flush=True,
        )
        return int(cli_ws_override), float(cli_lr_override)

    summary_result, status = _read_test4b_winning_ws()
    if status == "ok" and summary_result is not None:
        ws, lr = summary_result
        print(
            f"Test 10: inherited coherent pair (ws={ws}, base_lr={lr}) "
            f"from Test 4b winner",
            flush=True,
        )
        return ws, lr

    raise RuntimeError(
        f"Test 10 could not resolve (world_size, base_lr) (status={status!r}). "
        f"Either run Test 4b first so its summary exists at {TEST4B_SUMMARY}, "
        f"or pass BOTH --world-size AND --base-lr as a coherent pair."
    )


def _base(
    world_size: int,
    base_lr: float,
    precision: str,
    **overrides: Any,
) -> dict[str, Any]:
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
        "precision": precision,
        "local_attn_window": 0,
        "local_attn_heads": 1,
        "local_attn_dim": 64,
    }
    cfg.update(overrides)
    return cfg


def build_conditions(world_size: int, base_lr: float) -> dict[str, dict[str, Any]]:
    """Return {name: config}. The two conditions differ only in
    ``precision``; every other axis is locked to the Test 4b winner
    so the A/B is purely attributable to dtype.
    """
    return {
        "bf16": _base(world_size=world_size, base_lr=base_lr, precision="bf16"),
        "fp8": _base(world_size=world_size, base_lr=base_lr, precision="fp8"),
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
    # Per-seed bpb for paired t-test later; only seeds present in BOTH
    # conditions count toward the paired test.
    bpb_by_condition_seed: dict[str, dict[int, float]] = {}

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
        base_lr_by_seed: dict[int, float] = {}
        ws_by_seed: dict[int, int] = {}
        precision_by_seed: dict[int, str] = {}
        for seed, file in matches:
            data = json.loads(file.read_text())
            if not result_is_finite(data):
                invalid_results.append(f"{condition_name}_s{seed}")
                continue
            bpb_by_seed[seed] = float(data["eval"]["bpb"])
            loss_by_seed[seed] = float(data["train"]["final_loss"])
            steps_by_seed[seed] = int(data["train"]["steps"])
            sps_by_seed[seed] = float(data["train"]["steps_per_second"])
            # Read ACTUAL run parameters from the result JSON, not from
            # the reconstructed condition. Otherwise --summarize-only
            # after a Test 4b/5b rerun would silently relabel old results.
            base_lr_by_seed[seed] = float(data["config"]["base_lr"])
            ws_by_seed[seed] = int(
                data["config"].get("world_size", cfg.get("world_size", 0))
            )
            precision_by_seed[seed] = str(
                data["config"].get("precision", cfg.get("precision", "bf16"))
            )

        bpb_values = [bpb_by_seed[s] for s in sorted(bpb_by_seed)]
        stable_count = sum(1 for v in loss_by_seed.values() if _loss_is_stable(v))
        bpb_by_condition_seed[condition_name] = dict(bpb_by_seed)

        # Guard against JSONs with mixed configs in the same condition dir.
        for label, values in (
            ("world_size", set(ws_by_seed.values())),
            ("base_lr", set(base_lr_by_seed.values())),
            ("precision", set(precision_by_seed.values())),
        ):
            if len(values) > 1:
                raise RuntimeError(
                    f"condition {condition_name!r} has result JSONs with "
                    f"mixed {label} values {sorted(values)}; cannot "
                    f"summarize. Clear {RESULTS} and rerun."
                )

        ws = int(next(iter(ws_by_seed.values()))) if ws_by_seed else int(cfg.get("world_size", 0))
        actual_lr = (
            float(next(iter(base_lr_by_seed.values())))
            if base_lr_by_seed else float(cfg.get("base_lr", 0.0))
        )
        actual_precision = (
            str(next(iter(precision_by_seed.values())))
            if precision_by_seed else str(cfg.get("precision", "bf16"))
        )

        tokens_per_step = ws * int(cfg["batch_size"]) * int(cfg["seq_len"])
        mean_sps = _mean(list(sps_by_seed.values()))
        aggregate_tokens_per_sec = mean_sps * tokens_per_step

        rows.append({
            "name": condition_name,
            "precision": actual_precision,  # from JSON
            "world_size": ws,
            "base_lr": actual_lr,  # from JSON
            "bpb_by_seed": bpb_by_seed,
            "bpb_values": bpb_values,
            "mean_bpb": _mean(bpb_values),
            "se_bpb": sem(bpb_values),
            "ci_bpb": bootstrap_ci(bpb_values) if bpb_values else (0.0, 0.0),
            "stable_seed_count": stable_count,
            "total_seed_count": len(bpb_values),
            "expected_seed_count": len(SWEEP_SEEDS),
            "mean_steps_per_second": mean_sps,
            "tokens_per_step": tokens_per_step,
            "aggregate_tokens_per_sec": aggregate_tokens_per_sec,
        })

    # Sort rows by precision (bf16 then fp8) for consistent reporting.
    precision_order = {"bf16": 0, "fp8": 1}
    rows.sort(key=lambda row: precision_order.get(row["precision"], 99))
    if not rows:
        return summary

    print("\nTest 10 results — fp8 vs bf16 at Test 4b submission regime")
    hdr = (
        f"  {'condition':<8} {'prec':<5} {'ws':>3} {'lr':>7} {'stable':>8} "
        f"{'mean_bpb':>9} {'sem':>7} {'steps/s':>9} {'tok/s':>11}"
    )
    print(hdr)
    for row in rows:
        print(
            f"  {row['name']:<8} {row['precision']:<5} {row['world_size']:3d} "
            f"{row['base_lr']:7.4f} "
            f"{row['stable_seed_count']}/{row['total_seed_count']:<6} "
            f"{row['mean_bpb']:9.4f} {row['se_bpb']:7.4f} "
            f"{row['mean_steps_per_second']:9.3f} {row['aggregate_tokens_per_sec']:11,.0f}"
        )
        summary[row["name"]] = {
            "precision": row["precision"],
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

    decision: dict[str, Any] = {
        "oom_skipped_conditions": list(_OOM_SKIPPED),
        "invalid_result_files": invalid_results,
    }

    bf16_row = next((r for r in rows if r["name"] == "bf16"), None)
    fp8_row = next((r for r in rows if r["name"] == "fp8"), None)

    # Handle fp8-missing case explicitly — "fp8 OOMed or errored before
    # producing JSONs" is a decisive outcome ("submit at bf16"), not a
    # quiet absence of recommendation. Same logic for the degenerate
    # bf16-missing case (shouldn't happen in practice but handle it).
    fp8_missing = (fp8_row is None) or (fp8_row.get("total_seed_count", 0) == 0)
    bf16_missing = (bf16_row is None) or (bf16_row.get("total_seed_count", 0) == 0)
    fp8_in_oom_skipped = "fp8" in _OOM_SKIPPED
    bf16_in_oom_skipped = "bf16" in _OOM_SKIPPED

    if bf16_missing and fp8_missing:
        decision["recommendation"] = (
            "NO DATA — both conditions missing; re-run Test 10"
        )
    elif fp8_missing:
        reason = "OOMed" if fp8_in_oom_skipped else "errored or produced no JSONs"
        decision["recommendation"] = (
            f"fp8 {reason} — submit at bf16 (fp8 cannot complete at this regime)"
        )
    elif bf16_missing:
        # Unusual — typically fp8 is the one that breaks. But be
        # explicit rather than silent.
        reason = "OOMed" if bf16_in_oom_skipped else "errored or produced no JSONs"
        decision["recommendation"] = (
            f"bf16 {reason} — cannot make a comparison; investigate bf16 "
            f"before trusting any fp8-only data"
        )

    if bf16_row and fp8_row and not (bf16_missing or fp8_missing):
        # Throughput ratio: fp8 / bf16. >1.0 = fp8 wins; 1.3 is the
        # gate threshold for "meaningful" throughput speedup.
        if bf16_row["aggregate_tokens_per_sec"] > 0:
            throughput_ratio = (
                fp8_row["aggregate_tokens_per_sec"]
                / bf16_row["aggregate_tokens_per_sec"]
            )
        else:
            throughput_ratio = 0.0

        # Paired t-test over seeds that completed under BOTH conditions.
        bf16_seeds = bpb_by_condition_seed.get("bf16", {})
        fp8_seeds = bpb_by_condition_seed.get("fp8", {})
        paired_seeds = sorted(set(bf16_seeds) & set(fp8_seeds))
        fp8_minus_bf16: list[float] = []
        paired_bf16: list[float] = []
        paired_fp8: list[float] = []
        for seed in paired_seeds:
            paired_bf16.append(bf16_seeds[seed])
            paired_fp8.append(fp8_seeds[seed])
            fp8_minus_bf16.append(fp8_seeds[seed] - bf16_seeds[seed])
        if len(paired_seeds) >= 2:
            _, paired_p = paired_ttest(paired_fp8, paired_bf16)
            bpb_delta_paired = _mean(fp8_minus_bf16)
        else:
            paired_p = float("nan")
            bpb_delta_paired = float("nan")

        # Recommendation gate. A row is "fully stable" only if it has
        # ALL expected seeds AND all are stable. The previous gate only
        # checked stable_count == total_count, which green-lights a
        # partial 1-of-4 run. Require full coverage to gate a submission.
        def _fully_stable(row: dict[str, Any]) -> bool:
            expected = row.get("expected_seed_count", 0)
            return (
                row["total_seed_count"] == expected
                and row["stable_seed_count"] == expected
            )

        bf16_all_stable = _fully_stable(bf16_row)
        fp8_all_stable = _fully_stable(fp8_row)
        import math
        if not bf16_all_stable and not fp8_all_stable:
            recommendation = (
                "BOTH conditions incomplete or unstable at ws×LR regime; "
                "re-run before trusting a submission decision"
            )
        elif not fp8_all_stable:
            recommendation = (
                "fp8 UNSTABLE or incomplete — submit at bf16"
            )
        elif not bf16_all_stable:
            recommendation = (
                "bf16 UNSTABLE or incomplete — investigate before trusting fp8"
            )
        elif throughput_ratio < 1.3:
            recommendation = (
                f"fp8 throughput win insufficient ({throughput_ratio:.2f}× "
                f"< 1.3× gate) — submit at bf16"
            )
        elif (
            math.isfinite(bpb_delta_paired)
            and bpb_delta_paired > 0.02
            and math.isfinite(paired_p)
            and paired_p < 0.05
        ):
            recommendation = (
                f"fp8 degrades bpb (+{bpb_delta_paired:.4f}, p={paired_p:.4g}) "
                f"— submit at bf16 despite {throughput_ratio:.2f}× throughput"
            )
        else:
            recommendation = (
                f"fp8 winner — submit at fp8 ({throughput_ratio:.2f}× "
                f"throughput, bpb delta {bpb_delta_paired:+.4f})"
            )

        decision.update({
            "fp8_vs_bf16_throughput_ratio": throughput_ratio,
            "fp8_vs_bf16_bpb_delta_paired": bpb_delta_paired,
            "n_paired_seeds": len(paired_seeds),
            "paired_p": paired_p,
            "recommendation": recommendation,
        })

        print(
            f"\nComparison (fp8 vs bf16):"
            f"\n  throughput ratio (fp8/bf16): {throughput_ratio:.3f} "
            f"(1.3x = gate threshold)"
            f"\n  bpb delta paired (fp8-bf16): {bpb_delta_paired:+.4f} "
            f"(n={len(paired_seeds)} paired seeds, p={paired_p:.4f})"
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
        description="Exp 18 Test 10 launcher — fp8 vs bf16 at submission regime"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument(
        "--num-slots", type=int, default=None,
        help=(
            "Override the number of parallel slots. Defaults to "
            "4 // world_size on a 4-GPU pod (ws=4 -> 1 slot, ws=2 -> 2 slots)."
        ),
    )
    parser.add_argument(
        "--world-size", type=int, default=None,
        help=(
            "Override the DDP world size. Defaults to Test 4b's winning "
            "condition's world_size."
        ),
    )
    parser.add_argument(
        "--base-lr", type=float, default=None,
        help=(
            "Override the base LR. Defaults to Test 4b's winning "
            "condition's base_lr (scales with world_size per the "
            "linear-with-global-batch rule from Test 5b)."
        ),
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    world_size, base_lr = _resolve_ws_and_lr(args.world_size, args.base_lr)
    global CONDITIONS
    CONDITIONS = build_conditions(world_size, base_lr)

    # Pre-flight TE availability check. If TE isn't importable, the fp8
    # condition will hard-fail on every seed rather than OOM-skip cleanly
    # (autocast_context("fp8") raises RuntimeError before training starts,
    # not a CUDA OOM in torch allocator). Drop fp8 from the matrix up
    # front, run bf16 only, and mark the fp8 skip in _OOM_SKIPPED so the
    # summary reports "fp8 SKIPPED — TE unavailable on pod; submit at
    # bf16" instead of crashing the entire matrix.
    if not args.summarize_only and "fp8" in CONDITIONS:
        try:
            from chaoscontrol.precision import _check_te_available
        except Exception:
            _check_te_available = lambda: False  # type: ignore
        if not _check_te_available():
            print(
                "Test 10: transformer_engine is NOT importable on this pod "
                "(likely CUDA version mismatch). Dropping fp8 condition "
                "from the matrix; running bf16 only.",
                flush=True,
            )
            del CONDITIONS["fp8"]
            _OOM_SKIPPED.append("fp8")

    if not args.summarize_only:
        validate_data_paths(args.data_path, args.sp_model_path)

        # Single matrix call: both conditions share ws so a single
        # ``run_parallel_ddp_matrix`` can interleave them across slots.
        # On a 4-GPU pod: ws=4 -> 1 slot (conditions serialize),
        # ws=2 -> 2 slots (conditions can run in parallel within a wave).
        if args.num_slots is not None:
            num_slots = int(args.num_slots)
        else:
            num_slots = max(1, 4 // world_size)

        skipped = run_parallel_ddp_matrix(
            conditions=CONDITIONS,
            seeds=SWEEP_SEEDS,
            ws_per_slot=world_size,
            num_slots=num_slots,
            data_path=args.data_path,
            sp_model_path=args.sp_model_path,
            budget=args.budget,
            results_dir=RESULTS,
            runner_script=RUNNER_SSM,
            skip_oom_conditions=True,
        )
        _OOM_SKIPPED.extend(skipped)

    summary = summarize_results(CONDITIONS)
    summary["_world_size"] = world_size
    summary["_base_lr"] = base_lr
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "test10_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
