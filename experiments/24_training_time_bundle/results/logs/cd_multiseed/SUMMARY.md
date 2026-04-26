# cd_multiseed (5 arms × 5 seeds, 25 cells, 600s/cell, ws=1)

Run started 2026-04-25T01:59:18Z, completed 2026-04-25T06:55Z (4h 56m).
Pod: u9eypeygtv98ri (1× H100). Commit: 76c5bc4.

## final_loss table

| seed | treatment | telemetry | shuffled | budget_only | H_short |
|------|-----------|-----------|----------|-------------|---------|
| 17   | 3.9604    | 3.8796    | 3.8921   | 3.9590      | 3.8887  |
| 42   | 3.9540    | 3.8825    | 3.8904   | 3.9546      | 3.8860  |
| 1234 | 3.8847    | 3.9616    | 3.9592   | 3.8935      | 3.9612  |
| 1337 | 3.8884    | 3.8813    | 3.8880   | 3.8870      | 3.8301  |
| 2024 | 3.8897    | 3.8911    | 3.9580   | 3.8919      | 3.9617  |

## per-arm summary (n=5 seeds each)

| arm | final_loss mean ± std | bucket0 (rare) mean ± std | bucket1 mean ± std |
|---|---|---|---|
| treatment   | 3.9154 ± 0.0383 | 12.4288 ± 0.0370 | 8.0692 ± 0.0279 |
| telemetry   | 3.8992 ± 0.0351 | 12.4241 ± 0.0518 | 8.0501 ± 0.0241 |
| shuffled    | 3.9175 ± 0.0375 | 12.4566 ± 0.0489 | 8.0527 ± 0.0301 |
| budget_only | 3.9172 ± 0.0363 | 12.4219 ± 0.0865 | 8.0567 ± 0.0508 |
| H_short     | 3.9055 ± 0.0562 | 12.4296 ± 0.0533 | 8.0661 ± 0.0447 |

bucket_token_counts: [1314, 20259, 13.8M, 28.4M].

## Headline

**No signal.** Across 5 paired seeds, no arm separates from any other on
final_loss, rare-bucket CE (b0), or bucket-1. Between-arm differences in
means are smaller than within-arm seed-to-seed std (~0.04 on final_loss,
~0.04-0.09 on bucket-0).

The first cd_first_smoke run's apparent **−0.10 nat treatment-vs-telemetry
gap on rare CE was n=1 noise**. The treatment vs telemetry paired b0 delta
across these 5 seeds is +0.005 ± 0.030 — i.e., zero.

## Paired (treatment − telemetry) by seed

| seed | ΔFL    | Δb0    | Δb1    |
|------|--------|--------|--------|
| 17   | +0.081 | +0.017 | +0.077 |
| 42   | +0.072 | −0.009 | −0.006 |
| 1234 | −0.077 | +0.050 | −0.009 |
| 1337 | +0.007 | −0.032 | +0.016 |
| 2024 | −0.001 | −0.002 | +0.017 |
| mean | +0.016 | +0.005 | +0.019 |
| std  |  0.064 |  0.030 |  0.034 |

## What this means for CD

The criticality distillation mechanism, as currently configured
(weight=1e-3, hl=256, H=16, budget_frac=0.15) and run for 600s on a
10.7M-param model, **does not produce a measurable rare-bucket benefit
at this scale**.

The ablation triangle from the first smoke (treatment beats all 3
falsifiers on rare CE) was a single-seed artifact, not a mechanistic
signal.

## What's still worth checking

- The seat overlap / churn / score-corr telemetry IS now logged in every
  cell's JSON (`criticality_distillation_diagnostics`). Worth eyeballing
  whether the score Spearman correlation with criticality is non-trivial,
  whether seats stabilize across seeds, whether CD-loss trajectory is
  monotonic. These are *mechanism-internal* checks even when the
  *behavioral* outcome is null.
- Larger CD weight (1e-2, 5e-2) might produce a measurable signal at the
  cost of common-head CE — current weight is essentially in the noise floor.
- Larger model / longer budget — 10.7M @ 600s may be too small/short for
  selection-style mechanisms to bite.

## Files

- `exp24_cd_multiseed_*.json` (25): full result per cell with new
  telemetry (`criticality_distill_loss_trajectory`, `seat_indices_per_layer`,
  `score_p10/50/90/max_per_layer`, `per_window_bucket_ce_sum`).
- `exp24_cd_multiseed_*.log` (25): per-cell stdout from the runner.
- `cd_multiseed_run.log`: main matrix driver log.
- `cd_treatment_s1337.log`: treatment_s1337 fill-cell driver log.

(Per `experiments/*/results/*` `.gitignore`, raw artifacts are not
committed; this SUMMARY.md is also under that pattern but is the
canonical readable record.)
