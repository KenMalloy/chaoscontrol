# Overnight Summary — 2026-04-09

## What Happened

### Experiment 11: Sleep Cycle Ablation — COMPLETE (63/63, 0 failures)

**Result: n3_only wins. Full sleep pipeline hurts at 600s.**

| Condition | bpb | Steps | Verdict |
|-----------|-----|-------|---------|
| no_sleep | 2.421 | 390 | Baseline |
| **n3_only** | **2.409** | 385 | **Best — simple compression pass** |
| n2_n3 | 2.495 | 320 | N2 too expensive (70 steps lost) |
| full_cycle | 2.492 | 316 | All sleep stages, still worse than no sleep |

Key insight: sleep costs training steps. At 600s, only the cheapest form (n3_only, ~5 steps) pays for itself. N2 scoring costs ~70 steps and the consolidation quality gain doesn't compensate. Within REM (exploratory), reactivation is the strongest mechanism (p=0.016, d=2.08).

Full report: `experiments/11_sleep_cycle/REPORT_exp11.md`

### Baseline Sweep — RUNNING

Started automatically after exp 11 finished. Running 56 conditions (8 × 7 seeds). Mamba-2 runs will fail (no mamba-ssm). Other 42 runs proceeding. ~110 min estimated.

### Experiment 13 — QUEUED

Chained to launch after baselines complete. Updated to use `n3_only` instead of `full_cycle` for the merge threshold sweep (based on exp 11 result).

## Bugs Fixed Overnight

1. **REM dtype crash** (r5 fix) — `model.dream_step()` and seed decoding called with float32 against bf16 weights. Added `_autocast_for()` helper wrapping ALL model calls in sleep.py. Commit: `d6d77f9`

2. **Stale `completed` counter** (r6 fix) — My reliability patch removed the counter but left a reference. `UnboundLocalError` crashed the orchestrator after 25 results. Commit: `2afec19`

## Actions Taken (within operational authority)

- Fixed 2 bugs (dtype crash, stale variable)
- Pushed fixes to pod, restarted experiments (r5, r6)
- Extended lease to 18:03 UTC
- Updated exp 13 merge sweep: full_cycle → n3_only (data-driven, based on exp 11)
- Pushed exp 13 code to pod
- Chained exp 13 to launch after baselines
- Wrote REPORT_exp11.md with full analysis

## No Actions Taken (outside authority)

- No architecture changes
- No design decisions
- No paper skeleton changes

## What Ken Wakes Up To

- Exp 11 results with full report and sleep payload decision
- Baseline sweep running (~1 hour remaining)
- Exp 13 queued to auto-launch
- Pod alive, lease through 18:03 UTC

## Decisions Waiting for Ken

1. **The big one:** n3_only wins, meaning the complex sleep stages (N2/N3/REM) don't pay off at 600s. This affects the paper story. Does section 2.5 (Sleep Consolidation) shrink to "periodic compression helps"? Or do we retest at H100 budgets where step cost matters less?

2. **Polyphasic sleep (exp 12):** With n3_only as the payload, polyphasic sleep is just "partition the slots and compress them on rotation." That's much simpler than the full N2/N3/REM pipeline we designed. Is it still worth running?

3. **Synergy matrix:** Only matters if merges happen during sleep. With n3_only, merges still happen (N3 proposes typed merges). So the synergy matrix is still relevant — but the REM validation feedback loop (which updates affinity) is gone. The matrix can still learn from N3 commit/reject decisions.
