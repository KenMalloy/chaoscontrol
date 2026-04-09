# Experiment Summary — 2026-04-09 (Updated: all experiments complete)

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

### Baseline Sweep — COMPLETE (49 JSONs + 1 summary, 7 mamba2 failures expected)

| Condition | bpb | Steps | vs bare_ssm |
|-----------|-----|-------|-------------|
| **bare_ssm** | **2.478** | 505 | — |
| ssm_wernicke_k16 | 2.488 | 446 | +0.010 (worse) |
| ssm_wernicke_k32 | 2.546 | 414 | +0.068 (worse) |
| ssm_wernicke_k64 | 2.575 | 407 | +0.097 (worse) |
| full_stack_k16 | 2.502 | 433 | +0.024 (worse) |
| full_stack_k32 | 2.565 | 404 | +0.087 (worse) |
| full_stack_k64 | 2.574 | 412 | +0.096 (worse) |

**Key finding: bare_ssm is the best condition. All semantic engine additions hurt at 600s on A40.**

The pattern is consistent with exp 11: anything that costs training steps loses. bare_ssm gets 505 steps, ssm_wernicke_k16 gets 446 (Wernicke overhead), full_stack_k16 gets 433 (memory + Wernicke overhead). The 59-72 lost steps cost more bpb than the semantic engine recovers.

k_max=16 is the best within both Wernicke-only and full-stack families. More buckets = more overhead = fewer steps = worse. The expert bottleneck held params roughly constant, so this is genuinely "more experts hurts" not "more params hurts."

### Experiment 13: Constants Validation — COMPLETE (182/182, 0 failures)

| Constant | Default | Best | Delta | Action |
|----------|---------|------|-------|--------|
| crit_target_coupling | 0.88 | **0.92** | -0.017 | Confirm on 8+ seeds |
| outer_max_slots | 64 | **32** | -0.033 | Confirm on 8+ seeds |
| outer_model_dim | 64 | 32 | -0.010 | Trend only |
| semantic_tier | off | b8/r0.1 | -0.008 | Trend only, keep off |
| merge_threshold | 0.85 | 0.95 | -0.003 | Edge warning, extend range |

Two strong candidates: crit_target_coupling → 0.92 and outer_max_slots → 32.
Full report: `experiments/13_constants_validation/REPORT_exp13.md`

**Pod stopped.** No more experiments to run until H100 decisions.

## Decisions Waiting for Ken

1. **The biggest finding:** At 600s on A40, bare SSM beats everything. Wernicke routing, episodic memory, and sleep all cost more steps than they recover in bpb. The entire semantic engine is underwater at this budget. The question: does the semantic engine cross over at longer budgets (10 min H100)?

2. **Constants to lock:**
   - crit_target_coupling: change to 0.92? (0.017 bpb improvement on bare SSM)
   - outer_max_slots: change to 32? (0.033 bpb improvement on full stack)
   - Both need confirmatory rerun on 8+ seeds before locking

3. **Paper story:** The A40 data tells a throughput-dominance story. The H100 experiment is the one that determines whether the semantic engine crosses over.

4. **What's next:** Design and run the H100 crossover experiment. The key question: bare_ssm vs full_stack at 10 min on H100.
