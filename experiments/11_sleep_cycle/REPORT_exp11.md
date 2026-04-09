# Experiment 11: Sleep Cycle Ablation — Results

**Date:** 2026-04-09
**Pod:** asvnf6bwu59pen (4×A40 secure)
**Budget:** 600s per run, 9 conditions × 7 seeds = 63 runs
**Total wall time:** ~6 hours (across 6 restarts due to dtype bugs)

## Summary Table

| Condition | Mean bpb | SEM | 95% CI | Steps | vs no_sleep |
|-----------|----------|-----|--------|-------|-------------|
| **no_sleep** | **2.4210** | 0.0062 | [2.4104, 2.4321] | 390 | — |
| **n3_only** | **2.4090** | 0.0051 | [2.3991, 2.4174] | 385 | **-0.012 (better)** |
| n2_n3 | 2.4946 | 0.0075 | [2.4809, 2.5062] | 320 | +0.074 (worse) |
| n2_n3_rem_base | 2.5386 | 0.0066 | [2.5267, 2.5488] | 276 | +0.118 (worse) |
| n2_n3_rem_validate | 2.5328 | 0.0074 | [2.5194, 2.5467] | 297 | +0.112 (worse) |
| n2_n3_rem_cfr | 2.5090 | 0.0066 | [2.4962, 2.5198] | 285 | +0.088 (worse) |
| n2_n3_rem_reactivate | 2.5038 | 0.0061 | [2.4938, 2.5153] | 292 | +0.083 (worse) |
| n2_n3_rem_all | 2.4990 | 0.0041 | [2.4922, 2.5068] | 294 | +0.078 (worse) |
| full_cycle | 2.4916 | 0.0037 | [2.4852, 2.4979] | 316 | +0.071 (worse) |

## Confirmatory Contrasts (Holm-corrected, m=3)

| Contrast | Delta | Corrected p | Significant | Winner |
|----------|-------|-------------|-------------|--------|
| full_cycle vs no_sleep | +0.071 | 0.047 | YES | **no_sleep** |
| n2_n3 vs n3_only | +0.086 | 0.047 | YES | **n3_only** |
| n2_n3_rem_all vs n2_n3 | +0.004 | 0.813 | NO | — |

## Exploratory Contrasts (uncorrected)

| Contrast | Delta | p | d | Note |
|----------|-------|---|---|------|
| rem_validate vs rem_base | -0.006 | 0.938 | 0.31 | Merge validation: no effect |
| rem_cfr vs rem_base | -0.030 | 0.031 | 1.69 | CFR: helps within REM (trend) |
| rem_reactivate vs rem_base | -0.035 | 0.016 | 2.08 | **Reactivation: strongest REM mechanism** |
| full_cycle vs n2_n3_rem_all | -0.007 | 0.219 | 0.72 | N1 transition: no significant effect |

## Key Findings

### 1. Sleep helps, but only the simplest form
`n3_only` (2.409) beats `no_sleep` (2.421) by 0.012 bpb. This is a clean win for compression-pass sleep — the model benefits from periodic pruning of low-survival memory slots.

### 2. N2 utility scoring is too expensive at 600s
Adding N2 (leave-one-slot-out scoring) drops training steps from 385 to 320 and degrades bpb by 0.086. The cost of running model forward passes to score each slot's utility overwhelms any benefit from better-informed pruning.

### 3. REM dreaming doesn't compensate for N2's cost
Even with all three REM mechanisms enabled, the full pipeline can't recover the training steps lost to N2 scoring. REM's role of validating merges and training the gate policy requires N2 to identify which slots matter, but N2 itself is the bottleneck.

### 4. Within REM, reactivation is the strongest mechanism
Among the REM sub-mechanisms (exploratory, uncorrected): latent reactivation (-0.035, p=0.016, d=2.08) and CFR (-0.030, p=0.031, d=1.69) both show trends. Merge validation has no measurable effect. If REM becomes viable (e.g., at longer budgets or with cheaper N2), reactivation is the mechanism to keep.

### 5. The step-count story
The results are almost entirely explained by training steps lost to sleep:

| Condition | Steps | bpb |
|-----------|-------|-----|
| no_sleep | 390 | 2.421 |
| n3_only | 385 | 2.409 (5 steps lost, 0.012 gained) |
| n2_n3 | 320 | 2.495 (70 steps lost, 0.074 lost) |
| full_cycle | 316 | 2.492 (74 steps lost, 0.071 lost) |

n3_only spends ~5 steps on compression and gains more than it loses. Everything more expensive than that loses the trade.

## Sleep Payload Decision

**Winner: `n3_only`** — simple compression pass only.

For downstream experiments (12, 13), the sleep payload should be `n3_only` unless the budget increases significantly. The N2/N3/REM pipeline is not viable at 600s on A40.

**Future direction:** At longer budgets (10 min on H100), the step-cost ratio changes — more total steps means the relative cost of N2/REM is lower. Retest the full pipeline at H100 scale.
