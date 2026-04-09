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

### Experiment 13 — RUNNING

Launched after baselines completed. 182 runs (26 conditions × 7 seeds), 46 batches, ~8 hours. Running criticality sweep first.

## Decisions Waiting for Ken

1. **The biggest finding:** At 600s on A40, bare SSM beats everything. Wernicke routing, episodic memory, and sleep all cost more steps than they recover in bpb. The entire semantic engine is underwater at this budget. This is the same pattern as Phase 1 (memory hurt at 150s). The question: does the semantic engine cross over at longer budgets (10 min H100)? Phase 1→Phase 2 showed memory crossed over between 150s and 600s. Will Wernicke and sleep cross over between 600s and 10 min?

2. **Paper story impact:** If the semantic engine only pays off at H100 budgets, the A40 ablation tells a "throughput dominates at short budgets, but the semantic engine's value grows with budget" story. That's still a valid paper — it just needs the H100 data to land the punch.

3. **Polyphasic sleep (exp 12):** May not be worth running on A40 if even basic sleep barely helps. Save for H100.

4. **k_max:** k16 is the best. More buckets hurt. The expert bottleneck prevented the param confound, so this is a clean finding. Lock k_max=16 for now.

5. **Synergy matrix:** Hard to test when the base architecture (Wernicke + memory) is itself underwater. Save for H100.
