# Experiment 13: Constants Validation — Results

**Date:** 2026-04-09
**Pod:** asvnf6bwu59pen (4×A40 secure)
**Budget:** 600s per run, 26 conditions × 7 seeds = 182 runs, 0 failures
**Total wall time:** ~8 hours

## Sweep 1: Criticality Target (bare SSM)

| Condition | Mean bpb | Steps | vs default (0.88) |
|-----------|----------|-------|--------------------|
| crit_080 | 2.4986 | 474 | -0.007 (better) |
| crit_085 | 2.5074 | 465 | +0.002 |
| **crit_088** | **2.5052** | 468 | **default** |
| **crit_092** | **2.4887** | 484 | **-0.017 (best)** |
| crit_096 | 2.4956 | 480 | -0.010 (better) |

**Finding:** crit_092 is the best (2.489 vs 2.505 default). The optimal coupling is higher than our default — slightly closer to critical. The improvement is 0.017 bpb. Winner is NOT at the sweep edge, so the range is adequate.

**Recommendation:** CANDIDATE crit_092 for confirmatory rerun on 8+ seeds.

## Sweep 2: Memory Slot Dimension (full stack, no sleep)

| Condition | Mean bpb | Steps | vs default (64) |
|-----------|----------|-------|--------------------|
| **memdim_032** | **2.5148** | 413 | **-0.010 (best)** |
| memdim_064 | 2.5250 | 406 | default |
| memdim_128 | 2.5176 | 414 | -0.007 (better) |

**Finding:** memdim_032 is the best — smaller slot embeddings beat the default. But all three are very close (0.010 bpb range). Smaller dim means less overhead per slot operation.

**Recommendation:** TREND memdim_032 looks better but the effect is small. Keep 64 unless confirmed.

## Sweep 3: Max Slots (full stack, no sleep)

| Condition | Mean bpb | Steps | vs default (64) |
|-----------|----------|-------|--------------------|
| **slots_032** | **2.5036** | 423 | **-0.033 (best)** |
| slots_064 | 2.5364 | 391 | default |
| slots_128 | 2.5171 | 410 | -0.019 (better) |

**Finding:** slots_032 is the best (2.504 vs 2.536 default). The 64-slot default is the WORST. Fewer slots = less compression overhead = more training steps (423 vs 391). The compression frequency at 64 slots is eating too many steps.

**Recommendation:** CANDIDATE slots_032 for confirmatory rerun. This is a meaningful improvement (0.033 bpb).

## Sweep 4: Semantic Tier (full stack, bases × update_rate)

| Condition | Mean bpb | Steps | Note |
|-----------|----------|-------|------|
| **sem_off** | **2.5196** | 420 | **default (disabled)** |
| sem_b8_r1em01 | 2.5120 | 419 | best enabled config |
| sem_b4_r1em02 | 2.5230 | 398 | |
| sem_b4_r1em01 | 2.5298 | 401 | |
| sem_b4_r1em03 | 2.5322 | 396 | |
| sem_b8_r1em02 | 2.5255 | 399 | |
| sem_b8_r1em03 | 2.5278 | 405 | |
| sem_b16_r1em01 | 2.5271 | 398 | |
| sem_b16_r1em03 | 2.5409 | 382 | |
| sem_b16_r1em02 | 2.5535 | 368 | worst |

**Finding:** sem_b8_r1em01 (8 bases, 0.1 update rate) slightly beats sem_off (2.512 vs 2.520). But most enabled configs are WORSE than off. The dependency mattered: fast update rate (0.1) with moderate bases (8) is the only combination that helps. Slow rates (0.001, 0.01) hurt because the bases are stale.

**Recommendation:** TREND sem_b8_r1em01 looks marginally better than off, but 7/9 enabled configs are worse. The semantic tier is borderline — keep it off unless confirmed at 8+ seeds.

## Sweep 5: Merge Similarity Threshold (full stack + n3_only sleep)

| Condition | Mean bpb | Steps | vs default (0.85) |
|-----------|----------|-------|--------------------|
| merge_075 | 2.5284 | 403 | +0.017 |
| merge_080 | 2.5208 | 410 | +0.009 |
| **merge_085** | **2.5119** | 409 | **default** |
| merge_090 | 2.5199 | 409 | +0.008 |
| **merge_095** | **2.5088** | 424 | **-0.003 (best)** |

**Finding:** merge_095 is marginally best (2.509 vs 2.512 default), but the winner is at the sweep EDGE. Higher threshold = less merging = more slots retained = less compression = more steps (424 vs 409). The model prefers to keep slots rather than merge them.

**Recommendation:** WARNING winner at edge. Extend range to [0.95, 0.97, 0.99] before locking. Or interpret as "don't merge much at this budget."

## Summary of Decisions

| Constant | Default | Best | Delta | Action |
|----------|---------|------|-------|--------|
| crit_target_coupling | 0.88 | **0.92** | -0.017 | Confirm on 8+ seeds |
| outer_model_dim | 64 | 32 | -0.010 | Trend only, keep 64 |
| outer_max_slots | 64 | **32** | -0.033 | Confirm on 8+ seeds |
| semantic_tier | off | b8/r0.1 | -0.008 | Trend only, keep off |
| merge_threshold | 0.85 | 0.95 | -0.003 | Edge warning, extend range |

Two strong candidates for change: **crit_target_coupling → 0.92** and **outer_max_slots → 32**.
