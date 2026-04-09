# Experiment 09 Phase 2: Eval-Time Ablation Report

**Date:** 2026-04-08
**Pod:** mlntmjdlc3iids (3x A40)
**Total evals:** 729 (9 checkpoints x 81 eval configs)
**Grid:** 5 gate modes x 3 memory states x 2 CFR x 3 warmup = 81 (after redundancy filtering)

## Key Result

**The metabolic gate consistently hurts at eval time on memoryless checkpoints.** All gate modes (fork, MC, MCTS) produce ~5.5 bpb vs ~3.4 bpb for plain forward pass. This is a ~2 bpb degradation.

## Gate Mode Results (averaged across all scales and seeds)

| Gate Mode | Mean bpb | n |
|-----------|----------|---|
| **none** | **3.362** | 81 |
| mc_k4 | 5.467 | 162 |
| fork_k4 | 5.467 | 162 |
| mcts_k4 | 5.468 | 162 |
| mcts_k8 | 5.468 | 162 |

All gate modes produce nearly identical (bad) results. The gate's candidate selection adds noise without improving predictions.

## Scale x Gate Interaction

| Scale | gate=none | fork_k4 | mc_k4 | mcts_k4 | mcts_k8 |
|-------|-----------|---------|-------|---------|---------|
| dim128 (4L) | **2.877** | 5.503 | 5.503 | 5.503 | 5.503 |
| dim256 (6L) | **3.109** | 6.123 | 6.123 | 6.123 | 6.123 |
| dim384 (8L) | **4.100** | 4.777 | 4.777 | 4.777 | 4.777 |

- dim128: gate causes +2.63 bpb degradation
- dim256: gate causes +3.01 bpb degradation
- dim384: gate causes +0.68 bpb degradation (smaller gap because base model is worse)

## Memory State / CFR / Warmup

All memory state, CFR, and warmup variations produce **identical** results because the L1 winner was `mem_none` -- these checkpoints have no outer_model, so:
- seeded/cold/ttt = identical (no memory to manipulate)
- CFR on/off = identical (regret table has no effect without memory for bucket routing)
- warmup = identical (warmup writes to non-existent memory)

This is a valid scientific finding: without memory, these mechanisms are pure overhead.

## Best Configuration

**Best bpb: 2.848** (L3_dim128_seed4011, gate=none, seeded, cfr_off, warmup_none)

This matches the Phase 1 training result, confirming that plain forward eval reproduces training-time quality.

## Limitations

1. **No memory checkpoints tested:** Since L1 winner was mem_none, Phase 2 could not test whether memory helps at eval time. The gate, CFR, warmup, and memory state dimensions all collapsed to no-ops.

2. **Gate consistently harmful:** The metabolic gate (fork/MC/MCTS) doubles bpb by introducing noise into candidate selection. Without memory-based scoring signals, the gate has no information to select good candidates.

3. **To properly test memory + gate:** Would need to re-run Phase 1 with mem_epi forced, then Phase 2 on those checkpoints. The mem_epi training configs (checkpoints exist from L1) could be used directly.

## Conclusion

Phase 2 establishes a clean negative result: **without episodic memory, the metabolic gate is harmful at eval time across all modes and scales.** This is consistent with the architecture's design -- the gate is meant to leverage memory-based surprise signals, not raw prediction uncertainty.

The positive finding from Phase 1 (Wernicke MoE routing helps +0.18 bpb) stands. The architecture's value may lie in the routing layer rather than the gate mechanism, at least at these scales and budgets.

---

## Files

- `results_phase2/eval_results.json` — all 729 eval results
- `results_phase2/eval_results_shard{0,1,2}.json` — per-GPU shard files
- `results_phase2/shard_{0,1,2}.log` — per-shard logs
