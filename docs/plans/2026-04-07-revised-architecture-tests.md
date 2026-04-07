# Revised Architecture Tests — Design Doc

**Date:** 2026-04-07
**Context:** First 73-config run completed. SSM beats transformer (2.24 vs 2.48 bpb). Bio-inspired components inconclusive due to throughput confound (full A-mode matrix_exp limits training to 20-50 steps). Two research reports inform a redesigned architecture: game-theoretic exploration (micro-MCTS, CFR, Thompson sampling) and compressed-artifact memory (PQ + residual-sparse as consolidation mechanism).

## Goal

Research prototype (relaxed constraints) validating four new mechanisms and their interactions via a layered test matrix.

## Lesson from Round 1

Every component that adds per-step compute looks worse because it trains less. The fix: diag A-mode as primary (300+ steps in 300s), full A-mode as secondary validation at 1800s budget (~300 steps).

---

## New Components

### 1. Micro-MCTS Gate

Replace fork-and-pick with forward-only tree search triggered by surprise.

**Mechanism:**
- Surprise monitor fires when loss_spike / running_avg > threshold
- Clone current hidden state as root
- Run N rollouts of depth H tokens using the SSM as a world model:
  - At each node, select child via UCB (balance exploit/explore)
  - Step the SSM forward, accumulate value estimate
  - Terminal value = proxy score (memory consistency, confidence, lookahead loss)
- Back up returns to update visit counts and mean values per node
- Commit one token (proportional to root visit counts), advance
- Store rollout statistics (visit counts, values) in episodic memory keyed by Wernicke bucket ID

**Not triggered:** single cheap forward pass, zero overhead on easy tokens.

**Parameters to sweep:**
- Rollout count N: 4, 8, 16
- Horizon H: 8, 16
- UCB exploration constant c: 0.5, 1.0, 1.41

**Key difference from old fork:** the SSM IS the world model. Rollouts are cheap forward passes through the recurrence, not full model forward passes. Branching happens in latent space, not at the embedding level.

### 2. Memory Transformation with Latent Persistence

Replace fixed-size tensor slot storage with demand-driven consolidation where PQ compression IS the biological consolidation mechanism.

**Three memory states (demand-driven, not time-driven):**

| State | Trigger | Representation | Retrieval |
|-------|---------|---------------|-----------|
| **Full-fidelity** | Default (VRAM has space) | Raw tensor slot | Normal cue-dependent |
| **Compressed** | VRAM at capacity, lowest survival scores | Peaks progressively pruned, residual mass folded into bucket centroid | Reconstructed from centroid + remaining peaks |
| **Latent** | Fully compressed (all peaks pruned) | Bucket membership only | Silent under normal cues. Reactivatable under high-surprise + matching bucket |

**Consolidation pressure is capacity-driven:**
- While VRAM has space: all slots stay full-fidelity. No compression, no gist drift.
- At capacity: compress oldest/lowest-survival slots. Prune smallest peaks first, fold mass into bucket centroid. This IS the episodic-to-semantic transformation.
- Fully pruned slots become latent traces. They still exist (bucket membership recorded), but don't activate during normal retrieval. A high-surprise cue matching the bucket can reactivate by reconstructing from centroid + any residual structure.

**Wernicke bucket ID = PQ codebook index:**
- Each slot's bucket assignment doubles as its PQ cell
- The bucket centroid IS the semantic base for that memory type
- Compression = moving from (centroid + full peaks) toward (centroid only)
- Retrieval = rehydrate centroid + whatever peaks remain

**Compression-consequence feedback:**
- Bad merge (high quality_delta) = premature gist transformation
- Feeds back to Wernicke to refine bucket boundaries
- Prevents over-aggressive consolidation in buckets where episodic detail matters

### 3. CFR-Style Regret Tracking

Per Wernicke bucket, maintain a regret table over candidate actions. Forward-only bookkeeping — no weight updates at inference time.

**Mechanism:**
- After committing a token, estimate counterfactual value of alternatives via short lookahead (1-2 tokens forward for each unchosen candidate)
- Compute regret = counterfactual_value - committed_value for each alternative
- Accumulate positive regret per (bucket_id, action_index) in episodic memory
- Use regret-matching to bias future candidate selection: action_probability proportional to positive cumulative regret
- Negative-regret pruning with occasional full exploration (Pluribus-style)

**Information set key:** (Wernicke bucket_id, coarse hidden-state signature). This groups similar decision points so regret statistics transfer across contexts.

**Why this matters:** Without regret tracking, the gate treats every surprise as novel. With it, the gate remembers "in this type of context, candidate 3 tends to be underexplored and high-value" — learned exploration policy per semantic type.

### 4. Test-Time Memory Warmup

During eval forward pass, enable episodic writes. Later positions benefit from earlier observations.

**Mechanism:**
- consolidation_step runs during eval (currently only runs during training)
- Surprise-driven writes: high-surprise tokens during eval get written to episodic slots
- Subsequent positions can retrieve from these fresh slots
- The model "learns" the test distribution as it reads, without any weight updates

**This is legal under competition rules** (tokens already evaluated can inform future predictions).

---

## Test Matrix

### Budget
- Diag A-mode: 300s per config
- Full A-mode (Layer 5 only): 1800s per config

### Layer 1: Gate Modes (6 configs)
Isolate which exploration strategy works, no memory or Wernicke overhead.

| Config | A-mode | Memory | Wernicke | Gate | Warmup |
|--------|--------|--------|----------|------|--------|
| baseline_ssm | diag | none | none | none | cold |
| baseline_tfm | diag | none | none | none | cold |
| gate_fork_k4 | diag | none | none | fork_k4 | cold |
| gate_mc_k4 | diag | none | none | monte_carlo_k4 | cold |
| gate_mcts_k4 | diag | none | none | micro_mcts_k4_h8 | cold |
| gate_mcts_k8 | diag | none | none | micro_mcts_k8_h8 | cold |

**Decision gate:** best gate mode carries forward. If no gate beats baseline, gate is dropped.

### Layer 2: +Memory (6 configs)
Best gate from L1 × memory variants × warmup.

| Config | Memory | Warmup |
|--------|--------|--------|
| mem_none_cold | none | cold |
| mem_epi_cold | episodic | cold |
| mem_epi_warm | episodic | warmup |
| mem_both_cold | episodic+semantic | cold |
| mem_both_warm | episodic+semantic | warmup |
| mem_both_warm_latent | episodic+semantic (with latent persistence) | warmup |

**Decision gate:** best memory config carries forward.

### Layer 3: +Wernicke + CFR (4 configs)
Best gate + best memory × Wernicke × regret tracking.

| Config | Wernicke | CFR |
|--------|----------|-----|
| no_wernicke_no_cfr | none | none |
| wernicke_no_cfr | moe_16 | none |
| wernicke_cfr | moe_16 | regret tracking |
| wernicke_cfr_latent | moe_16 | regret tracking + latent reactivation |

**Decision gate:** best combo carries forward.

### Layer 4: Scaling (4 configs)
Best full stack at multiple model sizes + transformer baseline.

| Config | dim |
|--------|-----|
| full_stack_128 | 128 |
| full_stack_256 | 256 |
| full_stack_384 | 384 |
| transformer_384 | 384 |

### Layer 5: Full A-mode Validation (4 configs)
Layer 4 winners rerun on full A-mode at 1800s budget.

| Config | A-mode | Budget |
|--------|--------|--------|
| full_stack_128_full | full | 1800s |
| full_stack_256_full | full | 1800s |
| full_stack_384_full | full | 1800s |
| transformer_384 | (baseline, reuse L4 result) | — |

### Layer 6: Inference-Time Adaptation Depth (4 configs × 3 seeds)
Full stack winner. Tests how many memory tiers should participate during eval.
The SSM recurrence always adapts (it IS working memory). The question is
whether deeper tiers improve inference when allowed to run during the forward pass.

| Config | What adapts at eval time |
|--------|------------------------|
| wm_only | Just the recurrence (standard SSM inference) |
| wm_plus_episodic | Recurrence + surprise-gated episodic writes |
| wm_plus_all | Recurrence + episodic + semantic consolidation + latent reactivation |
| wm_plus_all_seeded | Same, but LTM starts from training (not cold start) |

This layer answers the artifact strategy question: is it worth shipping compressed
LTM seeds in the artifact (spending budget on memory instead of SSM weights), knowing
that the model can reconstitute memories during the eval forward pass?

---

## Total: ~28 configs, ~73 runs

- Layer 1-3: 17 configs × 3 seeds = 51 runs × 300s = 255 min
- Layer 3.5: 3 configs × 1 seed = 3 runs × 300s = 15 min
- Layer 4: 4 configs × 1 seed = 4 runs × 300s = 20 min
- Layer 5: 3 configs × 1 seed = 3 runs × 1800s = 90 min
- Layer 6: 4 configs × 3 seeds = 12 runs × 300s = 60 min
- **Total: ~7.3 hours GPU time**

- Layer 1-3: 16 configs × 300s = 80 min
- Layer 4: 4 configs × 300s = 20 min
- Layer 5: 3 configs × 1800s = 90 min
- **Total: ~3 hours GPU time**

Layers are sequential (each layer's best carries forward), but configs within a layer are independent and can run in parallel.

## Implementation Order

1. Micro-MCTS gate (new module, biggest risk)
2. Memory transformation + latent persistence (refactor existing MultiSlotOuterModel)
3. CFR regret tracking (new bookkeeping layer on top of memory)
4. Test-time warmup (small change to eval path)
5. Config generation + matrix runner
6. Run on H100, harvest, analyze

## Success Criteria

- At least one gate variant beats baseline_ssm on diag A-mode
- Memory warmup shows progressive improvement over eval sequence length
- Latent persistence reactivates at least some "forgotten" memories under surprise
- CFR regret tracking produces non-trivial bucket-level strategy differences
- Full A-mode at 1800s budget produces comparable step counts to diag at 300s
