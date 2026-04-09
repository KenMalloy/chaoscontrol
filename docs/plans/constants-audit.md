# Constants Audit

Every baked-in default in `ChaosControlConfig`, categorized by justification level. This determines what needs sweeping before the paper and what's defensible as-is.

## Categories

- **Validated**: swept or empirically tested in our experiments
- **Principled**: chosen from theory/convention, defensible without sweep
- **Inherited**: carried forward from parameter-golf, never validated in this architecture
- **Arbitrary**: no strong reason for this value, should be swept

---

## Model Architecture

| Constant | Value | Category | Justification | Action |
|----------|-------|----------|---------------|--------|
| `vocab_size` | 256 | **Principled** | Byte-level model, 256 is the only option | None |
| `model_dim` | 128 | **Validated** | Phase 1 L3 sweep: 128 > 256 > 384 at 150s | Retest at 600s (trunk scaling experiment) |
| `num_layers` | 4 | **Validated** | Swept alongside dim in Phase 1 L3 | Retest at 600s |
| `ff_mult` | 2 | **Arbitrary** | Standard MLP ratio, never swept | Low priority sweep |
| `a_mode` | "diag" | **Validated** | Phase 1 default, consistent with SSM literature | None |
| `a_full_rank` | 8 | **Inherited** | Only relevant for a_mode="full", which we don't use | None (dead code path) |
| `a_full_gamma` | 0.05 | **Inherited** | Same | None |

## Criticality

| Constant | Value | Category | Justification | Action |
|----------|-------|----------|---------------|--------|
| `crit_target_coupling` | 0.88 | **Inherited** | From parameter-golf, `log(0.88) ≈ -0.13` "slightly subcritical". Never swept. | **Sweep: [0.80, 0.85, 0.88, 0.92, 0.96]** |
| `crit_reg_alpha` | 0.01 | **Inherited** | Regularization strength, never swept | Sweep alongside coupling target |
| `crit_reg_beta` | 0.001 | **Inherited** | Secondary reg term, never swept | Low priority |

## Training

| Constant | Value | Category | Justification | Action |
|----------|-------|----------|---------------|--------|
| `base_lr` | 2e-3 | **Principled** | Standard for small models, AdamW convention | Could sweep [1e-3, 2e-3, 5e-3] but low priority |
| `weight_decay` | 1e-2 | **Principled** | Standard AdamW default | None |
| `grad_clip_norm` | 1.0 | **Principled** | Standard practice | None |
| `batch_size` | 64 | **Arbitrary** | Memory-limited guess, never swept | Low priority |
| `seq_len` | 256 | **Arbitrary** | Moderate context, never swept | Could test 128, 512 |
| `stride` | 128 | **Principled** | Half seq_len overlap, standard practice | None |

## Wernicke Routing

| Constant | Value | Category | Justification | Action |
|----------|-------|----------|---------------|--------|
| `wernicke_k_max` | 16 | **Arbitrary** | Chosen once, never swept | **Sweep: [16, 32, 64] (wired, param-controlled)** |
| `wernicke_window` | 8 | **Arbitrary** | Conv1d window for composition, never swept | Low priority |
| `wernicke_router` | "moe" | **Validated** | Phase 1 L2: MoE > VQ > none | None |
| `wernicke_balance_weight` | 0.01 | **Arbitrary** | Encourages uniform bucket usage, never swept | Low priority |

## Memory

| Constant | Value | Category | Justification | Action |
|----------|-------|----------|---------------|--------|
| `outer_model_dim` | 64 | **Arbitrary** | Slot embedding dimension, never swept | **Sweep: [32, 64, 128] (exp 13)** |
| `outer_max_slots` | 64 | **Arbitrary** | Max episodic slots before compression, never swept | **Sweep: [32, 64, 128] (exp 13)** |
| `outer_compress_ratio` | 2 | **Arbitrary** | Merge N into N/2, never swept | Low priority |
| `consolidation_ema_decay` | 0.99 | **Principled** | Standard EMA, slow-moving average | None |
| `consolidation_window` | 8 | **Arbitrary** | Steps after spike before flush, never swept | Low priority |

## Sleep Cycle

| Constant | Value | Category | Justification | Action |
|----------|-------|----------|---------------|--------|
| `sleep_interval` | 256 | **Arbitrary** | Wake steps between sleep cycles, never swept | **Test fixed vs adaptive (fatigue)** |
| `sleep_budget` | 128 | **Arbitrary** | Max ops per sleep cycle | Tied to interval (2:1 ratio) |
| `sleep_n2_budget` | 64 | **Arbitrary** | Sub-budget for N2 scoring | Low priority |
| `sleep_rem_budget` | 64 | **Arbitrary** | Sub-budget for REM | Low priority |
| `sleep_n2_batches` | 8 | **Arbitrary** | Cached batches for leave-one-out scoring | Low priority |
| `sleep_rem_dreams` | 4 | **Arbitrary** | Dreams per REM cycle | Low priority |
| `sleep_rem_length` | 128 | **Arbitrary** | Tokens per dream | Low priority |
| `sleep_merge_sim_threshold` | 0.85 | **Arbitrary** | Cosine sim for merge proposal, never swept | **Sweep: [0.75, 0.80, 0.85, 0.90, 0.95]** |
| `sleep_survival_floor` | 0.1 | **Arbitrary** | Below this, slots get pruned | Low priority |

## Polyphasic Sleep

| Constant | Value | Category | Justification | Action |
|----------|-------|----------|---------------|--------|
| `polyphasic_n_partitions` | 4 | **Principled** | Matches available GPUs for testing | Will scale to 8 on H100 |
| `polyphasic_k_awake` | 3 | **Principled** | N-1 default (one sleeping at a time) | **Sweep in exp 12: K=2, 3** |
| `polyphasic_swap_interval` | 256 | **Arbitrary** | Training steps between partition rotation | **Sweep: [64, 256, 1024]** |

## Metabolic Gate (inference only)

| Constant | Value | Category | Justification | Action |
|----------|-------|----------|---------------|--------|
| `metabolic_k` | 4 | **Arbitrary** | Candidate count for fork/MCTS | Low priority (inference) |
| `metabolic_threshold` | 0.1 | **Arbitrary** | Surprise ratio to trigger gate | Low priority (inference) |
| `metabolic_noise_std` | 0.01 | **Arbitrary** | Perturbation for candidates | Low priority (inference) |
| `mcts_horizon` | 8 | **Arbitrary** | Lookahead depth | Low priority (inference) |
| `mcts_ucb_c` | 1.41 | **Principled** | √2, standard UCB constant | None |

---

## Priority Sweeps (before paper)

These are the constants that could materially affect results and have never been validated:

### High Priority (affects core claims)

1. **`crit_target_coupling`** [0.80, 0.85, 0.88, 0.92, 0.96] — The SSM's temporal dynamics. Affects everything.
2. **`wernicke_k_max`** [16, 32, 64] — Already wired and param-controlled. Affects Wernicke and sleep.
3. **`sleep_merge_sim_threshold`** [0.75, 0.80, 0.85, 0.90, 0.95] — Controls how many slot pairs pass the merge gate. Candidates above this threshold are then ranked by `sim * affinity` — so the threshold controls *volume* of candidates, while affinity controls *priority*. Covered by Experiment 13 (provisional on exp 11 sleep payload and k_max lock).

### Medium Priority (refinement)

4. **`sleep_interval`** — Fixed vs adaptive fatigue (already implemented).
5. **`polyphasic_swap_interval`** [64, 256, 1024] — How often partitions rotate.
6. **`model_dim` / `num_layers`** at 600s — Phase 1 was at 150s, may differ.

### Low Priority (standard choices, unlikely to matter)

7. `base_lr`, `ff_mult`, `batch_size`, `seq_len` — Standard hyperparams.
8. Sleep sub-budgets — Internal allocation within a fixed total.
9. Metabolic gate params — Inference-only, not affecting training results.

---

## Cost Estimate

| Sweep | Conditions | Seeds | Runs | Est. time (4×A40) |
|-------|-----------|-------|------|--------------------|
| crit_target_coupling (5 values) | 5 | 7 | 35 | ~90 min |
| k_max (3 values, already wired) | 6 | 7 | 42 | ~110 min |
| merge_sim_threshold (5 values) | 5 | 7 | 35 | ~90 min |
| **Total high-priority** | | | **112** | **~5 hours** |
