# Sleep Cycle Design: Structured Memory Consolidation

## Motivation

Memory hurts at short training budgets (95 steps) but helps at longer ones (400+ steps). The architecture lacks offline consolidation — memory competes with gradient learning for every step. The metabolic gate hurts during training because it steals throughput.

Sleep addresses both: consolidation gets a dedicated phase without competing with gradients, and the gate trains during REM on internally-generated counterfactuals (dreams) rather than stealing wake-time steps.

## Core Thesis

Given a fixed compute budget, spending some of it on structured consolidation produces a better model than spending all of it on gradient updates.

## Architecture

### Module

`src/chaoscontrol/sleep.py` — a `SleepCycle` class. One entry point: `run()`. Called by the training loop when fatigue crosses threshold. Sleep counts against the training budget (wall time). If sleep doesn't overcome the lost training steps, it fails on its own terms.

### Fatigue Score

A dynamic system, not a weighted sum. Fatigue has state — it accumulates with inertia and decays during sleep.

Three input signals:
- **Surprise (low = sleepy):** When recent losses track the running average, the model isn't being challenged. The world is predictable enough to go offline.
- **Improvement rate (low = sleepy):** When loss isn't decreasing meaningfully per step, more training has diminishing returns.
- **Memory pressure (high = sleepy):** Slot count relative to max_slots, mean survival trending down, retrieval weight entropy flattening.

Dynamics:
```
fatigue += accumulation_rate * pressure_signal
fatigue -= decay_rate * (sleep_quality if sleeping else rest_rate)
fatigue = clamp(fatigue, 0, 1)
```

High surprise suppresses fatigue — the model stays alert when the environment demands attention. Fatigue accumulates across wake periods (debt). Sleep quality feeds back into recovery: a good consolidation reduces fatigue significantly; a poor one leaves residual debt.

Fatigue modulates sleep duration: light fatigue triggers a short sleep (one cycle). Heavy fatigue triggers longer sleep (multiple cycles, extended REM). Accumulated debt that isn't resolved triggers a longer consolidation — the day off.

Default wake:sleep ratio is 2:1 (256 wake steps : 128 sleep steps) but adapts based on fatigue.

**Ablation note:** The primary experiment (Experiment 11) uses a fixed sleep trigger interval and fixed sleep budget to isolate stage contributions. Adaptive fatigue is tested as a separate follow-up condition, not confounded with stage ablation.

### Wake-Time Caching

During wake, the training loop caches data needed by the sleep cycle:

- **High-signal moments:** Batches where surprise was unusually high or low (both directions), stored with their associated slot cues, bucket IDs, hidden states, and the real token continuations that followed.
- **Bucket distribution:** Running counts of Wernicke bucket assignments, used to weight dream scene frequency.
- **Recent hidden states:** A rolling buffer of hidden states from the last N wake steps, used by N2 for utility scoring.

This cache is the raw material for sleep. Without it, the sleep cycle has nothing to consolidate against.

### Sleep Stages

Four stages, executed in order within each sleep cycle. Ablation conditions add stages incrementally.

**N1 — Transition.** Mode switch. Freeze new slot creation. Snapshot recent unstable traces into a transition buffer. Prevents N2 from scoring slots that were written in the last few steps and haven't stabilized. This is not "learning rate to zero" — it's an explicit state change that says "we are no longer admitting new canonical episodes."

**N2 — Tag.** Re-evaluate every slot's utility via leave-one-slot-out delta loss. For each slot, temporarily remove it from the memory system, run the cached high-signal wake batches through the model, and measure the loss delta. Slots that hurt performance when removed are useful (high utility). Slots that make no difference are dead weight. This is more expensive than simple cue-matching but actually measures what the paper claims: slot contribution to wake-time prediction quality. Rescoring happens before compression, so N3 has real utility information.

**N3 — Rewrite.** Structural consolidation. Merges are *provisional* — proposed but not committed until REM validates (or committed immediately if running without REM in the `n3_only` ablation). Merge redundant slots (cosine similarity above threshold). Prune low-utility slots below a floor. Denoise surviving slots. Recompute semantic bases from the cleaned slot set. Produce latent traces from absorbed slots. Generate compression consequence candidates for Wernicke feedback.

**REM — Dream.** The model dreams. Backbone frozen. Dreams are not written as canonical memories.

#### Dream Generation

Dreams are directed by the day's significant experiences — high-signal moments, both good (positive surprise) and bad (negative surprise).

1. **Scene selection:** During wake, tag high-signal moments with their associated slot cues, bucket IDs, and hidden states. During REM, group these tagged moments by Wernicke bucket. Each group is a scene. Scene frequency reflects the wake-time bucket distribution — if bucket 3 handled 40% of tokens during the day, it gets proportionally more dream time. Dreams stress-test a world with realistic composition, not a uniform one.

2. **Background construction:** For each scene, reconstruct the context: load the relevant slots as memory state, set the bucket routing to match the original conditions.

3. **Dream execution:** From that context, decode seed tokens from slot centroids via the model's own decoder. Autoregressively generate forward using the model's predictions. The model is both dreamer and dream environment. Dream length scales with fatigue — more problems to process means longer REM. Dreams use the full forward path including Wernicke routing and memory retrieval (requires a `dream_step()` method that includes all tiers, unlike the reduced `model.step()` used for MCTS rollouts).

4. **Dream scoring:** Score against cached real continuations from the wake-time high-signal moments, not internal coherence. For each scene, the dream asks: "given this consolidated memory state, can I predict what actually happened?" This is teacher-forced cross-entropy on real targets, not self-referential plausibility. This anchors dream quality to wake-time bpb and prevents the system from learning self-consistent fantasies.

5. **Merge validation (rem_validate):** For each provisional N3 merge, run dreams seeded from the merged slot's context. If teacher-forced score worsens relative to pre-merge baseline: reject the merge and reactivate the latent trace (restore the absorbed slots with degradation noise). If score holds or improves: commit the merge. This gives REM a same-cycle repair channel without "changing the past" — N3 proposes, REM approves.

6. **Latent reactivation (rem_reactivate):** When a dream scene scores poorly (teacher-forced CE significantly above wake-time CE for the same cached moment), attempt `try_reactivate()` on the associated Wernicke bucket. The latent trace system already exists (`memory.py:501`) — compressed slots leave centroid traces that can be restored with degradation noise on high surprise. During wake, reactivation fires on surprise threshold. During REM, it fires on dream-diagnosed information loss. If reactivation improves the re-scored CE, the trace stays active. If not, it gets re-compressed. This is distinct from merge validation: validation prevents bad merges from committing; reactivation recovers information lost in *prior* compressions (from wake or previous sleep cycles).

7. **Gate policy (rem_cfr):** Gate decisions during dreams populate counterfactual values via perturbed forward passes. This is where CFR trains — on cheap internally-generated sequences with real-target scoring. The regret table updates bias future gate decisions toward actions that would have performed better during dreams.

8. **Updates from dreams:**
   - Slot survival: slots that seed dreams with good teacher-forced scores get boosted. Slots that seed poor scores get penalized.
   - Compression penalties: rejected merges and failed reactivations increase protection for similar slots in future N3 passes.

9. **What does NOT update:**
   - Model weights (backbone frozen — body paralysis)
   - Canonical episodic memory (dreams are not real experiences)
   - Semantic bases (only update from real slots during N3)

## Experiment 11: Sleep Cycle Ablation

### Structure

```
experiments/11_sleep_cycle/
  run_sleep_ablation.py
  configs/
  results/
  REPORT.md
```

### Conditions

All use the full stack (SSM + episodic memory + Wernicke MoE) at 600s budget, 5 seeds. Fixed sleep trigger interval (every 256 wake steps) and fixed sleep budget (128 steps) for clean ablation. N2 and REM get fixed sub-budgets (N2: 64 ops, REM: 64 ops) so stage attribution is not confounded by budget starvation. Realized ops per stage are logged alongside bpb. No adaptive fatigue in the primary experiment.

Each REM mechanism (merge validation, latent reactivation, gate policy) is independently toggleable so we can isolate their contributions without conflation.

| Condition | Stages | What it isolates |
|-----------|--------|-----------------|
| `no_sleep` | None | Baseline |
| `n3_only` | Compression pass | Deliberate compression vs overflow-triggered |
| `n2_n3` | Score + compress | Utility-based rescoring value |
| `n2_n3_rem_validate` | + merge validation | Dream-guided consolidation quality |
| `n2_n3_rem_cfr` | + gate policy | Dream-time gate/CFR training |
| `n2_n3_rem_reactivate` | + latent recovery | Dream-diagnosed information loss recovery |
| `n2_n3_rem_all` | + all three REM mechanisms | Combined REM value |
| `full_cycle` | N1 + N2 + N3 + all REM | N1 transition mechanism |

8 conditions x 5 seeds = 40 training runs. ~2.2 hours on 3x A40.

### Follow-up Conditions (separate from primary ablation)

| Condition | What it tests |
|-----------|---------------|
| `full_cycle_adaptive` | Fatigue-triggered sleep vs fixed interval |
| `full_cycle_ratio_3_1` | 3:1 wake:sleep ratio vs 2:1 |
| `full_cycle_ratio_4_1` | 4:1 wake:sleep ratio vs 2:1 |

### Statistical Analysis Plan

**Pre-specified contrasts (ordered by priority):**

1. Primary: `full_cycle` vs `no_sleep` — does the complete sleep cycle help?
2. Secondary: `n2_n3` vs `n3_only` — does utility scoring improve compression?
3. Secondary: `n2_n3_rem_all` vs `n2_n3` — do dreams add value beyond consolidation?
4. Exploratory: `n2_n3_rem_validate` vs `n2_n3` — is the lift from merge validation?
5. Exploratory: `n2_n3_rem_cfr` vs `n2_n3` — is the lift from gate policy?
6. Exploratory: `n2_n3_rem_reactivate` vs `n2_n3` — is the lift from latent recovery?
7. Exploratory: `full_cycle` vs `n2_n3_rem_all` — does N1 transition matter?

**For each contrast, report:**
- Mean delta bpb with 95% bootstrap CI (10,000 resamples)
- Wilcoxon signed-rank p-value (paired by seed)
- Holm-Bonferroni correction across all 7 contrasts
- Cohen's d effect size

With 5 seeds, Wilcoxon has 32 possible rank assignments — minimum achievable p is 0.0312. Sufficient for primary contrast significance at alpha=0.05 after Holm correction (primary contrast corrected threshold = 0.05/7 = 0.007, which requires 7 seeds to achieve). For the primary contrast alone at alpha=0.05 uncorrected, 5 seeds suffice. For stronger claims after full correction, increase to 7 seeds (minimum p = 0.0078).

### Decision Criteria

If `full_cycle` or `n2_n3_rem_all` significantly outperforms `no_sleep` (corrected p < 0.05) at 600s despite spending budget on consolidation instead of gradient steps: the sleep cycle works and the architecture is ready for H100 scale-up.

If no condition beats `no_sleep` after correction: consolidation costs more than it's worth at this scale and budget, and the architecture needs rethinking before scaling.

If individual REM mechanisms show isolated significant effects (e.g., reactivation helps but CFR doesn't), that guides which mechanisms to invest in at scale.
