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

### Sleep Stages

Four stages, executed in order within each sleep cycle. Ablation conditions add stages incrementally.

**N1 — Transition.** Mode switch. Freeze new slot creation. Snapshot recent unstable traces into a transition buffer. Prevents N2 from scoring slots that were written in the last few steps and haven't stabilized. This is not "learning rate to zero" — it's an explicit state change that says "we are no longer admitting new canonical episodes."

**N2 — Tag.** Re-evaluate every slot's survival. Run each slot's centroid as a retrieval cue against recent training hidden states (cached from the last wake period). Slots that don't match anything useful get survival reduced. Slots that were heavily cued during high-signal moments get survival boosted. This rescoring happens before compression, so N3 has better information about what to keep.

**N3 — Rewrite.** Structural consolidation. Merge redundant slots (cosine similarity above threshold). Prune low-survival slots below a floor. Denoise surviving slots. Recompute semantic bases from the cleaned slot set. Produce latent traces from absorbed slots. Generate compression consequence candidates for Wernicke feedback. This is the existing `_compress()` logic but run deliberately and thoroughly, not just when max_slots overflows.

**REM — Dream.** The model dreams. Backbone frozen. Dreams are not written as canonical memories.

#### Dream Generation

Dreams are directed by the day's significant experiences — high-signal moments, both good (positive surprise) and bad (negative surprise).

1. **Scene selection:** During wake, tag high-signal moments with their associated slot cues, bucket IDs, and hidden states. During REM, group these tagged moments by Wernicke bucket. Each group is a scene. Scene frequency should reflect the wake-time bucket distribution — if bucket 3 handled 40% of tokens during the day, it gets proportionally more dream time. Dreams stress-test a world with realistic composition, not a uniform one.

2. **Background construction:** For each scene, reconstruct the context: load the relevant slots as memory state, set the bucket routing to match the original conditions.

3. **Dream execution:** From that context, decode seed tokens from slot centroids via the model's own decoder. Autoregressively generate forward using the model's predictions. The model is both dreamer and dream environment. Dream length scales with fatigue — more problems to process means longer REM.

4. **Dream scoring:** Track whether the consolidated memory system composes coherently under the conditions that mattered today. Good surprises get rehearsed. Bad surprises get worked through.

5. **Updates from dreams:**
   - Regret table: gate decisions during dreams populate counterfactual values. This is where CFR trains.
   - Compression penalties: incoherent dreams flag bad merges. Penalties increase protection for similar slots in future N3 passes. Penalties do NOT undo merges — dreams can't change the past.
   - Slot survival: slots that seed coherent dreams get boosted. Slots that seed nonsense get penalized.

6. **What does NOT update:**
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

All use the full stack (SSM + episodic memory + Wernicke MoE) at 600s budget, 3 seeds.

| Condition | Stages | What it tests |
|-----------|--------|---------------|
| `no_sleep` | None | Baseline |
| `n3_only` | Compression pass | Does deliberate compression beat overflow-triggered? |
| `n2_n3` | Score + compress | Does rescoring before compression improve what survives? |
| `n2_n3_rem` | Score + compress + dreams | Do dreams improve gate policy and compression quality? |
| `full_cycle` | N1 + N2 + N3 + REM | Does the transition mode-switch matter? |

5 conditions x 3 seeds = 15 training runs. ~50 minutes on 3x A40.

### Claim Structure

Each row should beat the row above it, or we learn which stages don't pull their weight. The ablation table is the paper's core evidence.

### Decision Criteria

If `full_cycle` or `n2_n3_rem` significantly outperforms `no_sleep` at 600s despite spending budget on consolidation instead of gradient steps: the sleep cycle works and the architecture is ready for H100 scale-up.

If no condition beats `no_sleep`: consolidation costs more than it's worth at this scale and budget, and the architecture needs rethinking before scaling.
