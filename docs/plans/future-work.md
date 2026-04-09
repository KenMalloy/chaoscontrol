# Future Work

Ideas that emerged during the sleep cycle and polyphasic design sessions (2026-04-08). Ordered roughly by scientific priority, not implementation difficulty.

## Already Implemented (not future work — listed for reference)

- **Polyphasic partitioned sleep** — K-of-N scheduling, 3 topologies, partition-scoped sleep. Code in `partition.py`, integrated into training loop. Experiment 12 runner ready.
- **k_max sweep** — param-controlled via `wernicke_expert_dim` bottleneck. Wired in `experiments/baselines/run_mamba2_baseline.py`.
- **Adaptive sleep triggers (fatigue system)** — implemented in `fatigue.py`, disabled in experiments. Enable after experiment 11 confirms which stages help.
- **Adaptive sleep triggers (fatigue system)** — implemented in `fatigue.py`, disabled in experiments. Needs a follow-up experiment after exp 11 confirms which stages help: does fatigue-based triggering beat fixed interval? High priority — the model should sleep when it's tired, not on a schedule.
- **Mamba-2 baseline** — `Mamba2LM` in `baselines.py`, wired in runner. Waiting for H100-ready architecture before running.

---

## 1. Trunk Scaling (Cerebellum Hypothesis)

**Observation:** The elephant cerebellum has 97.5% of all neurons — a 3,300:1 ratio of prediction neurons to output neurons. Our SSM trunk (the prediction engine) is dim=128, 4 layers. The semantic engine (memory, Wernicke, sleep) gets substantial architectural investment relative to the trunk.

**Hypothesis:** Disproportionately scaling the SSM trunk (more layers, wider state) while keeping the semantic engine at current size will improve bpb more than scaling the semantic engine.

**Test A (scaling):** Trunk-scaling sweep at fixed semantic capacity:
- dim=128/4L (current) vs dim=256/4L vs dim=128/8L vs dim=256/8L
- All with identical memory config (64 slots, 16 buckets, same sleep)
- Fixed 600s budget
- If wider/deeper trunk wins despite fewer training steps (bigger model = slower per step), the cerebellum hypothesis holds

**Test B (reallocation):** Fixed total param budget, shift allocation:
- Current: ~60% trunk, ~30% semantic, ~10% routing
- Elephant-inspired: ~85% trunk, ~10% semantic, ~5% routing
- Same total parameter count, same training budget

Test A asks "does a bigger trunk help?" Test B asks "does *reallocating* from semantic to trunk help?" They can share a grid.

**Biological basis:** Herculano-Houzel et al. 2014 (elephant cerebellum 97.5%), Doya 1999 (cerebellum as supervised prediction engine)

---

## 2. Bucket Synergy Matrix (Learned Cross-Type Affinity)

**Observation:** Our merge rule is a hard gate: same bucket only. Biology does both separation (DG/CA3) and integration (CA1) simultaneously, depending on learned context.

**Mechanism:** A tiny N×N affinity matrix (16×16 = 256 floats) that learns which bucket pairs can safely merge:

```
merge_score = slot_similarity * bucket_affinity[a, b]
```

Affinity updates during sleep:
- Merge committed, bpb holds → affinity[a,b] += lr
- Merge rejected by REM → affinity[a,b] -= lr

Starts uniform (no opinion). Over time, the system discovers which semantic domains are friends and which are enemies. The matrix encodes the *structure* of the knowledge domain.

**The "kid cleaning up toys" principle:** Merges should flow like a gradient, not fire like a gate. Similar toys end up together naturally; the kid learns which bins accept mixed toys through experience.

**Also enables:** Data-driven partition topology for polyphasic sleep. High-affinity buckets should share a GPU group in `bucket_striped` mode.

**Biological basis:** Favila et al. 2016 (hippocampal differentiation), Molitor et al. 2021 (simultaneous separation + integration in different subregions)

---

## 3. CFR as a Subsystem

**Observation:** Counterfactual Regret minimization is currently a table that lives inside the training loop. It's updated during REM dreams and queried during inference gating. But it feels like it should be its own module with its own lifecycle.

**Questions to resolve:**
- Should the regret table have its own consolidation during sleep? (Separate from slot consolidation)
- Should it influence routing, not just gating? (If bucket 3's regret is high, route less to it)
- Should it feed back into the synergy matrix? (High regret on cross-bucket decisions → lower affinity)
- Is there a "regret decay" / forgetting mechanism? Old regrets may not apply to current data distribution
- Does it need its own partition in polyphasic sleep, or does it live globally?

**Biological basis:** OFC task-state labeling (Schuck et al. 2016), basal ganglia selective gating (Frank et al. 2001), distributed regret/counterfactual networks (meta-analyses linking OFC, cingulate, ventral striatum)

---

## 4. Context-Dependent Bucket Geometry

**Observation:** Our Wernicke buckets are a fixed partition of representation space (VQ codebook or MoE router weights). Biology shows that categorical boundaries sharpen or dissolve based on task context and attention.

**Idea:** Allow the routing to be conditioned on model state — not just the input token, but also the current surprise level, fatigue, or recent bucket distribution. When the model is "surprised," bucket boundaries could tighten (more separation). When processing predictable text, boundaries could soften (more integration).

**This is speculative.** The current fixed routing already works well (Wernicke MoE is a clean +0.18 bpb win). Context-dependent routing adds complexity and could easily overfit. Worth exploring only after the simpler ideas are validated.

**Biological basis:** Deep research report section on task-driven categorical representations, population coding

---

## 5. Hierarchical Buckets (Multi-Scale Organization)

**Observation:** Brains have multi-scale organization (microcircuits → columns → areas → networks). Our buckets are a single flat set.

**Idea:** Two-level hierarchy: coarse buckets (4-8, like "syntax" / "entities" / "discourse" / "other") containing fine buckets (4-8 each, like "verb morphology" / "noun phrases" / etc.). Merges allowed freely within fine buckets, allowed by synergy between fine buckets in the same coarse bucket, blocked across coarse buckets.

**Risk:** Over-engineering. The flat set might be sufficient for the 16MB artifact scale. Hierarchy matters more as the model grows.

---

## 6. Von Economo Neurons (Fast Cross-Partition Communication)

**Observation:** Dolphins and elephants (large-brained social mammals) have von Economo neurons — specialized cells thought to enable fast long-range communication between brain regions. The polyphasic sleep design has an "Invariant 4: cross-partition exchange is summarized." Von Economo neurons suggest there might be a role for fast, targeted communication between partitions beyond slow summaries.

**Very speculative.** Park this unless polyphasic experiments reveal that partition isolation hurts too much.

---

## Dependencies

```
Experiment 11 results
  ├── → Sleep payload decision (which stages?)
  ├── → k_max sweep (already wired)
  │     └── → Lock k_max for all downstream experiments
  ├── → Experiment 12 (polyphasic sleep, at winning k_max)
  │     └── → Partition topology comparison
  ├── → Experiment 10b (quantization robustness)
  │     └── → Artifact story
  └── → H100 training script

Synergy matrix (item 2)
  ├── depends on: basic sleep working (exp 11)
  └── enables: data-driven partition topology

Trunk scaling (item 1)
  ├── independent of sleep work
  └── can run in parallel with exp 12

Mamba-2 baseline (already wired)
  └── depends on: architecture locked for H100
```
