# Future Work

Research directions identified during the ChaosControl design process (April 2026). Each item includes the scientific hypothesis, proposed test, and biological basis.

---

## Implemented (pending experimental validation)

These are coded and tested but awaiting experiment results before activation or further development.

| Feature | Status | Code | Blocked on |
|---------|--------|------|------------|
| Offline sleep (N1/N2/N3/REM) | Running (Experiment 11) | `sleep.py`, `wake_cache.py`, `fatigue.py` | Results pending |
| Polyphasic partitioned sleep | Experiment 12 ready | `partition.py`, training loop integration | Exp 11 sleep payload decision |
| k_max sweep (16/32/64) | Wired, param-controlled | `wernicke.py` expert bottleneck, baseline runner | Pod time |
| Bucket synergy matrix | Implemented | `memory.py` affinity matrix, `sleep.py` joint merge ranking | Exp 11 (needs working sleep) |
| Adaptive fatigue triggers | Implemented, disabled | `fatigue.py` | Exp 11 (which stages help?) |
| Mamba-2 baseline | Wired | `baselines.py`, baseline runner | H100-ready architecture lock |
| Architecture baselines | Wired | Baseline runner (bare_ssm, ssm_wernicke, full_stack) | Pod time |

---

## 1. Trunk Scaling (Cerebellum Hypothesis)

The elephant cerebellum contains 97.5% of all neurons in a 3,300:1 ratio of prediction to output neurons. ChaosControl's SSM trunk is the analogous prediction engine. The hypothesis: disproportionately scaling the trunk while holding the semantic engine constant improves bpb more than scaling the semantic engine.

**Test A — Scaling.** Trunk sweep at fixed semantic capacity: dim=128/4L vs dim=256/4L vs dim=128/8L vs dim=256/8L. Identical memory config (64 slots, 16 buckets). Fixed 600s budget.

**Test B — Reallocation.** Fixed total parameter budget. Current split (~60% trunk, ~30% semantic, ~10% routing) vs elephant-inspired (~85% trunk, ~10% semantic, ~5% routing).

Independent of sleep work. Can run in parallel with Experiment 12.

**References:** Herculano-Houzel et al. 2014 (*Frontiers in Neuroanatomy*); Doya 1999 (*Neural Networks*)

---

## 2. Counterfactual Regret as a Subsystem

The regret table currently lives inside the training loop as a flat tensor updated during REM dreams and queried at inference gating. It should be its own module with a defined lifecycle.

**Open questions:**
- Should it consolidate during sleep independently of slot consolidation?
- Should accumulated regret influence Wernicke routing, not just metabolic gating?
- Should it feed back into the synergy matrix (high regret on cross-bucket decisions reduces affinity)?
- Does it need regret decay as the data distribution shifts?
- Does it partition in polyphasic sleep, or remain global?

**References:** Schuck et al. 2016 (*Neuron*) — OFC task-state labeling; Frank et al. 2001 (*CABN*) — basal ganglia selective gating

---

## 3. Context-Dependent Bucket Geometry

Wernicke buckets are a fixed partition of representation space. In biology, categorical boundaries sharpen or dissolve based on task context and attention. The idea: condition routing on model state (surprise level, fatigue, recent bucket distribution). When surprised, bucket boundaries tighten. When processing predictable text, boundaries soften.

Speculative. The current fixed routing works well (+0.18 bpb). Context-dependent routing adds complexity and overfitting risk. Explore only after simpler mechanisms are validated.

**References:** Deep research report on task-driven categorical representations and population coding

---

## 4. Hierarchical Buckets (Multi-Scale Organization)

Brains exhibit multi-scale organization (microcircuits, columns, areas, networks). ChaosControl's buckets are a single flat set. A two-level hierarchy — coarse buckets (4-8) containing fine buckets (4-8 each) — would allow free merging within fine buckets, synergy-gated merging within a coarse bucket, and blocked merging across coarse buckets.

Note: `affinity_clusters()` on the synergy matrix already discovers emergent coarse groupings. If those clusters are stable and interpretable, they provide the hierarchy without additional architecture.

Risk: over-engineering at the 16MB artifact scale. Hierarchy matters more as the model grows.

---

## 5. Von Economo Neurons (Fast Cross-Partition Communication)

Dolphins and elephants possess von Economo neurons — specialized cells thought to enable fast long-range communication between brain regions. The polyphasic design constrains cross-partition exchange to compact summaries (Invariant 4). Von Economo neurons suggest a role for fast, targeted communication beyond slow summaries.

Very speculative. Revisit only if polyphasic experiments reveal that partition isolation degrades performance.

---

## Dependency Graph

```
Experiment 11 results
  ├── Sleep payload decision (which stages earn their keep?)
  ├── k_max sweep (wired, auto-launches after exp 11)
  │     └── Lock k_max for all downstream experiments
  ├── Experiment 12: polyphasic sleep (at winning k_max)
  │     └── Partition topology comparison (slot_striped vs bucket_striped)
  ├── Experiment 10b: quantization robustness
  │     └── Artifact compression story
  └── H100 training script ("the ferrari")

Synergy matrix
  ├── Depends on: sleep working (exp 11)
  └── Enables: data-driven partition topology, emergent hierarchical buckets

Trunk scaling (item 1)
  ├── Independent of sleep work
  └── Can run in parallel with exp 12

Mamba-2 baseline
  └── Depends on: architecture locked for H100
```
