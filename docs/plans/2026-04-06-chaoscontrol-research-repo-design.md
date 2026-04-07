# ChaosControl Research Repository Design

## Thesis

An SSM built to mirror how the mind processes language — with typed compositional preprocessing, biologically-motivated episodic and semantic memory, critical dynamics, and metabolically-gated generation — outperforms both vanilla SSMs and transformer-patched SSMs on language modeling.

## Repo Structure

```
chaoscontrol/
├── README.md                      # thesis, abstract, how to run
├── pyproject.toml                 # pytest config, project metadata
├── .gitignore
│
├── src/chaoscontrol/
│   ├── __init__.py
│   ├── config.py                  # ChaosControlConfig dataclass
│   ├── core.py                    # ChaosSSMCore (diag/paired/full)
│   ├── routing.py                 # RichBNN, DistributedB
│   ├── memory.py                  # OuterModel, MultiSlotOuterModel, SemanticTier
│   ├── wernicke.py                # WernickeLayer
│   ├── metabolic.py               # metabolic_fork, structured projections
│   ├── model.py                   # ChaosSSMBlock, ChaosStudentLM
│   ├── training.py                # train_chaoscontrol_for_budget
│   ├── evaluation.py              # evaluate_chaoscontrol_bpb
│   ├── losses.py                  # criticality_loss
│   ├── baselines.py               # SimpleTransformerLM for comparison
│   └── data.py                    # batch_from_starts, build_lm_starts,
│                                  #   prepare_tokenized_enwik8_splits,
│                                  #   resolve_device, resolve_param_dtype,
│                                  #   maybe_autocast, maybe_sync_cuda
│
├── tests/
│   ├── test_core.py
│   ├── test_routing.py
│   ├── test_memory.py
│   ├── test_wernicke.py
│   ├── test_metabolic.py
│   ├── test_model.py
│   └── test_training.py
│
├── experiments/
│   ├── 01_baseline/
│   │   ├── README.md
│   │   ├── configs/
│   │   ├── run.sh
│   │   ├── analyze.py
│   │   └── results/.gitkeep
│   ├── 02_critical_dynamics/
│   ├── 03_state_dependent_routing/
│   ├── 04_long_term_memory/
│   ├── 05_typed_composition/
│   ├── 06_metabolic_gate/
│   ├── 07_full_system/
│   └── 08_gap_analysis/
│
├── analysis/
│   └── summarize_results.py
│
└── docs/
    ├── thesis.md
    ├── design.md
    └── research/
```

Source extracted from parameter-golf's `tools/chaoscontrol.py` (1796 LOC monolith) into focused modules. Utilities from `evolutionary_benchmark.py` and `spectral_flood_walk_v2a.py` copied into `data.py` — fully self-contained, no dependency on parameter-golf.

## Experiment Template

Each experiment directory follows:

```
experiments/XX_name/
├── README.md          # hypothesis, predictions, methodology
├── configs/           # YAML config variants for each ablation
├── run.sh             # runs all configs, checkpoints results
├── analyze.py         # reads results JSON, produces tables/plots
└── results/.gitkeep   # gitignored, populated by run.sh
```

Each README.md follows:

```markdown
# Experiment XX: [Title]

## Hypothesis
[One sentence — what we expect and why]

## Null hypothesis
[What would disprove it]

## Predictions
- [Specific, measurable prediction 1]
- [Specific, measurable prediction 2]

## Method
[What configs are tested, what's held constant, what's measured]

## Dependencies
[Which prior experiments must complete first, if any]

## Kill criteria
[When to stop early]
```

## Experiment Definitions

### 01 — Baseline

**Hypothesis:** Establish the floor (vanilla SSM) and ceiling (transformer) at matched parameter budgets.

**Null:** The transformer is strictly better at all sizes — this is expected and establishes the gap ChaosControl aims to close.

| Config | Architecture | model_dim | Params | Budget |
|---|---|---|---|---|
| ssm_small | ChaosSSM (A-diag, nothing else) | 128 | ~2MB | 300s |
| ssm_medium | ChaosSSM (A-diag, nothing else) | 256 | ~8MB | 300s |
| ssm_full | ChaosSSM (A-diag, nothing else) | 384 | ~15MB | 300s |
| tfm_small | Simple transformer | 128 | ~2MB | 300s |
| tfm_medium | Simple transformer | 256 | ~8MB | 300s |
| tfm_full | Simple transformer | 384 | ~15MB | 300s |

6 configs. Dependencies: none.

### 02 — Critical Dynamics

**Hypothesis:** Near-critical A parameterization improves loss over diagonal decay; oscillations emerge naturally in paired/full modes.

**Null:** No A-full variant beats A-diag.

| Config | A mode | Crit target | What it tests |
|---|---|---|---|
| diag | diag | n/a | Baseline (same as 01) |
| paired | paired | n/a | Do oscillations alone help? |
| full_088 | full | 0.88 | Does criticality at 0.88 help? |
| full_085 | full | 0.85 | More subcritical |
| full_092 | full | 0.92 | Less subcritical |
| full_095 | full | 0.95 | Near-critical |
| full_no_reg | full | n/a (reg off) | Does the regularizer matter? |

7 configs. All at Small first, winners at Medium/Full.

**Logging:** FFT of hidden state trajectories, Lyapunov exponent estimates, drift analysis for full_no_reg.

**Kill criteria:** None standalone — criticality may only help in combination with other components. Results are diagnostic.

Dependencies: 01 (for baseline comparison).

### 03 — State-Dependent Routing

**Hypothesis:** B(x,h) outperforms B(x); distributed topology outperforms monolithic NN.

**Null:** No rich B variant beats the no-rich-B baseline with the same A mode.

| Config | A mode | Rich B | Topology | Settling | What it tests |
|---|---|---|---|---|---|
| diag_none | diag | none | n/a | n/a | Baseline |
| diag_nn | diag | nn | n/a | n/a | Does B(x,h) help at all? |
| diag_hub | diag | hub | hub | n/a | Distributed, no interaction |
| diag_assembly_2 | diag | assembly | assembly | 2 | Settling with lateral messages |
| diag_assembly_4 | diag | assembly | assembly | 4 | More settling steps |
| diag_hybrid_2 | diag | hybrid | hybrid | 2 | Hub + settling |
| full_none | full | none | n/a | n/a | Criticality alone |
| full_nn | full | nn | n/a | n/a | Criticality + routing |
| full_hub | full | hub | hub | n/a | Criticality + distributed |
| full_assembly_2 | full | assembly | assembly | 2 | Criticality + settling |

10 configs.

**Key question:** Does distributed beat monolithic NN? If not, the brain-inspired topology doesn't matter — simple state-dependence is enough.

Dependencies: 01, 02 (to interpret interaction with A mode).

### 04 — Long-Term Memory

**Hypothesis:** Two-tier memory (episodic + semantic) outperforms single-tier and no-memory. Consolidation from episodic to semantic produces qualitatively different knowledge representations.

**Null:** No memory variant beats the no-memory baseline.

**Architecture:**
- **Episodic tier** (hippocampal): multi-slot, cue-dependent retrieval, pattern completion from partial cue, surprise-driven consolidation.
- **Semantic tier** (neocortical): slowly-updated basis vectors that provide always-on background bias to the recurrence. Not gated by surprise, not cue-dependent — a persistent prior extracted from episodic experience.
- **Consolidation**: when multiple episodic slots share structure along certain axes, extract that shared structure into the semantic tier. The episodic originals then decay naturally. Episodes → gist → knowledge.

**Retrieval paths:**
- Episodic: recurrence state cues episodic tier → pattern completion → reinstated into recurrence. Expensive, for novel/specific recall.
- Semantic: always-on bias added to recurrence at every step. Cheap, shapes all processing. For well-learned patterns.

| Config | Episodic | Semantic | Consolidation | Trigger | What it tests |
|---|---|---|---|---|---|
| no_memory | off | off | n/a | n/a | Baseline |
| episodic_only | multislot | off | n/a | immediate | Does episodic memory help? |
| semantic_only | off | basis vectors | n/a | n/a | Does background knowledge help? |
| both_no_transfer | multislot | basis vectors | independent | resolution | Both tiers, no consolidation between |
| both_with_transfer | multislot | basis vectors | episodic→semantic | resolution | Full system: gist extraction |
| both_transfer_typed | multislot (typed) | basis vectors (per type) | typed extraction | resolution | Typed consolidation |
| episodic_res | multislot | off | n/a | resolution | Resolution trigger vs immediate |
| episodic_win | multislot | off | n/a | windowed | Windowed trigger |
| episodic_pain | multislot | off | pain_biased | resolution | Pain-biased consolidation |
| episodic_survival_vs_uniform | multislot | off | survival vs uniform merge | resolution | Does impact scoring improve compression? |

10 configs.

**Key comparison:** `both_with_transfer` vs `both_no_transfer` — if transfer helps, consolidation (gist extraction) is doing real work.

Dependencies: 02, 03 (for A mode and rich B selections).

### 05 — Typed Composition (Wernicke)

**Hypothesis:** Wernicke layer discovers meaningful types; typed compression preserves more than untyped; compression-consequence training signal produces useful type distinctions.

**Null:** No Wernicke variant beats the untyped baseline.

| Config | Wernicke | Router | K_max | Typed storage | Typed consolidation | What it tests |
|---|---|---|---|---|---|---|
| no_wernicke | off | n/a | n/a | off | off | Memory without typing |
| vq_8 | on | VQ | 8 | on | off | Few types, VQ routing |
| vq_16 | on | VQ | 16 | on | off | Medium types |
| vq_32 | on | VQ | 32 | on | off | Many types |
| moe_8 | on | MoE | 8 | on | off | Few types, MoE routing |
| moe_16 | on | MoE | 16 | on | off | Medium types |
| typed_no_storage | on | VQ | 16 | off | off | Wernicke helps recurrence, memory untyped |
| typed_episodic | on | VQ | 16 | on | off | Types route to separate episodic banks |
| typed_both | on | VQ | 16 | on | on | Types route episodic + inform semantic |
| compression_consequence | on | VQ | 16 | on | on | Typing head learns from merge quality |

10 configs.

**Critical config:** `compression_consequence` — the novel training signal. If this works, types become meaningful for memory. If not, Wernicke is just a learned tokenizer.

**Logging:** Bucket utilization, sample inputs per bucket, loss delta after typed vs untyped merges, effective K over training.

Dependencies: 04 (for memory config).

### 06 — Metabolic Gate

**Hypothesis:** Generation+selection fork provides a small systematic advantage on high-surprise tokens. Structured projections ("choosing the question") outperform random noise.

**Null:** No gate variant beats the no-gate baseline.

| Config | Gate | K | Threshold | Mode | Scoring | Generation | What it tests |
|---|---|---|---|---|---|---|---|
| no_gate | off | n/a | n/a | n/a | n/a | n/a | Baseline |
| mem_consist_4 | on | 4 | 0.1 | fixed | memory_consistency | noise | Memory-guided selection |
| ensemble_4 | on | 4 | 0.1 | fixed | ensemble_agreement | noise | Consensus selection |
| lookahead_4 | on | 4 | 0.1 | fixed | loss_lookahead | noise | Confidence selection |
| mem_consist_8 | on | 8 | 0.1 | fixed | memory_consistency | noise | More candidates |
| adaptive_mem | on | 4 | 0.1 | adaptive | memory_consistency | noise | Learned gate frequency |
| high_thresh | on | 4 | 0.3 | fixed | memory_consistency | noise | Fork only on major surprises |
| low_thresh | on | 4 | 0.03 | fixed | memory_consistency | noise | Fork on mild surprises |
| noise_high | on | 4 | 0.1 | fixed | memory_consistency | noise (std=0.05) | More divergent candidates |
| noise_low | on | 4 | 0.1 | fixed | memory_consistency | noise (std=0.002) | Less divergent candidates |
| structured_proj | on | 4 | 0.1 | fixed | memory_consistency | learned projections | "Choosing the question" — K different views of input |

11 configs.

**Key insight from NFT:** The advantage is a compass, not a cannon. +1% compounding over 1000 timesteps is massive. Analysis should measure systematic directional bias across many sequences, not per-step improvement.

**Critical config:** `structured_proj` — tests whether structured generation (choosing what question to ask) outperforms random perturbation. This is the NFT-aligned mechanism.

**Logging:** Per-forked-step loss vs non-forked, fork rate over training, adaptive threshold trajectory, candidate divergence by scoring method.

Dependencies: 04, 05 (for memory and Wernicke config).

### 07 — Full System

**Hypothesis:** All components together outperform any subset. The system is synergistic.

**Null:** `full_system` does not beat `best_single`.

| Config | Components | What it tests |
|---|---|---|
| transformer | Transformer baseline from 01 | The ceiling |
| vanilla_ssm | A-diag, nothing else | The floor |
| best_single | Best single component from 02-06 | One piece alone |
| best_pair | Best two components from 02-06 | Pairwise synergy |
| full_no_gate | All except metabolic gate | Everything minus expensive fork |
| full_no_wernicke | All except Wernicke layer | Everything minus typed composition |
| full_no_memory | All except outer model | Everything minus LTM |
| full_system | All five components | The thesis |
| full_system_medium | All five, medium size | Does it scale? |
| full_system_full | All five, full size (15MB) | Does it scale further? |

10 configs. `best_single` and `best_pair` determined from prior results.

**The paper question:** Does `full_system` close the gap between `vanilla_ssm` and `transformer`? If it exceeds the transformer, that's the headline. If it closes the gap significantly, that's still a contribution. If it doesn't beat `best_single`, the synergy thesis is wrong.

Dependencies: 01-06 (all).

### 08 — Gap Analysis

**Hypothesis:** Test the weakest claims in the thesis explicitly.

| Config | What it tests | Why it's a gap |
|---|---|---|
| no_cue_proj | Remove cue projection, use raw recurrence state as retrieval key | Tests "recurrence is the index" — is scaffolding load-bearing? |
| compression_consequence | Wernicke typing trained from merge quality | The novel training signal — does it converge? |
| dynamic_crit_per_layer | Different criticality targets per layer | Tests dynamic criticality — subsystems tune differently |
| semantic_emergence | Full system at 10x budget | Does semantic tier content diverge from episodic over time? |
| structured_vs_noise | Structured projections vs isotropic noise in metabolic gate | Does generation mechanism matter? |
| survival_vs_random | Survival scoring vs random during compression | Is impact-based retention doing real work? |

6 configs.

**This experiment is the most important scientifically.** It tests the things we believe but haven't proven. Negative results here are as valuable as positive ones — they tell us which parts of the thesis to revise.

Dependencies: 07 (full system as base config).

## Total Experiment Count

| Experiment | Configs | Dependencies |
|---|---|---|
| 01 Baseline | 6 | none |
| 02 Critical Dynamics | 7 | 01 |
| 03 State-Dependent Routing | 10 | 01, 02 |
| 04 Long-Term Memory | 10 | 02, 03 |
| 05 Typed Composition | 10 | 04 |
| 06 Metabolic Gate | 11 | 04, 05 |
| 07 Full System | 10 | 01-06 |
| 08 Gap Analysis | 6 | 07 |
| **Total** | **70** | |

At 300s per config on a single GPU: ~5.8 hours for 01-06, ~1.4 hours for 07-08. Total: ~7.2 hours, ~$18.

## Implementation Notes

### Semantic Tier (new, not yet coded)

The semantic tier is the major new component that needs implementation. It differs from the episodic tier:

- **Storage:** Set of learned basis vectors (not slots). Updated slowly.
- **Retrieval:** Always-on. Added to recurrence at every step as a background bias. Not cue-dependent.
- **Consolidation process:** When multiple episodic slots share structure (high cosine similarity along principal axes), extract that shared structure as a new basis vector. Let the episodic originals decay.
- **Training:** The basis vectors update via a slow EMA of the extracted structure. No task-loss gradient. The semantic tier learns from what episodes have in common, not from what the task needs.

### Structured Projections for Metabolic Gate (new, not yet coded)

Replace isotropic noise with K learned projection heads. Each head emphasizes different features of the input. The system generates K different "views" of the same input and selects the best one. This is "choosing the question" — the NFT-aligned generation mechanism.

### Compression-Consequence Training Signal (not yet wired)

The Wernicke typing head needs a feedback loop from the outer model's compression outcomes. When a merge happens, compare pre-merge and post-merge retrieval quality. If quality dropped, signal the typing head that those slots should have been in different buckets. This signal is implemented as a gradient-free update to the router weights, similar to the learned consolidation weight.
