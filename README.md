# SemanticEngine SSM

*Codename: ChaosControl*

**A research repo for efficient typed-memory architectures on top of a state-space model.**

16MB artifact target. FineWeb validation. Bits-per-byte metric. Built for the OpenAI Parameter Golf competition.

## Current Status

As of **April 9, 2026**:

- **Bare SSM is the strongest confirmed system at 600s on A40.**
- **Wernicke routing helped in earlier lower-budget ablations, but recent 600s baseline sweeps show routing and full-stack memory are currently underwater on A40 because they cost too many training steps.**
- **Sleep/consolidation is no longer the active center of the project.** Experiment 11 showed that only the cheapest `n3_only` compression pass pays for itself at this budget.
- **The active hypothesis has shifted to Experiment 14:** typed, append-only KV memory with local retrieval may preserve the semantic-engine idea while removing most of the operational overhead.

This repo should be read as an active research program, not a claim of architectural victory. The codebase contains both:

- ideas that have survived ablation
- ideas that are still scientifically interesting but currently unsupported at the tested budget

## Thesis

The durable thesis is:

**An SSM trunk plus typed routing plus structured memory may beat a plain trunk if memory behaves like cheap state rather than expensive extra compute.**

That breaks into two layers:

### Core Invariants

- The SSM trunk should do the fast sequential prediction work.
- Typed routing should organize information before memory access.
- Memory should be structured by semantic type, not just raw recency.
- Efficiency matters as much as raw quality: extra mechanisms must earn their step cost.

### Replaceable Mechanisms

- surprise-gated writes
- offline sleep consolidation
- REM/CFR mechanisms
- retrieval strategy
- flat vs hierarchical Wernicke routing
- global semantic basis vs per-bucket prototypes

Those mechanisms are implementation choices, not the thesis itself.

## Architecture Evolution

### Historical emphasis (Experiments 09-13)

```text
raw bytes
  -> Wernicke typed routing
  -> SSM trunk
  -> surprise-gated episodic memory
  -> sleep-time consolidation (N1/N2/N3/REM)
  -> semantic / latent memory refinements
```

This version produced useful negative results:

- the metabolic gate hurts badly at eval time on memoryless checkpoints
- heavy sleep stages lose too many steps at 600s
- bare SSM wins the latest 600s A40 baseline sweep

### Current active emphasis (Experiment 14)

```text
raw bytes
  -> Wernicke typed routing
  -> append-only typed KV buffer
  -> within-bucket retrieval
  -> optional per-bucket prototypes
  -> SSM trunk
```

The active question is no longer "can elaborate consolidation help?" but:

**Can typed locality make memory cheap enough to matter?**

## What The Data Says So Far

### Confirmed Results

| Finding | Evidence |
|---|---|
| **Bare SSM is the current 600s A40 winner** | Baseline sweep summary in `experiments/OVERNIGHT_SUMMARY.md` |
| **`n3_only` is the only sleep payload that helps at 600s** | `experiments/11_sleep_cycle/REPORT_exp11.md` |
| **Metabolic gating hurts on memoryless checkpoints** | `experiments/09_revised_architecture/REPORT_phase2.md` |
| **`crit_target_coupling=0.92` and `outer_max_slots=32` look like the strongest constant improvements** | `experiments/13_constants_validation/REPORT_exp13.md` |

### Current Interpretation

- The original semantic-engine stack is **not yet compute-efficient enough** at the tested budget.
- The problem may be **mechanism placement**, not the entire idea.
- If ChaosControl wins, it is more likely to be through **typed routing + cheap typed memory** than through expensive offline maintenance.

## Experiments

| Experiment | What it tests | Status on April 9, 2026 |
|---|---|---|
| **09: Revised architecture** | Layered ablation: tokenizer, gate, memory, Wernicke, CFR | Complete |
| **10: Scaling laws** | Size and architecture scaling configs | Designed / partially wired |
| **10b: Quantization robustness** | Delta-bpb curves under int8/int6, reactivation isolation | Designed, not run |
| **11: Sleep cycle ablation** | 9 sleep conditions with REM mechanism isolation | Complete |
| **12: Polyphasic sleep** | K-of-N partition scheduling and topology comparison | Code ready, not yet run |
| **13: Constants validation** | Criticality, slot count, semantic tier, merge threshold | Complete |
| **14: VRAM typed buffer** | Append-only typed KV memory, retrieval ablations, hierarchical Wernicke, TTT warming curves | Active next architecture test |
| **Baselines** | Bare SSM, Wernicke variants, full stack, Mamba-2 wiring | Baseline sweep complete; Mamba-2 package still optional |

## Snapshot Results

### Experiment 11: Sleep Cycle Ablation (600s, 7 seeds)

| Condition | Mean bpb | Steps | Verdict |
|---|---|---|---|
| `no_sleep` | 2.4210 | 390 | Baseline |
| **`n3_only`** | **2.4090** | 385 | **Best sleep condition** |
| `n2_n3` | 2.4946 | 320 | Too expensive |
| `full_cycle` | 2.4916 | 316 | Too expensive |

### Baseline Sweep (600s, A40)

| Configuration | Mean bpb | Steps | Verdict |
|---|---|---|---|
| **bare_ssm** | **2.478** | 505 | **Current winner** |
| ssm_wernicke_k16 | 2.488 | 446 | Slightly worse |
| full_stack_k16 | 2.502 | 433 | Worse |

### Experiment 13: Constants Validation

| Constant | Default | Best candidate | Delta | Action |
|---|---|---|---|---|
| `crit_target_coupling` | 0.88 | **0.92** | -0.017 | confirm |
| `outer_max_slots` | 64 | **32** | -0.033 | confirm |
| `semantic_tier` | off | `b8/r0.1` | -0.008 | trend only |

## Experiment 14

Experiment 14 is the most important next-step architecture test in the repo.

It asks whether ChaosControl becomes more plausible when memory is treated as a **typed, rebuildable buffer** instead of a consolidation-heavy subsystem.

### Experiment 14 Claims

- **Claim 1:** A typed append-only KV buffer can give an SSM useful context access without transformer-style attention cost.
- **Claim 2:** If Claim 1 works, developmental fast weights and online defrag may further improve warmup and buffer quality.

### Experiment 14 Design

```text
Input bytes
  -> byte embedding
  -> Wernicke routing (flat or hierarchical)
  -> typed KV buffer read within bucket
  -> optional bucket prototype bias
  -> SSM backbone
  -> unconditional append to matching bucket
  -> LM head
```

### Experiment 14 Retrieval Modes

| Mode | Character |
|---|---|
| `bucket_mean` | routing does most of the work |
| `bucket_recent` | recency-biased local retrieval |
| `bucket_topk` | selective retrieval within bucket |
| `softmax_all` | legacy/global baseline |

### Experiment 14 Evaluation Story

The important metric is not just pretrain bpb. It is the **warming curve**:

- cold artifact performance with an empty buffer
- performance after 100 / 500 / 1000 / 5000 tokens of rebuild

If that curve is steep, then the typed buffer is doing real work.

## Quick Start

```bash
# Install
python -m venv .venv && .venv/bin/pip install torch numpy pyyaml

# Run tests
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q

# Reproduce the sleep ablation
PYTHONPATH=src .venv/bin/python experiments/11_sleep_cycle/run_sleep_ablation.py \
    --data-path /path/to/fineweb_data --budget 600 --num-gpus 4

# Run Experiment 14 Phase A
PYTHONPATH=src .venv/bin/python experiments/14_vram_buffer/run_exp14.py \
    --data-path /path/to/fineweb_data --budget 600 --num-gpus 8 --phase A
```

## Project Structure

```text
src/chaoscontrol/
  core.py              SSM recurrence (diag/paired/full A-modes)
  model.py             ChaosStudentLM wiring and Exp 14 buffer path
  wernicke.py          Flat + hierarchical typed routing
  memory.py            Multi-slot memory, typed buffer, prototypes, affinity
  sleep.py             Sleep cycle (N1/N2/N3/REM consolidation)
  wake_cache.py        High-signal moment cache for sleep
  fatigue.py           Dynamic fatigue tracker
  partition.py         Polyphasic partitions + scheduler
  metabolic.py         Fork/MC/MCTS gating experiments
  regret.py            Counterfactual regret table
  training.py          Training loop with sleep and typed-buffer integration
  evaluation.py        Eval + bpb / warming-curve calculation
  artifact.py          16MB artifact serialization
  baselines.py         SimpleTransformerLM, Mamba2LM
  config.py            ChaosControlConfig dataclass
  data.py              FineWeb data loading

experiments/
  09_revised_architecture/   Layered ablation
  10_scaling_laws/           Scaling-law sweep scaffolding
  11_sleep_cycle/            Sleep stage ablation
  12_polyphasic_sleep/       Partitioned sleep scheduling
  13_constants_validation/   Constant sweeps and confirmations
  14_vram_buffer/            Typed-buffer architecture test

docs/plans/                  Design documents and implementation plans
tools/                       RunPod deployment, polling, watchdog scripts
tests/                       Unit and integration coverage
```

## References

- McClelland, McNaughton & O'Reilly 1995 — Complementary learning systems
- Doya 1999 — Cerebellum/cortex/basal ganglia computational trichotomy
- Herculano-Houzel et al. 2014 — Elephant brain neuron distribution
- Favila et al. 2016 — Hippocampal pattern differentiation
- Molitor et al. 2021 — Simultaneous DG separation + CA1 integration
- Schuck et al. 2016 — OFC task-state representation
- Leutgeb et al. 2007 — Pattern separation in dentate gyrus and CA3
