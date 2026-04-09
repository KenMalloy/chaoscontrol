# SemanticEngine SSM

*Codename: ChaosControl*

**A biologically-inspired state-space model with structured memory consolidation.**

16MB artifact target. FineWeb validation. Bits-per-byte metric. Built for the OpenAI Parameter Golf competition.

## Thesis

A state-space model designed from biological principles — typed semantic routing, episodic/semantic memory with structured offline consolidation, metabolic gating with MCTS planning, and counterfactual regret tracking — can match or beat transformers on language modeling while offering O(d) inference and fixed-size state.

The central architectural claim: **the SSM trunk handles fast sequential prediction (cerebellum-like), while the semantic engine handles slow knowledge organization (hippocampal/cortical), and structured sleep consolidation bridges them.**

## Architecture

```
raw bytes (0-255)
  -> Wernicke layer (typed composition: VQ/MoE routing into semantic buckets)
  -> SSM recurrence (state-space model with diag/paired/full A-modes)
  -> Metabolic gate (fork/MC/MCTS search on high-surprise tokens, inference only)
  -> Memory tiers:
       episodic (multi-slot, cue-dependent retrieval, typed by bucket)
       semantic (slow background bases extracted from episodes)
       latent (compressed traces, reactivatable on surprise)
  -> Counterfactual regret tracking (exploration bias via bucket-keyed regret table)

Training-time only:
  -> Sleep cycle (N1 transition -> N2 utility scoring -> N3 prune/merge -> REM dreams)
  -> Bucket synergy matrix (learned cross-type merge affinity, updated during sleep)
  -> Polyphasic scheduling (K-of-N partition rotation for concurrent consolidation)
  -> Fatigue system (dynamic ODE triggering sleep based on surprise/improvement/pressure)
```

## Components

| Component | Biological analogy | Source |
|---|---|---|
| **Wernicke layer** | Typed composition (Wernicke's area) | `wernicke.py` |
| **SSM core** | Sequential prediction (cerebellum) | `core.py` |
| **Metabolic gate** | System 2 deliberation (prefrontal) | `metabolic.py` |
| **Episodic memory** | Fast encoding (hippocampus) | `memory.py` |
| **Semantic memory** | Slow consolidation (neocortex) | `memory.py` |
| **Latent traces** | Memory reactivation | `memory.py` |
| **Sleep cycle** | Offline consolidation (N1/N2/N3/REM) | `sleep.py` |
| **Bucket affinity** | Pattern separation + integration (DG/CA3/CA1) | `memory.py` |
| **Polyphasic sleep** | Distributed consolidation (dolphin sleep generalized) | `partition.py` |
| **Fatigue tracker** | Sleep pressure dynamics | `fatigue.py` |
| **Regret tracking** | Counterfactual evaluation (OFC/striatum) | `regret.py` |

## Experiments

| Experiment | What it tests | Status |
|---|---|---|
| **09: Revised architecture** | Layered ablation: tokenizer, gate, memory, Wernicke, CFR | Complete (Phase 1-2) |
| **10b: Quantization robustness** | Delta-bpb curves under int8/int6, reactivation mechanism isolation | Designed, not run |
| **11: Sleep cycle ablation** | 9 conditions: no_sleep through full_cycle with REM mechanism isolation | Running |
| **12: Polyphasic sleep** | K-of-N partition scheduling, 3 topologies (slot_striped, bucket_owned, bucket_striped) | Code ready |
| **Baselines** | Mamba-2, bare SSM, SSM+Wernicke, full stack; k_max sweep (16/32/64) | Wired |

## Key Results (from Phase 1-2)

| Configuration | bpb | Budget |
|---|---|---|
| Bare SSM | 2.90 | 150s |
| SSM + Wernicke MoE | 2.72 | 150s |
| Full stack (SSM + Wernicke + memory) | 2.42 | 600s |
| n3_only sleep (compression pass) | 2.41 | 600s |

Memory hurts at 150s but helps at 600s. Wernicke is a clean win at any budget. Sleep results (full ablation) pending.

## Quick Start

```bash
# Install
python -m venv .venv && .venv/bin/pip install torch numpy pyyaml

# Tests (288 passing)
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q

# Run experiment 11 (sleep ablation)
PYTHONPATH=src .venv/bin/python experiments/11_sleep_cycle/run_sleep_ablation.py \
    --data-path /path/to/fineweb_data --budget 600 --num-gpus 4
```

## Project Structure

```
src/chaoscontrol/
  core.py              SSM recurrence (diag/paired/full A-modes)
  model.py             ChaosStudentLM (full model wiring)
  wernicke.py          Wernicke layer (VQ/MoE typed routing)
  memory.py            Episodic/semantic/latent memory + bucket affinity
  sleep.py             Sleep cycle (N1/N2/N3/REM consolidation)
  wake_cache.py        High-signal moment cache for sleep
  fatigue.py           Dynamic fatigue tracker
  partition.py         Polyphasic partitions + scheduler
  metabolic.py         Fork/MC/MCTS metabolic gates
  regret.py            CFR regret table
  training.py          Training loop with sleep + polyphasic integration
  evaluation.py        Eval + bpb calculation
  runner.py            Experiment runner
  artifact.py          16MB artifact serialization (int8/int6 + LZMA)
  baselines.py         SimpleTransformerLM, Mamba2LM
  config.py            ChaosControlConfig dataclass
  data.py              FineWeb data loading

experiments/
  09_revised_architecture/   Layered ablation (Phase 1 training, Phase 2 eval)
  11_sleep_cycle/            Sleep stage ablation (9 conditions x 7 seeds)
  12_polyphasic_sleep/       Polyphasic partition scheduling
  baselines/                 Architecture baselines + k_max sweep

docs/plans/                  Design documents and implementation plans
tools/                       RunPod deployment, polling, watchdog scripts
tests/                       288 tests
```

## References

- McClelland, McNaughton & O'Reilly 1995 — Complementary learning systems
- Doya 1999 — Cerebellum/cortex/basal ganglia computational trichotomy
- Herculano-Houzel et al. 2014 — Elephant brain neuron distribution
- Favila et al. 2016 — Hippocampal pattern differentiation
- Molitor et al. 2021 — Simultaneous DG separation + CA1 integration
- Schuck et al. 2016 — OFC task-state representation
- Leutgeb et al. 2007 — Pattern separation in dentate gyrus and CA3
