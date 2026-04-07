# ChaosControl

**A biologically-inspired SSM for the OpenAI Parameter Golf competition.**

16MB artifact. FineWeb validation. Bits-per-byte.

## Thesis

A state-space model designed from biological principles — learned tokenization, typed composition, episodic/semantic memory with surprise-driven consolidation, metabolic gating with MCTS planning, and counterfactual regret tracking — can match or beat transformers on language modeling while offering O(d) inference and fixed-size state.

## Architecture

```
raw bytes (256)
  -> FixedStrideTokenizer (Stage 0: bytes -> learned VQ tokens)
  -> Wernicke (Stage 1: tokens -> typed semantic units)
  -> SSM recurrence (working memory, diag/paired/full A-modes)
  -> Metabolic gate (System 2: fork/MC/MCTS on surprise)
  -> Memory tiers (episodic -> semantic -> latent, demand-driven compression)
  -> CFR regret tracking (exploration bias via counterfactual lookahead)
```

## Components

| Component | Analogy | What it does |
|---|---|---|
| **Learned tokenizer** | Perceptual segmentation | Fixed-stride causal conv + VQ codebook; bytes -> tokens |
| **Wernicke layer** | Typed composition | VQ/MoE routing into semantic buckets |
| **SSM core** | Working memory | State-space recurrence with critical dynamics |
| **Metabolic gate** | System 2 reasoning | MCTS/fork/MC search on high-surprise tokens |
| **Episodic memory** | Hippocampal consolidation | Multi-slot memory with surprise-driven writes |
| **Semantic memory** | Neocortical knowledge | Background bias extracted from episodes |
| **Latent persistence** | Memory reactivation | Compressed traces reactivated on surprise |
| **CFR tracking** | Exploration policy | Counterfactual regret minimization over gate decisions |

## Quick start

```bash
# Experiment 09: layered ablation (L0 tokenizer through L6 inference adaptation)
bash experiments/09_revised_architecture/run.sh /path/to/fineweb_data --budget 150

# Tests
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

## Experiments

| Experiment | Layers | Configs | What it tests |
|---|---|---|---|
| **09: Revised architecture** | L0-L6 | ~35 | Tokenizer, gate modes, memory, Wernicke, CFR, scaling, inference adaptation |
| **10: Scaling laws** | -- | ~63 | SSM vs transformer vs Mamba-2 scaling curves, component ROI, quantization robustness |

## Baselines

| Baseline | bpb | Source |
|---|---|---|
| Competition transformer | 1.2244 | `baselines/parameter_golf/train_gpt.py` |
| Competition SOTA | 1.1147 | `baselines/parameter_golf/sota/` |
| Mamba-2 | TBD | `src/chaoscontrol/baselines.py` (requires `mamba-ssm`) |
| Our transformer | TBD | `src/chaoscontrol/baselines.py` |

## Project structure

```
src/chaoscontrol/        # core library
  core.py                #   SSM recurrence (diag/paired/full A-modes)
  model.py               #   ChaosStudentLM (SSM + Wernicke + memory + gate)
  tokenizer.py            #   FixedStrideTokenizer (VQ + reconstruction)
  alignment.py            #   codebook coupling losses
  vq.py                   #   vector quantization utilities
  metabolic.py            #   fork, monte carlo, micro-MCTS gates
  memory.py               #   episodic/semantic/latent memory tiers
  regret.py               #   CFR regret table
  training.py             #   training loop
  evaluation.py           #   eval + bpb calculation
  runner.py               #   experiment runner
  baselines.py            #   SimpleTransformerLM, Mamba2LM
  data.py                 #   enwik8 + FineWeb loaders
  config.py               #   ChaosControlConfig dataclass
experiments/09_*/         # active experiment (layered ablation)
baselines/parameter_golf/ # competition baseline + SOTA reference
docs/plans/               # design docs and implementation plans
tools/                    # deployment scripts (RunPod)
tests/                    # 177 tests
archive/round1/           # completed round 1 experiments (01-08)
```
