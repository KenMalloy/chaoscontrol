# ChaosControl

**An SSM built to mirror how the mind processes language.**

## Thesis

A state-space model designed from biological principles — typed compositional preprocessing, episodic and semantic memory with surprise-driven consolidation, near-critical dynamics, and metabolically-gated generation — outperforms both vanilla SSMs and transformer-patched SSMs on language modeling.

## Components

| Component | Brain analogy | What it does |
|---|---|---|
| **Critical dynamics** | Cortical criticality | A matrix parameterized for near-critical operation; perturbations propagate at all scales |
| **State-dependent routing** | Distributed semantic network | Input coupling B(x,h) depends on state; distributed sub-networks with lateral settling |
| **Episodic memory** | Hippocampal consolidation | Multi-slot VRAM memory with cue-dependent retrieval, surprise-driven writes, survival scoring |
| **Semantic memory** | Neocortical knowledge | Always-on background bias extracted from episodic experience over time |
| **Typed composition** | Wernicke's area | Composes bytes into units, assigns type buckets, routes to typed memory banks |
| **Metabolic gate** | Quantum navigational faculty | Generation+selection fork on high-surprise tokens; structured projections choose the question |

## Quick start

```bash
# Single experiment
PYTHONPATH=src .venv/bin/python -m chaoscontrol.runner \
    --config experiments/01_baseline/configs/ssm_small.yaml \
    --enwik8-path /path/to/enwik8 \
    --budget 300

# Full suite (~7 hours, single GPU)
bash run_all.sh /path/to/enwik8 --budget 300
```

## Experiments

| # | Name | Configs | Tests |
|---|---|---|---|
| 01 | Baseline | 6 | SSM floor, transformer ceiling |
| 02 | Critical dynamics | 7 | A parameterization, oscillations, target sweep |
| 03 | State-dependent routing | 10 | NN vs distributed, topology comparison |
| 04 | Long-term memory | 10 | Episodic, semantic, consolidation, two-tier |
| 05 | Typed composition | 10 | Wernicke layer, VQ vs MoE, compression-consequence |
| 06 | Metabolic gate | 11 | Scoring, threshold, structured projections |
| 07 | Full system | 10 | Synergy test, ablations, scaling |
| 08 | Gap analysis | 6 | Weakest claims tested explicitly |

**Total: 70 configs, ~7 hours on a single GPU**

## Project structure

```
src/chaoscontrol/     # modular source (core, routing, memory, wernicke, metabolic, model, training, evaluation, runner, baselines, data, config)
tests/                # 97 unit tests
experiments/          # 8 reproducible experiment directories
analysis/             # post-run analysis and winner promotion
docs/                 # design docs and research notes
```

## Test suite

```bash
.venv/bin/python -m pytest tests/ -v
```
