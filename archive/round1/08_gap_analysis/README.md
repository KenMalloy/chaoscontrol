# Experiment 08: Gap Analysis

## Hypothesis
Test the weakest parts of the thesis explicitly.

## Null hypothesis
All gap-analysis configs confirm prior results (no surprises). This would mean the thesis has no weak points, which is unlikely.

## Predictions
- no_cue_proj degrades performance (cue projection is load-bearing scaffolding)
- compression_consequence converges and produces meaningful type distinctions
- dynamic_crit_per_layer shows layer-specific criticality targets emerge naturally
- semantic_emergence at 10x budget shows qualitative divergence between semantic and episodic content
- structured_vs_noise shows structured projections outperform isotropic noise
- survival_vs_random shows impact-based retention outperforms random

## Method
Six configs, each testing a specific weak claim in the thesis.
Several require code modifications not yet implemented (noted in config comments).
Base config is the full system from experiment 07.
Budget varies (300s default, 3000s for semantic_emergence).

## Dependencies
07 (full system as base config).

## Kill criteria
This is the most important experiment scientifically. Negative results here are as valuable as positive ones — they tell us which parts of the thesis to revise.
