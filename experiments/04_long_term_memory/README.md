# Experiment 04: Long-Term Memory

## Hypothesis
Two-tier memory (episodic + semantic) outperforms single-tier and no-memory. Consolidation from episodic to semantic produces qualitatively different knowledge representations.

## Null hypothesis
No memory variant beats the no-memory baseline.

## Predictions
- Episodic-only outperforms no-memory (surprise-driven storage helps)
- Semantic-only provides a smaller but consistent improvement (background bias)
- both_with_transfer outperforms both_no_transfer (consolidation does real work)
- Resolution trigger outperforms immediate and windowed triggers
- Pain-biased consolidation preserves more useful episodes than uniform

## Method
Ten configs testing memory tiers (none, episodic, semantic, both) and consolidation variants.
All use a_mode=full, rich_b_mode=hub (best from prior experiments).
model_dim=128, num_layers=4. Fixed 300s budget.
Key comparison: both_with_transfer vs both_no_transfer.

## Dependencies
02, 03 (for A mode and rich B selections).

## Kill criteria
If no memory variant beats no-memory, the outer model architecture needs fundamental rethinking.
