# Experiment 05: Typed Composition (Wernicke)

## Hypothesis
Wernicke layer discovers meaningful types; typed compression preserves more than untyped; compression-consequence training signal produces useful type distinctions.

## Null hypothesis
No Wernicke variant beats the untyped baseline.

## Predictions
- Any Wernicke variant outperforms no_wernicke (typing helps recurrence)
- VQ routing outperforms MoE at matched K (simpler routing is enough)
- typed_both outperforms typed_episodic (typed consolidation adds value)
- compression_consequence produces more semantically coherent type buckets than standard VQ
- K=16 is the sweet spot (K=8 too few, K=32 fragmented)

## Method
Ten configs testing Wernicke layer presence, router type (VQ vs MoE), K_max, and typed storage/consolidation.
All use a_mode=full, rich_b_mode=hub, outer_model_dim=64, outer_model_type=multislot.
model_dim=128, num_layers=4. Fixed 300s budget.
Critical config: compression_consequence (novel training signal).
Logging: bucket utilization, sample inputs per bucket, loss delta after typed vs untyped merges.

## Dependencies
04 (for memory config).

## Kill criteria
If compression_consequence does not converge or produces degenerate buckets, the training signal needs redesign.
