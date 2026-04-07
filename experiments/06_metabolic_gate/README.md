# Experiment 06: Metabolic Gate

## Hypothesis
Generation+selection fork provides a small systematic advantage on high-surprise tokens. Structured projections ("choosing the question") outperform random noise.

## Null hypothesis
No gate variant beats the no-gate baseline.

## Predictions
- Any gated variant outperforms no_gate (fork-and-select helps on surprise tokens)
- memory_consistency scoring outperforms ensemble_agreement and loss_lookahead
- structured_proj outperforms noise variants (choosing the question beats random perturbation)
- Adaptive threshold learns to fork less frequently than fixed threshold
- The advantage is a compass, not a cannon: small per-step gain compounds over many steps

## Method
Eleven configs testing gate presence, K candidates, threshold, scoring method, noise level, and generation mechanism.
All use a_mode=full, rich_b_mode=hub, outer_model_dim=64.
model_dim=128, num_layers=4. Fixed 300s budget.
Critical config: structured_proj (NFT-aligned generation mechanism).
Logging: per-forked-step loss vs non-forked, fork rate, adaptive threshold trajectory.

## Dependencies
04, 05 (for memory and Wernicke config).

## Kill criteria
If no gate variant beats no_gate and the per-step advantage is not directionally consistent, the metabolic fork mechanism should be dropped.
