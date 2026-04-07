# Experiment 01: Baseline

## Hypothesis
A simple transformer outperforms a vanilla SSM at matched parameter budgets, establishing the floor and ceiling.

## Null hypothesis
The transformer is strictly better at all sizes — this is expected and establishes the gap ChaosControl aims to close.

## Predictions
- Transformer outperforms SSM at all three sizes (small, medium, full)
- The gap widens at larger sizes where attention's expressiveness matters more
- SSM trains faster per step but converges to worse BPB

## Method
Six configs: SSM and transformer at three sizes (128, 256, 384 model_dim, all 4 layers).
Matched parameter budgets (~2MB, ~8MB, ~15MB). Fixed 300s training budget.
Measured: final BPB, parameter count, training steps completed, wall time.

## Dependencies
None — this is the first experiment.

## Kill criteria
None — these baselines are needed by all subsequent experiments.
