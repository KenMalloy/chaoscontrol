# Experiment 02: Critical Dynamics

## Hypothesis
Near-critical A parameterization improves loss over diagonal decay; oscillations emerge naturally in paired/full modes.

## Null hypothesis
No A-full variant beats A-diag.

## Predictions
- full_088 or full_092 will produce the best BPB among the critical variants
- paired mode will show oscillatory hidden state trajectories (visible in FFT)
- full_no_reg will drift away from criticality and perform worse than regularized variants
- full_095 may be unstable (too close to critical boundary)

## Method
Seven configs: diagonal baseline, paired mode, four full-matrix variants at different criticality targets (0.85, 0.88, 0.92, 0.95), and full without regularization.
All at model_dim=128, num_layers=4. Fixed 300s budget.
Logging: FFT of hidden state trajectories, Lyapunov exponent estimates.

## Dependencies
01 (for baseline comparison).

## Kill criteria
None standalone — criticality may only help in combination with other components. Results are diagnostic.
