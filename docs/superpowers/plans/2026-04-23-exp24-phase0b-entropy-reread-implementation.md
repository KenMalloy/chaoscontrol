# Exp 24 Phase 0b Entropy-Reread Implementation Plan Stub

This is a stub, not the implementation plan.

Preregistered thesis:
`experiments/24_training_time_bundle/PHASE0B_ENTROPY_REREAD.md`

Locked base:

- Decision doc: `experiments/24_training_time_bundle/PHASE0_BASE_LOCK.md`
- Config: `experiments/24_training_time_bundle/configs/exp24_base.yaml`

The preregistration is frozen. Do not weaken the legality contract, controls,
success thresholds, or kill criteria without adding a revision-log entry in the
Phase 0 base-lock record.

## Placeholder Task List

The follow-up plan author should use `superpowers:writing-plans` to expand this
stub into a full implementation plan. Expected task areas:

1. Build `src/chaoscontrol/eval_stream/entropy_reread.py` with pre-score
   chunk-boundary state blending.
2. Build the causal direction picker and legality tests proving current-chunk
   targets cannot affect current-chunk scoring.
3. Build the compute-matched control harness:
   `score_only`, `scheduled_reread_compute_matched`,
   `entropy_reread_shift0_compute_matched`, and
   `entropy_reread_bidirectional_blend`.
4. Build CLI `scripts/run_exp20_entropy_reread.py`.
5. Calibrate thresholds on a held-out stream only.
6. Run the 4 compute-matched arms x 3 seeds against the locked Phase 0 base.
7. Evaluate results against the success, ambiguous, and shelve criteria in the
   preregistration.

Budget placeholder: estimate in the follow-up plan after implementation shape,
state-caching strategy, and control harness are designed.
