# Exp24 Training-Time Bundle

Exp24 tests training-time mechanisms on top of the current Exp23 fastest SSM
base. Evaluation remains the fixed post-training scorer; no eval-time TTT,
temporal heads, or scoring-time state changes belong in this bundle.

Current operational lock for optimizer work:

- `experiments/24_training_time_bundle/configs/exp24_base.yaml`
- `fast_slow_interval=64`
- `fast_slow_alpha=0.25`
- `fast_slow_eval_copy=slow`
- `dreamworld_enabled=false`
- `event_sleep_enabled=false`

Dreamworld and sleep remain follow-up mechanism lanes, but they are not part of
the locked static training base after the matched Phase 0 control.

## Status (2026-04-28)

Phase 0 trunk is locked at `phase0_fastslow_only_control`. The active
falsifier on top of that trunk is **CRCT v1** — Cache-Reweighted
Continuation Training in 3+1 mode, where rank 3 is the teacher/oracle
and train ranks never touch CRCT memory. See
`docs/crct-controller-architecture.md` for the architecture and
`build_crct_v1_matrix` in `exp24.py` for the live matrix
(`arm_a_fastslow_control` vs `arm_b_crct_controller`, 3 seeds each).

The 4×H100 phase 3 run is the calibrate-and-validate pass for CRCT;
the headline lives at 8×H100 (Phase 4 in
`docs/plans/2026-04-25-memory-aware-optimizer-plan.md`).

## Run Order

1. Ring 0 control: 2-3 seeds, 600s wall-clock, full validation after training.
2. Phase A sampling policy gate: no extra mechanism, same budget, compare to
   Ring 0 noise floor.
3. SemanticOptimizer overhead gate on 1xH100 before any 8xH100 semantic run.
4. First-wave mechanisms: fast/slow, spectral regularization, predictive
   auxiliary, scheduled Dreamworld hidden-state replay, and loss-triggered
   Dreamworld event replay.
5. Phase 3 falsifiers on the locked Phase 0 trunk — current path is
   `crct_v1` (rank-3 teacher, controller distillation, positive-only LM
   reweighting, gradient-conflict telemetry). Earlier phase-3 builders
   (`episodic_dw_curation_v1`, `episodic_controller_v1`, `episodic_ttt_v1`,
   `criticality_distillation_*`) remain in `exp24.py` as reproducible
   historical matrices.

## Dry Runs

```bash
.venv/bin/python experiments/24_training_time_bundle/run_exp24.py \
  --matrix ring0_control \
  --config experiments/23_fast_path/configs/base_seq_epoch_lr0064_full_corpus.yaml \
  --data-path baselines/parameter_golf/datasets/fineweb10B_sp16384 \
  --sp-model-path-16384 baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
  --output-dir /tmp/exp24_ring0_dry \
  --world-size 1 \
  --budget 5 \
  --dry-run
```

```bash
.venv/bin/python experiments/24_training_time_bundle/run_exp24.py \
  --matrix semantic_overhead_gate \
  --output-dir /tmp/exp24_semantic_gate_dry \
  --dry-run
```

```bash
.venv/bin/python experiments/24_training_time_bundle/run_exp24.py \
  --matrix first_wave \
  --seeds 1337 \
  --config experiments/23_fast_path/configs/base_seq_epoch_lr0064_full_corpus.yaml \
  --output-dir /tmp/exp24_first_wave_dry \
  --world-size 1 \
  --budget 5 \
  --dry-run
```

## Interpretation Rule

A mechanism whose BPB delta is smaller than the Ring 0 control sample standard
deviation is exploratory. It can motivate a follow-up; it is not a paper-quality
claim.

SGNS arms in this bundle answer the practical fast-path recipe question if they
are added later. They do not replace the historical Exp21
semantic-vs-distributional controls.
