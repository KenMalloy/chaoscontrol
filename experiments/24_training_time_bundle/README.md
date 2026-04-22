# Exp24 Training-Time Bundle

Exp24 tests training-time mechanisms on top of the current Exp23 fastest SSM
base. Evaluation remains the fixed post-training scorer; no eval-time TTT,
temporal heads, or scoring-time state changes belong in this bundle.

## Run Order

1. Ring 0 control: 2-3 seeds, 600s wall-clock, full validation after training.
2. Phase A sampling policy gate: no extra mechanism, same budget, compare to
   Ring 0 noise floor.
3. SemanticOptimizer overhead gate on 1xH100 before any 8xH100 semantic run.
4. First-wave mechanisms: fast/slow, spectral regularization, predictive
   auxiliary, scheduled Dreamworld hidden-state replay, and loss-triggered
   Dreamworld event replay.

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
