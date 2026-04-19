# Exp 22 - Temporal Heads

Attention heads diversify retrieval. Temporal heads diversify persistence.

Exp 22 tests whether a fixed SSM checkpoint can spend eval-time compute
better by carrying parallel recurrent states at multiple memory horizons,
then mixing their probability forecasts. The primary mechanism is
`log_a_shift`, not pre-softplus `delta_scale`.

## Runnable Conditions

`scripts/run_exp22_temporal_heads.py` currently supports:

| Condition | Meaning | Evidence label |
|---|---|---|
| `score_only` | Normal frozen checkpoint eval at `log_a_shift=0.0` | baseline |
| `single_horizon` | One shifted horizon, used for Phase 0 / best-single control | exploratory unless pre-registered |
| `temporal_heads` | Uniform probability mixture over shifted horizons | exploratory until matched seeds/full stream |
| `same_horizon_virtual_depth` | Equal-compute deterministic replay through shared SSM layers | control |

`gated_temporal_heads` intentionally fails fast in the runner until a
pre-registered previous-chunk gate is wired. We do not want an accidentally
always-on run labeled as gated.

## Phase 0 Pilot

Use the single-horizon template to sweep:

```text
log_a_shift in {-1.0, -0.5, 0.0, 0.5, 1.0}
```

The primary Phase A horizon set stays `[-0.5, 0.0, 0.5]` unless the pilot
shows severe OOD penalty or redundant horizons. If every non-base shift is
worse than base by more than `0.03` bpb on the calibration stream, Phase A may
still run, but should be labeled exploratory and Phase B temporal-scale
dropout becomes the real architecture test.

Example:

```bash
python scripts/run_exp22_temporal_heads.py \
  --config experiments/22_temporal_heads/configs/phase0_single_horizon_log_a_m050.json
```

Copy that config for the other pilot shifts, changing `horizon_shifts`,
`output_path`, and `summary_path`.

## Phase A

Run the score-only floor, the 3-head temporal mixture, and the same-horizon
virtual-depth control on the same checkpoint, tokenizer, stream, seed, and
budget accounting:

```bash
python scripts/run_exp22_temporal_heads.py \
  --config experiments/22_temporal_heads/configs/phaseA_score_only.json

python scripts/run_exp22_temporal_heads.py \
  --config experiments/22_temporal_heads/configs/phaseA_temporal_heads_3_uniform.json

python scripts/run_exp22_temporal_heads.py \
  --config experiments/22_temporal_heads/configs/phaseA_same_horizon_virtual_depth.json
```

Before running, replace every `TODO/...` path in the config files with the
actual final checkpoint, tokenizer, eval JSONL path, and result directory.

## Statistics

The primary unit is the checkpoint seed. If there are fewer than 3 matched
checkpoint seeds, results are exploratory. If there are 3 matched seeds, label
them provisional and include a pre-registered document bootstrap only as a
diagnostic. Seed-level p-values require matched seed deltas and should be
secondary to effect size, sign consistency, confidence intervals, and
`bpb_delta_per_extra_second`.

Do not claim "temporal heads" unless the multi-horizon run beats both:

1. the best pre-registered single horizon, and
2. the equal-compute same-horizon virtual-depth control.

If the best single horizon wins, report "single horizon retuning wins." If
same-horizon virtual depth wins, report "extra same-horizon compute wins."
