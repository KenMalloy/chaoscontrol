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
| `temporal_heads` | Probability mixture over shifted horizons | exploratory until matched seeds/full stream |
| `identical_heads_uniform` | Multiple independent `log_a_shift=0.0` traces with uniform probability mixture | sanity control |
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

Run the score-only floor, the identical-head sanity control, the 3-head
temporal mixtures, and the same-horizon virtual-depth control on the same
checkpoint, tokenizer, stream, seed, and budget accounting.
`phaseA_identical_heads_uniform` should match `score_only` within float noise;
if it does not, the temporal-head runner is not a clean self-ensemble scaffold.
`phaseA_temporal_heads_3_uniform` is the clean scientific baseline.
`phaseA_temporal_heads_3_base_prior` is the engineering guardrail: it keeps
80% prior weight on the base horizon and 10% on each shifted horizon, so a
badly OOD shifted head cannot divide a correct base forecast by three.
`phaseB_temporal_heads_3_online_exp_weights` is exploratory: it updates mixture
weights after each realized token loss, so token `t` is always scored with
weights computed before seeing token `t`.

```bash
python scripts/run_exp22_temporal_heads.py \
  --config experiments/22_temporal_heads/configs/phaseA_score_only.json

python scripts/run_exp22_temporal_heads.py \
  --config experiments/22_temporal_heads/configs/phaseA_identical_heads_uniform.json

python scripts/run_exp22_temporal_heads.py \
  --config experiments/22_temporal_heads/configs/phaseA_temporal_heads_3_uniform.json

python scripts/run_exp22_temporal_heads.py \
  --config experiments/22_temporal_heads/configs/phaseA_temporal_heads_3_base_prior.json

python scripts/run_exp22_temporal_heads.py \
  --config experiments/22_temporal_heads/configs/phaseA_same_horizon_virtual_depth.json

python scripts/run_exp22_temporal_heads.py \
  --config experiments/22_temporal_heads/configs/phaseB_temporal_heads_3_online_exp_weights.json
```

Before running, replace every `TODO/...` path in the config files with the
actual final checkpoint, tokenizer, eval JSONL path, and result directory.

## Analysis Sidecar

Temporal-head configs may set `analysis_path` to write one analysis-only JSONL
record per scored document. These fields must not affect scoring or gating:

- `winner_counts_by_shift`: per-token argmin-NLL counts for each horizon.
- `half_life_stats_by_shift`: per-layer p10/median/p90 implied half-life and
  the fraction of channels separated from the base horizon by at least 0.5
  octaves.
- `state_divergence_by_shift`: per-layer L2 and cosine distance from the base
  horizon state.

The half-life diagnostic uses `ln(2) / (delta * sigmoid(log_a + shift))`, with
`delta` measured from the same chunk forward pass. If shifted half-life
histograms overlap heavily with the base horizon, treat temporal heads as
redundant even before interpreting bpb.

## Online Mixing

`online_exp_weights_logprob` is the causal exponential-weights mixer. It starts
from fixed initial weights, scores each token using the current weights, then
updates weights from that token's already-revealed per-head log loss:

```text
log_w_{t+1,k} = log_softmax(log_w_{t,k} - eta * nll_{t,k})
```

It is not the same as an oracle per-token head chooser. Any oracle or
future-token weighting result must be labeled as a leaky diagnostic, never as a
language-model score.

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

## Positioning

Do not claim that multi-timescale recurrence is new. The defensible claim is
narrower: Exp 22 tests an inference-time multi-horizon self-ensemble for a
frozen recurrent/SSM language model, implemented as same-checkpoint parallel
traces with independent recurrent states and causal probability mixing, with no
new learned mixer parameters in the Phase A conditions.
