# Exp 22 - Temporal Heads for SSMs

**Date:** 2026-04-19
**Status:** Design - draft
**Assumption:** Exp 19, Exp 20, and Exp 21 are complete. Exp 22 starts from the final SSM checkpoint, final tokenizer/init decision, final scan path, and the legal eval-stream harness.

## Name

**Temporal head** is the load-bearing term for this experiment.

An attention head specializes *what to retrieve* from context. A temporal head specializes *how long to remember* the same incoming stream.

In implementation terms, a temporal head is a parallel recurrent trace that:

1. reuses the same SSM weights,
2. applies a different `delta_scale` / memory-horizon multiplier,
3. maintains its own recurrent state, and
4. contributes a separate log-probability view of the next token.

Parameter-free temporal heads add eval-time compute and recurrent state memory, but do not increase artifact size.

## Motivation

Exp 20 asks whether SSM-native test-time training can spend the eval budget productively. Exp 22 asks a narrower and cleaner question:

> Given a fixed eval-time budget, is it better for an SSM to spend extra compute by replaying the same horizon, or by evaluating several memory horizons in parallel?

This is the SSM-native analogue of attention heads. Transformers diversify retrieval. SSMs may need to diversify persistence.

The single-timescale bottleneck is the suspected failure mode. A standard SSM compresses the past through one learned time-horizon policy per channel. That may be too brittle under a fixed eval stream: some tokens want short memory, some want medium continuity, and some want slow belief persistence. Temporal heads test whether a frozen SSM already contains useful behavior at nearby horizons if we let multiple horizons vote.

## Core hypothesis

**Primary hypothesis:** A parameter-free multi-timeframe SSM improves bpb per extra eval second over both the best single timeframe and an equal-compute same-horizon recurrence control.

**Secondary hypotheses:**

1. **Temporal diversity matters.** Multi-timeframe heads beat the best single `delta_scale`, not merely the baseline `delta_scale=1.0`.
2. **Different horizons beat repeated horizon.** Multi-timeframe heads beat uniform depth recurrence or same-horizon extra compute at matched wall time.
3. **Conditional use matters.** Gated temporal heads improve bpb per second over always-on temporal heads when the slack budget is tight.
4. **Training can make it real.** If eval-only temporal heads help, train-time temporal-scale dropout should make the effect stronger and less out-of-distribution.

## Non-goals

- This is not a content-moderation or fact-checking experiment.
- This is not a new learned expert-SSM architecture in Phase 1.
- This is not a learned mixer experiment until the parameter-free controls are understood.
- This is not allowed to use current-chunk target loss to decide how to score that same chunk.

Small expert SSMs are deliberately deferred. Temporal heads are the safer first test because they reuse the final model and preserve artifact size.

## Mechanism

For a scale set such as:

```python
time_scales = [0.5, 1.0, 2.0]  # slow, base, fast if applied as reciprocal horizon multipliers
```

the eval loop carries one state bundle per scale:

```python
states_by_scale = {
    0.5: init_states(),
    1.0: init_states(),
    2.0: init_states(),
}
```

Each chunk is scored separately under each scale:

```python
for scale, states in states_by_scale.items():
    with DeltaModulator(model, delta_scale=scale):
        out = model(chunk, initial_states=states)
    logp_by_scale[scale] = log_softmax(out["logits"], dim=-1)
    states_by_scale[scale] = out["final_states"]
```

The first parameter-free mixer should be a uniform probability mixture, not a raw logits average:

```python
logp_mix = torch.logsumexp(
    torch.stack([
        logp_slow + math.log(1 / 3),
        logp_base + math.log(1 / 3),
        logp_fast + math.log(1 / 3),
    ]),
    dim=0,
)
```

This avoids treating logits from different horizons as calibrated in the same coordinate system. Raw-logit averaging may be tested later as an engineering variant, but it is not the primary scientific condition.

## Legality invariant

Exp 22 inherits the Exp 20 score-before-update rule. The extra wrinkle is routing.

The following are legal for deciding whether to spend extra compute on the current chunk:

- fixed always-on policy,
- previous-chunk loss,
- previous-chunk entropy,
- previous-chunk state norm / state delta,
- chunk index, doc boundary, and budget state,
- prefix-only online signals, if implemented causally.

The following are not legal for deciding how to score the current chunk:

- current-chunk target loss,
- current-chunk correctness,
- whole-current-chunk token statistics that influence earlier positions in the same chunk,
- an oracle per-token choice among temporal heads.

Current-chunk target loss may be logged and may schedule compute for future chunks or future documents. It must not change the score already being reported for the current chunk.

## Conditions

Phase A is the core experiment. It should run before any learned mixer or training change.

| Tag | Description | Purpose |
|---|---|---|
| `score_only` | Final checkpoint, normal eval, `delta_scale=1.0` | bpb and wall-clock floor |
| `single_delta_sweep` | Fixed single scales, e.g. `0.5`, `1.0`, `2.0`, `4.0` | find best single horizon |
| `uniform_depth_recur` | same-horizon extra compute via depth recurrence / repeated shared layers | equal-compute "more thinking" control |
| `temporal_heads_3_uniform` | slow/base/fast heads with uniform probability mixture | primary parameter-free temporal head test |
| `temporal_heads_5_uniform` | wider scale set if 3-head diversity is too low | checks whether scale coverage is the bottleneck |
| `gated_temporal_heads_3` | base path always, extra heads only when previous-chunk priority is high | tests bpb per slack second |
| `exp20_ttt_best` | best completed Exp 20 TTT policy under same slack accounting | strongest existing eval-time compute baseline |
| `temporal_heads_plus_ttt` | temporal heads first, TTT only if priority remains high | tests composition |

Phase B is only run if Phase A shows a non-trivial signal.

| Tag | Description | Purpose |
|---|---|---|
| `temporal_scale_dropout_train` | train with random per-batch or per-block `delta_scale` jitter, eval with temporal heads | reduces eval-only OOD risk |
| `tiny_mixer` | small frozen-checkpoint mixer trained on held-out calibration stream | measures learned composition without touching base weights |
| `tiny_ssm_sidecar` | small recurrent sidecar predicts logit deltas for high-priority chunks | deferred expert-SSM bridge |

## Statistical protocol

The primary unit is the checkpoint seed, not the document. If Exp 19-21 produce multiple final seeds, Exp 22 runs every condition on the same matched seeds and reports paired condition deltas.

```text
delta_seed_i = bpb_condition_seed_i - bpb_score_only_seed_i
```

Primary comparisons use paired deltas across seeds:

1. `temporal_heads_3_uniform` vs `score_only`,
2. `temporal_heads_3_uniform` vs best pre-registered single scale,
3. `temporal_heads_3_uniform` vs `uniform_depth_recur`,
4. `gated_temporal_heads_3` vs `temporal_heads_3_uniform` on bpb per extra second.

If only one final checkpoint exists, Exp 22 can still answer the engineering question, but it must not report seed-level p-values. Doc bootstrap confidence intervals may be included as diagnostics only; they do not replace independent checkpoint seeds.

Scale selection is also pre-registered. The primary scale set is `(0.5, 1.0, 2.0)`. Wider single-scale sweeps are exploratory unless they are chosen on a calibration stream and re-run, locked, on the primary eval stream.

## Evidence labels

Every Exp 22 result must be labeled before it is discussed in writeups or status summaries.

**Confirmatory-candidate** requires all of the following:

1. at least 5 matched checkpoint seeds,
2. fixed primary scale set `(0.5, 1.0, 2.0)`,
3. fixed gating threshold chosen off the primary eval stream,
4. full intended eval stream and budget regime,
5. passing legality tests,
6. paired seed-level analysis,
7. no learned mixer, tiny sidecar, or post-hoc oracle selection.

**Exploratory** is mandatory if any of the following are true:

- only one checkpoint seed is available,
- doc bootstrap is the only uncertainty estimate,
- scales or thresholds are chosen on the primary eval stream,
- the run uses a wider scale sweep not locked before primary eval,
- the run uses `tiny_mixer`, `tiny_ssm_sidecar`, or train-time temporal-scale dropout for the first time,
- the run uses a partial stream, degraded world size, or non-submission budget,
- any causal legality check is missing or not yet tested.

**Invalid** applies if current-chunk target-dependent information affects current-chunk scoring, or if artifact/budget accounting omits any temporal-head forward pass.

Seed-level p-values are secondary because `n=5` is small. The main report must show the paired seed deltas, mean delta, confidence interval, sign consistency, and bpb/sec. If p-values are reported, use paired tests and Holm correction across the four primary comparisons. Document-level bootstrap intervals may be included only as diagnostics.

## Compute policy

The first gated policy should be deliberately simple and pre-registered. It uses only information available before the chunk it affects.

```python
priority_next = (
    alpha * z(prev_entropy)
    + beta * z(prev_loss_spike)
    + gamma * z(prev_state_delta_norm)
    + delta * z(prev_head_disagreement)
)

if priority_next > threshold and slack_remaining > estimated_temporal_cost:
    action = "temporal_heads_3"
else:
    action = "base_only"
```

The threshold should be chosen on a calibration stream or smoke split, then frozen for the primary eval stream. If the threshold is tuned on the primary eval stream, the run must be labeled exploratory.

## Metrics

Primary:

```text
bpb_delta_per_extra_second = (bpb_score_only - bpb_condition) / extra_wall_seconds
```

Secondary:

- absolute bpb,
- extra wall seconds used,
- fraction of chunks receiving extra temporal heads,
- per-head bpb when evaluated alone,
- mixture bpb,
- head disagreement via pairwise KL or Jensen-Shannon divergence,
- selected-chunk vs non-selected-chunk gain,
- state norm drift per head,
- artifact bytes,
- peak VRAM,
- numerical failures / NaNs.

Useful diagnostic plots:

1. bpb improvement vs extra eval seconds,
2. head disagreement vs chunk priority,
3. slow/base/fast single-head bpb over doc position,
4. selected chunk fraction vs bpb gain,
5. temporal-head gain vs Exp 20 TTT gain at matched slack.

## Success criteria

**Promising:** `temporal_heads_3_uniform` beats `score_only` by at least `0.005` bpb and beats the best single scale or equal-compute recurrence on bpb per second.

**Strong:** `gated_temporal_heads_3` beats always-on temporal heads on bpb per second while using no more than 30% of chunks for extra heads.

**Thesis-level:** `temporal_heads_plus_ttt` beats the best Exp 20 TTT-only condition at the same total slack budget.

**Architecture-worthy:** train-time temporal-scale dropout improves the temporal-head gain or removes instability without increasing artifact size beyond the allowed budget.

## Kill criteria

- If the best single `delta_scale` matches or beats temporal heads at lower wall time, do not claim temporal heads. Report "single horizon retuning wins."
- If `uniform_depth_recur` matches or beats temporal heads at equal wall time, do not claim multiple timeframes. Report "extra same-horizon compute wins."
- If head disagreement is near zero across the eval stream, the chosen scales are redundant. Expand the scale set once, then park if still redundant.
- If temporal heads produce worse bpb than `score_only` by more than `0.005`, park eval-only temporal heads and only revisit under train-time scale dropout.
- If any gating policy uses current-chunk target-dependent information to affect current-chunk scoring, invalidate that run.
- If overhead consumes the slack needed for Exp 20 TTT and gives less bpb/sec than TTT, keep TTT as the eval-time compute path.

## Implementation sketch

Expected files:

- `src/chaoscontrol/eval_stream/temporal_heads.py`
- `scripts/run_exp22_temporal_heads.py`
- `tests/test_eval_stream_temporal_heads.py`
- `experiments/22_temporal_heads/README.md`
- `experiments/22_temporal_heads/configs/*.json`

Minimal API:

```python
@dataclass(frozen=True)
class TemporalHeadConfig:
    scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    mixer: Literal["uniform_logprob", "logit_mean"] = "uniform_logprob"
    policy: Literal["always", "previous_chunk_priority"] = "always"
    threshold: float | None = None
```

Runner contract:

```python
result = score_with_temporal_heads(
    model=model,
    streamer=streamer,
    head_config=head_config,
    budget=budget_tracker,
)
```

Load-bearing tests:

1. `scales=(1.0,)` matches `score_only` within float noise.
2. Uniform log-prob mixture with one head matches that head exactly.
3. Each head carries independent recurrent state.
4. Gating for chunk `k` cannot read target-dependent stats from chunk `k`.
5. Wall-clock accounting charges every temporal-head forward pass.
6. Artifact bytes are unchanged for parameter-free conditions.
7. `delta_scale=1.0` temporal path is bit-compatible with the existing Exp 20 `DeltaModulator` path.

## Risks

1. **Eval-only OOD.** The model was trained at one horizon, so alternative scales may be poorly calibrated. Mitigation: start parameter-free, then only run temporal-scale dropout training if Phase A shows signal.
2. **Causal leakage through routing.** Whole-chunk stats can accidentally leak future tokens into early-position scoring. Mitigation: always-on Phase A first; gated Phase A uses previous-chunk stats only.
3. **Miscalibrated mixtures.** Averaging logits can create false gains or losses. Mitigation: primary mixer is probability-space `logsumexp`.
4. **Compute tax.** Three heads roughly triple forward cost. Mitigation: compare by bpb per extra second and include gated variants.
5. **State memory.** Separate states per head increase VRAM. Mitigation: log peak VRAM; limit Phase A to 3 heads unless diversity is too low.
6. **Multiple comparisons.** Scale sweeps can overfit. Mitigation: lock primary scales before primary eval; treat wide sweeps as exploratory unless re-run.

## Decision rule

Exp 22 ships the term **Temporal Heads** only if the primary comparison is positive:

```text
multi-timeframe temporal heads > best single timeframe
and
multi-timeframe temporal heads > equal-compute same-horizon recurrence
```

If only the first inequality holds, the result is "multi-scale retuning helps." If only the second holds, the result is "ensembling horizons helps but scale choice is unresolved." If neither holds, temporal heads are a good name for a falsified idea, which is still useful.

## Condensed thesis

Attention heads diversify retrieval. Temporal heads diversify persistence.

Exp 22 tests whether SSMs can spend eval-time compute more effectively by running the same learned dynamics at several memory horizons, maintaining separate states, and mixing their probability forecasts under a fixed legal budget.
