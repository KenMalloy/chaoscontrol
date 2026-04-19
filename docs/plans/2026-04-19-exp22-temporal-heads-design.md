# Exp 22 - Temporal Heads for SSMs

**Date:** 2026-04-19
**Status:** Design - draft
**Assumption:** Exp 19, Exp 20, and Exp 21 are complete. Exp 22 starts from the final SSM checkpoint, final tokenizer/init decision, final scan path, and the legal eval-stream harness.

## Name

**Temporal head** is the load-bearing term for this experiment.

An attention head specializes *what to retrieve* from context. A temporal head specializes *how long to remember* the same incoming stream.

In implementation terms, a temporal head is a parallel recurrent trace that:

1. reuses the same SSM weights,
2. applies a different memory-horizon knob,
3. maintains its own recurrent state, and
4. contributes a separate log-probability view of the next token.

Parameter-free temporal heads add eval-time compute and recurrent state memory, but do not increase artifact size.

Novelty language should stay narrow. Multi-timescale recurrence, shared-weight
branches, and probability-space ensembling all have prior art. The defensible
claim is the composition: a frozen-checkpoint, same-model, separate-state,
multi-horizon recurrent/SSM self-ensemble with causal probability mixing and no
new learned mixer parameters in the Phase A conditions.

The primary horizon knob is `log_a_shift`, not the current pre-softplus `delta_scale` hook. `log_a_shift` directly changes the SSM pole term:

```python
a_base = sigmoid(log_a + shift)
decay = exp(-delta * a_base)
```

Negative shifts reduce `a_base` and lengthen memory; positive shifts increase `a_base` and shorten memory. Pre-softplus `delta_scale` remains a diagnostic or fallback knob because the current implementation scales `delta_proj(x)` before `softplus`, which can be asymmetric and weak near zero.

## Motivation

Exp 20 asks whether SSM-native test-time training can spend the eval budget productively. Exp 22 asks a narrower and cleaner question:

> Given a fixed eval-time budget, is it better for an SSM to spend extra compute by replaying the same horizon, or by evaluating several memory horizons in parallel?

This is the SSM-native analogue of attention heads. Transformers diversify retrieval. SSMs may need to diversify persistence.

The single-timescale bottleneck is the suspected failure mode. A standard SSM compresses the past through one learned time-horizon policy per channel. That may be too brittle under a fixed eval stream: some tokens want short memory, some want medium continuity, and some want slow belief persistence. Temporal heads test whether a frozen SSM already contains useful behavior at nearby horizons if we let multiple horizons vote.

## Core hypothesis

**Primary hypothesis:** A parameter-free multi-timeframe SSM improves bpb per extra eval second over both the best single horizon and an equal-compute same-horizon recurrence control.

**Secondary hypotheses:**

1. **Temporal diversity matters.** Multi-timeframe heads beat the best single horizon knob, not merely the baseline `log_a_shift=0.0`.
2. **Different horizons beat repeated horizon.** Multi-timeframe heads beat the pinned same-horizon virtual-depth control at matched wall time.
3. **Conditional use matters.** Gated temporal heads improve bpb per second over always-on temporal heads when the slack budget is tight.
4. **Training can make it real.** If eval-only temporal heads help, train-time temporal-scale dropout should make the effect stronger and less out-of-distribution.

## Non-goals

- This is not a content-moderation or fact-checking experiment.
- This is not a new learned expert-SSM architecture in Phase 1.
- This is not a learned mixer experiment until the parameter-free controls are understood.
- This is not allowed to use current-chunk target loss to decide how to score that same chunk.

Small expert SSMs are deliberately deferred. Temporal heads are the safer first test because they reuse the final model and preserve artifact size.

## Mechanism

For a shift set such as:

```python
horizon_shifts = [-0.5, 0.0, 0.5]  # slow, base, fast via log_a_shift
```

the eval loop carries one state bundle per shift:

```python
states_by_shift = {
    -0.5: init_states(),
    0.0: init_states(),
    0.5: init_states(),
}
```

Each chunk is scored separately under each shift:

```python
for shift, states in states_by_shift.items():
    with DeltaModulator(model, log_a_shift=shift):
        out = model(chunk, initial_states=states)
    logp_by_shift[shift] = log_softmax(out["logits"], dim=-1)
    states_by_shift[shift] = out["final_states"]
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

Phase 0 is a cheap calibration gate. It does not produce confirmatory claims.

| Tag | Description | Purpose |
|---|---|---|
| `horizon_response_pilot` | 10% calibration stream sweep of `log_a_shift in {-1.0, -0.5, 0.0, 0.5, 1.0}` plus current pre-softplus `delta_scale in {0.5, 1.0, 2.0}` | verify the horizon knob moves bpb/state norms monotonically enough to lock Phase A |

If every non-base `log_a_shift` costs more than `0.03` bpb on the pilot, Phase A still may run, but it is labeled exploratory unless Phase B train-time temporal-scale dropout is run. If pre-softplus `delta_scale` is noisy or nonmonotone in the pilot, it is not used as the primary temporal-head knob.

Phase A is the core experiment. It should run before any learned mixer or training change.

| Tag | Description | Purpose |
|---|---|---|
| `score_only` | Final checkpoint, normal eval, `log_a_shift=0.0` | bpb and wall-clock floor |
| `identical_heads_uniform` | three independent heads all pinned to `log_a_shift=0.0`, mixed uniformly | sanity check that same-checkpoint self-ensemble scaffolding is equivalent to the base predictor when horizons are identical |
| `single_horizon_sweep` | Fixed single shifts, pre-registered from pilot | find best single horizon |
| `same_horizon_virtual_depth` | deterministic replay of the same SSM layer group at `log_a_shift=0.0`; see control definition below | equal-compute "more thinking" control |
| `temporal_heads_3_uniform` | slow/base/fast heads with uniform probability mixture | primary parameter-free temporal head test |
| `temporal_heads_5_uniform` | wider shift set if 3-head diversity is too low | checks whether horizon coverage is the bottleneck |
| `gated_temporal_heads_3` | base path always, extra heads only when previous-chunk priority is high | tests bpb per slack second |
| `exp20_ttt_best` | best completed Exp 20 TTT policy under same slack accounting | strongest existing eval-time compute baseline |
| `temporal_heads_plus_ttt` | temporal heads first, TTT only if priority remains high | tests composition |

### Equal-compute same-horizon control

`same_horizon_virtual_depth` is pinned before implementation:

1. Load the same checkpoint weights.
2. Keep the horizon knob at `log_a_shift=0.0`.
3. Use one recurrent state bundle, not one state bundle per replay.
4. Replay the same contiguous SSM layer group through `ChaosStudentLM.depth_recurrence_shared_layers`.
5. Choose `depth_recurrence_count` by wall-clock matching against `temporal_heads_3_uniform`, not by nominal pass count.

If the final Exp 19 checkpoint was trained with an explicit depth-recurrence group, use that group. If it was not, the primary control uses all SSM blocks as the shared group. This control is deterministic. It is not a random-seed rescore, not an ensemble of identical base passes, and not a learned expert.

Phase B is run if Phase A shows a non-trivial signal, or if Phase A is null while the Phase 0 pilot suggests non-base horizons are suffering a large eval-only OOD penalty.

| Tag | Description | Purpose |
|---|---|---|
| `temporal_scale_dropout_train` | train with random per-batch or per-block horizon shifts, eval with temporal heads | reduces eval-only OOD risk |
| `tiny_mixer` | small frozen-checkpoint mixer trained on held-out calibration stream | measures learned composition without touching base weights |
| `tiny_ssm_sidecar` | small recurrent sidecar predicts logit deltas for high-priority chunks | deferred expert-SSM bridge |

## Statistical protocol

The primary unit is the checkpoint seed, not the document. If Exp 19-21 produce multiple final seeds, Exp 22 runs every condition on the same matched seeds and reports paired condition deltas.

```text
delta_seed_i = bpb_condition_seed_i - bpb_score_only_seed_i
```

Primary comparisons use paired deltas across seeds:

1. `temporal_heads_3_uniform` vs `score_only`,
2. `temporal_heads_3_uniform` vs best pre-registered single horizon,
3. `temporal_heads_3_uniform` vs `same_horizon_virtual_depth`,
4. `gated_temporal_heads_3` vs `temporal_heads_3_uniform` on bpb per extra second.

Before interpreting those comparisons, `identical_heads_uniform` must match
`score_only` within float noise. A failure there means the runner or state
threading is adding behavior beyond the declared horizon mechanism.

If only one final checkpoint exists, Exp 22 can still answer the engineering question, but it must not report seed-level p-values. Doc bootstrap confidence intervals may be included as diagnostics only; they do not replace independent checkpoint seeds.

Horizon selection is also pre-registered. The primary shift set is `(-0.5, 0.0, 0.5)` unless the Phase 0 calibration stream rejects it. Wider horizon sweeps are exploratory unless they are chosen on a calibration stream and re-run, locked, on the primary eval stream.

## Evidence labels

Every Exp 22 result must be labeled before it is discussed in writeups or status summaries.

**Confirmatory-candidate** requires all of the following:

1. at least 5 matched checkpoint seeds,
2. fixed primary horizon set,
3. fixed gating threshold chosen off the primary eval stream,
4. full intended eval stream and budget regime,
5. passing legality tests,
6. paired seed-level analysis,
7. no learned mixer, tiny sidecar, or post-hoc oracle selection.

**Provisional** is allowed when only 3 matched checkpoint seeds are feasible after the final submission checkpoint is chosen. Provisional results require the same fixed horizon set, full eval stream, legal accounting, paired seed deltas, and a pre-registered block bootstrap over documents as a diagnostic. Provisional results may guide engineering decisions, but they are not thesis-level and must not be presented as confirmatory evidence.

**Exploratory** is mandatory if any of the following are true:

- fewer than 3 matched checkpoint seeds are available,
- doc bootstrap is the only uncertainty estimate,
- horizons or thresholds are chosen on the primary eval stream,
- the run uses a wider horizon sweep not locked before primary eval,
- the run uses `tiny_mixer`, `tiny_ssm_sidecar`, or train-time temporal-scale dropout for the first time,
- the run uses a partial stream, degraded world size, or non-submission budget,
- any causal legality check is missing or not yet tested.

**Invalid** applies if current-chunk target-dependent information affects current-chunk scoring, or if artifact/budget accounting omits any temporal-head forward pass.

Seed-level p-values are secondary because `n=5` is small. The main report must show the paired seed deltas, mean delta, confidence interval, sign consistency, and bpb/sec. If p-values are reported, use paired tests and Holm correction across the four primary comparisons. Document-level bootstrap intervals may be included only as diagnostics.

## Compute policy

The first gated policy should be deliberately simple and pre-registered. It uses only information available before the chunk it affects. The primary gate does **not** use `prev_head_disagreement`, because disagreement is only observed after temporal heads have run and would become stale after base-only chunks.

```python
priority_next = (
    alpha * z(prev_entropy)
    + beta * z(prev_loss_spike)
    + gamma * z(prev_state_delta_norm)
)

if priority_next > threshold and slack_remaining > estimated_temporal_cost:
    action = "temporal_heads_3"
else:
    action = "base_only"
```

A secondary gated policy may add head disagreement through an exponentially decayed cache:

```python
if temporal_heads_ran:
    disagreement_ema = update_ema(disagreement_ema, current_head_disagreement)
else:
    disagreement_ema = decay_toward_prior(disagreement_ema)
```

That secondary policy is exploratory until it is frozen on a calibration stream and re-run on the primary stream. The threshold should be chosen on a calibration stream or smoke split, then frozen for the primary eval stream. If the threshold is tuned on the primary eval stream, the run must be labeled exploratory.

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

**Promising:** `temporal_heads_3_uniform` beats `score_only` by at least `0.005` bpb and beats the best single horizon or equal-compute recurrence on bpb per second.

**Strong:** `gated_temporal_heads_3` beats always-on temporal heads on bpb per second while using no more than 30% of chunks for extra heads.

**Thesis-level:** `temporal_heads_plus_ttt` beats the best Exp 20 TTT-only condition at the same total slack budget.

**Architecture-worthy:** train-time temporal-scale dropout improves the temporal-head gain or removes instability without increasing artifact size beyond the allowed budget.

**Phase B promotion rule:** A Phase A null does not falsify temporal heads as an architecture idea if the pilot shows large single-horizon OOD penalties. In that case, promote `temporal_scale_dropout_train` to mandatory before killing the line. Eval-only temporal heads and train-aware temporal heads are separate claims.

## Kill criteria

- If the best single horizon matches or beats temporal heads at lower wall time, do not claim temporal heads. Report "single horizon retuning wins."
- If `same_horizon_virtual_depth` matches or beats temporal heads at equal wall time, do not claim multiple timeframes. Report "extra same-horizon compute wins."
- If head disagreement is near zero across the eval stream, the chosen horizons are redundant. Expand the horizon set once on the calibration stream, then park if still redundant.
- If eval-only temporal heads produce worse bpb than `score_only` by more than `0.005`, park eval-only temporal heads and run Phase B only if the horizon-response pilot suggests OOD is the likely cause.
- If any gating policy uses current-chunk target-dependent information to affect current-chunk scoring, invalidate that run.
- If overhead consumes the slack needed for Exp 20 TTT and gives less bpb/sec than TTT, keep TTT as the eval-time compute path.
- If temporal-scale dropout training also fails to beat the same-horizon and best-single-horizon controls, kill the temporal-head architecture claim.

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
    horizon_shifts: tuple[float, ...] = (-0.5, 0.0, 0.5)
    head_ids: tuple[str, ...] | None = None  # required when shifts repeat
    horizon_knob: Literal["log_a_shift", "delta_post_scale", "delta_pre_scale"] = "log_a_shift"
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

1. `horizon_shifts=(0.0,)` matches `score_only` within float noise.
2. Uniform log-prob mixture with one head matches that head exactly.
3. Each head carries independent recurrent state.
4. Repeated `horizon_shifts=(0.0, 0.0, 0.0)` with explicit `head_ids` matches `score_only` within float noise while preserving independent state objects.
5. Gating for chunk `k` cannot read target-dependent stats from chunk `k`.
6. Wall-clock accounting charges every temporal-head forward pass.
7. Artifact bytes are unchanged for parameter-free conditions.
8. `log_a_shift=0.0` temporal path is bit-compatible with the existing Exp 20 `DeltaModulator` path.
9. `same_horizon_virtual_depth` uses one state bundle and a deterministic virtual-layer replay; it does not instantiate multiple identical temporal-head states.
10. Gating after a base-only chunk ignores head disagreement or uses only the pre-registered decayed cache.

## Risks

1. **Eval-only OOD.** The model was trained at one horizon, so alternative horizon shifts may be poorly calibrated. Mitigation: run the Phase 0 horizon-response pilot first; if OOD dominates, promote temporal-scale dropout before killing the architecture idea.
2. **Causal leakage through routing.** Whole-chunk stats can accidentally leak future tokens into early-position scoring. Mitigation: always-on Phase A first; gated Phase A uses previous-chunk stats only.
3. **Miscalibrated mixtures.** Averaging logits can create false gains or losses. Mitigation: primary mixer is probability-space `logsumexp`.
4. **Compute tax.** Three heads roughly triple forward cost. Mitigation: compare by bpb per extra second and include gated variants.
5. **State memory.** Separate states per head increase VRAM. Mitigation: log peak VRAM; limit Phase A to 3 heads unless diversity is too low.
6. **Multiple comparisons.** Horizon sweeps can overfit. Mitigation: lock primary horizons before primary eval; treat wide sweeps as exploratory unless re-run.

## Decision rule

Exp 22 ships the term **Temporal Heads** only if the primary comparison is positive:

```text
multi-timeframe temporal heads > best single horizon
and
multi-timeframe temporal heads > equal-compute same-horizon recurrence
```

If only the first inequality holds, the result is "multi-horizon retuning helps." If only the second holds, the result is "ensembling horizons helps but horizon choice is unresolved." If neither holds, temporal heads are a good name for a falsified idea, which is still useful.

## Condensed thesis

Attention heads diversify retrieval. Temporal heads diversify persistence.

Exp 22 tests whether SSMs can spend eval-time compute more effectively by running the same learned dynamics at several memory horizons, maintaining separate states, and mixing their probability forecasts under a fixed legal budget.
