# Learned Controller Action Space

**Status:** Design note for the controller-learning-gates branch.

**Thesis:** the stable endpoint is not a learned controller plus a permanent
hand-written governor. The stable endpoint is a learned event SSM whose action
heads are parameterized so most safety constraints are intrinsic to the action
space. A small invariant layer remains for physical impossibilities and
competition legality, and every intervention is logged.

## Frame

The controller should learn memory behavior directly:

```text
WRITE_EVENT / QUERY_EVENT / REPLAY_OUTCOME stream
        -> shared CPU SSM hidden state
        -> bounded action heads
        -> invariant checks
        -> accepted action + trace row
```

The old "governor" language is useful only if it means action-space geometry,
not a separate non-learned policy that constantly overrides a chaotic learner.
The learned system should express:

- which events deserve writes;
- which slots deserve protection or eviction;
- which memories deserve replay;
- when eval-time correction is worth spending;
- how exploration temperature, entropy pressure, and consolidation strength
  should move.

But it should express them in coordinates that cannot produce invalid physics.

## Design Rule

Learn preferences in bounded coordinates. Keep only hard invariants outside the
learner.

Examples:

```python
write_rate = min_write + sigmoid(z_write) * (max_write - min_write)
replay_budget = max_replay_tokens * sigmoid(z_replay)
ema_alpha = alpha_max * sigmoid(z_alpha)
entropy_beta = beta_max * sigmoid(z_entropy)
temperature = t_min + softplus(z_temperature)
```

The clamp is not a post-hoc patch. It is the coordinate system.

## Controller Shape

Use one shared recurrent controller unless profiling or learning dynamics argue
for separate controllers:

```text
Shared event SSM
  tracks cache pressure, reward drift, replay quality, entropy collapse,
  Gerber acceptance, bucket health, and wall-clock pressure.

Heads
  write_admission_head
  eviction_head
  replay_selection_head
  replay_timing_head
  simplex_selection_head
  eval_correction_gate_head
  budget_temperature_head
  consolidation_head
```

Per-head SSMs are a reasonable later specialization if interference appears:
write admission and eval correction may want different timescales. The first
implementation should prefer a shared SSM plus typed heads so all decisions see
the same cache health and reward history.

## Ordered Learning Without A Permanent Governor

Ordering should come from eligibility signals, not a static external governor.
Each head has an eligibility multiplier derived from measured readiness:

```text
write_head_eligible =
  finite_write_reward_rate high enough
  and admitted slots can be joined to replay outcomes
  and cache churn is bounded

replay_head_eligible =
  credited_actions > threshold
  and finite_reward_rate high enough
  and Gerber accept rate sane

eval_correction_eligible =
  legal eval-time reward path exists
  and correction budget accounting is exact
  and corrected spans improve without poisoning uncorrected spans
```

The action head can exist from the beginning, but its learning rate or action
amplitude is multiplied by eligibility. That gives ordered learning without
creating a second policy.

```python
proposal = head(ssm_state, event_features)
amplitude = readiness_ema.clamp(0.0, 1.0)
action = heuristic_anchor + amplitude * bounded_delta(proposal)
```

Early behavior stays close to the heuristic teacher. As evidence arrives, the
learned head earns control.

## Runaway Handling

Runaways can be stopped, but stopping them must produce telemetry. A stop that
is not logged is a hidden experimental confound.

Hard invariant stops:

- replay reward is NaN or missing;
- attribution key mismatch;
- event schema incomplete;
- eval correction would violate scoring legality;
- replay/correction/write request exceeds absolute wall-clock or token budget;
- candidate refers to a missing or sentinel-padded slot;
- learned budget moves faster than the allowed EMA rate.

Each intervention emits a trace row:

```text
gpu_step
event_type
head_name
raw_action
bounded_action
invariant_name
clamp_amount
readiness
reward_context
accepted
```

This turns safety into signal. If a head constantly hits a bound, that is not
only protection; it is evidence that the action parameterization, reward, or
budget range is wrong.

## What Should Become Learned

### Write Admission

Current behavior is pressure-times-CE ranking with a configured write rate.
Make the write head learn admission scores, but keep writes inside a bounded
rate coordinate.

First mature form:

```text
admit_score = write_head(event_features)
admit_count = bounded write budget
accepted = top admit_count by admit_score
```

Later mature form:

```text
admit_probability = sigmoid(write_head(...))
write_budget = bounded slow head
```

Never start by letting the model freely choose total write volume.

### Eviction

Eviction should become a learned protection/eviction score. Keep hard
protection for fresh slots and slots with unresolved replay credit.

Reward attribution: penalize evicting a slot that later would have produced
positive replay reward; reward evicting stale slots that repeatedly fail to pay
off.

### Replay Selection And Timing

Simplex selection is already the strongest learned piece. Extend the same
credit machinery to replay timing:

```text
selection head: which slot/candidate?
timing head: now or later?
budget head: how many replay tokens this window?
```

Replay timing is dangerous because it can consume wall-clock. Keep the timing
head in a token-budget coordinate and log every budget saturation.

### Eval-Time Correction

Eval correction should be a learned abstention policy, not a reflex.

```text
correct_probability = sigmoid(correction_head(...))
correction_fraction = bounded slow budget
accepted = top correction_fraction uncertain spans
```

Primary reward is BPB improvement on corrected spans with an explicit penalty
for unnecessary correction. Most spans should remain untouched.

### Consolidation / Fast-Slow

Fast-slow should move from fixed interval/alpha to bounded blend decisions:

```text
blend_alpha = alpha_max * sigmoid(alpha_head(...))
blend_now = sigmoid(blend_gate(...))
```

Hard invariant: no high-frequency thrash. Blend amplitude and cadence should
move on a slow EMA timescale.

### Exploration Knobs

Entropy beta, temperature, HxH residual strength, and Gerber tolerance should
be learned as bounded head outputs once the corresponding telemetry is stable.

These are not primary behaviors; they are meta-actions. They should learn
slower than selection/admission heads.

## Implementation Sequence

1. **Trace-only heads.** Add head outputs to telemetry without changing actions.
2. **Heuristic-anchored deltas.** Let heads perturb heuristic scores inside a
   small bounded delta.
3. **Full ranking control.** Let heads rank candidates while total budgets
   remain fixed.
4. **Bounded budget heads.** Let heads move write/replay/correction budgets
   through slow EMA coordinates.
5. **Meta-knob heads.** Learn entropy, temperature, Gerber tolerance, and
   consolidation strength only after primary decisions are stable.

This keeps the scientific ordering:

```text
learn ranking before learning rate
learn local action before learning global budget
learn behavior before learning meta-behavior
```

## Tests To Pin

- Head eligibility at zero must reproduce the heuristic action exactly.
- Bounded head outputs must never exceed documented min/max ranges.
- Every clamp or invariant stop must increment telemetry and write a trace row.
- A NaN reward path must block learning but not crash the run.
- Sentinel-padded slots must never receive replay credit or eviction reward.
- Fixed seed sampling must stay deterministic under learned bounded deltas.
- Budget heads must move only through the configured EMA rate.
- Trace-only mode must be bit-identical to the pre-head heuristic path.

## Success Criteria

Short-term success is not immediate BPB lift. Short-term success is a controller
that can be allowed to learn more behavior without making a run uninterpretable.

Healthy telemetry:

- nonzero credited actions;
- finite reward rate near 1.0 on OK outcomes;
- clamp events present but not saturating;
- entropy not collapsed unless reward justifies it;
- cache churn bounded;
- replay token budget respected;
- correction fraction respected;
- learned heads beat or match heuristic ranking before budget learning turns on.

## Non-Goals

- No direct rare-gradient actuator.
- No unconstrained learned replay rate.
- No unlogged safety override.
- No eval-time correction path that can violate the scoring contract.
- No permanent external policy that hides the behavior of the learned SSM.
