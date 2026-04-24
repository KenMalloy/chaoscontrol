# Criticality Distillation — Design

**Date:** 2026-04-24
**Status:** Draft. Replaces the actuator role ScOpt currently plays. Exp 25 entry.
**Supersedes design interaction with:** `docs/plans/2026-04-22-scarcity-optimizer-design.md` (ScOpt base), `docs/plans/2026-04-24-dual-timescale-rare-memory.md` (parked — the rare-gradient actuator path this design demotes).

## Thesis

ScOpt becomes a **teacher and instrument, not the main actuator.** Per-token pressure still identifies where rare signal is trying to flow, but it no longer shoves parameters directly. Instead, the SSM's own hidden-state history decides which channels actually preserved the rare signal, and a budgeted criticality control pulls those channels toward long-lived recurrence so ordinary CE + Muon can learn through the now-persistent traces.

One-liner for the paper:

> ScOpt does not teach the update direction. It teaches the model *where rare events occurred*; the SSM's own hidden-state history teaches *which channels preserved them*; criticality distillation pulls those channels toward longer memory.

Three roles, one thing each:

- **ScOpt** — event detector (sparse rare-event timestamps).
- **Recurrence state** — teacher (causal channel-survival evidence).
- **Criticality (log_a)** — actuator (budgeted long-memory allocation).

## Why we pivot from the direct actuator

Calibration sweep `results_scopt_cal_20260424T1200Z/` (2026-04-24) established that the rare-orthogonal actuator is structurally clipped, not miscalibrated:

| variant | rare_macro_cap_fired.median | effective_weight.median | fraction_positive |
|---|---:|---:|---:|
| step 1 base | 1.0 | 0.39 | 0.51 |
| baseline_d95 | 1.0 | 0.35 | 0.48 |
| pressure_c15 | 1.0 | 0.40 | 0.51 |
| rare_ema_095 | 1.0 | 0.30 | 0.51 |

All three knobs — baseline decay, pressure upper clamp, rare-EMA speed — failed to move cap_fired below 1.0 or effective_weight above ~0.4. The orth_norm/common_norm ratio is ~2.5× regardless, so the macro-Gerber cap is doing 100% of the stability work and the rare direction is neutered to 30–40% of intent. Pre-Gerber the same pathway diverged to final_loss 158. Between "clipped at 40%" and "diverges," there is no livable operating point for rare-orthogonal as a primary actuator.

Criticality Distillation dissolves the problem by not using the rare direction as an actuator at all.

## Algorithm

### Stage 1 — Sparse event detection

Pressure stops being a dense signed field and becomes a ranking:

```
pressure = scarcity_pressure_from_ce(...)                  # existing, unchanged
event_mask = pressure >= pressure.quantile(1 - event_frac) # top event_frac positions
```

Default `event_frac = 0.05`. This directly kills the `fraction_positive ≈ 0.5` problem — pressure-as-ranking doesn't care about the dense tail shape.

### Stage 2 — Rare Trace Bank (per-step aggregate, not per-event)

**Storage granularity matters.** At submission batch sizes (bs=512, seq=512, event_frac=0.05) a per-event bank would absorb ~13k events per layer per step and overwrite a fixed ring within one step. We do not want per-event identity; codex's "repeated events reinforce channel seats through fresh evidence" observation tells us the bank only needs aggregate, time-stamped evidence vectors.

**Bank entry = one `[D]` aggregate per (layer, step):**

```
bank_entry = {
    "step":   current_step,
    "layer":  l,
    "evidence":   aggregated_excess_energy_l,  # [D] mean over this step's event positions
    "event_count": n_events_this_step,          # for diagnostics and min-evidence gating
}
```

Aggregation inside Stage 3 is `mean_over_event_positions(future_excess_energy)`. If no events fire at layer `l` on step `t`, no entry is written (the step is invisible to the bank).

**Storage rule:** per-layer ring buffer keyed by step. Capacity = `trace_ttl_steps` (one slot per step, dropped on TTL). At `trace_ttl_steps = 1024` this is 1024 × D × fp32 per layer — a few MB per layer, not a few GB per step.

If we later want per-family evidence, we split `evidence` into `[n_buckets, D]` indexed by target frequency bucket. That's a v2 feature; first pass is a single `[D]` per (layer, step).

### Stage 3 — Post-event trace scoring

(Name note: not "causal" in the formal sense — `state_l[b, t+1:t+H]` includes contributions from *later input tokens* after the event, not only the event's own persistence. A matched-nearby or counterfactual control is required to claim causality. "Post-event trace scoring" is the honest label.)

**Recurrence-state capture API — required prerequisite.** Current SSM forward (`src/chaoscontrol/core.py:411-422`) computes the `[B, T, D]` state trajectory internally via `_diag_recurrence` and returns only `gate * states` → `out_proj(out)`. The existing ScOpt hooks (`experiments/23_fast_path/runner_fast_path.py:421`) capture projection inputs/outputs, not the recurrence states themselves. Criticality Distillation requires a new capture point:

- Modify `_forward_diag_scan` (and equivalents on other backends) to optionally return `states` — gated on a `capture_states: bool` param that defaults to False so production inference stays unchanged.
- A lightweight capture context manager on the model enables state return per-forward, gathers `states` into a per-layer buffer, and hands them to the Rare Trace Bank at the end of the step.
- Keep the capture buffer rank-local and short-lived; at end of step, Stage 3 consumes the `[B, T, D]` states, writes one `[D]` aggregate to the bank, and releases the raw states.

**Event mask timing.** Pressure is computed from CE after `model.encode` returns, so event positions are not known *during* the forward pass. Two viable resolutions:

- **(A) Two-pass within a step:** run the forward with state capture, compute CE + pressure, then accumulate excess energy over all positions and mask by event_mask *post-hoc*. This is cheap because excess-energy accumulation is O(B*T*D) per layer, well under the forward cost. The `states[B, T, D]` are already resident.
- **(B) One-step-lagged event mask:** apply step `t-1`'s pressure quantile to step `t`'s states. Saves the post-hoc masking but introduces a one-step lag; rare structure is slow-moving so this is likely fine, and it cleanly decouples capture from pressure.

Default to **(A)** for the first implementation because it keeps the event-state correspondence exact. Revisit (B) if the post-hoc pass turns out to be a measurable wall-clock hit.

**Scoring math:**

```
# For each position (b, t) in each layer l, over the trailing window [t+1, t+H]:
future_energy_l[b, t, c] = mean( states_l[b, t+1:t+H, c] ** 2 )   # [B, T, D]

# Baseline: running EMA of future_energy per channel, updated over non-event positions only.
baseline_future_energy_l[c] = ema_update( baseline, future_energy_l[~event_mask], decay=baseline_ema_decay )

# Excess per position:
excess_l[b, t, c] = relu(future_energy_l[b, t, c] - baseline_future_energy_l[c])

# Aggregate into the bank entry for this (layer, step):
bank_entry.evidence[c] = mean_over_event_positions( excess_l[event_mask, c] )   # [D]
```

The baseline is a cheap per-channel EMA of non-event future-energy. Matched-nearby control (paired non-event positions within the same sequence) is Ablation #3 — start with the EMA.

Aggregate score uses age-weighted evidence over the whole bank:

```
age_weight = exp( -(current_step - entry.step) / trace_half_life_steps )
score_l[c] = sum_entries( age_weight * entry.evidence[c] )
             / sum_entries( age_weight )
```

Evidence itself ages; seats do not age. See seat-relaxation section below.

### Stage 4 — Budgeted seat allocation (with evidence gate)

Every `seat_refresh_interval` steps, each layer recomputes its seat assignment from current score, **iff the evidence gate is satisfied**:

```
total_age_weighted_events_l = sum_entries( age_weight * entry.event_count )
if total_age_weighted_events_l < min_weighted_events_per_layer:
    # vacuous target — no gradient on log_a this refresh
    target_criticality_l = None
else:
    k = round(D * criticality_budget_frac)
    target_channels_l = topk(score_l, k)
    seat_mask_l = one_hot(target_channels_l, num_classes=D)   # [D] bool
    target_criticality_l = critical_value   # scalar applied only to seat channels
```

Defaults:
- `criticality_budget_frac = 0.15` (15% of channels critical per layer)
- `critical_value = 0.95` (target `1 - sigmoid(log_a) = 0.95`, near-critical)
- `min_weighted_events_per_layer = 256` (evidence floor before any seats bind)

Top-k is the first-pass allocator because it matches the "seats" semantics exactly. A soft Lagrange-multiplier variant (fix the mean criticality instead of the count) is an ablation, not the default.

### Stage 5 — Criticality actuator (masked, seat-only gradient)

A loss term, not an optimizer bias. This is the simplest and most composable choice: Muon just sees a gradient on `log_a`, and weight decay / fast-slow / param grouping all compose without special-casing.

**Critical subtlety:** non-seat channels must not feel the loss. Applying MSE over the full `[D]` vector with any default target actively pulls non-seats back to that target on every step, freezing ~85% of the recurrence spectrum. Mask by the seat selection:

```
criticality_l = 1.0 - sigmoid(log_a_l)                        # [D]

if target_criticality_l is None:
    loss_criticality_l = 0.0       # evidence gate not yet passed
else:
    seat_err   = (criticality_l - critical_value) ** 2         # [D]
    loss_criticality_l = (seat_err * seat_mask_l.float()).sum() / seat_mask_l.sum().clamp_min(1.0)

total_loss = ce_loss + criticality_distill_weight * sum_l(loss_criticality_l)
```

The mask means only the top-k seat channels receive criticality gradient; non-seats drift freely under whatever CE + Muon do to them. This matches the "no event owns a channel permanently" framing — a channel that loses its seat next refresh simply stops feeling the loss, it doesn't get yanked back to a default.

`critical_value` is applied directly as a scalar; there is no `default_value` because non-seats have no target. The `.detach()` on `seat_mask_l` and `target` is implicit — neither is differentiable with respect to anything by construction (they're top-k outputs over detached scores).

Default `criticality_distill_weight = 1e-3`. Sweep range 1e-4 to 1e-2 in ablations.

## Seat relaxation

Seats do not age. Evidence ages. Every refresh, the allocator re-reads the bank and assigns top-k from current age-weighted scores. This is the "diagnostic controller" framing: the bank is the sensor, the allocator is the controller, seats are the current decision.

Behavior under recurring events (worked example, `trace_half_life_steps = 256`):

- Event A at step `now - 1000`: weight ≈ 0.067
- Event A at step `now - 500`: weight ≈ 0.258
- Combined vote: ≈ 0.325

A one-off event from 1000 steps ago is nearly gone. A recurring rare pattern leaves a refreshed trail and keeps its seat by repeatedly earning it. Currently-active rare structure dominates allocation naturally.

This avoids the "one rare event holds a channel hostage forever" failure mode without introducing explicit event-identity tracking. If v2 wants per-family scoring, we can bucket by `target_bucket` at accumulation time.

## Defaults

```
event_frac                       = 0.05
trace_ttl_steps                  = 1024     # one bank slot per step, per layer
trace_half_life_steps            = 256
seat_refresh_interval            = 64       # steps
criticality_budget_frac          = 0.15
critical_value                   = 0.95     # target 1 - sigmoid(log_a) for seats
criticality_distill_weight       = 1e-3
baseline_ema_decay               = 0.99     # non-event future-energy EMA
min_weighted_events_per_layer    = 256      # evidence gate before seats bind
horizon_H                        = required sweep; initial values 16 / 64 / 256
```

## Diagnostics

Emit these into `scopt_trace_history` (or a new `criticality_distill_trace_history` field if it pollutes the ScOpt namespace):

| metric | healthy shape | read |
|---|---|---|
| `seat_churn` (fraction of seats changed per refresh) | high at warmup, decays to moderate | high-forever = noisy score; zero-too-early = stale bank or stuck budget |
| `budget_occupancy` (channels with criticality ≥ threshold / `budget_frac * D`) | ≈ 1.0 after warmup | < 1.0 = actuator isn't pulling hard enough; > 1.0 = overshoot |
| `score_criticality_corr` (rank correlation between score and final criticality per channel) | trends positive | near zero = distillation loss is not the dominant log_a gradient |
| `rare_bucket_ce` trajectory (per frequency bucket) | rare buckets improve faster than common | the one and only success metric |
| `event_rate` (events per step per layer) | ≈ `event_frac * batch * seq` | drops → pressure quantile broke |

Aggregate `final_loss` is dominated by common tokens and is **not** a valid success signal for this mechanism. `FrequencyBucketBaseline` already tracks per-bucket CE; we surface the rare-bucket trajectory as the primary success metric.

## Ablations (sequenced by information-per-cost)

1. **Horizon H** — values 16 / 64 / 256, fix everything else at defaults. Reads: does H couple to the fast/slow window? Our prior is H ≈ 2× slow-branch window.
2. **Per-layer vs cross-layer scoring** — score only the event's own layer, vs score all layers per event. Cross-layer is cheaper to motivate than to compute; if per-layer wins by ≥ 0.01 bpb on rare-bucket CE we ship per-layer and save the engineering.
3. **Baseline control — EMA vs matched-nearby** — EMA is the default; matched-nearby runs the same sequence's non-event positions as controls. Reads: does warm-up / phase-of-training contaminate the EMA control.
4. **Budget fraction** — 0.05 / 0.15 / 0.30. Reads: how much of the layer can be critical before stability suffers.
5. **Soft vs hard budget** — top-k vs Lagrange-multiplier soft budget. Only run if hard budget shows unhealthy churn.

The `horizon_H` and `per-layer vs cross-layer` ablations are blockers for committing a default; the others are refinements.

## Success metric and falsifier controls

**Primary metric (and only success signal):** rare-bucket CE on Param-Golf val improves relative to a matched ScOpt-as-telemetry-only control (criticality loss off, trace bank still collecting) AND relative to both falsifier controls below.

**Falsifier controls — first-class cells, not optional ablations:**

- **shuffled-teacher**: same trace bank, same budget, same scoring — but seat assignment uses a permuted `score_l` that destroys the channel identity. If rare-bucket CE improves in shuffled-teacher at the same rate as real teacher, the mechanism is "any budgeted criticality helps" and ScOpt-guided selection claims nothing.
- **budget-only**: no ScOpt guidance, no scoring, no bank. Uniformly set target for a random-but-fixed top-k channels per layer at startup, hold through training. Controls for "does having SOME critical modes help regardless of which ones."

Both are cheap variants (one config flag each) and devastating if they match treatment. They must ship in the first comparison matrix, not as v2.

Aggregate BPB is secondary; it's the submission number but not the mechanism test. If rare-bucket CE moves and aggregate doesn't, we have a real mechanism that needs scale; if rare-bucket CE also doesn't move, the hypothesis is falsified regardless of aggregate noise.

## Scope and sequencing

First-pass estimate (be skeptical; I tend to overcount):

| stage | effort | output |
|---|---|---|
| 1a. Recurrence-state capture API | 0.5 day | `_forward_diag_scan(capture_states=True)` + equivalents on chunked/tri backends, capture context mgr |
| 1b. Rare Trace Bank (per-step aggregate) + event-mask plumbing | 0.5 day | per-layer ring keyed by step, post-hoc event mask, unit tests |
| 2. Post-event trace scoring + age-weighted accumulator + baseline EMA | 0.5 day | `score_l` trajectory, baseline EMA over non-events |
| 3. Budget allocator (evidence-gated) + seat-masked criticality loss | 0.5 day | `loss_criticality`, target emission, integration with existing loss graph |
| 4. Diagnostics + per-bucket CE readout + shuffled-teacher / budget-only flags | 0.5 day | `seat_churn`, `rare_bucket_ce` per bucket over time, two falsifier-control config flags |
| 5. Smoke (treatment + telemetry-only + shuffled-teacher + budget-only, 4 cells) | 1 day | 1 × H100 smoke, mechanism read |
| 6. Ablation 1 (horizon_H 16/64/256) + writeup | 1 day | decision on default |
| 7. Ablation 2 (per-layer vs cross-layer) + writeup | 0.5 day | decision on default |

Realistically **4-5 working days to first mechanism read** (stages 1-5), then another 1-2 days of ablation before a default lands.

## Integration with existing code

- **Event detector** lives inside the existing activation-hook infrastructure in `_run_scopt_train_step` (`experiments/23_fast_path/runner_fast_path.py`). The hook already produces per-position pressure; switch to ranking + mask at capture time. Adds one tensor op per hook, no autograd change.
- **Trace bank** lives on the `ScarcityAwareOptimizer` instance as a per-layer ring of `[D]` fp32 tensors plus an int32 age buffer. Persisted in `state_dict` so it resumes from checkpoints.
- **Criticality loss** is composed in the runner's loss-accumulation block, alongside CE. `total_loss = ce + criticality_distill_weight * sum_layers(criticality_mse_l)`. Muon / AdamW / weight decay all see it as an extra gradient on `log_a` — no special-casing.
- **Config surface** adds the keys listed in Defaults above, all with docstring and sane fallbacks so existing configs continue to work with the actuator disabled.

## What ships first (inverted pyramid)

Minimum viable Exp 25 cell:
- Event detection + trace bank + scoring + top-k allocator + criticality MSE loss.
- `criticality_distill_weight = 0` as the default so the machinery can be merged and turned off.
- Control (`weight=0`) and treatment (`weight=1e-3`) as two cells of a calibration smoke.

If treatment shows `rare_bucket_ce` movement against control, proceed to horizon ablation. If not, falsify and publish the null.

## Explicitly parked

- Dual-timescale rare memory (`2026-04-24-dual-timescale-rare-memory.md`) — the rare-gradient EMA it would have sped up is demoted here. Not retired; if someone revives the direct actuator path, the dual-timescale design remains a cost reduction for it.
- Rare-orthogonal actuator (`rare_orthogonal_weight`) — default to 0, keep the code path for ablation, document that Criticality Distillation replaces its mechanism role.
- Soft-budget Lagrange formulation — parked until hard top-k is shown insufficient.

## Status

v2 draft. Not wired. Redline pass 2026-04-24 resolved: (1) per-step aggregate bank replaces per-event bank — at submission batch size per-event would overflow a fixed ring inside one step; (2) explicit recurrence-state capture API specified as a prerequisite — existing hooks do not expose `states`; (3) criticality loss is now seat-masked so non-seats drift freely; (4) evidence gate prevents allocation from junk; (5) shuffled-teacher and budget-only falsifier controls elevated to first-class cells; (6) Stage 3 renamed "post-event trace scoring" — matched-nearby control is required to claim causality; (7) `horizon_H` explicitly a required sweep parameter with initial values 16/64/256.

Next concrete action is approve-or-redline of this v2, then Stage 1 (state capture API + trace bank skeleton).
