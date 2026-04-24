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

### Stage 2 — Rare Trace Bank

Per-layer rolling database of post-event recurrence traces. On event positions only, we snapshot the hidden state and its near future:

```
event = {
    "step":    current_step,
    "layer":   l,
    "future_excess_energy": excess_l,   # [D], computed once per event
    "pressure":             pressure[b, t],
    "target_bucket":        freq_bucket[target[b, t]],  # reserved for v2 scoring
}
```

`excess_l` is the event contribution computed at capture time (see Stage 3), so the bank stores a small `[D]` vector per event, not `[H, D]`. Horizon `H` appears only inside the excess-energy computation and is released immediately.

**Storage rule:** fixed-size ring buffer per layer, capacity `trace_bank_capacity` events. TTL-based drop: any event older than `trace_ttl_steps` is evicted regardless of capacity. Combined with the age-weighted scoring below, the bank behaves as a rolling recency-weighted database.

### Stage 3 — Causal channel scoring

Channel score = excess future energy after an event, vs. a non-event baseline:

```
event_future_energy[c]    = mean( state_l[b, t+1:t+H][:, c] ** 2 )   # [D]
baseline_future_energy[c] = running EMA of mean-square state per channel during non-events
excess_l[c]               = relu(event_future_energy[c] - baseline_future_energy[c])
```

The baseline is a cheap per-channel EMA of non-event future-energy, updated every step at non-event positions. A matched-nearby control (paired non-event positions within the same sequence) is a v2 ablation; start with the EMA.

Aggregate score uses age-weighted evidence over the whole bank:

```
age_weight = exp( -(current_step - event.step) / trace_half_life_steps )
score_l[c] = sum_events( age_weight * event.future_excess_energy[c] )
             / sum_events( age_weight )
```

Events themselves age; seats do not age. See seat-relaxation section below.

### Stage 4 — Budgeted seat allocation

Every `seat_refresh_interval` steps, each layer recomputes its seat assignment from current score:

```
k = round(D * criticality_budget_frac)
target_channels_l = topk(score_l, k)
target_criticality_l = where(channel in target, critical_value, default_value)
```

Defaults:
- `criticality_budget_frac = 0.15` (15% of channels critical per layer)
- `critical_value = 0.95` (target `1 - sigmoid(log_a) = 0.95`, near-critical)
- `default_value = 1 - sigmoid(log_a_init)` (leave non-seat channels alone at their current initialization-driven setpoint)

Top-k is the first-pass allocator because it matches the "seats" semantics exactly. A soft Lagrange-multiplier variant (fix the mean criticality instead of the count) is an ablation, not the default.

### Stage 5 — Criticality actuator

A loss term, not an optimizer bias. This is the simplest and most composable choice: Muon just sees a gradient on `log_a`, and weight decay / fast-slow / param grouping all compose without special-casing.

```
criticality_l = 1.0 - sigmoid(log_a_l)                  # [D]
loss_criticality_l = mse(criticality_l, target_criticality_l.detach())
total_loss = ce_loss + criticality_distill_weight * sum_l(loss_criticality_l)
```

`target_criticality_l.detach()` ensures the actuator is a setpoint, not a differentiable signal that rare pressure can exploit. The `.detach()` is load-bearing.

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
trace_bank_capacity              = 512      # events per layer
trace_ttl_steps                  = 1024
trace_half_life_steps            = 256
seat_refresh_interval            = 64       # steps
criticality_budget_frac          = 0.15
critical_value                   = 0.95     # target 1 - sigmoid(log_a)
criticality_distill_weight       = 1e-3
horizon_H                        = TBD — first ablation
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

## Success metric (the only one)

> Rare-bucket CE on Param-Golf val improves relative to a matched ScOpt-as-telemetry-only control (no criticality loss) and to the locked fast-slow baseline at matched wall-clock.

Aggregate BPB is secondary; it's the submission number but not the mechanism test. If rare-bucket CE moves and aggregate doesn't, we have a real mechanism that needs scale; if rare-bucket CE also doesn't move, the hypothesis is falsified regardless of aggregate noise.

## Scope and sequencing

First-pass estimate (be skeptical; I tend to overcount):

| stage | effort | output |
|---|---|---|
| 1. Rare Trace Bank + event-mask plumbing | 1 day | per-layer ring buffer, event capture at hook level, unit tests |
| 2. Causal channel scoring + age-weighted accumulator | 0.5 day | `score_l` trajectory, basic diagnostics |
| 3. Budget allocator + criticality loss | 0.5 day | `loss_criticality`, target emission, integration with existing loss graph |
| 4. Diagnostics + per-bucket CE readout | 0.5 day | `seat_churn`, `rare_bucket_ce` per bucket over time |
| 5. Smoke + ablation 1 (horizon_H) | 1 day | 1 × H100 smoke per H value, results doc |
| 6. Ablation 2 (per-layer vs cross-layer) + writeup | 0.5 day | decision on default |

Realistically 3-5 working days to a smoke result. Not 1 week; not 1 afternoon.

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

Draft. Not wired. Next concrete action is approve-or-redline of this doc, then Stage 1 implementation.
