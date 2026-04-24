# Criticality Distillation — Runner Wiring Design

**Date:** 2026-04-24
**Status:** Draft. Depends on the v3 mechanism design at `docs/plans/2026-04-24-criticality-distillation.md` and the Stage 1a/1b/2/3 implementation now on `main` (terminal commit `c0318d8`).
**Goal:** Wire `CriticalityDistillation` into the training step and emit a first-read smoke matrix that can falsify or confirm the mechanism.

## What this design does NOT re-decide

Already settled in the mechanism design (`2026-04-24-criticality-distillation.md`) and implemented:

- Three roles: ScOpt = event detector (demoted here — we replace it with model-native surprise, see below), recurrence state = teacher, criticality = actuator.
- Per-step aggregate trace bank (not per-event).
- Post-event trace scoring with age-weighted evidence, seats do not age (only evidence ages).
- Evidence-gated top-k allocator with seat-masked MSE loss.

## Decisions made in this design

### 1. Pressure is model-native surprise, not ScOpt pressure

`CriticalityDistillation` does not depend on `ScarcityAwareOptimizer` being active. Pressure is computed independently inside the runner's training step:

```
pressure = relu(CE - H[p])                # [B, T]
```

where `H[p] = -Σ_i p_i · log p_i` is the per-position output-distribution entropy and `CE = -log p(target|context)` is the per-position cross-entropy against the ground-truth next token.

**Why this form** (interdisciplinary panel, 2026-04-24):
- Cognitive neuroscience: hippocampus gates consolidation on prediction error against own model, not stimulus frequency (Rescorla-Wagner; Lisman & Grace VTA-HC loop; Düzel contextual novelty).
- Active inference: `CE - H[p]` is the Shannon approximation of Bayesian surprise — observed surprisal minus expected surprisal. Precision-weighting `(CE - H) · H[p]` would be the next refinement; deferred.
- Developmental: fast-mapping (Carey, Horst & Samuelson) commits memory on "confident wrong" not "rare." `CE - H[p]` fires on confident-wrong (low entropy, high surprisal) and suppresses confused-wrong (high entropy, high surprisal).
- ML curiosity panel dissent (noisy-TV) is blunted by: (a) top-k is rank-based per batch so noise can't accumulate unbounded bonus; (b) CD's downstream channel-survival filter discards spurious events that don't produce distinctive persistence.

**Conditional top-k.** `compute_event_mask` marks only strictly-positive-pressure positions. Uniform-zero input produces no events, not k tied zeros. If no position in a batch has positive innovation, ingest writes nothing.

**Implementation.** Entropy emerges from the fused LM head kernel for free: `H[p] = lse - Σ_i p_i · logit_i`, and `Σ p_i · logit_i` can be accumulated in the streaming pass alongside `lse` with one extra scalar per row. No algorithmic cost. Fused kernel signature extends from `(loss, lse, per_token_ce)` to `(loss, lse, per_token_ce, per_token_entropy)`.

Non-fused fallback computes entropy in Python from the full-materialized softmax — negligible compared to the `[B, T, V]` forward.

### 2. Capture wiring — ExitStack at runner level

Around `model.encode(...)` in the training step:

```python
from contextlib import ExitStack

with ExitStack() as stack:
    getters = [stack.enter_context(layer.core.capture_states())
               for layer in model.layers]
    hidden = model.encode(inputs)
states_per_layer = [g() for g in getters]
```

Zero changes to `ChaosStudentLM` or `ChaosSSMCore` signatures (beyond the `capture_states()` context manager already landed in Stage 1a). The runner coordinates; the model does not grow a training-only kwarg on its production `encode` path.

### 3. Bank lives on CPU; `seat_mask` lives on GPU

CPU-resident:
- `bank_evidence[L, TTL, D]` fp32
- `bank_step[L, TTL]` int64
- `bank_event_count[L, TTL]` fp32
- `baseline_future_energy[L, D]` fp32
- `baseline_initialized[L]` bool

GPU-resident:
- `seat_mask[L, D]` bool (read by `criticality_loss` on every backward step)

**Why this split.** Bank is storage + arithmetic that happens off the critical path of the forward/backward. Moving it to CPU:
1. Frees HBM for training activations (submission is tight on the 80 GB H100).
2. Eliminates GPU stream synchronization points around `allocate_seats` (which otherwise blocks the main stream while reading `seat_mask`).
3. Enables CUDA-graph capture of the main training graph — all CD bookkeeping lives outside the captured region.
4. CPU RAM (251 GB on our pods) is effectively unbounded relative to any reasonable TTL, which lets the bank cover a long evidence window and removes TTL as a memory-constrained knob.

### 4. Two-phase ingest with async D2H

Per-step, inside the training loop after the forward:

```python
# Phase 1 (GPU). Cheap per-layer reduction, all on the training stream.
evidence_gpu = cd.ingest_gpu(
    pressure=pressure,                   # [B, T] GPU
    states_per_layer=states_per_layer,   # [B, T, D] each, GPU
    horizon_H=config.horizon_H,
    event_frac=config.event_frac,
)
# evidence_gpu: {"evidence": [L, D] fp32 GPU, "event_count": [L] fp32 GPU,
#                "future_energy_for_baseline": [L, B, T, D] fp32 GPU (or a
#                pre-aggregated non-event mean [L, D] for CPU-side EMA)}

# Phase 2 (PCIe + CPU, async on side stream). Non-blocking D2H + CPU append.
cd.ingest_cpu_async(evidence_gpu, step=current_step, event_mask=event_mask)
```

The CPU append reads the (already async-copied) `[L, D]` evidence vector and `[L]` event-count scalars, plus the non-event-mean-future-energy aggregate for baseline EMA. All CPU ops.

### 5. Seat refresh every 64 steps

```python
if current_step % cd.seat_refresh_interval == 0 and current_step >= warmup_steps:
    cd.allocate_seats_cpu(current_step=current_step)  # CPU compute
    cd.sync_seat_mask_to_gpu(non_blocking=True)       # H2D on side stream
```

Evidence gate inside `allocate_seats_cpu` returns `None` for any layer whose total weighted event count is below `min_weighted_events_per_layer`; those layers' seat slots are cleared (GPU `seat_mask` zeros out for the layer) and that layer contributes nothing to the criticality loss.

### 6. Loss composition

```python
total_loss = ce_loss + cd.criticality_loss(log_a_per_layer)
```

`criticality_loss` multiplies `criticality_distill_weight` internally (finding 4 fix, commit `c0318d8`). Runner does not multiply.

### 7. Defaults (rescaled for CPU-resident bank)

```
event_frac                       = 0.05
trace_half_life_steps            = 2048      # was 256
trace_ttl_steps                  = 20480     # derived: 10 × half_life
horizon_H                        = 64
seat_refresh_interval            = 64
criticality_budget_frac          = 0.15
critical_value                   = 0.95
criticality_distill_weight       = 1e-3
baseline_ema_decay               = 0.99
min_weighted_events_per_layer    = 256
```

Rationale for `trace_half_life_steps = 2048`: at ~10 step/s on a 600s Param-Golf run (~6K steps), 2048 is 1/3 of the run. Recent third contributes full weight; first third weighs ~12%. Matches the "evidence integrates over the run" framing without pinning seats to stale early-training evidence.

## Smoke matrix — 8 cells

Single seed, 1×H100, 600s budget per cell. Expected wall-clock ~80-90 min total.

Cells as registered in `experiments/24_training_time_bundle/exp24.py` under new builder `build_criticality_distillation_first_smoke_matrix`:

| cell name | pressure source | `trace_half_life_steps` | `horizon_H` | `criticality_distill_weight` | score / event variant |
|---|---|---:|---:|---:|---|
| `treatment` | `relu(CE - H[p])` | 2048 | 64 | 1e-3 | real score, top-k by positive innovation |
| `telemetry` | same | 2048 | 64 | **0** | bank collects, seats bind, but no loss gradient |
| `shuffled` | same | 2048 | 64 | 1e-3 | **score permuted across channels** before top-k |
| `budget_only` | **`torch.ones_like(ce)`** | 2048 | 64 | 1e-3 | uniform pressure, random top-k, uniform evidence → random seats via tiebreak |
| `hl_short` | `relu(CE - H[p])` | **256** | 64 | 1e-3 | bank-memory ablation (tight) |
| `hl_long` | same | **16384** | 64 | 1e-3 | bank-memory ablation (near-ceiling) |
| `H_short` | same | 2048 | **16** | 1e-3 | trace-window ablation (tight) |
| `H_long` | same | 2048 | **256** | 1e-3 | trace-window ablation (loose) |

### What each cell tells us

- **`treatment` vs `telemetry`:** is the loss gradient doing work, or is the bank + allocator alone (via observational effect on log_a's init gradient distribution) sufficient? Telemetry isolates the loss contribution.
- **`treatment` vs `shuffled`:** does *which* channels get seats matter, or does any budgeted criticality help? Shuffled breaks the score→channel identity link.
- **`treatment` vs `budget_only`:** does ScOpt-style event detection matter, or is the mechanism robust to random event assignment? Budget-only is the most stringent control for "does the pressure signal carry mechanism-relevant information?"
- **`hl_short` / `hl_long` vs `treatment`:** is there a sweet spot for bank memory window? Flat = memory window doesn't matter; monotonic-with-length = longer is better up to a ceiling; inverted-U = there's an optimum.
- **`H_short` / `H_long` vs `treatment`:** does the trace-attribution window matter? Strong H dependence = the definition of "this channel preserved the event" is sensitive to the window; weak H dependence = the mechanism is robust.

## Success metric

**Primary (and only gate):** rare-bucket CE trajectory on Param-Golf val. `treatment` must improve rare-bucket CE relative to `telemetry`, `shuffled`, AND `budget_only`.

- Rare-bucket CE is already tracked by `FrequencyBucketBaseline` during training; readout is direct from its EMA state.
- The falsifier cells ship in the first matrix, not as v2 ablations, because any one of them matching `treatment` falsifies the mechanism claim.

**Secondary (reported, not gating):**
- Aggregate val BPB per cell (the submission number, but not a mechanism test at this scale).
- `seat_churn` (fraction of seats changed per refresh) — healthy trajectory is high at warmup, declining to moderate at steady state.
- `budget_occupancy` (fraction of target-criticality channels at or above threshold) — should be near 1.0 after warmup.
- `score_criticality_corr` (rank correlation between score and final criticality per channel) — positive and rising confirms the distillation loss is the dominant log_a gradient source.
- `event_rate` (events per step per layer) — stable around `event_frac × batch × seq` after warmup; drops to zero if pressure distribution shifts pathologically.

## Engineering constraints

(per memory/feedback_risks_not_implementation_challenges.md — these are constraints, not risks)

- **Stream ordering.** The D2H copy of `evidence_gpu` lives on a side stream; the CPU append waits on an event recorded after the copy. `seat_mask` H2D is separately sequenced on a side stream so the main training stream reads the previous-refresh `seat_mask` until the new one is fully resident. `cuda.stream_wait_event` brackets this explicitly.
- **Checkpoint atomicity.** When saving, the CPU bank + GPU `seat_mask` are both serialized as registered buffers — `state_dict()` captures consistent state because CD is a single `nn.Module`. No split-brain between device halves.
- **Async CPU bookkeeping.** CD's CPU work (append + score + top-k) is measured in tens of microseconds per layer. Dataloader runs on separate processes. No CPU-thread contention expected at our workload.

## Wiring the falsifier variants

All four mechanism cells (`treatment` / `telemetry` / `shuffled` / `budget_only`) are reached by config flags on CD — no separate runner paths:

```python
# CriticalityDistillation constructor
cd = CriticalityDistillation(
    ...,
    criticality_distill_weight=weight,    # 0 for telemetry
    score_permute_before_topk=shuffled,   # True for shuffled
    uniform_pressure=budget_only,         # True for budget_only (bypasses the runner's CE-H[p] computation and forces pressure = ones)
)
```

`score_permute_before_topk` and `uniform_pressure` are new flags on CD, default False. The ablation variants (`hl_*`, `H_*`) only change hyperparameters, no new flags.

## Integration points in the runner

Changes in `experiments/23_fast_path/runner_fast_path.py` and `experiments/24_training_time_bundle/exp24.py`:

- **Extend fused LM head kernel** to emit per-token entropy alongside per-token CE.
- **Add CD capture + pressure + loss compose** inside the training step loop, gated on `cd is not None`. No impact on non-CD runs.
- **Register new matrix builder** `build_criticality_distillation_first_smoke_matrix` in `run_exp24.py` (argparse choice + default world size + default budget).
- **Stage 2.1 `compute_event_mask` update** for conditional top-k (strictly-positive only). This is a behavior change; regression pin in `tests/test_criticality_scoring.py`.

## What ships first

Minimum viable smoke:
- Stage 2.1 mask-function update (conditional top-k).
- Fused LM head entropy emission.
- Runner wiring (capture + pressure + ingest + loss compose).
- CD constructor flags for falsifier variants.
- Matrix builder + argparse registration.
- Smoke run (8 cells × 1 seed × 600s on 1×H100).

If the smoke's `treatment` cell moves rare-bucket CE and beats all three falsifiers, proceed to multi-seed + 4×H100 confirmation. If not, the four-cell matrix localizes *which* part of the mechanism is doing work (or isn't), and we redesign from that data.

## Parked for later

- Precision-weighted surprise: `(CE - H) · H[p]` second-pass refinement (active-inference panel's recommendation).
- Shuffled-teacher across layers rather than across channels within a layer.
- Matched-nearby baseline control (vs current EMA baseline) as a scoring ablation.
- Per-frequency-bucket evidence banks (bucket-keyed `bank_evidence`).
- Soft/Lagrange budget allocator vs current hard top-k.
- Multi-seed statistical power analysis before 4×H100 scale-up.
