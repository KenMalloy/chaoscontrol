# Dual-Timescale Rare Memory — Design

**Date:** 2026-04-24
**Status:** Design. Not on Exp24 critical path; target: Exp25+ once ScOpt base is calibrated and mechanism value is established.
**Depends on:** `docs/plans/2026-04-22-scarcity-optimizer-design.md` (base ScOpt).

## Thesis

ScOpt currently carries a single-timescale rare-gradient EMA (`rare_grad_ema`, decay=0.9, updated every split step). Single-timescale smoothing forces a bad tradeoff: short memory means the EMA is dominated by the last few batches' noise; long memory means it reacts too slowly to mechanism-relevant rare signal. The SSM fast/slow mechanism hit the same problem on parameters and resolved it by maintaining two coupled EMAs at different timescales. We expect the same structural solution to work for ScOpt's rare-gradient state.

The core unlock is *not raw speed* — the blend itself costs about the same as the single EMA. The unlock is that the slow branch's longer effective memory lets us *stretch `scopt_split_interval`* (currently 4) without the rare signal going stale. Fewer splits means directly less compute, because split steps are the expensive path (retain_graph + N per-layer `autograd.grad` calls).

## Update rule

```
fast = 0.90 * fast + 0.10 * rare_grad     # on every split step
slow = 0.99 * slow + 0.01 * fast          # on every split step

rare = alpha * slow + (1 - alpha) * fast  # used wherever rare_grad_ema is used today
```

Defaults to propose for a first pass:

```
rare_fast_decay  = 0.90     # matches current rare_ema_decay
rare_slow_decay  = 0.99
rare_blend_alpha = 0.80     # slow-dominant
```

`alpha` skews heavy toward slow because rare events are low-frequency by definition — the fast branch is present as a small batch-relevance correction, not as the primary signal.

## Effective window math

Effective EMA window ≈ 1 / (1 - decay):

| Branch  | Decay | Window |
| ------- | ----- | ------ |
| Current | 0.90  | ~10 samples |
| Fast    | 0.90  | ~10 samples (unchanged) |
| Slow    | 0.99  | ~100 samples |

The slow branch's window is 10× the fast one, which is what lets split cadence stretch.

## Compute multiplier from stretched split cadence

Every split step carries the retain_graph forward + N per-layer `autograd.grad` calls. At current `scopt_split_interval=4`, 25% of training steps pay that cost.

| `scopt_split_interval` | % of steps that are splits | Slow EMA updates per 100 steps | Fast coverage |
| ---------------------: | -------------------------: | -----------------------------: | ------------- |
| 4 (current)            | 25%                        | 25                             | ~10-sample window |
| 16                     | 6.25%                      | 6.25                           | noisier; ~10-sample window over 40-step training span |
| 32                     | 3.125%                     | 3.125                          | coarser; ~10-sample window over 80-step training span |

Split-step compute drops 4× at `split=16`, 8× at `split=32`, directly redeeming the "ScOpt is 9× slower than Muon" observation from the 2026-04-24 smoke.

## Where this plugs into ScOpt

Drop-in replacement for `rare_grad_ema` in two places, both in `src/chaoscontrol/optim/scopt.py`:

1. `set_rare_grad_ema` / `update_rare_grad_ema` — currently writes/updates one buffer; becomes two (`rare_fast`, `rare_slow`) updated together on each split step.
2. `_rare_adjusted_direction` — currently reads `state[p]["rare_grad_ema"]`; becomes `alpha * state[p]["rare_slow"] + (1 - alpha) * state[p]["rare_fast"]`.

Config knobs added under existing ScOpt group:

```yaml
scopt_rare_ema_decay_fast: 0.90
scopt_rare_ema_decay_slow: 0.99
scopt_rare_blend_alpha: 0.80
```

Back-compat: when the slow decay is unset or equal to the fast decay, behavior collapses to single-timescale and matches the current implementation bitwise.

## Ablations the design needs

1. **Matched-step quality at single vs dual timescale, `split=4`.** Isolates whether the slow branch alone improves rare signal quality, independent of cadence.
2. **Dual-timescale at `split=16` and `split=32` vs single-timescale at `split=4`.** Tests the real hypothesis — does stretched cadence with two timescales match or beat current cadence with one?
3. **`alpha` sweep at the winning cadence.** Slow-dominant (0.7–0.9) vs balanced (0.5) vs fast-dominant (0.1–0.3). We expect slow-dominant, but the right value probably depends on warmup state.

## Known open questions

- **Warmup behavior.** Slow EMA takes ~100 samples to reach a meaningful value. With `scopt_warmup_steps=200` and `split=16`, slow gets ~12 updates before scarcity activates — barely populated. Either warmup gates should consider slow-EMA readiness, or slow should initialize off the fast branch's first snapshot at scarcity-enable time.
- **Interaction with grad-clip rejection.** When a split step's rare contribution is discarded due to clip, both branches skip their update for that step (they're coupled). This is probably fine — mirroring the current single-EMA behavior — but worth verifying.
- **Recurrence scarcity.** `_apply_recurrence_scarcity` uses `_channel_pressure`, not `rare_grad_ema`. Dual-timescale doesn't directly change recurrence behavior. The channel pressure path might deserve its own dual-timescale treatment in a follow-up; keep that out of scope here.

## Not in scope

- No change to the row-pressure EMA or channel pressure. Those are derived from forward-pass CE signal, not rare-gradient state, and their time-scale tuning is a different problem.
- No change to the scarcity-factor math (`tanh` + floor), the row/column metric application, or the NS path.
- No change to the fused LM-head integration. Dual-timescale is orthogonal to whether per-token CE comes from the fused kernel or the unfused path.

## Relationship to the SSM fast/slow mechanism

The SSM's fast/slow is a parameter EMA; this is a rare-gradient EMA. Structurally identical (two coupled EMAs at different decays, blended at read time); semantically different (parameter weights vs optimizer state). The SSM literature and our own Phase 0 lock both support the structural pattern. We are betting it transfers.
