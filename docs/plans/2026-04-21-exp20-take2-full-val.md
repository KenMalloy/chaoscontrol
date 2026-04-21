# Exp 20 Take 2 — TTT Against Full Val, Budget-Bound

**Goal:** Test the SSM-native TTT thesis at the regime that actually matters: full 50k-doc validation, 600s wall-clock budget, record-eligible. Stop using 128-doc prefixes as headline cells.

**Status:** Planning (2026-04-21).

## Why take 1 wasn't enough

Take 1 measured TTT on 128 docs (~0.26% of val) in an un-budget-bounded regime where TTT adapted on every chunk. The pilot did surface signal: with reset persistence + steps=1, `delta_proj @ lr=0.004` beat the reset floor by 0.00246 bpb (1.54098 → 1.53852). That is not a null result. But it answers a different question than the one Parameter Golf asks.

**Three specific problems with take 1:**

1. **Wrong regime.** A record-eligible submission must complete all 50k docs inside 600s. At 8×H100 with ~82s score-only floor, that leaves ~518s TTT slack over ~171k chunks → budget forces adaptation on ~0.3% of chunks. The pilot adapted on 100% of chunks over 0.26% of val. *Which* chunks to adapt on is the load-bearing axis, and take 1 didn't test it.

2. **Non-robust signal.** The axis1 sentinel at 512 docs (`carry_state + log_a @ lr=0.016`) went 1.54191 at 128 docs → 1.54680 at 512 docs. Either noise or the carry_state regression deepening. Without full-val numbers we can't tell.

3. **carry_state loses to reset on 128 docs** (1.54347 vs 1.54098). The "SSM state as working memory" thesis is failing its cheapest sanity check — before TTT. Either the mechanism has a bug, or carrying state across heterogeneous FineWeb docs is actively harmful. Take 1 never diagnosed this.

## Protocol

### Phase 0 — Diagnose carry_state before any TTT (required)

No TTT experiment can be trusted while the no-TTT persistence baseline has a silent regression. One run at 4×H100, full 50k:

- `score-only, reset, state_norm logged every chunk`
- `score-only, carry_state, state_norm logged every chunk`

Hypotheses to distinguish from the logs:

| Hypothesis | Signature |
|---|---|
| State norm blowup | `‖h‖` grows unboundedly across docs under carry_state |
| Cross-doc interference | per-doc bpb fine early, degrades as more docs accumulated; resets recover |
| Threading bug | carry_state is numerically equivalent to reset (silent no-op) |
| Real phenomenon | `‖h‖` stable but carry still hurts; topic shifts contaminate next-doc prediction |

Outcome of Phase 0 decides whether TTT cells in Phase 2 should use reset or carry_state (or both).

### Phase 1 — Full-val score-only floors at 8×H100

Two runs:
- `reset, 8×H100, full 50k, record-eligible`
- `carry_state, 8×H100, full 50k, record-eligible` (only if Phase 0 didn't kill it)

Records `score_floor_seconds_actual` — the remaining TTT budget `usable_ttt_budget = 600 − score_floor − safety_margin` drives Phase 2 economics.

### Phase 2 — TTT schedule × adapt set, full-val, budget-bound

The axes that matter, given the budget:

**Axis A — adapt set** (which parameters respond):
- `delta_proj` (selective updating; best take-1 cell under reset)
- `log_a` (memory horizon)
- `log_a + delta_proj` (combined)
- `lm_head` (architecture-agnostic baseline)

**Axis B — schedule** (when to adapt, given ~500 steps of budget over ~171k chunks):
- `every_Kth_chunk`, K ∈ {128, 512, 2048}
- `doc_boundary` (adapt only at first chunk of each doc)
- `high_loss_gate` (adapt only on chunks above a per-doc bpb threshold — concentrates updates where model is actually struggling)

**Axis C — step hparams** (how to adapt):
- `steps_per_chunk` ∈ {1, 2, 4}
- `lr` log-spaced: {0.001, 0.004, 0.016, 0.064}

Full grid is 4 × 5 × 3 × 4 = 240 cells. Infeasible. **Plan: Bayesian / greedy search rather than grid.**

Starting seed (best Phase 0+pilot inheritance):
- `delta_proj, every_512th, steps=1, lr=0.004`
- `delta_proj, high_loss_gate, steps=2, lr=0.004`
- `log_a, doc_boundary, steps=1, lr=0.016`
- `log_a+delta_proj, every_2048th, steps=4, lr=0.008`

Each cell is one full-val run. Budget: ~600s/cell on 8×H100. 8 cells per day at one pod. Plan: 20–30 cells total across two days if signal is found; kill early if first 5 show no schedule beats the floor.

### Phase 3 — Δ-modulation fine sweep (no-grad, cheap)

Take 1 used coarse factor-2 perturbations. The optimal window is almost certainly tighter. Fine grid, all no-grad so per-cell cost ≈ score floor only:

- `delta_scale` ∈ {0.90, 0.95, 1.00, 1.05, 1.10, 1.20}
- `log_a_shift` ∈ {−0.2, −0.1, −0.05, 0, 0.05, 0.10, 0.20}

42 cells × ~82s each = ~1h total at ws=8. Cheap. Run as a single job.

### Phase 4 — Combined stack (only if 2 and 3 have signal)

- Best Δ-modulation × best TTT schedule × best adapt set × best persistence
- 3–5 cells max
- This is the candidate submission configuration

## Budget math (8×H100)

```
total_budget           = 600.0 s
score_floor_measured  ≈  82.0 s  (from Phase 1)
safety_margin         =  30.0 s
usable_ttt_budget     ≈ 488.0 s
```

At ~1s/adapt-step, ~488 steps available across ~171k chunks → schedule must adapt on ≤ 0.3% of chunks. This is the binding constraint.

`every_Kth_chunk` with K=512 → ~335 steps — fits budget.
`every_Kth_chunk` with K=128 → ~1340 steps — overruns budget by 3× unless `steps_per_chunk=0.25`, which means only 1-in-4 of those chunks actually adapts, which is just K=512 under another name.

`doc_boundary` → 50k steps — wildly overruns.
`doc_boundary` with throttle (adapt on first chunk of every Nth doc) → tractable; N=100 gives 500 steps.

`high_loss_gate` → variable; set threshold so expected activations ≈ 400.

## Files

New / modified:
- `experiments/20_ssm_native_ttt_v2/` — new dir; v1 stays as-is for history
- `experiments/20_ssm_native_ttt_v2/run_full_val.py` — thin wrapper that enforces full-50k completion + 600s budget as *required* (non-prefix)
- `src/chaoscontrol/eval_stream/schedule.py` — `AdaptSchedule` abstraction (`every_kth`, `doc_boundary`, `high_loss_gate`)
- `src/chaoscontrol/eval_stream/state_norm.py` — state-norm logger for Phase 0
- `tests/test_eval_stream_schedule.py` — unit coverage for adapt-schedule decisions
- `tests/test_eval_stream_state_norm.py` — unit coverage for norm logger

Reuses from v1: `LegalityController`, `BudgetTracker`, `StateManager`, `DeltaModulator`, `MetricsCollector`. No changes to those.

## Metrics

Primary:
- `bpb_full_50k` — only full-val runs count.
- `ttt_wall_seconds` — must stay under `usable_ttt_budget`.
- `record_eligible` — all of: full_validation_complete, under-budget, non-collapsed.

Secondary (for interpretability):
- `bpb_per_ttt_second` — gain per budget second spent adapting
- per-doc bpb trajectory — for detecting where in the stream TTT helps vs hurts
- `state_norm` trajectory — watchdog for carry_state stability
- `adapts_triggered` — count of chunks that actually ran a step (schedule's decision)

## Decision rule

Fire-or-scrap checkpoint after Phase 1 + first 5 Phase 2 cells:

| Outcome | Action |
|---|---|
| No Phase 2 cell beats the full-val reset floor | Scrap TTT thesis; write null-result post. Phase 3 alone may still be valuable. |
| Any Phase 2 cell beats floor by > 2σ of per-doc bpb | Continue Phase 2 with schedule refinement, then Phase 4. |
| Phase 3 Δ-modulation beats floor by > 0.005 bpb (cheap, no-grad) | Bank as submission candidate regardless of Phase 2 outcome. |

"> 2σ" means the beat is larger than the noise floor estimated from re-running the reset baseline 3× with different `stream_seed`.

## What would falsify the thesis

The reviewer's long-reasoning framing is compelling but testable. It's falsified if:

1. carry_state cannot be made non-regressing (Phase 0 fails to identify a fixable cause). Working-memory claim dies.
2. No TTT schedule beats the full-val floor by more than the per-seed noise (Phase 2 null).
3. Δ-modulation's fine sweep shows a flat response surface (Phase 3 null, meaning memory-horizon control doesn't matter at this scale).

All three null would settle it. Any one win would keep the thesis live and shape the submission.

## What take-2 is NOT

- Not a speed optimization of take 1. That belongs in a separate dir if needed.
- Not sweeping prefix lengths. 128/512/2k docs are diagnostic tools only.
- Not merged with Exp 23 (training throughput). Orthogonal experiment; different axes.
- Not attempting training-time TTT. This is eval-time only, same as v1.
