# Exp 24 - Rigor and Speed Session

**Date:** 2026-04-22
**Status:** Design - approved for implementation plan
**Scope:** one session, 1 H100 primary, CPU-gloo for multi-rank tests

## Goal

Leave the Exp 24 fast-path runner in a state where:

1. The `event_sleep` training-time mechanism has tests that guard the invariants its DDP and scheduling code actually promise, not just plumbing coverage.
2. The inside-600s-budget code paths are as kernel-tight as the existing custom-kernel surface (`_cublaslt`, `_lm_head_loss`, `_ssm_scan`, `fp8_linear`) allow.

ScOpt optimizer-step compute is explicitly out of scope for speedup because it is off-clock relative to the 600-second training budget. The spirit of the Parameter Golf problem is optimizer-independent. Muon is not in the Exp 24 loop.

## Test work - event_sleep supercar

Move the three event_sleep regression tests out of `tests/test_exp23_fast_path.py` into a new file `tests/test_exp24_event_sleep.py`. The Exp 23 file keeps generic fast-path-runner coverage; the Exp 24 file owns the `fast_slow_dreamworld_event_sleep` arm's invariants. The fast-path runner stays a shared substrate.

Replace the existing `dist.all_reduce` monkeypatch test with five load-bearing tests. Each one guards a claim the code actually makes.

### 1. Real multi-rank gloo lockstep

Spawn 4 ranks with `torch.multiprocessing.spawn`, gloo backend, CPU only. Feed each rank a divergent per-step loss sequence crafted so one rank's local `ratio > threshold` and another's is below. Assert that after `LossTriggeredReplayEMA.update` every rank sees identical `triggered`, `global_pressure`, and `fire_count`. This is the one load-bearing DDP invariant; the fake `dist.all_reduce` test never exercised a real collective.

Fold the `ratio == threshold` boundary assertion into this test. `local_pressure = max(0.0, ratio - threshold)` at equality should produce `local_fire = False`.

### 2. One-step-delay invariant

Build a deterministic loss sequence that forces a trigger at step K. Assert `dream_buffer.sample` is called exactly once, at step K+1, and never at step K. This is the central guarantee that keeps the optimizer step order aligned across ranks. Any regression that moves the decision call earlier becomes a silent DDP desync source.

### 3. Seed-anchored trigger trajectory

Fixed loss trajectory, fixed EMA decay, fixed threshold, fixed warmup. Assert an exact list of trigger step indices, for example `[47, 83, 112]`. Off-by-one errors in EMA ordering or warmup counting would otherwise slip through any per-step assertion.

### 4. Warmup-restore bit-equality

Drive the `_warmup` + `_restore_state_dict` path with `restore_after_warmup=True`. Snapshot the state dict before warmup; after restore, assert every tensor compares `torch.equal` against its snapshot. Parameter Golf's warmup contract is bit-exact, not "approximately equal". This test is the difference between the benchmarked bpb reflecting the submitted model and reflecting a drifted warmup state.

### 5. bf16 loss dtype

Push a `torch.bfloat16` loss tensor through `LossTriggeredReplayEMA.update`. Assert the returned `local_loss` and `ema_loss` round-trip correctly and the gate decision matches an fp32 reference run on the same loss values. `bf16` is a recurring foot-gun in this repo; silent downcasts here would corrupt the EMA.

### Backlog

Explicitly not in this session: `min_interval` enforcement, scheduled + event replay collision, empty-buffer guard, `world_size=1` DDP-on vs DDP-off equivalence, per-arm determinism, weight=0 baseline equivalence, artifact-size compliance. Each of these is plausible but not load-bearing for the science. Pick them up only if a bug demands.

## Kernel-level speed work

### Profile first

Run the `fast_slow_dreamworld_event_sleep` arm on 1 H100 for about 30 seconds, with CUDA event timings around forward, loss, backward, optimizer step, fast-slow EMA, event-sleep gate, and dreamworld replay. Rank the hotspots by wall-clock share inside the 600s training budget. Write up the ranking as a short section in the final session writeup.

### Candidate targets

These are real gaps identified from a code audit, not speculation. Each has to survive the profile - if a candidate's fraction of inside-budget wall clock is negligible, it drops.

- **Graph-break audit on the train step.** `_run_scopt_train_step` is the hot call. Every graph break between `model.encode`, logits head, CE, and backward costs kernel-launch overhead that matters at 10.7M parameters. Walk the step with `TORCH_LOGS=graph_breaks` and eliminate what can be eliminated without destabilizing training.

- **fast_slow weight EMA.** The slow-weight blend runs every `fast_slow_interval` steps and touches every trained parameter. If it is currently a Python loop over parameters, replace with `torch._foreach_lerp_` or an equivalent multi-tensor op.

- **Spectral regularizer.** Inside budget when `spectral_reg_lambda_*` is nonzero. If it iterates SSM layers in Python, vectorize across layers.

- **Dreamworld replay backward.** A full forward-plus-backward on captured entries, several times per second. Check whether activation recomputation or a shared-state reuse with the main step is tractable; if not, document why and move on.

- **Event_sleep gate GPU sync.** `LossTriggeredReplayEMA.update` currently calls `.item()` on the loss tensor every step even when the decision result is only consumed at the start of the next step. Keep the EMA on device, do the compare on device, defer the `.item()` until the decision is consulted. Expected wall-clock win is small but directly inside budget. This one is small enough that it lands even if the profile de-ranks it.

### What we do not touch

- `_ssm_scan` and `_lm_head_loss` kernels. Both are already fused. Rewriting either is a separate project with its own design doc and verification plan.
- fp8 path. Exp 18 Test 10 settled bf16 at 10.7M parameters.
- ScOpt optimizer step internals. Off-clock relative to the 600s budget. Kernel work there does not move the Parameter Golf score.
- Muon. Not in the Exp 24 loop.
- 8-GPU-specific optimizations. 1 H100 is the development target; any DDP-specific finding from the profile gets noted for a later session.

### Landing criteria

Each candidate either lands with measured before/after step-time numbers, or is explicitly marked "audited, already optimal" or "deferred, reason X" in the writeup. No silent drops.

## Matrix change

None. The `dreamworld + event_sleep` (no `fast_slow`) arm I had originally proposed for first-wave is dropped. Event_sleep's effect as part of the full stack is what Exp 24 is designed to measure.

## Non-goals

- Rewriting kernels.
- New experiment arms.
- Changes outside `experiments/23_fast_path/runner_fast_path.py`, `experiments/24_training_time_bundle/`, `src/chaoscontrol/optim/` (if and only if spectral reg lives there), and the two test files named above.
- 8-GPU work.
- Anything that changes the submitted artifact.

## Done-when

- `tests/test_exp24_event_sleep.py` exists and holds the five tests above, all green on CPU (gloo for test 1).
- Event-sleep tests are removed from `tests/test_exp23_fast_path.py`.
- The Exp 24 first-wave matrix compiles and its existing test passes (no matrix change this session).
- Profile report committed to `docs/plans/2026-04-22-exp24-runner-profile.md` (or appended to this design doc's writeup section).
- Each kernel candidate has a verdict: landed with numbers, audited-already-optimal, or deferred-with-reason.
- `git diff --check` clean, `pytest tests/test_exp23_fast_path.py tests/test_exp24_event_sleep.py tests/test_exp24_training_bundle.py tests/test_exp24_dreamworld.py` green.
