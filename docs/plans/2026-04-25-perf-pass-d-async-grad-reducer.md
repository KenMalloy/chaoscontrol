# Perf Pass D — Async grad reducer for episodic-only configs

> **Status:** Design draft. Implementation lands after Pass C is fully merged + reviewed; runs in parallel with Phase 3.6 drift correction.
> **Goal:** Re-enable `grad_allreduce_mode='async_param'` for episodic-only configurations (where ScOpt isn't active). Currently gated by ScOpt's bulk requirement; that gate is over-broad — it also blocks the async path for configs that don't use ScOpt at all.

## Why

`AsyncGradAllReducer` (`src/chaoscontrol/distributed.py:100`) overlaps gradient all-reduce with the backward pass — each parameter's all-reduce launches asynchronously as its grad materializes, instead of one bulk collective after all backwards complete. For models where backward dominates step time and the all-reduce window is fully overlap-able, this can cut step time by ~10-20%.

The runner currently requires `grad_allreduce_mode='bulk'` whenever ScOpt is active (`runner_fast_path.py:825`). The reason is that ScOpt does post-backward grad inspection (rare-grad EMA accumulation, channel pressure recording) and needs all grads materialized before the all-reduce so it can read them. Async per-param all-reduce reorders that.

But: the new `episodic_enabled=True` configs are **mutually-exclusive with ScOpt** (Phase 1 Task 1.3 incompatibility guard). So for those configs, the ScOpt-bulk constraint doesn't apply — async is safe.

## What changes

1. **Relax the bulk-only check** in `runner_fast_path.py` — instead of "ScOpt requires bulk", require "ScOpt OR async-incompatible-feature requires bulk." Episodic alone (no ScOpt) opts into async-eligible.
2. **Compose async with the new SUM/all_group/materialize_zeros path.** The existing async reducer uses default group + AVG. The new path uses `all_group` + SUM + materialize_zeros. `AsyncGradAllReducer` needs to accept the same kwargs `allreduce_grads` does (Pass C extended that signature; async needs the parallel extension).
3. **Verify ordering against `dist.gather`.** Pass C's gather collective is BEFORE the all-reduce; the async reducer fires per-grad as backward completes. Need to ensure: gather happens AFTER all backwards complete (sync barrier), THEN allreduce_grads fires (async or bulk). Otherwise gather races backward.

## Implementation tasks

- **D.1 — Extend `AsyncGradAllReducer` API.** Accept `group`, `op`, `materialize_zeros` kwargs matching Pass C's `allreduce_grads`. Default behavior unchanged (back-compat).
- **D.2 — Relax the runner guard.** Replace `if scopt and not bulk: raise` with `if (scopt OR predictive_aux OR ...) and not bulk: raise`. Episodic alone passes.
- **D.3 — Wire async + episodic.** When `episodic_enabled=True` and `grad_allreduce_mode='async_param'`, the train step uses `async_grad_reducer.wait()` after the gather collective (so gather doesn't race the per-grad all-reduces).
- **D.4 — Tests.** mp.spawn 4-rank gloo test that runs episodic + async together and verifies (a) no deadlock, (b) grads match the bulk-mode result bit-identically (within fp32 tolerance), (c) step time is ≤ bulk-mode time on a tiny model.

## Cost estimate

- Code: ~80 lines + ~150 lines of tests
- Wall-clock estimate: 1 implementer + 2 reviewers ≈ 1 session
- Throughput win: 10-20% on the 3+1 split if backward dominates step time. Larger win on 6+2 because more all-reduce traffic to overlap.

## What does NOT change

- ScOpt configs continue requiring bulk. The guard relaxes for episodic-without-ScOpt; ScOpt-with-bulk is untouched.
- Pass C's `dist.gather` semantics. Gather still fires synchronously before any all-reduce (async or bulk).

## Risk

- Async per-param all-reduce reorders grads' arrival on remote ranks vs bulk's deterministic single-collective ordering. For SGD-like optimizers this is fine; for Adam-family the second-moment estimate sees the same values either way (it integrates over many steps, the per-step ordering doesn't matter at steady state).
- If a future feature reads grads between backward and optimizer.step (the way ScOpt does), it would conflict. Add a "post-backward grad-read" registry so the guard can query it; flag this if it comes up.

## Sequencing

Pass D runs AFTER Pass C is fully merged. Can run in parallel with Phase 3.6 drift correction implementation since they touch different files.
