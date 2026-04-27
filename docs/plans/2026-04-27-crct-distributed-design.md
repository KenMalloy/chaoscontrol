# CRCT Distributed — Design

> **Status:** Approved. Implementation plan via `writing-plans`.
> **Scope:** `src/chaoscontrol/crct_distributed.py` — coordination layer for
> Cache-Reweighted Continuation Training (CRCT). Builds on the existing
> Pass C / Phase 1 3+1 episodic infra; does **not** rewrite it.
> **Non-goals:** scarcity-price math, oracle scoring math, controller-target
> definition, eviction policy. CRCT_distributed is the *transport + scheduler*;
> the math lives in callees that this module orchestrates.

## §1. Topology & process groups

```
ranks 0,1,2   = training workers; DDP wraps the model on train_pg only.
rank 3        = memory coprocessor (existing duties + new teacher work).

train_pg = new_group([0, 1, 2])     # DDP grad all-reduce, stop-flag MAX
crct_pg  = new_group([0, 1, 2, 3])  # gather(input_ids → 3); broadcast(targets ← 3)
sync_pg  = new_group([0, 3])        # state_dict snapshots, rank 0 → 3
```

`train_pg` is rank-3-free, so DDP collectives match the existing
`allreduce_grads(group=train_pg)` shape — no `materialize_zeros` needed
(rank 3 isn't in it). `crct_pg` mirrors Pass C's `dist.gather(dst=N-1)`
precedent for per-step batch fan-in; result fan-out is
`dist.broadcast(src=3)` on the same group. `sync_pg` is the snapshot bus
and never carries per-step traffic.

**Rank 3 stream layout** — two CUDA streams with priorities set at
coprocessor construction:

```python
high_stream = torch.cuda.Stream(priority=-1)   # episodic writer, controller_main, gpu_slot drain
low_stream  = torch.cuda.Stream(priority=0)    # teacher forward, controller-target compute
```

High-pri preempts low-pri inside the SM scheduler. CPU-side, the
coprocessor's `step_once()` runs high-pri reap **before** deciding
whether to enqueue new low-pri work — this is the load-bearing
scheduling rule, not the stream priority alone.

**Thin-scheduler invariant.** `Rank3MemoryCoprocessor` *imports and
orders* existing functions (`controller_main` driver, `EpisodicCache.append`,
`gpu_slot` drain, plus a caller-provided `score_fn` for oracle work). It
does **not** rewrite episodic / controller internals. Anything that
would touch those internals is out of scope and lives in a separate PR.

**Fallback at `world_size < 4`.** `create_crct_process_groups` returns
`(default_group_or_world, None, None)`. Both mailboxes detect the
`None` groups and switch to a same-process synchronous path:
training ranks call `score_fn` inline and `try_get` short-circuits to
the result they just computed. No CUDA streams, no extra processes, no
NCCL. Smoke tests, single-rank runs, and 2-rank gloo CPU tests all use
this path.

## §2. Components / API surface

### `create_crct_process_groups(world_size: int) -> tuple[PG | None, PG | None, PG | None]`

Returns `(train_pg, crct_pg, sync_pg)`. Every rank must call this in the
same order, even ranks not in a given group — that is
`torch.distributed.new_group`'s contract. `world_size < 4` returns
`(default_group_or_None, None, None)` and the caller falls back to
synchronous mode.

### `class WeightMailbox`

Async state_dict snapshot bus rank 0 → rank 3.

```python
WeightMailbox(model, sync_pg, *, sync_interval_steps=50,
              src_rank=0, my_rank, flat_buffer=None)
```

- `post_weights(global_step)` — rank-0-only. No-ops unless
  `global_step % sync_interval_steps == 0`. Flattens trainable params
  into a pre-allocated CUDA buffer on a worker stream, then a daemon
  thread issues one `dist.broadcast(flat_buffer, src=0, group=sync_pg,
  async_op=True)`. Returns immediately. The next `post_weights` skips
  if the prior broadcast hasn't completed (drop-don't-queue at the
  weight bus too).

- `maybe_sync(global_step)` — rank-3-only. Issues the matching
  `dist.broadcast` (recv side), checks `is_completed()`. When the broadcast
  finishes, unflattens into `model.state_dict()` and bumps a CPU-side
  version counter. If a new broadcast lands while the previous is still
  unflattening, the previous is discarded.

Single flat buffer (one NCCL call per snapshot) over per-tensor
broadcasts (one call per state_dict entry) — cuts NCCL launch overhead
from `O(num_params_named_tensors)` to `O(1)`. Same coalesce trick
`allreduce_grads` uses.

### `class TeacherResultMailbox`

Per-step result bus rank 3 → train ranks, fail-open on consumer side.

```python
TeacherResultMailbox(crct_pg, my_rank, num_train_ranks, *,
                     payload_shape, dtype=torch.float16, queue_depth=1)
```

- `post_result(step_id, target, conf, loss_weight)` — rank-3-only.
  Stages a fixed-shape payload tensor on `low_stream` (target [B,T,C]
  fp16 + conf [B,T] fp16 + loss_weight scalar fp16 + step_id int64
  packed via `view`). Records a CUDA event on `low_stream`. The
  scheduler issues `dist.broadcast(payload, src=3, group=crct_pg,
  async_op=True)` once the event's downstream stream-wait succeeds.

- `try_get(step) -> TeacherPayload | None` — train-rank-only. Issues a
  matching `dist.broadcast(buf, src=3, group=crct_pg, async_op=True)`
  once at construction (and once per consumed payload). Stores the
  work handle. On call: returns the payload if `work.is_completed()`,
  else returns None. On payload retrieval, validates `payload.step_id`
  is recent enough (default tolerance: `step - lag` for lag ≤ 4); stale
  payloads return None.

- **Bounded queue depth = 1.** Rank 3 holds at most one in-flight
  broadcast at a time. If `step_once` is asked to enqueue a new teacher
  job while the prior broadcast is still pending, it skips. Counter
  exposed via `coprocessor.metrics["teacher_drops"]`.

### `class Rank3MemoryCoprocessor`

The thin scheduler. Holds the two CUDA streams, the bounded queue, and
the existing-duty function references.

```python
Rank3MemoryCoprocessor(
    model_copy, weight_mailbox, result_mailbox,
    *,
    score_fn,                  # (model, input_ids, valid_mask) -> (target, conf, lw)
    episodic_drain_fn,         # () -> None — pulls from gpu_slot, calls cache.append
    controller_tick_fn,        # () -> None — drives controller_main one tick
    crct_pg, sync_pg,
    high_stream=None, low_stream=None,
)
```

`step_once(global_step)` runs in this order, deterministically:

1. **Reap teacher** — if low_stream has a completed score, hand
   payload to `result_mailbox.post_result`.
2. **Drain episodic gather** — `dist.gather(dst=3, group=crct_pg)` to
   receive the train ranks' batch + write events. Issued on
   high_stream.
3. **Episodic write commits** — `episodic_drain_fn()` on high_stream
   (Pass C drain).
4. **Controller tick** — `controller_tick_fn()` on high_stream.
5. **Maybe sync weights** — `weight_mailbox.maybe_sync(global_step)`,
   non-blocking.
6. **Maybe enqueue teacher** — if the bounded queue is empty and
   oracle is enabled this step, enqueue `score_fn` on low_stream
   against the just-received batch. Otherwise no-op.

The reap-before-enqueue order is what gives drop-don't-queue:
new teacher work is only scheduled if the previous broadcast is gone.

### `def rank3_coprocessor_loop(coprocessor, *, stop_flag) -> None`

The actual `while not stop:` driver on rank 3. Calls
`coprocessor.step_once(step_id)` per training step. Stop is signaled
via a separate `dist.broadcast` on `sync_pg` (rank 0 → 3); doesn't ride
on `train_pg`'s `should_stop_now` MAX collective because rank 3 isn't
in `train_pg`.

Single-process path (`world_size < 4`): function returns immediately;
training ranks invoke `score_fn` inline through the synchronous-mode
`try_get`.

## §3. Data flow per step

```
                  TRAIN RANKS [0,1,2]                          RANK 3
1. fetch batch         |                                         |
2.                     | --- gather(input_ids, dst=3, crct_pg) → | recv batch
3. forward+backward    |                                         | (high-pri reap teacher)
4. allreduce grads     | (train_pg, op=AVG)                      | (episodic drain)
5. try_get(step)       | ← broadcast(target, src=3, crct_pg) --- | (controller_main tick)
6. compute loss        |   None → fail-open; else apply target   | (maybe_sync weights)
7. optimizer.step      |                                         | (low-pri enqueue score)
8. (every N steps)     | rank 0 post_weights → sync_pg --------→ | maybe_sync receive
```

Causal order locks: rank 3 reads cache → scores → broadcasts targets →
**then** drains write events. The `dist.gather` at step 2 carries both
input_ids and the per-rank write-event payload (Pass C slot format);
rank 3's reap (step 1) and broadcast post (carried over from prior
iteration) happen before the new gather lands. Lag = 1 step is the
simplest start; bounded by `crct_pg` round-trip latency.

## §4. Failure / fallback

- **`world_size < 4`** — synchronous-on-train-rank path. `try_get`
  invokes `score_fn` inline against the train rank's own model copy.
  No streams, no NCCL. This is what gloo-CPU tests and single-rank
  smoke runs hit.
- **Rank 3 falls behind** — bounded queue depth = 1 enforces
  drop-don't-queue. `try_get` returns None on the train side; train
  takes the fail-open path. Counter `teacher_drops` exposed.
- **Weight snapshot stale** — rank 3 keeps using the previous snapshot.
  No barrier on either end of `maybe_sync`. The version counter is
  recorded on the next teacher payload so train-side observability can
  tell which snapshot scored a step.
- **Rank 3 dies** — out of scope for auto-recovery. NCCL collective on
  `crct_pg` will hang the train ranks; the runner's existing per-step
  health check catches this. CRCT is opt-in; documented as a known
  failure mode.
- **`train_pg` rank-3-free invariant** — assert at init:
  `dist.get_world_size(train_pg) == 3` and the runner's
  `allreduce_grads(group=train_pg, ...)` calls are routed through
  `train_pg`. A regression that puts rank 3 back into train_pg's
  collectives would silently corrupt gradient averages (3 train + 1
  zero-fill rank, but no longer materialize_zeros, so flatten shapes
  diverge).
- **Fail-open contract** — training side wraps the controller term as

  ```python
  payload = result_mailbox.try_get(step)
  if payload is None:
      loss = lm_loss + 0.0 * controller_logits.sum()
  else:
      loss = lm_loss + lambda_ctrl * controller_loss(payload)
  ```

  The `0.0 * controller_logits.sum()` keeps `controller_logits` (and
  every parameter that produced it) connected to the autograd graph.
  DDP's parameter-readiness machinery requires the **same** set of
  params to receive grads on every step; without the zero-product term,
  fail-open and fail-closed steps would have different grad sets and
  DDP would deadlock on the next bucket-allreduce. Load-bearing.

## §5. Tests

`tests/test_crct_distributed.py`. Mocked `torch.distributed` primitives
per Ken's spec. End-to-end gloo-spawn coverage rides on the existing
`test_runner_3plus1_skip_main.py` pattern as a follow-up integration
test, **not** in this module.

Test inventory:

1. **`create_crct_process_groups` correctness.** `world_size=4` returns
   three groups with the right rank lists; `world_size=2` returns
   `(default, None, None)`. Verified via a `FakeProcessGroup` that
   records `new_group` calls and rank lists.
2. **`WeightMailbox.post_weights` non-blocking on rank 0.** Returns
   within a tight wall-clock budget even when the mocked broadcast's
   work handle reports `is_completed=False`; daemon thread holds the
   handle.
3. **`WeightMailbox.maybe_sync` skips no-new-snapshot.** Repeat calls
   without a new post are idempotent; only one `load_state_dict` per
   snapshot version.
4. **`TeacherResultMailbox.try_get` returns None when work
   incomplete.** Fake work `is_completed()` returns False → None;
   True → payload.
5. **Drop-don't-queue invariant.** `step_once` skips enqueuing a new
   teacher job when the bounded queue is full; `metrics["teacher_drops"]`
   increments. Forced via a stub `score_fn` that never finishes.
6. **Causal order.** Stub all five callables with side-effects that
   record call order; assert sequence matches §2 step 1→6.
7. **Fail-open math.** Tiny `nn.Module`: with `payload=None`, loss
   evaluates to `lm_loss` numerically and `param.grad` materializes
   for every parameter that touched `controller_logits`. Without the
   `0.0 * sum`, the assertion that `controller_head.weight.grad is not
   None` fails — that's the regression guard.
8. **`world_size < 4` synchronous fallback.** `try_get` invokes the
   real `score_fn` inline and returns its result; no NCCL, no streams.

Out of scope (tracked separately):

- gloo-CPU 4-rank end-to-end ride-along — pattern exists in
  `test_runner_3plus1_skip_main.py`; lives next to the eventual
  coprocessor wiring under `experiments/26_crct/` (or wherever the
  runner lands).
- Score-function correctness (caller's responsibility; CRCT_distributed
  treats `score_fn` as a black box).
- Stream-priority observability under load (needs real CUDA + nsys;
  belongs in a perf-pass doc, not a unit test).
