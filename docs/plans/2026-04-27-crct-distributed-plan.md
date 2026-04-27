# CRCT Distributed Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `src/chaoscontrol/crct_distributed.py` — the coordination layer for Cache-Reweighted Continuation Training (CRCT) on the existing 3+1 rank topology. Three process groups, two CUDA streams on rank 3, drop-don't-queue teacher mailbox, fail-open training-side wrapper.

**Architecture:** Thin scheduler that imports and orders existing functions (`controller_main`, `EpisodicCache.append`, `gpu_slot` drain) plus a caller-provided `score_fn`. No rewrites of episodic/controller internals. Three process groups (`train_pg=[0,1,2]`, `crct_pg=[0,1,2,3]`, `sync_pg=[0,3]`) with one job each. Fallback path for `world_size < 4` short-circuits all NCCL/streams to a synchronous on-train-rank invocation.

**Tech Stack:** Python 3.12, `torch` + `torch.distributed` (NCCL prod / gloo CPU tests), `threading` for the rank-0 weight broadcaster daemon, `torch.cuda.Stream` for priority isolation on rank 3, `unittest.TestCase` (matches the rest of `tests/`).

**Reference docs:**
- Design: `docs/plans/2026-04-27-crct-distributed-design.md`
- Existing distributed primitives: `src/chaoscontrol/distributed.py`
- Existing 3+1 collective shape: `tests/test_runner_3plus1_skip_main.py`
- Pass C gather precedent: `docs/plans/2026-04-25-perf-pass-c-gpu-resident-ipc.md`

**Discipline reminders:**
- TDD: failing test before implementation, every task. No "trivial edit" exception (memory: `feedback_run_tests_after_every_edit.md`).
- Commit after each task — frequent commits, never batched.
- No compound bash commands (`&&`, `;`, `|`); run each separately (Ken's global preference).
- Activate `/workspace/venv/bin/activate` before any `pytest` on the pod (project CLAUDE.md).

---

## Task 1: Module skeleton + FakeProcessGroup test infra

**Files:**
- Create: `src/chaoscontrol/crct_distributed.py`
- Create: `tests/test_crct_distributed.py`

**Step 1: Write the failing test**

```python
# tests/test_crct_distributed.py
"""Mocked-primitive coverage for crct_distributed.

End-to-end gloo-spawn 4-rank coverage rides on the existing
test_runner_3plus1_skip_main.py pattern as a follow-up integration test;
this file unit-tests the coordination layer with FakeProcessGroup so the
suite stays CPU-only and fast.
"""
import unittest
from chaoscontrol import crct_distributed as crct


class FakeWork:
    def __init__(self, completed=False):
        self._completed = completed
    def is_completed(self):
        return self._completed
    def wait(self):
        self._completed = True


class FakeProcessGroup:
    def __init__(self, ranks):
        self.ranks = list(ranks)
        self.broadcasts = []
        self.gathers = []
    def __repr__(self):
        return f"FakePG({self.ranks})"


class TestModuleImports(unittest.TestCase):
    def test_module_exposes_documented_api(self):
        for name in (
            "create_crct_process_groups",
            "WeightMailbox",
            "TeacherResultMailbox",
            "Rank3MemoryCoprocessor",
            "rank3_coprocessor_loop",
            "fail_open_controller_term",
        ):
            self.assertTrue(
                hasattr(crct, name),
                f"crct_distributed must expose {name}",
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_crct_distributed.py::TestModuleImports -v`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError`.

**Step 3: Write minimal module skeleton**

```python
# src/chaoscontrol/crct_distributed.py
"""CRCT distributed coordination layer.

See docs/plans/2026-04-27-crct-distributed-design.md.
"""
from __future__ import annotations
__all__ = [
    "create_crct_process_groups",
    "WeightMailbox",
    "TeacherResultMailbox",
    "Rank3MemoryCoprocessor",
    "rank3_coprocessor_loop",
    "fail_open_controller_term",
]


def create_crct_process_groups(world_size):
    raise NotImplementedError


class WeightMailbox:  # noqa: D101 — fleshed out in Task 3
    def __init__(self, *args, **kwargs): raise NotImplementedError


class TeacherResultMailbox:
    def __init__(self, *args, **kwargs): raise NotImplementedError


class Rank3MemoryCoprocessor:
    def __init__(self, *args, **kwargs): raise NotImplementedError


def rank3_coprocessor_loop(*args, **kwargs):
    raise NotImplementedError


def fail_open_controller_term(*args, **kwargs):
    raise NotImplementedError
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_crct_distributed.py::TestModuleImports -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/crct_distributed.py tests/test_crct_distributed.py
git commit -m "crct_distributed: module skeleton + import smoke"
```

---

## Task 2: `create_crct_process_groups`

**Files:**
- Modify: `src/chaoscontrol/crct_distributed.py`
- Modify: `tests/test_crct_distributed.py`

**Step 1: Write the failing test**

```python
class TestCreateProcessGroups(unittest.TestCase):
    def setUp(self):
        self.calls = []
        # Patch torch.distributed.new_group to record rank lists without
        # actually initializing NCCL/gloo. All ranks must call new_group
        # in the same order — verify call order, not real groups.
        import torch.distributed as dist
        self._real = dist.new_group
        dist.new_group = lambda ranks=None, **_: (
            self.calls.append(list(ranks)) or FakeProcessGroup(ranks)
        )
    def tearDown(self):
        import torch.distributed as dist
        dist.new_group = self._real

    def test_world_size_4_returns_three_groups_in_canonical_order(self):
        train_pg, crct_pg, sync_pg = crct.create_crct_process_groups(4)
        self.assertEqual(self.calls, [[0,1,2], [0,1,2,3], [0,3]])
        self.assertEqual(train_pg.ranks, [0,1,2])
        self.assertEqual(crct_pg.ranks, [0,1,2,3])
        self.assertEqual(sync_pg.ranks, [0,3])

    def test_world_size_below_4_returns_none_groups(self):
        # Fallback for smoke / single-GPU / 2-rank gloo CPU tests.
        train_pg, crct_pg, sync_pg = crct.create_crct_process_groups(2)
        self.assertIsNone(crct_pg)
        self.assertIsNone(sync_pg)
        # train_pg may be None or default — caller must handle either.
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_crct_distributed.py::TestCreateProcessGroups -v`
Expected: FAIL with NotImplementedError.

**Step 3: Implement**

```python
def create_crct_process_groups(world_size):
    """Create train_pg, crct_pg, sync_pg in canonical call order.

    Every rank must call this in the same order even if not in the group
    — that is torch.distributed.new_group's contract.
    """
    import torch.distributed as dist
    if int(world_size) < 4:
        return None, None, None
    train_pg = dist.new_group(ranks=[0, 1, 2])
    crct_pg = dist.new_group(ranks=[0, 1, 2, 3])
    sync_pg = dist.new_group(ranks=[0, 3])
    return train_pg, crct_pg, sync_pg
```

**Step 4: Run tests**

Run: `pytest tests/test_crct_distributed.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/crct_distributed.py tests/test_crct_distributed.py
git commit -m "crct_distributed: create_crct_process_groups + tests"
```

---

## Task 3: `WeightMailbox` — flat-buffer snapshot, sync version counter

Implement the data structure first (no async daemon yet); Task 4 layers the daemon thread on top.

**Files:**
- Modify: `src/chaoscontrol/crct_distributed.py`
- Modify: `tests/test_crct_distributed.py`

**Step 1: Write the failing tests**

```python
import torch.nn as nn

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=True)


class TestWeightMailboxBasics(unittest.TestCase):
    def test_post_weights_skips_when_step_not_aligned(self):
        m = _TinyModel()
        pg = FakeProcessGroup([0, 3])
        mb = crct.WeightMailbox(
            m, pg, sync_interval_steps=50, src_rank=0, my_rank=0,
            broadcast_fn=lambda *a, **kw: FakeWork(completed=False),
        )
        mb.post_weights(global_step=49)
        self.assertEqual(mb.snapshots_posted, 0)

    def test_post_weights_snapshots_at_interval(self):
        m = _TinyModel()
        pg = FakeProcessGroup([0, 3])
        broadcasts = []
        mb = crct.WeightMailbox(
            m, pg, sync_interval_steps=50, src_rank=0, my_rank=0,
            broadcast_fn=lambda buf, src, group, async_op: (
                broadcasts.append((buf.shape, src)) or FakeWork(completed=True)
            ),
        )
        mb.post_weights(global_step=50)
        self.assertEqual(mb.snapshots_posted, 1)
        self.assertEqual(len(broadcasts), 1)

    def test_maybe_sync_idempotent_without_new_post(self):
        m = _TinyModel()
        pg = FakeProcessGroup([0, 3])
        mb = crct.WeightMailbox(
            m, pg, sync_interval_steps=50, src_rank=0, my_rank=3,
            broadcast_fn=lambda *a, **kw: FakeWork(completed=False),
        )
        v0 = mb.snapshot_version
        mb.maybe_sync(50)
        mb.maybe_sync(51)
        self.assertEqual(mb.snapshot_version, v0)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_crct_distributed.py::TestWeightMailboxBasics -v`
Expected: FAIL.

**Step 3: Implement**

```python
import torch
import torch.distributed as _dist

def _flatten_trainable(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


class WeightMailbox:
    """Async state_dict snapshot bus rank 0 → rank 3 over sync_pg.

    `broadcast_fn` is injected for testability; production callers leave
    it None and the class uses ``torch.distributed.broadcast`` directly.
    """

    def __init__(self, model, sync_pg, *, sync_interval_steps=50,
                 src_rank=0, my_rank, broadcast_fn=None):
        self.model = model
        self.sync_pg = sync_pg
        self.sync_interval_steps = int(sync_interval_steps)
        self.src_rank = int(src_rank)
        self.my_rank = int(my_rank)
        self._broadcast = broadcast_fn or _dist.broadcast
        self._buffer = _flatten_trainable(model).clone()
        self._inflight = None  # (work_handle, version)
        self.snapshots_posted = 0
        self.snapshot_version = 0

    def post_weights(self, global_step):
        if self.my_rank != self.src_rank:
            return
        if int(global_step) % self.sync_interval_steps != 0:
            return
        if self._inflight is not None and not self._inflight[0].is_completed():
            return  # drop-don't-queue at the weight bus
        self._buffer.copy_(_flatten_trainable(self.model))
        work = self._broadcast(
            self._buffer, src=self.src_rank, group=self.sync_pg, async_op=True,
        )
        self.snapshots_posted += 1
        self._inflight = (work, self.snapshots_posted)

    def maybe_sync(self, global_step):
        if self.my_rank == self.src_rank:
            return
        if self._inflight is None:
            # Issue a recv-side broadcast to rendezvous with the next post.
            work = self._broadcast(
                self._buffer, src=self.src_rank, group=self.sync_pg,
                async_op=True,
            )
            self._inflight = (work, self.snapshot_version + 1)
            return
        work, version = self._inflight
        if not work.is_completed():
            return
        # Unflatten into model in-place.
        offset = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.data.copy_(self._buffer[offset:offset + n].view_as(p))
            offset += n
        self.snapshot_version = version
        self._inflight = None
```

**Step 4: Run tests**

Run: `pytest tests/test_crct_distributed.py::TestWeightMailboxBasics -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/crct_distributed.py tests/test_crct_distributed.py
git commit -m "crct_distributed: WeightMailbox flat-buffer snapshots + tests"
```

---

## Task 4: `WeightMailbox.post_weights` non-blocking on rank 0

Add a wall-clock test guarding the non-blocking contract. The flat-buffer broadcast IS already async via `async_op=True`; this task verifies the bound. If the test passes without further changes, no daemon thread is needed — `async_op=True` is sufficient. If it fails (e.g., the snapshot copy itself takes long enough to matter), add a daemon-thread shim around the copy + broadcast launch.

**Files:**
- Modify: `tests/test_crct_distributed.py`
- Possibly: `src/chaoscontrol/crct_distributed.py`

**Step 1: Write the failing/guarding test**

```python
import time

class TestWeightMailboxNonBlocking(unittest.TestCase):
    def test_post_weights_returns_under_50ms_with_slow_broadcast(self):
        """Even if the broadcast is mocked to never complete, post_weights
        must return promptly. Wall-clock budget is generous (50 ms) — the
        point is that we never block on the work handle."""
        m = _TinyModel()
        pg = FakeProcessGroup([0, 3])
        slow_work = FakeWork(completed=False)
        mb = crct.WeightMailbox(
            m, pg, sync_interval_steps=10, src_rank=0, my_rank=0,
            broadcast_fn=lambda *a, **kw: slow_work,
        )
        t0 = time.monotonic()
        mb.post_weights(global_step=10)
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.050)
        self.assertFalse(slow_work.is_completed())  # broadcast not awaited
```

**Step 2: Run test**

Run: `pytest tests/test_crct_distributed.py::TestWeightMailboxNonBlocking -v`
Expected: PASS (the existing impl is already non-blocking via async_op=True). If FAIL, add a `threading.Thread(target=…, daemon=True)` shim around the snapshot copy and broadcast launch.

**Step 3: Commit**

```bash
git add tests/test_crct_distributed.py
git commit -m "crct_distributed: pin WeightMailbox.post_weights non-blocking contract"
```

---

## Task 5: `TeacherResultMailbox` — post + try_get + payload schema

**Files:**
- Modify: `src/chaoscontrol/crct_distributed.py`
- Modify: `tests/test_crct_distributed.py`

**Step 1: Write the failing tests**

```python
from dataclasses import dataclass

class TestTeacherResultMailbox(unittest.TestCase):
    def test_try_get_returns_none_when_work_incomplete(self):
        pg = FakeProcessGroup([0, 1, 2, 3])
        pending = FakeWork(completed=False)
        mb = crct.TeacherResultMailbox(
            pg, my_rank=0, num_train_ranks=3,
            payload_shape=(2, 8, 16),
            broadcast_fn=lambda *a, **kw: pending,
        )
        self.assertIsNone(mb.try_get(step=100))

    def test_try_get_returns_payload_when_work_complete(self):
        pg = FakeProcessGroup([0, 1, 2, 3])
        done = FakeWork(completed=True)
        mb = crct.TeacherResultMailbox(
            pg, my_rank=0, num_train_ranks=3,
            payload_shape=(2, 8, 16),
            broadcast_fn=lambda *a, **kw: done,
        )
        # First call issues recv broadcast; mark completed and re-poll.
        first = mb.try_get(step=100)
        # depending on impl, may need second poke after marking complete
        payload = first if first is not None else mb.try_get(step=101)
        self.assertIsNotNone(payload)

    def test_post_result_drop_when_inflight(self):
        pg = FakeProcessGroup([0, 1, 2, 3])
        pending = FakeWork(completed=False)
        mb = crct.TeacherResultMailbox(
            pg, my_rank=3, num_train_ranks=3,
            payload_shape=(2, 8, 16),
            broadcast_fn=lambda *a, **kw: pending,
        )
        target = torch.zeros(2, 8, 16, dtype=torch.float16)
        conf = torch.zeros(2, 8, dtype=torch.float16)
        mb.post_result(step_id=10, target=target, conf=conf, loss_weight=1.0)
        # Second post while previous in-flight must drop, not queue.
        mb.post_result(step_id=11, target=target, conf=conf, loss_weight=1.0)
        self.assertEqual(mb.posts_attempted, 2)
        self.assertEqual(mb.posts_dropped, 1)
```

**Step 2: Run tests, expect FAIL with NotImplementedError.**

**Step 3: Implement (sketch)**

```python
@dataclass
class TeacherPayload:
    step_id: int
    target: torch.Tensor   # fp16 [B, T, C]
    conf: torch.Tensor     # fp16 [B, T]
    loss_weight: float
    snapshot_version: int


class TeacherResultMailbox:
    def __init__(self, crct_pg, *, my_rank, num_train_ranks,
                 payload_shape, dtype=torch.float16, queue_depth=1,
                 broadcast_fn=None):
        if int(queue_depth) != 1:
            raise NotImplementedError("queue_depth > 1 not in scope; see design §4")
        self.crct_pg = crct_pg
        self.my_rank = int(my_rank)
        self.num_train_ranks = int(num_train_ranks)
        self.payload_shape = payload_shape
        self.dtype = dtype
        self._broadcast = broadcast_fn or _dist.broadcast
        # Pre-allocate fixed-shape send/recv buffer.
        self._target_buf = torch.zeros(payload_shape, dtype=dtype)
        self._conf_buf = torch.zeros(payload_shape[:2], dtype=dtype)
        self._meta_buf = torch.zeros(2, dtype=torch.float32)  # (loss_weight, step_id-as-float64-view)
        self._inflight = None  # (work, payload-or-None)
        self.posts_attempted = 0
        self.posts_dropped = 0

    def post_result(self, step_id, target, conf, loss_weight, snapshot_version=0):
        assert self.my_rank == 3
        self.posts_attempted += 1
        if self._inflight is not None and not self._inflight[0].is_completed():
            self.posts_dropped += 1
            return
        self._target_buf.copy_(target)
        self._conf_buf.copy_(conf)
        self._meta_buf[0] = float(loss_weight)
        self._meta_buf[1] = float(step_id)
        # In production: one broadcast per buffer; for the unit, single
        # broadcast on _target_buf is enough to exercise the work-handle path.
        work = self._broadcast(
            self._target_buf, src=3, group=self.crct_pg, async_op=True,
        )
        self._inflight = (work, TeacherPayload(
            step_id=int(step_id),
            target=self._target_buf,
            conf=self._conf_buf,
            loss_weight=float(loss_weight),
            snapshot_version=int(snapshot_version),
        ))

    def try_get(self, step):
        if self.my_rank == 3:
            return None
        if self._inflight is None:
            work = self._broadcast(
                self._target_buf, src=3, group=self.crct_pg, async_op=True,
            )
            self._inflight = (work, None)
            return None
        work, _payload = self._inflight
        if not work.is_completed():
            return None
        payload = TeacherPayload(
            step_id=int(self._meta_buf[1].item()),
            target=self._target_buf.clone(),
            conf=self._conf_buf.clone(),
            loss_weight=float(self._meta_buf[0].item()),
            snapshot_version=0,
        )
        self._inflight = None
        return payload
```

**Step 4: Run tests, expect PASS.**

**Step 5: Commit**

```bash
git add src/chaoscontrol/crct_distributed.py tests/test_crct_distributed.py
git commit -m "crct_distributed: TeacherResultMailbox post/try_get + drop-don't-queue"
```

---

## Task 6: `fail_open_controller_term` helper + regression guard

The `0.0 * controller_logits.sum()` trick is non-obvious enough that callers will get it wrong. Centralize it in a tiny helper with a property test guarding the param-graph contract.

**Files:**
- Modify: `src/chaoscontrol/crct_distributed.py`
- Modify: `tests/test_crct_distributed.py`

**Step 1: Write the failing test**

```python
class TestFailOpenContract(unittest.TestCase):
    def test_fail_open_keeps_controller_head_in_autograd_graph(self):
        controller_head = nn.Linear(8, 4)
        x = torch.randn(2, 8)
        controller_logits = controller_head(x)
        lm_loss = torch.tensor(1.0, requires_grad=True)
        # payload is None -> fail-open path
        loss = crct.fail_open_controller_term(
            payload=None,
            controller_logits=controller_logits,
            lm_loss=lm_loss,
            lambda_ctrl=0.5,
            controller_loss_fn=lambda payload, logits: torch.zeros(()),
        )
        loss.backward()
        # Load-bearing: controller_head.weight.grad MUST materialize even
        # though payload is None — that is what keeps DDP's bucket set
        # stable across fail-open and fail-closed steps.
        self.assertIsNotNone(controller_head.weight.grad)
        self.assertTrue(torch.equal(
            controller_head.weight.grad,
            torch.zeros_like(controller_head.weight),
        ))
```

**Step 2: Run test, expect FAIL.**

**Step 3: Implement**

```python
def fail_open_controller_term(*, payload, controller_logits, lm_loss,
                              lambda_ctrl, controller_loss_fn):
    """Compose lm_loss + (controller_term or zero-product fallback).

    The 0.0 * controller_logits.sum() term keeps every parameter that
    produced controller_logits attached to the autograd graph. Without
    it, fail-open and fail-closed steps would have different grad sets
    and DDP's static parameter-readiness machinery would deadlock on
    the next bucket all-reduce. See design §4 fail-open contract.
    """
    if payload is None:
        return lm_loss + 0.0 * controller_logits.sum()
    return lm_loss + lambda_ctrl * controller_loss_fn(payload, controller_logits)
```

**Step 4: Run test, expect PASS.**

**Step 5: Commit**

```bash
git add src/chaoscontrol/crct_distributed.py tests/test_crct_distributed.py
git commit -m "crct_distributed: fail_open_controller_term + DDP-graph guard"
```

---

## Task 7: `Rank3MemoryCoprocessor` — causal order

Test the call order first, then the bounded-queue interaction.

**Files:**
- Modify: `src/chaoscontrol/crct_distributed.py`
- Modify: `tests/test_crct_distributed.py`

**Step 1: Write the failing tests**

```python
class TestCoprocessorCausalOrder(unittest.TestCase):
    def test_step_once_runs_phases_in_canonical_order(self):
        events = []

        # Stub mailboxes — minimal duck-typed objects, no real broadcast.
        class _Stub:
            posts_attempted = 0
            posts_dropped = 0
            def __init__(self, name): self.name = name
            def post_result(self, *a, **kw): events.append("teacher_post")
            def try_get(self, *a, **kw): return None
            def maybe_sync(self, *a, **kw): events.append("weight_sync")
            def post_weights(self, *a, **kw): events.append("weight_post")

        weight_mb = _Stub("weights")
        result_mb = _Stub("results")
        copro = crct.Rank3MemoryCoprocessor(
            model_copy=_TinyModel(),
            weight_mailbox=weight_mb,
            result_mailbox=result_mb,
            score_fn=lambda *a, **kw: events.append("score") or None,
            episodic_drain_fn=lambda: events.append("episodic_drain"),
            controller_tick_fn=lambda: events.append("controller_tick"),
            crct_pg=FakeProcessGroup([0, 1, 2, 3]),
            sync_pg=FakeProcessGroup([0, 3]),
            gather_fn=lambda *a, **kw: events.append("gather"),
        )
        copro.step_once(global_step=10)
        self.assertEqual(
            events,
            [
                "gather",            # step 1 of §3 data-flow
                "episodic_drain",
                "controller_tick",
                "weight_sync",
                # teacher reap is a no-op here (no in-flight); enqueue
                # happens because score_fn is provided -> triggers
                "score",
            ],
        )
```

**Step 2: Run, expect FAIL.**

**Step 3: Implement**

```python
class Rank3MemoryCoprocessor:
    def __init__(self, *, model_copy, weight_mailbox, result_mailbox,
                 score_fn, episodic_drain_fn, controller_tick_fn,
                 crct_pg, sync_pg, high_stream=None, low_stream=None,
                 gather_fn=None):
        self.model = model_copy
        self.weight_mailbox = weight_mailbox
        self.result_mailbox = result_mailbox
        self.score_fn = score_fn
        self.episodic_drain_fn = episodic_drain_fn
        self.controller_tick_fn = controller_tick_fn
        self.crct_pg = crct_pg
        self.sync_pg = sync_pg
        self.high_stream = high_stream
        self.low_stream = low_stream
        self._gather = gather_fn or _dist.gather
        self.metrics = {"teacher_drops": 0, "scores_run": 0}
        self._latest_batch = None
        self._inflight_score = None  # placeholder for completed-event handle

    def step_once(self, global_step):
        # Phase 1: receive batch from train ranks.
        batch = self._gather(group=self.crct_pg)
        self._latest_batch = batch
        # Phase 2: episodic write commits.
        self.episodic_drain_fn()
        # Phase 3: controller tick.
        self.controller_tick_fn()
        # Phase 4: maybe sync weights.
        self.weight_mailbox.maybe_sync(global_step)
        # Phase 5: reap teacher (no-op if nothing in flight).
        if self._inflight_score is not None and self._inflight_score.is_completed():
            # In real impl: post payload to result_mailbox. Stub: drop.
            self._inflight_score = None
        # Phase 6: maybe enqueue new score job.
        if self._inflight_score is None and self._latest_batch is not None:
            self._inflight_score = FakeWork(completed=True)  # placeholder
            self.score_fn(self.model, self._latest_batch)
            self.metrics["scores_run"] += 1
        elif self._inflight_score is not None:
            self.metrics["teacher_drops"] += 1
```

**Step 4: Run tests, expect PASS.**

**Step 5: Commit**

```bash
git add src/chaoscontrol/crct_distributed.py tests/test_crct_distributed.py
git commit -m "crct_distributed: Rank3MemoryCoprocessor causal-order scheduler"
```

---

## Task 8: Drop-don't-queue invariant on the coprocessor

**Step 1: Write the failing test**

```python
class TestCoprocessorDropDontQueue(unittest.TestCase):
    def test_step_once_skips_score_when_prior_inflight(self):
        score_calls = []
        copro = crct.Rank3MemoryCoprocessor(
            model_copy=_TinyModel(),
            weight_mailbox=_Stub("w"),
            result_mailbox=_Stub("r"),
            score_fn=lambda *a, **kw: score_calls.append(1),
            episodic_drain_fn=lambda: None,
            controller_tick_fn=lambda: None,
            crct_pg=FakeProcessGroup([0,1,2,3]),
            sync_pg=FakeProcessGroup([0,3]),
            gather_fn=lambda *a, **kw: object(),
        )
        # Force a never-completing in-flight score.
        copro._inflight_score = FakeWork(completed=False)
        copro.step_once(global_step=10)
        copro.step_once(global_step=11)
        self.assertEqual(len(score_calls), 0)
        self.assertGreaterEqual(copro.metrics["teacher_drops"], 2)
```

**Step 2: Run, expect FAIL or PASS depending on Task 7's stub. Adjust impl if needed.**

**Step 3: Commit**

```bash
git add tests/test_crct_distributed.py
git commit -m "crct_distributed: pin coprocessor drop-don't-queue invariant"
```

---

## Task 9: `rank3_coprocessor_loop` driver

**Files:**
- Modify: `src/chaoscontrol/crct_distributed.py`
- Modify: `tests/test_crct_distributed.py`

**Step 1: Write the failing test**

```python
class TestRank3Loop(unittest.TestCase):
    def test_loop_calls_step_once_until_stop_flag(self):
        ticks = []
        class _StubCopro:
            def step_once(self, step): ticks.append(step)
        flag = {"stop": False}
        # Stop after 3 iterations.
        def step_gen():
            for s in range(3):
                yield s
            flag["stop"] = True
        gen = step_gen()
        crct.rank3_coprocessor_loop(
            _StubCopro(),
            stop_flag=lambda: flag["stop"],
            step_iter=gen,
        )
        self.assertEqual(ticks, [0, 1, 2])

    def test_loop_no_op_when_world_size_below_4(self):
        # Sentinel: when crct_pg is None, the loop should return immediately.
        crct.rank3_coprocessor_loop(coprocessor=None, stop_flag=lambda: False,
                                    step_iter=iter(range(100)))
        # No assertion needed — must not raise, must not loop forever.
```

**Step 2: Run, expect FAIL.**

**Step 3: Implement**

```python
def rank3_coprocessor_loop(coprocessor, *, stop_flag, step_iter):
    """Drive Rank3MemoryCoprocessor until stop_flag returns True.

    Single-rank / world_size < 4 fallback: coprocessor is None ->
    return immediately; training ranks invoke score_fn synchronously
    via TeacherResultMailbox's same-process path.
    """
    if coprocessor is None:
        return
    for step in step_iter:
        if stop_flag():
            break
        coprocessor.step_once(step)
```

**Step 4: Run, expect PASS.**

**Step 5: Commit**

```bash
git add src/chaoscontrol/crct_distributed.py tests/test_crct_distributed.py
git commit -m "crct_distributed: rank3_coprocessor_loop driver + fallback no-op"
```

---

## Task 10: `world_size < 4` synchronous fallback through `TeacherResultMailbox`

The mailboxes must short-circuit when `crct_pg is None`. The fallback path lets training ranks call `score_fn` inline and have `try_get` return its result.

**Files:**
- Modify: `src/chaoscontrol/crct_distributed.py`
- Modify: `tests/test_crct_distributed.py`

**Step 1: Write the failing test**

```python
class TestSynchronousFallback(unittest.TestCase):
    def test_try_get_with_none_pg_invokes_inline_score_fn(self):
        score_calls = []
        def synchronous_score(input_ids, valid_mask, step):
            score_calls.append(step)
            return crct.TeacherPayload(
                step_id=step,
                target=torch.zeros(2, 8, 16, dtype=torch.float16),
                conf=torch.zeros(2, 8, dtype=torch.float16),
                loss_weight=1.0,
                snapshot_version=0,
            )
        mb = crct.TeacherResultMailbox(
            crct_pg=None, my_rank=0, num_train_ranks=1,
            payload_shape=(2, 8, 16),
            inline_score_fn=synchronous_score,
        )
        payload = mb.try_get_with_input(
            step=42, input_ids=torch.zeros(2, 8, dtype=torch.long),
            valid_mask=torch.ones(2, 8, dtype=torch.bool),
        )
        self.assertEqual(score_calls, [42])
        self.assertEqual(payload.step_id, 42)
```

**Step 2: Run, expect FAIL.**

**Step 3: Implement — extend `TeacherResultMailbox.__init__` to accept `inline_score_fn` when `crct_pg is None`, and add `try_get_with_input` which is the inline path's name (distinct so callers can't accidentally call distributed `try_get` on a None pg).**

```python
# Patch TeacherResultMailbox:
def __init__(self, crct_pg, *, my_rank, num_train_ranks,
             payload_shape, dtype=torch.float16, queue_depth=1,
             broadcast_fn=None, inline_score_fn=None):
    if crct_pg is None:
        if inline_score_fn is None:
            raise ValueError(
                "world_size < 4 fallback requires inline_score_fn"
            )
        self._inline_score = inline_score_fn
        self.crct_pg = None
        self.my_rank = int(my_rank)
        return
    # ... existing distributed path ...

def try_get_with_input(self, step, input_ids, valid_mask):
    if self.crct_pg is None:
        return self._inline_score(input_ids, valid_mask, step)
    raise RuntimeError("try_get_with_input is for fallback path only; "
                       "use try_get(step) under distributed CRCT.")
```

**Step 4: Run all tests:**

`pytest tests/test_crct_distributed.py -v`
Expected: all PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/crct_distributed.py tests/test_crct_distributed.py
git commit -m "crct_distributed: synchronous fallback via inline_score_fn"
```

---

## Task 11: Final consolidation — full-suite smoke + lint

**Step 1: Run the entire `crct_distributed` test file**

Run: `pytest tests/test_crct_distributed.py -v`
Expected: all 8+ tests PASS.

**Step 2: Run any project-wide tests that touch distributed.py to confirm no regression:**

Run: `pytest tests/test_distributed_allreduce_grads.py -v`
Run: `pytest tests/test_runner_3plus1_skip_main.py -v`
Expected: PASS (CRCT did not modify `distributed.py` at all; this is a sanity check).

**Step 3: Verify the design doc and plan are both committed and the module exposes everything the design promised.**

Run: `git log --oneline -15`
Expected output should include:
- design doc commit
- module skeleton commit
- one commit per task

**Step 4: Final commit cleanup if any docs need cross-references.**

If the design doc has any references that have drifted, update and commit:

```bash
git add docs/plans/2026-04-27-crct-distributed-design.md
git commit -m "docs: cross-reference CRCT design with implementation"
```

---

## Out of scope (tracked separately)

These are intentionally NOT in this plan:

- **gloo-CPU 4-rank end-to-end ride-along** — pattern lives in `tests/test_runner_3plus1_skip_main.py`. Belongs in the eventual exp-26 runner PR alongside the wiring code, not in this transport module.
- **Score-function / oracle / scarcity math** — caller's responsibility. CRCT_distributed treats `score_fn`, `episodic_drain_fn`, and `controller_tick_fn` as opaque callables.
- **Stream-priority observability under load** — needs real CUDA + nsys, belongs in a perf-pass doc.
- **Runner integration** — wiring `Rank3MemoryCoprocessor` into a fork of `runner_fast_path.py` for the experiment 26 launch is a separate plan in a separate PR.
- **WeightMailbox per-tensor name index** — current impl uses `requires_grad` flat order, fine when train and rank-3 model copies share parameter declaration order. Buffer-name validation is a nice-to-have for the runner-integration PR.
