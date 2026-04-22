# Exp 24 Rigor-and-Speed Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Before starting the first code task, run @superpowers:receiving-code-review on the plan itself and fix anything it flags (per project memory: dry-run plan review before executing).

**Goal:** Add five load-bearing regression tests for the `event_sleep` training-time mechanism and land measured kernel-level speedups on the `fast_slow_dreamworld_event_sleep` arm, inside one session on 1 H100.

**Architecture:** Event-sleep tests move into `tests/test_exp24_event_sleep.py` so the Exp 23 fast-path test file stays generic-substrate. The runner's inside-600s-budget hotspots get audited against a 1 H100 profile; kernel wins land with before/after wall-clock numbers. ScOpt and Muon are out of scope for speedup (off-clock and not in the loop respectively).

**Tech Stack:** Python 3.12, PyTorch 2.x, `torch._foreach_*` multi-tensor ops, `torch.multiprocessing.spawn` with gloo backend, `torch.cuda.Event` for profiling, pytest, existing `_lm_head_loss` and `_ssm_scan` kernels (not modified).

---

## File Structure

- Create `tests/test_exp24_event_sleep.py`: all event_sleep regression coverage.
- Create `experiments/24_training_time_bundle/profile_event_sleep_arm.py`: 1 H100 profile harness for `fast_slow_dreamworld_event_sleep`.
- Create `docs/plans/2026-04-22-exp24-runner-profile.md`: profile ranking and per-candidate verdicts.
- Modify `tests/test_exp23_fast_path.py`: remove three event_sleep tests after copy.
- Modify `experiments/23_fast_path/training_hooks.py`: `FastSlowConsolidator.after_optimizer_step` → `torch._foreach_lerp_`.
- Modify `experiments/23_fast_path/runner_fast_path.py`: `LossTriggeredReplayEMA` on-device EMA refactor; other kernel-candidate edits conditional on profile verdicts.
- Modify `docs/plans/2026-04-22-exp24-rigor-and-speed-design.md`: append a short "results" section linking to the profile doc after landing.

## Phase 1 - Test migration

### Task 1: Create `tests/test_exp24_event_sleep.py` and move existing event_sleep tests

**Files:**
- Create: `tests/test_exp24_event_sleep.py`
- Modify: `tests/test_exp23_fast_path.py` (delete three tests)

**Tests to move:**

1. `test_event_sleep_gate_reduces_rank_loss_pressure` (gate plumbing; will be replaced in Task 2)
2. `test_event_sleep_gate_waits_for_warmup_and_threshold` (gate warmup/threshold)
3. `test_train_fast_for_budget_runs_event_sleep_replay` (loop integration)

**Step 1: Create the new file skeleton**

```python
"""Regression coverage for the Exp 24 `event_sleep` training-time mechanism.

The hot-loop implementation lives with the Exp 23 fast-path runner
(`experiments/23_fast_path/runner_fast_path.py`) which is the shared
training substrate. This file owns the Exp 24-specific invariants:
DDP lockstep, one-step delay, seed-anchored trigger trajectory,
warmup-restore bit equality, and bf16 loss handling.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


_RUNNER_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "23_fast_path"
    / "runner_fast_path.py"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "runner_fast_path_event_sleep", _RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

- [ ] Create the file with the header and `_load_runner_module` helper above.

**Step 2: Move the two gate tests**

- [ ] Copy `test_event_sleep_gate_waits_for_warmup_and_threshold` from `tests/test_exp23_fast_path.py` into the new file verbatim.
- [ ] Copy `test_event_sleep_gate_reduces_rank_loss_pressure` into the new file verbatim (it will be deleted in Task 2).

**Step 3: Move the loop test**

- [ ] Copy `test_train_fast_for_budget_runs_event_sleep_replay` into the new file. It depends on a `_TinyTokenTrainModel` helper and `_load_runner_module`; inline the model class into the new file (copy its definition from `test_exp23_fast_path.py`) so the new file does not import from the old one.

**Step 4: Delete from old file**

- [ ] Delete all three tests from `tests/test_exp23_fast_path.py`.

**Step 5: Run both test files**

```bash
pytest tests/test_exp23_fast_path.py tests/test_exp24_event_sleep.py -q
```

Expected: both files pass, no import errors, no duplicate test names.

**Step 6: Commit**

```bash
git add tests/test_exp24_event_sleep.py tests/test_exp23_fast_path.py
git commit -m "tests: move event_sleep regression tests to exp24 file"
```

## Phase 2 - Five supercar tests

Every Phase 2 task is TDD-shaped: write test, run expecting either pass (coverage added for behavior that already works) or fail (bug found - fix before proceeding), then commit.

### Task 2: Real multi-rank gloo DDP lockstep test (replaces fake-reduce test)

**Files:**
- Modify: `tests/test_exp24_event_sleep.py`

**Step 1: Delete the fake-reduce test**

- [ ] Remove `test_event_sleep_gate_reduces_rank_loss_pressure` from `tests/test_exp24_event_sleep.py`. Its assertion surface is fully subsumed by the real DDP test below.

**Step 2: Add the gloo multi-rank test**

Add this test to `tests/test_exp24_event_sleep.py`:

```python
import os
import tempfile

import torch.distributed as dist
import torch.multiprocessing as mp


def _gate_worker(rank, world_size, init_file, per_rank_loss, result_path):
    """Spawned-process worker: init gloo, run one gate update, write result."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "0"  # gloo ignores; init_file carries rendezvous
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mod = _load_runner_module()
        gate = mod.LossTriggeredReplayEMA(decay=0.5, warmup_steps=1)
        # Seed the EMA so the next update is post-warmup.
        gate.update(torch.tensor(per_rank_loss[0]))
        decision = gate.update(
            torch.tensor(per_rank_loss[1]),
            threshold=1.10,
            pressure_threshold=0.05,
            ddp_active=True,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        payload = {
            "triggered": bool(decision.triggered),
            "global_pressure": float(decision.global_pressure),
            "fire_count": int(decision.fire_count),
            "local_fire": bool(decision.local_fire),
            "local_ratio": float(decision.local_ratio),
        }
        torch.save(payload, f"{result_path}.rank{rank}")
    finally:
        dist.destroy_process_group()


def test_gloo_four_rank_lockstep_and_boundary(tmp_path):
    """All ranks see identical triggered / global_pressure / fire_count.

    Also covers the `ratio == threshold` boundary on rank 2: its local_fire
    must be False because local_pressure = max(0, ratio - threshold) = 0.
    """
    world_size = 4
    # First loss seeds the EMA (decay=0.5 means ema_after = 2.0 for all ranks).
    # Second loss drives the ratio; rank 2 sits exactly at threshold.
    per_rank_loss = {
        0: (2.0, 2.60),  # ratio 1.30, above threshold 1.10: fires
        1: (2.0, 2.00),  # ratio 1.00, below: no fire
        2: (2.0, 2.20),  # ratio 1.10, exactly at threshold: no fire (boundary)
        3: (2.0, 2.80),  # ratio 1.40, above: fires
    }

    result_path = tmp_path / "decisions"
    init_file = tmp_path / "rendezvous"
    init_file.touch()

    mp.spawn(
        _gate_worker,
        args=(world_size, str(init_file), None, str(result_path)),
        nprocs=world_size,
        join=True,
    )

    decisions = [
        torch.load(f"{result_path}.rank{r}") for r in range(world_size)
    ]

    # Per-rank local_fire must match the pattern above.
    expected_local_fire = [True, False, False, True]
    for rank, dec in enumerate(decisions):
        assert dec["local_fire"] is expected_local_fire[rank], (
            f"rank {rank}: local_fire {dec['local_fire']} != "
            f"{expected_local_fire[rank]}"
        )

    # Rank 2 boundary: ratio exactly 1.10 -> local_pressure exactly 0 -> no fire.
    assert abs(decisions[2]["local_ratio"] - 1.10) < 1e-6
    assert decisions[2]["local_fire"] is False

    # Lockstep: every rank must agree on the global decision fields.
    fire_count_ref = decisions[0]["fire_count"]
    global_pressure_ref = decisions[0]["global_pressure"]
    triggered_ref = decisions[0]["triggered"]
    assert fire_count_ref == 2, "ranks 0 and 3 fired -> fire_count = 2"
    for rank, dec in enumerate(decisions):
        assert dec["fire_count"] == fire_count_ref
        assert dec["global_pressure"] == pytest.approx(global_pressure_ref)
        assert dec["triggered"] is triggered_ref
```

A note on the worker signature: the fourth positional arg is unused (`None`) because `_gate_worker` does not take `per_rank_loss` via mp.spawn directly; each rank constructs its own loss from the `rank` index. The cleaner version passes a literal dict through; amend the worker if the linter complains.

Actually, amend the worker to receive the per-rank loss map:

```python
def _gate_worker(rank, world_size, init_file, per_rank_loss_map, result_path):
    # ... same body, but use `per_rank_loss_map[rank]` instead of `per_rank_loss`
```

And update the call:

```python
    mp.spawn(
        _gate_worker,
        args=(world_size, str(init_file), per_rank_loss, str(result_path)),
        nprocs=world_size,
        join=True,
    )
```

- [ ] Add the worker function and test to the file with the dict-threaded version.

**Step 3: Run the test**

```bash
pytest tests/test_exp24_event_sleep.py::test_gloo_four_rank_lockstep_and_boundary -v
```

Expected: PASS. Total runtime should be under 10 seconds on a laptop.

If it fails: check `dist.init_process_group` file permissions and MASTER_PORT resolution. The memory entry `feedback_mp_spawn_flake.md` flags occasional hangs after a killed prior pytest - rerun once.

**Step 4: Commit**

```bash
git add tests/test_exp24_event_sleep.py
git commit -m "tests: add real gloo 4-rank event_sleep lockstep test"
```

### Task 3: One-step-delay invariant test

**Files:**
- Modify: `tests/test_exp24_event_sleep.py`

**Step 1: Add the test**

The test forces a trigger at step K and asserts the very next `dream_buffer.sample` call happens at step K+1. We instrument `DreamReplayBuffer` with a real buffer seeded with one entry so `len(buffer) >= min_size`, let the gate fire on a specific step, and record sample-call step indices.

```python
def test_event_sleep_replay_lands_one_step_after_trigger(monkeypatch):
    mod = _load_runner_module()

    sampled_steps: list[int] = []

    class InstrumentedBuffer(mod.DreamReplayBuffer):
        def sample(self, *, generator, current_step):
            sampled_steps.append(int(current_step))
            return super().sample(generator=generator, current_step=current_step)

    monkeypatch.setattr(mod, "DreamReplayBuffer", InstrumentedBuffer)

    # Gate that fires exactly at step index 5 (0-indexed), never before.
    fire_steps = {5}

    class ScriptedGate:
        def __init__(self, **kwargs):
            self._step = 0

        def update(self, loss, **kwargs):
            step = self._step
            self._step += 1
            if step in fire_steps:
                return mod.LossTriggeredReplayDecision(
                    local_loss=1.0,
                    ema_loss=1.0,
                    local_ratio=1.3,
                    local_pressure=0.2,
                    global_pressure=0.2,
                    fire_count=1,
                    local_fire=True,
                    triggered=True,
                )
            return mod.LossTriggeredReplayDecision(
                local_loss=1.0,
                ema_loss=1.0,
                local_ratio=0.95,
                local_pressure=0.0,
                global_pressure=0.0,
                fire_count=0,
                local_fire=False,
                triggered=False,
            )

    monkeypatch.setattr(mod, "LossTriggeredReplayEMA", ScriptedGate)

    # Cheap no-op capture and replay so the loop exercises event_sleep dispatch.
    monkeypatch.setattr(
        mod,
        "capture_dream_entry",
        lambda model, inputs, **kwargs: type(
            "E",
            (),
            {
                "step": kwargs["step"],
                "states": [torch.zeros(inputs.size(0), 4)],
                "replay_tokens": inputs[:, :3],
            },
        )(),
    )
    monkeypatch.setattr(
        mod,
        "dreamworld_replay_backward",
        lambda *a, **k: torch.tensor(0.1),
    )

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=6,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=10,
        prefetch_batches=False,
        dreamworld_enabled=True,
        dreamworld_cache_interval=1,
        dreamworld_interval=0,  # scheduled replay disabled
        dreamworld_weight=0.25,
        dreamworld_prefix_tokens=3,
        dreamworld_replay_tokens=2,
        dreamworld_min_size=1,
        event_sleep_enabled=True,
        event_sleep_min_interval=1,
        event_sleep_weight=0.5,
    )

    # Trigger happened at step 5; replay sample must be at step 6. No sample
    # at step 5 itself. No sample at any other step (scheduled replay off).
    assert 5 not in sampled_steps, (
        f"replay leaked into trigger step: {sampled_steps}"
    )
    assert 6 in sampled_steps, (
        f"replay did not land one step after trigger: {sampled_steps}"
    )
    assert sampled_steps == [6], (
        f"replay fired on unexpected steps: {sampled_steps}"
    )
```

- [ ] Add the test to the file.

**Step 2: Run**

```bash
pytest tests/test_exp24_event_sleep.py::test_event_sleep_replay_lands_one_step_after_trigger -v
```

Expected: PASS. If it fails with `sampled_steps == [5]` or similar, the replay branch in `train_fast_for_budget` is evaluating the trigger too early - a real bug. Fix by ensuring the gate `update` call happens *after* the step's `_run_scopt_train_step` so the decision comes from step N's completed loss, and the resulting `event_sleep_pending = True` is checked at the top of step N+1.

**Step 3: Commit**

```bash
git add tests/test_exp24_event_sleep.py
git commit -m "tests: pin event_sleep one-step-delay invariant"
```

### Task 4: Seed-anchored trigger trajectory test

**Files:**
- Modify: `tests/test_exp24_event_sleep.py`

**Step 1: Add the test**

Feed a crafted loss sequence into the bare `LossTriggeredReplayEMA` and assert the exact list of step indices that trigger. This is a tight regression sentinel against EMA order-of-operations bugs.

```python
def test_gate_produces_exact_trigger_trajectory():
    mod = _load_runner_module()

    # 40-step loss sequence with three deliberate spikes at steps 12, 24, 35.
    # Between spikes the loss drifts slowly down so the EMA tracks it.
    losses = []
    base = 2.5
    for step in range(40):
        base = 0.995 * base  # slow exponential decay
        spike = 0.0
        if step in (12, 24, 35):
            spike = 0.8
        losses.append(base + spike)

    gate = mod.LossTriggeredReplayEMA(decay=0.9, warmup_steps=5)

    fired_steps: list[int] = []
    for step, loss in enumerate(losses):
        decision = gate.update(
            torch.tensor(loss),
            threshold=1.10,
            pressure_threshold=0.02,
            ddp_active=False,
            world_size=1,
            device=torch.device("cpu"),
        )
        if decision is not None and decision.triggered:
            fired_steps.append(step)

    # Recomputed manually from the sequence above; this anchor must match
    # exactly. If it drifts, inspect the change to decay/warmup/threshold or
    # to the ratio computation order inside update().
    assert fired_steps == [12, 24, 35], (
        f"trigger trajectory drifted: got {fired_steps}, expected [12, 24, 35]"
    )
```

- [ ] Add the test.

**Step 2: Run**

```bash
pytest tests/test_exp24_event_sleep.py::test_gate_produces_exact_trigger_trajectory -v
```

Expected: PASS. If it fails: write the actual `fired_steps` into the test as the new anchor ONLY if you understand why the trajectory changed (e.g. you intentionally modified the EMA order). Silent anchor updates defeat the point of this test.

**Step 3: Commit**

```bash
git add tests/test_exp24_event_sleep.py
git commit -m "tests: seed-anchor event_sleep trigger trajectory"
```

### Task 5: Warmup-restore bit-equality test

**Files:**
- Modify: `tests/test_exp24_event_sleep.py`

**Step 1: Add the test**

The Parameter Golf warmup contract is bit-exact. Run `_warmup` with `restore_after_warmup=True`, snapshot state before, assert `torch.equal` against every restored tensor after.

```python
def test_warmup_restore_is_bit_equal():
    mod = _load_runner_module()

    model = _TinyTokenTrainModel()

    # Snapshot pre-warmup state (deep clone so the model can mutate freely).
    pre_state = {
        name: param.detach().clone() for name, param in model.state_dict().items()
    }

    # Drive _warmup with a restore_after_warmup semantics call. The actual
    # production call path is inside run_condition(); here we exercise the
    # primitive directly.
    config = {
        "warmup_steps": 4,
        "restore_after_warmup": True,
        "seed": 1337,
    }
    mod._warmup(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=6,
        batch_size=2,
        device=torch.device("cpu"),
        config=config,
        vocab_size=6,
    )
    # _warmup should have restored state internally when restore_after_warmup is True.

    post_state = model.state_dict()

    for name, pre_tensor in pre_state.items():
        post_tensor = post_state[name]
        assert torch.equal(pre_tensor, post_tensor), (
            f"param {name} drifted after warmup-restore: "
            f"max diff {(pre_tensor - post_tensor).abs().max().item()}"
        )
```

Before writing the test: read `_warmup` at `experiments/23_fast_path/runner_fast_path.py:1900` and the `restore_after_warmup` handling at line 2016, then adjust the call signature above to match. The test body above is the assertion contract; the exact call shape depends on the private function's kwargs.

- [ ] Read `_warmup` and `_restore_state_dict` source.
- [ ] Add the test, adjusting the call signature to match.

**Step 2: Run**

```bash
pytest tests/test_exp24_event_sleep.py::test_warmup_restore_is_bit_equal -v
```

Expected: PASS. If it fails: the warmup path is mutating state that `_restore_state_dict` does not roll back - optimizer state, RNG, or buffers. Fix by snapshotting and restoring whatever is mutating.

**Step 3: Commit**

```bash
git add tests/test_exp24_event_sleep.py
git commit -m "tests: pin warmup-restore bit equality contract"
```

### Task 6: bf16 loss dtype test

**Files:**
- Modify: `tests/test_exp24_event_sleep.py`

**Step 1: Add the test**

```python
def test_gate_handles_bf16_loss_tensor():
    mod = _load_runner_module()

    losses_fp32 = [2.5, 2.48, 2.46, 2.45, 2.44, 3.20]
    losses_bf16 = [
        torch.tensor(v, dtype=torch.bfloat16).float().item() for v in losses_fp32
    ]  # round-trip through bf16 so the comparison uses the same values

    gate_fp32 = mod.LossTriggeredReplayEMA(decay=0.5, warmup_steps=1)
    gate_bf16 = mod.LossTriggeredReplayEMA(decay=0.5, warmup_steps=1)

    fp32_decisions = []
    bf16_decisions = []
    for loss_fp32, loss_bf16 in zip(losses_fp32, losses_bf16, strict=True):
        d_fp32 = gate_fp32.update(
            torch.tensor(loss_fp32, dtype=torch.float32),
            threshold=1.10,
            pressure_threshold=0.05,
        )
        d_bf16 = gate_bf16.update(
            torch.tensor(loss_bf16, dtype=torch.bfloat16),
            threshold=1.10,
            pressure_threshold=0.05,
        )
        fp32_decisions.append(d_fp32)
        bf16_decisions.append(d_bf16)

    for d_fp32, d_bf16 in zip(fp32_decisions, bf16_decisions, strict=True):
        if d_fp32 is None:
            assert d_bf16 is None
            continue
        assert d_bf16 is not None
        assert d_fp32.triggered == d_bf16.triggered
        assert d_fp32.local_fire == d_bf16.local_fire
        # local_loss is a float inside the gate; must be finite and not nan.
        assert torch.isfinite(torch.tensor(d_bf16.local_loss))
        assert torch.isfinite(torch.tensor(d_bf16.ema_loss))
```

- [ ] Add the test.

**Step 2: Run**

```bash
pytest tests/test_exp24_event_sleep.py::test_gate_handles_bf16_loss_tensor -v
```

Expected: PASS. If `d_fp32.triggered != d_bf16.triggered` for some step, bf16's reduced precision has pushed the ratio across the threshold. That is legitimate numerical behavior only if the ratio is within 1e-3 of threshold; otherwise investigate whether `loss.detach().float().item()` at `runner_fast_path.py:573` is silently losing precision.

**Step 3: Commit**

```bash
git add tests/test_exp24_event_sleep.py
git commit -m "tests: pin event_sleep gate bf16 loss dtype path"
```

## Phase 3 - Profile

### Task 7: Build profile harness and run on 1 H100

**Files:**
- Create: `experiments/24_training_time_bundle/profile_event_sleep_arm.py`
- Create: `docs/plans/2026-04-22-exp24-runner-profile.md`

**Step 1: Write the profile script**

Create `experiments/24_training_time_bundle/profile_event_sleep_arm.py` with CUDA-event timings around the train step's major sections. The script should: load a minimal model, construct a single-seed `fast_slow_dreamworld_event_sleep` config, run ~30 seconds (or 500 steps, whichever is first), emit a JSON with per-section wall-clock milliseconds and per-section share of the inside-budget total.

Sections to time (wrap each in a `torch.cuda.Event` pair with `elapsed_time`):

- `encode_forward` - `model.encode(...)` call
- `logits_and_loss` - LM head + CE
- `backward` - `loss.backward()`
- `spectral_reg` - `spectral_regularization_loss(...)` call if enabled
- `predictive_aux` - aux loss computation if enabled
- `dreamworld_replay` - `dreamworld_replay_backward(...)` call on replay steps
- `event_sleep_gate` - `LossTriggeredReplayEMA.update(...)` call
- `optimizer_step` - `optimizer.step()` + `optimizer.zero_grad()`
- `fast_slow_ema` - `fast_slow.after_optimizer_step(...)` call

Emit results to `experiments/24_training_time_bundle/profile_event_sleep_arm_out.json` with the schema:

```json
{
  "arm": "fast_slow_dreamworld_event_sleep",
  "steps": 500,
  "total_inside_budget_ms": 12345.6,
  "sections": [
    {"name": "encode_forward", "mean_ms": 2.1, "share": 0.34},
    ...
  ]
}
```

- [ ] Write the profile script.
- [ ] Run locally with `--steps 10` on CPU as a smoke test.

**Step 2: Run on 1 H100**

On the pod (remember `source /workspace/venv/bin/activate` first per project CLAUDE.md):

```bash
source /workspace/venv/bin/activate
python experiments/24_training_time_bundle/profile_event_sleep_arm.py --steps 500
```

Expected: JSON output file, total runtime under 2 minutes.

- [ ] Run on 1 H100.
- [ ] Copy JSON output back locally.

**Step 3: Write up ranking**

Create `docs/plans/2026-04-22-exp24-runner-profile.md` with:

- Total inside-budget wall-clock per step (mean and p50/p95).
- Ranked hotspot table (name, mean ms, share %).
- Initial verdict per candidate from the design doc: "landing-target", "needs-audit", "negligible-skip".

- [ ] Write the report.

**Step 4: Commit**

```bash
git add experiments/24_training_time_bundle/profile_event_sleep_arm.py \
        experiments/24_training_time_bundle/profile_event_sleep_arm_out.json \
        docs/plans/2026-04-22-exp24-runner-profile.md
git commit -m "exp24: profile fast_slow_dreamworld_event_sleep on 1 H100"
```

## Phase 4 - Kernel-level speedups

All Phase 4 tasks are conditional on the profile ranking. Tasks marked "lands regardless" execute even if their profile share is small because they are cheap and correct.

### Task 8: `FastSlowConsolidator.after_optimizer_step` → `torch._foreach_lerp_`

**Lands regardless** (small but inside budget and the Python-loop pattern is already visible in source).

**Files:**
- Modify: `experiments/23_fast_path/training_hooks.py:43-55`
- Modify: `tests/test_exp24_training_hooks.py` (or the existing file that covers FastSlowConsolidator)

**Step 1: Write the equivalence test first**

Before changing the implementation, add a test that two `FastSlowConsolidator` instances with identical configs produce bit-equal `slow_state` after N model steps, one using the current loop and one using a reference loop. This guards against silent divergence when we swap the loop for `_foreach_lerp_`.

Actually simpler: the test asserts the *current* behavior against a hand-computed reference, so the replaced implementation has a target to match.

```python
def test_fastslow_after_optimizer_step_matches_reference_lerp():
    import torch
    from experiments._23_fast_path.training_hooks import FastSlowConsolidator
    # import path may differ; adjust after reading training_hooks.py

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    consolidator = FastSlowConsolidator.from_config(
        model, {"fast_slow_enabled": True, "fast_slow_interval": 1, "fast_slow_alpha": 0.1}
    )

    # Capture initial slow state.
    pre_slow = {name: t.clone() for name, t in consolidator.slow_state.items()}

    # Mutate model params.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))

    consolidator.after_optimizer_step(model, step=1)

    model_params = dict(model.named_parameters())
    for name, pre_slow_t in pre_slow.items():
        expected = torch.lerp(pre_slow_t, model_params[name].detach(), 0.1)
        actual = consolidator.slow_state[name]
        assert torch.equal(actual, expected), (
            f"{name} lerp mismatch: max diff "
            f"{(actual - expected).abs().max().item()}"
        )
```

- [ ] Locate the existing FastSlowConsolidator test file (`tests/test_exp24_training_hooks.py` is likely).
- [ ] Add the equivalence test.
- [ ] Run it against the current implementation; should PASS.

**Step 2: Refactor `after_optimizer_step` to `_foreach_lerp_`**

Replace lines 47-55 of `experiments/23_fast_path/training_hooks.py`:

```python
    def after_optimizer_step(self, model: torch.nn.Module, *, step: int) -> None:
        if not self.enabled or self.interval <= 0 or step % self.interval != 0:
            return

        model_params = dict(model.named_parameters())
        slow_list: list[torch.Tensor] = []
        fast_list: list[torch.Tensor] = []
        for name, slow_param in self.slow_state.items():
            model_param = model_params.get(name)
            if model_param is None:
                continue
            # _foreach_lerp_ requires matching dtype/device across the list.
            # Fast param must be on the same device/dtype as slow param.
            if (
                model_param.device != slow_param.device
                or model_param.dtype != slow_param.dtype
            ):
                # Fall back to per-tensor lerp for mismatched tensors.
                with torch.no_grad():
                    slow_param.lerp_(model_param.detach(), self.alpha)
                continue
            slow_list.append(slow_param)
            fast_list.append(model_param.detach())

        if slow_list:
            with torch.no_grad():
                torch._foreach_lerp_(slow_list, fast_list, self.alpha)

        self.sync_count += 1
```

- [ ] Apply the refactor.

**Step 3: Run equivalence + full training_hooks tests**

```bash
pytest tests/test_exp24_training_hooks.py -v
```

Expected: PASS including the new equivalence test.

**Step 4: Measure**

Re-run the profile script (Task 7) and compare `fast_slow_ema` section mean ms before/after. Append the delta to `docs/plans/2026-04-22-exp24-runner-profile.md` under a "Verdicts" section.

- [ ] Re-run profile.
- [ ] Write delta in verdicts section.

**Step 5: Commit**

```bash
git add experiments/23_fast_path/training_hooks.py tests/test_exp24_training_hooks.py \
        docs/plans/2026-04-22-exp24-runner-profile.md
git commit -m "exp23: fast_slow EMA via torch._foreach_lerp_"
```

### Task 9: Event-sleep EMA on-device refactor

**Lands regardless** (small sync per step inside budget; reviewer-visible fix).

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py:537-607` (LossTriggeredReplayEMA)

**Step 1: Update the EMA to keep state on the loss tensor's device**

Current `LossTriggeredReplayEMA` stores `self.value` as a Python float and calls `.item()` on the loss every step. The refactor keeps `self.value` as a same-device tensor and only syncs when the DDP all-reduce already requires a sync.

Key change outline:

- `self.value: torch.Tensor | None = None` instead of `float | None`.
- First update: `self.value = loss.detach().float()` (no `.item()`).
- Subsequent updates: compute ratio / pressure as tensor ops on-device.
- The `all_reduce` path already requires a device tensor; assemble `[local_pressure, float(local_fire)]` directly from tensor ops.
- Single `.item()` calls remain at the very end to fill the `LossTriggeredReplayDecision` dataclass fields (which are declared as Python floats). Those syncs are effectively the same as the existing ones for `global_pressure` and `fire_count`.

The net win: one fewer GPU→CPU sync per step at the top of `update()`, and the ratio compare happens on-device.

- [ ] Implement the refactor.

**Step 2: Run existing event_sleep tests**

```bash
pytest tests/test_exp24_event_sleep.py -v
```

Expected: all 5 tests PASS, including bf16 and the seed-anchored trajectory. If the trajectory test fails, the on-device EMA may accumulate differently than the float path. Either make the computations bit-match (e.g. keep all operations in `float32` on-device) or update the anchor with a clear note.

**Step 3: Measure**

Re-run the profile script. Compare `event_sleep_gate` section mean ms before/after. Append delta.

- [ ] Re-run profile.
- [ ] Write delta.

**Step 4: Commit**

```bash
git add experiments/23_fast_path/runner_fast_path.py docs/plans/2026-04-22-exp24-runner-profile.md
git commit -m "exp23: keep LossTriggeredReplayEMA state on device"
```

### Task 10: Spectral regularizer vectorization (conditional)

**Condition:** Execute only if profile shows `spectral_reg` section > 1% of inside-budget wall clock. Otherwise mark "negligible - skipped" in the verdicts section and move on.

**Files (if executing):**
- Modify: `experiments/23_fast_path/training_hooks.py:96-115`

**Step 1: Read the current implementation**

`spectral_regularization_loss` at line 96 iterates `iter_log_a_params`, computes a per-layer penalty (relu-hinges + pow(2)), and `torch.stack(penalties).mean()` at the end.

The vectorizable observation: if all `log_a` tensors have the same dtype (they do; all are float parameters), we can `torch.cat([log_a for _, log_a in iter_log_a_params(model)])` once, compute the full hinge loss on the concatenated tensor, and mean across all elements.

Note the current behavior computes `penalty.mean()` per layer and then means across layers. A direct concat+mean is *not* equivalent if layers have different sizes - it weights by element count instead of by layer. Preserve the current semantics: mean-per-layer then mean-across-layers.

**Step 2: Write equivalence test**

```python
def test_spectral_reg_vectorized_matches_reference():
    import torch
    from experiments._23_fast_path.training_hooks import spectral_regularization_loss
    # ... build a model with multiple SSM layers of varying size, compute old
    # and new spectral loss, assert torch.equal on both.
```

- [ ] Write the test.

**Step 3: Refactor preserving semantics**

Use `torch.nested` tensors or a Python-level prescan to compute per-layer means then stack, which can still be a single fused op chain. Alternative: if mean-per-layer is numerically fine to approximate as mean-across-elements (investigate whether the layers are same-sized in practice), simplify and document the change.

- [ ] Implement.
- [ ] Run tests.

**Step 4: Measure and commit**

- [ ] Re-run profile.
- [ ] Commit with numbers.

### Task 11: Graph-break audit on `_run_scopt_train_step` (conditional)

**Condition:** Execute if profile shows the combined `encode_forward + logits_and_loss + backward` share is above 60% AND `torch._logging.set_logs(graph_breaks=True)` reveals breaks inside hot sections.

**Files (if executing):**
- Modify: `experiments/23_fast_path/runner_fast_path.py` (various, depending on findings)

**Step 1: Enable graph break logging**

```bash
TORCH_LOGS=graph_breaks python -c "
import torch
# set up a one-step call to _run_scopt_train_step and run it
"
```

- [ ] Run graph-break logging on one train step.
- [ ] Document each break with file:line and reason.

**Step 2: Fix the cheap ones**

- `.item()` calls inside the compiled region break graphs. Move them outside.
- Python-side conditionals that depend on tensor values break graphs. Replace with `torch.where` or lift to outside the compile region.
- Dict lookups on `.named_parameters()` can break depending on tracing.

- [ ] Apply fixes one at a time, retest, commit individually.

**Step 3: Measure and commit**

- [ ] Re-run profile after fixes.
- [ ] Commit with numbers per fix.

### Task 12: Dreamworld replay backward audit (informational only)

**Execute:** read-only audit, write up findings. Do not modify unless profile shows `dreamworld_replay` above 15% of inside-budget wall clock.

**Files:**
- Modify: `docs/plans/2026-04-22-exp24-runner-profile.md` (findings only)

**Step 1: Read `dreamworld_replay_backward`**

- [ ] Read `experiments/23_fast_path/dreamworld.py:197` and understand what the replay backward does.
- [ ] Document in the profile report: is it a full forward+backward, or does it reuse cached activations? Is the gradient scope correctly limited to replay-induced params?

**Step 2: Write verdict**

- [ ] Append verdict to the profile report: one of "already optimal", "activation-reuse candidate - deferred", or "concrete fix - file a task for next session".

## Phase 5 - Verify and close out

### Task 13: Final verification

**Files:**
- None (verification only)

**Step 1: Run the full affected test set**

```bash
pytest tests/test_exp23_fast_path.py tests/test_exp24_event_sleep.py tests/test_exp24_training_bundle.py tests/test_exp24_dreamworld.py tests/test_exp24_training_hooks.py -q
```

Expected: all PASS.

- [ ] Run and confirm.

**Step 2: `git diff --check`**

```bash
git diff --check
```

Expected: no output (no trailing whitespace, no conflict markers).

- [ ] Run and confirm.

**Step 3: Update the design doc with a short results section**

Append to `docs/plans/2026-04-22-exp24-rigor-and-speed-design.md`:

```markdown

## Results

Implementation landed 2026-04-22. Link: `docs/plans/2026-04-22-exp24-runner-profile.md`.

- Tests: five event_sleep tests in `tests/test_exp24_event_sleep.py`, all green.
- Kernel wins landed: (fill in from profile verdicts)
- Deferred with reason: (fill in)
```

- [ ] Append the section.
- [ ] Commit.

**Step 4: Final commit**

```bash
git add docs/plans/2026-04-22-exp24-rigor-and-speed-design.md
git commit -m "exp24: close out rigor-and-speed session with results"
```

## Notes for the implementer

- **Pod venv**: all profile runs go on the pod, which requires `source /workspace/venv/bin/activate` before any Python invocation. Project CLAUDE.md is explicit about this.
- **Committed code on pods**: per memory, commit everything before deploying to the pod. Never rsync uncommitted changes.
- **No compound bash**: Ken's global CLAUDE.md forbids `cmd1 && cmd2` and similar. Split commands.
- **Run tests after every edit**: no "trivial edit" exception.
- **If a test anchor changes** (seed-anchored trajectory, warmup-restore bit-equality), update the anchor only with a written justification in the commit message. Silent anchor updates defeat the point.
- **Profile-before-fix discipline**: Tasks 10 and 11 are conditional. Do not refactor speculatively.
- **Off-clock boundaries**: ScOpt internals and Muon orthogonalization are out of scope even if they show up in the profile.
