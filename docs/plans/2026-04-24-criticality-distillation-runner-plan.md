# Criticality Distillation Runner Wiring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task. Use @superpowers:test-driven-development for every task (red-fail → implement → green-pass → commit). Apply @superpowers:verification-before-completion before claiming any step is done.

**Goal:** Wire the already-landed `CriticalityDistillation` module (Stages 1a/1b/2/3, terminal commit `c0318d8`) into the training step of `experiments/23_fast_path/runner_fast_path.py` and emit an 8-cell first-read smoke matrix under the new `scopt_criticality_distillation_smoke` matrix name.

**Architecture:** CD pressure = `relu(CE - H[p])` computed independently of ScOpt. Bank + baseline EMA + allocator live on CPU; `seat_mask` lives on GPU. ExitStack at runner level for state capture. Two-phase ingest with async D2H. All per the design doc at `docs/plans/2026-04-24-criticality-distillation-runner-design.md` (commit `8293653`).

**Tech Stack:** PyTorch 2.9, pytest, chaoscontrol runner_fast_path, exp24 matrix builders.

**Background:** required reading —
- Mechanism design: `docs/plans/2026-04-24-criticality-distillation.md` (v3, commit `4b7d77c`)
- Runner-wiring design: `docs/plans/2026-04-24-criticality-distillation-runner-design.md` (commit `8293653`)
- Implemented module: `src/chaoscontrol/optim/criticality.py`

For entropy: the first smoke uses the **non-fused LM head path** for CD-active cells so entropy can be computed from full logits in Python (no kernel rewrite needed for first read). Fused-kernel entropy emission is a v2 throughput upgrade, deferred.

---

## Conventions

- Every task is TDD: write the failing test, run to confirm red, implement, run to confirm green, commit.
- Regression tests: a test that does not fail on the pre-fix code is NOT a regression test. See `feedback_regression_is_never_build_error.md`.
- Commits follow `component: imperative sentence` style.
- Test runner on macOS: `/opt/homebrew/bin/python3.11 -m pytest ...`
- All pod-only tests are marked `@pytest.mark.gpu` or skipped under `torch.cuda.is_available()` check; the plan target runs on CPU with contrived tensors so macOS CI stays green.

---

## Stage A — `compute_event_mask` conditional top-k

### Task A.1 — Conditional top-k (strictly-positive only) + update existing test

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py` (the module-level `compute_event_mask`)
- Modify: `tests/test_criticality_scoring.py`

**Step 1: Update the failing-when-spec-is-right test.**

The existing `test_event_mask_handles_all_equal_pressure` asserts uniform-ones input produces `k` events. Under the new spec (conditional top-k on strict positives), uniform-ones still produces `k` events because all positions have positive pressure. But a **new** test must be added for the key invariant: uniform-ZERO input produces zero events.

Append to `tests/test_criticality_scoring.py`:

```python
def test_event_mask_returns_empty_when_no_positive_pressure():
    pressure = torch.zeros(2, 10)
    mask = compute_event_mask(pressure, event_frac=0.5)
    assert mask.sum().item() == 0, (
        "uniform-zero pressure must produce no events (conditional top-k)"
    )


def test_event_mask_only_marks_strictly_positive_positions():
    pressure = torch.tensor([[-1.0, 0.0, 0.5, 0.0, -2.0, 3.0, 0.0, 0.0, 1.5, 0.1]])
    # Strictly positive positions: indices 2, 5, 8, 9 (values 0.5, 3.0, 1.5, 0.1).
    # Zero-pressure positions (indices 1, 3, 6, 7) must NOT be selected.
    # event_frac=0.5 -> k=5, but only 4 strictly-positive positions exist,
    # so we get exactly 4 events at the positive indices.
    mask = compute_event_mask(pressure, event_frac=0.5)
    assert mask.sum().item() == 4
    positive_indices = torch.tensor([2, 5, 8, 9])
    for idx in positive_indices.tolist():
        assert mask[0, idx].item() is True
    # Zero and negative positions never selected.
    for idx in [0, 1, 3, 4, 6, 7]:
        assert mask[0, idx].item() is False
```

**Step 2: Run to confirm failure.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_scoring.py::test_event_mask_returns_empty_when_no_positive_pressure tests/test_criticality_scoring.py::test_event_mask_only_marks_strictly_positive_positions -v
```
Expected: both FAIL (current impl returns k positions regardless of sign).

**Step 3: Update `compute_event_mask`.**

Replace the existing function body with:

```python
def compute_event_mask(pressure: torch.Tensor, event_frac: float) -> torch.Tensor:
    """Top-`event_frac` positions of STRICTLY-POSITIVE pressure become True.

    Events only fire on positions where the model had something to be
    surprised about (positive innovation). Uniform or all-negative
    pressure produces zero events — the bank writes nothing that step.

    Args:
        pressure: any shape; values ranked in decreasing magnitude.
        event_frac: fraction in [0, 1].

    Returns:
        Boolean tensor, same shape as `pressure`.
    """
    if not 0.0 <= event_frac <= 1.0:
        raise ValueError(f"event_frac must be in [0, 1]; got {event_frac}")
    total = pressure.numel()
    k = int(round(event_frac * total))
    if k == 0:
        return torch.zeros_like(pressure, dtype=torch.bool)
    positive = pressure > 0.0
    n_positive = int(positive.sum().item())
    if n_positive == 0:
        return torch.zeros_like(pressure, dtype=torch.bool)
    k = min(k, n_positive)
    flat = pressure.reshape(-1)
    _, idx = torch.topk(flat, k=k, largest=True)
    mask = torch.zeros(total, dtype=torch.bool, device=pressure.device)
    mask[idx] = True
    return mask.reshape(pressure.shape) & positive
```

**Step 4: Run full scoring + distillation suites.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_scoring.py tests/test_criticality_distillation.py -q
```

The existing `test_event_mask_handles_all_equal_pressure` still passes (uniform-ONES has all-positive pressure, so 10 events fire as before). The two new tests pass. Bank-related tests in `test_criticality_distillation.py` may need to be audited — any test that relied on uniform/zero pressure producing events must now tolerate zero events. Audit and update any that break.

**Step 5: Commit.**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_scoring.py
git commit -m "criticality: conditional top-k — event mask only marks strictly-positive pressure"
```

---

## Stage B — CD knobs for falsifier cells

### Task B.1 — `score_permute_before_topk` flag for shuffled-teacher cell

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test.**

Append:

```python
def test_shuffled_teacher_permutes_score_before_topk():
    """With score_permute_before_topk=True, seat assignment no longer
    tracks channel identity of the score peaks — any seat can be
    allocated. This is the shuffled-teacher falsifier."""
    torch.manual_seed(42)
    cd = CriticalityDistillation(
        num_layers=1, dim=10, trace_ttl_steps=4,
        trace_half_life_steps=100.0,
        criticality_budget_frac=0.3,
        min_weighted_events_per_layer=1.0,
        score_permute_before_topk=True,
    )
    # Evidence concentrated on channels 0, 1, 2. With permutation, top-k
    # should NOT match the evidence peaks exactly.
    evidence = torch.zeros(10)
    evidence[0] = 10.0
    evidence[1] = 9.0
    evidence[2] = 8.0
    cd.add_step_evidence(layer=0, step=0, evidence=evidence, event_count=10.0)
    cd.allocate_seats(current_step=1)
    peak_channels = {0, 1, 2}
    selected = set(cd.seat_mask[0].nonzero().flatten().tolist())
    # With score permutation, the overlap between selected seats and peak
    # channels is determined by a random permutation. Assert the selected
    # set is not exactly the peaks (flake-prone in the 10!/(7!3!)=120
    # permutations = 1/120 = 0.83% probability) — manual_seed(42) pins it.
    assert selected != peak_channels, (
        f"shuffled-teacher must not match peaks; got {selected}"
    )


def test_default_allocate_seats_does_not_permute():
    """Without the flag, allocate_seats picks the true top-k by score
    (regression pin against Task 3.1 behavior)."""
    cd = CriticalityDistillation(
        num_layers=1, dim=10, trace_ttl_steps=4,
        trace_half_life_steps=100.0,
        criticality_budget_frac=0.3,
        min_weighted_events_per_layer=1.0,
    )
    evidence = torch.zeros(10)
    evidence[0] = 10.0
    evidence[1] = 9.0
    evidence[2] = 8.0
    cd.add_step_evidence(layer=0, step=0, evidence=evidence, event_count=10.0)
    cd.allocate_seats(current_step=1)
    assert set(cd.seat_mask[0].nonzero().flatten().tolist()) == {0, 1, 2}
```

**Step 2: Run to confirm failure.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_distillation.py::test_shuffled_teacher_permutes_score_before_topk tests/test_criticality_distillation.py::test_default_allocate_seats_does_not_permute -v
```
Expected: first FAILS (constructor doesn't accept `score_permute_before_topk`), second PASSES (existing behavior).

**Step 3: Add the flag + implement permutation.**

In `CriticalityDistillation.__init__`, add after existing params:

```python
        score_permute_before_topk: bool = False,
```

Store `self.score_permute_before_topk = bool(score_permute_before_topk)`.

In `allocate_seats`, modify the block that computes top-k:

```python
            if weighted_events_per_layer[layer].item() < self.min_weighted_events_per_layer:
                self.seat_mask[layer].fill_(False)
                continue
            layer_scores = scores[layer]
            if self.score_permute_before_topk:
                perm = torch.randperm(self.dim, device=layer_scores.device)
                layer_scores = layer_scores[perm]
            topk = torch.topk(layer_scores, k=k, largest=True)
            mask = torch.zeros(self.dim, dtype=torch.bool, device=self.seat_mask.device)
            if self.score_permute_before_topk:
                # Map permuted indices back to original channel positions.
                original_indices = perm[topk.indices]
                mask[original_indices] = True
            else:
                mask[topk.indices] = True
            self.seat_mask[layer] = mask
```

**Step 4: Run.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_distillation.py -q
```
Expected: all pass.

**Step 5: Commit.**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: score_permute_before_topk flag for shuffled-teacher falsifier"
```

---

## Stage C — CPU/GPU split infrastructure

This stage restructures CD's mechanism internals so bank/baseline/allocator naturally run on CPU and `seat_mask` on GPU, without breaking existing tests. The existing API stays backward-compatible — if everything is on CPU, the existing tests still pass.

### Task C.1 — `sync_seat_mask_to_device` helper + explicit device placement

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test.**

Append:

```python
def test_seat_mask_can_live_on_different_device_than_bank():
    """In production the bank lives on CPU (cheap storage, infrequent
    access) while seat_mask lives on GPU (hot path in criticality_loss).
    Verify that moving seat_mask while keeping bank on CPU works: the
    loss reads the correct device, and allocate_seats rebuilds the GPU
    seat_mask from CPU-side score data."""
    cd = CriticalityDistillation(
        num_layers=1, dim=4, trace_ttl_steps=2,
        criticality_budget_frac=0.5,
        min_weighted_events_per_layer=0.0,
    )
    # Simulate the production placement: everything on CPU except
    # seat_mask (which should accept a device move).
    cd.bank_evidence.data = cd.bank_evidence.data.to('cpu')
    # On CPU-only environments this is a no-op but the method must not error.
    cd.sync_seat_mask_to_device(torch.device('cpu'))
    assert cd.seat_mask.device == torch.device('cpu')
    # Populate evidence, allocate, confirm seat_mask is still on the
    # requested device.
    cd.add_step_evidence(layer=0, step=0, evidence=torch.tensor([1.0, 2.0, 0.0, 0.0]), event_count=1.0)
    cd.allocate_seats(current_step=1)
    assert cd.seat_mask.device == torch.device('cpu')
    # And criticality_loss reads seat_mask from its current device.
    log_a = torch.zeros(4, requires_grad=True)
    loss = cd.criticality_loss([log_a])
    assert loss.device == torch.device('cpu')
```

**Step 2: Run to confirm failure.**

Expected: FAIL on `sync_seat_mask_to_device` AttributeError.

**Step 3: Add the helper method.**

In `CriticalityDistillation`:

```python
    def sync_seat_mask_to_device(self, device: torch.device) -> None:
        """Move (or keep) seat_mask on the specified device.

        In production, bank buffers stay on CPU while seat_mask is moved
        to the training GPU after each allocate_seats call. This helper
        makes that move explicit and testable.
        """
        self.seat_mask.data = self.seat_mask.data.to(device)
```

No other changes.

**Step 4: Run.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_distillation.py -q
```
Expected: all pass.

**Step 5: Commit.**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: sync_seat_mask_to_device for CPU-bank + GPU-seat_mask split"
```

---

### Task C.2 — `ingest_gpu` and `ingest_cpu_from_prepared` for two-phase ingest

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test.**

```python
def test_ingest_two_phase_matches_single_phase_output():
    """Two-phase ingest (ingest_gpu -> ingest_cpu_from_prepared) must
    produce the same bank state as the single-call ingest_step that
    already passes tests."""
    cd_single = CriticalityDistillation(
        num_layers=1, dim=3, trace_ttl_steps=4,
        baseline_ema_decay=0.0,
    )
    cd_split = CriticalityDistillation(
        num_layers=1, dim=3, trace_ttl_steps=4,
        baseline_ema_decay=0.0,
    )
    states_l0 = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]])
    pressure = torch.tensor([[10.0, 0.0, 0.0]])

    cd_single.ingest_step(
        step=0, pressure=pressure, states_per_layer=[states_l0],
        horizon_H=2, event_frac=0.34,
    )
    prepared = cd_split.ingest_gpu(
        pressure=pressure, states_per_layer=[states_l0],
        horizon_H=2, event_frac=0.34,
    )
    cd_split.ingest_cpu_from_prepared(step=0, prepared=prepared)

    assert torch.allclose(cd_single.bank_evidence, cd_split.bank_evidence)
    assert torch.equal(cd_single.bank_step, cd_split.bank_step)
    assert torch.allclose(cd_single.bank_event_count, cd_split.bank_event_count)
    assert torch.allclose(cd_single.baseline_future_energy, cd_split.baseline_future_energy)
    assert torch.equal(cd_single.baseline_initialized, cd_split.baseline_initialized)


def test_ingest_gpu_returns_serializable_evidence_dict():
    cd = CriticalityDistillation(num_layers=2, dim=3, trace_ttl_steps=4)
    states = [torch.randn(1, 4, 3), torch.randn(1, 4, 3)]
    pressure = torch.randn(1, 4)
    prepared = cd.ingest_gpu(
        pressure=pressure, states_per_layer=states, horizon_H=2, event_frac=0.5,
    )
    assert set(prepared.keys()) == {
        "event_mask", "aggregated_excess_per_layer",
        "non_event_mean_future_energy_per_layer",
        "event_count_per_layer",
    }
    assert prepared["event_mask"].shape == (1, 4)
    assert prepared["event_mask"].dtype == torch.bool
    assert prepared["aggregated_excess_per_layer"].shape == (2, 3)
    assert prepared["non_event_mean_future_energy_per_layer"].shape == (2, 3)
    assert prepared["event_count_per_layer"].shape == (2,)
```

**Step 2: Run to confirm failure.**

Expected: both FAIL on method-not-found.

**Step 3: Implement.**

Add methods to `CriticalityDistillation`:

```python
    @torch.no_grad()
    def ingest_gpu(
        self,
        *,
        pressure: torch.Tensor,
        states_per_layer: list,
        horizon_H: int,
        event_frac: float,
    ) -> dict:
        """Phase 1 of two-phase ingest. Runs on whatever device
        `pressure` and `states_per_layer` live on. Returns a dict of
        aggregated tensors ready for async D2H transfer."""
        if len(states_per_layer) != self.num_layers:
            raise ValueError(
                f"states_per_layer must have {self.num_layers} entries; got {len(states_per_layer)}"
            )
        event_mask = compute_event_mask(pressure, event_frac=event_frac)
        # Per-layer aggregates.
        aggregated_excess: list[torch.Tensor] = []
        non_event_mean_future_energy: list[torch.Tensor] = []
        event_count: list[torch.Tensor] = []
        flat_mask = event_mask.reshape(-1)
        flat_non_event = ~flat_mask
        for layer, states in enumerate(states_per_layer):
            if states.shape[-1] != self.dim:
                raise ValueError(
                    f"layer {layer}: states last dim {states.shape[-1]} != self.dim {self.dim}"
                )
            future_energy = compute_future_energy(states, horizon_H=horizon_H)  # [B,T,D]
            flat_fe = future_energy.reshape(-1, self.dim)  # [B*T, D]
            # Non-event-mean future energy for baseline EMA consumption.
            if flat_non_event.any():
                nonevt_mean = flat_fe[flat_non_event].mean(dim=0)
            else:
                # No non-event positions this batch — emit a sentinel zero
                # vector; ingest_cpu_from_prepared will skip the baseline
                # EMA update when event_count_per_layer indicates this.
                nonevt_mean = torch.zeros(self.dim, dtype=flat_fe.dtype, device=flat_fe.device)
            non_event_mean_future_energy.append(nonevt_mean)
            # Aggregated excess over event positions (needs current baseline).
            baseline = self.baseline_future_energy[layer].to(flat_fe.device)
            excess = (future_energy - baseline).clamp_min(0.0)
            flat_excess = excess.reshape(-1, self.dim)
            if flat_mask.any():
                agg = flat_excess[flat_mask].mean(dim=0)
                cnt = flat_mask.sum().to(torch.float32)
            else:
                agg = torch.zeros(self.dim, dtype=flat_excess.dtype, device=flat_excess.device)
                cnt = torch.zeros((), dtype=torch.float32, device=flat_excess.device)
            aggregated_excess.append(agg)
            event_count.append(cnt)
        return {
            "event_mask": event_mask,
            "aggregated_excess_per_layer": torch.stack(aggregated_excess, dim=0),
            "non_event_mean_future_energy_per_layer": torch.stack(non_event_mean_future_energy, dim=0),
            "event_count_per_layer": torch.stack(event_count, dim=0),
        }

    @torch.no_grad()
    def ingest_cpu_from_prepared(self, *, step: int, prepared: dict) -> None:
        """Phase 2 of two-phase ingest. Consumes the `prepared` dict
        (after any cross-device transfer the caller needs to do) and
        updates CPU-side bank + baseline EMA state."""
        agg = prepared["aggregated_excess_per_layer"].to(
            device=self.bank_evidence.device, dtype=self.bank_evidence.dtype
        )
        nonevt = prepared["non_event_mean_future_energy_per_layer"].to(
            device=self.baseline_future_energy.device, dtype=self.baseline_future_energy.dtype
        )
        counts = prepared["event_count_per_layer"].to(
            device=self.bank_event_count.device, dtype=self.bank_event_count.dtype
        )
        event_mask_had_non_events = bool(
            (~prepared["event_mask"]).any().item()
        )
        decay = self.baseline_ema_decay
        for layer in range(self.num_layers):
            # Baseline EMA update — uses the pre-aggregated non-event mean.
            if event_mask_had_non_events:
                obs = nonevt[layer]
                if not bool(self.baseline_initialized[layer].item()):
                    self.baseline_future_energy[layer].copy_(obs)
                    self.baseline_initialized[layer] = True
                else:
                    self.baseline_future_energy[layer].mul_(decay).add_(obs, alpha=(1.0 - decay))
            # Bank write — only if this layer had events.
            cnt = float(counts[layer].item())
            if cnt > 0:
                self.add_step_evidence(
                    layer=layer,
                    step=step,
                    evidence=agg[layer],
                    event_count=cnt,
                )
```

**Step 4: Run.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_distillation.py -q
```
Expected: all pass.

**Step 5: Commit.**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: two-phase ingest (ingest_gpu + ingest_cpu_from_prepared)"
```

---

## Stage D — Runner integration

### Task D.1 — Pressure computation helper

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Create: `tests/test_runner_criticality_pressure.py`

**Step 1: Write the failing test.**

Create `tests/test_runner_criticality_pressure.py`:

```python
"""Runner-side helpers for criticality-distillation pressure.

Uses a tiny Python-only path that mirrors the production runner's
pressure computation without requiring the full SSM forward.
"""
import torch
import pytest

# The helper will be exposed at module level from runner_fast_path.
# We import lazily via the runner module's loader.
from experiments._23_fast_path_runner_helpers import (  # noqa
    compute_ce_minus_entropy_pressure,
)


def test_pressure_fires_on_confident_wrong():
    """Confident-wrong: peaky distribution, target in the tail.
    pressure should be high (positive)."""
    # logits shape [B=1, T=1, V=3]; target = class 2 (the weak class)
    logits = torch.tensor([[[5.0, 5.0, 0.0]]])
    targets = torch.tensor([[2]])
    pressure = compute_ce_minus_entropy_pressure(logits, targets)
    assert pressure.shape == (1, 1)
    # CE = -log p(target=2). softmax: ~[0.5, 0.5, 0.0]. p(2) ≈ 0.0067.
    # CE ≈ 5.0. Entropy ≈ ln(2) ≈ 0.693. innovation ≈ 4.3.
    assert pressure[0, 0].item() > 3.0


def test_pressure_suppresses_confused_wrong():
    """Confused-wrong: uniform distribution, target doesn't matter.
    pressure should be ~0."""
    logits = torch.zeros(1, 1, 10)  # uniform
    targets = torch.tensor([[3]])
    pressure = compute_ce_minus_entropy_pressure(logits, targets)
    # CE = ln(10); H = ln(10); innovation ≈ 0.
    assert abs(pressure[0, 0].item()) < 1e-5


def test_pressure_nonnegative_always():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8)
    targets = torch.randint(0, 8, (2, 4))
    pressure = compute_ce_minus_entropy_pressure(logits, targets)
    assert (pressure >= 0.0).all(), "pressure must be relu'd to non-negative"
```

**Step 2: Run to confirm failure.**

Expected: ImportError.

**Step 3: Implement the helper.**

Decide on placement. The runner already has a `FrequencyBucketBaseline` import and helpers. Add the new helper in a sibling module so the test can import it without pulling the whole runner (and thus all torch.distributed setup).

Create `experiments/_23_fast_path_runner_helpers.py` (note the `_` prefix so it's a private cousin module that won't be collected as a test target):

```python
"""Runner-side helpers shared between runner_fast_path and tests."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_ce_minus_entropy_pressure(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Per-position `relu(CE - H[p])` pressure for Criticality Distillation.

    Args:
        logits: `[B, T, V]` unnormalized logits.
        targets: `[B, T]` int64 target indices.

    Returns:
        `[B, T]` fp32 non-negative pressure. Events are selected via
        top-k of this field.
    """
    # Per-position CE: -log p(target | context).
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
    ce = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T]
    # Per-position entropy: H[p] = -sum(p * log p).
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, T]
    return (ce - entropy).clamp_min(0.0)
```

Add a re-export to `experiments/23_fast_path/runner_fast_path.py` so the runner imports it from one place:

```python
from experiments._23_fast_path_runner_helpers import (  # noqa: E402
    compute_ce_minus_entropy_pressure,
)
```

**Step 4: Run tests.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_runner_criticality_pressure.py -v
```
Expected: 3 passed.

**Step 5: Commit.**

```bash
git add experiments/_23_fast_path_runner_helpers.py experiments/23_fast_path/runner_fast_path.py tests/test_runner_criticality_pressure.py
git commit -m "runner: compute_ce_minus_entropy_pressure helper for criticality distillation"
```

---

### Task D.2 — Add CriticalityDistillation construction + train-step plumbing in runner

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Modify: `tests/test_exp23_fast_path.py`

**Step 1: Write the failing test.**

In `tests/test_exp23_fast_path.py`, append a test that runs `train_fast_for_budget` with criticality-distillation config and verifies:
- The CD module is constructed.
- Per-step, the runner calls `ingest_gpu` then `ingest_cpu_from_prepared`.
- `criticality_loss` is added to total_loss.
- seat_mask sync happens at the configured interval.

Since this is an integration-style test, use monkeypatching to observe calls.

```python
def test_train_fast_for_budget_wires_criticality_distillation_when_configured(monkeypatch):
    mod = _load_runner_module()
    from chaoscontrol.optim.criticality import CriticalityDistillation

    model = _TinyTokenTrainModel()  # existing fixture — see prior tests in this file
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    ingest_gpu_calls = []
    ingest_cpu_calls = []
    sync_calls = []
    allocate_calls = []

    orig_ingest_gpu = CriticalityDistillation.ingest_gpu
    orig_ingest_cpu = CriticalityDistillation.ingest_cpu_from_prepared
    orig_sync = CriticalityDistillation.sync_seat_mask_to_device
    orig_allocate = CriticalityDistillation.allocate_seats

    def spy_ingest_gpu(self, **kwargs):
        ingest_gpu_calls.append(kwargs)
        return orig_ingest_gpu(self, **kwargs)

    def spy_ingest_cpu(self, **kwargs):
        ingest_cpu_calls.append(kwargs)
        return orig_ingest_cpu(self, **kwargs)

    def spy_sync(self, device):
        sync_calls.append(device)
        return orig_sync(self, device)

    def spy_allocate(self, **kwargs):
        allocate_calls.append(kwargs)
        return orig_allocate(self, **kwargs)

    monkeypatch.setattr(CriticalityDistillation, "ingest_gpu", spy_ingest_gpu)
    monkeypatch.setattr(CriticalityDistillation, "ingest_cpu_from_prepared", spy_ingest_cpu)
    monkeypatch.setattr(CriticalityDistillation, "sync_seat_mask_to_device", spy_sync)
    monkeypatch.setattr(CriticalityDistillation, "allocate_seats", spy_allocate)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=3,
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
        precision="fp32",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=4,
        prefetch_batches=False,
        # CD config:
        criticality_distill_enabled=True,
        criticality_distill_weight=1e-3,
        criticality_distill_budget_frac=0.25,
        criticality_distill_half_life_steps=100.0,
        criticality_distill_ttl_steps=400,
        criticality_distill_horizon_H=2,
        criticality_distill_event_frac=0.25,
        criticality_distill_seat_refresh_interval=2,
        criticality_distill_min_weighted_events_per_layer=0.0,
        criticality_distill_uniform_pressure=False,
        criticality_distill_score_permute_before_topk=False,
    )

    # Ingest must fire on every step.
    assert len(ingest_gpu_calls) == 4
    assert len(ingest_cpu_calls) == 4
    # Seat-refresh at interval=2 over 4 steps => 2 allocate calls.
    assert len(allocate_calls) == 2
    # Sync must follow allocate.
    assert len(sync_calls) >= 2
```

**Step 2: Run to confirm failure.**

Expected: FAIL because `criticality_distill_*` kwargs don't exist yet, and no CD plumbing in `train_fast_for_budget`.

**Step 3: Implement the plumbing in `train_fast_for_budget`.**

In `experiments/23_fast_path/runner_fast_path.py`:

- Add the `criticality_distill_*` kwargs to `train_fast_for_budget` signature (all with sensible defaults matching an `enabled=False` no-op path).
- At the top of the function (after the existing config normalization block), construct the CD module if enabled:

```python
    criticality = None
    if criticality_distill_enabled:
        from contextlib import ExitStack  # noqa: E402
        from chaoscontrol.optim.criticality import CriticalityDistillation
        # Count diag-mode SSM cores in the model — these are the layers we
        # capture. Layers with a_mode != "diag" skip CD for this first smoke.
        ssm_cores = []
        for layer in model.layers:
            core = getattr(layer, "core", None)
            if core is not None and getattr(core, "a_mode", None) == "diag":
                ssm_cores.append(core)
        if not ssm_cores:
            raise ValueError(
                "criticality_distill_enabled=True requires at least one diag-mode SSM core"
            )
        dim = ssm_cores[0].dim
        for c in ssm_cores:
            if c.dim != dim:
                raise ValueError(
                    f"CD currently requires all captured cores share dim; got {c.dim} vs {dim}"
                )
        criticality = CriticalityDistillation(
            num_layers=len(ssm_cores),
            dim=dim,
            trace_ttl_steps=int(criticality_distill_ttl_steps),
            trace_half_life_steps=float(criticality_distill_half_life_steps),
            seat_refresh_interval=int(criticality_distill_seat_refresh_interval),
            criticality_budget_frac=float(criticality_distill_budget_frac),
            critical_value=float(criticality_distill_critical_value),
            min_weighted_events_per_layer=float(criticality_distill_min_weighted_events_per_layer),
            criticality_distill_weight=float(criticality_distill_weight),
            score_permute_before_topk=bool(criticality_distill_score_permute_before_topk),
        )
        # Buffers built on CPU via the module's default — bank stays CPU,
        # seat_mask moved to the training device.
        criticality.sync_seat_mask_to_device(device)
```

In the inner training loop, around the forward pass, wrap capture in an ExitStack and route pressure:

```python
        if criticality is not None:
            with ExitStack() as stack:
                getters = [stack.enter_context(c.capture_states()) for c in ssm_cores]
                # Existing forward + CE computation...
                # logits and CE are produced by the forward.
            states_per_layer = [g() for g in getters]
            if bool(criticality_distill_uniform_pressure):
                pressure = torch.ones_like(ce)
            else:
                pressure = compute_ce_minus_entropy_pressure(logits, targets)
            prepared = criticality.ingest_gpu(
                pressure=pressure,
                states_per_layer=states_per_layer,
                horizon_H=int(criticality_distill_horizon_H),
                event_frac=float(criticality_distill_event_frac),
            )
            # For the first smoke: synchronous D2H. Async streams are a
            # v2 throughput refinement.
            prepared_cpu = {
                k: (v.to('cpu', non_blocking=False) if isinstance(v, torch.Tensor) else v)
                for k, v in prepared.items()
            }
            criticality.ingest_cpu_from_prepared(step=step_count, prepared=prepared_cpu)
            if step_count % criticality.seat_refresh_interval == 0:
                criticality.allocate_seats(current_step=step_count)
                criticality.sync_seat_mask_to_device(device)
            log_a_per_layer = [c.log_a for c in ssm_cores]
            cd_loss = criticality.criticality_loss(log_a_per_layer)
            total_loss = ce_loss + cd_loss
        else:
            total_loss = ce_loss
```

Write per the existing runner's structure — this pseudocode is illustrative; adapt to actual variable names (`ce`, `logits`, `targets`, `step_count` may have different names in the runner).

Surface CD telemetry into the returned result dict:

```python
        if criticality is not None:
            result["criticality_distillation"] = {
                "seat_mask_fraction_per_layer": [
                    float(m.float().mean().item()) for m in criticality.seat_mask
                ],
                "bank_occupancy_per_layer": [
                    float((step_tensor >= 0).float().mean().item())
                    for step_tensor in criticality.bank_step
                ],
                "baseline_initialized_per_layer": criticality.baseline_initialized.tolist(),
                "criticality_distill_weight": criticality.criticality_distill_weight,
                "trace_half_life_steps": criticality.trace_half_life_steps,
                "trace_ttl_steps": criticality.trace_ttl_steps,
                "horizon_H": int(criticality_distill_horizon_H),
                "event_frac": float(criticality_distill_event_frac),
                "uniform_pressure": bool(criticality_distill_uniform_pressure),
                "score_permute_before_topk": bool(criticality_distill_score_permute_before_topk),
            }
```

**Step 4: Run.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_exp23_fast_path.py::test_train_fast_for_budget_wires_criticality_distillation_when_configured -v
```
Expected: pass.

Also run the full file to catch regressions:

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_exp23_fast_path.py -q
```
Expected: all existing tests still pass.

**Step 5: Commit.**

```bash
git add experiments/23_fast_path/runner_fast_path.py tests/test_exp23_fast_path.py
git commit -m "runner: criticality distillation wiring (capture, ingest, seat refresh, loss compose)"
```

---

### Task D.3 — Rare-bucket CE readout from FrequencyBucketBaseline

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Modify: `tests/test_exp23_fast_path.py`

**Step 1: Write the failing test.**

```python
def test_train_fast_for_budget_returns_rare_bucket_ce_trajectory():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    token_frequencies = torch.tensor([100.0, 50.0, 20.0, 10.0, 5.0, 1.0])

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=3,
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
        precision="fp32",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=4,
        prefetch_batches=False,
        rare_bucket_ce_enabled=True,
        rare_bucket_ce_token_frequencies=token_frequencies,
        rare_bucket_ce_num_buckets=4,
    )
    # Every training step emits one bucket-CE snapshot.
    traj = result.get("rare_bucket_ce_trajectory")
    assert traj is not None
    assert len(traj) == 4
    # Each snapshot has all buckets.
    for snap in traj:
        assert len(snap) == 4  # num_buckets
        for val in snap:
            assert isinstance(val, float)
```

**Step 2: Run to confirm failure.**

Expected: FAIL on missing kwargs / None result key.

**Step 3: Implement.**

Add `rare_bucket_ce_enabled`, `rare_bucket_ce_token_frequencies`, `rare_bucket_ce_num_buckets` kwargs to `train_fast_for_budget`. When enabled, construct a `FrequencyBucketBaseline` alongside CD (independent of ScOpt — we want rare-bucket CE tracking even when ScOpt is off for the budget-only cell).

At step end, read `baseline._ema` and append to a list. Return under `rare_bucket_ce_trajectory` in the result dict.

**Step 4: Run.**

Pass all runner tests.

**Step 5: Commit.**

```bash
git add experiments/23_fast_path/runner_fast_path.py tests/test_exp23_fast_path.py
git commit -m "runner: rare_bucket_ce_trajectory readout from FrequencyBucketBaseline"
```

---

## Stage E — Smoke matrix

### Task E.1 — `build_criticality_distillation_first_smoke_matrix`

**Files:**
- Modify: `experiments/24_training_time_bundle/exp24.py`
- Modify: `experiments/24_training_time_bundle/run_exp24.py`

**Step 1: Write the failing test.**

Create `tests/test_exp24_cd_smoke_matrix.py`:

```python
import sys
sys.path.insert(0, "experiments/23_fast_path")
sys.path.insert(0, "experiments/24_training_time_bundle")
from exp24 import build_criticality_distillation_first_smoke_matrix


def test_matrix_emits_eight_cells_with_expected_names():
    entries = build_criticality_distillation_first_smoke_matrix(
        speed_config={}, seed=1337, budget_seconds=600.0,
    )
    names = [e["name"] for e in entries]
    expected_suffixes = [
        "treatment", "telemetry", "shuffled", "budget_only",
        "hl_short", "hl_long", "H_short", "H_long",
    ]
    for suffix in expected_suffixes:
        assert any(suffix in n for n in names), (
            f"missing cell for {suffix}; got names {names}"
        )


def test_matrix_cell_flags_match_design():
    entries = build_criticality_distillation_first_smoke_matrix(
        speed_config={}, seed=1337, budget_seconds=600.0,
    )
    by_suffix = {e["name"].rsplit("_s", 1)[0].split("cd_")[-1]: e for e in entries}
    # treatment: surprise pressure, weight 1e-3, real score, hl 2048, H 64
    t = by_suffix["treatment"]
    assert t["criticality_distill_weight"] == 1e-3
    assert t["criticality_distill_uniform_pressure"] is False
    assert t["criticality_distill_score_permute_before_topk"] is False
    assert t["criticality_distill_half_life_steps"] == 2048
    assert t["criticality_distill_horizon_H"] == 64
    # telemetry: weight 0
    assert by_suffix["telemetry"]["criticality_distill_weight"] == 0.0
    # shuffled: permute True
    assert by_suffix["shuffled"]["criticality_distill_score_permute_before_topk"] is True
    # budget_only: uniform pressure True
    assert by_suffix["budget_only"]["criticality_distill_uniform_pressure"] is True
    # hl_short/hl_long
    assert by_suffix["hl_short"]["criticality_distill_half_life_steps"] == 256
    assert by_suffix["hl_long"]["criticality_distill_half_life_steps"] == 16384
    # H_short/H_long
    assert by_suffix["H_short"]["criticality_distill_horizon_H"] == 16
    assert by_suffix["H_long"]["criticality_distill_horizon_H"] == 256
```

**Step 2: Run to confirm failure.**

Expected: ImportError on the builder name.

**Step 3: Implement.**

In `experiments/24_training_time_bundle/exp24.py`, add builder following the pattern of `build_scopt_calibration_sweep_matrix`:

```python
def build_criticality_distillation_first_smoke_matrix(
    *,
    speed_config: dict[str, Any],
    seed: int = 1337,
    world_size: int = 1,
    budget_seconds: float = 600.0,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """First-read Criticality Distillation matrix — 8 cells.

    Measures whether CD's treatment-cell improves rare-bucket CE
    relative to three falsifier controls (telemetry, shuffled-teacher,
    budget-only) and maps sensitivity to half-life and horizon_H.
    """
    base = _base_entry(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
    )
    base["artifact_impact"] = ARTIFACT_CHANGES_WEIGHTS_ONLY
    base["batch_size"] = int(batch_size)
    base["optimizer_param_grouping"] = "ssm_three_group"
    base["optimizer_dynamics_lr_mul"] = 0.1
    base["optimizer"] = "muon"  # ScOpt OFF — CD is independent

    # CD defaults per design doc.
    base["criticality_distill_enabled"] = True
    base["criticality_distill_weight"] = 1e-3
    base["criticality_distill_budget_frac"] = 0.15
    base["criticality_distill_critical_value"] = 0.95
    base["criticality_distill_half_life_steps"] = 2048
    base["criticality_distill_ttl_steps"] = 20480
    base["criticality_distill_horizon_H"] = 64
    base["criticality_distill_event_frac"] = 0.05
    base["criticality_distill_seat_refresh_interval"] = 64
    base["criticality_distill_min_weighted_events_per_layer"] = 256
    base["criticality_distill_uniform_pressure"] = False
    base["criticality_distill_score_permute_before_topk"] = False

    base["rare_bucket_ce_enabled"] = True
    base["rare_bucket_ce_num_buckets"] = 16

    def _cell(arm_suffix: str, overrides: dict[str, Any]) -> dict[str, Any]:
        cell = copy.deepcopy(base)
        cell.update(overrides)
        return _named_entry(
            base=cell,
            phase="smoke",
            mechanism="cd_first_smoke",
            arm=f"cd_{arm_suffix}",
            seed=seed,
        )

    return [
        _cell("treatment", {}),
        _cell("telemetry", {"criticality_distill_weight": 0.0}),
        _cell("shuffled", {"criticality_distill_score_permute_before_topk": True}),
        _cell("budget_only", {"criticality_distill_uniform_pressure": True}),
        _cell("hl_short", {"criticality_distill_half_life_steps": 256,
                            "criticality_distill_ttl_steps": 2560}),
        _cell("hl_long", {"criticality_distill_half_life_steps": 16384,
                           "criticality_distill_ttl_steps": 163840}),
        _cell("H_short", {"criticality_distill_horizon_H": 16}),
        _cell("H_long", {"criticality_distill_horizon_H": 256}),
    ]
```

In `run_exp24.py`:
- Add the import at the top-of-file `from exp24 import` block.
- Add the dispatch branch inside `run_matrix`.
- Add `"cd_first_smoke"` to the argparse choices list.
- Add `cd_first_smoke` to `_default_world_size_for_matrix` (returns 1) and `_default_budget_for_matrix` (returns 600.0).

**Step 4: Run tests.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_exp24_cd_smoke_matrix.py -q
```
Expected: 2 passed.

Also verify the runner integration:

```
/opt/homebrew/bin/python3.11 experiments/24_training_time_bundle/run_exp24.py --matrix cd_first_smoke --seeds 1337 --dry-run
```
Expected: prints 8 matrix entries without erroring.

**Step 5: Commit.**

```bash
git add experiments/24_training_time_bundle/exp24.py experiments/24_training_time_bundle/run_exp24.py tests/test_exp24_cd_smoke_matrix.py
git commit -m "exp24: register cd_first_smoke matrix (8 cells: treatment + 3 falsifiers + 4 ablations)"
```

---

## Final verification

Before handing back control:

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_distillation.py tests/test_criticality_scoring.py tests/test_ssm_state_capture.py tests/test_runner_criticality_pressure.py tests/test_exp24_cd_smoke_matrix.py tests/test_exp23_fast_path.py -q
```

Expected: all passing, count increased over the current 34 by the 15-ish new tests added.

```
git log --oneline 8293653..HEAD
```

Expected: 8-10 commits, each a task above.

```
git status
```

Expected: clean.

```
/opt/homebrew/bin/python3.11 experiments/24_training_time_bundle/run_exp24.py --matrix cd_first_smoke --seeds 1337 --dry-run
```

Expected: 8 entries printed, no errors.

---

## Out of scope for this plan (follow-ups)

- **Fused-kernel entropy emission.** The first smoke uses the non-fused LM head path for CD cells so we can compute entropy in Python. Extending the fused kernel to emit per-token entropy alongside per-token CE is a throughput win for scale-up runs but not required for a mechanism read. Follow-up plan.
- **Async side-stream D2H.** Task D.2 uses synchronous `non_blocking=False`. Upgrading to a dedicated CUDA stream + event ordering is a throughput refinement; the first smoke does not need it.
- **Multi-seed + 4×H100 confirmation** after first smoke shows treatment wins.
- **Precision-weighted surprise** `(CE - H[p]) · H[p]` — active-inference panel's suggested v2.
- **Matched-nearby baseline control** as a scoring ablation.
- **Per-frequency-bucket evidence banks** (bucket-keyed `bank_evidence`).
- **Paired / full SSM modes.** Currently `capture_states()` raises `NotImplementedError` for non-diag; CD runner wiring inherits that restriction.
