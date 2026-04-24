# Criticality Distillation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task. Use @superpowers:test-driven-development for every task (red-fail, implement, green-pass, commit). Apply @superpowers:verification-before-completion before claiming any step is done.

**Goal:** Implement the Criticality Distillation mechanism (Stages 1a, 1b, 2, 3 of the design doc) — a new `CriticalityDistillation(nn.Module)` that captures SSM recurrence states, scores channels by post-event trace survival, allocates a budgeted set of near-critical seats, and emits a seat-masked loss on `log_a`. Target: a working module with green tests that an integration stage can wire into the runner.

**Architecture:** A separate `nn.Module` (not part of ScOpt) owns the Rare Trace Bank as registered buffers, consumes pressure from ScOpt's existing hooks, consumes recurrence states via a new `ChaosSSMCore.capture_states()` context manager, and emits a seat-masked MSE loss. Diag-backend only in stage 1. Per-step aggregate bank (one `[D]` per layer per step) replaces the per-event bank that would overflow at submission batch sizes.

**Tech Stack:** PyTorch 2.9, pytest, chaoscontrol (local repo). No new external deps.

**Background:** read `docs/plans/2026-04-24-criticality-distillation.md` (v3, commit `4b7d77c`) for the full mechanism rationale and falsifier controls. This plan implements that design and does not reopen any of its decisions.

---

## Conventions for every task

- Every task starts with a failing test, per @superpowers:test-driven-development.
- After every green commit, run the **full relevant test file** once (not just the single test) to ensure no regression in neighbors: `pytest tests/test_criticality_distillation.py -q`.
- All tensors default to `torch.float32` unless otherwise specified. Buffers that track ages may be `int64` (wall-clock step counters).
- Commits are small (one task per commit). Commit messages follow existing repo style: `component: imperative sentence`.
- Do not claim a task is done until the test actually fails on the pre-fix code AND passes on the post-fix code (per `feedback_regression_is_never_build_error.md`).
- Do not add features beyond what the task specifies. YAGNI applies.

---

## Stage 1a — Recurrence-state capture API

**Files created:** none in this stage.
**Files modified:** `src/chaoscontrol/core.py`, new test file `tests/test_ssm_state_capture.py`.

### Task 1a.1 — Capture state wiring in `_forward_diag_scan`

**Files:**
- Modify: `src/chaoscontrol/core.py` (around `ChaosSSMCore.__init__` and `_forward_diag_scan`)
- Create: `tests/test_ssm_state_capture.py`

**Step 1: Write the failing test**

Create `tests/test_ssm_state_capture.py`:

```python
import torch
import pytest
from chaoscontrol.core import ChaosSSMCore


def _make_core(dim: int = 8) -> ChaosSSMCore:
    torch.manual_seed(0)
    return ChaosSSMCore(dim=dim, a_mode="diag")


def test_capture_states_context_manager_records_shape_via_helper():
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8)
    with core.capture_states() as get_states:
        _ = core._forward_diag_scan(x)
        captured = get_states()
    assert captured is not None, "capture_states must populate states"
    assert captured.shape == (2, 5, 8)
    assert captured.requires_grad is False, "captured states must be detached"


def test_capture_states_context_clears_after_exit():
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8)
    with core.capture_states() as get_states:
        _ = core._forward_diag_scan(x)
    assert core._captured_states is None
    assert core._capture_states_enabled is False
```

**Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ssm_state_capture.py -q`
Expected: two tests fail with `AttributeError: 'ChaosSSMCore' object has no attribute 'capture_states'` (or similar).

**Step 3: Minimal implementation**

In `src/chaoscontrol/core.py`, at the top of the file if not already present:

```python
from contextlib import contextmanager
```

In `ChaosSSMCore.__init__`, after the existing assignments, add:

```python
        self._capture_states_enabled = False
        self._captured_states: torch.Tensor | None = None
```

Below `__init__`, add:

```python
    @contextmanager
    def capture_states(self):
        """Enable recurrence-state capture for the enclosed forward(s).

        Inside the context, `_forward_diag_scan` and the diag fast-path branch
        in `forward()` write the per-step state trajectory into
        `self._captured_states` as a detached `[B, T, D]` tensor. The buffer
        is cleared on exit so we do not leak `[B, T, D]` tensors across
        unrelated forwards.
        """
        if self.a_mode != "diag":
            raise NotImplementedError(
                f"capture_states() is only implemented for a_mode='diag'; "
                f"got a_mode={self.a_mode!r}."
            )
        self._capture_states_enabled = True
        self._captured_states = None
        try:
            yield lambda: self._captured_states
        finally:
            self._capture_states_enabled = False
            self._captured_states = None
```

Modify `_forward_diag_scan` to write the buffer when enabled:

```python
    def _forward_diag_scan(self, x: torch.Tensor) -> torch.Tensor:
        decay, update, gate = self._diag_terms(x)
        states = _diag_recurrence(decay, update)
        if self._capture_states_enabled:
            self._captured_states = states.detach()
        out = gate * states
        return self.out_proj(out)
```

**Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ssm_state_capture.py -q`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/core.py tests/test_ssm_state_capture.py
git commit -m "ssm: add capture_states() context manager for recurrence-state capture"
```

### Task 1a.2 — Capture in the inlined fast-path of `forward()` (the production path)

**Files:**
- Modify: `src/chaoscontrol/core.py` at the diag fast-path branch inside `forward()` (currently around line 573-583)
- Modify: `tests/test_ssm_state_capture.py`

**Step 1: Write the failing test**

Append to `tests/test_ssm_state_capture.py`:

```python
def test_capture_via_top_level_forward_diag_fast_path():
    """Production model.encode() path routes through forward()'s inlined
    diag fast-path, not _forward_diag_scan. Must capture there too."""
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8)
    with core.capture_states() as get_states:
        _ = core(x)  # forward, not _forward_diag_scan
        captured = get_states()
    assert captured is not None, "forward() diag fast-path must capture too"
    assert captured.shape == (2, 5, 8)


def test_capture_is_disabled_by_default_no_overhead_path():
    """Capture is off by default; no attribute should be populated without
    the context manager."""
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8)
    _ = core(x)
    assert core._captured_states is None
    assert core._capture_states_enabled is False
```

**Step 2: Run tests to verify the first one fails, second passes**

Run: `pytest tests/test_ssm_state_capture.py::test_capture_via_top_level_forward_diag_fast_path -q`
Expected: FAIL — captured is None because forward() doesn't populate it.

Run: `pytest tests/test_ssm_state_capture.py::test_capture_is_disabled_by_default_no_overhead_path -q`
Expected: PASS (already disabled by default after 1a.1).

**Step 3: Minimal implementation**

Locate the inlined diag fast-path in `ChaosSSMCore.forward()` (search for `if rich_b is None and initial_state is None:` inside the `if self.a_mode == "diag":` branch). Currently:

```python
            if rich_b is None and initial_state is None:
                decay, update, gate = self._diag_terms(x)
                states = _diag_recurrence(decay, update)
                out = gate * states
                y = self.out_proj(out)
```

Add capture between the `_diag_recurrence` call and the `out = gate * states` line:

```python
            if rich_b is None and initial_state is None:
                decay, update, gate = self._diag_terms(x)
                states = _diag_recurrence(decay, update)
                if self._capture_states_enabled:
                    self._captured_states = states.detach()
                out = gate * states
                y = self.out_proj(out)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ssm_state_capture.py -q`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/core.py tests/test_ssm_state_capture.py
git commit -m "ssm: capture recurrence states in forward() diag fast-path branch"
```

### Task 1a.3 — Numerical invariance under capture (output must not change)

**Files:**
- Modify: `tests/test_ssm_state_capture.py`

No implementation change expected; this locks in an invariant so future refactors of the diag path can't silently corrupt outputs.

**Step 1: Write the failing test**

Append to `tests/test_ssm_state_capture.py`:

```python
def test_forward_output_is_identical_with_and_without_capture():
    core = _make_core(dim=8)
    x = torch.randn(2, 5, 8)
    y_no_capture = core(x)
    with core.capture_states() as _:
        y_with_capture = core(x)
    assert torch.equal(y_no_capture, y_with_capture), (
        "capture must not change the forward output"
    )
```

**Step 2: Run the test**

Run: `pytest tests/test_ssm_state_capture.py::test_forward_output_is_identical_with_and_without_capture -q`
Expected: PASS (if capture is implemented correctly — it only writes a detached copy, never modifies the forward graph).

If this test fails, the capture logic is perturbing the computation — investigate before proceeding.

**Step 3: Commit**

```bash
git add tests/test_ssm_state_capture.py
git commit -m "ssm: pin capture cannot change forward output"
```

### Task 1a.4 — Non-diag modes raise `NotImplementedError` on capture request

**Files:**
- Modify: `tests/test_ssm_state_capture.py`

**Step 1: Write the failing test**

Append:

```python
def test_capture_states_raises_for_paired_mode():
    torch.manual_seed(0)
    core = ChaosSSMCore(dim=8, a_mode="paired")
    with pytest.raises(NotImplementedError, match="capture_states"):
        with core.capture_states():
            pass


def test_capture_states_raises_for_full_mode():
    torch.manual_seed(0)
    core = ChaosSSMCore(dim=8, a_mode="full")
    with pytest.raises(NotImplementedError, match="capture_states"):
        with core.capture_states():
            pass
```

**Step 2: Run tests**

Run: `pytest tests/test_ssm_state_capture.py -q`
Expected: 6 passed (the new two pass because 1a.1 already added the `a_mode != "diag"` guard).

**Step 3: Commit**

```bash
git add tests/test_ssm_state_capture.py
git commit -m "ssm: pin capture_states NotImplementedError for paired/full modes"
```

---

## Stage 1b — Rare Trace Bank scaffold

**Files created:** `src/chaoscontrol/optim/criticality.py`, `tests/test_criticality_distillation.py`.

### Task 1b.1 — `CriticalityDistillation` skeleton with registered buffers

**Files:**
- Create: `src/chaoscontrol/optim/criticality.py`
- Create: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test**

Create `tests/test_criticality_distillation.py`:

```python
import torch
import pytest
from chaoscontrol.optim.criticality import CriticalityDistillation


def test_constructs_with_expected_buffer_shapes():
    cd = CriticalityDistillation(
        num_layers=3,
        dim=16,
        trace_ttl_steps=8,
        trace_half_life_steps=4,
        seat_refresh_interval=2,
        criticality_budget_frac=0.25,
        critical_value=0.95,
        min_weighted_events_per_layer=1.0,
        criticality_distill_weight=1e-3,
        baseline_ema_decay=0.99,
    )
    # Evidence bank: [num_layers, trace_ttl_steps, dim]
    assert cd.bank_evidence.shape == (3, 8, 16)
    # Per-slot step counter: -1 means "empty"
    assert cd.bank_step.shape == (3, 8)
    assert cd.bank_event_count.shape == (3, 8)
    # Baseline EMA per layer per channel
    assert cd.baseline_future_energy.shape == (3, 16)
    # Current seat assignment per layer (bool)
    assert cd.seat_mask.shape == (3, 16)
    assert cd.seat_mask.dtype == torch.bool
    # All buffers start zero / empty
    assert torch.equal(cd.bank_evidence, torch.zeros_like(cd.bank_evidence))
    assert torch.equal(cd.bank_step, torch.full_like(cd.bank_step, -1))
    assert torch.equal(cd.bank_event_count, torch.zeros_like(cd.bank_event_count))
    assert not cd.seat_mask.any()


def test_buffers_register_for_state_dict():
    cd = CriticalityDistillation(num_layers=2, dim=4, trace_ttl_steps=3)
    sd = cd.state_dict()
    assert "bank_evidence" in sd
    assert "bank_step" in sd
    assert "bank_event_count" in sd
    assert "baseline_future_energy" in sd
    assert "seat_mask" in sd
```

**Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: 2 tests fail — `ModuleNotFoundError` for the import.

**Step 3: Minimal implementation**

Create `src/chaoscontrol/optim/criticality.py`:

```python
"""Criticality Distillation mechanism (design: docs/plans/2026-04-24-criticality-distillation.md).

ScOpt produces per-token pressure; CriticalityDistillation consumes pressure
and recurrence-state traces, scores channels by post-event trace survival,
allocates a budgeted set of near-critical seats, and emits a seat-masked MSE
loss on `log_a`. Diag-backend only in stage 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CriticalityDistillation(nn.Module):
    """Criticality distillation loss generator + trace bank.

    All mechanism state (bank, baseline EMA, current seats) is stored as
    registered buffers so checkpointing is automatic via `state_dict`.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        dim: int,
        trace_ttl_steps: int = 1024,
        trace_half_life_steps: float = 256.0,
        seat_refresh_interval: int = 64,
        criticality_budget_frac: float = 0.15,
        critical_value: float = 0.95,
        min_weighted_events_per_layer: float = 256.0,
        criticality_distill_weight: float = 1e-3,
        baseline_ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        if not 0.0 < criticality_budget_frac < 1.0:
            raise ValueError(
                f"criticality_budget_frac must be in (0, 1); got {criticality_budget_frac}"
            )
        if critical_value <= 0.0 or critical_value >= 1.0:
            raise ValueError(
                f"critical_value must be in (0, 1); got {critical_value}"
            )
        self.num_layers = int(num_layers)
        self.dim = int(dim)
        self.trace_ttl_steps = int(trace_ttl_steps)
        self.trace_half_life_steps = float(trace_half_life_steps)
        self.seat_refresh_interval = int(seat_refresh_interval)
        self.criticality_budget_frac = float(criticality_budget_frac)
        self.critical_value = float(critical_value)
        self.min_weighted_events_per_layer = float(min_weighted_events_per_layer)
        self.criticality_distill_weight = float(criticality_distill_weight)
        self.baseline_ema_decay = float(baseline_ema_decay)

        # Per-layer ring buffer keyed by step index (one evidence vector per
        # (layer, step) that had at least one event).
        self.register_buffer(
            "bank_evidence",
            torch.zeros(self.num_layers, self.trace_ttl_steps, self.dim, dtype=torch.float32),
        )
        # Slot's originating step; -1 means "empty".
        self.register_buffer(
            "bank_step",
            torch.full((self.num_layers, self.trace_ttl_steps), -1, dtype=torch.int64),
        )
        # Number of events contributing to this slot's evidence.
        self.register_buffer(
            "bank_event_count",
            torch.zeros(self.num_layers, self.trace_ttl_steps, dtype=torch.float32),
        )
        # Per-layer per-channel EMA of non-event future energy.
        self.register_buffer(
            "baseline_future_energy",
            torch.zeros(self.num_layers, self.dim, dtype=torch.float32),
        )
        # Current seat assignment per layer (top-k channels that feel the loss).
        self.register_buffer(
            "seat_mask",
            torch.zeros(self.num_layers, self.dim, dtype=torch.bool),
        )
```

Also add this to the package's `__init__.py` if `src/chaoscontrol/optim/__init__.py` exists and exports named members:

```python
# (only if the file has explicit re-exports)
from chaoscontrol.optim.criticality import CriticalityDistillation  # noqa: F401
```

Check with: `grep -l "ScarcityAwareOptimizer" src/chaoscontrol/optim/__init__.py` — if the file re-exports ScOpt explicitly, add the CD re-export too; otherwise skip.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: scaffold CriticalityDistillation with registered buffers"
```

### Task 1b.2 — Bank ingest with TTL-based ring slot assignment

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test**

Append:

```python
def test_add_step_evidence_writes_into_correct_slot_and_ttl_wraps():
    cd = CriticalityDistillation(num_layers=1, dim=4, trace_ttl_steps=3)
    e0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    e1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    e2 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    e3 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    cd.add_step_evidence(layer=0, step=0, evidence=e0[0], event_count=10.0)
    cd.add_step_evidence(layer=0, step=1, evidence=e1[0], event_count=11.0)
    cd.add_step_evidence(layer=0, step=2, evidence=e2[0], event_count=12.0)
    # At this point all 3 slots occupied for layer 0.
    assert set(cd.bank_step[0].tolist()) == {0, 1, 2}

    # Wrap: step=3 must evict the oldest (step=0).
    cd.add_step_evidence(layer=0, step=3, evidence=e3[0], event_count=13.0)
    assert set(cd.bank_step[0].tolist()) == {1, 2, 3}
    # Evidence for step=3 present
    slot = (cd.bank_step[0] == 3).nonzero(as_tuple=True)[0].item()
    assert torch.equal(cd.bank_evidence[0, slot], e3[0])
    assert cd.bank_event_count[0, slot].item() == pytest.approx(13.0)


def test_add_step_evidence_rejects_wrong_layer_or_shape():
    cd = CriticalityDistillation(num_layers=2, dim=4, trace_ttl_steps=3)
    with pytest.raises((IndexError, ValueError)):
        cd.add_step_evidence(layer=5, step=0, evidence=torch.zeros(4), event_count=1.0)
    with pytest.raises((RuntimeError, ValueError)):
        cd.add_step_evidence(layer=0, step=0, evidence=torch.zeros(8), event_count=1.0)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_criticality_distillation.py::test_add_step_evidence_writes_into_correct_slot_and_ttl_wraps -q`
Expected: FAIL — `AttributeError: 'CriticalityDistillation' object has no attribute 'add_step_evidence'`.

**Step 3: Minimal implementation**

In `src/chaoscontrol/optim/criticality.py`, inside `CriticalityDistillation`:

```python
    def add_step_evidence(
        self,
        *,
        layer: int,
        step: int,
        evidence: torch.Tensor,
        event_count: float,
    ) -> None:
        """Write one (layer, step) evidence vector into the bank.

        Slot selection rule:
          * If there is any empty slot (bank_step == -1), fill the smallest
            empty index.
          * Else evict the slot with the oldest bank_step value (lowest step).
        This gives us a TTL-wrapped ring without tracking a separate write
        pointer — the aging math naturally demotes the oldest evidence.
        """
        if not 0 <= layer < self.num_layers:
            raise IndexError(
                f"layer={layer} out of range for num_layers={self.num_layers}"
            )
        if evidence.shape != (self.dim,):
            raise ValueError(
                f"evidence must have shape ({self.dim},); got {tuple(evidence.shape)}"
            )

        slots = self.bank_step[layer]  # [trace_ttl_steps]
        empty = (slots == -1).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            slot = int(empty[0].item())
        else:
            # Evict oldest
            slot = int(slots.argmin().item())

        self.bank_evidence[layer, slot] = evidence.to(
            dtype=self.bank_evidence.dtype, device=self.bank_evidence.device
        )
        self.bank_step[layer, slot] = int(step)
        self.bank_event_count[layer, slot] = float(event_count)
```

**Step 4: Run tests**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: add_step_evidence writes into oldest-slot / empty-slot with validation"
```

### Task 1b.3 — Age-weighted `score` with known math

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test**

Append:

```python
def test_score_age_weights_match_hand_computation():
    cd = CriticalityDistillation(
        num_layers=1,
        dim=4,
        trace_ttl_steps=3,
        trace_half_life_steps=2.0,  # half-life = 2 steps for hand math
    )
    # One evidence vector at step=0, another at step=2.
    cd.add_step_evidence(
        layer=0, step=0,
        evidence=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        event_count=1.0,
    )
    cd.add_step_evidence(
        layer=0, step=2,
        evidence=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        event_count=1.0,
    )
    # Score at current_step=4.
    # age_weight uses half-life formulation: w = 2 ** (-age / half_life)
    # age_0 = 4 (entry at step 0) -> w_0 = 2**(-4/2) = 0.25
    # age_1 = 2 (entry at step 2) -> w_1 = 2**(-2/2) = 0.5
    # expected: (0.25 * [1,0,0,0] + 0.5 * [0,1,0,0]) / (0.25 + 0.5)
    #         = [1/3, 2/3, 0, 0]
    score = cd.score(current_step=4)
    expected = torch.tensor([[1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0]])
    assert torch.allclose(score, expected, atol=1e-6), f"{score} != {expected}"


def test_score_returns_zeros_when_bank_empty():
    cd = CriticalityDistillation(num_layers=2, dim=3, trace_ttl_steps=4)
    score = cd.score(current_step=0)
    assert score.shape == (2, 3)
    assert torch.equal(score, torch.zeros_like(score))
```

Important: the test pins the age-weight formula as `2 ** (-age / half_life)` (equivalent to `exp(-age * ln(2) / half_life)`). Pick one convention in the implementation and stay consistent.

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_criticality_distillation.py::test_score_age_weights_match_hand_computation -q`
Expected: FAIL — `AttributeError: 'CriticalityDistillation' object has no attribute 'score'`.

**Step 3: Minimal implementation**

Add to `CriticalityDistillation`:

```python
    def score(self, current_step: int) -> torch.Tensor:
        """Age-weighted average of evidence across the bank.

        Returns `[num_layers, dim]` fp32 score. Empty bank (no valid slots)
        produces zeros.

        Age weight is `2 ** (-age / trace_half_life_steps)`, so `age ==
        trace_half_life_steps` carries weight 0.5, and `age == 0` carries
        weight 1.0.
        """
        valid = self.bank_step >= 0  # [L, T]
        age = (int(current_step) - self.bank_step).clamp_min(0).to(dtype=torch.float32)
        weight = torch.pow(
            torch.tensor(2.0, dtype=torch.float32), -age / self.trace_half_life_steps
        )
        weight = weight * valid.to(dtype=torch.float32)  # zero-out empty slots
        weight_sum = weight.sum(dim=1, keepdim=True)  # [L, 1]
        weighted_evidence = (weight.unsqueeze(-1) * self.bank_evidence).sum(dim=1)  # [L, D]
        safe_denom = weight_sum.clamp_min(1e-12)
        score = weighted_evidence / safe_denom
        # Layers with zero total weight -> zeros (not NaN).
        score = torch.where(
            weight_sum > 0,
            score,
            torch.zeros_like(score),
        )
        return score
```

**Step 4: Run tests**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: 6 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: age-weighted score over per-layer bank"
```

### Task 1b.4 — `state_dict` round-trip test

**Files:**
- Modify: `tests/test_criticality_distillation.py`

No new implementation — this pins the buffer-registration decision against regression (codex amendment 3).

**Step 1: Write the failing test**

Append:

```python
def test_state_dict_round_trip_preserves_bank_and_baseline():
    cd1 = CriticalityDistillation(num_layers=1, dim=4, trace_ttl_steps=3, trace_half_life_steps=2.0)
    cd1.add_step_evidence(layer=0, step=0, evidence=torch.tensor([1.0, 0.0, 0.0, 0.0]), event_count=3.0)
    cd1.add_step_evidence(layer=0, step=1, evidence=torch.tensor([0.0, 1.0, 0.0, 0.0]), event_count=5.0)
    cd1.baseline_future_energy.fill_(0.42)
    cd1.seat_mask[0, 0] = True

    sd = cd1.state_dict()

    cd2 = CriticalityDistillation(num_layers=1, dim=4, trace_ttl_steps=3, trace_half_life_steps=2.0)
    cd2.load_state_dict(sd)

    assert torch.equal(cd2.bank_evidence, cd1.bank_evidence)
    assert torch.equal(cd2.bank_step, cd1.bank_step)
    assert torch.equal(cd2.bank_event_count, cd1.bank_event_count)
    assert torch.equal(cd2.baseline_future_energy, cd1.baseline_future_energy)
    assert torch.equal(cd2.seat_mask, cd1.seat_mask)

    # Score must match after round-trip.
    assert torch.allclose(cd2.score(current_step=2), cd1.score(current_step=2))
```

**Step 2: Run the test**

Run: `pytest tests/test_criticality_distillation.py::test_state_dict_round_trip_preserves_bank_and_baseline -q`
Expected: PASS (because all mechanism state is on registered buffers).

**Step 3: Commit**

```bash
git add tests/test_criticality_distillation.py
git commit -m "criticality: pin state_dict round-trip preserves bank and baseline"
```

---

## Stage 2 — Scoring mechanics (event mask, future energy, baseline EMA)

**Files created:** `tests/test_criticality_scoring.py`.
**Files modified:** `src/chaoscontrol/optim/criticality.py`.

### Task 2.1 — `compute_event_mask`

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Create: `tests/test_criticality_scoring.py`

**Step 1: Write the failing test**

Create `tests/test_criticality_scoring.py`:

```python
import torch
import pytest
from chaoscontrol.optim.criticality import compute_event_mask


def test_event_mask_selects_top_event_frac_positions():
    pressure = torch.arange(100, dtype=torch.float32).reshape(2, 50)
    mask = compute_event_mask(pressure, event_frac=0.1)  # top 10%
    assert mask.shape == pressure.shape
    assert mask.dtype == torch.bool
    # Top 10% of 100 = 10 positions marked True
    assert mask.sum().item() == 10
    # Those 10 positions are the highest-pressure ones
    assert (pressure[mask] >= pressure[~mask].max()).all()


def test_event_mask_handles_all_equal_pressure():
    pressure = torch.ones(2, 10)
    mask = compute_event_mask(pressure, event_frac=0.5)
    # Ties can resolve either way; just assert the count is correct
    assert mask.sum().item() == 10  # 0.5 * 20 total positions


def test_event_mask_empty_at_zero_frac_and_full_at_one():
    pressure = torch.randn(3, 4)
    assert compute_event_mask(pressure, event_frac=0.0).sum().item() == 0
    assert compute_event_mask(pressure, event_frac=1.0).sum().item() == pressure.numel()
```

**Step 2: Run tests**

Run: `pytest tests/test_criticality_scoring.py -q`
Expected: 3 tests fail — `ImportError: cannot import name 'compute_event_mask'`.

**Step 3: Minimal implementation**

Append to `src/chaoscontrol/optim/criticality.py`:

```python
def compute_event_mask(pressure: torch.Tensor, event_frac: float) -> torch.Tensor:
    """Top-`event_frac` positions of pressure become True.

    Args:
        pressure: any shape; absolute magnitude determines rank.
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
    if k >= total:
        return torch.ones_like(pressure, dtype=torch.bool)
    flat = pressure.reshape(-1)
    threshold = torch.topk(flat, k=k, largest=True).values[-1]
    return pressure >= threshold
```

Note on ties: `pressure >= threshold` may mark slightly more than exactly `k` positions when there are ties at the threshold. The second test tolerates this (total elements = 20, event_frac=0.5, so all 20 are marked). For the first test (distinct values from `arange`) this is exact.

**Step 4: Run tests**

Run: `pytest tests/test_criticality_scoring.py -q`
Expected: 3 passed.

**Step 5: Commit**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_scoring.py
git commit -m "criticality: compute_event_mask top-k by pressure"
```

### Task 2.2 — `compute_future_energy`

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_scoring.py`

**Step 1: Write the failing test**

Append to `tests/test_criticality_scoring.py`:

```python
from chaoscontrol.optim.criticality import compute_future_energy


def test_future_energy_matches_hand_computation_small_case():
    # states shape [B=1, T=4, D=2]
    states = torch.tensor([
        [[1.0, 0.0],
         [2.0, 1.0],
         [3.0, 2.0],
         [4.0, 3.0]]
    ])
    # Horizon H=2. For t=0, future = [t+1:t+3] = [[2,1],[3,2]]
    #   energy = mean([4, 1, 9, 4] grouped by channel) -> [mean(4, 9)=6.5, mean(1, 4)=2.5]
    # For t=1, future = [[3,2],[4,3]] -> [mean(9, 16)=12.5, mean(4, 9)=6.5]
    # For t=2, future = [[4,3]] only (t+H goes past end) -> [16.0, 9.0]
    # For t=3, no future window -> [0, 0] (convention: empty window -> zero)
    out = compute_future_energy(states, horizon_H=2)
    assert out.shape == (1, 4, 2)
    assert torch.allclose(out[0, 0], torch.tensor([6.5, 2.5]))
    assert torch.allclose(out[0, 1], torch.tensor([12.5, 6.5]))
    assert torch.allclose(out[0, 2], torch.tensor([16.0, 9.0]))
    assert torch.allclose(out[0, 3], torch.tensor([0.0, 0.0]))
```

**Step 2: Run the test**

Run: `pytest tests/test_criticality_scoring.py::test_future_energy_matches_hand_computation_small_case -q`
Expected: FAIL — `cannot import name 'compute_future_energy'`.

**Step 3: Minimal implementation**

Append to `src/chaoscontrol/optim/criticality.py`:

```python
def compute_future_energy(states: torch.Tensor, horizon_H: int) -> torch.Tensor:
    """Per-position mean-square energy over the trailing window `[t+1, t+H]`.

    Args:
        states: `[B, T, D]` recurrence states.
        horizon_H: window length (strictly positive).

    Returns:
        `[B, T, D]` — empty windows (tail where `t+1 >= T`) produce zeros.
    """
    if horizon_H < 1:
        raise ValueError(f"horizon_H must be >= 1; got {horizon_H}")
    B, T, D = states.shape
    sq = states.pow(2)  # [B, T, D]
    out = torch.zeros_like(sq)
    for t in range(T):
        start = t + 1
        stop = min(t + 1 + horizon_H, T)
        if start >= stop:
            continue  # empty window -> zeros
        out[:, t, :] = sq[:, start:stop, :].mean(dim=1)
    return out
```

This is an `O(T)` Python loop for clarity and correctness. Vectorize later if it becomes a hot path. (Unrolled cumulative-sum trick is possible but premature.)

**Step 4: Run the test**

Run: `pytest tests/test_criticality_scoring.py::test_future_energy_matches_hand_computation_small_case -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_scoring.py
git commit -m "criticality: compute_future_energy per-position trailing window"
```

### Task 2.3 — Baseline EMA update (non-event positions only)

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test**

Append to `tests/test_criticality_distillation.py`:

```python
def test_update_baseline_ema_only_reads_non_event_positions():
    cd = CriticalityDistillation(
        num_layers=1, dim=2, trace_ttl_steps=4,
        baseline_ema_decay=0.5,
    )
    # future_energy shape [B=1, T=4, D=2]
    future_energy = torch.tensor([[
        [1.0, 10.0],  # t=0, event
        [2.0, 20.0],  # t=1, non-event
        [3.0, 30.0],  # t=2, non-event
        [4.0, 40.0],  # t=3, event
    ]])
    event_mask = torch.tensor([[True, False, False, True]])
    # Mean over non-event positions: channel 0 -> (2+3)/2 = 2.5
    #                                channel 1 -> (20+30)/2 = 25.0
    # Baseline starts at zero. One update with decay=0.5:
    #   new = 0.5 * old + 0.5 * obs = 0.5 * 0 + 0.5 * [2.5, 25.0] = [1.25, 12.5]
    cd.update_baseline_ema(layer=0, future_energy=future_energy, event_mask=event_mask)
    assert torch.allclose(cd.baseline_future_energy[0], torch.tensor([1.25, 12.5]))


def test_update_baseline_ema_no_nonevent_positions_is_noop():
    cd = CriticalityDistillation(num_layers=1, dim=2, trace_ttl_steps=4, baseline_ema_decay=0.9)
    cd.baseline_future_energy.fill_(7.0)
    future_energy = torch.randn(1, 4, 2)
    event_mask = torch.ones(1, 4, dtype=torch.bool)  # every position is an event
    cd.update_baseline_ema(layer=0, future_energy=future_energy, event_mask=event_mask)
    assert torch.equal(cd.baseline_future_energy[0], torch.full((2,), 7.0))
```

**Step 2: Run tests**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: FAIL — no `update_baseline_ema`.

**Step 3: Minimal implementation**

In `CriticalityDistillation`:

```python
    @torch.no_grad()
    def update_baseline_ema(
        self,
        *,
        layer: int,
        future_energy: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> None:
        """Update per-channel baseline EMA using only non-event positions."""
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer={layer}")
        if future_energy.shape[-1] != self.dim:
            raise ValueError(
                f"future_energy last dim must be {self.dim}; got {tuple(future_energy.shape)}"
            )
        non_event = ~event_mask  # [B, T]
        if not non_event.any():
            return  # no new information; leave EMA alone
        flat_fe = future_energy.reshape(-1, self.dim)  # [B*T, D]
        flat_m = non_event.reshape(-1)  # [B*T]
        obs = flat_fe[flat_m].mean(dim=0)  # [D]
        decay = self.baseline_ema_decay
        self.baseline_future_energy[layer].mul_(decay).add_(obs.to(self.baseline_future_energy.dtype), alpha=(1.0 - decay))
```

**Step 4: Run tests**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: all passing (including the two new ones).

**Step 5: Commit**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: update_baseline_ema over non-event positions only"
```

### Task 2.4 — End-to-end evidence aggregation: `ingest_step`

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

This ties stages 2.1-2.3 together: given per-layer pressure + states + step, run event mask, future energy, baseline update, excess energy, aggregate, and write one evidence entry per layer.

**Step 1: Write the failing test**

Append:

```python
def test_ingest_step_writes_one_entry_per_layer_with_events():
    cd = CriticalityDistillation(
        num_layers=2, dim=3, trace_ttl_steps=4,
        baseline_ema_decay=0.0,  # baseline = observation (no smoothing) for easier math
    )
    states_l0 = torch.tensor([[
        [1.0, 0.0, 0.0],  # t=0 event
        [0.0, 1.0, 0.0],  # t=1 non-event
        [0.0, 0.0, 1.0],  # t=2 non-event (future window for event at t=0 covers t=1..end)
    ]])
    states_l1 = torch.tensor([[
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]])
    pressure = torch.tensor([[[10.0, 0.0, 0.0]]]).reshape(1, 3)  # top 1/3 at t=0
    # event_frac chosen so only t=0 is an event; event_mask = [True, False, False]
    cd.ingest_step(
        step=0,
        pressure=pressure,  # [B=1, T=3]
        states_per_layer=[states_l0, states_l1],
        horizon_H=2,
        event_frac=0.34,  # round(0.34 * 3) = 1 position
    )
    # Layer 0: future at t=0 over [t+1:t+3] = rows 1 and 2 -> mean([[0,1,0],[0,0,1]]**2, dim=0) = [0, 0.5, 0.5]
    # Baseline from non-event positions t=1..2: future at t=1 over [t+1:t+3] = [row 2] -> [0, 0, 1]
    #                                            future at t=2 over [t+1:t+3] = [] -> [0, 0, 0]
    # non-event future mean = ([0,0,1] + [0,0,0]) / 2 = [0, 0, 0.5]
    # With decay=0 baseline = observation -> [0, 0, 0.5]
    # excess = relu([0, 0.5, 0.5] - [0, 0, 0.5]) = [0, 0.5, 0]
    # Aggregated over 1 event position = [0, 0.5, 0]
    l0_slot = (cd.bank_step[0] == 0).nonzero(as_tuple=True)[0].item()
    assert torch.allclose(cd.bank_evidence[0, l0_slot], torch.tensor([0.0, 0.5, 0.0]), atol=1e-6)
    assert cd.bank_event_count[0, l0_slot].item() == pytest.approx(1.0)


def test_ingest_step_no_events_writes_nothing():
    cd = CriticalityDistillation(num_layers=1, dim=3, trace_ttl_steps=4)
    states = [torch.randn(1, 3, 3)]
    pressure = torch.zeros(1, 3)
    cd.ingest_step(step=0, pressure=pressure, states_per_layer=states, horizon_H=2, event_frac=0.0)
    assert (cd.bank_step == -1).all()
```

**Step 2: Run tests — they fail**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: FAIL — no `ingest_step`.

**Step 3: Minimal implementation**

Add to `CriticalityDistillation`:

```python
    @torch.no_grad()
    def ingest_step(
        self,
        *,
        step: int,
        pressure: torch.Tensor,
        states_per_layer: list,
        horizon_H: int,
        event_frac: float,
    ) -> None:
        """Full per-step evidence ingestion.

        Args:
            step: current training step index.
            pressure: `[B, T]` pressure field (any real-valued tensor).
            states_per_layer: list of length `num_layers`, each entry a
                `[B, T, D]` captured states tensor.
            horizon_H: trailing window for post-event energy.
            event_frac: fraction of positions to mark as events.
        """
        if len(states_per_layer) != self.num_layers:
            raise ValueError(
                f"states_per_layer must have {self.num_layers} entries; got {len(states_per_layer)}"
            )
        event_mask = compute_event_mask(pressure, event_frac=event_frac)  # [B, T]
        n_events = int(event_mask.sum().item())
        if n_events == 0:
            return
        for layer, states in enumerate(states_per_layer):
            if states.shape[-1] != self.dim:
                raise ValueError(
                    f"layer {layer}: states last dim {states.shape[-1]} != self.dim {self.dim}"
                )
            future_energy = compute_future_energy(states, horizon_H=horizon_H)  # [B, T, D]
            self.update_baseline_ema(
                layer=layer, future_energy=future_energy, event_mask=event_mask
            )
            baseline = self.baseline_future_energy[layer]  # [D]
            excess = (future_energy - baseline).clamp_min(0.0)  # [B, T, D]
            # Aggregate: mean over event positions.
            flat_excess = excess.reshape(-1, self.dim)  # [B*T, D]
            flat_mask = event_mask.reshape(-1)  # [B*T]
            aggregate = flat_excess[flat_mask].mean(dim=0)  # [D]
            self.add_step_evidence(
                layer=layer,
                step=step,
                evidence=aggregate,
                event_count=float(flat_mask.sum().item()),
            )
```

**Step 4: Run tests**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: all passing.

**Step 5: Commit**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: end-to-end ingest_step ties event mask, future energy, baseline EMA"
```

---

## Stage 3 — Allocator + seat-masked loss

### Task 3.1 — Evidence-gated top-k seat allocation

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test**

Append:

```python
def test_allocate_seats_respects_evidence_gate():
    cd = CriticalityDistillation(
        num_layers=1, dim=10, trace_ttl_steps=4,
        trace_half_life_steps=1.0,
        criticality_budget_frac=0.3,
        min_weighted_events_per_layer=100.0,  # unreachable with small input
    )
    cd.add_step_evidence(layer=0, step=0, evidence=torch.ones(10), event_count=1.0)
    cd.allocate_seats(current_step=1)
    assert not cd.seat_mask[0].any(), "evidence gate must suppress seat assignment"


def test_allocate_seats_top_k_when_gate_passes():
    cd = CriticalityDistillation(
        num_layers=1, dim=10, trace_ttl_steps=4,
        trace_half_life_steps=100.0,  # slow aging, so recent events count fully
        criticality_budget_frac=0.3,  # 3 seats per layer
        min_weighted_events_per_layer=1.0,
    )
    # Channels 2, 5, 7 have highest evidence.
    evidence = torch.zeros(10)
    evidence[2] = 3.0
    evidence[5] = 5.0
    evidence[7] = 1.0
    evidence[0] = 0.5
    cd.add_step_evidence(layer=0, step=0, evidence=evidence, event_count=10.0)
    cd.allocate_seats(current_step=1)
    assert cd.seat_mask[0].sum().item() == 3
    # Top-3 by score: channels 5, 2, 7 (in order of magnitude)
    assert cd.seat_mask[0, 5].item() is True
    assert cd.seat_mask[0, 2].item() is True
    assert cd.seat_mask[0, 7].item() is True
```

**Step 2: Run tests — fail**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: FAIL — no `allocate_seats`.

**Step 3: Minimal implementation**

Add to `CriticalityDistillation`:

```python
    @torch.no_grad()
    def allocate_seats(self, *, current_step: int) -> None:
        """Recompute per-layer seat assignment from current age-weighted score.

        Gate: a layer's total age-weighted event count must exceed
        `min_weighted_events_per_layer`; otherwise its seats are cleared.
        """
        valid = self.bank_step >= 0
        age = (int(current_step) - self.bank_step).clamp_min(0).to(dtype=torch.float32)
        weight = torch.pow(
            torch.tensor(2.0, dtype=torch.float32), -age / self.trace_half_life_steps
        )
        weight = weight * valid.to(dtype=torch.float32)  # [L, T]
        weighted_events_per_layer = (weight * self.bank_event_count).sum(dim=1)  # [L]

        k = max(1, int(round(self.dim * self.criticality_budget_frac)))
        scores = self.score(current_step=current_step)  # [L, D]

        for layer in range(self.num_layers):
            if weighted_events_per_layer[layer].item() < self.min_weighted_events_per_layer:
                self.seat_mask[layer].fill_(False)
                continue
            topk = torch.topk(scores[layer], k=k, largest=True)
            mask = torch.zeros(self.dim, dtype=torch.bool)
            mask[topk.indices] = True
            self.seat_mask[layer] = mask
```

**Step 4: Run tests**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: all passing.

**Step 5: Commit**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: evidence-gated top-k seat allocator"
```

### Task 3.2 — Seat-masked criticality loss

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test**

Append:

```python
def test_criticality_loss_is_zero_when_no_seats():
    cd = CriticalityDistillation(num_layers=1, dim=4, trace_ttl_steps=2)
    # seat_mask is all False by default.
    log_a_per_layer = [torch.zeros(4, requires_grad=True)]
    loss = cd.criticality_loss(log_a_per_layer)
    assert loss.item() == 0.0


def test_criticality_loss_values_match_hand_mse_on_seats_only():
    cd = CriticalityDistillation(
        num_layers=1, dim=4, trace_ttl_steps=2,
        critical_value=0.9,
        criticality_distill_weight=1.0,  # isolate per-layer mse
    )
    cd.seat_mask[0] = torch.tensor([True, False, True, False])
    # 1 - sigmoid(log_a=0) = 0.5 on every channel.
    log_a_per_layer = [torch.zeros(4, requires_grad=True)]
    # For seat channels: (0.5 - 0.9)^2 = 0.16. Mean over 2 seats = 0.16.
    loss = cd.criticality_loss(log_a_per_layer)
    assert torch.allclose(loss, torch.tensor(0.16), atol=1e-6)
```

**Step 2: Run — fail**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: FAIL — no `criticality_loss`.

**Step 3: Minimal implementation**

Add to `CriticalityDistillation`:

```python
    def criticality_loss(self, log_a_per_layer: list) -> torch.Tensor:
        """Seat-masked MSE loss pulling `1 - sigmoid(log_a[seat])` toward
        `critical_value`.

        Non-seat channels contribute exactly zero to the loss (and therefore
        exactly zero gradient to their log_a).

        Returns:
            Scalar tensor. Weight of this term in the total loss is applied
            externally (`criticality_distill_weight` is not multiplied here).
        """
        if len(log_a_per_layer) != self.num_layers:
            raise ValueError(
                f"log_a_per_layer must have {self.num_layers} entries"
            )
        total = torch.zeros((), dtype=torch.float32, device=self.seat_mask.device)
        any_seats = False
        for layer, log_a in enumerate(log_a_per_layer):
            mask = self.seat_mask[layer]
            if not mask.any():
                continue
            any_seats = True
            criticality = 1.0 - torch.sigmoid(log_a.to(dtype=torch.float32))
            err = (criticality - self.critical_value) ** 2
            # Select seat entries explicitly so non-seats contribute no op that
            # could produce grad through masking arithmetic.
            seat_err = err[mask]
            total = total + seat_err.mean()
        if not any_seats:
            return torch.zeros((), dtype=torch.float32)
        return total
```

**Step 4: Run tests**

Run: `pytest tests/test_criticality_distillation.py -q`
Expected: all passing.

**Step 5: Commit**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: seat-masked MSE loss on log_a (scalar output)"
```

### Task 3.3 — Non-seat `log_a` receives exactly zero gradient (codex amendment)

**Files:**
- Modify: `tests/test_criticality_distillation.py`

This is the explicit regression test codex asked for.

**Step 1: Write the failing test**

Append:

```python
def test_non_seat_log_a_gets_exactly_zero_gradient_from_criticality_loss():
    cd = CriticalityDistillation(
        num_layers=1, dim=6, trace_ttl_steps=2,
        critical_value=0.9,
    )
    cd.seat_mask[0] = torch.tensor([True, False, True, False, False, True])
    log_a = torch.zeros(6, requires_grad=True)
    loss = cd.criticality_loss([log_a])
    loss.backward()
    # Non-seat entries are indices 1, 3, 4 — their grad MUST be exactly zero.
    non_seat_grad = log_a.grad[~cd.seat_mask[0]]
    assert torch.equal(non_seat_grad, torch.zeros_like(non_seat_grad)), (
        f"non-seat log_a must have exactly zero grad; got {non_seat_grad}"
    )
    # Seat entries must receive a nonzero grad (sanity — loss depends on them).
    seat_grad = log_a.grad[cd.seat_mask[0]]
    assert (seat_grad != 0.0).all(), (
        f"seat log_a must have nonzero grad; got {seat_grad}"
    )
```

**Step 2: Run — should PASS**

Run: `pytest tests/test_criticality_distillation.py::test_non_seat_log_a_gets_exactly_zero_gradient_from_criticality_loss -q`
Expected: PASS (implementation in 3.2 already uses `err[mask]` which excludes non-seat entries from the loss graph).

If it fails, the implementation is sneaking a non-seat gradient through — investigate.

**Step 3: Commit**

```bash
git add tests/test_criticality_distillation.py
git commit -m "criticality: pin non-seat log_a receives exactly zero gradient"
```

### Task 3.4 — Seat target is non-differentiable

**Files:**
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test**

Append:

```python
def test_criticality_loss_has_no_grad_path_to_pressure_or_states():
    cd = CriticalityDistillation(
        num_layers=1, dim=4, trace_ttl_steps=2,
        critical_value=0.9, min_weighted_events_per_layer=0.0,
    )
    # Populate one evidence entry and seat.
    evidence = torch.tensor([1.0, 0.0, 2.0, 0.0])
    cd.add_step_evidence(layer=0, step=0, evidence=evidence, event_count=1.0)
    cd.allocate_seats(current_step=1)

    # Pressure and states are usually produced with grad in the training
    # graph — but the criticality loss should only depend on log_a.
    log_a = torch.zeros(4, requires_grad=True)
    loss = cd.criticality_loss([log_a])
    # If the loss depends on anything other than log_a, gradients on a
    # fresh unrelated tensor should cause an error when we try to extract
    # them. Assert directly: loss.backward consumes log_a only.
    assert loss.requires_grad
    loss.backward()
    assert log_a.grad is not None
    # seat_mask is a registered buffer and should not receive grads.
    assert not cd.seat_mask.requires_grad
    # baseline_future_energy should not receive grads either.
    assert not cd.baseline_future_energy.requires_grad
```

**Step 2: Run the test**

Run: `pytest tests/test_criticality_distillation.py::test_criticality_loss_has_no_grad_path_to_pressure_or_states -q`
Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_criticality_distillation.py
git commit -m "criticality: pin loss has no grad path outside log_a"
```

---

## Stage 3 validation — full mechanism integration test

### Task 3.5 — End-to-end: capture → ingest → allocate → loss moves seat log_a

**Files:**
- Modify: `tests/test_criticality_distillation.py`

This is the single most load-bearing test. It simulates a training-step loop and asserts that the seat `log_a` values move more than non-seat `log_a` values.

**Step 1: Write the failing test**

Append:

```python
def test_full_mechanism_moves_seat_log_a_more_than_non_seat():
    """After N training steps, seat-channel log_a values should move
    meaningfully while non-seat log_a values stay pinned."""
    from chaoscontrol.core import ChaosSSMCore

    torch.manual_seed(0)
    dim = 8
    core = ChaosSSMCore(dim=dim, a_mode="diag")
    cd = CriticalityDistillation(
        num_layers=1,
        dim=dim,
        trace_ttl_steps=16,
        trace_half_life_steps=4.0,
        seat_refresh_interval=1,  # refresh every step for this test
        criticality_budget_frac=0.25,  # 2 seats
        critical_value=0.99,
        min_weighted_events_per_layer=0.0,  # no gate for this integration test
        criticality_distill_weight=1.0,
    )

    # Snapshot initial log_a.
    log_a_init = core.log_a.detach().clone()

    # Optimizer for log_a only (isolate the mechanism).
    opt = torch.optim.SGD([core.log_a], lr=0.5)

    # Force a specific seat pattern by biasing pressure to only channels
    # 0 and 1 — we expect them to become seats.
    for step in range(10):
        x = torch.randn(2, 6, dim)
        with core.capture_states() as get_states:
            _ = core(x)
            states = get_states()
        # Pressure field biased to T=0, T=1 positions (so events concentrate there).
        pressure = torch.zeros(2, 6)
        pressure[:, 0] = 10.0
        pressure[:, 1] = 8.0
        # Construct states that light up channels 0 and 1 after events.
        fake_states = torch.zeros_like(states)
        fake_states[:, 1:, 0] = 1.0  # channel 0 persists after t=0 events
        fake_states[:, 2:, 1] = 1.0  # channel 1 persists after t=0, 1 events

        cd.ingest_step(
            step=step,
            pressure=pressure,
            states_per_layer=[fake_states],
            horizon_H=4,
            event_frac=2.0 / 12.0,  # top ~2 positions per 12
        )
        cd.allocate_seats(current_step=step + 1)

        opt.zero_grad()
        loss = cd.criticality_loss([core.log_a])
        if loss.requires_grad:
            loss.backward()
            opt.step()

    # Seat channels (0 and 1) should have moved; non-seat channels stay put.
    log_a_delta = (core.log_a.detach() - log_a_init).abs()
    seats = cd.seat_mask[0]
    # At least one of the seat channels moved
    assert log_a_delta[seats].max().item() > 1e-3, (
        f"seat log_a did not move: {log_a_delta[seats]}"
    )
    # No non-seat channel moved
    assert log_a_delta[~seats].max().item() < 1e-6, (
        f"non-seat log_a moved: {log_a_delta[~seats]}"
    )
```

**Step 2: Run the test**

Run: `pytest tests/test_criticality_distillation.py::test_full_mechanism_moves_seat_log_a_more_than_non_seat -q`
Expected: PASS.

If it fails, that means one of: (a) seat allocator isn't firing (check `cd.seat_mask[0]` intermediate values), (b) loss has a bad grad path (re-check 3.3/3.4), (c) baseline EMA is eating all signal (try decay=0 for the test).

**Step 3: Commit**

```bash
git add tests/test_criticality_distillation.py
git commit -m "criticality: end-to-end integration test — seat log_a moves, non-seat stays"
```

---

## Out of scope for this plan

Deliberately *not* in this plan (follow-ups after the above lands green):

- Wiring `CriticalityDistillation` into `runner_fast_path.py` training loop.
- Exp 24 matrix configs for the 4-cell first smoke (treatment / telemetry-only / shuffled-teacher / budget-only).
- `rare_bucket_ce` diagnostic readout.
- Ablations (`horizon_H`, per-layer vs cross-layer, baseline control type, budget fraction).
- Non-diag backend support (explicitly raises `NotImplementedError` at `capture_states()` entry per 1a.1).

These belong in the next plan, written once Stage 3 is green and the mechanism is proven to move log_a correctly in isolation.

---

## Final verification (end of plan)

Before declaring the plan complete, run the full test suite that covers this work:

```bash
pytest tests/test_ssm_state_capture.py tests/test_criticality_distillation.py tests/test_criticality_scoring.py -v
```

Expected: all passing. Count and list the tests; verify none are unexpectedly skipped.

Verify with a clean state:

```bash
git status      # must be clean; no untracked or modified files
git log --oneline -20    # every task above landed a commit
```
