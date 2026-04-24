# Criticality Distillation Runner Wiring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task. Use @superpowers:test-driven-development for every task (red-fail → implement → green-pass → commit). Apply @superpowers:verification-before-completion before claiming any step is done.

**Goal:** Wire the already-landed `CriticalityDistillation` module (Stages 1a/1b/2/3, terminal commit `c0318d8`) into `experiments/23_fast_path/runner_fast_path.py` and emit an 8-cell `cd_first_smoke` matrix. Every cell rides on the locked `control_fastslow_only_i64a025` operational stack (fast/slow interval 64, alpha 0.25, eval copy = slow, Dreamworld off) so the mechanism read isolates CD's effect on top of the actual submission base rather than a generic Muon base.

**Architecture:** CD pressure = `relu(CE - H[p])` computed independently of ScOpt. Bank + baseline EMA + allocator live on CPU; `seat_mask` lives on GPU. ExitStack at runner level for state capture. Two-phase ingest with synchronous D2H for first smoke (async side-stream deferred). Non-fused LM head path on CD cells so full logits exist for entropy.

**Tech Stack:** PyTorch 2.9, pytest, chaoscontrol runner_fast_path, exp24 matrix builders.

**Required reading before starting:**
- Mechanism design: `docs/plans/2026-04-24-criticality-distillation.md` (v3, commit `4b7d77c`)
- Runner-wiring design: `docs/plans/2026-04-24-criticality-distillation-runner-design.md` (commit `8293653`)
- Locked fast/slow base: `build_phase0_fastslow_only_control` in `experiments/24_training_time_bundle/exp24.py` (the knobs CD rides on)
- Implemented module: `src/chaoscontrol/optim/criticality.py`

**Non-gating feedback files** (read these, they'll save time):
- `feedback_regression_is_never_build_error.md` — tests must fail on the bug, not just collect
- `feedback_risks_not_implementation_challenges.md` — no "risk" padding in reports
- `feedback_estimate_calibration.md` — subtask-based effort estimates, no "multi-day" hand-waves

---

## Conventions

- Every task is TDD: red, implement, green, commit.
- Commits: `component: imperative sentence`.
- Test runner on macOS: `/opt/homebrew/bin/python3.11 -m pytest ...`
- All tests run on CPU using contrived tensors. GPU-only behavior gated behind `torch.cuda.is_available()` skips.

---

## Stage A — `compute_event_mask` conditional top-k

### Task A.1 — Strictly-positive top-k, event mask returns empty on zero-pressure

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py` (the module-level `compute_event_mask`)
- Modify: `tests/test_criticality_scoring.py`

**Step 1: Write the failing tests.**

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
    mask = compute_event_mask(pressure, event_frac=0.5)
    # Strict positives: indices 2, 5, 8, 9. event_frac 0.5 → k=5 requested,
    # only 4 positives exist, so 4 events returned at the positive indices.
    assert mask.sum().item() == 4
    for idx in [2, 5, 8, 9]:
        assert mask[0, idx].item() is True
    for idx in [0, 1, 3, 4, 6, 7]:
        assert mask[0, idx].item() is False
```

**Step 2: Run — expect FAIL.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_scoring.py::test_event_mask_returns_empty_when_no_positive_pressure tests/test_criticality_scoring.py::test_event_mask_only_marks_strictly_positive_positions -v
```
Current `compute_event_mask` returns k positions regardless of sign — both new tests fail.

**Step 3: Fix `compute_event_mask`.**

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

**Step 4: Run the full scoring + distillation suites.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_scoring.py tests/test_criticality_distillation.py -q
```

The existing `test_event_mask_handles_all_equal_pressure` still passes (uniform-ONES is all-positive, so k=10 events fire). Any distillation test that relied on zero/uniform pressure producing events must be audited and updated — none currently do, but spot-check.

**Step 5: Commit.**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_scoring.py
git commit -m "criticality: conditional top-k — event mask only marks strictly-positive pressure"
```

---

## Stage B — CD knobs for falsifier cells

### Task B.1 — `score_permute_before_topk` (shuffled-teacher falsifier, genuinely random)

**Context.** The first draft of this plan had a bug: permuting scores and then mapping top-k indices back through the permutation recovers the original score peaks exactly — the permutation is silently undone. The correct shuffled-teacher semantics is "seat assignment is random w.r.t. the score." Simplest implementation: when the flag is on, bypass score-based top-k entirely and pick k random channels per layer.

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing tests.**

Append:

```python
def test_shuffled_teacher_does_not_select_score_peaks():
    """With score_permute_before_topk=True, seat assignment is random —
    NOT the top-k by score. The whole point of the falsifier is to break
    the channel-identity correspondence between score peaks and seats.
    """
    torch.manual_seed(42)
    cd = CriticalityDistillation(
        num_layers=1, dim=10, trace_ttl_steps=4,
        trace_half_life_steps=100.0,
        criticality_budget_frac=0.3,  # k=3
        min_weighted_events_per_layer=1.0,
        score_permute_before_topk=True,
    )
    # Evidence concentrated on channels 0, 1, 2.
    evidence = torch.zeros(10)
    evidence[0] = 10.0
    evidence[1] = 9.0
    evidence[2] = 8.0
    cd.add_step_evidence(layer=0, step=0, evidence=evidence, event_count=10.0)
    cd.allocate_seats(current_step=1)
    peak_channels = {0, 1, 2}
    selected = set(cd.seat_mask[0].nonzero().flatten().tolist())
    assert len(selected) == 3
    # With seed 42 and random k-of-10 selection, the probability of
    # exactly recovering {0,1,2} is 1 / C(10,3) = 1/120. The seed pins
    # the non-match.
    assert selected != peak_channels, (
        f"shuffled-teacher must not track score peaks; got {selected}"
    )


def test_default_allocate_seats_still_picks_score_peaks():
    """Regression pin against the fix — the non-shuffled path is
    unchanged."""
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

**Step 2: Run — expect FAIL.**

The new tests fail: constructor doesn't accept `score_permute_before_topk`.

**Step 3: Implement.**

In `CriticalityDistillation.__init__`, add after the existing parameters:

```python
        score_permute_before_topk: bool = False,
```

Store `self.score_permute_before_topk = bool(score_permute_before_topk)`.

In `allocate_seats`, replace the top-k selection block:

```python
            if weighted_events_per_layer[layer].item() < self.min_weighted_events_per_layer:
                self.seat_mask[layer].fill_(False)
                continue
            mask = torch.zeros(self.dim, dtype=torch.bool, device=self.seat_mask.device)
            if self.score_permute_before_topk:
                # Random k-of-D channels — ignores score entirely. The
                # evidence gate is still respected above, so layers
                # without evidence get no seats even in this mode.
                perm = torch.randperm(self.dim, device=self.seat_mask.device)
                mask[perm[:k]] = True
            else:
                topk = torch.topk(scores[layer], k=k, largest=True)
                mask[topk.indices] = True
            self.seat_mask[layer] = mask
```

**Step 4: Run.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_distillation.py -q
```
All pass.

**Step 5: Commit.**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: score_permute_before_topk flag — random k-of-D for shuffled-teacher falsifier"
```

### Task B.2 — `fixed_random_seats` flag (design-faithful budget-only falsifier)

**Context.** The original design's `budget_only` cell is "no scoring, no bank, random fixed top-k seats held constant throughout training." This tests "does any fixed criticality budget help, regardless of channel selection?" Implementation: at init, pick random k channels per layer and never change them. Skip ingest and allocate entirely.

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing tests.**

Append:

```python
def test_fixed_random_seats_binds_once_at_init_and_never_changes():
    torch.manual_seed(7)
    cd = CriticalityDistillation(
        num_layers=2, dim=8, trace_ttl_steps=4,
        criticality_budget_frac=0.25,  # k=2 per layer
        fixed_random_seats=True,
    )
    # Seats bound at init.
    assert cd.seat_mask[0].sum().item() == 2
    assert cd.seat_mask[1].sum().item() == 2
    initial_mask_0 = cd.seat_mask[0].clone()
    initial_mask_1 = cd.seat_mask[1].clone()
    # Per-layer masks differ (different random draws).
    if initial_mask_0.shape == initial_mask_1.shape:
        # Not strictly required but the test seed makes it likely.
        pass
    # allocate_seats is a no-op.
    cd.add_step_evidence(layer=0, step=0, evidence=torch.tensor([5.0]*8), event_count=1.0)
    cd.allocate_seats(current_step=1)
    assert torch.equal(cd.seat_mask[0], initial_mask_0)
    assert torch.equal(cd.seat_mask[1], initial_mask_1)


def test_fixed_random_seats_with_default_false_does_not_prebind():
    torch.manual_seed(7)
    cd = CriticalityDistillation(
        num_layers=1, dim=8, trace_ttl_steps=4,
        criticality_budget_frac=0.25,
    )
    # Default False — no seats pre-bound.
    assert cd.seat_mask.sum().item() == 0
```

**Step 2: Run — expect FAIL.**

Constructor doesn't accept `fixed_random_seats`.

**Step 3: Implement.**

In `__init__`, add the flag:

```python
        fixed_random_seats: bool = False,
```

Store `self.fixed_random_seats = bool(fixed_random_seats)`.

After the buffer registrations in `__init__`, add:

```python
        if self.fixed_random_seats:
            k = max(1, int(round(self.dim * self.criticality_budget_frac)))
            for layer in range(self.num_layers):
                perm = torch.randperm(self.dim)
                self.seat_mask[layer, perm[:k]] = True
```

In `allocate_seats`, add an early return at the top:

```python
    @torch.no_grad()
    def allocate_seats(self, *, current_step: int) -> None:
        """..."""
        if self.fixed_random_seats:
            # Seats bound once at init and never change — no-op.
            return
        # ... existing body unchanged ...
```

**Step 4: Run.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_criticality_distillation.py -q
```
All pass.

**Step 5: Commit.**

```bash
git add src/chaoscontrol/optim/criticality.py tests/test_criticality_distillation.py
git commit -m "criticality: fixed_random_seats flag for design-faithful budget-only falsifier"
```

---

## Stage C — CPU/GPU split infrastructure

### Task C.1 — `sync_seat_mask_to_device` helper

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Write the failing test.**

```python
def test_seat_mask_can_live_on_different_device_than_bank():
    cd = CriticalityDistillation(
        num_layers=1, dim=4, trace_ttl_steps=2,
        criticality_budget_frac=0.5,
        min_weighted_events_per_layer=0.0,
    )
    cd.sync_seat_mask_to_device(torch.device('cpu'))
    assert cd.seat_mask.device == torch.device('cpu')
    cd.add_step_evidence(layer=0, step=0, evidence=torch.tensor([1.0, 2.0, 0.0, 0.0]), event_count=1.0)
    cd.allocate_seats(current_step=1)
    assert cd.seat_mask.device == torch.device('cpu')
    log_a = torch.zeros(4, requires_grad=True)
    loss = cd.criticality_loss([log_a])
    assert loss.device == torch.device('cpu')
```

**Step 2-5:** Same pattern as before. Implement:

```python
    def sync_seat_mask_to_device(self, device: torch.device) -> None:
        """Move (or keep) seat_mask on the specified device."""
        self.seat_mask.data = self.seat_mask.data.to(device)
```

Commit: `criticality: sync_seat_mask_to_device for CPU-bank + GPU-seat_mask split`

### Task C.2 — Two-phase ingest (`ingest_gpu` + `ingest_cpu_from_prepared`)

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Failing test — split produces same bank state as single-call.**

```python
def test_ingest_two_phase_matches_single_phase_output():
    cd_single = CriticalityDistillation(num_layers=1, dim=3, trace_ttl_steps=4, baseline_ema_decay=0.0)
    cd_split = CriticalityDistillation(num_layers=1, dim=3, trace_ttl_steps=4, baseline_ema_decay=0.0)
    states_l0 = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]])
    pressure = torch.tensor([[10.0, 0.0, 0.0]])
    cd_single.ingest_step(step=0, pressure=pressure, states_per_layer=[states_l0], horizon_H=2, event_frac=0.34)
    prepared = cd_split.ingest_gpu(pressure=pressure, states_per_layer=[states_l0], horizon_H=2, event_frac=0.34)
    cd_split.ingest_cpu_from_prepared(step=0, prepared=prepared)
    assert torch.allclose(cd_single.bank_evidence, cd_split.bank_evidence)
    assert torch.equal(cd_single.bank_step, cd_split.bank_step)
    assert torch.allclose(cd_single.bank_event_count, cd_split.bank_event_count)
    assert torch.allclose(cd_single.baseline_future_energy, cd_split.baseline_future_energy)
    assert torch.equal(cd_single.baseline_initialized, cd_split.baseline_initialized)


def test_ingest_gpu_returns_expected_dict_keys():
    cd = CriticalityDistillation(num_layers=2, dim=3, trace_ttl_steps=4)
    prepared = cd.ingest_gpu(
        pressure=torch.randn(1, 4),
        states_per_layer=[torch.randn(1, 4, 3), torch.randn(1, 4, 3)],
        horizon_H=2,
        event_frac=0.5,
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

**Step 2: Run — expect FAIL (methods don't exist).**

**Step 3: Implement.**

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
        if len(states_per_layer) != self.num_layers:
            raise ValueError(
                f"states_per_layer must have {self.num_layers} entries; got {len(states_per_layer)}"
            )
        event_mask = compute_event_mask(pressure, event_frac=event_frac)
        flat_mask = event_mask.reshape(-1)
        flat_non_event = ~flat_mask
        aggregated_excess = []
        non_event_mean_future_energy = []
        event_count = []
        for layer, states in enumerate(states_per_layer):
            if states.shape[-1] != self.dim:
                raise ValueError(
                    f"layer {layer}: states last dim {states.shape[-1]} != self.dim {self.dim}"
                )
            future_energy = compute_future_energy(states, horizon_H=horizon_H)
            flat_fe = future_energy.reshape(-1, self.dim)
            if flat_non_event.any():
                nonevt_mean = flat_fe[flat_non_event].mean(dim=0)
            else:
                nonevt_mean = torch.zeros(self.dim, dtype=flat_fe.dtype, device=flat_fe.device)
            non_event_mean_future_energy.append(nonevt_mean)
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
        agg = prepared["aggregated_excess_per_layer"].to(
            device=self.bank_evidence.device, dtype=self.bank_evidence.dtype,
        )
        nonevt = prepared["non_event_mean_future_energy_per_layer"].to(
            device=self.baseline_future_energy.device, dtype=self.baseline_future_energy.dtype,
        )
        counts = prepared["event_count_per_layer"].to(
            device=self.bank_event_count.device, dtype=self.bank_event_count.dtype,
        )
        had_non_events = bool((~prepared["event_mask"]).any().item())
        decay = self.baseline_ema_decay
        for layer in range(self.num_layers):
            if had_non_events:
                obs = nonevt[layer]
                if not bool(self.baseline_initialized[layer].item()):
                    self.baseline_future_energy[layer].copy_(obs)
                    self.baseline_initialized[layer] = True
                else:
                    self.baseline_future_energy[layer].mul_(decay).add_(obs, alpha=(1.0 - decay))
            cnt = float(counts[layer].item())
            if cnt > 0:
                self.add_step_evidence(
                    layer=layer,
                    step=step,
                    evidence=agg[layer],
                    event_count=cnt,
                )
```

**Step 4: Run — all tests pass.**
**Step 5: Commit:** `criticality: two-phase ingest (ingest_gpu + ingest_cpu_from_prepared)`

---

## Stage D — Runner integration

### Task D.1 — Pressure computation helper

**Files:**
- Create: `experiments/_23_fast_path_runner_helpers.py`
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Create: `tests/test_runner_criticality_pressure.py`

**Step 1: Write failing tests.**

`tests/test_runner_criticality_pressure.py`:

```python
import torch
import pytest
from experiments._23_fast_path_runner_helpers import compute_ce_minus_entropy_pressure


def test_pressure_fires_on_confident_wrong():
    logits = torch.tensor([[[5.0, 5.0, 0.0]]])
    targets = torch.tensor([[2]])
    pressure = compute_ce_minus_entropy_pressure(logits, targets)
    assert pressure.shape == (1, 1)
    assert pressure[0, 0].item() > 3.0


def test_pressure_suppresses_confused_wrong():
    logits = torch.zeros(1, 1, 10)
    targets = torch.tensor([[3]])
    pressure = compute_ce_minus_entropy_pressure(logits, targets)
    assert abs(pressure[0, 0].item()) < 1e-5


def test_pressure_nonnegative_always():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8)
    targets = torch.randint(0, 8, (2, 4))
    pressure = compute_ce_minus_entropy_pressure(logits, targets)
    assert (pressure >= 0.0).all()
```

**Step 2: Run — expect ImportError.**

**Step 3: Implement.**

`experiments/_23_fast_path_runner_helpers.py`:

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
        `[B, T]` fp32 non-negative pressure.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    ce = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return (ce - entropy).clamp_min(0.0)
```

Add a re-export in `runner_fast_path.py`:

```python
from experiments._23_fast_path_runner_helpers import (  # noqa: E402
    compute_ce_minus_entropy_pressure,
)
```

**Step 4: Run — all pass.**
**Step 5: Commit:** `runner: compute_ce_minus_entropy_pressure helper for criticality distillation`

### Task D.2 — CD construction + train-step plumbing with lm_head_backward_mode validation

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Modify: `tests/test_exp23_fast_path.py`

**Step 1: Failing tests.**

```python
def test_train_fast_for_budget_raises_if_cd_enabled_with_fused_lm_head():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="criticality_distill_enabled.*fused"):
        mod.train_fast_for_budget(
            model,
            train_tokens=torch.arange(128, dtype=torch.int16) % 6,
            train_num_tokens=128,
            stride=4, seq_len=3, batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer,
            budget_seconds=300.0,
            chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
            rank=0, world_size=1, seed=123, precision="fp32",
            stop_check_interval=1, stop_margin_seconds=0.0,
            vocab_size=6, max_steps=1, prefetch_batches=False,
            lm_head_backward_mode="fused_streaming_cached",
            criticality_distill_enabled=True,
        )


def test_train_fast_for_budget_wires_criticality_distillation_when_configured(monkeypatch):
    mod = _load_runner_module()
    from chaoscontrol.optim.criticality import CriticalityDistillation

    model = _TinyTokenTrainModel()
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

    mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4, seq_len=3, batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=4, prefetch_batches=False,
        lm_head_backward_mode="single",  # CD requires non-fused for entropy
        criticality_distill_enabled=True,
        criticality_distill_weight=1e-3,
        criticality_distill_budget_frac=0.25,
        criticality_distill_half_life_steps=100.0,
        criticality_distill_ttl_steps=400,
        criticality_distill_horizon_H=2,
        criticality_distill_event_frac=0.25,
        criticality_distill_seat_refresh_interval=2,
        criticality_distill_min_weighted_events_per_layer=0.0,
        criticality_distill_critical_value=0.95,
        criticality_distill_uniform_pressure=False,
        criticality_distill_score_permute_before_topk=False,
        criticality_distill_fixed_random_seats=False,
    )

    assert len(ingest_gpu_calls) == 4
    assert len(ingest_cpu_calls) == 4
    assert len(allocate_calls) == 2  # seat_refresh_interval=2 over 4 steps
    assert len(sync_calls) >= 2


def test_train_fast_for_budget_cd_fixed_random_seats_skips_ingest_and_allocate(monkeypatch):
    """Fixed-random-seats cell: no ingest, no allocate, seats bound at
    init and held."""
    mod = _load_runner_module()
    from chaoscontrol.optim.criticality import CriticalityDistillation

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ingest_calls = []

    monkeypatch.setattr(
        CriticalityDistillation,
        "ingest_gpu",
        lambda self, **kw: (ingest_calls.append(kw) or {"event_mask": torch.zeros(1,1,dtype=torch.bool),
                                                        "aggregated_excess_per_layer": torch.zeros(1,1),
                                                        "non_event_mean_future_energy_per_layer": torch.zeros(1,1),
                                                        "event_count_per_layer": torch.zeros(1)})
    )

    mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4, seq_len=3, batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=4, prefetch_batches=False,
        lm_head_backward_mode="single",
        criticality_distill_enabled=True,
        criticality_distill_fixed_random_seats=True,
        # Other CD knobs at defaults.
    )
    assert len(ingest_calls) == 0, (
        "fixed_random_seats cell must skip ingest entirely"
    )
```

**Step 2: Run — all fail.**

**Step 3: Implement plumbing in `train_fast_for_budget`.**

Add these kwargs (all default so existing runs unchanged):

```python
    # CD kwargs
    criticality_distill_enabled: bool = False,
    criticality_distill_weight: float = 1e-3,
    criticality_distill_budget_frac: float = 0.15,
    criticality_distill_critical_value: float = 0.95,
    criticality_distill_half_life_steps: float = 2048.0,
    criticality_distill_ttl_steps: int = 20480,
    criticality_distill_horizon_H: int = 64,
    criticality_distill_event_frac: float = 0.05,
    criticality_distill_seat_refresh_interval: int = 64,
    criticality_distill_min_weighted_events_per_layer: float = 256.0,
    criticality_distill_baseline_ema_decay: float = 0.99,
    criticality_distill_uniform_pressure: bool = False,
    criticality_distill_score_permute_before_topk: bool = False,
    criticality_distill_fixed_random_seats: bool = False,
```

After the existing config normalization, add validation:

```python
    if criticality_distill_enabled:
        mode_lc = str(lm_head_backward_mode).strip().lower()
        if mode_lc in _FUSED_LM_HEAD_MODES:
            raise ValueError(
                "criticality_distill_enabled=True requires a non-fused LM-head "
                "backward mode so the full logits exist for per-position entropy "
                f"computation. Got lm_head_backward_mode={lm_head_backward_mode!r}. "
                "Use lm_head_backward_mode='single' or another non-fused variant."
            )
```

Construct CD if enabled (after model construction, before training loop):

```python
    from contextlib import ExitStack
    criticality = None
    ssm_cores: list = []
    if criticality_distill_enabled:
        from chaoscontrol.optim.criticality import CriticalityDistillation
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
                    f"CD requires all captured cores share dim; got {c.dim} vs {dim}"
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
            baseline_ema_decay=float(criticality_distill_baseline_ema_decay),
            score_permute_before_topk=bool(criticality_distill_score_permute_before_topk),
            fixed_random_seats=bool(criticality_distill_fixed_random_seats),
        )
        criticality.sync_seat_mask_to_device(device)
```

In the inner training loop — pseudocode, adapt to the runner's actual variable names:

```python
        if criticality is not None and not criticality.fixed_random_seats:
            with ExitStack() as stack:
                _ = [stack.enter_context(c.capture_states()) for c in ssm_cores]
                # Existing forward that produces logits + ce
                hidden = model.encode(inputs)
                logits = model.lm_head(model.final_norm(hidden))
                ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction='none').view(B, T)
                ce_loss = ce.mean()
            states_per_layer = [c._captured_states for c in ssm_cores]
            # Pressure
            if bool(criticality_distill_uniform_pressure):
                pressure = torch.ones_like(ce)
            else:
                pressure = compute_ce_minus_entropy_pressure(logits, targets)
            # Ingest
            prepared = criticality.ingest_gpu(
                pressure=pressure,
                states_per_layer=states_per_layer,
                horizon_H=int(criticality_distill_horizon_H),
                event_frac=float(criticality_distill_event_frac),
            )
            prepared_cpu = {
                k: (v.to('cpu') if isinstance(v, torch.Tensor) else v)
                for k, v in prepared.items()
            }
            criticality.ingest_cpu_from_prepared(step=steps, prepared=prepared_cpu)
            # Seat refresh
            if steps % criticality.seat_refresh_interval == 0:
                criticality.allocate_seats(current_step=steps)
                criticality.sync_seat_mask_to_device(device)
        elif criticality is not None and criticality.fixed_random_seats:
            # No ingest, no allocate — run plain forward + CE.
            hidden = model.encode(inputs)
            logits = model.lm_head(model.final_norm(hidden))
            ce_loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        else:
            # Existing non-CD forward path, unchanged.
            ...

        # Loss compose
        if criticality is not None:
            log_a_per_layer = [c.log_a for c in ssm_cores]
            cd_loss = criticality.criticality_loss(log_a_per_layer)
            total_loss = ce_loss + cd_loss
        else:
            total_loss = ce_loss
```

**Implementation note:** the runner's existing structure uses `_run_train_step` to encapsulate the forward + loss. The cleanest integration is a new `_run_criticality_distillation_train_step` that replaces `_run_train_step` when CD is active, rather than adding branches inside the existing function. Match whichever pattern the rest of the file uses.

**Step 4: Run full file.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_exp23_fast_path.py -q
```

All pass, including the three new CD tests.

**Step 5: Commit:** `runner: criticality distillation wiring with non-fused LM head requirement`

### Task D.3 — Val-time per-bucket CE in the submission scorer

**Context.** Primary success metric is rare-bucket CE on Param-Golf val — NOT the training-time `FrequencyBucketBaseline._ema`. The training EMA is telemetry that we'll emit per-step (Task D.4 diagnostic), but the GATE is val-time per-bucket CE. This task extends the existing full-val scorer to emit per-bucket CE alongside the aggregate BPB it already produces.

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py` (locate the full-val scoring code)
- Modify: `tests/test_exp23_fast_path.py`
- Also: inspect where `FrequencyBucketBaseline` is persisted at eval time — the val scorer needs access to token frequencies.

**Step 1: Failing test.**

```python
def test_full_val_scorer_emits_per_bucket_ce():
    """The full-val scorer's result dict must include
    per_bucket_val_ce: List[float], one entry per frequency bucket.
    Rare-bucket CE is the first N entries (lowest log-freq buckets)."""
    mod = _load_runner_module()
    # Construct a minimal val scorer invocation with a stub model +
    # known token frequencies, then verify the output shape + dtype.
    # (Exact call shape depends on runner internals — adapt.)
    #
    # Minimal behavior:
    result = mod.run_full_val_score(
        model=_TinyTokenTrainModel(),
        val_tokens=torch.arange(64, dtype=torch.int16) % 6,
        token_frequencies=torch.tensor([100.0, 50.0, 20.0, 10.0, 5.0, 1.0]),
        num_buckets=4,
        device=torch.device("cpu"),
    )
    assert "per_bucket_val_ce" in result
    assert len(result["per_bucket_val_ce"]) == 4
    for v in result["per_bucket_val_ce"]:
        assert isinstance(v, float)
    assert "rare_bucket_val_ce" in result
    assert isinstance(result["rare_bucket_val_ce"], float)  # aggregate of lowest bucket(s)
```

**Step 2: Run — expect FAIL (function or key missing).**

**Step 3: Implement.**

Locate the full-val scorer in `runner_fast_path.py` (search for `full_val`, `val_score`, or similar). Extend its per-token-CE accumulator to bucket by target token frequency:

- Build a `FrequencyBucketBaseline`-style bucketizer from `token_frequencies + num_buckets`.
- For each val batch, compute per-position CE, then scatter-add into `bucket_sum[num_buckets]` and `bucket_count[num_buckets]`.
- At the end, `per_bucket_val_ce[b] = bucket_sum[b] / max(bucket_count[b], 1)`.
- `rare_bucket_val_ce` is the mean of the lowest `max(1, num_buckets // 4)` buckets (quarter-most-rare).

Emit both in the scorer's result dict. Propagate through `train_fast_for_budget`'s return so exp24 summaries can read it.

**Step 4: Run full file — all pass.**
**Step 5: Commit:** `runner: per-bucket val CE in full-val scorer (primary CD metric)`

### Task D.4 — Diagnostics: seat_churn, budget_occupancy, score_criticality_corr, event_rate

**Files:**
- Modify: `src/chaoscontrol/optim/criticality.py`
- Modify: `experiments/23_fast_path/runner_fast_path.py`
- Modify: `tests/test_criticality_distillation.py`

**Step 1: Failing tests.**

```python
def test_diagnostics_snapshot_shape():
    cd = CriticalityDistillation(
        num_layers=2, dim=8, trace_ttl_steps=4,
        criticality_budget_frac=0.25,
    )
    cd.seat_mask[0, :2] = True
    cd.seat_mask[1, 2:4] = True
    # Populate baseline and bank minimally.
    cd.add_step_evidence(layer=0, step=0, evidence=torch.ones(8), event_count=3.0)
    cd.add_step_evidence(layer=1, step=0, evidence=torch.ones(8), event_count=5.0)
    cd.baseline_initialized.fill_(True)

    log_a_per_layer = [torch.zeros(8), torch.zeros(8)]
    snap = cd.diagnostics_snapshot(log_a_per_layer, current_step=1)
    assert "seat_churn_per_layer" in snap  # fraction of seats changed since last snapshot
    assert "budget_occupancy_per_layer" in snap  # fraction of seat channels above critical_value threshold
    assert "score_criticality_corr_per_layer" in snap
    assert "event_rate_per_layer" in snap
    assert len(snap["seat_churn_per_layer"]) == 2
    assert len(snap["budget_occupancy_per_layer"]) == 2
    assert len(snap["score_criticality_corr_per_layer"]) == 2
    assert len(snap["event_rate_per_layer"]) == 2


def test_seat_churn_zero_when_seats_unchanged_between_snapshots():
    cd = CriticalityDistillation(num_layers=1, dim=6, trace_ttl_steps=4, criticality_budget_frac=0.5)
    cd.seat_mask[0, :3] = True
    log_a = [torch.zeros(6)]
    _ = cd.diagnostics_snapshot(log_a, current_step=0)
    # Same seat_mask on next snapshot.
    snap2 = cd.diagnostics_snapshot(log_a, current_step=1)
    assert snap2["seat_churn_per_layer"][0] == 0.0


def test_seat_churn_nonzero_when_seats_shift():
    cd = CriticalityDistillation(num_layers=1, dim=6, trace_ttl_steps=4, criticality_budget_frac=0.5)
    cd.seat_mask[0, :3] = True
    log_a = [torch.zeros(6)]
    _ = cd.diagnostics_snapshot(log_a, current_step=0)
    cd.seat_mask[0].zero_()
    cd.seat_mask[0, 3:] = True  # all seats moved
    snap2 = cd.diagnostics_snapshot(log_a, current_step=1)
    assert snap2["seat_churn_per_layer"][0] > 0.9
```

**Step 2-5: Implement + commit.**

Add to `CriticalityDistillation`:

```python
    def diagnostics_snapshot(
        self,
        log_a_per_layer: list,
        current_step: int,
    ) -> dict:
        """Emit a per-layer diagnostic snapshot.

        Must be called with a stable cadence (e.g. at each seat refresh)
        so seat_churn is interpretable.
        """
        if len(log_a_per_layer) != self.num_layers:
            raise ValueError(f"need {self.num_layers} log_a tensors")
        # seat_churn — needs a previous snapshot of seat_mask. Store
        # lazily.
        if not hasattr(self, "_last_seat_mask"):
            self._last_seat_mask = torch.zeros_like(self.seat_mask)
        churn = (self.seat_mask != self._last_seat_mask).float().mean(dim=-1)  # [L]
        self._last_seat_mask = self.seat_mask.clone()
        # budget_occupancy
        occupancy = []
        for layer, log_a in enumerate(log_a_per_layer):
            mask = self.seat_mask[layer]
            if not mask.any():
                occupancy.append(0.0)
                continue
            criticality = 1.0 - torch.sigmoid(log_a.to(torch.float32))
            above = criticality >= self.critical_value * 0.9  # 90% of target
            occupancy.append(float((mask & above.to(mask.device)).float().sum().item() / mask.float().sum().item()))
        # score_criticality_corr — Spearman-lite (rank correlation) per layer.
        scores = self.score(current_step=current_step)  # [L, D]
        corrs = []
        for layer, log_a in enumerate(log_a_per_layer):
            criticality = 1.0 - torch.sigmoid(log_a.detach().to(torch.float32)).cpu()
            s = scores[layer].cpu()
            # Pearson over rank — cheap proxy for Spearman.
            rs = torch.argsort(torch.argsort(s)).float()
            rc = torch.argsort(torch.argsort(criticality)).float()
            rs = rs - rs.mean()
            rc = rc - rc.mean()
            denom = (rs.norm() * rc.norm()).clamp_min(1e-12)
            corrs.append(float(((rs * rc).sum() / denom).item()))
        # event_rate — average bank_event_count over populated slots.
        event_rates = []
        for layer in range(self.num_layers):
            valid = self.bank_step[layer] >= 0
            if not valid.any():
                event_rates.append(0.0)
                continue
            event_rates.append(float(self.bank_event_count[layer][valid].mean().item()))
        return {
            "seat_churn_per_layer": churn.tolist(),
            "budget_occupancy_per_layer": occupancy,
            "score_criticality_corr_per_layer": corrs,
            "event_rate_per_layer": event_rates,
        }
```

In the runner, emit `diagnostics_snapshot` at every seat refresh into the result dict as `criticality_distillation_diagnostics: List[dict]` (one entry per refresh).

Commit: `criticality: diagnostics_snapshot (seat_churn, budget_occupancy, score_criticality_corr, event_rate)`

### Task D.5 — Config-threading test through the full matrix → runner path

**Context.** `train_fast_for_budget` is called from the orchestrator via a config dict that must thread CD kwargs correctly. This class of bug has bitten us repeatedly — silent kwarg drops. Test directly.

**Files:**
- Create: `tests/test_cd_config_threading.py`

**Step 1: Write the test (will fail if the dispatch is broken).**

```python
"""Config threading test for Criticality Distillation.

Builds a matrix entry via the exp24 builder, then exercises the
full dispatch path to confirm CD kwargs arrive at train_fast_for_budget
unmodified.
"""
import sys
sys.path.insert(0, "experiments/23_fast_path")
sys.path.insert(0, "experiments/24_training_time_bundle")


def test_matrix_entry_threads_all_cd_kwargs_to_runner(monkeypatch):
    from exp24 import build_criticality_distillation_first_smoke_matrix

    entries = build_criticality_distillation_first_smoke_matrix(
        speed_config={}, seed=1337, budget_seconds=600.0,
    )
    assert len(entries) == 8
    treatment = next(e for e in entries if "treatment" in e["name"])

    # The orchestrator threads entries through run_condition -> runner
    # kwargs. We monkeypatch train_fast_for_budget to capture the kwargs
    # it actually receives.
    import runner_fast_path  # type: ignore
    captured = {}

    def capture_kwargs(*args, **kwargs):
        captured.update(kwargs)
        # Return a minimal result dict so the orchestrator doesn't
        # choke on downstream processing.
        return {
            "steps": 0, "final_loss": 0.0, "initial_loss": 0.0,
            "aggregate_tokens_per_sec": 0.0, "peak_vram_mb": 0.0,
            "tokens_per_step": 0, "elapsed_s": 0.0, "world_size": 1,
            "loss_delta": 0.0, "loss_trajectory": [],
        }

    monkeypatch.setattr(runner_fast_path, "train_fast_for_budget", capture_kwargs)

    # Now invoke the dispatch path the orchestrator uses. If the project
    # has a helper like run_matrix_entries or run_condition, call that.
    # Fall back to a direct config -> kwargs manual flatten if no such
    # helper exists.
    from launch import run_matrix_entries  # adapt to actual module
    run_matrix_entries(
        entries=[treatment],
        output_dir="/tmp/test_cd_threading",
        world_size=1,
        dry_run=False,
    )

    required_cd_keys = {
        "criticality_distill_enabled",
        "criticality_distill_weight",
        "criticality_distill_budget_frac",
        "criticality_distill_critical_value",
        "criticality_distill_half_life_steps",
        "criticality_distill_ttl_steps",
        "criticality_distill_horizon_H",
        "criticality_distill_event_frac",
        "criticality_distill_seat_refresh_interval",
        "criticality_distill_min_weighted_events_per_layer",
        "criticality_distill_uniform_pressure",
        "criticality_distill_score_permute_before_topk",
        "criticality_distill_fixed_random_seats",
        "lm_head_backward_mode",
        "rare_bucket_ce_num_buckets",
    }
    missing = required_cd_keys - set(captured.keys())
    assert not missing, f"kwargs dropped in dispatch: {missing}"
    # Spot-check a few critical values survive intact.
    assert captured["criticality_distill_enabled"] is True
    assert captured["criticality_distill_weight"] == 1e-3
    assert captured["lm_head_backward_mode"] == "single"
```

**Step 2-5: Run — if the dispatch drops kwargs, this test catches it BEFORE first smoke burns pod time.** Fix the dispatch (usually in `launch.py` or wherever `run_condition`/`run_matrix_entries` flattens config dicts into kwargs) until the test passes. Commit: `runner: config-threading test confirms CD kwargs reach train_fast_for_budget`

---

## Stage E — Smoke matrix (on the locked fast/slow base)

### Task E.1 — `build_criticality_distillation_first_smoke_matrix` on fast_slow base

**Files:**
- Modify: `experiments/24_training_time_bundle/exp24.py`
- Modify: `experiments/24_training_time_bundle/run_exp24.py`
- Create: `tests/test_exp24_cd_smoke_matrix.py`

**Step 1: Failing tests.**

```python
import sys
sys.path.insert(0, "experiments/23_fast_path")
sys.path.insert(0, "experiments/24_training_time_bundle")
from exp24 import build_criticality_distillation_first_smoke_matrix


def test_matrix_emits_eight_cells_with_expected_names():
    entries = build_criticality_distillation_first_smoke_matrix(
        speed_config={}, seed=1337, budget_seconds=600.0,
    )
    assert len(entries) == 8
    expected_suffixes = [
        "treatment", "telemetry", "shuffled", "budget_only",
        "hl_short", "hl_long", "H_short", "H_long",
    ]
    for suffix in expected_suffixes:
        assert any(suffix in e["name"] for e in entries), (
            f"missing cell for {suffix}"
        )


def test_every_cell_rides_on_locked_fast_slow_base():
    """Every cell must inherit the control_fastslow_only_i64a025 lock
    exactly. Otherwise matrix results conflate CD effect with base
    differences."""
    entries = build_criticality_distillation_first_smoke_matrix(
        speed_config={}, seed=1337, budget_seconds=600.0,
    )
    for e in entries:
        assert e.get("fast_slow_enabled") is True, e["name"]
        assert e.get("fast_slow_interval") == 64, e["name"]
        assert e.get("fast_slow_alpha") == 0.25, e["name"]
        assert e.get("fast_slow_eval_copy") == "slow", e["name"]
        assert e.get("dreamworld_enabled") is False, e["name"]
        assert e.get("dreamworld_cache_interval") == 0, e["name"]
        assert e.get("dreamworld_interval") == 0, e["name"]
        assert e.get("dreamworld_weight") == 0.0, e["name"]
        assert e.get("dreamworld_replay_batch_size") == 0, e["name"]


def test_every_cell_uses_non_fused_lm_head_backward():
    entries = build_criticality_distillation_first_smoke_matrix(
        speed_config={}, seed=1337, budget_seconds=600.0,
    )
    for e in entries:
        mode = str(e.get("lm_head_backward_mode", "")).strip().lower()
        assert mode in {"single", "chunked"}, (
            f"{e['name']} uses {mode!r}, must be non-fused for CD entropy"
        )


def test_cell_flags_match_design():
    entries = build_criticality_distillation_first_smoke_matrix(
        speed_config={}, seed=1337, budget_seconds=600.0,
    )
    by_suffix = {}
    for e in entries:
        # Strip _s<seed> and cd_ prefix to get the arm suffix.
        name = e["name"]
        arm = name.rsplit("_s", 1)[0].split("cd_")[-1]
        by_suffix[arm] = e

    # treatment: surprise pressure, weight 1e-3, real score, hl 2048, H 64
    t = by_suffix["treatment"]
    assert t["criticality_distill_enabled"] is True
    assert t["criticality_distill_weight"] == 1e-3
    assert t["criticality_distill_uniform_pressure"] is False
    assert t["criticality_distill_score_permute_before_topk"] is False
    assert t["criticality_distill_fixed_random_seats"] is False
    assert t["criticality_distill_half_life_steps"] == 2048
    assert t["criticality_distill_horizon_H"] == 64

    # telemetry: weight 0
    assert by_suffix["telemetry"]["criticality_distill_weight"] == 0.0

    # shuffled: score_permute_before_topk True
    assert by_suffix["shuffled"]["criticality_distill_score_permute_before_topk"] is True

    # budget_only: fixed_random_seats True (design-faithful: no scoring, no bank)
    assert by_suffix["budget_only"]["criticality_distill_fixed_random_seats"] is True

    # Ablations
    assert by_suffix["hl_short"]["criticality_distill_half_life_steps"] == 256
    assert by_suffix["hl_long"]["criticality_distill_half_life_steps"] == 16384
    assert by_suffix["H_short"]["criticality_distill_horizon_H"] == 16
    assert by_suffix["H_long"]["criticality_distill_horizon_H"] == 256
```

**Step 2: Run — expect ImportError on builder.**

**Step 3: Implement.**

In `experiments/24_training_time_bundle/exp24.py`:

```python
def build_criticality_distillation_first_smoke_matrix(
    *,
    speed_config: dict[str, Any],
    seed: int = 1337,
    world_size: int = 1,
    budget_seconds: float = 600.0,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """First-read Criticality Distillation matrix — 8 cells, every cell
    riding on the locked control_fastslow_only_i64a025 operational stack.

    Measures whether treatment improves rare-bucket val CE relative to
    three falsifier controls (telemetry, shuffled-teacher, budget-only
    random-fixed-seats) and maps sensitivity to half-life and horizon_H.
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
    base["optimizer"] = "muon"

    # LOCKED fast/slow-only base — every cell inherits this exactly.
    base["fast_slow_enabled"] = True
    base["fast_slow_interval"] = 64
    base["fast_slow_alpha"] = 0.25
    base["fast_slow_eval_copy"] = "slow"
    base["dreamworld_enabled"] = False
    base["dreamworld_cache_interval"] = 0
    base["dreamworld_interval"] = 0
    base["dreamworld_weight"] = 0.0
    base["dreamworld_replay_batch_size"] = 0

    # Non-fused LM head required for CD entropy.
    base["lm_head_backward_mode"] = "single"

    # CD defaults per design doc (CPU-resident bank, rescaled long windows).
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
    base["criticality_distill_baseline_ema_decay"] = 0.99
    base["criticality_distill_uniform_pressure"] = False
    base["criticality_distill_score_permute_before_topk"] = False
    base["criticality_distill_fixed_random_seats"] = False

    # Val-time per-bucket CE readout.
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
        _cell("budget_only", {"criticality_distill_fixed_random_seats": True}),
        _cell("hl_short", {"criticality_distill_half_life_steps": 256,
                            "criticality_distill_ttl_steps": 2560}),
        _cell("hl_long", {"criticality_distill_half_life_steps": 16384,
                           "criticality_distill_ttl_steps": 163840}),
        _cell("H_short", {"criticality_distill_horizon_H": 16}),
        _cell("H_long", {"criticality_distill_horizon_H": 256}),
    ]
```

In `run_exp24.py`:
- Add `from exp24 import build_criticality_distillation_first_smoke_matrix` alongside other builder imports.
- Add dispatch branch inside `run_matrix`:
  ```python
  if matrix == "cd_first_smoke":
      entries: list[dict[str, Any]] = []
      for seed in seeds:
          entries.extend(
              build_criticality_distillation_first_smoke_matrix(
                  speed_config=speed_config, seed=seed,
                  world_size=world_size, budget_seconds=budget_seconds,
              )
          )
      return entries
  ```
- Add `"cd_first_smoke"` to argparse `choices`.
- Add to `_default_world_size_for_matrix`: returns 1.
- Add to `_default_budget_for_matrix`: returns 600.0.

**Step 4: Run tests.**

```
/opt/homebrew/bin/python3.11 -m pytest tests/test_exp24_cd_smoke_matrix.py -q
```
All pass.

Dry-run check:
```
/opt/homebrew/bin/python3.11 experiments/24_training_time_bundle/run_exp24.py --matrix cd_first_smoke --seeds 1337 --dry-run
```
Prints 8 entries.

**Step 5: Commit:** `exp24: register cd_first_smoke on locked fast_slow-only base (8 cells)`

---

## Final verification

```
/opt/homebrew/bin/python3.11 -m pytest \
  tests/test_criticality_distillation.py \
  tests/test_criticality_scoring.py \
  tests/test_ssm_state_capture.py \
  tests/test_runner_criticality_pressure.py \
  tests/test_exp24_cd_smoke_matrix.py \
  tests/test_cd_config_threading.py \
  tests/test_exp23_fast_path.py -q
```

Expected: all passing, test count increased over current 34 by ~25 new tests.

```
git log --oneline 0c62203..HEAD | wc -l
```

Expected: 10-13 commits, one per task.

```
/opt/homebrew/bin/python3.11 experiments/24_training_time_bundle/run_exp24.py --matrix cd_first_smoke --seeds 1337 --dry-run
```

Expected: 8 entries printed; every entry has `fast_slow_enabled=True`, `fast_slow_interval=64`, `fast_slow_alpha=0.25`, `lm_head_backward_mode="single"`.

---

## Out of scope for this plan

- **Fused-kernel entropy emission.** First smoke uses non-fused LM head. Follow-up plan: extend fused kernel to emit per-token entropy as a fourth output, at no algorithmic cost.
- **Async side-stream D2H.** First smoke uses synchronous non_blocking=False. Side-stream with event-ordered sync is a throughput refinement.
- **Multi-seed + 4×H100 confirmation** after first smoke shows treatment wins against all three falsifiers.
- **Precision-weighted surprise** `(CE - H[p]) · H[p]` — active-inference panel's suggested v2.
- **Matched-nearby baseline control** — current design uses EMA; matched-nearby is a v2 scoring ablation.
- **Per-frequency-bucket evidence banks.**
- **Paired / full SSM modes.** Currently `capture_states()` raises `NotImplementedError` for non-diag.
- **Staggered CD seat-refresh vs fast/slow interval.** Both are 64 in the first smoke by design (legible control loop). If `seat_churn` diagnostics show chattering, v2 staggers CD to 32 or 128.
