# Criticality Distillation Runner Wiring — Implementation Plan (v3)

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task. Use @superpowers:test-driven-development for every task (red-fail → implement → green-pass → commit). Apply @superpowers:verification-before-completion before claiming any step is done.

**Goal:** Wire `CriticalityDistillation` (terminal commit `c0318d8`) into the training step with a speed-first design: fused LM-head entropy emission, GPU reduces to tiny tensors, CPU runs a low-latency incremental-accumulator controller, pinned double-buffered async D2H. Emit an 8-cell `cd_first_smoke` matrix riding the locked `control_fastslow_only_i64a025` fast/slow base. Measure CD overhead vs a CD-off baseline and report (not hard-fail).

**Architecture (three tiers, each does one thing):**
- **H100:** fused CE + entropy kernel, event-mask top-k, future-energy reduction, log_a loss compose.
- **CPU:** decayed score accumulators (`score_num`, `score_den`, `event_mass`), non-event baseline EMA, falsifier policy, top-k seat allocation. Ring bank exists only for TTL correction + checkpoint debug.
- **H100 (again):** consumes the latest seat_mask for `criticality_loss` on every backward.

The "clever CPU" point: we do NOT rescan `[L, TTL, D]` every refresh. Each step applies one decay multiplier, one add. Top-k runs on the maintained accumulator. Ring bank is for TTL-exact correction (subtract evicted contribution at its current decay) and checkpoint serialization. Full-bank `score()` remains as a v2 diagnostic / consistency-check path.

**Tech Stack:** PyTorch 2.9, pytest, chaoscontrol runner_fast_path, fused `_lm_head_loss` kernel.

**Required reading before starting:**
- Mechanism design: `docs/plans/2026-04-24-criticality-distillation.md` (v3, `4b7d77c`)
- Runner-wiring design: `docs/plans/2026-04-24-criticality-distillation-runner-design.md` (`8293653`)
- Locked fast/slow base: `build_phase0_fastslow_only_control` in `experiments/24_training_time_bundle/exp24.py`
- Fused LM head kernels: `src/chaoscontrol/kernels/_lm_head_loss/` (C++ bindings, CUDA sources)
- Feedback memory: `feedback_regression_is_never_build_error.md`, `feedback_risks_not_implementation_challenges.md`, `feedback_estimate_calibration.md`.

---

## Conventions

- Every task is TDD: red, implement, green, commit.
- Commits: `component: imperative sentence`.
- macOS test runner: `/opt/homebrew/bin/python3.11 -m pytest ...`
- CPU-only tests use contrived tensors. GPU-only behavior: `pytest.importorskip('torch.cuda')` or `pytest.mark.skipif(not torch.cuda.is_available())`.

---

## Stage A — `compute_event_mask` conditional top-k (unchanged from v2)

### Task A.1 — Strictly-positive top-k; zero pressure → zero events

**Files:** `src/chaoscontrol/optim/criticality.py`, `tests/test_criticality_scoring.py`.

**Step 1:** Append two tests that verify uniform-zero → empty mask, and only strictly-positive positions are selected (exact test bodies in v2 are correct).

**Step 2:** Run — expect FAIL.

**Step 3:** Replace `compute_event_mask` body with:

```python
def compute_event_mask(pressure: torch.Tensor, event_frac: float) -> torch.Tensor:
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

**Step 4:** Run full scoring + distillation suites. All pass.

**Step 5:** `git commit -m "criticality: conditional top-k — event mask only marks strictly-positive pressure"`

---

## Stage B — CD falsifier flags

### Task B.1 — `score_permute_before_topk` → random k-of-D (shuffled teacher)

Exact as v2. In `allocate_seats`, when flag is True: `mask[torch.randperm(D, device=seat_mask.device)[:k]] = True`. Test pins `selected != peak_channels`.

Commit: `criticality: score_permute_before_topk flag — random k-of-D for shuffled-teacher falsifier`

### Task B.2 — `fixed_random_seats` flag (design-faithful budget-only)

Exact as v2. Constructor flag, seats bound once at init, `allocate_seats` is a no-op when True. Test pins init-bound + no-op on second call.

Commit: `criticality: fixed_random_seats flag for design-faithful budget-only falsifier`

---

## Stage C — CPU-as-controller with incremental accumulators

### Task C.1 — Register accumulator buffers

**Files:** `src/chaoscontrol/optim/criticality.py`, `tests/test_criticality_distillation.py`.

**Step 1:** Failing test.

```python
def test_accumulator_buffers_register_and_initialize_to_zero():
    cd = CriticalityDistillation(num_layers=3, dim=8, trace_ttl_steps=16)
    # score_num: running age-weighted sum of evidence contributions.
    assert cd.score_num.shape == (3, 8)
    assert torch.equal(cd.score_num, torch.zeros_like(cd.score_num))
    # score_den: running age-weighted count.
    assert cd.score_den.shape == (3,)
    assert torch.equal(cd.score_den, torch.zeros_like(cd.score_den))
    # event_mass: running age-weighted event count for the gate.
    assert cd.event_mass.shape == (3,)
    # last_decay_step: last step we applied decay to accumulators.
    assert cd.last_decay_step.item() == -1
    # All in state_dict (buffers registered).
    sd = cd.state_dict()
    for key in ("score_num", "score_den", "event_mass", "last_decay_step"):
        assert key in sd
```

**Step 2-3:** Add these buffer registrations in `__init__` alongside existing ones:

```python
        self.register_buffer(
            "score_num", torch.zeros(self.num_layers, self.dim, dtype=torch.float32)
        )
        self.register_buffer(
            "score_den", torch.zeros(self.num_layers, dtype=torch.float32)
        )
        self.register_buffer(
            "event_mass", torch.zeros(self.num_layers, dtype=torch.float32)
        )
        self.register_buffer(
            "last_decay_step", torch.tensor(-1, dtype=torch.int64)
        )
```

**Step 4-5:** Run tests, commit: `criticality: register accumulator buffers (score_num, score_den, event_mass, last_decay_step)`

### Task C.2 — `_step_decay_accumulators` method

**Files:** `src/chaoscontrol/optim/criticality.py`, `tests/test_criticality_distillation.py`.

**Step 1:** Failing test.

```python
def test_step_decay_applies_half_life_factor_to_all_accumulators():
    cd = CriticalityDistillation(
        num_layers=1, dim=4, trace_ttl_steps=8,
        trace_half_life_steps=2.0,  # decay factor per step = 2^(-1/2) ≈ 0.7071
    )
    cd.score_num.fill_(1.0)
    cd.score_den.fill_(4.0)
    cd.event_mass.fill_(10.0)
    cd.last_decay_step.fill_(0)
    # Advance to step 2. Total decay = 2^(-2/2) = 0.5.
    cd._step_decay_accumulators(current_step=2)
    assert torch.allclose(cd.score_num, torch.full_like(cd.score_num, 0.5))
    assert torch.allclose(cd.score_den, torch.tensor([2.0]))
    assert torch.allclose(cd.event_mass, torch.tensor([5.0]))
    assert cd.last_decay_step.item() == 2


def test_step_decay_is_idempotent_when_called_with_same_step():
    cd = CriticalityDistillation(num_layers=1, dim=2, trace_ttl_steps=4, trace_half_life_steps=2.0)
    cd.score_num.fill_(1.0)
    cd.last_decay_step.fill_(5)
    cd._step_decay_accumulators(current_step=5)
    assert torch.allclose(cd.score_num, torch.full_like(cd.score_num, 1.0))
```

**Step 2:** Run — FAIL on method not found.

**Step 3:** Add method to `CriticalityDistillation`:

```python
    @torch.no_grad()
    def _step_decay_accumulators(self, current_step: int) -> None:
        """Apply age decay to running accumulators between
        last_decay_step and current_step. Idempotent when called with
        the same step."""
        if int(current_step) <= int(self.last_decay_step.item()):
            return
        dt = int(current_step) - int(self.last_decay_step.item())
        if int(self.last_decay_step.item()) < 0:
            # First time — accumulators are all zero, no decay needed.
            self.last_decay_step.fill_(int(current_step))
            return
        factor = 2.0 ** (-float(dt) / self.trace_half_life_steps)
        self.score_num.mul_(factor)
        self.score_den.mul_(factor)
        self.event_mass.mul_(factor)
        self.last_decay_step.fill_(int(current_step))
```

**Step 4-5:** Run, commit: `criticality: _step_decay_accumulators applies half-life decay between ingests`

### Task C.3 — `_add_contribution` method

**Files:** same as C.2.

**Step 1:** Failing test.

```python
def test_add_contribution_updates_numerator_denominator_and_event_mass():
    cd = CriticalityDistillation(
        num_layers=2, dim=3, trace_ttl_steps=8,
        trace_half_life_steps=4.0,
    )
    # Pre-step: call _step_decay (no-op on zero accumulators; just syncs last_decay_step).
    cd._step_decay_accumulators(current_step=0)
    cd._add_contribution(
        layer=0,
        evidence=torch.tensor([1.0, 2.0, 3.0]),
        event_count=5.0,
    )
    # Numerator: evidence * event_count = [5, 10, 15]
    assert torch.allclose(cd.score_num[0], torch.tensor([5.0, 10.0, 15.0]))
    # Denominator: event_count = 5
    assert cd.score_den[0].item() == 5.0
    # Event mass: event_count
    assert cd.event_mass[0].item() == 5.0
    # Layer 1 untouched.
    assert torch.equal(cd.score_num[1], torch.zeros(3))
```

**Step 2:** Run — FAIL.

**Step 3:** Implement:

```python
    @torch.no_grad()
    def _add_contribution(
        self,
        *,
        layer: int,
        evidence: torch.Tensor,
        event_count: float,
    ) -> None:
        """Incremental additive update of accumulators. Weight for this
        step's contribution is always 1.0 (the age is zero). Evidence
        tensor must match `self.dim`."""
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"layer={layer}")
        if evidence.shape != (self.dim,):
            raise ValueError(
                f"evidence must have shape ({self.dim},); got {tuple(evidence.shape)}"
            )
        ec = float(event_count)
        ev = evidence.to(dtype=self.score_num.dtype, device=self.score_num.device)
        self.score_num[layer].add_(ev, alpha=ec)
        self.score_den[layer].add_(ec)
        self.event_mass[layer].add_(ec)
```

**Step 4-5:** Commit: `criticality: _add_contribution for incremental accumulator update`

### Task C.4 — `_subtract_expired_contribution` method (TTL correction)

**Files:** same.

**Step 1:** Failing test.

```python
def test_subtract_expired_removes_contribution_at_its_current_decay_weight():
    cd = CriticalityDistillation(
        num_layers=1, dim=2, trace_ttl_steps=4,
        trace_half_life_steps=2.0,
    )
    # Simulate: entry added at step=0 with evidence=[1, 2] and event_count=3.
    # Current step=4. Entry's current age = 4 -> decay weight 2^(-4/2) = 0.25.
    # Before subtraction:
    cd.score_num[0] = torch.tensor([0.25 * 1 * 3, 0.25 * 2 * 3])  # [0.75, 1.5]
    cd.score_den[0] = 0.25 * 3  # 0.75
    cd.event_mass[0] = 0.25 * 3  # 0.75
    # Subtract as if that slot is now being overwritten.
    cd._subtract_expired_contribution(
        layer=0,
        evicted_step=0,
        current_step=4,
        evicted_evidence=torch.tensor([1.0, 2.0]),
        evicted_event_count=3.0,
    )
    assert torch.allclose(cd.score_num[0], torch.zeros(2), atol=1e-6)
    assert abs(cd.score_den[0].item()) < 1e-6
    assert abs(cd.event_mass[0].item()) < 1e-6
```

**Step 2:** Run — FAIL.

**Step 3:** Implement:

```python
    @torch.no_grad()
    def _subtract_expired_contribution(
        self,
        *,
        layer: int,
        evicted_step: int,
        current_step: int,
        evicted_evidence: torch.Tensor,
        evicted_event_count: float,
    ) -> None:
        """Remove an expired/evicted slot's contribution at its current
        decay weight. Call this BEFORE writing the new entry into a
        slot that was occupied."""
        age = max(0, int(current_step) - int(evicted_step))
        factor = 2.0 ** (-float(age) / self.trace_half_life_steps)
        ec = float(evicted_event_count) * factor
        ev = evicted_evidence.to(
            dtype=self.score_num.dtype, device=self.score_num.device
        )
        self.score_num[layer].sub_(ev, alpha=ec)
        self.score_den[layer].sub_(ec / factor if ec != 0 else 0.0)  # raw ec without factor... see note
        # Correction: evicted slot's unweighted event_count was
        # `evicted_event_count`. Its current contribution to score_den
        # is `evicted_event_count * factor`. Remove that.
        # (Refactor the above to be clearer.)
```

Actually, re-think the math. The contribution the slot ADDED at time `evicted_step` was:
- score_num[layer] += evicted_evidence * evicted_event_count * (weight at time of add = 1.0)
- score_den[layer] += evicted_event_count * 1.0
- event_mass[layer] += evicted_event_count * 1.0

Between then and `current_step`, all accumulators have been multiplied by `2^(-(current_step - evicted_step) / half_life) = factor`. So the slot's **remaining** contribution is:
- score_num: `evicted_evidence * evicted_event_count * factor`
- score_den: `evicted_event_count * factor`
- event_mass: `evicted_event_count * factor`

Subtract these. Cleaner implementation:

```python
    @torch.no_grad()
    def _subtract_expired_contribution(
        self,
        *,
        layer: int,
        evicted_step: int,
        current_step: int,
        evicted_evidence: torch.Tensor,
        evicted_event_count: float,
    ) -> None:
        age = max(0, int(current_step) - int(evicted_step))
        factor = 2.0 ** (-float(age) / self.trace_half_life_steps)
        remaining_ec = float(evicted_event_count) * factor
        ev = evicted_evidence.to(
            dtype=self.score_num.dtype, device=self.score_num.device
        )
        self.score_num[layer].sub_(ev, alpha=remaining_ec)
        self.score_den[layer].sub_(remaining_ec)
        self.event_mass[layer].sub_(remaining_ec)
```

**Step 4-5:** Run test, commit: `criticality: _subtract_expired_contribution for TTL-exact accumulator correction`

### Task C.5 — `score_from_accumulators` replaces hot-path score

**Files:** same.

**Step 1:** Failing test.

```python
def test_score_from_accumulators_matches_full_bank_score_after_ingest_sequence():
    """The incremental accumulator must produce the same score as the
    full-bank scan after any sequence of ingests. This is the
    consistency pin between the two implementations."""
    cd = CriticalityDistillation(
        num_layers=1, dim=3, trace_ttl_steps=8,
        trace_half_life_steps=4.0,
    )
    ingests = [
        (0, torch.tensor([1.0, 0.0, 0.0]), 2.0),
        (1, torch.tensor([0.0, 1.0, 0.0]), 3.0),
        (3, torch.tensor([0.0, 0.0, 2.0]), 1.0),
    ]
    for step, evidence, ec in ingests:
        cd._step_decay_accumulators(current_step=step)
        cd._add_contribution(layer=0, evidence=evidence, event_count=ec)
        # Also write to the ring bank for the full-scan comparison.
        cd.add_step_evidence(layer=0, step=step, evidence=evidence, event_count=ec)
    # Advance decay to score time.
    current_step = 5
    cd._step_decay_accumulators(current_step=current_step)
    accumulator_score = cd.score_from_accumulators()
    full_scan_score = cd.score(current_step=current_step)
    # Tolerance accounts for the fact that the accumulator's
    # denominator treats `event_count` as weight while the full-scan
    # score uses equal weight per entry. The two are semantically
    # different if event_counts differ — so we compare the SIGN and
    # NEARLY-EQUAL PER-CHANNEL RATIOS, not raw magnitudes.
    # [Actually — align the two definitions so this matches. See note.]
    # For now assert the peak channel matches.
    assert accumulator_score[0].argmax().item() == full_scan_score[0].argmax().item()
```

**Implementation note for the subagent:** the full-scan `score()` uses `sum(age_weight * evidence) / sum(age_weight)` — equal weight per entry regardless of event_count. The accumulator weights entries by `event_count`. These are different semantics.

**Decision:** align `score()` (the full-scan) with the accumulator semantics: weight each entry by its `event_count` when computing the age-weighted mean. Update both tests that depend on `score()` to the new convention.

The math: `score_accumulator[c] = score_num[c] / score_den` where both have been age-decayed together. Equivalent to `(Σ_entries w_e · count_e · evidence_e[c]) / (Σ_entries w_e · count_e)`. This is a count-weighted mean, which is what we want — a step with 100 events contributes more than a step with 1 event.

**Step 2:** Run — FAIL.

**Step 3:** Implement two things:

(a) Add to `CriticalityDistillation`:

```python
    def score_from_accumulators(self) -> torch.Tensor:
        """Age-weighted count-weighted mean of evidence, read from
        running accumulators. Hot-path scorer."""
        denom = self.score_den.clamp_min(1e-12).unsqueeze(-1)
        raw = self.score_num / denom  # [L, D]
        # Layers with zero event_mass contribute zero (not NaN).
        valid = self.event_mass > 0
        return torch.where(valid.unsqueeze(-1), raw, torch.zeros_like(raw))
```

(b) Update `score()` (the full-scan) to weight by `bank_event_count` so it matches the accumulator semantics. Any existing tests that break under this change are testing the old unweighted semantics — update their expected values using the new weighted math.

**Step 4:** Run. Expect some existing tests to break — audit each, update expected values or explain why they should be unchanged.

**Step 5:** Commit: `criticality: score_from_accumulators as hot-path, align score() semantics`

### Task C.6 — `allocate_seats_from_accumulators` + gate on `event_mass`

**Files:** same.

**Step 1:** Failing test.

```python
def test_allocate_seats_from_accumulators_picks_topk_by_accumulator_score():
    cd = CriticalityDistillation(
        num_layers=1, dim=8, trace_ttl_steps=8,
        trace_half_life_steps=100.0,
        criticality_budget_frac=0.25,  # k=2
        min_weighted_events_per_layer=0.5,
    )
    cd._step_decay_accumulators(current_step=0)
    cd._add_contribution(layer=0, evidence=torch.tensor([0.1,0.2,5.0,0.0,0.0,9.0,0.0,0.0]), event_count=1.0)
    cd.allocate_seats_from_accumulators(current_step=1)
    assert cd.seat_mask[0].sum().item() == 2
    assert cd.seat_mask[0, 5].item() is True  # peak 1
    assert cd.seat_mask[0, 2].item() is True  # peak 2


def test_allocate_seats_from_accumulators_respects_event_mass_gate():
    cd = CriticalityDistillation(
        num_layers=1, dim=8, trace_ttl_steps=8,
        trace_half_life_steps=100.0,
        criticality_budget_frac=0.25,
        min_weighted_events_per_layer=100.0,  # unreachable
    )
    cd._step_decay_accumulators(current_step=0)
    cd._add_contribution(layer=0, evidence=torch.ones(8), event_count=1.0)
    cd.allocate_seats_from_accumulators(current_step=1)
    assert not cd.seat_mask[0].any()
```

**Step 2:** Run — FAIL.

**Step 3:** Implement:

```python
    @torch.no_grad()
    def allocate_seats_from_accumulators(self, *, current_step: int) -> None:
        """Hot-path seat allocator. Respects fixed_random_seats flag."""
        if self.fixed_random_seats:
            return
        self._step_decay_accumulators(current_step=current_step)
        k = max(1, int(round(self.dim * self.criticality_budget_frac)))
        scores = self.score_from_accumulators()
        for layer in range(self.num_layers):
            if self.event_mass[layer].item() < self.min_weighted_events_per_layer:
                self.seat_mask[layer].fill_(False)
                continue
            mask = torch.zeros(self.dim, dtype=torch.bool, device=self.seat_mask.device)
            if self.score_permute_before_topk:
                perm = torch.randperm(self.dim, device=self.seat_mask.device)
                mask[perm[:k]] = True
            else:
                topk = torch.topk(scores[layer], k=k, largest=True)
                mask[topk.indices] = True
            self.seat_mask[layer] = mask
```

**Step 4-5:** Run, commit: `criticality: allocate_seats_from_accumulators (hot-path, event_mass-gated)`

### Task C.7 — Update `ingest_cpu_from_prepared` to drive accumulators

**Files:** same.

**Step 1:** Failing test.

```python
def test_ingest_cpu_from_prepared_advances_accumulators():
    cd = CriticalityDistillation(num_layers=1, dim=3, trace_ttl_steps=8, trace_half_life_steps=4.0)
    prepared = {
        "event_mask": torch.tensor([[True, False, True]]),
        "aggregated_excess_per_layer": torch.tensor([[1.0, 2.0, 3.0]]),
        "non_event_mean_future_energy_per_layer": torch.tensor([[0.5, 0.5, 0.5]]),
        "event_count_per_layer": torch.tensor([2.0]),
    }
    cd.ingest_cpu_from_prepared(step=0, prepared=prepared)
    # Accumulators updated: score_num = evidence * count = [2, 4, 6];
    # score_den = 2; event_mass = 2.
    assert torch.allclose(cd.score_num[0], torch.tensor([2.0, 4.0, 6.0]))
    assert cd.score_den[0].item() == 2.0
    assert cd.event_mass[0].item() == 2.0
    # Ring bank also written for TTL/checkpoint state.
    slot = (cd.bank_step[0] == 0).nonzero(as_tuple=True)[0].item()
    assert torch.allclose(cd.bank_evidence[0, slot], torch.tensor([1.0, 2.0, 3.0]))
```

**Step 2:** Run — FAIL (existing method doesn't update accumulators).

**Step 3:** Update `ingest_cpu_from_prepared` to call `_step_decay_accumulators + _add_contribution` alongside the existing ring-bank write + baseline EMA update. When the ring evicts a slot, first call `_subtract_expired_contribution` with the evicted slot's data.

Updated flow:

```python
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
        # Advance accumulator decay to this step.
        self._step_decay_accumulators(current_step=step)
        for layer in range(self.num_layers):
            if had_non_events:
                obs = nonevt[layer]
                if not bool(self.baseline_initialized[layer].item()):
                    self.baseline_future_energy[layer].copy_(obs)
                    self.baseline_initialized[layer] = True
                else:
                    self.baseline_future_energy[layer].mul_(decay).add_(obs, alpha=(1.0 - decay))
            cnt = float(counts[layer].item())
            if cnt <= 0:
                continue
            # Incremental accumulator update.
            self._add_contribution(
                layer=layer,
                evidence=agg[layer],
                event_count=cnt,
            )
            # Ring-bank write with TTL-exact correction.
            self._write_ring_slot(
                layer=layer,
                step=step,
                evidence=agg[layer],
                event_count=cnt,
                current_step=step,
            )
```

Add helper `_write_ring_slot` that subtracts an evicted slot's contribution before overwriting:

```python
    @torch.no_grad()
    def _write_ring_slot(
        self, *, layer: int, step: int, evidence: torch.Tensor,
        event_count: float, current_step: int,
    ) -> None:
        slots = self.bank_step[layer]
        empty = (slots == -1).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            slot = int(empty[0].item())
        else:
            # Evicting oldest — first subtract its remaining contribution.
            slot = int(slots.argmin().item())
            evicted_step = int(slots[slot].item())
            evicted_ev = self.bank_evidence[layer, slot].clone()
            evicted_cnt = float(self.bank_event_count[layer, slot].item())
            self._subtract_expired_contribution(
                layer=layer,
                evicted_step=evicted_step,
                current_step=current_step,
                evicted_evidence=evicted_ev,
                evicted_event_count=evicted_cnt,
            )
        self.bank_evidence[layer, slot] = evidence.to(
            dtype=self.bank_evidence.dtype, device=self.bank_evidence.device,
        )
        self.bank_step[layer, slot] = int(step)
        self.bank_event_count[layer, slot] = float(event_count)
```

**Step 4:** Run full suite. The existing `test_add_step_evidence_writes_into_correct_slot_and_ttl_wraps` still passes (ring behavior unchanged). Accumulator test passes.

**Step 5:** Commit: `criticality: ingest drives incremental accumulators, ring handles TTL correction`

### Task C.8 — Equivalence test: accumulator score vs full-bank score

**Files:** `tests/test_criticality_distillation.py`.

**Step 1:** A sharper equivalence test than C.5's shape-only check.

```python
def test_accumulator_score_equals_full_bank_score_within_fp32_tolerance():
    """Incremental accumulator and full-bank scan must agree exactly
    (modulo fp32 rounding) after any ingest sequence and any step
    advance."""
    cd = CriticalityDistillation(
        num_layers=2, dim=5, trace_ttl_steps=10,
        trace_half_life_steps=3.0,
    )
    torch.manual_seed(7)
    steps = [0, 1, 3, 5, 8, 13, 21]
    for step in steps:
        cd._step_decay_accumulators(current_step=step)
        for layer in range(2):
            evidence = torch.randn(5).abs() + 0.1
            cnt = float(torch.randint(1, 10, (1,)).item())
            cd._add_contribution(layer=layer, evidence=evidence, event_count=cnt)
            cd._write_ring_slot(
                layer=layer, step=step, evidence=evidence,
                event_count=cnt, current_step=step,
            )
    current_step = 30
    cd._step_decay_accumulators(current_step=current_step)
    acc = cd.score_from_accumulators()
    full = cd.score(current_step=current_step)
    assert torch.allclose(acc, full, atol=1e-4, rtol=1e-4), (
        f"accumulator diverged from full scan: acc={acc} full={full}"
    )
```

**Step 2-3:** Run — the test may initially fail due to subtle off-by-one in TTL handling OR because `score()` semantics aren't aligned. Fix whichever side is wrong until they match within fp32 tolerance. If they diverge, this is the regression pin that keeps the two impls in agreement.

**Step 4-5:** Run, commit: `criticality: pin accumulator / full-scan equivalence to fp32 tolerance`

---

## Stage D — Fused LM head entropy emission (time-boxed, measured)

**This is the one stage in this plan where the individual implementation step is genuinely >5 minutes.** Time it. Ken and I agreed ~30 min is the realistic estimate. Report actual elapsed in the commit message.

**Outline:**

The existing fused forward emits `(loss, lse, per_token_ce)` (or `(loss, lse, per_token_ce, logits_cache)` in the cached variant). Add a fourth scalar accumulator per row in the streaming pass:

```
sum_exp_logit[row] = Σ_v exp(logit_v - lse[row]) · logit_v
```

Then `entropy[row] = lse[row] - sum_exp_logit[row]` (since probs sum to 1 by the lse normalization).

We extend **only one mode** to start (the current default: `fused_streaming_cached`). Other modes stay as-is and are unsupported for CD.

### Task D.1 — CUDA kernel: accumulate `sum_exp_logit` in the streaming pass

**Files:** `src/chaoscontrol/kernels/_lm_head_loss/src/linear_ce.cu`

**Start timer. Target: 30 min for D.1 through D.4.**

**Step 1:** Find the streaming cached forward kernel (search for `linear_ce_streaming_cached_forward` or similar). Identify the row-level accumulation loop.

**Step 2:** Add a new output buffer `per_token_entropy` emitted alongside `per_token_ce`. In the tile loop, after `max_logit` and `lse` are computed, add a second accumulation:

```cpp
// After lse is known for this row:
float sum_exp_logit = 0.0f;
for (int v = 0; v < vocab; ++v) {
    float l = logits[row * vocab + v];
    sum_exp_logit += expf(l - lse_row) * l;
}
per_token_entropy[row] = lse_row - sum_exp_logit;
```

In the actual streaming pass this is not a second pass — fold it into the existing tile loop that computes lse (or the loss-compute pass that reads logits again for CE).

**Step 3:** Unit test at the C++/CUDA level (if the project has a way to call kernels from pytest) OR skip to D.4's Python-level test.

**Step 4:** Build + load the extension:

```bash
MAX_JOBS=6 TORCH_CUDA_ARCH_LIST="9.0" /opt/homebrew/bin/python3.11 -m pip install -e . --no-build-isolation 2>&1 | tail -5
```

(Note: this builds CUDA code. On macOS without CUDA, the build will skip or error. **This task and D.2 / D.3 should be done by a subagent on the pod, not macOS.** The plan splits execution: write the code changes on macOS, verify against a Python reference in D.4, actual kernel execution + numerical test runs on the pod.)

**Step 5:** Commit: `lm_head_loss: emit per_token_entropy in streaming_cached forward kernel`

### Task D.2 — C++ binding: expose entropy output in pybind signature

**Files:** `src/chaoscontrol/kernels/_lm_head_loss/src/rms_norm_binding.cpp`

**Step 1-3:** Extend the return tuple of `linear_ce_streaming_cached_forward` (or define a new entrypoint `linear_ce_streaming_cached_forward_with_entropy`) to include `per_token_entropy` of shape `[B*T]` fp32.

**Step 4-5:** Commit: `lm_head_loss: bind per_token_entropy in streaming_cached forward`

### Task D.3 — Python autograd.Function returns entropy

**Files:** `src/chaoscontrol/kernels/_lm_head_loss/__init__.py`

**Step 1:** Add a new public API:

```python
def fused_lm_head_forward_with_ce_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    tile_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (loss, lse, per_token_ce, per_token_entropy). Entropy is
    detached (non-differentiable); the other three tensors behave as
    before."""
    ...
```

This wraps the existing `fused_lm_head_forward` or the cached variant, with an additional per-token-entropy output.

**Step 4-5:** Commit: `lm_head_loss: fused_lm_head_forward_with_ce_entropy API`

### Task D.4 — Python test: kernel entropy matches softmax reference

**Files:** `tests/test_lm_head_loss_kernel.py` (or new file `tests/test_lm_head_loss_entropy.py`).

**This is a pod-only GPU test.** Mark skipped if no CUDA.

**Step 1:** Failing test.

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_entropy_matches_softmax_reference_within_fp32_tolerance():
    from chaoscontrol.kernels._lm_head_loss import fused_lm_head_forward_with_ce_entropy
    import torch.nn.functional as F
    torch.manual_seed(0)
    B, T, D, V = 2, 8, 32, 128
    x = torch.randn(B, T, D, device='cuda', dtype=torch.float32)
    weight = torch.randn(V, D, device='cuda', dtype=torch.float32)
    target = torch.randint(0, V, (B, T), device='cuda', dtype=torch.int64)
    loss, lse, per_token_ce, per_token_entropy = fused_lm_head_forward_with_ce_entropy(
        x.reshape(-1, D), weight, target.reshape(-1),
        tile_size=8192,
    )
    # Reference via softmax.
    logits = x.reshape(-1, D) @ weight.T  # [B*T, V]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ref_entropy = -(probs * log_probs).sum(dim=-1)
    assert torch.allclose(per_token_entropy, ref_entropy, atol=1e-4, rtol=1e-4), (
        f"max abs diff = {(per_token_entropy - ref_entropy).abs().max().item()}"
    )
```

**Step 2-5:** Pod-only build + run. Commit: `lm_head_loss: test fused per-token entropy matches softmax reference`

**Stop timer. Report elapsed in the final commit message of this stage** (e.g. `"actual elapsed: 37 min"` in `git notes` or the commit body).

---

## Stage E — Runner wiring with fused entropy + async D2H

### Task E.1 — Pressure helper consumes fused (ce, entropy) pair

**Files:** `experiments/_23_fast_path_runner_helpers.py` (new), `experiments/23_fast_path/runner_fast_path.py`, `tests/test_runner_criticality_pressure.py`.

**Step 1:** Failing test (same structure as v2 Task D.1 but consumes pre-computed per-token CE and entropy tensors, not logits):

```python
from experiments._23_fast_path_runner_helpers import compute_ce_minus_entropy_pressure_from_fused


def test_pressure_from_fused_pair_nonnegative_and_ranks_by_innovation():
    ce = torch.tensor([[3.0, 0.5, 2.0]])
    entropy = torch.tensor([[0.1, 2.0, 2.5]])
    # innovation = ce - entropy = [2.9, -1.5 -> 0, -0.5 -> 0]
    pressure = compute_ce_minus_entropy_pressure_from_fused(ce, entropy)
    assert (pressure >= 0).all()
    assert pressure.argmax().item() == 0  # highest innovation at index 0
```

**Step 2-5:** Implement as a one-liner. Commit: `runner: compute_ce_minus_entropy_pressure_from_fused helper`

### Task E.2 — CD construction + entropy-capable mode validation

**Files:** `experiments/23_fast_path/runner_fast_path.py`, `tests/test_exp23_fast_path.py`.

**Step 1:** Failing tests for:
- `criticality_distill_enabled=True` with an entropy-capable mode (`fused_streaming_cached_with_entropy`) succeeds.
- `criticality_distill_enabled=True` with a non-entropy-capable mode raises.
- `criticality_distill_enabled=False` works with any mode (existing behavior).

**Step 2-5:** Implement kwarg plumbing + validation. Define constant `_ENTROPY_CAPABLE_LM_HEAD_MODES = {"fused_streaming_cached_with_entropy"}` (initially only one mode). Commit: `runner: criticality distillation requires entropy-capable LM-head mode`

### Task E.3 — Training-step wiring: capture, pressure, pinned async D2H, accumulator update

**Files:** `experiments/23_fast_path/runner_fast_path.py`, `tests/test_exp23_fast_path.py`.

**Step 1:** Failing integration-style test that runs 4 steps through the CD pipeline and verifies:
- `ingest_gpu` called 4× (per-step ingest).
- Pinned host buffers used for D2H (verify by asserting `tensor.is_pinned()` in a spy).
- Accumulator state advances as steps go by.
- Seat refresh fires every `seat_refresh_interval` steps.
- `criticality_loss` is added to `total_loss` when CD has seats.

**Step 2-5:** Implement the inner training step.

Key structural change over v2's wiring: use pinned double-buffered host tensors and async D2H.

```python
# At CD construction time, allocate two pinned host buffers per prepared-key.
def _alloc_pinned_evidence_buffers(num_layers: int, dim: int) -> dict:
    return {
        "A": {
            "aggregated_excess_per_layer": torch.empty(num_layers, dim, pin_memory=True, dtype=torch.float32),
            "non_event_mean_future_energy_per_layer": torch.empty(num_layers, dim, pin_memory=True, dtype=torch.float32),
            "event_count_per_layer": torch.empty(num_layers, pin_memory=True, dtype=torch.float32),
            "event_mask": torch.empty((1,), pin_memory=True, dtype=torch.bool),  # placeholder — resized per batch
        },
        "B": {...},  # mirror
    }

# Per-step, inside the training loop (with CD active):
parity = step % 2  # ping-pong A/B
host_slot = pinned_buffers["A" if parity == 0 else "B"]

# Forward + CE + entropy via the fused entropy-capable kernel.
with ExitStack() as stack:
    _ = [stack.enter_context(c.capture_states()) for c in ssm_cores]
    hidden = model.encode(inputs)
    # Fused LM head with entropy:
    loss, lse, per_token_ce, per_token_entropy = fused_lm_head_forward_with_ce_entropy(...)
states_per_layer = [c._captured_states for c in ssm_cores]

# Pressure.
if criticality_distill_uniform_pressure:
    pressure = torch.ones_like(per_token_ce).reshape(B, T)
else:
    pressure = compute_ce_minus_entropy_pressure_from_fused(
        per_token_ce.reshape(B, T), per_token_entropy.reshape(B, T),
    )

# GPU reduction.
prepared_gpu = criticality.ingest_gpu(
    pressure=pressure,
    states_per_layer=states_per_layer,
    horizon_H=int(criticality_distill_horizon_H),
    event_frac=float(criticality_distill_event_frac),
)
# Async D2H into pinned buffer.
host_slot["aggregated_excess_per_layer"].copy_(prepared_gpu["aggregated_excess_per_layer"], non_blocking=True)
host_slot["non_event_mean_future_energy_per_layer"].copy_(prepared_gpu["non_event_mean_future_energy_per_layer"], non_blocking=True)
host_slot["event_count_per_layer"].copy_(prepared_gpu["event_count_per_layer"], non_blocking=True)
# event_mask resized as needed.
ev_mask_cpu = prepared_gpu["event_mask"].to("cpu", non_blocking=True)

# Record event so the CPU consumer can wait.
copy_done = torch.cuda.Event()
copy_done.record()

# ... training backward + optimizer step happens on main stream ...
loss.backward()
optimizer.step()

# After the backward, CPU accumulator update (D2H will be done by then).
copy_done.synchronize()
prepared_cpu = {
    "aggregated_excess_per_layer": host_slot["aggregated_excess_per_layer"],
    "non_event_mean_future_energy_per_layer": host_slot["non_event_mean_future_energy_per_layer"],
    "event_count_per_layer": host_slot["event_count_per_layer"],
    "event_mask": ev_mask_cpu,
}
criticality.ingest_cpu_from_prepared(step=step, prepared=prepared_cpu)

# Seat refresh every N steps.
if step % criticality.seat_refresh_interval == 0 and step > 0:
    criticality.allocate_seats_from_accumulators(current_step=step)
    criticality.sync_seat_mask_to_device(device)
```

Commit: `runner: CD wiring with fused entropy, pinned async D2H, accumulator update`

### Task E.4 — Diagnostics snapshot + val-time per-bucket CE

**Files:** `src/chaoscontrol/optim/criticality.py`, `experiments/23_fast_path/runner_fast_path.py`, `tests/test_criticality_distillation.py`, `tests/test_exp23_fast_path.py`.

Same as v2 Tasks D.3 + D.4. Commit: `runner: diagnostics_snapshot + per-bucket val CE`

### Task E.5 — Config-threading test

Same as v2 Task D.5, with the added required keys `lm_head_backward_mode` = entropy-capable mode name. Commit: `runner: config-threading test confirms CD kwargs reach train_fast_for_budget`

---

## Stage F — Smoke matrix on fast/slow base

### Task F.1 — `build_criticality_distillation_first_smoke_matrix`

Same as v2 Stage E with two differences:
1. `base["lm_head_backward_mode"] = "fused_streaming_cached_with_entropy"` (entropy-capable fused mode, not `"single"`).
2. Matrix test asserts every cell has this mode value.

Commit: `exp24: register cd_first_smoke on locked fast_slow base with fused-entropy LM head`

---

## Stage G — Topology preflight + overhead measurement (measure-only, no hard fail)

### Task G.1 — Topology snapshot emitted at training start

**Files:** `experiments/23_fast_path/runner_fast_path.py` (one new helper), `tests/test_exp23_fast_path.py`.

**Step 1:** Failing test — runner emits `topology_snapshot` in the result dict when enabled:

```python
def test_runner_emits_topology_snapshot_when_requested():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = mod.train_fast_for_budget(
        model, ..., emit_topology_snapshot=True, max_steps=1,
    )
    snap = result.get("topology_snapshot")
    assert snap is not None
    assert "lscpu" in snap or "cpu_info" in snap
    assert "nvidia_smi_topo" in snap or "gpu_topo" in snap or snap.get("gpu_topo_unavailable") is True
    assert "numactl_h" in snap or "numa_unavailable" in snap
```

**Step 2-5:** Implement the helper as subprocess calls to `lscpu`, `nvidia-smi topo -m`, `numactl -H`, capturing stdout into the result dict. Each call wrapped in try/except to tolerate missing binaries.

Commit: `runner: topology_snapshot for CPU flags, GPU interconnect, NUMA`

### Task G.2 — CD overhead measurement

**Files:** `experiments/23_fast_path/runner_fast_path.py`, `tests/test_exp23_fast_path.py`.

**Step 1:** Failing test that runs the same config twice — once with `criticality_distill_enabled=False` and once with `True` (same steps, same seed) — and verifies the result dict contains an `cd_overhead` block with `tokens_per_sec_baseline`, `tokens_per_sec_treatment`, `overhead_fraction` fields.

**Step 2-5:** Implement the measurement pattern. This is opt-in; only smoke cells that request it pay the double-run cost. For the first smoke, only `treatment` and `budget_only` need it (they're the "can CD ship" cells).

Commit: `runner: CD overhead measurement (baseline vs CD-active tokens/sec, report only)`

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
  tests/test_exp23_fast_path.py \
  tests/test_lm_head_loss_entropy.py -q
```

Expected: all passing.

```
/opt/homebrew/bin/python3.11 experiments/24_training_time_bundle/run_exp24.py --matrix cd_first_smoke --seeds 1337 --dry-run
```

Expected: 8 cells, each with `fast_slow_enabled=True`, `lm_head_backward_mode="fused_streaming_cached_with_entropy"`.

---

## Out of scope for this plan

- **Multi-seed + 4×H100 confirmation** after first smoke shows treatment wins.
- **Precision-weighted surprise** `(CE - H[p]) · H[p]`.
- **Matched-nearby baseline control** (v2 scoring ablation).
- **Per-frequency-bucket evidence banks** (bucketed `bank_evidence`).
- **Paired / full SSM modes.**
- **Staggered CD seat-refresh vs fast/slow interval** (both at 64 in first smoke; follow-up if `seat_churn` chatters).
- **Hard overhead-gate enforcement.** First smoke measures and reports; future runs may promote to a gate.
- **CPU worker thread on a separate Python thread.** First smoke does CPU controller work on the main thread after `copy_done.synchronize()`. If profiling shows the sync is in the critical path, follow-up promotes to a dedicated thread with lock-free pinned ringbuffer.

---

## Execution notes

- **macOS tasks:** Stages A, B, C, E.1, E.4 (partial), E.5, F.1, G.* — these have CPU-only tests, run on macOS first.
- **Pod tasks:** Stage D (kernel), E.3's integration test on CUDA, Stage D.4 (pod-only). These require CUDA + kernel build; run after macOS tasks land on `main`.
- **Time the kernel stage (D.1 through D.4).** Report actual elapsed vs the 30-minute estimate in the commit message body of D.4. This calibrates future kernel-change estimates.
