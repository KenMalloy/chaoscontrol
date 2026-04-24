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

### Task C.9 — Vectorized `compute_future_energy` + `ingest_gpu` API

**Context.** The existing `compute_future_energy` uses `for t in range(T)`. At B=512, T=512 on GPU that's 512 Python iterations per layer per step — horrendous. Replace with a vectorized `cumsum + windowed difference` form. Then define `ingest_gpu` as a proper CD method that takes GPU tensors and returns **only small aggregates** — no `[B, T]` masks crossing PCIe.

**Files:** `src/chaoscontrol/optim/criticality.py`, `tests/test_criticality_scoring.py`, `tests/test_criticality_distillation.py`.

**Step 1: Failing tests.**

Append to `tests/test_criticality_scoring.py`:

```python
def test_future_energy_vectorized_matches_reference_on_large_tensor():
    """Vectorized form must match the slow reference on a shape that
    actually matters."""
    torch.manual_seed(0)
    B, T, D, H = 4, 64, 32, 8
    states = torch.randn(B, T, D)
    # Reference (Python loop — the old slow one, re-written locally).
    def _ref(states, H):
        sq = states.pow(2)
        out = torch.zeros_like(sq)
        for t in range(T):
            s, e = t + 1, min(t + 1 + H, T)
            if s < e:
                out[:, t, :] = sq[:, s:e, :].mean(dim=1)
        return out
    expected = _ref(states, H)
    actual = compute_future_energy(states, horizon_H=H)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5), (
        f"max abs diff {(actual - expected).abs().max().item()}"
    )
```

Append to `tests/test_criticality_distillation.py`:

```python
def test_ingest_gpu_returns_only_small_aggregates():
    """ingest_gpu must NOT return [B, T] event_mask in the cross-PCIe
    payload. Only layer-reduced tensors + scalar counts."""
    cd = CriticalityDistillation(num_layers=2, dim=3, trace_ttl_steps=4)
    states = [torch.randn(1, 6, 3), torch.randn(1, 6, 3)]
    pressure = torch.randn(1, 6)
    prepared = cd.ingest_gpu(
        pressure=pressure, states_per_layer=states, horizon_H=2, event_frac=0.5,
    )
    # Expected keys after finding #4 fix: no event_mask in the payload.
    assert set(prepared.keys()) == {
        "aggregated_excess_per_layer",  # [L, D]
        "non_event_mean_future_energy_per_layer",  # [L, D]
        "event_count_per_layer",  # [L]
        "n_events_scalar",  # scalar int-like
        "n_non_events_scalar",  # scalar int-like
    }
    assert prepared["aggregated_excess_per_layer"].shape == (2, 3)
    assert prepared["non_event_mean_future_energy_per_layer"].shape == (2, 3)
    assert prepared["event_count_per_layer"].shape == (2,)
    assert prepared["n_events_scalar"].numel() == 1
    assert prepared["n_non_events_scalar"].numel() == 1


def test_ingest_gpu_parity_with_ingest_step():
    """Two-phase (ingest_gpu -> ingest_cpu_from_prepared) must match
    single-call ingest_step on accumulator state."""
    cd_single = CriticalityDistillation(num_layers=1, dim=3, trace_ttl_steps=8, baseline_ema_decay=0.0)
    cd_split = CriticalityDistillation(num_layers=1, dim=3, trace_ttl_steps=8, baseline_ema_decay=0.0)
    torch.manual_seed(11)
    states = [torch.randn(2, 8, 3)]
    pressure = torch.randn(2, 8).abs()  # non-negative so events fire
    cd_single.ingest_step(step=0, pressure=pressure, states_per_layer=states, horizon_H=3, event_frac=0.25)
    prepared = cd_split.ingest_gpu(pressure=pressure, states_per_layer=states, horizon_H=3, event_frac=0.25)
    cd_split.ingest_cpu_from_prepared(step=0, prepared=prepared)
    assert torch.allclose(cd_single.score_num, cd_split.score_num, atol=1e-5)
    assert torch.allclose(cd_single.score_den, cd_split.score_den, atol=1e-5)
    assert torch.allclose(cd_single.event_mass, cd_split.event_mass, atol=1e-5)
    assert torch.allclose(cd_single.baseline_future_energy, cd_split.baseline_future_energy, atol=1e-5)
```

**Step 2: Run — expect FAIL.**

**Step 3: Implement.**

Replace `compute_future_energy` body with vectorized cumsum form:

```python
def compute_future_energy(states: torch.Tensor, horizon_H: int) -> torch.Tensor:
    """Per-position mean-square energy over the trailing window
    `[t+1, t+H]`. Vectorized — no Python loop over T.

    Args:
        states: `[B, T, D]` recurrence states.
        horizon_H: window length (strictly positive).

    Returns:
        `[B, T, D]` — empty-window tail positions produce zeros.
    """
    if horizon_H < 1:
        raise ValueError(f"horizon_H must be >= 1; got {horizon_H}")
    B, T, D = states.shape
    sq = states.pow(2)  # [B, T, D]
    zero_pad = torch.zeros(B, 1, D, dtype=sq.dtype, device=sq.device)
    csum = torch.cat([zero_pad, sq.cumsum(dim=1)], dim=1)  # [B, T+1, D]; csum[:, t, :] = Σ_{i<t} sq[:, i, :]
    t = torch.arange(T, device=sq.device)
    t_start = t + 1  # inclusive first future index
    t_end_excl = torch.clamp(t + 1 + horizon_H, max=T)  # exclusive
    valid = t_start < t_end_excl  # [T]
    # Clamp indices to [0, T] for safe gather when invalid (masked later).
    safe_start = torch.where(valid, t_start, torch.zeros_like(t_start))
    safe_end = torch.where(valid, t_end_excl, torch.zeros_like(t_end_excl))
    # sum[b, t, d] = csum[b, end, d] - csum[b, start, d]
    sum_energy = csum[:, safe_end, :] - csum[:, safe_start, :]  # [B, T, D]
    count = torch.where(
        valid,
        (t_end_excl - t_start).to(torch.float32),
        torch.zeros_like(t_start, dtype=torch.float32),
    )  # [T]
    safe_count = count.clamp_min(1.0).view(1, T, 1)
    out = sum_energy / safe_count
    return torch.where(valid.view(1, T, 1), out, torch.zeros_like(out))
```

Replace `ingest_gpu` implementation (the v3 version returned `event_mask` in the dict — drop it and add scalar counts):

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
        """Phase 1 of two-phase ingest. Runs on the pressure/states
        device. Returns only small aggregates + scalar counts — no
        [B, T] masks in the payload."""
        if len(states_per_layer) != self.num_layers:
            raise ValueError(
                f"states_per_layer must have {self.num_layers} entries; got {len(states_per_layer)}"
            )
        event_mask = compute_event_mask(pressure, event_frac=event_frac)  # [B, T]
        flat_mask = event_mask.reshape(-1)
        flat_non_event = ~flat_mask
        n_events = flat_mask.sum().to(torch.int64)
        n_non_events = flat_non_event.sum().to(torch.int64)
        aggregated_excess = []
        non_event_mean_future_energy = []
        event_count = []
        for layer, states in enumerate(states_per_layer):
            if states.shape[-1] != self.dim:
                raise ValueError(
                    f"layer {layer}: states last dim {states.shape[-1]} != self.dim {self.dim}"
                )
            future_energy = compute_future_energy(states, horizon_H=horizon_H)  # [B, T, D]
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
            "aggregated_excess_per_layer": torch.stack(aggregated_excess, dim=0),
            "non_event_mean_future_energy_per_layer": torch.stack(non_event_mean_future_energy, dim=0),
            "event_count_per_layer": torch.stack(event_count, dim=0),
            "n_events_scalar": n_events,
            "n_non_events_scalar": n_non_events,
        }
```

Update `ingest_cpu_from_prepared` to read `n_non_events_scalar` instead of `event_mask`:

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
        had_non_events = int(prepared["n_non_events_scalar"].item()) > 0
        decay = self.baseline_ema_decay
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
            self._add_contribution(layer=layer, evidence=agg[layer], event_count=cnt)
            self._write_ring_slot(
                layer=layer, step=step, evidence=agg[layer],
                event_count=cnt, current_step=step,
            )
```

**Step 4:** Run full criticality + scoring suite — all pass.

**Step 5:** Commit: `criticality: vectorized compute_future_energy + ingest_gpu returns only tiny aggregates`

---

## Stage D — Fused LM head entropy emission (time-boxed, measured)

**This is the one stage in this plan where the individual implementation step is genuinely >5 minutes.** Time it. Ken and I agreed ~30 min is the realistic estimate. Report actual elapsed in the commit message.

**Outline:**

The existing fused forward emits `(loss, lse, per_token_ce)` (cached variant also emits `logits_cache`). Current implementation in `src/chaoscontrol/kernels/_lm_head_loss/src/rms_norm_binding.cpp` computes `row_max` and `row_sum` across tiles first, then finalizes `lse = row_max + log(row_sum)` after the outer tile loop. Writing entropy "after lse is known" would require a second pass.

**One-pass path** — use the online-softmax running-max trick, adding one more accumulator (`row_exp_logit_sum`) that is kept in sync with `row_sum` whenever the running max updates:

```
initialize:
  m        = -inf            # running max
  s        = 0.0             # running exp sum       (current row_sum)
  exl      = 0.0             # running exp · logit sum (NEW)

for each tile:
  m_tile    = max over this tile's logits
  m_new     = max(m, m_tile)
  scale     = exp(m - m_new)
  # rescale old accumulators to the new max
  s       := s * scale
  exl     := exl * scale
  # add this tile's contribution under the new max
  for each v in tile:
    e   = exp(logit_v - m_new)
    s  += e
    exl += logit_v * e
  m := m_new

after loop:
  lse     = m + log(s)
  entropy = lse - exl / s
```

This keeps the forward pass single-pass over logits. Zero extra memory loads beyond the single `logit_v` value already loaded for `row_sum`.

**Integration strategy (addresses allowlist-churn finding #3).** Do NOT introduce a new `lm_head_backward_mode` value. Keep `lm_head_backward_mode="fused_streaming_cached"` unchanged. Add a separate boolean flag `lm_head_emit_entropy` that causes the Python wrapper to call an entropy-emitting variant of the forward while leaving the backward identical. Two C++ entrypoints sharing most code:

- `linear_ce_streaming_cached_forward(...)` — existing, unchanged.
- `linear_ce_streaming_cached_forward_with_entropy(...)` — same body plus the `row_exp_logit_sum` accumulator, returns `(loss, lse, per_token_ce, logits_cache, per_token_entropy)`.

No mode-allowlist plumbing needed. No test-enumeration churn.

### Task D.1 — CUDA kernel: add `row_exp_logit_sum` accumulator to the online-max loop

**Files:** `src/chaoscontrol/kernels/_lm_head_loss/src/linear_ce.cu` (or wherever the streaming forward kernel lives — locate via grep for the online-max tile loop).

**Start timer. Target: ~30 min for D.1 through D.4.**

**Step 1:** Locate the existing online-max tile loop used for `lse`. Identify where `row_max` and `row_sum` are maintained across tiles with the scale-by-exp-of-max-diff correction.

**Step 2:** Add a new device-side accumulator `row_exp_logit_sum[row]` initialized to 0. At every point where `row_sum` is rescaled by `exp(m_old - m_new)` on a max update, rescale `row_exp_logit_sum` by the same factor. Inside the per-element tile inner loop where `row_sum += exp(logit - m_new)`, also add `row_exp_logit_sum += logit * exp(logit - m_new)` (reusing the already-computed `exp(logit - m_new)` scalar).

Concretely, the minimal kernel delta is: rename the inner-loop local `float e = expf(logit - m_new);` if not already a local, then add `row_exp_logit_sum += logit * e;` on the same line where `row_sum += e;` happens. Plus one rescale line in the max-update branch.

**Step 3:** After the outer tile loop, compute `entropy_row = lse_row - row_exp_logit_sum / row_sum` and store to `per_token_entropy[row]`.

**Step 4:** Build:

```bash
MAX_JOBS=6 TORCH_CUDA_ARCH_LIST="9.0" /opt/homebrew/bin/python3.11 -m pip install -e . --no-build-isolation 2>&1 | tail -5
```

(Pod-only — macOS without CUDA will skip the CUDA compile path.)

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

### Task E.2 — CD construction + `lm_head_emit_entropy` flag

**Context.** Per codex finding #3, we are NOT introducing a new `lm_head_backward_mode` value. Instead, keep `lm_head_backward_mode="fused_streaming_cached"` and add a separate `lm_head_emit_entropy: bool` kwarg. When CD is enabled, this flag must be True. The Python wrapper picks the entropy-emitting C++ entrypoint based on this flag; the backward path is identical either way.

**Files:** `experiments/23_fast_path/runner_fast_path.py`, `tests/test_exp23_fast_path.py`.

**Step 1:** Failing tests for:
- `criticality_distill_enabled=True, lm_head_emit_entropy=True` succeeds regardless of `lm_head_backward_mode`.
- `criticality_distill_enabled=True, lm_head_emit_entropy=False` raises a clear `ValueError` (CD requires entropy).
- `criticality_distill_enabled=False` works with any `lm_head_emit_entropy` value (no coupling when CD is off).

**Step 2-5:** Implement kwarg plumbing + validation. No constants named after a mode; the flag is orthogonal to the mode. Commit: `runner: CD requires lm_head_emit_entropy=True; no new mode name`

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

**Per codex finding #4:** only layer-aggregate tensors + scalar counts cross PCIe. No `[B, T]` event_mask. Total per-step D2H at `L=24, D=512`: `2 × [24, 512] fp32 + [24] fp32 + 2 scalars ≈ 98 KB`.

```python
# At CD construction time, allocate two pinned host buffers per prepared-key.
def _alloc_pinned_evidence_buffers(num_layers: int, dim: int) -> dict:
    def _slot():
        return {
            "aggregated_excess_per_layer": torch.empty(num_layers, dim, pin_memory=True, dtype=torch.float32),
            "non_event_mean_future_energy_per_layer": torch.empty(num_layers, dim, pin_memory=True, dtype=torch.float32),
            "event_count_per_layer": torch.empty(num_layers, pin_memory=True, dtype=torch.float32),
            "n_events_scalar": torch.empty((), pin_memory=True, dtype=torch.int64),
            "n_non_events_scalar": torch.empty((), pin_memory=True, dtype=torch.int64),
        }
    return {"A": _slot(), "B": _slot()}

# Per-step, inside the training loop (with CD active):
parity = step % 2  # ping-pong A/B
host_slot = pinned_buffers["A" if parity == 0 else "B"]

# Forward + CE + entropy via the fused entropy-emitting kernel
# (lm_head_emit_entropy=True selects the entrypoint; lm_head_backward_mode
# is unchanged at "fused_streaming_cached").
with ExitStack() as stack:
    _ = [stack.enter_context(c.capture_states()) for c in ssm_cores]
    hidden = model.encode(inputs)
    loss, lse, per_token_ce, per_token_entropy = fused_lm_head_forward_with_ce_entropy(...)
states_per_layer = [c._captured_states for c in ssm_cores]

# Pressure.
if criticality_distill_uniform_pressure:
    pressure = torch.ones_like(per_token_ce).reshape(B, T)
else:
    pressure = compute_ce_minus_entropy_pressure_from_fused(
        per_token_ce.reshape(B, T), per_token_entropy.reshape(B, T),
    )

# GPU reduction: returns tiny aggregates + scalar counts, NO event_mask.
prepared_gpu = criticality.ingest_gpu(
    pressure=pressure,
    states_per_layer=states_per_layer,
    horizon_H=int(criticality_distill_horizon_H),
    event_frac=float(criticality_distill_event_frac),
)
# Async D2H of small aggregates + scalars only.
for key in ("aggregated_excess_per_layer",
            "non_event_mean_future_energy_per_layer",
            "event_count_per_layer",
            "n_events_scalar",
            "n_non_events_scalar"):
    host_slot[key].copy_(prepared_gpu[key], non_blocking=True)

# Record event so the CPU consumer can wait.
copy_done = torch.cuda.Event()
copy_done.record()

# ... training backward + optimizer step happens on main stream ...
loss.backward()
optimizer.step()

# After backward, CPU accumulator update (D2H is done by now).
copy_done.synchronize()
criticality.ingest_cpu_from_prepared(step=step, prepared=host_slot)

# Seat refresh every N steps.
if step % criticality.seat_refresh_interval == 0 and step > 0:
    criticality.allocate_seats_from_accumulators(current_step=step)
    criticality.sync_seat_mask_to_device(device)
```

Commit: `runner: CD wiring with fused entropy, tiny-aggregate pinned async D2H, accumulator update`

### Task E.4 — Diagnostics snapshot + val-time per-bucket CE (explicit result schema)

**Files:** `src/chaoscontrol/optim/criticality.py`, `experiments/23_fast_path/runner_fast_path.py`, `tests/test_criticality_distillation.py`, `tests/test_exp23_fast_path.py`.

**Context.** The design gate is rare-bucket CE on Param-Golf val, not the training-time EMA (codex finding #5). Current fast scorer emits aggregate BPB; this task extends it to per-bucket and adds the CD diagnostic snapshots.

**Required output schema on `train_fast_for_budget`'s result dict when CD is active:**

```python
{
  # Primary success metric (val-time, gates the mechanism).
  "per_bucket_val_ce": list[float],   # length = num_buckets, lowest log-freq bucket first (index 0 = rarest)
  "rare_bucket_val_ce": float,         # mean of indices [0 : max(1, num_buckets // 4)]
  "val_bucket_num_buckets": int,       # bookkeeping
  "val_bucket_token_counts": list[int], # per-bucket token count seen on val (for noise-aware reading)

  # Diagnostic snapshots — one per seat refresh during training.
  "criticality_distillation_diagnostics": list[dict],   # each dict:
  #   {
  #     "step": int,
  #     "seat_churn_per_layer": list[float],            # fraction of seats changed since previous snapshot
  #     "budget_occupancy_per_layer": list[float],      # fraction of seat channels with criticality >= 0.9 * critical_value
  #     "score_criticality_corr_per_layer": list[float], # rank correlation (score vs current criticality) per layer
  #     "event_rate_per_layer": list[float],            # mean event_count per populated bank slot, per layer
  #     "seat_mask_fraction_per_layer": list[float],    # fraction of channels currently seated
  #   }

  # Overhead measurement (Task G.2 — only when enabled).
  "cd_overhead": {
    "tokens_per_sec_baseline": float,
    "tokens_per_sec_treatment": float,
    "overhead_fraction": float,  # 1 - treatment/baseline
  } | None,
}
```

**Step 1:** Failing tests.

```python
def test_train_result_contains_per_bucket_val_ce_when_rare_bucket_ce_enabled():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    token_frequencies = torch.tensor([100.0, 50.0, 20.0, 10.0, 5.0, 1.0])
    result = mod.train_fast_for_budget(
        model, train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128, stride=4, seq_len=3, batch_size=2,
        device=torch.device("cpu"), optimizer=optimizer,
        budget_seconds=300.0, chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=4, prefetch_batches=False,
        rare_bucket_ce_enabled=True,
        rare_bucket_ce_token_frequencies=token_frequencies,
        rare_bucket_ce_num_buckets=4,
    )
    assert "per_bucket_val_ce" in result
    assert len(result["per_bucket_val_ce"]) == 4
    assert "rare_bucket_val_ce" in result
    assert isinstance(result["rare_bucket_val_ce"], float)
    assert "val_bucket_num_buckets" in result
    assert result["val_bucket_num_buckets"] == 4
    assert "val_bucket_token_counts" in result


def test_cd_diagnostics_emitted_at_every_seat_refresh():
    mod = _load_runner_module()
    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = mod.train_fast_for_budget(
        model, train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128, stride=4, seq_len=3, batch_size=2,
        device=torch.device("cpu"), optimizer=optimizer,
        budget_seconds=300.0, chunk_size=2, grad_clip_norm=0.0, fused_grad_clip=False,
        rank=0, world_size=1, seed=123, precision="fp32",
        stop_check_interval=1, stop_margin_seconds=0.0,
        vocab_size=6, max_steps=8, prefetch_batches=False,
        lm_head_backward_mode="fused_streaming_cached",
        lm_head_emit_entropy=True,
        criticality_distill_enabled=True,
        criticality_distill_seat_refresh_interval=2,
        # ... other CD defaults
    )
    diags = result["criticality_distillation_diagnostics"]
    # 8 steps / refresh every 2 = 4 snapshots (plus possibly step 0).
    assert len(diags) >= 3
    for snap in diags:
        for key in ("step", "seat_churn_per_layer", "budget_occupancy_per_layer",
                    "score_criticality_corr_per_layer", "event_rate_per_layer",
                    "seat_mask_fraction_per_layer"):
            assert key in snap, f"diagnostic snapshot missing {key}"
```

**Step 2-5:** Implement:

- **Per-bucket val CE.** In the full-val scorer, build a frequency-bucketizer (same log-binning as `FrequencyBucketBaseline`) from `rare_bucket_ce_token_frequencies + num_buckets`. For each val batch, compute per-position CE (already computed), scatter-add into `bucket_sum[B]` and `bucket_count[B]`. At the end, `per_bucket_val_ce[b] = bucket_sum[b] / max(bucket_count[b], 1)`. Mean of first `max(1, num_buckets // 4)` buckets → `rare_bucket_val_ce`. Emit both + `val_bucket_num_buckets` + `val_bucket_token_counts` in the result.
- **Diagnostics.** `CriticalityDistillation.diagnostics_snapshot(log_a_per_layer, current_step)` method returns the per-layer lists defined above, AND includes `seat_mask_fraction_per_layer`. Called at every seat refresh inside the runner training loop; append to `result["criticality_distillation_diagnostics"]`.

Commit: `runner: val-time per-bucket CE + CD diagnostics snapshot (explicit result schema)`

### Task E.5 — Config-threading test

**Files:** `tests/test_cd_config_threading.py`.

Same shape as v2 Task D.5, updated required-key set for v3:

```python
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
    "lm_head_emit_entropy",   # new: entropy-flag approach, not mode renaming
    "rare_bucket_ce_enabled",
    "rare_bucket_ce_num_buckets",
    "rare_bucket_ce_token_frequencies",
}
```

Commit: `runner: config-threading test confirms CD + entropy-flag + per-bucket CE kwargs reach train_fast_for_budget`

---

## Stage F — Smoke matrix on fast/slow base

### Task F.1 — `build_criticality_distillation_first_smoke_matrix`

Same as v2 Stage E with these differences for v3 (codex findings #3, #6):

1. `base["lm_head_backward_mode"] = "fused_streaming_cached"` (unchanged mode name), plus `base["lm_head_emit_entropy"] = True` (new flag that drives the entropy-emitting forward entrypoint).
2. Matrix test asserts every cell has both: `lm_head_backward_mode == "fused_streaming_cached"` AND `lm_head_emit_entropy is True`.
3. Explicitly register the world size in `run_exp24.py`'s `_default_world_size_for_matrix`: add a branch `if matrix == "cd_first_smoke": return 1`. The runner design says 1×H100 for first smoke; 4×H100 reserved for confirmation. Default-returning 8 is wrong for this matrix.

**Additional test for #6:**

```python
def test_run_exp24_defaults_cd_first_smoke_to_world_size_1():
    from run_exp24 import _default_world_size_for_matrix
    assert _default_world_size_for_matrix("cd_first_smoke") == 1
```

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

Expected: 8 cells, each with `fast_slow_enabled=True`, `lm_head_backward_mode="fused_streaming_cached"`, `lm_head_emit_entropy=True`, `world_size=1`.

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
