# Codex Handoff Briefs — Phase B4 through F3

**Usage:**
- Each section below is a self-contained brief that can be pasted into Codex.
- Codex commits + pushes to `origin/main`; the controller (Claude) reviews after each push.
- One brief per Codex session for clean file ownership.

**Sequence (sequential dependency chain):**
1. **B4** — real shm-ring producer wiring (replaces B1/B2/B3 placeholder lists)
2. **C1 → C2 → C3 → C4 → C5 → C6 → C7 → C8 → C9 → C10** — controller main loop + credit assignment + SGD + fast/slow EMA. Each builds on prior; do not parallelize.
3. **E1 → E2 → E3 → E4** — AMX/AVX-512 perf swap. After C correctness is established.
4. **F1 → F2 → F3** — matrix arms → analysis → pod run. F1 can start once C3+ lands (matrix references controller config flags). F3 is a runbook, not a Codex coding task.

**Parallel-safe with the above:**
- **B5 NDJSON follow-up** (anytime; touches a single file)

**Canonical references** (Codex should read these first when picking up a brief):
- `docs/plans/2026-04-25-cpu-ssm-controller-design.md` — design source of truth, refreshed at `98b8006`
- `docs/plans/2026-04-26-cpu-ssm-controller.md` — execution plan with task-level specs

---

## B4 — Real shm-ring producer wiring

Working dir: `/Users/kennethmalloy/Local Documents/Developer/chaoscontrol`. Branch: main. `git pull --rebase origin main` before start and again before push.

### Scene-setting

Last piece of Phase B. Phase A built the shm rings (commits `57d608f` → `0a89338` for A1-A5 plus review fixes). Phase B's producers (B1 `ef5e81b`, B2 `74954e7`, B3 `8420d45`) currently push WRITE/QUERY/REPLAY_OUTCOME dicts onto **placeholder Python lists** held on consumer state. B4 swaps those for the real shm-backed rings exposed by A5 (`ShmRingWriteEvent`, `ShmRingQueryEvent`, `ShmRingReplayOutcome` in `chaoscontrol.kernels._cpu_ssm_controller`).

The placeholder lists were always meant to be temporary — every B1/B2/B3 test docstring says "Phase B4 will replace the in-process list with shm-ring pushes once Phase A4 (ShmRing) lands."

B5 (commit `47ff3d6`) just landed the pre/post replay CE pair — the reward signal is now real. Phase C (controller process consuming the rings) is next.

### Architectural recap — three producer shapes

Per the design-doc refresh ("Producer-shape differs per event type"):

- **WRITE events** — emitted on each TRAIN RANK directly via `select_writes`. **Per-train-rank ring producer** (one ring per train rank).
- **QUERY events** — emitted on the EPISODIC RANK after `dist.gather`, aggregated per source-rank. Single ring on episodic rank.
- **REPLAY_OUTCOME events** — episodic rank produces, single ring.

**Total rings for 3-train + 1-episodic deployment:** 3 write_rings + 1 query_ring + 1 replay_outcome_ring = 5 rings.

### Naming

POSIX shm names start with `/`. Use PID-suffixed names so concurrent runs / tests don't collide:

- Write rings: `/cc_episodic_write_rank{R}_pid{PID}` for train rank R
- Query ring: `/cc_episodic_query_pid{PID}`
- Replay-outcome ring: `/cc_episodic_replay_pid{PID}`

Stash the resolved names on consumer state (`consumer.write_ring_name: str`, etc.) so test code reads them without reconstructing.

### Files to modify

**`experiments/23_fast_path/runner_fast_path.py`:**
- Init region / `_attach_episodic_consumer`: when `episodic_event_log_enabled=True`, allocate the rings via `_ext.ShmRingX.create(name)`. Hold them on consumer state.
- Producer sites: replace `consumer.write_event_log.append(d)` with `consumer.write_ring.push(d)`. `push` returns `False` on full → increment a counter (e.g., `consumer.write_ring_drops`); don't raise; don't block.
- Cleanup on shutdown: `_ext.ShmRingX.unlink(name)` for each ring.

**`src/chaoscontrol/optim/episodic_writer.py`:** B1's WRITE_EVENT push site — same swap.

**`src/chaoscontrol/eval_stream/types.py`:** existing `episodic_event_log_enabled` flag already gates the chain. No new flag needed.

### Test infrastructure

The existing B1/B2/B3 tests inspect placeholder lists. After B4, migrate them to attach to the rings + pop:

- **In-process attach** (simpler, faster): `ring = _ext.ShmRingWriteEvent.attach(consumer.write_ring_name)` then `ring.pop()`. Use this for the existing test migrations.
- **Cross-process attach via os.fork()** (matches A3/A4 pattern): use this for a NEW integration test.

### Existing tests to migrate

- `tests/test_write_event_producer.py` — list inspection → `ShmRingWriteEvent.attach(...).pop()`
- `tests/test_query_event_producer.py` — same
- `tests/test_replay_outcome_producer.py` — same

### New cross-process test (`tests/test_real_shm_ring_producers.py`)

```python
"""Cross-process shm ring producer test (Phase B4)."""
import os
import time
from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_write_event_ring_cross_process_round_trip():
    # Producer: simulate the runner emitting 100 WRITE_EVENTs
    # Consumer: child process via os.fork attaches by name, pops, verifies
    pass


def test_query_event_ring_cross_process_round_trip():
    pass


def test_replay_outcome_ring_cross_process_round_trip():
    pass


def test_ring_full_drops_increment_counter_does_not_crash():
    """Pushing 32K events through a 16K-capacity write ring without
    a consumer running should drop ~16K and increment the drops
    counter, not crash."""
    pass
```

### Lifecycle invariants

- Create on rank init when `episodic_event_log_enabled=True`. PID-suffixed names.
- Hold the ring object on consumer state.
- Producer push returns False on full → increment `*_ring_drops`; don't raise; don't block.
- Cleanup unlinks the ring name on runner shutdown. Idempotent.
- `episodic_event_log_enabled=False` (default) → no rings allocated, no producers fire, bit-identical to pre-B4 (I4's invariant).

### Commit + push

```
git pull --rebase origin main
git add experiments/23_fast_path/runner_fast_path.py \
        src/chaoscontrol/optim/episodic_writer.py \
        src/chaoscontrol/eval_stream/types.py \
        tests/test_write_event_producer.py \
        tests/test_query_event_producer.py \
        tests/test_replay_outcome_producer.py \
        tests/test_real_shm_ring_producers.py
git commit -m "episodic: B4 — swap producer placeholder lists for real shm rings"
git push origin main
```

### Self-review

- [ ] All three producer sites push to ShmRing instead of list
- [ ] Per-train-rank write ring; single query + replay rings on episodic
- [ ] PID-suffixed names; held on consumer state; unlinked on shutdown
- [ ] Drops counter; producer doesn't raise on full
- [ ] Existing producer tests migrated; new cross-process test in place
- [ ] `episodic_event_log_enabled=False` → no rings, bit-identical (I4 invariant)

### Out of scope

- Phase C (consuming the rings)
- Phase E (AMX)
- F1 (matrix arms)
- B5 NDJSON divergence follow-up

### When done

Report: commit SHA, total test count, any test that required design adjustments (e.g., FIFO ordering vs list ordering), the actual `*_ring_drops` behavior in fixtures, anything surprising about the runner's shutdown path. Under 300 words.

---

## C1 — Controller main loop + ring polling

Working dir same as B4. `git pull --rebase` before start and before push.

### Scene-setting

Phase C builds the trained CPU SSM controller process — the consumer of B4's shm rings. C1 is the entry point: a long-running C++ function that polls all rings round-robin, dispatches events to per-type handlers, and exits cleanly when signaled.

Read the canonical spec at `docs/plans/2026-04-26-cpu-ssm-controller.md` Task C1 for the full design. Read the design-doc's "Architecture spine" + "Event-log schema" sections at `docs/plans/2026-04-25-cpu-ssm-controller-design.md`.

### Files

- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/controller_main.cpp` (the polling loop + exit-flag handling)
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/cpu_ssm_controller.cpp` (`#include "controller_main.h"`, add pybind11 binding)
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/__init__.py` (re-export `controller_main`)
- Create: `tests/test_controller_main.py` (Python test: spawn a controller process, push 10 WriteEvents from a producer, verify processed)

### `controller_main` design

```cpp
// src/controller_main.h
#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "shm_ring.h"
#include "wire_events.h"

// Polls all rings round-robin until exit_flag is set. Idle sleep when
// no rings have events. Returns total event count processed (for the
// test harness; production caller ignores it).
//
// ring_names: { "/cc_episodic_write_rank0_pidX", ..., "/cc_episodic_query_pidX", "/cc_episodic_replay_pidX" }
// exit_flag_shm_name: a small POSIX shm region (1 byte) the parent
//   process writes 1 to when shutdown is requested.
uint64_t controller_main(
    const std::vector<std::string>& write_ring_names,
    const std::string& query_ring_name,
    const std::string& replay_ring_name,
    const std::string& exit_flag_shm_name,
    uint32_t idle_sleep_ns = 100
);
```

C1 only stubs the per-event handlers (each handler increments a counter, no real logic yet). C2 wires real dispatch.

### TDD

Per the canonical plan Task C1 step list. Test pattern: spawn a Python subprocess that calls `_ext.controller_main(...)`, push events from the test process, verify counters via shared-memory introspection (e.g., a small counter region the controller updates).

### Commit + push

```
git add src/chaoscontrol/kernels/_cpu_ssm_controller/src/controller_main.cpp \
        src/chaoscontrol/kernels/_cpu_ssm_controller/src/controller_main.h \
        src/chaoscontrol/kernels/_cpu_ssm_controller/src/cpu_ssm_controller.cpp \
        src/chaoscontrol/kernels/_cpu_ssm_controller/__init__.py \
        tests/test_controller_main.py
git commit -m "ssm_controller: C1 main loop with round-robin ring polling"
git push origin main
```

### Out of scope

- Per-event handlers beyond stubs (C2)
- Action history / credit assignment (C3+)
- SGD / EMA (C8/C9)

---

## C2 — Event dispatch by type

### Scene-setting

C1 stubs each event-type handler as a counter increment. C2 makes them real dispatch points: `handle_write(WriteEvent&)`, `handle_query(QueryEvent&)`, `handle_replay_outcome(ReplayOutcome&)`. Each is still a placeholder (records into per-type ring buffers or counters), but the dispatch table is wired.

### Files

- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/event_handlers.h`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/event_handlers.cpp`
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/controller_main.cpp` (call dispatch table)
- Modify: `tests/test_controller_main.py` (test that 1 WriteEvent + 1 QueryEvent + 1 ReplayOutcome each call exactly the right handler)

### Design

```cpp
// event_handlers.h
class EventHandlers {
public:
    EventHandlers();
    void handle_write(const WriteEvent& ev);
    void handle_query(const QueryEvent& ev);
    void handle_replay_outcome(const ReplayOutcome& ev);

    // For tests
    uint64_t write_count() const { return write_count_; }
    uint64_t query_count() const { return query_count_; }
    uint64_t replay_outcome_count() const { return replay_outcome_count_; }

private:
    uint64_t write_count_ = 0;
    uint64_t query_count_ = 0;
    uint64_t replay_outcome_count_ = 0;
};
```

C3+ extends each `handle_*` with the real action history + credit assignment + SSM forward. C2 just establishes the dispatch surface.

### Commit

```
git commit -m "ssm_controller: C2 event dispatch by type"
```

---

## C3 — Per-slot action history

### Scene-setting

The credit-assignment substrate. When a REPLAY_OUTCOME arrives for slot S with reward R, the controller walks S's history of past actions (admission, replay-selection, utility-update, edge-update) and credits each. C3 builds the data structure that holds this history.

Per the design doc's "Credit assignment" section + the plan's Task C3.

### Files

- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/action_history.h`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/action_history.cpp`
- Test: `tests/test_action_history.py` (Python test via pybind11 binding)

### Design

```cpp
// action_history.h
struct ActionHistoryEntry {
    uint8_t  action_type;             // 0=admission, 1=replay_selection, 2=utility_update, 3=edge_update
    uint64_t gpu_step;
    uint32_t policy_version;
    float    output_logit;
    uint8_t  selected_rank;           // valid only for replay_selection
    uint32_t neighbor_slot;           // valid only for edge_update
    // Hidden state checkpoints — variable size; store as separate
    // contiguous arrays per slot history. Sizes from controller config.
    // (In V1: D_global=128, D_slot=16; total ~288 bytes per entry.)
    std::vector<float> global_state;
    std::vector<float> slot_state;
};

class PerSlotActionHistory {
public:
    PerSlotActionHistory(uint32_t num_slots, uint32_t max_entries_per_slot);

    void append(uint32_t slot_id, ActionHistoryEntry entry);

    // Walk backward from the most recent entry. Returns a span/reference
    // into the slot's ring; caller must not retain across other appends
    // to the same slot.
    const std::vector<ActionHistoryEntry>& history(uint32_t slot_id) const;

    // GC: when slot S is evicted, retain history for `gc_lookahead_events`
    // more events (cover in-flight replays), then drop.
    void mark_evicted(uint32_t slot_id, uint64_t current_event_id);
    void gc(uint64_t current_event_id, uint64_t gc_lookahead);
};
```

### Test

Append 100 actions for slot 0; walk back; verify all 100 in reverse order. Append for slot 1; verify slot 0 unchanged. Mark slot 0 evicted; gc after gc_lookahead events; verify slot 0 history dropped.

### Commit

```
git commit -m "ssm_controller: C3 per-slot action history with eviction-triggered GC"
```

---

## C4 — Recency decay

### Scene-setting

Small focused task. The credit attribution multiplies the reward by `gamma ^ (T - P)` where T is the current gpu_step and P is the action's gpu_step at decision time. Half-life ~140 steps with `gamma = 0.995`.

### Files

- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/credit.h`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/credit.cpp`
- Test: `tests/test_credit_recency.py`

### Design

```cpp
// credit.h
inline float recency_decay(float reward, uint64_t T, uint64_t P, float gamma = 0.995f) {
    if (T < P) return reward;  // shouldn't happen but be defensive
    return reward * std::pow(gamma, static_cast<float>(T - P));
}
```

### Test

`recency_decay(R=1.0, T=1000, P=860, gamma=0.995)` → `0.995^140` ≈ 0.496 (half-life check). Boundary: `recency_decay(R=1.0, T=P, P=P)` → 1.0 exactly. Boundary: very large `T - P` → bounded > 0.

### Commit

```
git commit -m "ssm_controller: C4 recency decay credit factor"
```

---

## C5 — Gerber-statistic off-policy correction

### Scene-setting

Standard off-policy correction is importance-ratio clipping (`clip(π_current / π_V, 0.5, 2.0)`). Ken's call: use the Gerber statistic instead — a robust co-movement metric from finance (Gerber/Markowitz/Pujara) that asks "do current and behavior policies confidently AGREE on the action's polarity above noise?" Concordant beyond noise → full credit. Inactive (either logit within noise band) or discordant → drop credit (weight 0).

Avoids the importance-ratio division-by-near-zero pathology entirely.

### Files

- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/credit.{h,cpp}`
- Test: `tests/test_gerber_correction.py`

### Design

```cpp
// credit.h
// Returns 1.0 if both logits are concordant beyond noise threshold H,
// 0.0 if either is in the noise band OR if they're discordant beyond H.
inline float gerber_weight(float L_v, float L_current, float H) {
    bool v_active = std::abs(L_v) > H;
    bool c_active = std::abs(L_current) > H;
    if (!v_active || !c_active) return 0.0f;  // inactive
    if ((L_v > 0.0f) != (L_current > 0.0f)) return 0.0f;  // discordant
    return 1.0f;  // concordant beyond noise
}

// Rolling stddev helper for σ_logit per decision type. H = c * sigma,
// c=0.5 default. Use Welford's online variance for numerical stability.
class RollingStddev {
public:
    explicit RollingStddev(float decay = 0.99f);
    void update(float x);
    float stddev() const;
private:
    float ema_ = 0.0f;
    float ema_sq_ = 0.0f;
    float decay_;
};
```

### Test

- Both logits at +1.0, H=0.5 → 1.0 (concordant positive)
- Both at -1.0, H=0.5 → 1.0 (concordant negative)
- One +1.0, one -1.0, H=0.5 → 0.0 (discordant)
- One 0.1, one 1.0, H=0.5 → 0.0 (inactive — first below H)
- RollingStddev: feed N samples from N(0, sigma), assert estimated stddev within 5% of true sigma after >100 samples

### Commit

```
git commit -m "ssm_controller: C5 Gerber-statistic off-policy correction"
```

---

## C6 — Rank-aware credit + assembled credit attribution

### Scene-setting

Combines C3 (history) + C4 (recency) + C5 (Gerber). When REPLAY_OUTCOME arrives, walk the slot's history backward; for each action compute `credit = R · γ^(T-P) · gerber_weight(L_V, L_current, H)`; for REPLAY_SELECTION actions also multiply by `1 / (selected_rank + 1)` (top-1 full credit, top-K attenuated).

### Files

- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/credit.{h,cpp}` (add the assembled `attribute_credit` function)
- Test: `tests/test_credit_attribution.py`

### Design

```cpp
struct CreditedAction {
    ActionHistoryEntry entry;
    float credit;  // signed; sign comes from reward
};

std::vector<CreditedAction> attribute_credit(
    uint32_t slot_id,
    const ReplayOutcome& outcome,
    const PerSlotActionHistory& history,
    const RollingStddevPerType& sigma,
    float gamma,
    float gerber_c
);
```

### Test

Five actions in slot's history; REPLAY_OUTCOME with R=+0.5 arrives 100 steps later; each action gets `R * gamma^(T-P) * gerber * rank_factor` credit. Rank factor is `1/(rank+1)` for replay_selection, 1.0 for others. Verify sums match.

### Commit

```
git commit -m "ssm_controller: C6 assembled credit attribution per replay outcome"
```

---

## C7 — SSM forward + backward through saved hidden state

### Scene-setting

The substrate's `cpu_ssm_controller.cpp` already has an fp32 diagonal SSM reference forward step. C7 extends with the BACKWARD pass: given the saved hidden state from a past action and a target gradient on the output, compute gradients on the SSM weights. Diagonal SSM is friendly: `dh/dx = decay * dh_next/dx + W_in.T @ dy`, linear in time, no recurrence-unrolling required.

### Files

- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/cpu_ssm_controller.cpp` (extend with backward path)
- Test: `tests/test_ssm_backward.py`

### Design

The backward needs to handle the per-slot recurrent state (D_slot=8-16) plus the global state (D_global=64-128). Each layer's diagonal recurrence has `decay`, `w_in`, `w_out`, `bias` — gradients w.r.t. each.

Reference: PyTorch's autograd on the same architecture (already in `experiments/25_controller_pretrain/pretrain_controller.py`). Cross-language parity test: run forward + backward in C++ vs PyTorch on the same input, verify weight gradients match within 1e-5.

### Test

```python
def test_ssm_backward_matches_pytorch_reference():
    # Construct a 4-layer diagonal SSM in both C++ and PyTorch
    # with identical weights
    # Forward through both, get the same output (already tested)
    # Backward through both with a unit gradient on output
    # Verify all per-parameter gradients match within 1e-5
```

### Commit

```
git commit -m "ssm_controller: C7 SSM backward + Python parity test"
```

---

## C8 — SGD step on fast weights

### Scene-setting

Plain SGD, no momentum. Fast/slow EMA (C9) provides the temporal smoothing; momentum on top is doubly-smoothed and sluggish under non-stationary RL. Per Ken's call.

### Files

- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/optimizer.h`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/optimizer.cpp`
- Test: `tests/test_sgd_step.py`

### Design

```cpp
class SgdStep {
public:
    SgdStep(float lr) : lr_(lr) {}
    void apply(float* weights, const float* gradients, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            weights[i] -= lr_ * gradients[i];
        }
    }
private:
    float lr_;
};
```

### Test

`SgdStep(0.1).apply(w=1.0, g=1.0)` → w = 0.9. Vector test: 100 weights, all g=1.0 → all weights decrease by lr.

### Commit

```
git commit -m "ssm_controller: C8 SGD step on fast weights"
```

---

## C9 — Fast/slow EMA blend

### Scene-setting

Mirrors the core SSM's fast_slow pattern. Every `interval=64` events, blend `slow = (1-α)·slow + α·fast`, α=0.25. Eval uses slow.

### Files

- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/optimizer.{h,cpp}`
- Test: `tests/test_fast_slow_ema.py`

### Design

```cpp
class FastSlowEma {
public:
    FastSlowEma(float alpha = 0.25f, uint64_t interval = 64)
        : alpha_(alpha), interval_(interval) {}

    void tick_event() { ++event_count_; }

    bool should_blend() const { return event_count_ % interval_ == 0; }

    void blend(float* slow, const float* fast, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            slow[i] = (1.0f - alpha_) * slow[i] + alpha_ * fast[i];
        }
    }
private:
    float alpha_;
    uint64_t interval_;
    uint64_t event_count_ = 0;
};
```

### Test

- `FastSlowEma(0.25, 64)`: tick 63 times → `should_blend() == false`. Tick 1 more → true.
- `blend(slow=1.0, fast=2.0, n=1)` with α=0.25 → slow = 0.75 + 0.5 = 1.25.

### Commit

```
git commit -m "ssm_controller: C9 fast/slow EMA blend (α=0.25, interval=64)"
```

---

## C10 — Online learning loop

### Scene-setting

The integration task. C2's `handle_replay_outcome` becomes real: walk the slot's action history (C3), compute credit (C6 = C4 × C5 × rank), recompute the SSM forward from the saved state (substrate), backward (C7), accumulate gradients, every 256 actions or 1s wall-clock apply SGD (C8), every 64 events blend slow weights (C9).

Also: `handle_write` and `handle_query` write entries into the action history (C3) — admission and replay-selection actions with their saved hidden states.

### Files

- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/event_handlers.cpp`
- Test: `tests/test_online_learning_loop.py`

### Design

```cpp
void EventHandlers::handle_replay_outcome(const ReplayOutcome& ev) {
    // 1. Look up slot's action history (C3)
    const auto& history = action_history_->history(ev.slot_id);

    // 2. Walk backward, credit each action (C6)
    auto credited = attribute_credit(ev.slot_id, ev, *action_history_, sigma_, gamma_, gerber_c_);

    // 3. For each credited action: recompute SSM forward from saved state,
    //    backward against credit-weighted target, accumulate gradients (C7)
    for (const auto& ca : credited) {
        if (ca.credit == 0.0f) continue;
        ssm_forward(ca.entry.global_state, ca.entry.slot_state, ...);
        ssm_backward(target = ca.credit, accumulate into grad buffers);
    }

    // 4. Periodic SGD step (C8) and slow EMA (C9)
    fast_slow_.tick_event();
    if (++actions_since_step_ >= 256 || /* wall clock check */) {
        sgd_.apply(weights_fast_, grad_buf_, n_params_);
        std::fill(grad_buf_, grad_buf_ + n_params_, 0.0f);  // zero
        actions_since_step_ = 0;
    }
    if (fast_slow_.should_blend()) {
        fast_slow_.blend(weights_slow_, weights_fast_, n_params_);
    }
}
```

### Test

End-to-end: feed 100 WriteEvents and 50 ReplayOutcomes with synthetic rewards correlated to a hidden feature. Verify the controller's policy learns to upweight that feature within 50 steps (output_logit on the feature direction increases monotonically).

### Commit

```
git commit -m "ssm_controller: C10 online learning loop wired to replay outcomes"
```

---

## E1 — Detect AMX/AVX-512 at runtime

### Scene-setting

Phase E starts. The C++ runtime up to C10 uses generic fp32 matmul + scalar diagonal recurrence. E1-E4 swap to AMX BF16 + AVX-512 intrinsics on Sapphire Rapids; falls back to the generic path on darwin / non-AMX hosts.

### Files

- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/cpu_ssm_controller.cpp` (add CPUID detection)
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/cpu_features.{h,cpp}`
- Test: `tests/test_cpu_capability_detection.py`

### Design

CPUID-based detection. Gate AMX behind `has_amx()`; gate AVX-512 behind `has_avx512f()`. On darwin / Apple Silicon, both return false → generic fallback. Log on first call so the test harness can verify which path is hot.

### Test

On darwin: `has_amx() == False` AND `has_avx512f() == False`. On a Sapphire Rapids pod (Linux x86_64 with the right CPUID bits): both True. Test gated by platform (skip-if-not-Linux for the True case).

### Commit

```
git commit -m "ssm_controller: E1 CPUID-based AMX/AVX-512 detection with generic fallback"
```

---

## E2 — AMX BF16 matmul kernel

### Scene-setting

The hot-path swap. Replace the generic fp32 matmul in the SSM forward with AMX BF16 tile ops (`_tile_loadd`, `_tile_dpbf16ps`). ~10x throughput speedup on Sapphire Rapids vs generic.

### Files

- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/amx_matmul.cpp`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/amx_matmul.h`
- Test: `tests/test_amx_matmul.py` (gated on `has_amx()`)

### Design

AMX uses 8 tile registers, each up to 1KB. For BF16 matmul `C[16,16] += A[16,32] @ B[32,16]`:
1. `LDTILECFG` to set up tile shapes
2. `_tile_loadd(0, A_ptr, A_stride)` — load A into tile 0
3. `_tile_loadd(1, B_ptr, B_stride)` — load B into tile 1
4. `_tile_dpbf16ps(2, 0, 1)` — accumulate `tile2 += dot(tile0, tile1)` in fp32
5. `_tile_stored(2, C_ptr, C_stride)` — store C from tile 2

For larger matmuls, tile over the 16x16 block size. Cross-platform: `#ifdef __AMX_BF16__` (gcc/clang macro).

Reference: `intrin.h` documentation, Intel's AMX programming guide.

### Test

Construct (16, 32) and (32, 16) BF16 matrices, call AMX matmul, compare against fp32 reference. Match within 1e-3 (BF16 precision).

### Commit

```
git commit -m "ssm_controller: E2 AMX BF16 matmul kernel"
```

---

## E3 — AVX-512 diagonal recurrence

### Scene-setting

The diagonal SSM recurrence is element-wise: `h_next[i] = decay[i] * h[i] + W_in[i,:] @ x[:]`. Element-wise FMA is AVX-512's bread and butter. 16 fp32 elements per cycle per AVX unit.

### Files

- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/avx512_recurrence.cpp`
- Test: `tests/test_avx512_recurrence.py` (gated on `has_avx512f()`)

### Design

```cpp
// avx512_recurrence.cpp
#ifdef __AVX512F__
#include <immintrin.h>

void diagonal_recurrence_avx512(
    const float* __restrict__ decay,
    const float* __restrict__ x,
    float* __restrict__ h,
    std::size_t D
) {
    for (std::size_t i = 0; i < D; i += 16) {
        __m512 d = _mm512_load_ps(decay + i);
        __m512 xi = _mm512_load_ps(x + i);
        __m512 hi = _mm512_load_ps(h + i);
        // h = d * h + x  (FMA)
        hi = _mm512_fmadd_ps(d, hi, xi);
        _mm512_store_ps(h + i, hi);
    }
}
#endif
```

D should be a multiple of 16 (the SSM is configured this way per the design). If a residual remainder exists, fall back to scalar for the tail.

### Test

D=512 random vectors; AVX-512 path matches scalar reference within 1e-5.

### Commit

```
git commit -m "ssm_controller: E3 AVX-512 diagonal recurrence kernel"
```

---

## E4 — Pod-side throughput verification

### Scene-setting

Not a Codex task per se — this is a benchmarking script + a documentation artifact. Run on a Sapphire Rapids pod, document the speedup. The output of E4 is a benchmark JSON + a one-page markdown summary.

### Files

- Create: `experiments/25_controller_pretrain/bench_amx.py`
- Create: `experiments/25_controller_pretrain/bench_amx.md` (results + interpretation)

### Design

The benchmark script:
1. Runs the controller's online-learning hot path (forward + backward + SGD per event) for 100K synthetic events.
2. Measures per-event latency in three modes: (a) generic fp32, (b) AVX-512 only, (c) AMX BF16 + AVX-512.
3. Outputs a JSON with mean / median / p99 latency per mode.

Interpretation goes into `bench_amx.md`. Expect ~10x speedup AMX vs generic.

### Commit

```
git add experiments/25_controller_pretrain/bench_amx.py \
        experiments/25_controller_pretrain/bench_amx.md
git commit -m "bench: AMX vs generic CPU SSM throughput on Sapphire Rapids"
```

---

## F1 — Define falsifier matrix arms

### Scene-setting

Take the trained controller to the falsifier matrix. Six arms × three seeds = 18 cells.

Read the canonical plan at `docs/plans/2026-04-26-cpu-ssm-controller.md` Task F1 for the full arm spec.

### Arms

- `arm_a_control` — no episodic
- `arm_b_heuristic_cold` — heuristic controller (from X-merged commits), cold cache at eval
- `arm_b_heuristic_warm` — heuristic, warm cache loaded from trainer
- `arm_c_trained_cold_frozen` — trained controller, cold cache, no eval-time TTT
- `arm_d_trained_cold_online` — trained controller, cold cache, eval-time TTT enabled
- `arm_e_trained_warm_online` — trained controller, warm cache, eval-time TTT enabled

### Files

- Modify: `experiments/24_training_time_bundle/exp24.py` (add `build_episodic_controller_v1_matrix`)
- Test: `tests/test_exp24_training_bundle.py` (assert all 6 arms × 3 seeds present + flags differ as expected)

### Test

```python
def test_episodic_controller_v1_matrix_has_six_arms_three_seeds():
    matrix = build_episodic_controller_v1_matrix()
    assert len(matrix) == 18  # 6 × 3
    arms = {entry["arm"] for entry in matrix}
    assert arms == {
        "arm_a_control", "arm_b_heuristic_cold", "arm_b_heuristic_warm",
        "arm_c_trained_cold_frozen", "arm_d_trained_cold_online",
        "arm_e_trained_warm_online"
    }
    # Verify arm-specific flags
    for entry in matrix:
        if entry["arm"] == "arm_a_control":
            assert entry["episodic_cache_enabled"] is False
        elif entry["arm"] == "arm_c_trained_cold_frozen":
            assert entry["controller_train_online"] is False
        # etc.
```

### Commit

```
git commit -m "exp24: episodic_controller_v1 matrix (6 arms × 3 seeds)"
```

---

## F2 — Phase 3 analysis script extensions

### Scene-setting

Currently the analysis flow is: matrix runs → JSON results harvest → ad-hoc Python interpretation. F2 formalizes the pairwise comparisons + significance tests + plots so the V1 deliverable has a reproducible analysis surface.

### Files

- Modify (or create): `experiments/24_training_time_bundle/analyze_phase3.py` (or wherever the existing Phase 3 analysis lives)
- Test: `tests/test_phase3_analysis_controller.py`

### Design

Three pairwise comparisons:
1. **Trained vs heuristic** — `arm_d_trained_cold_online` vs `arm_b_heuristic_cold` (cold-cache held constant; the controller policy is the only difference). Per-seed BPB, paired t-test, mean/std bars.
2. **Cold vs warm** — `arm_e_trained_warm_online` vs `arm_d_trained_cold_online` (online controller held constant; cache init is the only difference). Same stats.
3. **Frozen vs online TTT** — `arm_c_trained_cold_frozen` vs `arm_d_trained_cold_online` (cold cache + trained controller held constant; only TTT toggles). Same stats.

Plus per-arm summary table: arm name, mean BPB, std BPB, fraction of seeds beating arm_a_control.

### Test

Feed synthetic results JSON with known per-arm BPB; verify the analysis script computes the right pairwise comparisons + stats.

### Commit

```
git commit -m "exp24: Phase 3 analysis extensions for controller arms"
```

---

## F3 — Pod run + harvest (RUNBOOK, NOT CODEX)

This is **not** a Codex task. It's a runbook for Ken (or whoever runs the pod). Reproduced here for completeness.

### Sequence

1. Push everything green to `origin/main`.
2. Launch a 4×H100 + 26-vCPU Sapphire Rapids pod via runpodctl.
3. SSH in. Activate `/workspace/venv`. Pull latest.
4. Build the C++ extension on the pod: `python src/chaoscontrol/kernels/_cpu_ssm_controller/setup_ext.py build_ext --inplace`. Verify AMX path is hot via the E1 capability log.
5. Launch the matrix: `python experiments/24_training_time_bundle/run_exp24.py --matrix episodic_controller_v1 --seeds 1234,1337,42 --budget 600`.
6. Monitor via `/loop` per `feedback_monitor_via_loop.md` — slow loop + ScheduleWakeup beats one-shot bash watchers.
7. Harvest on completion (per the cd_multiseed pattern, commit `81a509c`):
   - rsync logs + JSON results to local repo `experiments/24_training_time_bundle/results/logs/episodic_controller_v1/`
   - verify file count (18 cells × 1 JSON each = 18 + bundle log)
   - commit + push (force-add over `experiments/*/results/*` gitignore per the cd_multiseed pattern)
8. Run the F2 analysis script on the harvested results. Read the verdict.

### Pod stop policy

Per `feedback_always_stop_pods.md`: only stop on prior agreement. The matrix run takes ~6h (3 seeds × 6 arms × 600s training + 600s eval + overhead). Confirm with Ken before stopping.

---

## B5 NDJSON divergence follow-up

### Scene-setting

B5 (commit `47ff3d6`) added the post-step CE pair to the wire-side REPLAY_OUTCOME. But the diagnostic NDJSON log in `src/chaoscontrol/episodic/diagnostics.py` writes its row BEFORE `optimizer.step()` — so the NDJSON's `replay_grad_norm` and `replay_loss` columns have the pre-step values while the wire-side has the post-step values. Two sources of truth diverge.

D4's BC pretrain pipeline reads the NDJSON. If we train on stale `ce_after_replay`, the value head learns the wrong target.

### Fix

Reorder: write the NDJSON row AFTER `optimizer.step()` and AFTER `_run_post_step_replay_ce` patches the dict. The NDJSON write should consume the same dict the wire-side ring sees.

### Files

- Modify: `experiments/23_fast_path/runner_fast_path.py` (move the NDJSON write site)
- Test: `tests/test_replay_outcome_log_schema.py` (add assertion that NDJSON row's CE matches the wire-side dict's CE)

### Commit

```
git commit -m "episodic: B5 follow-up — NDJSON row writes after post-step CE patch"
```

---

## End

All briefs above are self-contained. Codex picks one, executes, commits + pushes. The controller (Claude) reviews each commit and merges any review fixes (Codex pushes directly to main; review fixes are separate follow-up commits per the convention established in commits `bc7548c` (A1), `8e7a21a` (A2)).
