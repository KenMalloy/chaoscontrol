# CPU SSM Controller Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Build the trained CPU SSM controller's runtime + online learning + bootstrap pipeline on top of the substrate landed in commit `1295301`. Final deliverable: a controller that learns to curate the cache during the 600s training window, ships a slow-snapshot in the artifact, and continues TTT-style learning during the 600s eval window.

**Architecture:** Separate-process streaming CPU SSM that consumes a shared-memory event log of GPU writes / queries / replay outcomes, writes curation actions back to the cache the GPUs read from, and learns online via per-slot action history with Gerber-statistic-based off-policy correction. Mirrors the core SSM architecturally (TTT at eval, fast/slow weight pair, eval uses slow). C++ inference + online learning on the Sapphire Rapids target; PyTorch only for the offline bootstrap pretrain.

**Tech Stack:** C++ (the controller runtime); Python + PyTorch (offline pretrain, glue, tests); POSIX shm + lock-free SPSC ring buffers (event transport); pybind11 (or raw CPython if `_cpu_ssm_controller` already uses raw); safetensors (artifact format); pytest (tests).

**Source design:** `docs/plans/2026-04-25-cpu-ssm-controller-design.md` is the spec. This plan is the bite-sized execution decomposition.

**Pre-built substrate (already landed in `1295301`):**
- EpisodicCache extended with `pressure_at_write`, `source_write_id`, `write_bucket`, `slot_state[N, D_slot]`, `simplex_edges[N, K_max]` + round-trip via `to_dict`/`from_dict`
- Replay diagnostics carry the full REPLAY_OUTCOME schema columns (NaN-reserved for Phase 4 grad_cos_*)
- Trainer checkpoint includes `ckpt["episodic_cache"]`
- C++ extension scaffold at `src/chaoscontrol/kernels/_cpu_ssm_controller/` with fp32 diagonal SSM reference step (cross-language parity tested)
- Replay fanout default = 0 (unbounded), preserving pre-landing behavior

---

## Phase A: Shared-memory event rings (foundation)

The wire that carries WRITE_EVENT, QUERY_EVENT, REPLAY_OUTCOME records from each GPU rank to the controller process. Per-rank SPSC, POSIX shm. Three rings per rank (one per event type). Lock-free, fixed-size slots, header with read/write indices.

### Task A1: Define wire record structs in C++

**Files:**
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/wire_events.h`
- Test: `tests/cpp/test_wire_events.cpp` (build via `setup_ext.py` test target — add target if not present)

**Sizes (corrected from initial draft — 552/528 were unreachable arithmetic with K=256 + the field set; corrected target sizes are derived from the body + minimal 8-byte alignment padding):**
- WriteEvent: 568 bytes (body 564, _pad1[4] aligns to 8)
- QueryEvent: 544 bytes (body 544, naturally 8-aligned, no _pad1 needed)
- ReplayOutcome: 96 bytes (body 94, _pad1[2] aligns to 8)

**Step 1: Write the failing test**

```cpp
// tests/cpp/test_wire_events.cpp
#include "wire_events.h"

int main() {
    static_assert(sizeof(WriteEvent) == 568, "WriteEvent must be 568 bytes");
    static_assert(sizeof(QueryEvent) == 544, "QueryEvent must be 544 bytes");
    static_assert(sizeof(ReplayOutcome) == 96, "ReplayOutcome must be 96 bytes");
    static_assert(alignof(WriteEvent) == 8, "WriteEvent must be 8-byte aligned");
    return 0;
}
```

**Step 2: Run test to verify it fails**

`make test_wire_events` → expect compile error: `wire_events.h` not found.

**Step 3: Write minimal implementation**

```cpp
// src/wire_events.h
#pragma once
#include <cstdint>

constexpr int KEY_REP_DIM_DEFAULT = 256;
constexpr int SPAN_LENGTH_DEFAULT = 4;

#pragma pack(push, 1)

struct WriteEvent {
    uint8_t  event_type;            // = 1
    uint8_t  source_rank;
    uint8_t  write_bucket;
    uint8_t  _pad0[5];              // align to 8
    uint64_t candidate_id;          // (source_rank << 56) | rank_seq
    uint64_t gpu_step;
    uint64_t key_fp;
    uint16_t key_rep[KEY_REP_DIM_DEFAULT];   // f16 storage as u16
    uint16_t value_tok_ids[SPAN_LENGTH_DEFAULT];
    uint32_t value_anchor_id;
    float    pressure_at_write;
    float    pre_write_ce;
    uint8_t  _pad1[4];              // align body 564 → 568 (8-byte boundary)
};

struct QueryEvent {
    uint8_t  event_type;            // = 2
    uint8_t  source_rank;
    uint8_t  bucket;
    uint8_t  _pad0[5];
    uint64_t query_id;
    uint64_t gpu_step;
    uint16_t query_rep[KEY_REP_DIM_DEFAULT];
    float    pressure;
    float    pre_query_ce;
    // body 544 — already 8-aligned, no _pad1 required
};

struct ReplayOutcome {
    uint8_t  event_type;            // = 3
    uint8_t  selected_rank;
    uint8_t  outcome_status;
    uint8_t  _pad0[5];
    uint64_t replay_id;
    uint64_t gpu_step;
    uint64_t query_event_id;
    uint64_t source_write_id;
    uint32_t slot_id;
    uint32_t policy_version;
    uint64_t selection_step;
    float    teacher_score;
    float    controller_logit;
    float    ce_before_replay;
    float    ce_after_replay;
    float    ce_delta_raw;
    float    bucket_baseline;
    float    reward_shaped;
    float    grad_cos_rare;         // NaN until Phase 4
    float    grad_cos_total;        // NaN until Phase 4
    uint16_t flags;
    uint8_t  _pad1[2];              // align body 94 → 96 (8-byte boundary)
};

#pragma pack(pop)

static_assert(sizeof(WriteEvent) == 568);
static_assert(sizeof(QueryEvent) == 544);
static_assert(sizeof(ReplayOutcome) == 96);
```

**Step 4: Run test to verify it passes**

`make test_wire_events` → exit 0.

**Step 5: Commit**

```bash
git add src/chaoscontrol/kernels/_cpu_ssm_controller/src/wire_events.h \
        tests/cpp/test_wire_events.cpp
git commit -m "ssm_controller: wire-event structs for shm transport"
```

### Task A2: SPSC ring buffer in C++

**Files:**
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/spsc_ring.h`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/spsc_ring.cpp`
- Test: `tests/cpp/test_spsc_ring.cpp`

**Steps:**
1. Write failing test: producer writes 1024 events, consumer reads 1024, all match, ring is empty after.
2. Run test → fail (header missing).
3. Implement `SpscRing<T>`: header with `std::atomic<uint64_t> write_idx, read_idx`, slot array of N elements, push/pop with `memory_order_acquire/release`. Use cacheline padding to avoid false sharing.
4. Run test → pass.
5. Add second test: consumer reads while producer writes from another thread; verify no torn reads.
6. Commit: `ssm_controller: lock-free SPSC ring buffer for event transport`

### Task A3: POSIX shm wrapper

**Files:**
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/posix_shm.h`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/posix_shm.cpp`
- Test: `tests/cpp/test_posix_shm.cpp`

**Steps:**
1. Test: create shm region, write a byte, read it back, unlink.
2. Implement `PosixShm`: `shm_open`, `ftruncate`, `mmap`, `munmap`, `shm_unlink`. RAII wrapper.
3. Test: two processes (fork) share the same shm region, one writes, other reads.
4. Commit: `ssm_controller: POSIX shm wrapper with RAII`

### Task A4: Compose Ring + Shm into ShmRing

**Files:**
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/shm_ring.h`
- Test: `tests/cpp/test_shm_ring.cpp`

**Steps:**
1. Test: producer process creates a `ShmRing<WriteEvent>` of capacity 1024, pushes 100 events; consumer process attaches by name, pops all 100.
2. Implement `ShmRing<T>` = `PosixShm` + `SpscRing<T>` placed in the shm region. Ring header at offset 0, slots from offset `sizeof(header)` rounded to cacheline.
3. Commit: `ssm_controller: ShmRing composing SPSC + POSIX shm`

### Task A5: Python ShmRing bindings

**Files:**
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/cpu_ssm_controller.cpp` (extend bindings)
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/__init__.py` (export ShmRing)
- Test: `tests/test_shm_ring_python.py`

**Steps:**
1. Test: from Python, create a write-event ring of capacity 1024, push a `WriteEvent` from Python (numpy struct array or dict), attach from a child process, pop and verify all fields.
2. Implement bindings: `ShmRing.create(name, capacity, event_type)`, `.push(event_dict)`, `.pop()`, `.attach(name)`, `.unlink()`.
3. Commit: `ssm_controller: Python bindings for ShmRing`

---

## Phase B: GPU-side event producers

The rank process (CPU) packs the events into rings as the GPU produces tensors. Three producers, each gated by a config flag so this lands as a no-op for non-controller runs (preserves bit-identity).

### Task B1: WRITE_EVENT producer in select_writes

**Files:**
- Modify: `src/chaoscontrol/optim/episodic_writer.py` (add ring push after write decision)
- Modify: `experiments/23_fast_path/runner_fast_path.py` (allocate per-rank ring, pass to writer)
- Test: `tests/test_write_event_producer.py`

**Steps:**
1. Test: configure runner with `episodic_event_log_enabled=True`, run one training step with one write, attach to the rank's shm ring from the test, verify exactly one WriteEvent record present with the right fields.
2. Test: with `episodic_event_log_enabled=False` (default), no ring is allocated and no records are pushed (back-compat).
3. Implement: in `select_writes`, after the admission decision, build a `WriteEvent` dict and push to the rank's `write_ring`. Include `(source_rank, rank_seq, gpu_step, key_fp, key_rep, value_tok_ids, value_anchor_id, pressure_at_write, pre_write_ce, write_bucket)`.
4. Commit: `episodic: WRITE_EVENT producer in select_writes`

### Task B2: QUERY_EVENT producer

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py` (in the query emission path, push to query_ring)
- Test: `tests/test_query_event_producer.py`

**Steps:**
1. Test: run one training step that triggers a query, attach to query_ring, verify one QueryEvent.
2. Implement: at the point where the controller_query_queue is filled today, also push a QueryEvent to the per-rank query_ring (when ring is allocated).
3. Commit: `episodic: QUERY_EVENT producer in runner query path`

### Task B3: REPLAY_OUTCOME producer

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py` (in the replay drain path, push to replay_ring after each replay)
- Test: `tests/test_replay_outcome_producer.py`

**Steps:**
1. Test: pre-populate tagged_replay_queue with 2 entries, run one episodic step, verify 2 ReplayOutcome records in replay_ring with the right (replay_id, slot_id, ce_before_replay, ce_after_replay) values.
2. Implement: after each replay's CE delta is computed, push a ReplayOutcome record. Compute `bucket_baseline` from a per-bucket EMA (state lives on the runner; persisted via cache snapshot for now).
3. Commit: `episodic: REPLAY_OUTCOME producer in replay drain`

### Task B4: Per-rank ring allocation + naming

**Files:**
- Modify: `experiments/23_fast_path/runner_fast_path.py` (init region — allocate rings under names like `cc_episodic_write_rank{R}`)
- Modify: `src/chaoscontrol/episodic/cpu_ssm_controller.py` (controller spawn — attach to all ranks' rings)
- Test: `tests/test_ring_naming.py`

**Steps:**
1. Test: with world_size=4 and `episodic_event_log_enabled=True`, verify all 12 rings (4 ranks × 3 events) are allocated with the expected names.
2. Implement: ring allocation in init region, names parameterized by rank + event_type. Cleanup on shutdown via shm_unlink.
3. Commit: `episodic: per-rank ring allocation + lifecycle`

---

## Phase C: Controller event consumer + credit assignment

The C++ controller process reads all rings round-robin, dispatches by event type, maintains per-slot action history, computes gradients via Gerber-statistic credit attribution, and applies SGD.

### Task C1: Controller main loop + ring polling

**Files:**
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/controller_main.cpp`
- Test: `tests/cpp/test_controller_main.cpp`

**Steps:**
1. Test: spawn a controller process, push 10 WriteEvents from a producer, verify the controller processes all 10 (verifiable via a counter in shared state).
2. Implement: `controller_main(num_ranks, exit_flag_shm)` polls all rings round-robin in a tight loop; sleep_ns(100) when idle; exit on flag set.
3. Commit: `ssm_controller: main loop with round-robin ring polling`

### Task C2: Event dispatch by type

**Files:**
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/controller_main.cpp`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/event_handlers.h`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/event_handlers.cpp`
- Test: `tests/cpp/test_event_dispatch.cpp`

**Steps:**
1. Test: push 1 WriteEvent + 1 QueryEvent + 1 ReplayOutcome; verify each handler is called exactly once.
2. Implement: `handle_write(WriteEvent&)`, `handle_query(QueryEvent&)`, `handle_replay_outcome(ReplayOutcome&)`. Each is a stub for now (just increments a counter).
3. Commit: `ssm_controller: event dispatch by type`

### Task C3: Per-slot action history

**Files:**
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/action_history.h`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/action_history.cpp`
- Test: `tests/cpp/test_action_history.cpp`

**Steps:**
1. Test: append 100 actions for slot 0, walk back, verify all 100 in reverse order. Append for slot 1, verify slot 0's history unchanged.
2. Test: evict slot 0, verify history retained for `gc_lookahead_events=10000` more events, then dropped.
3. Implement: per-slot ring buffer of `ActionHistoryEntry { action_type, gpu_step, policy_version, global_state[D_global], slot_state[D_slot], output_logit, ... }`. GC on slot eviction.
4. Commit: `ssm_controller: per-slot action history with eviction-triggered GC`

### Task C4: Recency decay

**Files:**
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/credit.h`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/credit.cpp`
- Test: `tests/cpp/test_credit_recency.cpp`

**Steps:**
1. Test: `recency_decay(R=1.0, T=1000, P=860)` returns `0.995^140` ≈ 0.496 (half-life check).
2. Implement: `recency_decay(R, T, P, gamma=0.995) -> float = R * pow(gamma, T - P)`.
3. Commit: `ssm_controller: recency decay credit factor`

### Task C5: Gerber-statistic off-policy correction

**Files:**
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/credit.cpp`
- Test: `tests/cpp/test_gerber_correction.cpp`

**Steps:**
1. Test: with `H=0.5`, both logits at 1.0 (concordant positive) → weight 1.0. Both at -1.0 (concordant negative) → weight 1.0. One +1.0 one -1.0 (discordant) → weight 0. One 0.1 one 1.0 (inactive) → weight 0.
2. Implement: `gerber_weight(L_v, L_current, H) -> float`.
3. Add `RollingStddev` helper for σ_logit per decision type. `H = c * sigma`, c=0.5.
4. Commit: `ssm_controller: Gerber-statistic off-policy correction`

### Task C6: Rank-aware credit + assembled credit attribution

**Files:**
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/credit.cpp`
- Test: `tests/cpp/test_credit_attribution.cpp`

**Steps:**
1. Test: REPLAY_OUTCOME with `selected_rank=0` → rank multiplier 1.0. `selected_rank=4` → 0.2.
2. Test: full credit walk — 5 actions in slot's history, REPLAY_OUTCOME arrives, verify each action gets `R * recency * gerber * rank_factor` for its type.
3. Implement: `attribute_credit(slot_id, replay_outcome) -> vector<CreditedAction>` walks history backward, computes credit per action.
4. Commit: `ssm_controller: assembled credit attribution per replay outcome`

### Task C7: SSM forward + backward through saved hidden state

**Files:**
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/cpu_ssm_controller.cpp` (extend the existing fp32 reference step with a backward path)
- Test: `tests/cpp/test_ssm_backward.cpp` + `tests/test_cpu_ssm_controller_runtime.py` (Python parity)

**Steps:**
1. Test: forward through a 4-layer diagonal SSM; backward against a unit gradient on the output; verify the input-side gradient matches a numerical-differentiation reference.
2. Implement: backward pass in C++. Diagonal SSM is friendly: `dh/dx = decay * dh_next/dx + W_in.T @ dy`. Linear in time, no recurrence-unrolling required if hidden state is saved.
3. Add Python parity test: forward + backward in C++ matches PyTorch reference to 1e-5.
4. Commit: `ssm_controller: backward pass + Python parity test`

### Task C8: SGD step on fast weights

**Files:**
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/optimizer.h`
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/optimizer.cpp`
- Test: `tests/cpp/test_sgd_step.cpp`

**Steps:**
1. Test: SGD on a single weight, gradient = 1.0, lr = 0.1 → weight decreases by 0.1.
2. Implement: `SgdStep::apply(weights, gradients, lr)`. No momentum.
3. Commit: `ssm_controller: SGD step on fast weights`

### Task C9: Fast/slow EMA blend

**Files:**
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/optimizer.cpp`
- Test: `tests/cpp/test_fast_slow_ema.cpp`

**Steps:**
1. Test: `slow=1.0, fast=2.0, alpha=0.25` → `slow' = 0.75 * 1.0 + 0.25 * 2.0 = 1.25`.
2. Test: triggered every 64 events, not every event.
3. Implement: `FastSlowEma::tick_event()` increments counter, `maybe_blend()` triggers blend when counter % 64 == 0.
4. Commit: `ssm_controller: fast/slow EMA blend (α=0.25, interval=64)`

### Task C10: Wire credit attribution + SGD into REPLAY_OUTCOME handler

**Files:**
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/event_handlers.cpp`
- Test: `tests/cpp/test_online_learning_loop.cpp`

**Steps:**
1. Test: end-to-end — feed 100 WriteEvents and 50 ReplayOutcomes (with synthetic rewards correlated to a hidden feature), verify the controller's policy learns to upweight that feature within 50 steps.
2. Implement: in `handle_replay_outcome`, run `attribute_credit`, recompute SSM forward from saved states, backward against credit-weighted target, accumulate gradients, apply SGD every 256 actions or 1s wall-clock, blend slow weights every 64 events.
3. Commit: `ssm_controller: online learning loop wired to replay outcomes`

---

## Phase D: Trace logging + offline PyTorch bootstrap pretrain

The controller starts cold without bootstrap. Pretraining on heuristic traces avoids the random-init exploration tax.

### Task D1: ADMISSION trace logging on heuristic writer

**Files:**
- Modify: `src/chaoscontrol/optim/episodic_writer.py` (log every admit/reject decision)
- Test: `tests/test_admission_trace_logging.py`

**Steps:**
1. Test: enable trace logging, run 1 training step with 4 candidate writes, verify the trace file has 4 ADMISSION rows with `(candidate_id, decision, gpu_step, pressure, pre_write_ce, key_fp, key_rep_l2)`.
2. Implement: when `episodic_admission_trace_path` is set, append NDJSON rows to that path. No-op when unset (back-compat).
3. Commit: `episodic: admission trace logging on heuristic writer`

### Task D2: Eviction trace logging on cache

**Files:**
- Modify: `src/chaoscontrol/optim/episodic_cache.py` (log every eviction in `append` when capacity is full)
- Test: `tests/test_eviction_trace_logging.py`

**Steps:**
1. Test: fill cache to capacity, append one more, verify EVICTION row with `(evicted_slot_id, evicted_key_fp, evicted_utility_at_eviction, gpu_step)`.
2. Implement: optional `eviction_trace_path` on EpisodicCache.
3. Commit: `episodic: eviction trace logging on cache`

### Task D3: Replay-outcome NDJSON consolidator

**Files:**
- Modify: `src/chaoscontrol/episodic/diagnostics.py` (already writes per-replay logs; verify schema includes new fields from substrate landing)
- Test: `tests/test_replay_outcome_log_schema.py`

**Steps:**
1. Test: assert the NDJSON rows from a real run contain all REPLAY_OUTCOME schema columns (the diagnostic log was extended in `1295301`).
2. Implement: confirm — likely no work, but pin the test as a regression catch.
3. Commit: `episodic: pin replay-outcome NDJSON schema for offline pretrain`

### Task D4: PyTorch BC + value-prediction pretrain pipeline (offline)

**Files:**
- Create: `experiments/25_controller_pretrain/pretrain_controller.py`
- Create: `experiments/25_controller_pretrain/__init__.py`
- Create: `tests/test_controller_pretrain.py`

**Steps:**
1. Test: synthetic dataset of (event_features, heuristic_action, observed_reward) tuples; assert pretrain converges to non-trivial accuracy on policy head + non-trivial R^2 on value head within 100 epochs on a tiny model.
2. Implement: load NDJSON trace files, batch into (event, heuristic_action, delayed_reward) tuples, train a small SSM (D_global=128, 4 layers) with two heads — policy (cross-entropy on heuristic's top-K) + value (regression on `reward_shaped`).
3. Run on a real trace file from a heuristic-only training run (TBD when trace data is harvested).
4. Commit: `controller_pretrain: offline BC + value-prediction pipeline`

### Task D5: Weight dump format (PyTorch → C++ binary)

**Files:**
- Create: `experiments/25_controller_pretrain/dump_to_cpp.py`
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/__init__.py` (add `load_weights_from_path`)
- Test: `tests/test_controller_weight_dump.py`

**Steps:**
1. Test: train a tiny SSM in PyTorch, dump to a binary file, load into the C++ runtime, verify forward outputs match within 1e-5.
2. Implement: dump format = header (n_layers, D_global, D_slot, dtype) + concatenated f16 weights in a documented order.
3. Commit: `controller_pretrain: PyTorch → C++ weight dump format`

---

## Phase E: AMX intrinsics for hot path (Sapphire Rapids only)

The substrate uses generic fp32 matmul. AMX BF16 + AVX-512 deliver the throughput the design relies on. This phase is a pure performance optimization; correctness is already established.

### Task E1: Detect AMX/AVX-512 at runtime

**Files:**
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/cpu_ssm_controller.cpp` (add CPUID detection)
- Test: `tests/test_cpu_capability_detection.py`

**Steps:**
1. Test: on a Sapphire Rapids pod, `has_amx()` returns True. On darwin / non-AMX hosts, returns False with a clear log.
2. Implement: CPUID-based detection. Fallback paths use the existing fp32 reference.
3. Commit: `ssm_controller: runtime AMX/AVX-512 capability detection`

### Task E2: AMX BF16 matmul kernel

**Files:**
- Create: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/amx_matmul.cpp`
- Test: `tests/cpp/test_amx_matmul.cpp` (gated by AMX availability)

**Steps:**
1. Test: AMX matmul of (16x16) BF16 against an fp32 reference matches to 1e-3.
2. Implement: AMX tile config + `_tile_loadd` + `_tile_dpbf16ps`.
3. Commit: `ssm_controller: AMX BF16 matmul kernel`

### Task E3: AVX-512 diagonal recurrence

**Files:**
- Modify: `src/chaoscontrol/kernels/_cpu_ssm_controller/src/cpu_ssm_controller.cpp`
- Test: `tests/cpp/test_avx512_recurrence.cpp` (gated by AVX-512)

**Steps:**
1. Test: AVX-512 diagonal recurrence over D=512 matches fp32 reference to 1e-5.
2. Implement: `_mm512_load_ps`, `_mm512_fmadd_ps` for the diagonal multiply-accumulate.
3. Commit: `ssm_controller: AVX-512 diagonal recurrence kernel`

### Task E4: Pod-side throughput verification

**Files:**
- Create: `experiments/25_controller_pretrain/bench_amx.py`

**Steps:**
1. Run on Sapphire Rapids pod: benchmark generic vs AMX path on 100K events. Expect ~10x speedup.
2. Document results in `bench_amx.md`.
3. Commit: `bench: AMX vs generic CPU SSM throughput on Sapphire Rapids`

---

## Phase F: Falsifier matrix arms

Take the trained controller to the falsifier matrix and answer: does it beat the heuristic?

### Task F1: Define matrix arms

**Files:**
- Modify: `experiments/24_training_time_bundle/exp24.py` (add `build_episodic_controller_v1_matrix`)
- Test: `tests/test_exp24_training_bundle.py` (new test asserting the matrix has the expected arms)

**Arms:**
- `arm_a_control` — no episodic
- `arm_b_heuristic_cold` — heuristic controller, cold cache
- `arm_b_heuristic_warm` — heuristic controller, warm cache
- `arm_c_trained_cold_frozen` — trained controller, cold cache, no eval-time learning
- `arm_d_trained_cold_online` — trained controller, cold cache, eval-time TTT
- `arm_e_trained_warm_online` — trained controller, warm cache, eval-time TTT

3 seeds per arm = 18 cells.

**Steps:**
1. Test: assert all 6 arms with all 3 seeds present, assert the controller-config flags differ as expected per arm.
2. Implement: extend the matrix builder.
3. Commit: `exp24: episodic_controller_v1 matrix (6 arms × 3 seeds)`

### Task F2: Analysis script extensions

**Files:**
- Modify: `experiments/24_training_time_bundle/analyze_phase3.py` (or wherever the Phase 3 analysis lives)
- Test: `tests/test_phase3_analysis_controller.py`

**Steps:**
1. Test: feed synthetic results with known per-arm BPB, verify the analysis script computes the right comparisons (heuristic vs trained, cold vs warm, frozen vs online-TTT).
2. Implement: pairwise comparisons + significance tests + plots.
3. Commit: `exp24: Phase 3 analysis extensions for controller arms`

### Task F3: Pod run + harvest

**No tasks here — this is a runbook step.** Once F1+F2 land:
1. Launch the matrix on a 4xH100 + 26-vCPU Sapphire Rapids pod
2. Monitor via /loop (per the runpodcli + monitor_via_loop conventions)
3. Harvest logs + JSON results (per the cd_multiseed harvest pattern from `81a509c`)
4. Commit the harvest + push

---

## Sequencing notes

- **A and B can interleave per-event-type:** A1+A2+A3+A4 is the foundation, then per-event-type slices land as A5+B1, A5+B2, A5+B3.
- **C depends on B:** can't test online learning without events flowing.
- **D can start in parallel with C:** trace logging hooks live on the heuristic, no controller needed.
- **E depends on C7+C8 working correctness-first:** AMX is a perf swap, not a feature.
- **F depends on A through E (or at least D):** matrix runs need the trained controller to exist.

## Open dependencies to land first (NOT in this plan)

- **#103** trainer cache save: per the C1 reviewer's heads-up, `_construct_episodic_cache` at `runner_fast_path.py:1610-1618` needs `fingerprint_window=int(config.get("episodic_fingerprint_window", 8))` added. Else silent miss when config sets W ≠ 8. This blocks a real Phase F run.
- **#104** `run_exp20_fast_score.py` cache-field wiring: blocks the score-only mode of any cache-aware arm.
- **#95** ScOpt allreduce migration: orthogonal but on the path to Phase 4 reward shaping (`grad_cos_rare`).
