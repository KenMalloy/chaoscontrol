# CPU Evidence Engine — design

Status: proposal
Author: Ken / Claude
Date: 2026-04-29
Supersedes: the `cpu_scorer=...` integration that lets `chunked_nll_from_hidden`
be reached from the exact-oracle path.

## Why

The current code pulls CPU AMX scoring into the exact-oracle path through a
shared `chunked_nll_from_hidden(..., cpu_scorer=...)` hook. Once enabled, that
hook makes the CPU score the full
`(16 hide variants + force_on + memory_off) × B × T × V` row count of an Exp26
maintenance chunk, which lands at ~46 s per 16-slot chunk. The CPU is fast in
absolute terms; it is fast for the wrong job.

The role the CPU was meant to play — bounded controller-side evidence, ranking,
prefilter, scheduling — is correct and useful. But the implementation expressed
that role as a generic `score_hidden(arbitrary_shape)` primitive, which made it
trivially possible for a caller with B×T×V-shaped work to reuse it. That is the
shape of the bug, and it will recur as long as the type signature of CPU
scoring matches the type signature of the exact oracle.

## Goal

A CPU-only memory-evidence engine that keeps the episodic memory plane
continuously supplied with ranked, actionable maintenance work without becoming
visible to the training trunk or starving GPU3. The engine is *not* a scorer,
not a fallback oracle, and not a knob anyone can flip. It is a streaming
pipeline whose only public surface is "advance the stream."

## Hard requirements

1. **CPU-only.** AMX, AVX-512, thread pinning, huge pages, cache-pinned buffers
   are all on the table. No GPU kernels.
2. **Non-blocking to the trunk.** Lanes never sit on the GPU0–2 critical path.
   If the engine falls behind, it drops, merges, or lowers priority work
   internally rather than slowing training.
3. **Fast enough to be invisible.** Operationally: queue depth bounded, frame
   age below TTL, GPU3 never idle waiting on CPU scheduling. The wall-clock
   cost of one engine tick is fixed by tile geometry, not by call shape.
4. **Streaming over a Factual-Counterfactual Table.** Evidence advances in
   fixed-size, cache-aligned tiles with predictable latency. There is no API
   that takes arbitrary `(B, T, D)` hidden states and returns NLL.

## Architectural shape

Three roles, separated at the type/signature level, not by convention:

```
A — Oracle Producer (GPU3 only)
    Reads ArmMaintenanceJob from the job ring.
    Runs force_on / memory_off / hide-slot / refresh-candidate physics.
    Writes ArmMaintenanceResult onto the result ring.
    The ONLY producer of source=gpu_exact rows.

B — Evidence Stream (CPU lanes — this design)
    Drains the result ring into the FCT.
    Streams tiles over the FCT, refreshing per-slot rolling state.
    Optionally fills source=cpu_estimate rows via a tile-bounded AMX probe.
    Emits ranked SlotWorkItems back onto the job ring.

C — Controller / Action Loop (CPU FullA + simplex; existing)
    Reads per-slot rolling state from the engine.
    Decides PRESERVE / DECAY / EVICT / REFRESH / QUARANTINE / DISTILL.
```

A and B communicate only through SPSC rings. B and C communicate only through
read-only views of FCT slot state. There is no shared scoring API between A
and B.

## The Factual-Counterfactual Table (FCT)

Cache-resident, ring-allocated, AoS, 64-byte rows (one cache line each).

```c
struct FCTRow {                  // 64 bytes
    uint32_t slot_id;            // 4
    uint32_t frame_id;           // 4
    uint16_t bucket;             // 2   token / position bucket
    uint8_t  source;             // 1   0=stale 1=cpu_estimate 2=gpu_exact
    uint8_t  confidence;         // 1   q8 in [0, 255]
    uint16_t flags;              // 2   proposed_action bits, contradiction
    uint16_t age_ticks;          // 2   since last refresh
    bf16     factual_loss;       // 2
    bf16     factual_entropy;    // 2
    bf16     counterfactual_dlt; // 2   memory_off − baseline
    bf16     retrieval_mass;     // 2
    bf16     drift;              // 2
    bf16     contradiction;      // 2
    bf16     peak_utility;       // 2
    bf16     sharpness;          // 2
    bf16     prefilter_score;    // 2
    bf16     reserved0;          // 2
    uint64_t timestamp_ns;       // 8
    uint32_t reserved1[2];       // 8
};
```

- Capacity = `N_slots × N_buckets_per_slot`. 4096 × 8 = 32 768 rows = 2 MiB.
  Comfortably fits in shared L3 on Sapphire Rapids/EMR; per-lane shards
  (~256 KiB each) fit in L2.
- Sharded by `slot_id mod N_lanes`. Lanes never write outside their shard,
  so the hot path is lock-free.
- AoS, not SoA: per-tile work is row-local (read all metrics, compute
  priority, write `prefilter_score`). One cache line per row keeps that loop
  vectorizable and prefetcher-friendly.
- **Residency target.** Per-lane shard is L2-resident; the active tile is
  L1-hot during processing. The lane's cue cache (below) shares L1 with the
  tile, so D and `T_PROBE_MAX` are sized so cue + tile fit in ~24 KiB of
  L1d, leaving headroom for scratch.

## The tile

A tile is K consecutive FCT rows, K chosen so K × 64 B ≤ L1d.

- Default `K = 64`. One tile = 4 KiB, leaves ~28 KiB of L1d for scratch and
  the lane's per-tick cue summary.
- Each lane processes a fixed number of tiles per engine tick
  (`tiles_per_tick`, default 4). Cost per lane per tick = 4 × 64 = 256 rows.
- Tile geometry maps cleanly onto AMX: 16-row × 32-col bf16 tiles for the
  bounded estimator path, with K split across two AMX outer iterations.

## The lane

```
Lane (one per pinned physical core):
    shard:        FCTRow[]   (this lane's slice of the FCT)
    cursor:       u32         (next tile index in the round-robin)
    job_inbox:    SPSC<TileTicket>           from engine
    work_outbox:  SPSC<SlotWorkItem>         to job ring
    packet_inbox: SPSC<EvidencePacket>       from GPU3 result ring
    amx:          AMXEstimator               cue-bounded NLL primitive
    cue_cache:    bf16[T_PROBE_MAX, D]       latest frame cue, L1-hot
    cue_nll:      f32[T_PROBE_MAX]           cached cue NLL profile
    ranker_w:     bf16[M_metrics, M_metrics] tiny learned ranker
    scratch:      aligned 4 KiB
```

Lane tick (fixed schedule):

1. **Drain `packet_inbox`** until empty or budget hit. Each packet writes one
   or more rows with `source=gpu_exact`, refreshes timestamp, clears stale
   flags. Bounded by ring depth.
2. **Advance the streaming cursor** by `tiles_per_tick`. For each tile:
   - decay metrics toward zero, increment `age_ticks` for each row,
   - run **AMX ranker matmul** once over the whole tile —
     `[K_TILE, M_metrics] @ ranker_w → [K_TILE, M_metrics]` followed by a
     dot with the cached `cue_nll` projection — to refresh
     `prefilter_score` for all 64 rows in a single AMX call,
   - for rows whose `prefilter_score` clears the dispatch threshold and
     whose `source` is stale or low-confidence, append a `SlotWorkItem` to
     `work_outbox`.
3. **AMX cue-NLL refresh** (only when the lane sees a fresh frame). On a
   frame-ingest signal, copy the cue summary into `cue_cache` and run
   `amx_bf16_nll(cue_cache, cue_targets, packed_head)` once. Result lives in
   `cue_nll[T_PROBE_MAX]` and is reused for every subsequent tile until the
   next frame replaces it. **No per-slot NLL.** Slot-specific NLL never
   happens on the lane; it is GPU3 territory.
4. **Yield.** No blocking primitives. Next tick comes from the engine.

The hidden-state provenance for AMX is therefore narrow and explicit: a
single per-frame `[T_PROBE_MAX, D]` cue digest, copied into pinned per-lane
memory at ingest. The lane never receives variant hidden states, never
receives per-slot hidden states, and has no API surface that accepts
arbitrary `(B, T, D)` tensors.

## The engine

Native C++ object owning the lane pthreads and the SPSC rings. Python sees
a thin pybind11 surface:

```python
class CpuEvidenceEngine:
    def ingest_frame(self, cue_hidden, cue_targets,
                     frame_id, step, stream_id) -> None: ...
    def slot_state(self, slot_id: int) -> SlotState: ...   # read-only view
    def diagnostics(self) -> dict[str, Any]: ...
    def shutdown(self) -> None: ...
```

There is no `evidence.score(...)`, no `advance()`, no `absorb_results()`,
no `drain_work()`. Lanes run continuously; they drain the GPU3 result ring
and push to the GPU3 job ring directly via SHM, with no Python on the hot
path.

Per maintenance tick on the Python side, the only evidence-engine work is:

```python
def tick(self, *, model, step):
    if frame_ready:
        self._evidence.ingest_frame(
            cue_hidden=frame.cue_hidden_pinned,
            cue_targets=frame.cue_targets_pinned,
            frame_id=frame.frame_id,
            step=step,
            stream_id=frame.stream_id,
        )
    # the controller reads slot state through self._evidence.slot_state(...)
    # to decide actions; no other coupling
    return TickResult(...)
```

The only way `cpu_estimate` rows enter the FCT is through the lane's tile
loop, which is bounded by geometry. The only way `gpu_exact` rows enter
is through the lane's drain of the GPU3 result ring, whose only producer
is the GPU3 oracle worker.

## CPU API surface — every entry point and its shape constraint

The promise that "CPU APIs cannot accept oracle-shaped work" is only as
strong as the enumeration. Below is the complete list of CPU-side entry
points after this change, with the shape each one enforces at the
type/binding level. Anything not on this list does not exist.

### Python-visible surface (pybind11 bindings)

| Entry point                                              | Argument shapes (enforced at binding)                                            | Output                              |
|----------------------------------------------------------|----------------------------------------------------------------------------------|-------------------------------------|
| `CpuEvidenceEngine.ingest_frame`                         | `cue_hidden: bf16[T_PROBE_MAX, D]`, `cue_targets: int32[T_PROBE_MAX]`, scalars   | None                                |
| `CpuEvidenceEngine.slot_state(slot_id)`                  | `slot_id: u32`                                                                   | `SlotState` (copy-by-value POD)     |
| `CpuEvidenceEngine.diagnostics()`                        | no args                                                                          | `dict`                              |
| `CpuEvidenceEngine.shutdown()`                           | no args                                                                          | None                                |

The pybind11 binding asserts the cue tensor shape (`shape == (T_PROBE_MAX,
D)`, `dtype == bf16`, `device == cpu`, `is_pinned()`) before copying. A
caller passing a 1024×512 hidden batch raises `ValueError` at the binding,
not deep in the kernel.

What is **not** on this list, and will not be added:

- No `score(hidden, targets)` of any shape.
- No `advance(rows: int)` or `advance(slots: list[int])` with caller-set
  shape.
- No `absorb(packets: list)` that accepts arbitrary Python objects.
- No exposure of `amx_bf16_nll` to Python. The kernel's pybind11 binding
  is removed by this change; `amx_bf16_nll` becomes a TU-internal symbol
  used only by `FctLane::cue_nll_refresh`.

### Internal C++ surface (TU-private after this change)

| Function                                                  | Sole caller                                                  | Shape constraint                                                         |
|-----------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------------------|
| `FctLane::tick()`                                         | the lane's pinned pthread main loop                          | no inputs                                                                |
| `FctLane::drain_packets()`                                | `FctLane::tick()`                                            | reads up to `MAX_PACKETS_PER_TICK` from the lane's SPSC packet ring      |
| `FctLane::process_tile(tile)`                             | `FctLane::tick()`                                            | `tile: FCTRow[K_TILE]` — `K_TILE` is a `constexpr` (compile-time fixed)  |
| `FctLane::cue_nll_refresh()`                              | `FctLane::tick()` only on a fresh-frame signal               | reads lane-private `cue_hidden_pinned[T_PROBE_MAX, D]`; no caller shape  |
| `amx_bf16_nll(hidden, targets, packed_head)`              | `FctLane::cue_nll_refresh()` only                            | `hidden: bf16[T_PROBE_MAX, D]`, `targets: int32[T_PROBE_MAX]` (asserted) |

`K_TILE` and `T_PROBE_MAX` are `constexpr` in `fct_row.h`. A caller asking
for a 1024-row tile or a 1024-token cue refresh fails to compile.

### What this guarantees

The full list of "places where CPU runs an NLL" after this change is **one
function**: `FctLane::cue_nll_refresh()`, called only from
`FctLane::tick()` on a fresh-frame signal, with shapes fixed at compile
time. The kernel function `amx_bf16_nll` is not callable from Python, not
callable from any other CPU module, and its only C++ caller is the lane.

A future Codex session asking "can the CPU score this oracle batch?" finds
no surface to attach to. That is the architecture preventing the class of
bug, not convention.

## Telemetry — starvation attribution

The hard requirement "GPU3 should not idle waiting on CPU" is only
enforceable if we can attribute idle time. Every diagnostics dump
includes a single `gpu3_starvation_reason` enum recording the first-hit
reason for the most recent observed-idle window, plus the per-reason
seconds counters.

Reasons:

| Reason                | Meaning                                                                  |
|-----------------------|--------------------------------------------------------------------------|
| `ok`                  | GPU3 has work and is consuming it; not idle                              |
| `no_slots`            | no slots are flagged across all FCT shards (dispatch criteria empty)     |
| `confidence_gate`     | candidates exist but all below the dispatch threshold                    |
| `frame_stale`         | no fresh cue has been ingested within `frame_ttl_steps`                  |
| `scheduler_behind`    | lane tile drops > 0 in the last interval (CPU couldn't keep cadence)     |
| `job_ring_empty`      | CPU produced no `SlotWorkItem`s in the last interval despite candidates  |
| `result_ring_full`    | GPU3 cannot write results because the lane hasn't drained                |
| `oracle_saturated`    | job ring backed up; GPU3 busy and CPU is producing faster than it drains |

Counters in `CpuEvidenceEngine.diagnostics()`:

```
lane_tile_drops_total
lane_tile_advances_total
lane_cue_nll_refreshes_total
lane_cue_nll_seconds_total
lane_packets_absorbed_total
lane_work_items_emitted_total
engine_job_ring_full_drops_total
engine_result_ring_drain_seconds_total
gpu3_idle_seconds_total
gpu3_idle_seconds_by_reason: dict[str, float]
```

These roll into the existing `replay_eviction.diagnostics()` payload so
shadow-mode runs can confirm the engine is healthy without going active.

## How the bug becomes structurally impossible

After the refactor:

- `chunked_nll_from_hidden(...)` is renamed `gpu_chunked_nll_from_hidden(...)`
  and no longer accepts a `cpu_scorer=...` argument. It is GPU-only at the
  type level.
- `oracle_confirm_slots(...)` and `oracle_confirm_refresh_candidates(...)`
  drop the `cpu_scorer=...` parameter. Their `backend` literal is
  `Literal["gpu_exact"]`. Adding a CPU literal would require explicit code
  change, not a config flip.
- `CpuMemoryScorer.score_hidden(arbitrary_shape)` is replaced by
  `AMXEstimator.estimate_tile(rows: ≤K, tokens: ≤T_PROBE_MAX, head: PackedHead)`.
  The signature itself rejects oracle-shaped work — passing more than `K`
  rows or more than `T_PROBE_MAX` tokens raises.
- The `cpu_scorer_backend` / `cpu_scorer_lanes` / `cpu_scorer_*` config knobs
  are removed. They are replaced by `evidence_engine_lanes`,
  `evidence_engine_tiles_per_tick`, `evidence_engine_estimate_budget`. The
  knob names cannot be repurposed for oracle scoring without code changes.

This is the central design rule: **CPU may decide what deserves oracle time;
GPU3 adjudicates physics.** The signatures enforce it.

## Bounds (the "fast enough to be invisible" contract)

The engine separates the **coordinator tick** (Python, synchronous, called
from `replay_eviction.tick`) from the **lane work** (native C++, fully
asynchronous, running in pinned pthreads). The coordinator's job is bounded
ring drains and pushes; lane work runs continuously at its own cadence.

| Dimension                       | Default                                |
|---------------------------------|----------------------------------------|
| Lanes                           | 8 (pinned P-cores, SMT off on lane set)|
| Tile size K                     | 64 rows / 4 KiB                        |
| Tiles per advance per lane      | 4                                      |
| Coordinator wall budget         | < 1 ms (ring ops only, no NLL)         |
| AMX cue-NLL cost per frame      | one `[T_PROBE_MAX × D] @ [V × D]^T`    |
| Per-frame AMX call shape        | 32 × 384 @ 16 384 × 384 → 32 × 16 384  |
| Per-frame AMX work              | ~1 ms per lane (compute-bound)         |
| Per-tile matmul cost            | `[K × M] @ [M × M]` on L1-resident data|
| Per-tile wall (steady state)    | ~150 ns (load/store dominated)         |
| Sustained advance rate per lane | ~10 M FCT rows/s in cache-warm steady  |
| Total advance rate, 8 lanes     | ~80 M rows/s peak; 1–10 M/s realistic  |
| FCT capacity                    | 32 768 rows total                      |
| Full-table revisit time         | < 100 ms even at conservative throughput|

The lane is dramatically faster than the FCT it has to scan, so steady-state
CPU on the evidence engine is small; the lane count is sized for **burst
absorption** (when frames ingest faster than usual or GPU3 result packets
arrive in a clump), not for keeping up with the tile loop.

The number that matters is not "wall time per coordinator tick" — that is
trivially small because the coordinator does no NLL. The number that matters
is **rows of evidence the engine can advance per second**. Because the lane
schedule is fixed (tiles_per_tick × K × tick_rate), this number is set by
geometry, not by call shape. The engine cannot be handed a 9 M-row job; it
has no API for it.

Earlier drafts of this document specified a `M_EST × T_PROBE_MAX` per-tile
NLL cap, which would have implied ~16 K NLL evaluations per coordinator tick
(~70 ms at 230 K rows/s). That cap is removed. The lane runs **one** AMX NLL
per frame (cue-only) and reuses the result across all subsequent tiles. NLL
is a per-frame cost, not a per-tile cost.

## Failure modes (graceful degradation)

1. **GPU3 saturated, result ring fills.** Lanes drain whatever fits; the rest
   waits. FCT rows just stay stale longer; `age_ticks` rises;
   `prefilter_score` reflects that and slots get re-prioritized. No blocking.
2. **CPU lane falls behind tick cadence.** Engine drops the oldest tile work
   for that lane (`tiles_advanced` < `tiles_requested`). Telemetry counter
   `tile_drops_total`. Trunk untouched.
3. **Job ring full.** Engine drops the lowest-priority `SlotWorkItem` rather
   than waiting; counter `work_drops_total`. GPU3 keeps draining what's
   already queued.
4. **AMX estimator fails or unavailable.** Lane disables AMX, falls back to
   metadata-only `prefilter_score`. There is no fallback into the trunk path,
   ever.

Every drop and every fallback emits telemetry. No silent absorption.

## What gets deleted, kept, added

**Delete.**
- `cpu_scorer=...` parameter from `chunked_nll_from_hidden`,
  `oracle_confirm_slots`, `oracle_confirm_refresh_candidates`.
- `_cpu_scorer_for(model, step)` and the three call sites that read it.
- `cpu_scorer_backend`, `cpu_scorer_lanes`, `cpu_scorer_vocab_tile_size`,
  `cpu_scorer_row_chunk_size`, `cpu_scorer_parallel_threshold_rows`,
  `cpu_scorer_weight_sync_interval_steps` config knobs and their telemetry
  fields.

**Keep.**
- The AMX BF16 NLL kernel itself (`_cpu_ssm_controller._ext.amx_bf16_nll`,
  `amx_pack_b_vnni`, `has_amx_bf16`, `amx_bf16_kernel_available`). Used by
  the lane's `cue-NLL refresh` — exactly one shape, exactly one caller.
- `CpuMemoryScorerWeights` (rename → `EvidenceWeights`). The weight snapshot
  bookkeeping is the same.
- The native `ArmMaintenanceScheduler`, `ArmMaintenanceJob`,
  `ArmMaintenanceResult`, and `ShmRing*` machinery — the engine uses them.
- The `FullAControllerState` and `MaintenancePolicy` action loop. The engine
  feeds them with cleaner inputs.
- `CpuRefreshProposalModel` (Role C). The proposal generator that the
  controller uses to produce refresh candidates remains as-is; the engine
  ranks and dispatches what the controller proposes, it does not generate
  proposals itself.

**Add.**
- `chaoscontrol/evidence/engine.py` — `CpuEvidenceEngine` Python wrapper.
  The only Python-side object `ReplayEvictionLoop` holds. Has no hot-path
  logic; just owns the lifetime and exposes
  `ingest_frame / slot_state / diagnostics / shutdown`.
- `chaoscontrol/oracle_worker/` — the Role A long-lived process. Entry
  point at `__main__.py`, model loader, ring drain loop, oracle physics
  dispatcher. Started/stopped by `runner_fast_path.py`.
- C++ in `_cpu_ssm_controller`:
  - `fct_row.h` — `FCTRow` POD, 64-byte cache-line aligned.
  - `fct_shard.h/.cpp` — per-lane shard, ring index helpers, AoS scan.
  - `fct_lane.h/.cpp` — pinned-pthread lane, owns drain + advance + AMX
    cue-NLL + per-tile ranker matmul.
  - `evidence_engine.h/.cpp` — coordinator that owns lanes, wires SPSC
    rings, exposes the pybind11 surface listed above.
  - `evidence_packet.h`, `slot_work_item.h` — POD types matching the
    shapes above; new SHM ring instantiations
    (`ShmRingEvidencePacket`, `ShmRingSlotWorkItem`,
    `ShmRingArmMaintenanceResult`).
- `chaoscontrol/evidence/__init__.py` — re-exports the Python wrapper.

**Rename.**
- `CpuMemoryScorer` → `AMXEstimator`. The new name carries the contract:
  it estimates, it does not produce truth.
- `chunked_nll_from_hidden` → `gpu_chunked_nll_from_hidden`. The new name
  forbids the CPU caller pattern.

## Alternatives considered

1. **Strict role split with the shared scoring API kept.** Rename callers,
   document that `cpu_scorer=...` must never be passed into oracle paths.
   Discipline-dependent; the next person who finds the hook will use it. The
   current bug *was* this approach. Rejected.

2. **CPU does only metadata-based prefilter, no scoring at all.** Simpler.
   Loses AMX as a controller-evidence tool. Gives up something we paid to
   build (the AMX BF16 kernel works). Rejected — the streaming FCT design
   keeps AMX for what it's good at (bounded dense scoring of small tiles)
   without letting it inherit oracle-shaped work.

3. **Streaming FCT (this design).** Type-level separation, cache-resident,
   tile-bounded, single-purpose lanes. Matches the project's existing native
   ARM scheduler / SHM ring infrastructure.

## Out of scope (for this design)

- Refresh candidate generation algorithm (FullA / simplex; lives in C).
- The action policy itself (MaintenancePolicy / values; lives in C).
- GPU3-side oracle physics (Role A; unchanged).
- Trunk-side training (Roles 0–2; unchanged).

## CPU budget on the deployment target (Xeon Platinum 8480+)

Single-socket Sapphire Rapids, 56 P-cores, AMX BF16 per core, 8-channel
DDR5-4800 (~300 GB/s memory bandwidth), 105 MiB shared L3, 2 MiB private
L2 per core. Existing concurrent CPU consumers (audited from the codebase
and project memories):

| Consumer                                                         | Cores  |
|------------------------------------------------------------------|--------|
| Python main, CUDA launch overhead, NCCL/DDP coordination         | 2      |
| Data loaders (DataLoader workers; SP-shard streaming)            | 4–6    |
| FullA controller + simplex policy + online learning (Exp 25)     | 4–6    |
| `CrctGradientConflictMonitor`, gradient sketches                 | 1–2    |
| `CpuRefreshProposalModel`, `MaintenancePolicy`, action evidence  | 1–2    |
| `ScarcityAwareMemoryOptimizer.dual_step`                         | <1     |
| Replay eviction Python orchestration, telemetry, trace flush     | 1–2    |
| ARM scheduler, SHM ring drains                                   | <1     |
| GPU3 oracle worker process (CPU side: ring drain + frame setup)  | 1–2    |
| **Existing baseline (no evidence engine)**                       | **15–22** |

Reserved: 4 cores idle for system / kernel / NCCL bursts. Lane allocation
chooses from the remaining 30–37 cores.

Lane sizing is set by **burst absorption**, not steady throughput:

- Steady AMX work per lane = one cue-NLL per frame ingest (~1 ms) +
  per-tile matmul (~150 ns × tiles_per_advance). At 10 frames/s ingest,
  CPU duty is ~1.5 % per lane.
- Burst absorption: when GPU3 returns a clump of result packets and a new
  frame ingests in the same window, the lane has to drain packets, run
  cue-NLL, and advance several tiles back-to-back. 8 lanes processes that
  burst in ~2 ms wall time without queuing.

**Default: 8 lanes.** Configurable; 4 is the lower bound (loses burst
margin), 16 is the upper bound (no measurable benefit on this workload).
The lanes pin to P-cores 0–7 with SMT disabled on those cores; the rest
of the socket keeps SMT enabled.

## Closed design choices

(These were open questions in earlier drafts; resolved by the author and
recorded here as decisions.)

1. **Lane count: 8.** Calculated above against the 8480+ baseline; 25 %
   of the socket, leaving ~30 cores headroom.
2. **C++-everywhere production path.** The lane, the FCT, the SPSC rings,
   and the engine coordinator all live in `_cpu_ssm_controller` as native
   C++ behind a thin pybind11 surface. The Python wrapper class exists
   only to expose `ingest_frame`, `slot_state`, `diagnostics`, `shutdown`
   — it owns no hot-path logic. Lanes drain the GPU3 result ring and push
   to the GPU3 job ring directly via SHM; Python is on the path only for
   frame ingest and read-only slot state lookups. Python is allowed in
   tests; not in production tick paths.
3. **Async GPU3 oracle, no staging.** The synchronous `_select_native_job`
   `push(job)` → `pop(worker_ring)` pattern is removed in this work, not
   later. The implementation includes the long-lived GPU3 oracle worker
   process (Role A) and converts the oracle path to fully async-via-rings
   in the same change.

## The GPU3 oracle worker (Role A) — added by this design

A long-lived worker process pinned to GPU3 (`cuda:3`) that:

- Attaches to the job ring (`ShmRingArmMaintenanceJob`) as a consumer.
- Holds a model handle on `cuda:3` and the SP-tokenized cache structures.
- Pops `ArmMaintenanceJob`s, runs the exact `force_on / memory_off /
  hide-slot / refresh-candidate` physics, and writes
  `ArmMaintenanceResult`s to a new result ring
  (`ShmRingArmMaintenanceResult`, with the same SHM-ring semantics).
- Has a graceful shutdown path triggered by a sentinel job or SIGTERM.

The worker is started by the runner (`runner_fast_path.py`) before training
begins and stopped after the run finishes. It does not share Python state
with the training process; it talks to the rest of the system only through
the shared-memory rings.

`replay_eviction.py` no longer pops from any worker ring. Lanes drain the
result ring directly via SHM in their pthread loop; that drain feeds the
FCT. The feedback loop closes through the FCT itself: tick `N` dispatches
a `SlotWorkItem` (lane → job ring directly), GPU3 produces a result some
`k` ticks later, the lane absorbs it on its next pthread iteration, the
controller sees fresh evidence on tick `N + k + 1` via `slot_state(...)`.
There is no synchronous wait point on the trunk side and no Python on
the result-drain path.

## Open questions for the author

(All material design choices are now closed. Anything below is calibration,
not architecture.)

- Default `T_PROBE_MAX` for cue-NLL refresh: 32 fits comfortably in L1 with
  D=384. If we move to D=512 we should drop to T_PROBE_MAX=24.
- Whether to expose lane count as a per-experiment knob or a per-pod knob.
  Recommendation: per-pod, set once in runner config, not in `exp26.py`.
