# Memory-Aware Optimizer — Design

**Date:** 2026-04-25
**Status:** Design draft. Not on Exp24 critical path; Exp25+ territory once ScOpt base is calibrated and Criticality Distillation has either landed or been parked.
**Depends on:** `docs/plans/2026-04-22-scarcity-optimizer-design.md` (ScOpt base), `docs/plans/2026-04-24-criticality-distillation.md` (the recurrence-state telemetry path this design reuses), `docs/plans/2026-04-24-dual-timescale-rare-memory.md` (the rare-gradient EMA timescale this design's hints derive from).

> **Section structure note.** This document answers nine numbered design questions. Sections #6 (IPC) and #9 (failure modes) are pinned by the refinement that arrived 2026-04-25. The remaining seven are a working decomposition of the "memory-aware optimizer" architecture as described in that refinement plus the ScOpt thesis; they are not a transcription of an external spec.

## Thesis (#1)

ScOpt and Criticality Distillation answer "where is rare signal trying to flow on this batch?" They do not answer "what rare events did this model see two thousand steps ago that it has since smoothed away?" The recurrence integrates; the optimizer EMAs decay; the rare-grad EMA's effective window is at most ~100 split steps even at the slow timescale. Rare events that fired more than a few thousand steps back are, for practical purposes, gone.

A *memory-aware* optimizer maintains a separate store — outside the SSM's recurrent state, outside the optimizer's EMA buffers — of rare-event evidence with a lifetime measured in the whole training run, not in EMA windows. The optimizer reads from this store when it computes its update direction. A CPU-side controller curates the store: deciding what to keep, what to consolidate, what to evict, and what to re-project as the backbone drifts under it.

The biological analogy holds at the architecture level: the GPU optimizer plays the role of the awake cortex (fast, online, narrow-window), and the CPU controller plays the role of the sleeping hippocampus (slow, offline, replay-driven, autonomous). Hippocampal replay during sleep is autonomous but biased toward circuits that were recently active. We mirror that here: the controller has its own loop and its own loss function, but its work-selection prior is informed by attention hints the optimizer posts during training.

We make no claim that this beats Muon at 10M parameters on the 600s budget. The thesis is that *if* Criticality Distillation's "channel-survival evidence outlives the rare-grad EMA" intuition is correct, then a longer-lived store buys more of the same effect, and the bottleneck becomes managing that store coherently rather than producing the evidence in the first place.

## Architecture (#2)

Three components, each with one responsibility.

**GPU optimizer (the fast process).** Lives where ScOpt lives — inside the train step. Per split step, it produces the same per-token pressure, per-channel scarcity, and rare-grad evidence ScOpt produces today. New responsibilities over base ScOpt: (a) on a write trigger, package an "append event" message and post it to the controller; (b) on a read, look up entries from a small read-side mirror of the store; (c) post attention hints to the controller (see #6). The optimizer never blocks on the controller — every IPC operation is non-blocking, and the read path tolerates a stale mirror.

**CPU controller (the slow process).** Runs in a separate Python process per DDP rank, communicating with the optimizer over shared memory and a small bounded message queue. Its loop is autonomous: drain pending optimizer messages, advance its own work scheduler, run one or more curation steps (re-projection, eviction, consolidation), publish the new mirror of the store back to the optimizer. The controller has its own internal loss function — channel-survival reconstruction error against fresh evidence — and uses the optimizer's attention hints to *prioritize* which entries to spend its compute budget on, not to dictate what it does with them.

**The store.** A bounded, addressable cache of rare-event evidence. Each entry is a fixed-shape record (see #3). The store is materialized in pinned host memory; the optimizer accesses it through a read-side mirror that the controller refreshes on its own cadence. The store is rank-local — DDP ranks each maintain their own controller and their own store, with periodic rank-0 checkpointing only.

The optimizer cannot brute-force-evaluate the whole cache every cycle, and we explicitly do not require it to. The controller's autonomy plus the attention-hint protocol exists precisely so that exhaustive scans never have to happen on either side.

```
                                                          rank-local
   ┌──────────────────────────────────────┐  shared mem  ┌──────────────────────┐
   │  GPU optimizer (every split step)    │ ───────────▶ │  CPU controller      │
   │  - ScOpt update direction            │   messages   │  - work scheduler    │
   │  - read mirror lookup                │ ◀─────────── │  - re-project / evict│
   │  - posts append/drift/feedback/hints │  ack/state   │  - consolidate       │
   │  - reads mirror (no block)           │              │  - publishes mirror  │
   └──────────────────────────────────────┘              └──────────────────────┘
                       ▲                                              │
                       │                                              ▼
                       │                                       ┌──────────────┐
                       └────── refreshed mirror ───────────────│  store       │
                                                               │  bounded     │
                                                               │  pinned host │
                                                               └──────────────┘
```

## Cache contents and lifecycle (#3)

A store entry is a fixed-shape record. Indicative schema (V1):

| Field            | Shape          | Dtype | Notes                                                           |
| ---------------- | -------------- | ----- | --------------------------------------------------------------- |
| `evidence`       | `[L, D]`       | fp16  | Per-layer channel-survival evidence; same units as the Trace Bank in Criticality Distillation. |
| `target_token`   | scalar         | i32   | Token ID associated with the rare event, or `-1` if none.       |
| `pressure`       | scalar         | fp16  | Pressure value at write time.                                   |
| `step_written`   | scalar         | i32   | Training step on which this entry was created.                  |
| `step_last_used` | scalar         | i32   | Last step the optimizer read this entry.                        |
| `utility_ema`    | scalar         | fp16  | Retrieval-utility EMA; updated on every optimizer-side read.    |
| `version`        | scalar         | i32   | Bumped by the controller every time it re-projects this entry.  |

At V=16384, L=4, D=256 the record is ~2KB. A 4096-entry cache is ~8MB per rank. Rank-local; not shipped in the artifact.

**Lifecycle.**

- *Append.* Triggered on the GPU side when an event's pressure exceeds a quantile-based threshold *and* the in-flight backbone has produced credible channel-survival evidence (Criticality Distillation's Stage 3 output for that step). The optimizer posts an append-event message; the controller is the sole writer to the store. There is no synchronous handshake — the optimizer never waits for the entry to land in the mirror.
- *Re-projection.* Backbone weights drift continuously. An entry's `evidence` vector was meaningful relative to the backbone at `step_written`. The controller's primary job is to maintain re-projection of stale entries against the current backbone, prioritized by attention hints (see #6).
- *Eviction.* Bounded store, so something has to leave. The controller picks evictions using its own scoring function — typically `score = α·age + β·(1 - utility_ema) + γ·redundancy(entry)` — but always at the controller's discretion. The optimizer's attention hints can flag entries as "currently relevant," which biases the controller *away* from evicting them, but does not pin them.
- *Consolidation.* Multiple entries that point to the same channel structure get merged. The controller decides when to consolidate; attention hints flag clusters as candidates.
- *Read.* The optimizer reads from a read-side mirror (lock-free, double-buffered). On a read, the optimizer increments a per-entry hit counter in shared memory; the controller folds those into `utility_ema` on its next loop iteration.

The lifecycle is asymmetric on purpose: writes are buffered (the store may lag the optimizer by tens to hundreds of steps), reads are immediate (against whatever mirror is current). Stale reads are tolerable; missed writes are not.

## GPU optimizer side — the read path (#4)

ScOpt's update rule (Step 5 in `2026-04-22-scarcity-optimizer-design.md`) gains one term:

```
direction = common + β · r_orth + γ · m_proj
```

where `m_proj` is the projection onto parameter space of the relevant store entries' `evidence`. Concretely, on the GPU side per split step:

1. Compute pressure, common gradient, rare gradient, and rare-channel pressure as ScOpt does today.
2. Use the rare-channel-pressure vector as a *query* against the read mirror. The query is a top-K nearest-neighbour lookup over `evidence` aggregated to the per-layer channel axis. K is small (default 16). Query cost is O(K · L · D); fits in a few µs at the cache sizes proposed.
3. Aggregate the retrieved entries into `m_proj` per layer, weighted by their `utility_ema` and a freshness term `exp(-(step - step_last_used) / τ)`.
4. Bumps go in shared memory (per-entry hit counter, retrieval-feedback message in #6).
5. Write the new `direction`, fall through to ScOpt's per-role geometric integration (Step 6) unchanged.

The new term is `γ · m_proj`. Default `γ = 0` at warmup, ramped to a small value (0.1–0.3) once the cache has populated. This keeps the memory-aware path off the load-bearing critical path until the store is non-degenerate.

`γ = 0` collapses the design to base ScOpt bitwise — the back-compat lever.

## CPU controller side — autonomous curation loop (#5)

The controller runs an unbounded loop bounded only by wall-clock budget per iteration. One iteration:

1. Drain the optimizer→controller queue (bounded; #6). Apply append events, fold drift summaries into a per-channel drift budget, fold retrieval-feedback into per-entry `utility_ema`, fold attention hints into the work-selection prior.
2. Run the work scheduler. The scheduler maintains a priority over store entries; the priority is the controller's own composite of `attention_hint_score`, `staleness`, `drift_pressure`, and `utility_ema`. The optimizer's hint contributes one term among several. The controller is free to demote a hinted entry if other terms outweigh it.
3. Pop one or more entries off the priority queue and run a curation step: re-project against the current backbone snapshot, evict, consolidate, or no-op. The controller's own loss is the post-curation reconstruction error of `evidence` against fresh per-channel survival evidence sampled from the optimizer's outbound stream.
4. Optionally rebuild the read mirror. The mirror is double-buffered: write into the inactive buffer, atomically flip the active pointer. The optimizer's reads always see one consistent snapshot.
5. Send any controller→optimizer messages (cache-state summary, eviction notifications; #6).

Compute budget per iteration is set so the controller's wall-clock cost is comfortably below the GPU step time at 1×H100. At 8×H100 the per-rank budget is the same; the controllers don't talk to each other.

The controller has no gradient. Its loss is offline reconstruction error; the only "optimizer" it runs is a small linear re-projection of `evidence` vectors using the current backbone's projection weights, fetched via a read-only shared-memory snapshot the GPU side publishes every N steps.

## IPC (#6) — bidirectional message protocol

The IPC layer is the load-bearing structural choice of this design. The optimizer is fast and narrow-window; the controller is slow and wide-window; the protocol is the place where their semantics meet without the optimizer blocking and without the controller pretending to be exhaustive.

### Optimizer → controller messages

Posted from the GPU side at the end of each split step (or each step, for the lighter-weight messages). All non-blocking; all bounded-queue with controller-side drop-on-overflow plus a counter.

- **Append events.** New write candidates. Payload: `{layer_idx, target_token, pressure, evidence_handle, step}`. `evidence_handle` is a pinned-memory pointer the controller copies out of before the GPU side overwrites the slot. Frequency: bounded by `event_frac · split_steps_per_run`, typically a few thousand events per run per rank.
- **Backbone drift summaries.** What channels moved this cycle, and by how much. Payload: `{layer_idx, drift_norm[D], step}`, sent every N steps (default N = 32). Used by the controller to rank entries by re-projection urgency without having to recompute each entry's drift directly.
- **Retrieval feedback.** Entry E was retrieved at step T with utility U. Payload: `{entry_id, step, utility}`. Frequency: one per retrieval, but folded into a per-step batch on the GPU side before posting (typical batch size ~K ≤ 16, so a few hundred per second). The controller folds U into `utility_ema`.
- **Channel-attention hints.** Channels with heavy gradient updates this cycle. Payload: `{layer_idx, channel_attention[D], step}`, sent every N steps (matched to drift cadence). Derived from rare-channel pressure plus a fast EMA of `‖rare_grads‖` per channel. The controller uses this to bias work-selection toward entries whose `evidence` vector has heavy mass on currently-active channels — same biological principle as hippocampal replay biased toward recently-active circuits.

### Controller → optimizer messages

Posted lazily; the optimizer reads them at convenient points (typically once per split step, on the same cadence as the read-mirror flip).

- **Cache state summary.** `{count, count_dirty, fraction_stale, fraction_high_drift, mirror_version}`. Lets the optimizer adapt: if the cache is mostly dirty, drop `γ`; if it's mostly fresh, raise `γ`.
- **Eviction notifications.** `{evicted_entry_ids[], step}`. The optimizer uses this to clear any handle it might still hold from a prior retrieval and to update its in-step bookkeeping.

### Autonomy contract

The controller decides *how* to handle every entry in its work queue: re-project, evict, consolidate, leave alone. The optimizer decides *which* entries deserve attention via channel-attention hints, retrieval feedback, and append events. The controller is still running its own loop with its own loss function; its work-selection prior is informed by optimizer hints rather than being uniform-random or exhaustive.

This is the pattern of a database query optimizer receiving hints, a GC write barrier informing the collector about generational pressure, or a branch predictor receiving training events from retired instructions. Fast processes post attention hints to slow processes so slow processes don't scan exhaustively. We are not inventing this pattern; we are applying it to a memory cache for an optimizer.

### Transport details

- Shared memory backed by `/dev/shm`. POSIX shm + a small lock-free SPSC ring buffer per direction per rank. Python-side accessor uses `multiprocessing.shared_memory.SharedMemory` plus a `numpy` view over the buffer; controller is a separate process per rank.
- Bounded queue sizes: 64K append-event slots, 4K drift summaries, 64K retrieval-feedback slots, 4K hint slots. On overflow, the oldest message in that lane is dropped and a `dropped_<lane>_count` counter is bumped — never blocks the optimizer.
- One mirror version per controller cycle. Mirror flip is a single 8-byte CAS on the active-pointer slot.

## Integration with ScOpt (#7)

The memory-aware path is a strict extension of base ScOpt. Reusing existing pieces:

- *Pressure, rare gradient, rare-channel pressure, rare-grad EMA.* All produced as today (single-timescale or dual-timescale per `2026-04-24-dual-timescale-rare-memory.md`). The dual-timescale design already gives us a `slow` branch with a ~100-sample window; the memory-aware path extends that window to whole-run scale by externalizing the state.
- *Recurrence-state capture API.* Already a prerequisite for Criticality Distillation. The same `capture_states: bool` hook produces the channel-survival evidence the store stores. We do not introduce a second capture path.
- *Update rule.* Adds `γ · m_proj` to the direction (#4). With `γ = 0`, base ScOpt is recovered bitwise.
- *DDP all-reduce contract.* The memory-aware term `m_proj` is rank-local (each rank has its own store). To keep the optimizer step deterministic across ranks, `m_proj` is *all-reduced* like every other scarcity-derived quantity in the ScOpt step. This is a design tradeoff: fully rank-local stores would diverge faster across ranks; all-reducing `m_proj` keeps the optimizer ranks coherent at the cost of one extra collective per split step.

Knobs added under existing ScOpt config group:

```yaml
scopt_memory_enable: false                   # default off; off-by-default rule per ScOpt back-compat
scopt_memory_cache_entries: 4096
scopt_memory_gamma: 0.0                      # set 0.1–0.3 once cache is non-degenerate
scopt_memory_gamma_warmup_steps: 1000
scopt_memory_query_topk: 16
scopt_memory_controller_budget_ms: 50        # per-iteration wall-clock cap
scopt_memory_drift_cadence_steps: 32
scopt_memory_hint_cadence_steps: 32
```

## Engineering and DDP semantics (#8)

- **Process model.** One controller process per DDP rank, launched at runner init via `multiprocessing.get_context("spawn")`. Death of the controller bumps a status flag the optimizer polls; the optimizer falls back to `γ = 0` and continues. Never crashes the train step.
- **Pinned memory ownership.** The store and the read mirror live in pinned host memory allocated by the controller and shared with the GPU process. The GPU side reads through a `numpy`/`torch.from_numpy` view; writes only happen through the IPC ring (the GPU side never touches the store directly).
- **Backbone snapshot for re-projection.** Every M steps (default M = 1024), the GPU side publishes a read-only snapshot of the projection weights the controller uses for re-projection. Snapshot is small (a few MB); cost is negligible.
- **DDP semantics.** The store itself is rank-local; `m_proj` is all-reduced in the same bucket as `rare_grads`. Cache-state summaries and eviction notifications are rank-local — no cross-rank coordination on cache curation. This means rank-0's controller may consolidate an entry that rank-1's controller has already evicted; that is fine as long as the optimizer's `m_proj` is reduced.
- **Checkpointing.** Rank-0's store is checkpointed alongside the model state every C steps (default C = 8192), as a single file under the run dir. Other ranks' stores are not checkpointed (they will repopulate from new appends on resume). Checkpointing is the controller's job, not the optimizer's.
- **Throughput contract.** The Tier -1 overhead gate from the ScOpt design extends here. We add: (a) end-to-end tokens/sec must remain ≥ 0.75× base ScOpt; (b) the controller's per-iteration budget must fit comfortably under the optimizer's split-step wall clock; (c) shared-memory IPC must not be a measurable fraction of optimizer step time. If any of these fail, drop to `γ = 0` (no memory-aware term) or park the design.
- **Determinism.** With `γ = 0` and the memory-aware path disabled, the runner is bitwise identical to base ScOpt. With `γ > 0`, the runner is deterministic only in the limit where mirror-version, append latency, and eviction order are the same across runs. We do not promise bit-exact determinism for the memory-aware path; we promise statistical equivalence under the noise-aware decision gate.

## Failure modes (#9)

The memory-aware path has more moving parts than base ScOpt and accordingly more ways to fail. The list below is not speculative; each item is a concrete failure mechanism with a specific symptom and a specific check. The new failure mode introduced by the IPC refinement — attention hints saturating or misleading the controller — is called out as **NEW**.

1. **Cache populated with stale entries; `m_proj` consistently points away from the rare direction.** Symptom: bpb regression scales with `γ`. Check: cosine between `m_proj` and `r_orth` per-layer; if median < 0, the cache is anti-correlated with the live signal. Mitigation: shorter re-projection horizon, more aggressive controller budget, stricter `utility_ema` floor for retention.
2. **Controller falls behind the optimizer; mirror persistently stale.** Symptom: `fraction_stale` from the cache-state summary climbs above 0.5 and stays there. Mitigation: drop `γ` to 0 automatically when `fraction_stale > 0.5`; the runner falls back to base ScOpt without intervention. This is the same auto-disable pattern Exp24 uses for fast/slow when slow EMA is undertrained.
3. **Store overflows; controller drops appends.** Symptom: `dropped_append_count` non-zero. Either the store capacity is too small or the optimizer is over-firing. Mitigation: raise capacity (within the rank-local memory budget), tighten the pressure quantile that triggers writes.
4. **NEW: optimizer attention hints saturate the controller's work scheduler.** When the channel-attention vector is concentrated on a small number of channels (e.g. one or two SSM layers dominate), the controller's hint-driven priority will repeatedly select entries from those channels, starving entries from less-active channels. Eventually those starved entries become uniformly stale and get evicted en masse, after which the cache is effectively a small-channel-cluster cache and no longer reflects the historical event distribution. Symptom: histogram of `evidence`'s active channels collapses over training; entries-per-layer distribution becomes peaky. Check: log per-layer entry count and per-channel `evidence` mass distribution; alert if any layer's share rises above a threshold (default 60%). Mitigation: cap the hint-derived priority term at a fraction of the total priority (default 0.3), keeping the hints as one of several inputs rather than the dominant one. The controller's autonomy is the structural defense: the controller is free to weight its own staleness/drift terms above the hint, and the cap enforces this. Falsifier for this failure mode: holding `γ` constant and disabling hints, the cache-state telemetry should not drift toward the same channel-collapse pattern. If it does, the failure has a different cause.
5. **NEW: optimizer attention hints mislead the controller into prioritizing entries the optimizer doesn't actually retrieve.** Hints reflect *current gradient activity*, not *retrieval pattern*. If the model's heavily-updated channels in some training phase are not the channels whose historical evidence the optimizer queries, the controller spends its budget re-projecting entries that no read path ever uses. The structural defense is partial: per #4 and #6, the read query and the channel-attention hint share the same source (rare-channel pressure), so divergence is bounded to the residual reweighting that `utility_ema` and the freshness term apply on the read side. The failure mode lives in that residual — when `utility_ema` and freshness pull retrieval toward a different subset than channel-attention pulls re-projection toward. Symptom: high re-projection rate combined with low retrieval count for the same entries. Check: cross-correlation between `version` (re-projections) and `step_last_used` (retrievals) per entry; if those are uncorrelated, the controller is mistuned. Mitigation: weight retrieval-feedback higher than channel-attention in the controller's priority; this is the per-class attention-hint hierarchy (retrieval > drift > channel) rather than treating them as equal sources.
6. **DDP all-reduce of `m_proj` becomes a step-time bottleneck.** Symptom: 8×H100 throughput regression vs 1×H100. Check: collective time in the Tier -1 profile. Mitigation: bucket the `m_proj` reduce with the `rare_grads` reduce so it is one collective; if still problematic, drop `m_proj` from the all-reduce and accept rank-local divergence (different design tradeoff).
7. **Controller crash during a long run.** Symptom: status flag flips to dead; `γ` auto-clamps to 0. The runner continues as base ScOpt. The check is the auto-disable; the failure mode is the crash itself, which we treat as "park and investigate" rather than "retry," since silently restarting the controller mid-run risks an inconsistent cache.
8. **Pressure quantile drift over training.** As the model improves, the absolute pressure threshold for "rare event" shifts; the cache's contents drift toward whatever the model is currently bad at. Symptom: `target_token` distribution in the cache shifts with training step. Mitigation: this may be desirable behavior (the cache tracks the live frontier of rare-event difficulty); only treat as a failure mode if it correlates with bpb regression in the Tier 1 placebo.
9. **Determinism break on resume.** The cache is not bit-exact across resumes (mirror flip ordering, eviction ordering). Symptom: bpb-after-resume diverges from bpb-from-scratch by more than the noise-aware decision gate (`δ_bpb = 0.005`). Mitigation: anchor the controller to a deterministic seed for tie-breaking in eviction and consolidation; accept that the determinism is statistical, not bit-exact.

The first three plus item 7 are ScOpt-class failure modes generalized to a longer-lived store. Items 4 and 5 are new: they are the specific failure modes the attention-hint protocol introduces. Items 6, 8, and 9 are engineering footnotes that need empirical confirmation rather than mitigation up front.

## Open questions

1. **`γ` schedule.** Default ramp from 0 over `gamma_warmup_steps`. Right schedule probably depends on cache population rate; sweep once Tier -1 lands.
2. **Query semantics.** Top-K nearest-neighbour over per-layer channel axes is the obvious choice; an attention-style softmax read is the alternative. Both produce `m_proj`; the softmax form is differentiable but doesn't add anything since we don't backprop through `m_proj` anyway. Default top-K.
3. **Per-rank vs cross-rank store.** Default rank-local + all-reduced `m_proj`. Cross-rank consolidation (one shared store) is a v2 question; deferred until rank-local proves at all.
4. **Controller process count.** One per rank seems right; one global controller is also possible (NUMA, shared memory across ranks on the same node). Default one-per-rank for cleanliness.
5. **Hint composition.** The refinement message lists four optimizer→controller message types: append events, drift summaries, retrieval feedback, channel-attention hints. They are not all the same kind of signal — append events are state changes, drift and channel are gradient-derived, retrieval is read-pattern. The controller's priority function should weight them per-class, not as a uniform sum. Default weights pinned in #9 mitigation 5.

## Scope discipline

- **Not** an Exp24 first-wave arm. First-wave matrix (fast_slow, spectral, predictive_aux, scheduled dreamworld, dreamworld event_sleep) ships first.
- **Not** a replacement for Criticality Distillation. The memory-aware optimizer is a strict superset: it reuses CD's recurrence-state telemetry and its trace bank shape, just on a longer timescale and with an external curator. If CD is parked, this design is parked too.
- **Not** an architectural-law claim. Same scope as ScOpt: "our SSM at this scale on this benchmark." Transformer comparison is a follow-up.
- **Not** the dual-timescale rare-EMA design. Dual-timescale extends the within-optimizer EMA window from 10 samples to 100; memory-aware extends it from 100 samples to whole-run. These compose: the dual-timescale slow branch is what feeds the cache.

## References

- Internal: `docs/plans/2026-04-22-scarcity-optimizer-design.md`, `docs/plans/2026-04-24-criticality-distillation.md`, `docs/plans/2026-04-24-dual-timescale-rare-memory.md`.
- Internal memory: `project_scarcity_optimizer_thesis.md`, `feedback_risks_not_implementation_challenges.md` (failure-mode framing in #9), `feedback_wiring_vs_existence.md` (the controller-IPC layer needs the wiring proof, not just the existence proof).
- Engineering precedents for the hint-protocol pattern: database query optimizer hints, GC write barriers, branch predictor training events. The biological precedent is hippocampal replay biased toward recently-active circuits — autonomous loop, peripheral signals shape priority.
