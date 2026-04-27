# Memory-Aware Optimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the curated-Dreamworld memory-aware architecture: an episodic cache on a dedicated GPU feeds Dreamworld replay via a CPU controller. Validated against the existing falsifier — uncurated Dreamworld did NOT transfer to artifact (`phase0_dw_sweep`, val_bpb 1.484, lost to fast_slow-only at 1.479) — and the prediction is that curation flips this.

---

## Thesis selection (load-bearing — read first)

The 2026-04-25 design doc (`docs/plans/2026-04-25-memory-aware-optimizer-design.md`) and this plan describe two **distinct cousin mechanisms**, not the same architecture:

| | **Thesis A: m_proj (design doc)** | **Thesis B: curated DW (this plan)** |
|---|---|---|
| Where cache plugs in | Optimizer's update rule | Backward pass |
| What's stored | Channel-survival evidence vectors | Token-ID spans + late-residual key |
| Read mechanism | Direct projection into param space | Cosine retrieval → tag for replay |
| Reaches weights via | `direction = common + β·r_orth + γ·m_proj` | Replay backward → `param.grad` → all-reduce → optimizer |
| Forward-pass effect | None | None |

**This plan implements Thesis B.** Conversation evolution on 2026-04-25 (residual stream as cache key, controller-driven querying, Dreamworld as consolidation engine) selected curated-DW over m_proj. Thesis A remains a documented, coherent alternative; if Phase 3 falsifies curated-DW, m_proj is the natural next thing to try, not a vague pivot.

The two are non-mutually-exclusive in principle — both could coexist in a future design — but conflating them produces incoherent specs. We pick one and ship it.

---

## Architecture (Thesis B, locked 2026-04-25)

- **Train GPUs (3 of 4 in test, 6 of 8 in prod).** Run the SSM forward + main backward as today. Two new responsibilities, both cheap and synchronous in the train step: (a) `select_writes` payloads get dropped into a per-rank SPSC write ring, (b) the **same** top-p positions (already selected by pressure × CE for writes) also publish `(b, t, residual[D])` tuples to a per-rank query-candidate SPSC ring. One source signal, two consumers (cache for writes, controller for queries). The SSM forward stays uninstrumented.
- **Episodic GPU as DDP rank N−1 (rank 3 in test, ranks 6 + 7 in prod).** Same torchrun process group, same model copy, same all-reduce. Holds the cache. Drains write rings. Serves cosine-similarity queries to the controller. Runs Dreamworld replay forward+backward on tagged entries; gradients accumulate into the standard `param.grad` and flow through the existing all-reduce path.
- **CPU controller (one separate child process per node).** Drains the per-rank query-candidate rings. Schedules cache queries on the episodic GPU. Schedules Dreamworld replay on tagged entries. Owns its own loop, its own wall-clock budget, never blocks the SSM. Touches no gradients, no GPU memory directly — its plane is control + analytics.
- **Cache read mechanism: utility-weighted cosine.** `score = cosine_sim(query_residual, cache.key_rep[i]) × cache.utility_u[i]`. Two GPU vector ops per candidate. No learned predictor in the read path.
- **No forward-time injection.** Cache content reaches the SSM only via Dreamworld replay gradients. Val time has no cache access.

### All-reduce topology with the asymmetric split

The 3+1 split has different ranks contributing different gradient sources, and the previous "two-collective" framing didn't compose cleanly (AVG over groups can't reconcile main-only and replay-only contributions on the same params). Replaced with **a single SUM all-reduce per step over the all-rank group, with pre-scaled main gradients**:

1. **Train ranks (0..N-2)** run main backward. Before the collective, they multiply their main gradients by `1 / (N-1)` so the upcoming SUM gives the main *average* over train ranks.
2. **Episodic rank (N-1)** runs replay backward (or materializes explicit zero grads if no replay item is available this step — see Decision 0.10 for the unconditional-replay-collective rule). No pre-scaling on replay grads — they enter the SUM at full magnitude.
3. **All N ranks** call `allreduce_grads(model, group=all_group, materialize_zeros=True)`. The collective is `op=SUM`. Output on every rank: `(main_avg) + (replay_full)` for params touched by both; `main_avg` for main-only params; `replay_full` for replay-only params.
4. Optimizer step proceeds on every rank with the combined gradient.

**This is a deliberate scaling asymmetry, not a bug.** Main contributions land at average scale (`/(N-1)`); replay contributions land at full magnitude. Replay weight is therefore controlled by `dreamworld_weight` in the existing config, applied at replay-backward time. If Phase 3 results suggest replay grads should be down-scaled further, drop `dreamworld_weight`; don't try to fold scaling into the collective.

**Footnote — Adam β₂ × replay_full timing.** On parameters touched by both main and replay (shared layers: embedding, final norm, lm_head, and any layer the SSM trunk touches that replay also touches), the optimizer's second-moment estimate becomes the moving average of `(main_avg + replay_full)²` rather than `main²` alone. With Adam-style optimizers `β₂ ≈ 0.999`, the variance estimate adapts on a slow timescale. Phase 1-3 are safe because (a) `dreamworld_weight` is constant from t=0 — the optimizer adapts ONCE during the cache-fill warm-up and is stable thereafter, and (b) the warm-up is symmetric across all four falsifier arms (cache starts empty in all of them). **If Phase 5+ ever schedules `dreamworld_weight` (e.g., a ramp from 0 to a target value):** the schedule must be slow relative to β₂ — ~10 × β₂'s effective window, so ~5000-10000 steps for the ramp — or the second-moment estimate gets perpetually mis-calibrated on shared params. Add a `dreamworld_weight_warmup_steps` analog to Thesis A's `gamma_warmup_steps` if scheduling lands.

**Why a single SUM instead of two AVG collectives:** AVG-then-AVG over different subsets double-counts the first AVG result. SUM with pre-scaling gives full control and a single round-trip. NCCL subgroups are still useful for *non-gradient* coordination (heartbeats, status flags) but not for the main gradient path.

---

## Reference docs

- Design (note: 2026-04-25 conversation supersedes the doc on storage-as-token-IDs, GPU residence, no-mirror, controller-as-canonical-querier, no forward injection, *and* mechanism choice — the doc's m_proj thesis is parked, not implemented): `docs/plans/2026-04-25-memory-aware-optimizer-design.md`
- ScOpt design: `docs/plans/2026-04-22-scarcity-optimizer-design.md`
- Built modules: `src/chaoscontrol/optim/episodic_cache.py` (15 tests green), `src/chaoscontrol/optim/episodic_writer.py` (20 tests green)
- Failed baseline (the experiment we're flipping): `experiments/24_training_time_bundle/phase0_dw_sweep_4x_20260422T224040Z` — uncurated Dreamworld at val_bpb 1.484, lost to fast_slow-only at 1.479
- Falsifier: per-bucket val CE on 4 buckets `[1314, 20259, 13.8M, 28.4M]` tokens; rare bucket (≈1314 tokens) is load-bearing
- Importing `runner_fast_path`: use `importlib.util.spec_from_file_location` (see `tests/test_cd_config_threading.py:14`); the directory `experiments/23_fast_path/` is not a Python package

## Pre-conditions (already true)

- ScOpt built + integrated (`src/chaoscontrol/optim/scopt.py`)
- Per-token CE exposed via fused LM-head kernels
- Per-bucket val CE diagnostics in place
- Manual all-reduce DDP (Dreamworld backward syncs through it correctly — DDP-bypass concern retracted 2026-04-25)
- Episodic cache + writer modules built and tested

---

## Phase 0 — Settle defaults (no code, ~30 min reading)

### Decision 0.1 — `key_rep` captured at write, **never refreshed in Phase 1-3**

`key_rep` = encode-output residual `hidden[b, t]` at the write position. Phase 1-3 captures once at write, never refreshes. Drift is handled by eviction (utility EMA + age) rather than refresh. Phase 5 may introduce a refresh mechanism, but only after a *drift model* is in place (Codex: "we need to know whether key_rep remains anchored to the original event or becomes a self-referential synthetic-context vector"). Refresh-via-replay was tempting but, as currently specified, replays the value tokens out of their original context — the new key_rep would be from a synthetic, decontextualized residual. Not safe to ship without the drift model.

### Decision 0.2 — Read = utility-weighted cosine top-K

`score(i) = cosine_sim(query_residual, cache.key_rep[i]) × cache.utility_u[i]` for occupied slots `i`. Top-K = 16 default. Two GPU vector ops per candidate; no learned predictor in the read path; "close to atomic." Entries that retrieve well by similarity but never produce useful replay get demoted via the utility factor.

**Cold-start handling (resolved 2026-04-25 reviewer round 3):** `utility_u` initializes to **1.0** at append time (was 0.0 — the bug both reviewers caught). New entries enter at full retrieval weight; subsequent replay-driven `update_utility` calls pull utility down for entries that don't help. Floor is implicit (utility EMA blend with non-negative signals stays non-negative under the default utility transform — see Decision 0.10).

Phase 5+ research candidate: per-entry stored predictor vectors (`(key_rep·q)(predictor_w·q)·utility_ema`) — deferred until Phase 3 says whether utility-weighted cosine alone suffices.

### Decision 0.3 — Write rate

`top_p = 1.0 / (B * T)` → ~1 position per micro-batch step. Same top-p drives query-candidate publication (one source signal, two consumers).

### Decision 0.4 — Cache capacity

`capacity=4096`, `span_length=4`, `key_rep_dim=model_dim`, `grace_steps=1000`. ~40 MB on episodic GPU at `model_dim=256`.

### Decision 0.5 — Falsifier with statistical-power escalation

Three arms × three seeds × per-bucket val CE. Pass: rare-bucket CE delta ≤ −0.005 bpb with stderr clearing zero. **Pre-committed escalation rule:** if `σ` across the 3 seeds on the rare-bucket delta exceeds 0.008 bpb, run 3 additional seeds before declaring pass/fail. Pinned now to avoid negotiating it under time pressure.

### Decision 0.6 — Hardware target: 3+1 from day one

4×H100 with 3 train + 1 episodic ranks (same torchrun group) for Phase 1-3. Phase 4 scales to 8×H100 (6+2). No single-GPU prototype — controller-on-separate-process and episodic-GPU-as-DDP-rank are structural, and a single-GPU smoke test would be a different system.

### Decision 0.7 — Process model: episodic GPU is a DDP rank, controller is a child process

- **Episodic GPU = DDP rank N-1** in the same torchrun process group. Has its own model copy (already replicated by DDP init), its own `param.grad`. Replay gradients flow through the standard all-reduce. NO `multiprocessing.spawn` for the episodic worker.
- **CPU controller = one child process per node** spawned via `multiprocessing.get_context("spawn")` at runner init. Touches no gradients, no GPU memory directly. Communicates with train ranks and the episodic rank via shared-memory rings only. On controller death, train ranks see a stale heartbeat flag and continue without cache effects (writes still drain to the episodic rank, the cache fills, replay path goes silent).

### Decision 0.8 — IPC: N SPSC rings, POSIX shm, drop-on-overflow

- **Write rings:** one per train rank, SPSC. Train rank writes WritePayload structs; episodic rank drains all rings each cycle.
- **Query-candidate rings:** one per train rank, SPSC. Train rank publishes `(b, t, pressure, residual[D])` tuples for the top-p positions per step. Controller drains all rings each cycle.
- **Tagged-replay ring:** SPSC, controller → episodic rank. Controller pushes slot indices to replay; episodic rank drains.
- All bounded; on overflow, oldest slot is overwritten and a `dropped_<ring>_count` counter is bumped. Never blocks any rank.

### Decision 0.9 — Diagnostic schema as DuckDB-friendly time-series substrate

Per-replay diagnostics from Phase 3 are logged as a flat append-only event stream with a fixed columnar schema (NDJSON or Parquet rows). Schema is chosen to be queryable by DuckDB without transformation. Phase 5's time-series analytics (cohort analysis, drift trajectory, cross-entry correlation, surprise frontier, adaptive replay budget) build on top of this log. Phase 1-3 just produces the log; Phase 5 adds the query layer.

Schema (rows appended once per replay event):

```
step                       : int64    -- training step at which replay fired
slot                       : int64    -- cache slot that was replayed
key_fp                     : int64    -- entry's fingerprint
write_step                 : int64    -- step at which entry was originally written
write_pressure             : float64  -- pressure at write time
write_bucket               : int8     -- token-bucket index (0..3) of value_anchor_id
query_cosine               : float64  -- cosine sim that retrieved this entry
utility_pre                : float64  -- utility_ema before this replay
replay_loss                : float64  -- CE on value tokens after replay forward
replay_grad_norm           : float64
replay_grad_cos_common     : float64  -- cosine vs live common-grad direction
replay_grad_cos_rare       : float64  -- cosine vs live rare-grad direction
replay_grad_cos_total      : float64  -- cosine vs total grad
utility_signal_raw         : float64  -- raw signal fed to update_utility (signed)
utility_signal_transformed : float64  -- transformed/clamped value actually applied
utility_post               : float64  -- updated utility_ema after this replay
```

### Decision 0.10 — Utility signal: clamped cosine to live rare-grad direction

`update_utility(slot, ce_delta=...)` is called once per replay event. The `ce_delta` parameter is overloaded by name (carryover from earlier design); the actual signal we feed is:

```
utility_signal_raw         = replay_grad_cos_rare              # signed [-1, 1]
utility_signal_transformed = max(0.0, replay_grad_cos_rare)    # clamped [0, 1]
update_utility(slot, ce_delta=utility_signal_transformed)
```

Rationale (Codex round 3): "if utility is just 'was hard when replayed,' the cache may preserve difficult junk." The signal we want is "did this replay produce gradients that align with the live rare-gradient direction" — *useful*, not *hard*. Cosine to rare-grad gives this directly. Clamp to non-negative because:
- A negative cosine means the replay grad pushed the model in the wrong direction; we shouldn't reward it, but neither should we punish into negative `utility_u` (which would interact pathologically with `score = cosine × utility_u` — entries with negative utility would invert the cosine ordering)
- Clamping to 0 means "this replay didn't help"; eviction handles entries that never produce above-zero signal

Both raw and transformed values are logged so Phase 4 / Phase 5 can sweep alternative transforms (sigmoid, square, etc.) without re-running matrices.

**Phase 5+ alternatives to evaluate** (not implemented now): replay loss improvement (`pre_ce - post_ce` on value span), gradient alignment to rare grad weighted by per-channel pressure, predicted future rare-bucket val CE delta. The diagnostic log keeps enough columns to backtest these without new pod runs.

---

## Phase 1 — Bring up the 3+1 hardware split with synchronous writes

End state: SSM trains on 3 ranks; episodic rank holds the cache; per-rank write rings drop payloads from train ranks; episodic rank drains them. NCCL subgroups in place. No reads, no replay yet. Eight tasks.

### Task 1.0: Commit the existing cache + writer modules

```bash
pytest tests/test_episodic_cache.py tests/test_episodic_writer.py -v
git add src/chaoscontrol/optim/episodic_cache.py src/chaoscontrol/optim/episodic_writer.py tests/test_episodic_cache.py tests/test_episodic_writer.py
git commit -m "optim: episodic cache substrate + write-trigger logic (components 1-2)"
```

### Task 1.1: SPSC ring buffer over POSIX shared memory

**Files:** Create `src/chaoscontrol/episodic/ipc.py`, `src/chaoscontrol/episodic/__init__.py`; test `tests/test_episodic_ipc.py`.

5-step TDD. Tests cover: write/read in order; overflow drops oldest and bumps `dropped_count`; multi-attach (one writer process attaches by name, one reader process attaches by same name) round-trips. Implementation uses `multiprocessing.shared_memory.SharedMemory` for a slot table + a small separate SHM for head/tail/dropped counters. Only ONE writer per ring at any time — multi-writer is a Phase-1.4 task using N rings, not a multi-writer ring.

```bash
git commit -m "episodic: SPSC shm ring buffer (drop-on-overflow, never blocks)"
```

### Task 1.2: Modify `allreduce_grads` to take a process group + materialize-zero option

**Files:** Modify `src/chaoscontrol/distributed.py`; test `tests/test_distributed_allreduce_grads.py`.

The current `allreduce_grads(model, world_size)` (`distributed.py:40-65`) only operates on the default process group, uses `op=AVG`, and returns early when no grads exist. Three changes required for the 3+1 split:

1. Accept a `group: dist.ProcessGroup | None` parameter (default = WORLD).
2. Accept a `materialize_zeros: bool` parameter (default = False). When True, before the collective, set `param.grad` to `torch.zeros_like(param)` for any param where `param.grad is None`. This guarantees the collective sees identical shapes on every rank — the "explicit zero path" Codex flagged.
3. Switch from `op=AVG` to `op=SUM`. Caller is responsible for any pre-scaling (per the all-reduce topology section: train ranks pre-scale main grads by `1/(N-1)` before calling).

5-step TDD on the gloo backend with `mp.spawn(world_size=4)`. Tests must cover:
- Single SUM all-reduce with pre-scaled grads gives expected `main_avg + replay_full` on every rank.
- Materialize-zeros mode produces consistent shapes when one rank has fewer non-None grads.
- Old call sites (`allreduce_grads(model, world_size)`) still work unchanged (preserve back-compat with the existing AVG path via a `op` kwarg or a separate function — pick whichever keeps existing exp23 cells bit-for-bit identical).

```bash
git commit -m "distributed: allreduce_grads gains group + materialize_zeros + SUM mode"
```

### Task 1.3: Episodic-rank skip-main + unconditional replay collective

**Files:** Modify `experiments/23_fast_path/runner_fast_path.py` (init: detect episodic rank, skip main forward/backward path; restructure step loop so all ranks unconditionally enter the all_group all-reduce after main+replay phases); test `tests/test_runner_3plus1_skip_main.py`.

Two structural changes:

**1. Episodic rank skips main.** Detect `is_episodic_rank = (rank == world_size - 1)` when `episodic_enabled=True`. On that rank, skip main forward+backward each step. The episodic rank's local `param.grad` after main-phase is therefore None for all params (or the replay grads from Phase 3 onward).

**2. Replay collective is unconditional, every step, every rank.** Codex round 3: "every rank needs to enter the replay phase every step, even when no replay item is available, with explicit zero grads when idle. Otherwise 'episodic rank has replay this step' can diverge from 'train ranks skipped replay this step.'" Concretely:

- Train ranks 0..N-2: after main backward, multiply `param.grad` by `1/(N-1)` (the pre-scale per the topology section). They have no replay grads.
- Episodic rank N-1: if a tagged-replay queue item is available, run replay backward; otherwise materialize zeros (handled by `materialize_zeros=True` in the upcoming all-reduce).
- ALL N ranks: call `allreduce_grads(model, group=all_group, materialize_zeros=True, op=SUM)`. Single collective, every step. Phase 1 has no replay yet, so the episodic rank always submits zeros — but the collective fires regardless.

5-step TDD on CPU/gloo smoke. Test must verify the single-collective shape is consistent across ranks even when replay is absent.

```bash
git commit -m "exp23: episodic rank skip-main + unconditional all-rank replay collective"
```

### Task 1.4: Wire `select_writes` into per-rank write ring

> **CURRENT 2026-04-26:** Perf Pass C's synchronous `dist.gather` replacement was retired. The trunk-throughput invariant is now explicit: train ranks publish WRITE_EVENT records into per-rank `ShmRingWriteEvent` rings and continue; the episodic rank drains asynchronously. Memory may lag/drop under backpressure, but it must not add a train-step collective.

**Files:** Modify `experiments/23_fast_path/runner_fast_path.py` (in `_build_optimizer` region for ring setup; in train step body after `per_token_ce`/`pressure` are computed); test `tests/test_runner_episodic_writes.py`.

Two pieces:
1. Setup: when `episodic_enabled=True`, create one write ring per train rank (named `episodic_write_ring_rank{R}`) and one query-candidate ring per train rank (`episodic_query_ring_rank{R}`). Episodic rank attaches as reader to all train rings.
2. In-step: shape adapter for `per_token_ce` (`[B, T-1] → [B, T]` via right-pad with zeros) goes here. The top_p selection runs once and produces both write payloads AND query-candidate tuples. Write payloads → `write_ring[rank].try_write(payload)`; query-candidates → `query_ring[rank].try_write((b, t, pressure[b,t], residual[b,t,:]))`.

5-step TDD.

```bash
git commit -m "exp23: train step writes payloads + query candidates to per-rank rings"
```

### Task 1.5: Episodic rank drains write rings into the cache

> **CURRENT 2026-04-26:** The live drain is `_drain_episodic_write_rings` plus the `episodic_write_drain` daemon thread in `runner_fast_path.py`. `_drain_episodic_payloads_gpu` remains only as a legacy direct-test helper for the old slot-tensor format.

**Files:** Modify `experiments/23_fast_path/runner_fast_path.py` (episodic rank's per-step loop body); test `tests/test_runner_episodic_drain.py`.

On each step, the episodic rank drains every train rank's write ring and calls `cache.append(...)` for each payload. Heartbeat counter increments per cycle.

5-step TDD on a CPU multi-rank smoke (mp.spawn world_size=4 with gloo).

```bash
git commit -m "exp23: episodic rank drains write rings into cache"
```

### Task 1.6: 4×H100 (3+1) end-to-end smoke

Live pod test. NCCL backend. 100-step training run with `episodic_enabled=True`. Cache fills monotonically on episodic rank; train val_bpb at end of 100 steps within float-noise of a **3-rank no-episodic baseline (NOT 4-rank)** at matched config — Codex round 3 flagged that the 4-rank baseline confounds "episodic-write effect" with "loss-of-a-train-rank effect." The right comparison: 3 train ranks with episodic disabled vs. 3 train ranks with episodic enabled and the 4th rank as the episodic worker. This isolates the write-only effect to ZERO behavioral change. No reads, no replay yet.

**Pod lifecycle (per `feedback_no_subagent_runpod.md`):** subagent may invoke `runpodctl` IF it runs in the foreground with the main agent monitoring. For Task 1.6's ~30-min smoke that's appropriate. The subagent owns: bring up pod, sync code, run the 3-rank baseline + 3+1-episodic configs, capture results, and stop the pod. The main agent monitors completion and reads results. Do NOT dispatch this in `run_in_background` mode without a follow-up `/loop` watcher.

Use `runpodcli` to bring up 4×H100. `/loop` for monitoring per `feedback_monitor_via_loop.md`.

Smoke results JSON committed; runtime telemetry shows cache_len > 0 and write-ring `dropped_count` = 0.

```bash
git commit -m "exp23: 4xH100 (3+1) write-only smoke (cache fills, val_bpb unchanged)"
```

### Task 1.7: Telemetry pass

Per-step train telemetry: `episodic_writes_this_step`, `episodic_query_candidates_this_step`, `episodic_write_ring_dropped`, `episodic_query_ring_dropped`, `episodic_rank_heartbeat_age_steps`. Logged via the existing rank0 jsonl logger.

```bash
git commit -m "exp23: episodic cache + ring telemetry per step"
```

**Phase 1 exit criterion:** 4×H100 (3+1) runs 100 steps; cache fills monotonically; main val_bpb within float-noise of 4-rank baseline (no reads, no replay). All ring drop-counts zero under steady state.

---

## Phase 2 — Bring up the CPU controller + cache queries (4×H100, 3+1)

End state: controller polls query-candidate rings, runs cosine queries on episodic rank, fills tagged-replay ring. Replay still doesn't fire — that's Phase 3. Five tasks.

### Task 2.1: Spawn the controller process

**Files:** Create `src/chaoscontrol/episodic/controller.py`; test `tests/test_episodic_controller.py`.

Controller process spawns at runner init via `multiprocessing.get_context("spawn")`. Holds reader handles to all per-rank query-candidate rings + writer handle to the tagged-replay ring + a query-request/response ring pair to the episodic rank. Runs an unbounded loop bounded by per-cycle wall budget (default 50 ms). Heartbeat counter in shared memory.

5-step TDD with CPU smoke harness.

```bash
git commit -m "episodic: CPU controller process scaffold (loop + heartbeat)"
```

### Task 2.2: Controller drains query rings, requests top-K from episodic rank

**Files:** Modify `src/chaoscontrol/episodic/controller.py`; test extends `tests/test_episodic_controller.py`.

Per cycle: drain all train-rank query rings → for each candidate, send a query request to the episodic rank → drain query responses → push tagged slot indices to the tagged-replay ring.

5-step TDD.

```bash
git commit -m "episodic: controller drains query rings + requests top-K matches"
```

### Task 2.3: Episodic rank serves queries: utility-weighted cosine top-K

**Files:** Modify `experiments/23_fast_path/runner_fast_path.py` (episodic rank's per-step loop also drains query-request ring); test `tests/test_runner_episodic_query.py`.

Episodic rank exposes a query API:

```python
def query_topk(query: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Returns [k] tensor of slot indices in descending utility-weighted-cosine
    order. Slots with occupied=False are excluded."""
    occupied = self.cache.occupied
    if not occupied.any():
        return torch.empty(0, dtype=torch.int64)
    keys = self.cache.key_rep[occupied]
    util = self.cache.utility_u[occupied]
    q = query / (query.norm() + 1e-8)
    keys_n = keys / (keys.norm(dim=1, keepdim=True) + 1e-8)
    cosines = keys_n @ q
    scores = cosines * util  # utility-weighted cosine, Decision 0.2
    k_eff = min(k, scores.size(0))
    _, top = torch.topk(scores, k_eff, largest=True)
    occupied_idx = occupied.nonzero().squeeze(-1)
    return occupied_idx[top]
```

5-step TDD.

```bash
git commit -m "episodic: utility-weighted cosine top-K query (Decision 0.2)"
```

### Task 2.4: Tagged-replay ring fills end-to-end

**Files:** Integration test `tests/test_runner_episodic_writes.py`.

End-to-end: train step → write payload → episodic rank drains → cache fills → train step publishes query candidates → controller drains → controller requests top-K → episodic rank computes utility-weighted cosine → controller pushes slot indices to tagged-replay ring → tagged-replay ring depth grows.

```bash
git commit -m "exp23: tagged-replay ring fills under end-to-end smoke"
```

### Task 2.5: Phase 2 4×H100 (3+1) smoke

Live pod test. 100 steps with `episodic_enabled=True`. Cache fills (Phase 1); residuals publish; controller queries; tagged-replay ring fills with non-zero entries at non-zero cosine match. Still no replay → val_bpb unchanged from baseline.

```bash
git commit -m "exp23: 4xH100 (3+1) controller + query smoke (val_bpb unchanged)"
```

**Phase 2 exit criterion:** tagged-replay ring fills end-to-end; controller heartbeat steady; val_bpb still within float-noise of the no-episodic baseline.

---

## Phase 3 — Curated Dreamworld replay validation (the falsifier)

End state: tagged entries get replayed; per-replay diagnostic log fills; per-bucket val CE delta is measured against a 3+1-matched uncurated DW control. Four tasks.

### Task 3.1: Episodic rank drains tagged-replay ring + runs Dreamworld backward

**Files:** Modify `experiments/23_fast_path/dreamworld.py` (add `dreamworld_replay_from_cache_entry(model, cache, slot, weight, ...)` parallel to existing `dreamworld_replay_backward`); modify `experiments/23_fast_path/runner_fast_path.py` (episodic rank's per-step loop drains tagged-replay ring); test `tests/test_dreamworld_replay_from_cache.py`.

Reads `cache.value_tok_ids[slot]` + `cache.value_anchor_id[slot]`, builds a synthetic input batch (the value tokens themselves), runs forward → CE → backward through the episodic rank's model copy. Gradients accumulate into `param.grad` on the episodic rank; at the end of the step, all 4 ranks call `allreduce_grads(within=all_group)` with explicit zeros from train ranks for any params the replay touched.

5-step TDD on CPU smoke.

```bash
git commit -m "dreamworld: replay backward from cache entry (parallel to online buffer)"
```

### Task 3.2: Per-replay diagnostic log (DuckDB-friendly schema)

**Files:** Create `src/chaoscontrol/episodic/diagnostics.py`; modify `experiments/23_fast_path/dreamworld.py` to emit log rows per replay event; test `tests/test_episodic_diagnostics.py`.

Schema per Decision 0.9. Log emitted as NDJSON to a per-rank file under the run dir (`run_dir/episodic_replay_log_rank{R}.ndjson`). DuckDB can ingest these via `read_json_auto(...)` without transformation. Log includes the replay-grad-vs-live-grad cosines (Codex's "(c) retrieval diagnostics").

5-step TDD.

```bash
git commit -m "episodic: per-replay diagnostic log (DuckDB-ready NDJSON schema)"
```

### Task 3.3: Falsifier matrix — `episodic_dw_curation_v1` (4 arms × 3 seeds, with escalation)

**Files:** Modify `experiments/24_training_time_bundle/exp24.py` (new matrix builder `build_episodic_dw_curation_v1_matrix`); modify `experiments/24_training_time_bundle/run_exp24.py` (register matrix); shape pin in `tests/test_exp24_training_bundle.py`.

Four arms × three seeds = 12 cells. **All four arms must be topologically identical** — 3+1 rank layout, episodic rank present, same all-reduce path, same replay cadence, same `dreamworld_weight`, same wall budget. The ONLY difference between arms is the replay-candidate-selection mechanism; everything else (config, ScOpt, fused-CE backend, fast_slow recipe, seeds) is matched. Codex round 3: "uncurated DW should use the exact same episodic rank, replay cadence, loss weight, and all_group averaging as curated DW — otherwise the A/B delta can still be replay topology rather than memory curation."

- **Arm A — uncurated DW:** episodic rank reads replay candidates from the existing online buffer (current `dreamworld.py` path); `episodic_enabled=False` for cache writes/queries. Replay backward runs on the episodic rank, gradients flow through the same all_group SUM all-reduce as Arms B/B'.
- **Arm B — curated DW (cosine + utility-weighted retrieval):** episodic rank reads replay candidates from the cache's tagged-replay ring; `episodic_enabled=True` for cache writes/queries. Retrieval: `score = cosine_sim(query_residual, key_rep) × utility_u` (Decision 0.2). Replay backward, all-reduce, replay weight, and per-step replay frequency identical to Arms A/B'.
- **Arm B' — curated DW (pressure-only retrieval) — MECHANISM-SPECIFICITY ARM:** identical to Arm B *except* retrieval ignores cosine similarity and utility_u entirely. Cache fills the same way (write trigger unchanged); replay candidates are selected by `score = pressure_at_write` only. **Purpose:** distinguish "memory persistence" from "any rare-grad-aligned retrieval policy." If Arm B beats both A and B' on rare-bucket val CE, the result is mechanism-specific (cosine + utility_u retrieval is doing the work). If Arm B ties Arm B', the thesis collapses to "replay-grad-rare-alignment" rather than memory persistence — Phase 4 ablation would catch this expensively; Arm B' catches it cheaply.
- **Arm C — no DW reference:** episodic rank present, `dreamworld_weight=0` so replay backward fires but contributes zero gradient. Establishes the 3+1 topology baseline without any DW signal at all. Same all-reduce path so it pays the same overhead.

Seeds: 1337, 2674, 4011. Budget 600s/cell + ~250s full-val/cell × 9 cells ≈ ~7-8h wall on 4×H100.

**Pre-committed escalation rule (Decision 0.5):** if `σ(rare-bucket δ_bpb)` across 3 seeds on Arm B > 0.008 bpb, run 3 additional seeds on Arms A, B, and B' before declaring pass/fail.

**Updated decision gates (4-arm version):**
- **Pass + mechanism-specific:** Arm B beats both Arm A AND Arm B' on rare-bucket val CE by ≥ 0.005 bpb. The thesis (memory persistence × similarity-based recall) is supported. Phase 4 unlocks; the ablation matrix expands the lesion set.
- **Pass + mechanism-agnostic:** Arm B ties Arm B' (within stderr) but both beat Arm A. The thesis collapses to "any rare-grad-aligned replay curation works." Memory persistence is NOT load-bearing. Phase 4 still unlocks but the predictor-vector research (Phase 5.4) becomes less interesting and Phase 5.3 (refresh) likely never pays. Document and proceed.
- **Mixed/null:** Arm B ties Arm A. Cache curation didn't help. See diagnostic checks below.
- **Regression:** Treatment worse than control. Disable, root-cause.

**Cost delta vs. 3-arm version:** +3 cells × 600s training × 4×H100 ≈ +90 min training + ~15 min full-val × 3 = ~2.25h additional wall-clock. Total Phase 3 falsifier wall: ~7.5h on a warm pod.

```bash
git commit -m "exp24: episodic_dw_curation_v1 matrix (3+1 topology, 3 arms x 3 seeds)"
```

### Task 3.4: Run on 4×H100 (3+1), commit results, run analysis

Use `runpodcli`. `/loop` for monitoring. Budget ~8h wall.

After results land:

```bash
python experiments/24_training_time_bundle/analyze_episodic_dw_curation_v1.py
git commit -m "exp24: record episodic_dw_curation_v1 results"
```

**Decision gate after Task 3.4:**

- **Pass:** Arm B rare-bucket val CE delta vs. Arm A ≤ −0.005 bpb (or ≤ −0.005 after seed-escalation if the σ rule fires). Curation flips Dreamworld's transfer problem. Phase 4 unlocks.
- **Mixed/null:** Curation didn't beat uncurated under matched topology. Use the diagnostic log to diagnose:
  - Cache fill rate too low? (telemetry)
  - Cosine retrieval not surfacing useful entries? (replay-grad-cos-rare in the log — if it's near zero across high-utility entries, retrieval is wrong)
  - Replay rate too low? (controller cycle budget vs. step cadence)
  - Consolidation works but doesn't transfer? (replay loss curves drop but val_bpb doesn't)
  - Then either (a) iterate curation logic, (b) try Thesis A (m_proj) as the alternative, or (c) park
- **Regression:** Treatment worse than uncurated. Disable, root-cause, do not proceed.

---

**Phase 3 exit criterion:** decision gate result documented in a memory entry under `2026-04-25-memory-aware-optimizer-plan` with rare-bucket delta + σ + per-arm bucket table.

---

## Phase 3.5 — DuckDB analytics layer (resequenced 2026-04-25 after spec review)

**Why pulled forward from Phase 5:** the 8-GPU ablation matrix in Phase 4 is expensive and a null/mixed result is hard to interpret without queryable diagnostics. Without an analytics layer, the controller is still doing pressure-only candidate selection in Phase 4 — which means Phase 4 ablation can't distinguish "controller policy works" from "the headline curated-vs-uncurated effect." DuckDB lands BEFORE any 8-GPU run.

Tasks (detailed):

- **3.5.1 — DuckDB ingestion layer** for the Phase 3 NDJSON event log. Schema is already DuckDB-friendly per Decision 0.9. Build a small Python module (`src/chaoscontrol/episodic/analytics.py`) that wraps `duckdb.connect()` + `read_json_auto(...)` for the per-replay log, with helpers for the common queries.
- **3.5.2 — Common query helpers:** cohort analysis (group entries by write-time channel mass), drift trajectory aggregation (avg key_rep displacement per entry per N steps), surprise frontier rollup (pressure × CE distribution over training), retrieval-utility correlation (does cosine retrieval predict replay grad cosine to rare-grad direction).
- **3.5.3 — Controller-side query hooks:** wire the controller's per-cycle priority function to use DuckDB queries instead of pressure-only candidate selection. Replaces the Phase 2 default. This is what makes Phase 4 ablation testable beyond the headline experiment.

**Phase 3.5 exit criterion:** controller demonstrably uses query-derived priority on a 4×H100 (3+1) smoke run; per-cycle wall budget for DuckDB queries stays under the controller's 50ms target. If queries are too slow at cache size 4096, profile and reduce; the analytics layer can NOT become a step-time bottleneck.

**Drift snapshots (originally 5.2) are absorbed into Phase 3** — they're a cheap observability-only addition during the falsifier matrix run, NOT a separate task. Add a periodic `snapshot_key_reps_to_log()` call (every M=512 steps) that writes one NDJSON row per occupied cache slot to a parallel `episodic_drift_log_rank{R}.ndjson`. Rolled into Task 3.2's diagnostic-log work.

---

## Phase 3.6 — Drift correction implementation (4×H100 work)

**Why pulled forward from Phase 5.3 (resequenced 2026-04-25):** Ken's framing — "I am really assuming it's going to be needed." Drift correction is the mechanism that determines whether `key_rep` stays anchored to its original event under continued backbone training. We bake the implementation in BEFORE 8-GPU work because (a) implementing it on cheap 4×H100 cycles before scaling is the right cost order, and (b) the 8-GPU ablation matrix in Phase 4 should test the system *with* drift correction active, not test "no drift correction" as a sneaky bonus arm.

**Hard prerequisite:** Phase 3.5 outputs. The drift correction design picks among the three `key_rep` refresh options (full SSM state capture / prefix-token re-encode / a hybrid) based on what Phase 3.5's empirical drift trajectory data actually shows. Specifically, the DuckDB queries answer:
- How fast does `key_rep` drift in practice (norm/step)?
- Which cohorts drift fastest (early-layer vs late-layer residuals, high-pressure vs low-pressure writes)?
- Does drift correlate with retrieval failure (cosine match producing replay grads that don't align with rare-grad direction)?

Without those answers, we'd be guessing at refresh cadence, refresh trigger threshold, and refresh mechanism. With them, the design is data-driven.

**Tasks (sequenced after Phase 3.5):**
- **3.6.1 — Drift trajectory analysis report.** Run the DuckDB queries from 3.5 against the Phase 3 falsifier results' drift logs. Output a brief markdown report under `experiments/24_training_time_bundle/results/` summarizing measured drift, cohort splits, and retrieval-failure correlation. This is the spec for 3.6.2.
- **3.6.2 — Refresh mechanism implementation.** Pick the option (or hybrid) the trajectory report supports. Wire it through the cache + worker. Add a per-entry `birth_embedding_version` increment on refresh (already in the cache schema).
- **3.6.3 — Drift-correction unit tests.** Verify refresh produces the expected `key_rep` change for synthetic inputs; verify the eviction-to-refresh policy doesn't ping-pong; verify back-compat (drift correction OFF = Phase 1-3 behavior bit-identical).

**Phase 3.6 exit criterion:** drift correction is opt-in via a new config flag (default OFF for back-compat with Phase 3 cells); unit tests green; the Phase 3 falsifier results stay reproducible bit-identically with the flag OFF.

---

## Phase 3.7 — Second 4×H100 falsifier with drift correction active

**Goal:** validate that drift correction is the multiplier we expect on cheap compute before committing 8-GPU cycles.

**Matrix `episodic_dw_curation_v2`:** 6 cells = 2 arms × 3 seeds.
- **Arm B' — curated DW + drift correction ON:** same config as Phase 3 Arm B, plus the new drift-correction flag enabled.
- **Arm B (reference) — curated DW + drift correction OFF:** REUSE the Phase 3 Arm B results (don't re-run; same seeds, same config). The ScOpt + episodic_enabled config combo (currently guarded; relies on Task 95) must be functional.

Statistical comparison: B' vs B per-bucket val CE delta. Pre-committed escalation rule from Decision 0.5 (σ > 0.008 bpb on rare bucket → run 3 more seeds).

**Phase 3.7 exit criterion:** decision gate.
- **Pass:** B' improves rare-bucket val CE over B by ≥ 0.005 bpb. Drift correction works at scale-down. Phase 4 unlocks; the 8-GPU ablation matrix now tests the *full* system.
- **Mixed/null:** drift correction didn't help on 4×H100. Either (a) drift wasn't the bottleneck (then Phase 4 is meaningful without 3.6's contribution), or (b) the refresh mechanism was wrong and a different option from 3.6.2 should be tried. Decision driven by the diagnostic logs from Phase 3.5.
- **Regression:** B' worse than B. Disable, root-cause. Don't proceed to 8-GPU.

**Hard prerequisite for proceeding to Phase 4:** either Phase 3.7 PASS, or an explicit decision (with documented reasoning) to scale up without drift correction.

---

## Phase 4 — Scale to 8×H100 (6+2) + ablation matrix (outline only)

Tasks (not detailed; sequence after Phase 3 pass):

- **4.1: Multi-episodic-rank topology.** 2 episodic GPUs. Sharded cache vs. replicated vs. one-master-one-replay-only — decide based on Phase 3 cache fill rate and replay throughput.
- **4.2: 8-rank performance.** All-reduce timing at 6+2 vs. 8-rank symmetric. Verify the topology cost is below the curation gain.
- **4.3: Falsifier rerun on 8×H100.** Confirm Phase 3 finding holds at scale. Same 3 arms, same 3 seeds.
- **4.4: Ablation matrix `episodic_dw_curation_v2`.** If 4.3 passes, run the full ablation Codex named: random cache replay (cache as storage with random retrieval), pressure-only replay (no cosine, just pressure rank), stale-key replay (key_rep frozen at first write, never updated even by Phase 5 refresh), key-shuffled replay (permute key_reps to break the retrieval signal). Each arm × 3 seeds. Distinguishes which curation component matters.

**Phase 4 exit criterion:** rare-bucket val CE delta on 8×H100 (6+2) within noise of the 4×H100 (3+1) Phase 3 result; ablation isolates the curation mechanism that drives the result.

---

## Phase 5 — Research extensions (outline only)

**Resequenced 2026-04-25:** former 5.1 (DuckDB analytics), 5.2 (drift model snapshots), and 5.3 (key_rep refresh) are pulled forward. They land as Phase 3 (drift snapshots), Phase 3.5 (DuckDB), and Phase 3.6 (refresh implementation) respectively, validated on a second 4×H100 falsifier (Phase 3.7) before any 8-GPU work. Phase 5 is now strictly post-Phase-4 research extensions.

- **5.4: Predictor-vectors-per-entry research.** Per-entry stored `predictor_w[D]` updated by replay history. Retrieval score becomes `(key_rep · q)(predictor_w · q)·utility_ema`. Atomic at retrieval; learned per-entry. Triggered only if Phase 4 ablation shows utility-weighted cosine retrieval is selecting wrong entries even with drift correction active.
- **5.5: Cross-rank cache coherence (6+2).** Sharded cache across 2 episodic GPUs vs. replicated. Decide based on Phase 4 throughput.

---

## Tracking

- Plan owner: Ken
- Date scope: 2026-04-25 → estimated 2 sessions for Phase 1, 1 session for Phase 2, Phase 3 matrix is one ~8h pod run with offline analysis. Per `feedback_estimate_calibration.md` rule of dividing by 10×, real estimates may compress further.
- Cache + writer modules untracked on `main`; commit as Task 1.0 first.
- Phase 3.5, 3.6, 3.7 detail will firm up after Phase 3's first 4×H100 result lands. Phase 4-5 stay outline-only.

**Phase sequencing (locked 2026-04-25):**
1. Phase 1 — 3+1 hardware bring-up (in flight)
2. Phase 2 — controller + queries
3. Phase 3 — first 4×H100 falsifier (drift snapshots logged inline)
4. Phase 3.5 — DuckDB analytics layer (queryable diagnostics)
5. Phase 3.6 — drift correction implementation (informed by 3.5 outputs)
6. Phase 3.7 — second 4×H100 falsifier with drift correction
7. Phase 4 — scale to 8×H100 + ablation matrix (only after 3.7 passes)
8. Phase 5 — research extensions (predictor vectors, cross-rank coherence)

## Cross-cutting reminders

- Tests after every edit (`feedback_run_tests_after_every_edit.md`).
- No warnings; root-cause failures (`feedback_no_warnings.md`).
- Verify before claiming wiring exists (`feedback_wiring_vs_existence.md`).
- Don't auto-stop pods unilaterally (`feedback_always_stop_pods.md`).
- A regression is never a build error (`feedback_regression_is_never_build_error.md`).
- 3+1 hardware split from day one (Decision 0.6); no single-GPU prototype detour.
- Cache content reaches the SSM only via gradients in this plan (Thesis B). Forward-time injection and direct-optimizer-direction (Thesis A / m_proj) are documented elsewhere and parked, not silently bolted on.
- Controller's DuckDB analytics live OFF the cache read path; cache reads stay close-to-atomic (utility-weighted cosine, two GPU vector ops).
- Diagnostics from day one in Phase 3; null results without per-replay logs are uninterpretable.
