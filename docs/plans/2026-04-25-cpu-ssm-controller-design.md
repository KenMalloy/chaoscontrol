# CPU SSM Controller Design

**Status:** Brainstormed and approved 2026-04-25. Implementation plan is the next step (`writing-plans` skill).

**Supersedes:** task #105's original "Simplex + controller-memory pivot (Pass E)" framing. The cosine-utility heuristic in X-controller becomes the bootstrap teacher, not the deliverable.

## Goal

Replace the hand-tuned cache-curation heuristics (admission, eviction, utility maintenance, retrieval scoring) with a learned policy embodied in a small CPU SSM that runs as a separate process, writes to the cache that the GPUs read from, and learns online during training from delayed reward.

## Constraints (locked during brainstorming)

- **Hardware target:** Sapphire Rapids (Intel Xeon Platinum 8470, 26 vCPU). AMX BF16/INT8 + AVX-512 available.
- **Implementation language:** C++ inference and online learning. AMX intrinsics for matmul, AVX-512 for diagonal recurrence. PyTorch only for the offline bootstrap pretrain. Revisit C++ if it becomes a barrier.
- **Process model:** separate process from the GPU train ranks. Reads event records from per-rank shared-memory rings on its own loop. Writes to the shared cache structure that GPUs read from. Not in the GPU query path.
- **Architectural symmetry with core SSM:** TTT at eval, fast/slow weight pair (α=0.25, interval=64 events, eval uses slow).
- **Artifact budget:** ≤16MB total. Existing model + tokenizer ~11.5MB. Controller weights ~100–500KB at int8. Cache config (no contents by default) ~100B. Cold-cache-default ships ~600KB delta; warm-cache-supported ships ~2MB delta.

## Architecture spine

The cache is the shared memory. GPUs read it via the existing `EpisodicCache.query()`. The trained controller — separate-process streaming CPU SSM, C++ inference + online learning, AMX/AVX-512 — writes to the cache (admission, eviction, utility, simplex edges) by consuming a shared-memory event log of GPU writes, queries, and replay outcomes. D_global=64–128, D_slot=8–16 (per slot), 4–8 diagonal layers. fast/slow (α=0.25, interval=64 events, eval uses slow). Learns at runtime during the 600s training window from delayed reward; TTT continues at eval. Bootstrap: PyTorch pretrain on heuristic traces (behavior cloning + reward prediction), seeds the C++ runtime weights. Reward = `ce_delta_raw − bucket_baseline` (V1) with `grad_cos_rare` reserved as a Phase 4 swap-in.

## What's being learned (the heuristics we currently hardcode)

| Hardcoded today | Learned by the controller |
|---|---|
| `pressure × per-token CE quantile` in writer's `select_writes` | Admission policy per write event |
| Hardcoded `utility_u = 1.0` at admit | Initial utility prior, conditional on event features |
| Fixed EMA toward signal in `cache.update_utility` | Utility delta per replay outcome |
| `grace_steps` + utility-ranked in cache | Eviction target selection (at admission time when cache is full) |
| `cosine × utility` in `query.query_topk` | The scores the GPU's self-serve query ranks against (query primitive stays GPU-side; controller learns what `utility` and edge-derived bonuses mean) |
| Does not exist yet | Simplex edge add / remove / strengthen after each replay outcome |

Each row corresponds to an interpretation of the SSM's output for the corresponding event type.

## Event-log schema

Two-process boundary: GPU produces tensors → rank process (CPU) packs compact records into SPSC shared-memory rings → controller polls all rings round-robin. Per-rank SPSC because multi-producer would be MPSC. POSIX shm + lock-free SPSC ring per producer rank (the pre-Pass-C pattern; CPU↔GPU is the case it was designed for).

### WRITE_EVENT (~552B)

| Field | Type |
|---|---|
| event_type | u8 = 1 |
| candidate_id | u64 = `(source_rank << 56) \| rank_seq` (rank-prefixed monotonic, no global counter) |
| gpu_step | u64 |
| source_rank | u8 |
| key_fp | u64 |
| key_rep | f16[D≈256] |
| value_tok_ids | u16[span≈4] |
| value_anchor_id | u32 |
| pressure_at_write | f32 |
| pre_write_ce | f32 |
| write_bucket | u8 (token-freq bucket 0–3) |

Controller action: `(admit, initial_utility, simplex_parent, evict_target_if_full)`.

### QUERY_EVENT (~528B)

| Field | Type |
|---|---|
| event_type | u8 = 2 |
| query_id | u64 (rank-prefixed) |
| gpu_step | u64 |
| source_rank | u8 |
| query_rep | f16[D≈256] |
| pressure | f32 |
| pre_query_ce | f32 |
| bucket | u8 |

### REPLAY_OUTCOME (~96B)

| Field | Type |
|---|---|
| event_type | u8 = 3 |
| replay_id | u64 (controller-issued action id, echoed back) |
| gpu_step | u64 |
| query_event_id | u64 (links to triggering query) |
| source_write_id | u64 (links to original candidate) |
| slot_id | u32 |
| selection_step | u64 (controller step at selection time) |
| policy_version | u32 |
| selected_rank | u8 |
| teacher_score | f32 (cosine × utility from heuristic) |
| controller_logit | f32 |
| ce_before_replay | f32 |
| ce_after_replay | f32 |
| ce_delta_raw | f32 |
| bucket_baseline | f32 |
| reward_shaped | f32 (= raw − baseline; V1 default) |
| grad_cos_rare | f32 (NaN until Phase 4 rare-grad wiring; column reserved per Decision 0.10's pattern) |
| grad_cos_total | f32 (NaN until Phase 4) |
| outcome_status | u8 (ok / slot_missing / stale / nan / skipped) |
| flags | u16 |

### Throughput / sizing

At ~2M tok/sec ≈ 10K writes/sec × 552B = ~5.5MB/sec write traffic per rank, ~640KB/sec replay traffic. Per-rank ring sizing: 16MB write + 8MB query + 1MB replay-outcome = ~25MB/rank, ~100MB total across 4 ranks. Negligible against 251GB pod RAM.

## Controller-internal action logs

NOT in the shared-memory rings. Append-only NDJSON files on the controller process for audit + off-policy correction. Keep the wire schema about facts crossing the boundary; keep controller actions in the controller.

- `ADMISSION`: `(candidate_id, decision, assigned_slot_id, evicted_slot_id, initial_utility, simplex_parent, gpu_step, policy_version)`
- `REPLAY_SELECTION`: `(replay_id, query_event_id, candidates_considered, selected_slots, selection_step, policy_version)`
- `UTILITY_UPDATE`: `(replay_id, slot_id, utility_delta, gpu_step, policy_version)`
- `EDGE_UPDATE`: `(replay_id, slot_a, slot_b, edge_delta, gpu_step, policy_version)`

## Credit assignment

When REPLAY_OUTCOME arrives for slot S with reward R, the controller walks S's per-slot action history and credits each contributing action.

### Per-slot action history (in-memory ring per slot)

| Action | Stored at decision time |
|---|---|
| `ADMISSION` | candidate_id, gpu_step, policy_version, global_state (D_global), slot_state (D_slot), output_logit |
| `REPLAY_SELECTION` | replay_id, gpu_step, policy_version, global_state, slot_state, output_logit, selected_rank, query_event_id |
| `UTILITY_UPDATE` | replay_id, gpu_step, policy_version, global_state, slot_state, output_logit (= utility_delta) |
| `EDGE_UPDATE` | replay_id, gpu_step, policy_version, global_state, slot_state, output_logit (= edge_delta), neighbor_slot |

State checkpoint cost: ~1K actions/sec × (D_global + D_slot) ≈ 144 fp16 ≈ 288 B/action ≈ 170MB over a 600s training run. Trivial.

### Credit attribution (gpu_step T, slot S, reward R)

Walk S's history backward; for each action at gpu_step P with policy_version V:
- **Recency:** `credit = R · γ^(T−P)`, γ ≈ 0.995 per gpu_step (half-life ~140 steps — captures typical write→replay horizon).
- **Off-policy correction (Gerber-statistic-based, NOT importance-ratio clipping):** for the action stored at policy_version V with logit L_V, recompute current policy's logit L_current for the same action. Noise threshold `H = c · σ_logit` where σ_logit is a rolling stddev per decision type (c ≈ 0.5).

| Case | Condition | Credit weight |
|---|---|---|
| Concordant | both \|L_V\| > H AND \|L_current\| > H AND same sign | 1.0 |
| Inactive | either logit within ±H noise band | 0 (drop) |
| Discordant | both > H, opposite signs | 0 (drop) |

  Avoids the importance-ratio division-by-near-zero pathology that hard clipping at [0.5, 2.0] still has to navigate — no ratio computed at all.

- **Rank:** REPLAY_SELECTION credit multiplied by `1 / (selected_rank + 1)` — top-1 full credit, top-K attenuated.

Recompute SSM forward from the saved global_state + slot_state, get fresh output, backward against the credit-weighted target. Accumulate gradients into fast weights.

### Optimizer

SGD on fast weights, no momentum. Online RL is non-stationary; Adam's `m`/`v` moments drift, and the fast/slow EMA already provides smoothing — momentum on top is doubly-smoothed and sluggish to react. Step every 256 actions or every 1s wall-clock, whichever first. Slow weights EMA every 64 events: `slow = 0.75·slow + 0.25·fast`. Eval uses slow.

### GC

When slot S is evicted, retain its action history for ~10K events (cover in-flight replays), then drop.

### Reward shaping

V1: `reward_shaped = ce_delta_raw − bucket_baseline` (already in REPLAY_OUTCOME). The reserved `grad_cos_rare` column lets us swap to `max(0, grad_cos_rare)` once Phase 4 wires the rare-grad direction — no schema migration.

## Bootstrap (offline PyTorch pretrain)

Avoids the random-init thousand-step exploration tax during the 600s training window.

1. **Trace logging:** instrument the existing heuristic to log every decision (admit / utility / eviction) plus per-replay reward outcomes. Y-merged `episodic_replay_log_rank{R}.ndjson` already captures the replay side; add equivalent on the writer side (`select_writes` decisions) and on the cache (eviction decisions). One cd_smoke run gives ~50K+ events.
2. **Offline training in PyTorch:** small SSM, single GPU, days not weeks. Two-head pretrain:
   - **Policy head:** imitate heuristic's top-K selections (cross-entropy on score distribution over slots).
   - **Value head:** predict delayed replay reward for selected slot (regression on `reward_shaped`).
3. **Weight dump:** export trained weights to the C++ runtime's binary format. Weight surgery: PyTorch fp32 → C++ BF16 (training) + INT8 (artifact-shipping path).
4. **Online continuation:** C++ controller starts from the dumped weights. Same credit-assignment logic continues during the 600s training window from the bootstrap seed.

## Artifact layout

≤16MB total. Existing model + tokenizer ~11.5MB. Two cache modes — cold-default (V1 spec), warm-supported (matrix arm).

| Component | Status | Size (cold) | Size (warm) |
|---|---|---|---|
| Core SSM model (slow copy, bf16) | existing | ~10.5MB | ~10.5MB |
| Tokenizer (SP) | existing | ~1MB | ~1MB |
| Cache config (no contents) | NEW | ~100B | ~100B |
| Cache contents (key_rep at int8, rest fp16) | optional | 0 | ~1.4MB |
| Controller weights (slow copy, int8) | NEW | ~100–500KB | ~100–500KB |
| Running stats (bucket baselines, σ_logit, EMAs) | NEW | ~10KB | ~10KB |
| **Total delta from existing** | | ~600KB | ~2.0MB |

**Cold-default rationale:** cache contents = training-derived data (key residuals from training docs, value tokens). Borderline under Param Golf's "training-derived artifacts" framing — the same bucket as "tokenizer + offline-pretrain legality pending." The trained controller policy IS shippable (learned weights, like model weights). Cold-cache controller beating heuristic is a stronger result; attribution is to the policy, not to training-time memories.

**Warm-supported rationale:** if room exists in budget AND warm beats cold by enough to outweigh attribution-cleanliness, ship warm. Empirically resolvable.

### Format

Safetensors blob. Sapphire Rapids' big DRAM means parse overhead is irrelevant at load time. Python eval driver loads, hands shm pointers to the C++ controller process.

### Extended cache schema (in-memory only — never serialized into the eval artifact under cold mode)

- Existing W-substrate fields: capacity, span_length, key_rep_dim, fingerprint_window, grace_steps, key_fp[N], key_rep[N,D], value_tok_ids[N,span], value_anchor_id[N], utility_u[N], write_step[N], last_fired_step[N], occupied[N], fp_index
- NEW: `slot_state[N, D_slot]` — per-slot SSM state, fp16, 4096×16×2 = 128KB in memory
- NEW: `simplex_edges[N, K_max]` — per-slot outgoing edges as `(slot_id u32, edge_weight f16)`, K_max=4, 96KB in memory

C3's `to_dict`/`from_dict` are still load-bearing for trainer-side checkpointing (#103) and intra-training state, but eval artifact ships only `cache_config` under cold mode.

### Load path at eval start

1. `run_exp20_eval.py` opens the artifact via safetensors.
2. **Cold mode:** `cache = EpisodicCache(**blob["cache_config"])` — fresh empty cache, right shape. **Warm mode:** `cache = EpisodicCache.from_dict(blob["cache_full"])` — pre-populated via C3 plumbing.
3. `controller = ControllerState.from_dict(blob["controller"])` — slow weights, running stats.
4. Spawn C++ controller process; pass shm cache + controller-state pointers.
5. Eval loop runs as today; queries route to cache; writes route through controller's event log; if `controller_train_online=True`, controller's SGD updates fast → slow during eval window (mirrors training fast/slow).

## Sequencing / dependencies

- **Substrate (landed):** W's eval-cache wiring + C1/C2/C3 substrate fixes (commits `7e1cf68`, `c763892`, `a370551` on main).
- **Pre-pod prereqs (pending):**
  - **#103** trainer-side cache save: must add `fingerprint_window=` to `_construct_episodic_cache(...)` per the C1 reviewer's heads-up; otherwise silent miss the moment a config sets W ≠ 8.
  - **#104** `run_exp20_fast_score.py` cache-field wiring.
  - **#101** `pressure_at_write` field on EpisodicCache (for proper Arm B' semantics; orthogonal to controller design but unblocks pressure_only matrix arm).
  - **#95** ScOpt allreduce migration (orthogonal).
- **This design's implementation = task #105 rescoped.** Old #105 framing ("simplex + controller-memory pivot, heuristic with lifecycle defenses") is superseded.

### Implementation plan tasks (high level — feeds writing-plans skill)

1. C++ runtime scaffold: SSM forward + diagonal recurrence (AVX-512) + AMX BF16 matmul stub; weight dump format.
2. Per-slot action history + credit attribution + Gerber off-policy correction (C++).
3. Shared-memory ring buffers (POSIX shm SPSC) + protocol for the 3 wire events.
4. GPU-side instrumentation: emit WRITE_EVENT / QUERY_EVENT / REPLAY_OUTCOME from the rank processes.
5. Cache schema extensions: `slot_state` and `simplex_edges`. Update C3 `to_dict`/`from_dict` round-trip + tests.
6. Trace logging hooks on the existing heuristic (writer, cache, replay).
7. PyTorch offline pretrain pipeline (BC + reward prediction).
8. Weight dump C++ format + import path.
9. Controller spawn / lifecycle from `runner_fast_path.py` and `run_exp20_eval.py`.
10. Matrix arms: cold-cache-trained-controller, warm-cache-trained-controller, frozen-trained-controller (no eval TTT), heuristic-cold-cache (control), heuristic-warm-cache (control).
11. Falsifier analysis script extensions for the new arms.

## Open questions / TBDs

- **D_global, D_slot, num_layers:** ablation grid TBD. Initial proposal D_global=128, D_slot=16, 4 layers.
- **Recency γ:** initial 0.995 per gpu_step. Sensitivity ablation TBD.
- **Gerber noise threshold c:** initial 0.5. Sensitivity TBD.
- **Outcome_status enum exhaustiveness:** start with `{ok, slot_missing, stale, nan, skipped}`; expect to extend as edge cases surface.
- **Per-slot action history capacity:** start unbounded with GC at slot eviction; revisit if memory pressure surfaces.

## Out of scope (V1)

- Phase 4 rare-grad direction wiring (`grad_cos_rare` reward shaping). Schema-reserved, NaN-logged, no migration cost when wired.
- ScOpt / episodic compatibility (ScOpt currently gated incompatible per Decision 0.10).
- Multi-controller process fan-out (single controller is well within Sapphire Rapids budget).
- Cross-eval-document persistence of cache state (per-doc reset semantics retained from W).
