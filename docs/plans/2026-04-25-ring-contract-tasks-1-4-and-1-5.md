# Ring Contract — Phase 1 Tasks 1.4 & 1.5

> **Purpose:** Lock the IPC contract between train ranks (writers) and the episodic rank (consumer + cache) so Tasks 1.4 and 1.5 can be implemented in parallel worktrees against a stable interface. The merge happens 1.4 first (defines the interface), then 1.5 (consumes it).

This document is the source of truth. If either implementer needs to deviate, they MUST flag it before merging.

---

## Ring inventory

For a runner with `world_size = N` and `episodic_enabled = True`:

| Ring name pattern | Producer (creates) | Consumer (attaches) | Capacity (config key) | Slot dtype |
|---|---|---|---|---|
| `episodic_write_ring_rank{R}` (R = 0..N-2) | Train rank R | Episodic rank N-1 | `episodic_write_ring_capacity` (default 256) | `_WRITE_PAYLOAD_DTYPE` (see below) |
| `episodic_query_ring_rank{R}` (R = 0..N-2) | Train rank R | (Phase 2 controller, not Phase 1) | `episodic_query_ring_capacity` (default 256) | `_QUERY_CANDIDATE_DTYPE` (see below) |

**Phase 1 scope:** Task 1.4 creates BOTH ring kinds on train ranks (the query rings will fill but are not yet drained — Phase 2 wires the controller). Task 1.5 drains ONLY the write rings on the episodic rank. The query ring is producer-only-active in Phase 1.

---

## Slot dtypes

Both dtypes are `numpy.dtype` structs (named fields). They are defined ONCE in `src/chaoscontrol/episodic/payload_dtypes.py` (new file, owned by Task 1.4) and imported by both Task 1.4 (producer) and Task 1.5 (consumer).

### `_WRITE_PAYLOAD_DTYPE`

Mirrors `chaoscontrol.optim.episodic_writer.WritePayload`. Single 1-D slot per write:

```python
import numpy as np

# These two dimensions parameterize the dtype at runner init time.
def make_write_payload_dtype(span_length: int, key_rep_dim: int) -> np.dtype:
    return np.dtype([
        ("key_fp",          np.int64),                      # rolling-hash fingerprint
        ("key_rep",         np.float32, (key_rep_dim,)),    # late-residual at write position
        ("value_tok_ids",   np.int64,   (span_length,)),    # next S target tokens
        ("value_anchor_id", np.int64),                      # value_tok_ids[0]
    ])
```

The exact tuple `(span_length, key_rep_dim)` is read from runner config. Default in Phase 1: `span_length=4`, `key_rep_dim=model_dim` (typically 256 in current configs).

**At runner init**, the dtype is computed once and passed to `ShmRing.create(dtype=...)`. Both producer and consumer use this single dtype.

### `_QUERY_CANDIDATE_DTYPE`

Single 1-D slot per query candidate:

```python
def make_query_candidate_dtype(key_rep_dim: int) -> np.dtype:
    return np.dtype([
        ("batch_index",  np.int64),
        ("position",     np.int64),
        ("pressure",     np.float32),
        ("residual",     np.float32, (key_rep_dim,)),  # late-layer residual at (b, t)
    ])
```

Phase 1 fills these but does not consume them. Phase 2 Task 2.x adds the controller-side reader.

---

## Lifetime + ownership

**At runner init, after `dist.init_process_group(...)` and before the train loop:**

- **Train rank R** (where `R != world_size - 1`):
  - `write_ring = ShmRing.create(name=f"episodic_write_ring_rank{R}", slot_shape=(), dtype=_WRITE_PAYLOAD_DTYPE, capacity=config["episodic_write_ring_capacity"])`
  - `query_ring = ShmRing.create(name=f"episodic_query_ring_rank{R}", slot_shape=(), dtype=_QUERY_CANDIDATE_DTYPE, capacity=config["episodic_query_ring_capacity"])`
  - Stash both in a runner-state struct accessible to `_run_train_step` (e.g., a dict keyed by ring kind, or new fields on the train-step closure).

- **Episodic rank** (where `R == world_size - 1`):
  - `write_rings = [ShmRing.attach(name=f"episodic_write_ring_rank{R}", ...) for R in range(world_size - 1)]`
  - `cache = EpisodicCache(capacity=..., span_length=..., key_rep_dim=..., grace_steps=..., utility_ema_decay=...)` — same kwargs the runner reads from config (defaults match Decision 0.4 of the plan: capacity=4096, span_length=4, key_rep_dim=model_dim, grace_steps=1000, utility_ema_decay=0.99).
  - Heartbeat counter (a single int, in-process; episodic rank increments each step).

- **Synchronization at init:** Train ranks must `dist.barrier()` AFTER `ShmRing.create()` so the episodic rank's `attach()` doesn't race the create. Use the existing `all_group` from Task 1.3 (it's the WORLD-equivalent group when episodic_enabled=True).

**At runner shutdown:**
- Train ranks call `write_ring.close_and_unlink()` and `query_ring.close_and_unlink()` (each rank owns the unlink for its own rings).
- Episodic rank calls `close()` (consumer side, no unlink) on each attached write ring.
- Episodic rank does NOT touch query rings in Phase 1.

---

## In-step behavior

### Train ranks (Task 1.4)

In `_run_train_step`'s train-rank branch (the `else` of `is_episodic_rank`), AFTER the existing main forward+backward+pressure computation but BEFORE the train-rank pre-scaling and all-reduce:

```python
# Existing: compute pressure, per_token_ce, hidden states (the "residual" we need)
# pressure, per_token_ce, hidden are already in scope at this point

# 1. Shape adapter: per_token_ce is [B, T-1]; pad to [B, T] with zero on last col
pressure_full = _right_pad_per_token_signal(pressure, T=input_ids.size(1))
ce_full       = _right_pad_per_token_signal(per_token_ce, T=input_ids.size(1))
# (helper from Phase 1 Task 1.4 of plan; copy from plan if not yet defined)

# 2. Top-p selection — the SAME positions drive both writes and query candidates
write_signal  = pressure_full * ce_full
positions     = select_top_p_positions(write_signal, top_p=config["episodic_top_p"])
# positions: [K, 2] of (batch_index, position)

# 3. Build write payloads (using existing build_write_payload from episodic_writer)
for k in range(positions.shape[0]):
    b, t = int(positions[k, 0]), int(positions[k, 1])
    payload = build_write_payload(
        batch_index=b, position=t,
        input_ids=input_ids,
        target_ids=target_ids,
        key_rep_per_position=hidden.detach(),  # late-layer residual is the encode output
        fingerprint_window=config["episodic_fingerprint_window"],
        span_length=config["episodic_span_length"],
    )
    if payload is None:
        continue
    # Pack into the numpy struct dtype
    slot = np.zeros((), dtype=_WRITE_PAYLOAD_DTYPE)
    slot["key_fp"]          = payload.key_fp
    slot["key_rep"]         = payload.key_rep.cpu().numpy()
    slot["value_tok_ids"]   = payload.value_tok_ids.cpu().numpy()
    slot["value_anchor_id"] = payload.value_anchor_id
    write_ring.try_write(slot)

    # Same position → also a query candidate (Phase 2 will consume)
    qslot = np.zeros((), dtype=_QUERY_CANDIDATE_DTYPE)
    qslot["batch_index"] = b
    qslot["position"]    = t
    qslot["pressure"]    = float(pressure_full[b, t].item())
    qslot["residual"]    = hidden[b, t].detach().cpu().numpy().astype(np.float32)
    query_ring.try_write(qslot)
```

### Episodic rank (Task 1.5)

In `_run_train_step`'s episodic-rank branch (the `if is_episodic_rank` block from Task 1.3, currently returning a zero placeholder), BEFORE the placeholder return:

```python
# Drain every train rank's write ring this step.
total_drained = 0
for ring in write_rings:
    while True:
        slot = ring.try_read()
        if slot is None:
            break
        cache.append(
            key_fp=int(slot["key_fp"]),
            key_rep=torch.from_numpy(slot["key_rep"].copy()),
            value_tok_ids=torch.from_numpy(slot["value_tok_ids"].copy()),
            value_anchor_id=int(slot["value_anchor_id"]),
            current_step=current_step,
            embedding_version=embedding_version,
        )
        total_drained += 1

heartbeat += 1
# (existing) return zero placeholder loss
```

**`current_step`** comes from the train loop's step counter (already in scope on every rank).
**`embedding_version`** is 0 for Phase 1; Phase 5 introduces refresh and bumps this.

---

## Telemetry contract (consumed by Task 1.7)

Task 1.4 must emit per-step:
- `episodic_writes_this_step` — number of payloads pushed to write_ring (per train rank)
- `episodic_query_candidates_this_step` — number pushed to query_ring (per train rank)
- `episodic_write_ring_dropped` — current `dropped_count()` from the train rank's write ring

Task 1.5 must emit per-step (rank 0 logs since episodic rank's data flows back via existing logging):
- `episodic_drained_this_step` — total payloads drained across all train ranks' write rings
- `episodic_cache_len` — `len(cache)` after the drain
- `episodic_rank_heartbeat_age_steps` — steps since last heartbeat advance (always 0 in steady state)

The exact JSONL schema is Task 1.7's job; both 1.4 and 1.5 just need to publish these into the runner's existing telemetry dict.

---

## Config keys introduced by Tasks 1.4 + 1.5

Add to runner config validation (with defaults):

| Key | Default | Owner | Notes |
|---|---|---|---|
| `episodic_top_p` | `1.0 / (B * T)` (computed) | 1.4 | One position per batch step in expectation |
| `episodic_fingerprint_window` | `8` | 1.4 | Tokens preceding write position for rolling hash |
| `episodic_span_length` | `4` | 1.4 | Cache value span length |
| `episodic_write_ring_capacity` | `256` | 1.4 | Per-rank write-ring buffer size |
| `episodic_query_ring_capacity` | `256` | 1.4 | Per-rank query-ring buffer size |
| `episodic_capacity` | `4096` | 1.5 | Cache capacity |
| `episodic_grace_steps` | `1000` | 1.5 | Cache eviction grace period |
| `episodic_utility_ema_decay` | `0.99` | 1.5 | Cache utility EMA decay |

`episodic_enabled` and `episodic_key_rep_dim` were introduced in Task 1.3 (or are derivable from `model_dim`).

---

## Merge protocol

1. **Task 1.4 lands first** on main. This defines `payload_dtypes.py`, the ring naming convention, and the producer side.
2. **Task 1.5's worktree rebases on the post-1.4 main** before merging. The 1.5 implementer reads `payload_dtypes.py` from the rebased state and verifies the slot dtype matches what they coded against.
3. After both land, the integrated 4×H100 (3+1) smoke (Task 1.6) verifies the full pipeline.

## What each implementer MUST verify against this contract

Before reporting "done":
- [ ] Ring names match the patterns above EXACTLY (consumer-side attach must succeed against producer-side create).
- [ ] Slot dtypes match (verified via `ShmRing.attach` metadata validation from Task 2.0 — mismatch raises ValueError).
- [ ] All config keys land in the runner's config dict with the documented defaults.
- [ ] `dist.barrier()` synchronization at init is in place so attach doesn't race create.
- [ ] Task 1.5 closes attached rings (consumer `close()`, no unlink) at runner shutdown.
- [ ] Task 1.4 unlinks created rings (producer `close_and_unlink()`) at runner shutdown.

If your implementation deviates from any line in this document, flag it in the implementer report. Don't silently change the contract.
