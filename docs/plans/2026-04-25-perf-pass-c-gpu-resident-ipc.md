# Perf Pass C — GPU-resident episodic IPC

> **Status:** Superseded on 2026-04-26. This draft's `dist.gather` path proved to be a synchronous train-step bottleneck. The current runner uses async per-rank `ShmRingWriteEvent` producers plus an episodic-rank drain thread: write memory is publish-or-drop and never adds a trunk collective.
>
> Kept for historical context because it explains the slot-tensor format and the abandoned gather tradeoff.
>
> **Original status:** Design draft. Implementation lands after Phase 1 (Tasks 1.4 + 1.5) merges.
> **Goal:** Eliminate CPU↔GPU traversal in the episodic write path. Replace POSIX shm rings with a single per-step `dist.gather` collective on GPU. Collapses items 7 + 8 from the perf hit list into one architectural pivot.

## Why

The Phase 1 IPC pipeline is:
```
train_rank.GPU_hidden → .cpu().numpy() → POSIX shm slot → episodic_rank.CPU buffer
                                                        ↓
                                              torch.from_numpy.copy() → episodic_rank.GPU cache
```

Two sync GPU↔CPU transfers per slot. At K=1 per step it's invisible; at K>10 it starts showing. More importantly, it contradicts the locked architectural commitment: **cache lives in GPU memory; the CPU controller orchestrates without owning data** (decision 0.7). The shm path forces data through CPU on every write — a structural mismatch.

NCCL gather sends payloads GPU-to-GPU directly. Train ranks emit; episodic rank receives. Zero CPU traversal.

## Topology

World size N (4 in test, 8 in prod). Ranks 0..N-2 are train. Rank N-1 is episodic.

**Per training step, after main backward:**
```
all ranks → dist.gather(slot_tensor, dst=N-1) → episodic rank holds [N, K_max, slot_dim] tensor
                                                                    ↓
                                                              filter by valid_mask
                                                                    ↓
                                                          cache.append(...) per valid slot
                                                                    ↓
                                                  push (b, t, pressure, residual) to controller queue
```

Episodic rank's `gather` receives a stack of all ranks' contributions; train ranks each emit their own padded tensor and continue.

The `gather` is a single collective — implicit synchronization, no extra `dist.barrier` needed.

## Slot format — single fp32 tensor

One slot is a contiguous fp32 tensor of shape `[slot_dim]` with this layout:

| Offset | Field | Width | dtype (interpretation) |
|---|---|---|---|
| 0 | `valid_mask` | 1 | fp32 (0.0 or 1.0) |
| 1 | `pressure` | 1 | fp32 |
| 2 | `key_fp` | 2 | int64 reinterpret |
| 4 | `value_anchor_id` | 2 | int64 reinterpret |
| 6 | `value_tok_ids` | 2*S | int64 reinterpret of S elements |
| 6+2S | `key_rep` | D | fp32 |
| 6+2S+D | `residual` | D | fp32 |

`slot_dim = 6 + 2S + 2D`. At S=4, D=256: `slot_dim = 526` fp32 = 2104 bytes per slot.

`int64` fields occupy 2 fp32 slots via `view(torch.int64)` reinterpret. This avoids needing a struct-dtype mixed payload, keeps the gather a single contiguous tensor.

K_max = 16 default (sweepable knob). Per-rank emit tensor: `[K_max, slot_dim]` = 16 × 2104 = ~33 KB per rank per step. Trivial NCCL bandwidth.

**Single slot type for write + query.** The slot carries both the write payload (key_fp, key_rep, value_tok_ids, value_anchor_id) AND the query candidate fields (pressure, residual). Episodic rank routes to two consumers:
- Cache writer: appends `(key_fp, key_rep, value_tok_ids, value_anchor_id)` to `EpisodicCache`
- Controller queue: pushes `(rank, b, t, pressure, residual)` for Phase 2 query handling

This collapses two POSIX shm rings into one NCCL channel. Item 8 of the perf list becomes free.

## Train-rank emit logic (replaces the K-loop write_ring + query_ring code from Task 1.4)

```python
# After main backward, per_token_ce in scope
T = input_ids.size(1)
pressure_full = _right_pad_per_token_signal(pressure, T)  # [B, T]
ce_full       = _right_pad_per_token_signal(per_token_ce, T)
positions     = select_top_p_positions(pressure_full * ce_full, top_p=top_p)  # [K, 2]

# Build batched payloads on GPU (no .cpu() calls)
slot_tensor = torch.zeros(K_max, slot_dim, device=device, dtype=torch.float32)
n_valid = positions.size(0)  # ≤ K_max
for k in range(n_valid):
    b, t = int(positions[k, 0]), int(positions[k, 1])
    if t < W or t + S > T:  # boundary check, same as build_write_payload
        continue
    slot_tensor[k, 0] = 1.0  # valid_mask
    slot_tensor[k, 1] = pressure_full[b, t]
    # key_fp: rolling hash; computed in Python (small) and packed via int64 view
    fp = fingerprint_tokens(input_ids[b, t-W:t])
    slot_tensor[k, 2:4].view(torch.int64)[0] = fp
    slot_tensor[k, 4:6].view(torch.int64)[0] = int(target_ids[b, t].item())  # value_anchor_id
    slot_tensor[k, 6:6+2*S].view(torch.int64).copy_(target_ids[b, t:t+S].to(torch.int64))
    slot_tensor[k, 6+2*S:6+2*S+D].copy_(hidden[b, t].detach())  # key_rep
    slot_tensor[k, 6+2*S+D:].copy_(hidden[b, t].detach())  # residual (same vector for now)

# Single GPU-to-GPU collective; dst = episodic rank
gather_list = [torch.zeros_like(slot_tensor) for _ in range(world_size)] if rank == episodic_rank else None
dist.gather(slot_tensor, gather_list=gather_list, dst=episodic_rank, group=all_group)
```

Note: `key_rep` and `residual` are the SAME vector (the late-residual at write position). The contract treats them as conceptually distinct (write key vs query vector), but they're computed from the same source. We just store once; the consumer routes.

The Python loop over K is fine because K is small (default top_p × B × T ≈ 1, ramps to ~16 at max). For larger K the loop body could be vectorized; defer until measurement says it matters.

The fingerprint is computed in Python and packed via int64-view into the fp32 slot. This is the only CPU work per write — fingerprint computation is unavoidable (it's a sequence hash) but takes O(W) per payload, ~µs.

## Episodic-rank drain logic (replaces the ring-attach + drain loop from Task 1.5)

```python
# all_received: [N, K_max, slot_dim] after gather
slots = torch.cat(gather_list, dim=0)  # [N * K_max, slot_dim]
valid = slots[:, 0] > 0.5  # [N * K_max] bool
valid_slots = slots[valid]  # [n_valid, slot_dim]

for i in range(valid_slots.size(0)):
    slot = valid_slots[i]  # [slot_dim] still on GPU
    key_fp           = int(slot[2:4].view(torch.int64).item())
    value_anchor_id  = int(slot[4:6].view(torch.int64).item())
    value_tok_ids    = slot[6:6+2*S].view(torch.int64).clone()  # [S] int64 GPU tensor
    key_rep          = slot[6+2*S:6+2*S+D].clone()              # [D] fp32 GPU tensor
    residual         = slot[6+2*S+D:].clone()                   # [D] fp32 GPU tensor
    pressure         = float(slot[1].item())

    cache.append(
        key_fp=key_fp,
        key_rep=key_rep,
        value_tok_ids=value_tok_ids,
        value_anchor_id=value_anchor_id,
        current_step=current_step,
        embedding_version=embedding_version,
    )

    # Phase 2 controller queue (in-process; the controller process reads from
    # this in a Phase 2 task. For Phase 1 we just count.)
    controller_query_queue.append({
        "step": current_step,
        "rank": int(i // K_max),
        "pressure": pressure,
        "residual_handle": residual,  # GPU tensor, valid until next step's clone() overwrites
    })
```

The `.item()` calls for `key_fp`, `value_anchor_id`, and `pressure` are unavoidable — the cache schema and controller queue both need scalar values, and these are tiny syncs (~µs each, K total per step).

`EpisodicCache.append` already moves tensors to its internal storage device on append (cache lives on episodic rank's GPU; storage tensors are pre-allocated on that device). The `.clone()` calls ensure the slot tensor can be reused next step.

## Synchronization model

**No init barrier for IPC setup.** There's no shm to coordinate; the cache is local to the episodic rank, and the gather collective IS the rendezvous. The first `dist.gather` happens in step 0, after `dist.init_process_group` returns and `all_group` is created.

**Per step, the gather is a hard sync point** for all ranks. Train ranks block at the gather until episodic rank reaches it (and vice versa). This is the same semantic as the existing `allreduce_grads` collective — no new sync class.

**Order of collectives per step:**
1. Train ranks: forward → backward → pre-scale grads
2. Episodic rank: skip-main → ready
3. ALL ranks: `dist.gather` (this task's new collective)
4. Episodic rank: drain into cache (CPU work, no collective)
5. ALL ranks: `allreduce_grads(SUM, group=all_group, materialize_zeros=True)`
6. ALL ranks: optimizer.step()

The gather happens BEFORE the all-reduce because the cache append is a side-effect we want completed before the optimizer step (in case Phase 3's replay backward fires this step and depends on the cache being current). For Phase 1, replay isn't in the picture, so the order is structural-only.

## Failure modes

- **NCCL timeout** if one rank stalls. Default 30 min on NCCL backend; gloo backend has shorter timeouts. Same risk class as existing all-reduce; no new mitigation needed.
- **Rank death.** Communicator goes bad; subsequent collectives fail. Same as existing path.
- **Episodic rank's drain too slow.** The gather is synchronous; if drain takes longer than the train ranks' next-step prep, train ranks block. At K_max=16 and ~µs per cache.append, drain is microseconds — way under any train-step budget.
- **K > K_max.** `select_top_p_positions` returns up to K = round(B*T*top_p) positions. We bound the slot tensor at K_max. If a config wants K > K_max, the slot tensor truncates silently. Add a runtime assert: `K <= K_max`. Default K_max = 16 covers `top_p ≤ 16/(B*T)` which is way above Phase 1's `1/(B*T)`.

## What this obsoletes from Phase 1

- `src/chaoscontrol/episodic/ipc.py` (the POSIX shm SPSC ring) — no longer used in episodic path. Keep the module for now (it has 11 tests pinning behavior; might find another use), but mark as deprecated for the episodic IPC path.
- `src/chaoscontrol/episodic/payload_dtypes.py` (numpy struct dtypes for shm slots) — same. The new path uses fp32 tensors, not numpy structs.
- Task 1.4's `_create_episodic_rings`, ring `try_write` calls, `close_and_unlink` shutdown.
- Task 1.5's `_attach_episodic_consumer`, ring `try_read` drain loop, ring `close()` shutdown.
- Task 2.0's metadata validation in `ShmRing.attach` — irrelevant if we're not using ShmRing for episodic.
- The contract doc (`docs/plans/2026-04-25-ring-contract-tasks-1-4-and-1-5.md`) — supplanted by this design.

This is a lot of code that gets ripped out. Roughly 800 lines of POSIX shm machinery → ~150 lines of NCCL gather + slot pack/unpack. Net reduction.

The throwaway is intentional. Phase 1 was about *whether the IPC path works at all*; Pass C is about whether it can work fast enough for Phase 3+ scale.

## Implementation tasks (Pass C, sequenced)

- **C.1: Slot format helpers.** New module `src/chaoscontrol/episodic/gpu_slot.py`. `pack_payload(slot, b, t, ...)` and `unpack_payload(slot)`. Tests pin the int64-reinterpret round-trip + boundary-skip semantics.
- **C.2: Train-rank emit replacement.** Strip `_create_episodic_rings`, ring writes, ring shutdown from runner. Insert the slot-tensor + `dist.gather` block in the train-rank branch of `_run_train_step`.
- **C.3: Episodic-rank drain replacement.** Strip `_attach_episodic_consumer`, ring drain. Insert `dist.gather` (receiver side) + slot unpack + `cache.append` in the episodic-rank branch of `_run_train_step`.
- **C.4: Controller query queue.** In-process Python list on the episodic rank for Phase 2 readers. Replaces the POSIX query-ring.
- **C.5: Tests.** mp.spawn 4-rank gloo smoke proving end-to-end (write → gather → drain → cache fills with correct contents). Reuse the test patterns from Tasks 1.4 + 1.5.
- **C.6: Update plan doc.** Mark items 1, 2, 5 of the perf hit list as obsolete (subsumed by C). Update the architecture section.

## Open questions for review

- **K_max choice.** 16 is conservative. At Phase 1 default top_p ≈ 1, almost always 1-2 valid slots per rank. Larger K_max wastes bandwidth (padded zeros) but raises the safe ceiling. Pick 16 unless someone has a reason for higher.
- **Slot fp32 reinterpretation safety.** `view(torch.int64)` requires aligned memory and matching strides. The slot tensor is contiguous fp32, so view-as-int64 works on the offsets we picked (all even). Verify with a unit test.
- **`gather` vs `all_gather`.** Episodic rank is the only consumer; `gather` to dst=episodic_rank is correct. `all_gather` would broadcast to every rank, costing N× bandwidth for no benefit.
- **Should we keep `src/chaoscontrol/episodic/ipc.py` or delete it?** It has 11 tests + clean implementation. Could be useful for non-episodic IPC (e.g., the controller process talking to the runner). Recommend keep + mark deprecated for episodic; revisit for delete after Phase 5.
