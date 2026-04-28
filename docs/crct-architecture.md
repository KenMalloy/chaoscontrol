# Rank-3 Asynchronous Cache-Utility Teacher for ChaosControl

Expert architecture review (via Codex). This is the reference implementation plan for CRCT.

## Core Design Principle

Treat rank 3 as an **asynchronous utility teacher**, not part of the training DDP graph.

```
ranks 0,1,2:
    train normal ChaosStudentLM / train_ssm path
    DDP group contains only ranks 0,1,2
    optionally consume token weights from rank 3
    never wait for rank 3
    if weights are late: train with weight = 1.0
rank 3:
    no backward
    no optimizer step
    holds stale-but-recent model copy + cache
    scores future batches or prefetched batches
    computes dense utility[token] = NLL_no_cache[token] - NLL_with_cache[token]
    converts utility to bounded loss weights
    sends fp16 weights back to owning training rank
```

**Critical invariant: Rank 3 may fall behind. Rank 3 may be stale. Rank 3 may drop a batch. Ranks 0-2 must never wait for it.**

## Key Repo-Specific Findings

1. `ChaosStudentLM.encode()` returns pre-head hidden state, has no memory-write side effects — perfect hook for rank-3 scoring
2. Current typed-buffer read path computes dominant bucket per sample, reads once per sample/bucket, broadcasts across sequence. True per-token reads were avoided because of O(batch*seq) cost — but rank 3 is idle, so rank 3 is where expensive per-token retrieval belongs
3. train_ssm.py already uses chunked LM backward to avoid materializing huge logits — CRCT patches into this cleanly

## Implementation: 8 Pieces

### 1. Patch encode() with cache-read toggle

```python
def encode(self, input_ids: torch.Tensor, *, cache_read: bool = True) -> torch.Tensor:
```

Wrap only the cache/episodic read path with `if cache_read and ...`

### 2. Cache utility scoring (src/chaoscontrol/cache_utility.py)

```python
@torch.inference_mode()
def nll_from_hidden_chunked(model, hidden, targets, *, chunk_size=128):
    B, T, _ = hidden.shape
    out = []
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        h = hidden[:, s:e]
        y = targets[:, s:e]
        logits = model.lm_head(model.final_norm(h)).float()
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).view(B, e - s)
        out.append(nll)
    return torch.cat(out, dim=1)

def utility_to_loss_weight(utility, valid_mask, *, tau=0.10, strength=0.25, w_min=0.75, w_max=1.25):
    signed = torch.tanh(utility.float() / tau)
    weights = 1.0 + strength * signed
    weights = weights.clamp(w_min, w_max) * valid_mask.float()
    # Mean-1 normalize to preserve effective LR
    mean_w = weights.sum().clamp_min(1e-6) / valid_mask.sum().clamp_min(1.0)
    return (weights / mean_w.detach()).detach()
```

### 3. Process groups (ranks 0-2 in DDP, rank 3 separate)

```python
TRAIN_RANKS = (0, 1, 2)
UTILITY_RANK = 3

def make_utility_groups():
    train_pg = dist.new_group(list(TRAIN_RANKS))
    util_pgs = {r: dist.new_group([r, UTILITY_RANK]) for r in TRAIN_RANKS}
    sync_pg = dist.new_group([0, UTILITY_RANK])
    return train_pg, util_pgs, sync_pg
```

All DDP, Muon, fused grad clip collectives MUST use train_pg, not default world group.

### 4. Async weight mailbox (non-blocking P2P on dedicated CUDA stream)

- Dedicated CUDA stream for communication
- Ring buffer of depth=8 receive slots
- `try_get(step)` never blocks — returns None if weights aren't ready
- `post_recv(step)` posts future receives
- Training ranks use `utility_ahead_steps = 4` prefetch

### 5. Patch chunked loss in train_ssm.py

Multiply per-token CE by optional token_weights in the chunked head path. `token_weights.detach()` always.

### 6. Rank-3 sender

Deterministic per-owner-rank FIFO order (NCCL has no tags). Separate process group per training rank.

### 7. Future batch source

Option A (preferred): Rank 3 reconstructs future batches from deterministic SP-shard iterators
Option B: Ranks 0-2 send future input_ids as int32 (~2MB per batch)

### 8. Model-copy sync

Sync rank-3 model every 32-64 optimizer steps. Clone+flatten bf16 snapshot on rank 0, isend to rank 3 on sync_pg. Never inside forward/backward critical path.

## Synchronization Pitfalls

1. Never block on utility at loss time — use try_get, fallback to weight=1.0
2. P2P on dedicated CUDA stream, never default training stream
3. No NCCL tags — use per-pair process groups + FIFO
4. Rank 3 must not participate in training collectives
5. Don't reuse send buffers before completion
6. Cache reads must be read_only=True, no LRU updates, no writes
7. No hidden syncs in hot loop (no .item(), .cpu(), print, synchronize, barrier)

## Tuning Parameters

```
tau = 0.10        # tanh temperature (±0.2 nats ce_delta → strong response)
strength = 0.25   # weight deviation from 1.0
w_min = 0.75      # clamp floor
w_max = 1.25      # clamp ceiling
```

## Rank-3 Scoring Summary

```python
hidden = model.encode(x, cache_read=False)
cache_delta = cache.read_dense(hidden.detach(), read_only=True, per_token=True)
nll_base = CE(head(hidden), y)
nll_mem = CE(head(hidden + cache_delta), y)
utility = nll_base - nll_mem
weights = clamp_norm(1 + strength * tanh(utility / tau))
# send fp16 weights to owning rank
```
