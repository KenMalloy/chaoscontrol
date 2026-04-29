# CRCT Evidence Substrate And Packet Lane

> **Status note (2026-04-29):** This document describes the CRCT
> teacher/oracle substrate and the Exp26 trunk packet contract. Earlier
> drafts included a trunk-local `ControllerMLP`; that path has been removed.
> Exp26 uses an always-present async episodic residual lane: GPU3 emits
> `{memory_residual, memory_gate}` packets, train ranks consume them with
> `memory_mode="packet"`, and `memory_mode="off"` is retained only as a GPU3
> counterfactual/oracle mode. See `experiments/26_arm/README.md`.

## Core Conceptual Correction

The trunk should build semantic gist and cue locally, but the targeted
episodic choice happens off the trunk hot path. The utility label is defined by
GPU3 physics, not by a train-rank gate head.

### The Clean Split

```
rank-3 oracle:
    asks: "If memory were forcibly injected here, would it help this token?"
    computes dense signed utility
    exports latest-complete residual packets

CPU / maintenance controller:
    maintains evidence and schedules which memory work GPU3 should verify
    learns commit/maintenance choices from oracle feedback
    does not run on the trunk hot path

training ranks:
    use oracle utility as:
        1. token loss weights (mild, positive-only initially)
        2. residual packet gates from the teacher payload
    consume packet residuals before recurrence
    do not read CRCT memory during trunk encode
```

## Critical Correction: encode() memory_mode

The exact teacher should compare semantic stream with memory disabled vs memory forcibly injected at the REAL injection point. NOT just `hidden + memory_delta` after the fact.

### encode() API Used By CRCT

```python
def encode(self, input_ids, *, memory_mode="packet",
           cache_read_cutoff=None,
           episodic_residual=None, episodic_gate=None,
           return_memory_meta=False):
    """
    CRCT uses:
        "packet"   -> train-rank trunk encode with latest-complete residual
        "off"      -> rank-3 oracle no-memory side
        "force_on" -> rank-3 oracle memory side and packet source
    cache_read_cutoff:
        optional MVCC event-id cutoff for append-only multislot memory reads
    """
```

## Packet Design

The gate is continuous, but it is supplied by the off-path memory plane rather
than computed by a trunk-local MLP:

```python
x = x + episodic_gate[..., None] * episodic_residual
```

- missing packet → zero-residual no-op
- stale-but-safe packet → same fixed tensor lane, no trunk wait
- latest-complete packet → targeted episodic residual enters before recurrence
- negative/anti-memory residuals should be named as a separate mechanism

## Rank-3 Teacher Scoring

```python
@torch.inference_mode()
def score_memory_teacher(model, input_ids, valid_mask, *, head_chunk_size=128):
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    
    txn = cache.begin_batch()
    h_off = model.encode(x, memory_mode="off", cache_read_cutoff=txn.read_cutoff)
    h_mem = model.encode(x, memory_mode="force_on", cache_read_cutoff=txn.read_cutoff)
    
    nll_off = chunked_nll(model, h_off, y, head_chunk_size)
    nll_mem = chunked_nll(model, h_mem, y, head_chunk_size)
    
    utility = (nll_off - nll_mem) * mask.float()
    
    return {
        "loss_weights": utility_to_loss_weight(utility, mask),
        "memory_residual": force_on_memory_meta["memory_residual"],
        "memory_gate": utility_to_packet_gate(utility, mask),
        "utility": utility,
    }
```

Writes from the scored batch must reserve event ids strictly newer than
`txn.read_cutoff` before they enter append-only memory. This is not just
metadata: `MultiSlotOuterModel.read(...)` and `read_bucket(...)` filter slots
by `cache_read_cutoff`, so both oracle encode passes see the same snapshot.

## Runtime Memory Ownership

Current Exp26 ARM uses rank 3 as the teacher/oracle memory owner. Train ranks
do **not** read CRCT slots on the trunk forward path; they run the fixed packet
encoder, consume async teacher tensors
(`target`/`confidence`/`loss_weight`/`utility` plus optional
`memory_residual`/`memory_gate`) when available, and treat a missing packet as a
zero-residual no-op. If the teacher falls behind, the train rank fails open for
that step rather than waiting on memory.

The teacher-memory contract is:

```text
train ranks:
    never append CRCT slots locally
    never read CRCT slots during the timed trunk forward

rank 3:
    owns the teacher cache/memory
    scores the oracle comparison
    writes accepted candidates after scoring
```

There is no train-rank memory snapshot mode in the CRCT matrix. The data
flowing back from the teacher path is the async per-token/per-batch payload:
`target`, `confidence`, `loss_weight`, `utility`, `memory_residual`, and
`memory_gate`. Result JSON records
`mechanisms.crct.memory_owner="rank3_teacher_only"`,
`trunk_memory_mode="packet"`, zero train-rank slot reads/writes, and rank-3
`teacher_memory_slots` so a run that accidentally re-couples trunk and memory
is visible immediately.

Rank 3 scores opportunistically on the memory plane. Exp26 gives the teacher an
every-step opportunity and relies on the latest-only mailbox/backpressure to
decide what actually gets scored. Ranks 1 and 2 never enter teacher traffic;
they stay purely on the train-rank gradient clock. Result JSON records payload
lag, stale drops, packet counts/bytes, score timings, and train-rank slot
read/write counts so a no-teacher, stale-teacher, no-packet, or accidentally
all-rank run is visible from the data.

CRCT-only training syncs trunk gradients over the train-rank subgroup. Rank 3
does not join the gradient all-reduce; it receives a coalesced rank0->rank3
parameter refresh on the teacher cadence through the same two-rank teacher
group. Telemetry records `grad_sync_group="train_ranks"`,
`memory_rank_joins_grad=false`, `teacher_coordinator_rank=0`, and
parameter-sync timings so accidental re-coupling is caught.

## Gradient Conflict Sensor

Gradient conflict detection is a controller input, not a replacement
controller.  Rank 3 can cheaply sketch the LM-head gradient for tokens that
would otherwise be written to memory:

```python
logits = lm_head(final_norm(hidden_candidate))
p = softmax(logits)
p[target] -= 1
grad_sketch = normalize(p @ lm_head.weight)
conflict_cos = dot(grad_sketch, recent_accepted_grad_ema)
```

The live runner exposes this as `crct_gradient_conflict_enabled`. When enabled,
the scorer computes sketches only for the top `crct_memory_write_tokens_per_step`
write candidates. Normal controller targets, confidence, and LM loss weights do
not change. The monitor only adjusts the append-side write score; the hard
`crct_gradient_conflict_catastrophic_threshold` is a circuit breaker for
pathological anti-alignment. Soft gating is configurable but defaults to `0.0`
so the first pod runs can use it as telemetry.

Result JSON records `mechanisms.crct.gradient_conflict` from the memory rank
with counters for `calls`, `candidates_seen`, `candidates_compared`,
`admitted_candidates`, `guardrail_suppressed_candidates`,
`soft_gated_candidates`, conflict min/max/mean, and the last write-token limit.
Optional `crct_gradient_conflict_trace_path` writes sampled NDJSON rows for the
top write candidates: `step`, `candidate_rank`, `token_id`, `utility`,
`conflict_cos`, `gate`, `suppressed`, and `reason`. That is the post-run answer
to "did the memory gate suppress because it was protecting the trunk, or did the
controller simply not find useful candidates?"

## Controller Target Transform

```python
def utility_to_controller_target(utility, *, tau=0.10, floor=0.05, ceil=0.95):
    target = torch.sigmoid(utility.float() / tau)
    return target.clamp(floor, ceil).detach()
```

Scale: utility +0.20 nats → target ≈ 0.88, 0.00 → 0.50, -0.20 → 0.12

## Training-Rank Loss

```python
lm_loss = weighted_chunked_lm_loss(hidden, targets, token_weights=loss_weights)
loss = lm_loss
```

## KEY INSIGHT: Don't Downweight Hard Tokens

Negative utility means "memory hurts here", NOT "this token is unimportant."

Use negative utility to CLOSE THE GATE, not to ignore the token.

First version: positive-only LM weighting:
```python
def positive_only_lm_weight(utility, valid_mask, *, tau=0.10, strength=0.15, w_max=1.20):
    pos = torch.relu(torch.tanh(utility.float() / tau))
    w = 1.0 + strength * pos
    w = w.clamp(1.0, w_max) * valid_mask.float()
    mean_w = w.sum().clamp_min(1e-6) / valid_mask.sum().clamp_min(1.0)
    return (w / mean_w.detach()).detach()
```

## Staged Implementation

### Stage 0: Shadow Mode
Rank 3 computes utility but ranks 0-2 ignore it. Measure: step time impact, utility distribution sanity, correlation with existing ce_delta_raw.

### Stage 1: Controller Distillation Only
`loss = lm_loss + 0.01 * ctrl_loss`. No LM reweighting. Test: can controller learn memory-help/hurt boundary?

### Stage 2: Mild Positive-Only LM Weighting
`alpha = 0.05` initially, maybe up to 0.15.

### Stage 3: Symmetric Weighting (only if proven)

## Critical Warnings

1. **Causal leakage**: Score batch BEFORE writing it to cache. Wrong order will look amazing and fail eval.
2. **Don't let controller define teacher labels**: force_on vs off, NOT predicted-gate vs off.
3. **Stale teacher is OK for controller supervision, less OK for aggressive LM reweighting.**
4. **Rank 3 is opportunistic. Late labels are dropped. Training never waits.**

## One-Sentence Summary

The rank-3 oracle idea is wise; using it as controller supervision is the clean win; using it to aggressively reshape LM loss is the part that could send you into lala land.
