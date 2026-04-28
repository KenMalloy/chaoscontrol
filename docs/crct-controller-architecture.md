# Codex Architecture Review Part 2: Controller as Distillation Target

## Core Conceptual Correction

The controller is NOT unnecessary. It is a distilled utility predictor, but it
does not put cache reads in the timed train-rank trunk path and it must not
define the utility label.

### The Clean Split

```
rank-3 oracle:
    asks: "If memory were forcibly injected here, would it help this token?"
    bypasses controller
    computes dense signed utility

controller:
    asks: "Can I cheaply predict that utility before seeing the target?"
    learns the dense utility/readiness target

training ranks:
    use oracle utility as:
        1. token loss weights (mild, positive-only initially)
        2. controller distillation targets
    do not read CRCT memory during trunk encode
```

## Critical Correction: encode() memory_mode

The exact teacher should compare semantic stream with memory disabled vs memory forcibly injected at the REAL injection point. NOT just `hidden + memory_delta` after the fact.

### encode() API Used By CRCT

```python
def encode(self, input_ids, *, memory_mode="off",
           cache_read_cutoff=None, teacher_gate=None,
           return_controller_logits=False, return_memory_meta=False):
    """
    CRCT uses:
        "off"      -> train-rank trunk encode and oracle no-memory side
        "force_on" -> rank-3 oracle memory side
    cache_read_cutoff:
        optional MVCC event-id cutoff for append-only multislot memory reads
    """
```

## Controller Design

Continuous gate, not binary:

```python
p_help = torch.sigmoid(controller_logits)
gate = torch.clamp((p_help - 0.5) * 2.0, 0.0, 1.0)
x = x + gate[..., None] * mem_delta
```

- p_help <= 0.50 → no memory injection
- p_help = 0.75 → half-strength
- p_help = 1.00 → full injection
- Never use negative gate (anti-memory is a different mechanism)

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
        "controller_targets": sigmoid(utility / tau),  # dense calibrated signal
        "utility": utility,
    }
```

Writes from the scored batch must reserve event ids strictly newer than
`txn.read_cutoff` before they enter append-only memory. This is not just
metadata: `MultiSlotOuterModel.read(...)` and `read_bucket(...)` filter slots
by `cache_read_cutoff`, so both oracle encode passes see the same snapshot.

## Runtime Memory Ownership

Current Exp23/24 CRCT uses rank 3 as the teacher/oracle memory owner. Train
ranks do **not** read CRCT slots on the trunk forward path; they run the normal
pre-memory encoder, consume small async teacher tensors
(`target`/`confidence`/`loss_weight`/`utility`) when available, and distill the
controller head from that pre-memory hidden stream. If the teacher falls behind,
the train rank fails open for that step rather than waiting on memory.

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

There is no train-rank memory snapshot mode in the CRCT matrix. The only data
flowing back from the teacher path is the small async per-token payload:
`target`, `confidence`, `loss_weight`, and `utility`. Result JSON records
`mechanisms.crct.memory_owner="rank3_teacher_only"`,
`trunk_memory_mode="off"`, zero train-rank slot reads/writes, and rank-3
`teacher_memory_slots` so a run that accidentally re-couples trunk and memory
is visible immediately.

Rank 3 scores opportunistically rather than every train step. The default
`crct_teacher_score_interval_steps=64` keeps the teacher useful without making
oracle scoring the cadence of the trunk all-reduce. Transport is sparse and
two-rank only: rank 0 broadcasts one matched batch to rank 3 on score
boundaries, and rank 3 broadcasts the small teacher payload back to rank 0 on
the score boundary plus the next sparse publish slot. Ranks 1 and 2 never enter
teacher collectives; they stay purely on the train-rank gradient clock. Result
JSON records `teacher_transport_participant`, `teacher_bypass_steps`,
`request_interval_skips`, `broadcast_interval_skips`, payload lag, stale drops,
and score timings so a no-teacher, stale-teacher, or accidentally all-rank run
is visible from the data. The Exp24 CRCT cell writes 128 memory candidates per
teacher score; at the default cadence this still fills the 4096-slot teacher
cache during a 600s run without overfeeding the memory side.

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
ctrl_loss = BCE(controller_logits, controller_targets)
loss = lm_loss + lambda_controller * ctrl_loss  # lambda = 0.01
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
