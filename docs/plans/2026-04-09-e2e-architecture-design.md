# Experiment 14: VRAM-Resident Memory at Scale

## Grand Thesis

Two stacked claims:

**Claim 1 (primary):** A rebuildable typed KV buffer can buy an SSM
transformer-quality context access without paying transformer-style
attention cost. The SSM trunk IS the attention (O(1) per step). The buffer
IS the lossless backup. Retrieval is O(k) within a typed bucket. The buffer
rebuilds at test time from the input stream — nothing ships but weights
and structure.

**Claim 2 (secondary, contingent on Claim 1):** Developmental fast weights
and online defrag may further improve buffer quality and cold-start
performance. Test only after the typed-buffer story is proven.

Steps are sacred. Memory is state, not compute. Even a 1%-correct retrieval
signal, compounded over thousands of steps, closes the gap.

## Architecture

```
Input (raw bytes)
  |
  v
Byte Embedding ---- No learned tokenizer. Raw bytes.
  |
  v
Wernicke Routing --- MoE: type input into k_max semantic buckets.
  |                   Possibly hierarchical (2 layers for finer typing
  |                   at lower routing cost). Ships in the artifact.
  v
KV Buffer ---------- VRAM-resident, append-only during training.
  |                   Every token's KV pair stored unconditionally.
  |                   No surprise gating. No compression. No maintenance.
  |                   Retrieval: within matching Wernicke bucket only.
  |                   At test time, rebuilds from the input stream.
  |                   At experiment scale (~1500 steps), buffer fits
  |                   easily in VRAM. Capacity is not a constraint.
  v
Bucket Prototypes -- Per-bucket semantic priors (one per Wernicke type).
  |                   NOTE: current SemanticTier is a global basis bank.
  |                   Per-bucket conditioning is NEW architecture work.
  |                   Ships in artifact for cold-start context (~2KB).
  v
SSM Backbone ------- dim=128, 4 layers, diag A-mode, criticality
  |                   target=0.92 (from exp 13). State update IS the
  |                   attention: h_{t+1} = A*h_t + B*x_t. O(1) per step.
  v
LM Head ------------ Predict next byte.
```

**Cut from prior architecture:** Metabolic gate (violates step-count
thesis — costs forward passes for marginal benefit).

### Forward pass (training and inference — identical)

1. Embed raw bytes
2. Wernicke types the input, routes through expert(s)
3. Buffer read: project query, retrieve within matching bucket
4. Bucket prototype: add per-type prior bias
5. SSM recurrence through 4 layers
6. Buffer write: append KV pair to matching bucket (unconditional)
7. Predict next byte

### Buffer retrieval modes (to be ablated)

| Mode | Mechanism | Character |
|------|-----------|-----------|
| bucket_mean | Average all entries in bucket | Simplest — routing IS the retrieval |
| bucket_recent | Most recent k entries in bucket | Recency bias |
| bucket_topk | Top-k by dot product similarity | Selective attention within bucket |
| softmax_all | Softmax over all slots (current) | Baseline, no bucket scoping |

### Hierarchical Wernicke

Two cheap routing decisions beat one hard one. Cost comparison at
dim=128, batch=32, seq=256 (~2.7 GFLOP SSM backbone per step):

| Config | Total buckets | Cost/token | % of SSM |
|--------|--------------|-----------|----------|
| flat_16 | 16 | ~35K FLOP | ~13% |
| flat_64 | 64 | ~41K FLOP | ~16% |
| hier_8x8 | 64 | ~70K FLOP | ~27% |
| hier_8x32 | 256 | ~71K FLOP | ~27% |

Hierarchical gives 256 buckets for ~27% overhead vs flat_256 at ~25%.
Nearly same cost but routing accuracy should be higher and cache miss
rate lower (coarse bucket as fallback when fine bucket is sparse).

## Headline Metrics

bpb_pretrain is a training diagnostic only — it reflects performance
with a long-lived buffer that doesn't ship. Not a headline.

| Metric | Buffer state | Role |
|--------|-------------|------|
| **bpb_artifact_cold** | Empty (16MB artifact only) | How good are the weights alone? |
| **bpb_ttt_after_N** | Rebuilding from test data | Production performance |

Report bpb_ttt as a warming curve: bpb at N=0 (cold), 100, 500, 1000,
5000 tokens. The curve shape IS the thesis — steep warming = the typed
buffer works. The gap between cold and warm = buffer value.

### TTT Evaluation Contract

To keep TTT results fair and reproducible, the protocol is fixed:

1. Use a locked held-out validation slice that is never used for T2-T6
   model selection.
2. Build fixed evaluation segments of length at least `N_max + 1024`
   tokens (e.g. 8192 bytes for `N_max=5000`).
3. For each segment and each `N` in `{0, 100, 500, 1000, 5000}`:
   - reset SSM state, buffer, and any semantic cache to empty
   - consume the first `N` tokens with writes enabled but no scoring
   - score the next `1024` tokens only
4. Reset again before the next segment.

If reliable document boundaries are available, segment on document
boundaries; otherwise use fixed non-overlapping windows. All models and
seeds must use the exact same evaluation segments.

## The 16MB Artifact

- SSM weights (dim=128, 4L, criticality): ~200KB at int6 + LZMA
- Wernicke router + experts: ~100KB compressed
- Bucket prototypes (k_max x dim=64): ~2KB
- Total: well under 16MB

The training buffer does NOT ship. At test time the buffer rebuilds
from the input stream — like a transformer filling its KV cache.

## VRAM Budget (per H100, 80GB)

| Use | Size | Notes |
|-----|------|-------|
| Model + optimizer | ~200-400MB | Small model, mostly free |
| Activations | varies | Skip gradient checkpointing — store all |
| KV buffer | ~1-10MB | Trivial at experiment scale |
| FW matrix (Claim 2) | ~200MB | If testing developmental phases |
| **Free** | **~78GB** | Available for larger batch sizes |

**Batch size tuning:** At dim=128, the H100 is likely underutilized at
batch=32. Increasing batch size to saturate the GPU's tensor cores is
free performance — more tokens per step at ~same wall time per step.
Benchmark on pod boot to find the largest batch that doesn't slow the
step. This trades spare VRAM for training throughput.

## Engineering Changes (Claim 1)

### 1. Unconditional buffer append

Remove surprise gating from consolidation_step(). Append every token's
KV pair to the buffer, organized by Wernicke bucket_id. Pure VRAM
append.

### 2. Within-bucket retrieval (replaces softmax over all slots)

For each query: identify Wernicke bucket, retrieve within that bucket
only. Multiple retrieval modes to ablate (mean, recent, topk, softmax).
Retrieval cost bounded by bucket size, not total buffer size.

### 3. Per-bucket semantic prototypes (replaces global basis bank)

One prototype per Wernicke bucket, EMA-updated from bucket entries.
Per-type prior for cold-start context. New architecture work.

### 4. Hierarchical Wernicke option

Stack 2 Wernicke layers: coarse routing (8 types) then fine routing
(8-32 subtypes). Better routing accuracy at matched compute cost.
Hierarchical cache fallback for sparse fine buckets.

## Engineering Changes (Claim 2 — secondary)

### 5. Fast weight matrix

Trainable (dim_fw x dim_fw) matrix. Write: outer product update.
Read: matrix-vector multiply. Freeze at configurable step.
dim_fw and freeze_step are independent knobs.

### 6. Online defrag (replaces sleep cycle)

Amortized buffer maintenance. Score one entry per step. Prune when
score drops below threshold. Zero additional forward passes. Only
relevant at longer training horizons when buffer grows large. At
experiment scale (1500 steps), the buffer never needs maintenance.

## Ablation Plan (8x H100, 600s budget)

### Phase A: Claim 1 (typed buffer)

#### T2: Retrieval mode x capacity bridge

Primary question: does typed within-bucket retrieval help?

Bridge question: how much of the gain comes from uncapping memory rather
than from the retrieval rule itself?

| Condition | max_slots | Retrieval |
|-----------|-----------|-----------|
| softmax_all_32 | 32 | softmax over all (current baseline) |
| softmax_all_uncapped | unlimited | softmax over all |
| mean_uncapped | unlimited | bucket mean |
| recent_uncapped | unlimited | most recent k=8 in bucket |
| topk_32_8 | 32 | top-8 in bucket |
| topk_uncapped_4 | unlimited | top-4 in bucket |
| topk_uncapped_8 | unlimited | top-8 in bucket |
| topk_uncapped_16 | unlimited | top-16 in bucket |

8 conditions x 7 seeds = 56 runs

#### T3: Wernicke structure (param-matched)

Hold total expert parameter budget constant. Scale expert_dim inversely
with k_max to isolate typing granularity from model size.

Report realized total parameter counts for every T3 condition in the
summary table. Treat this as approximately parameter-matched, not
perfectly identical.

| Condition | Structure | Total buckets |
|-----------|-----------|--------------|
| flat_8 | 1 layer, k=8 | 8 |
| flat_16 | 1 layer, k=16 | 16 |
| flat_64 | 1 layer, k=64 | 64 |
| hier_8x8 | 2 layers, 8x8 | 64 |
| hier_8x32 | 2 layers, 8x32 | 256 |

5 conditions x 7 seeds = 35 runs

### Phase B: Claim 2 (developmental + maintenance)

Run only if Claim 1 shows positive results.

#### T4: Append-only vs online defrag

This is NOT part of the core 600s experiment. At ~1500 training steps
the buffer should still fit easily in VRAM and maintenance pressure is
likely weak. Only run T4 as a longer-horizon follow-up (for example 5x
the budget) if realized buffer occupancy, retrieval dilution, or
throughput suggests maintenance has become relevant.

| Condition | Maintenance |
|-----------|-------------|
| append_only | none (Claim 1 winner) |
| defrag_global | score+prune, all buckets |
| defrag_per_bucket | score+prune, within bucket |

3 conditions x 7 seeds = 21 runs

#### T5: Developmental fast weights

fw_matrix_dim = FIXED (10K, ~200MB VRAM)
fw_freeze_step = SWEPT

| Condition | fw_freeze_step | Childhood % |
|-----------|---------------|-------------|
| buffer_only | n/a | 0% |
| fw_freeze_150 | 150 | 10% |
| fw_freeze_450 | 450 | 30% |
| fw_freeze_750 | 750 | 50% |
| fw_never | never | 100% |

5 conditions x 7 seeds = 35 runs

### Phase C: Composition + Confirmation

#### T6: Composition (exploratory)

| Condition | Config |
|-----------|--------|
| bare_ssm | No memory, no Wernicke |
| current_best | Exp 9/11/13 winners (old architecture) |
| claim1_winner | T2 + T3 winners |
| full_winner | + T4 + T5 winners (if Claim 2 tested) |
| transformer | Transformer baseline, same wall time |

5 conditions x 7 seeds = 35 runs

#### T7: Confirmation (fresh seeds)

Single pre-specified contrast only:

- locked winner from T6
- strongest baseline from T6

Run on 8 FRESH seeds and the locked TTT evaluation protocol above.
This is the real statistical test. Everything before is exploratory.
No extra pairwise tests in T7.

2 conditions x 8 fresh seeds = 16 runs

### Totals

| Phase | Runs | Batches (8 GPU) | Wall time |
|-------|------|----------------|-----------|
| A (T2+T3) | 91 | 12 | ~120 min |
| B (T5 core) | 35 | 5 | ~50 min |
| C (T6+T7) | 51 | 7 | ~70 min |
| **Total** | **177** | **24** | **~4 hours** |

T4 is a separate longer-horizon follow-up and is excluded from these
totals. Phase A is the core experiment. B is contingent. C requires A
results.

## Operational Plan

1. Write all code targeting 8x H100 (local development)
2. Spin up 8x H100 pod + network disk (high-availability region)
   -> benchmark batch size, verify setup -> tear down
3. Spin up CPU-only pod + same network disk
   -> download/prep 130GB prerequisite data -> tear down
4. Spin up 8x H100 -> Phase A (~120 min) -> analyze -> tear down
5. Decide: run Phase B? Select winners.
6. Spin up 8x H100 -> Phase B if needed + Phase C (~2 hrs) -> tear down
7. Optional: run T4 only as a longer-horizon follow-up if Claim 1 wins
   and the buffer shows real maintenance pressure

Total H100 time: ~4 hours across 2-3 sessions.
