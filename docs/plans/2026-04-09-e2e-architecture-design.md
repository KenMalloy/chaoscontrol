# Experiment 14: VRAM-Resident Memory at Scale

## Grand Thesis

An SSM with a VRAM-resident typed KV buffer achieves transformer-quality
language modeling at SSM throughput. The SSM trunk IS the attention mechanism
(O(1) per step). The buffer IS the lossless backup for what the SSM state
forgets. Retrieval is recovery, not computation. O(n^2) becomes O(1) + O(k).

Steps are sacred. Memory is state, not compute. Even a 1%-correct retrieval
signal, compounded over thousands of steps, closes the gap.

## Architecture (with brain labels)

```
Input (raw bytes)
  |
  v
Sensory Cortex ---- Byte embedding (no learned tokenizer)
  |
  v
Wernicke's Area ---- MoE routing: type the input into k_max semantic buckets
  |                   Ships in the artifact. Structure from birth.
  v
Hippocampus -------- VRAM KV buffer. Append every token's key-value pair.
  |                   No surprise gating. No compression during training.
  |                   Retrieval: top-k within the matching Wernicke bucket.
  |                   At test time, rebuilds from the input stream.
  v
Neocortex ---------- Semantic tier prototypes (one per bucket type).
  |                   Slow EMA distillation from buffer entries.
  |                   Ships in artifact as per-type priors (2KB total).
  v
SSM Backbone ------- The trunk. dim=128, 4 layers, diag A-mode.
  |                   State update IS the attention: h_{t+1} = A*h_t + B*x_t
  |                   Maximum step count. Never throttled by memory ops.
  v
Output ------------- LM head, predict next byte
```

### Forward pass (training)

1. Embed raw bytes
2. Wernicke types the input, routes through experts
3. Buffer read: project query from embedding, top-k retrieval within
   the matching bucket, weighted sum of values
4. Semantic tier: add per-bucket prototype bias
5. SSM recurrence through 4 layers
6. Buffer write: append this token's KV pair to the matching bucket
   (unconditional, no surprise gating, pure VRAM append)
7. Predict next token via LM head

### Forward pass (inference / TTT)

Identical. The buffer starts empty and grows as the model reads.
After a few tokens per bucket, retrieval becomes useful.
The warming curve is steep because Wernicke pre-organizes the buffer.

## Three-Number Eval

| Metric | Buffer state | What it measures |
|--------|-------------|-----------------|
| bpb_pretrain | Full (trained with growing buffer) | Training quality |
| bpb_artifact | Empty (cold start, 16MB artifact only) | Weight quality without buffer |
| bpb_ttt | Rebuilding (buffer grows from test data) | Production performance |

- **bpb_pretrain**: Best number. SSM + full buffer during training.
- **bpb_artifact**: Worst number. SSM weights + Wernicke routing + semantic
  prototypes. No buffer. This measures how good the weights are alone.
- **bpb_ttt**: The real number. Start cold, process test data, buffer grows.
  After ~100-500 tokens, the model has useful typed context. The gap between
  artifact and TTT measures buffer value.

## The 16MB Artifact

Contents:
- SSM weights (dim=128, 4L): ~200KB at int6 + LZMA
- Wernicke router + experts: ~100KB compressed
- Semantic tier prototypes (k_max prototypes at dim=64): ~2KB
- Total model: well under 16MB

The training buffer does NOT ship. It was scaffolding — it made training
better (richer gradients), but the artifact is just weights + structure.
At test time the buffer rebuilds from the input stream.

## Developmental Phases During Training

### Phase 1: Infancy (fast weight matrix)

Steps 0 to ~fw_dim. A large fast weight matrix (dim_fw x dim_fw)
accumulates all token representations via outer product updates:
W += eta * v @ k.T. Read is W @ query. O(1) read, O(1) write.

The matrix absorbs everything. High plasticity. No explicit slots yet.
This builds a foundation of "crystallized patterns" before the explicit
buffer system takes over.

The matrix should be large enough to use meaningful VRAM. At dim_fw=10K:
200MB VRAM, 0.003ms read, ~10K pattern capacity.

### Phase 2: Adolescence (transition)

At approximately step = fw_dim (when interference begins degrading the
matrix), the FW matrix freezes. It becomes a permanent background bias —
the gist of everything learned in infancy.

Simultaneously, the explicit KV buffer activates. The model starts
appending per-token KV pairs and learning to retrieve from them.

### Phase 3: Adulthood (production mode)

The KV buffer grows. Retrieval is hierarchical: Wernicke bucket -> top-k
within bucket. The frozen FW matrix provides background context. The
semantic tier slowly distills prototypes from buffer entries.

This is the mode the artifact runs in. The model must spend enough
training time here to master structured retrieval, because that is
all it has at inference time.

### The Golden Ratio

The transition point (fw_dim / total_steps) determines how much training
is childhood vs adulthood. The model needs enough adulthood to master
the inference-time memory system. Empirically: test fw_dim values that
give 10-30% childhood.

## Engineering Changes Required

### 1. Unconditional buffer append (replaces surprise-gated consolidation)

Current: consolidation_step() runs every step, writes only when
surprise > 0.01 * running_avg (~98% of steps anyway).

Change: remove surprise gating. Append every token's KV pair to the
buffer unconditionally. The buffer is organized by Wernicke bucket_id.
Pure VRAM operation.

### 2. Top-k retrieval within bucket (replaces softmax over all slots)

Current: softmax over all slots -> weighted sum.

Change: for each query, identify the Wernicke bucket, compute dot
products within that bucket only, take top-k, softmax over those k
candidates, weighted sum. k is a tunable parameter (likely 4-16).

This prevents signal dilution as the buffer grows and naturally limits
retrieval cost to O(k) regardless of buffer size.

### 3. Fast weight matrix layer (new)

Add a trainable fast weight matrix (dim_fw x dim_fw) with:
- Write: W += eta * value_proj(x) @ key_proj(x).T (outer product)
- Read: output = W @ query_proj(x)
- Freeze mechanism: at step = fw_dim, stop writes, keep reads

### 4. Micro-op sleep (replaces inline batch sleep)

Current: sleep_cycle.run() blocks the training loop for 64-300+
forward passes every N steps.

Change: amortize across training steps. Score ONE slot per step as a
side-effect. Prune when score drops below threshold. Zero additional
forward passes. The sleep "cycle" is spread across hundreds of
training steps instead of happening in one blocking burst.

This only matters during Phase 3 (adulthood) when the buffer is large
enough to need maintenance.

## Ablation Plan (8x H100, 600s budget, ~1500 steps/run)

### T2: Buffer scaling (max_slots x retrieval_k)

Does uncapped buffer with top-k retrieval beat capped buffer?

| Condition | max_slots | retrieval_k |
|-----------|-----------|-------------|
| capped_softmax | 32 | all (current) |
| capped_topk | 32 | 8 |
| uncapped_topk_4 | unlimited | 4 |
| uncapped_topk_8 | unlimited | 8 |
| uncapped_topk_16 | unlimited | 16 |

5 conditions x 7 seeds = 35 runs

### T3: Wernicke bucket count (k_max)

What granularity of semantic typing is optimal?

| k_max | Buckets per H100 |
|-------|-----------------|
| 4 | 0.5 |
| 8 | 1 |
| 16 | 2 |
| 32 | 4 |
| 64 | 8 |

5 conditions x 7 seeds = 35 runs

### T4: Micro-op sleep x partitioning

Does zero-cost consolidation help, and does semantic partitioning matter?

| Condition | Sleep | Partitioned | Bucket-aware |
|-----------|-------|-------------|-------------|
| no_sleep | off | - | - |
| micro_global | n3 micro-op | no | - |
| micro_slot_striped | n3 micro-op | yes | no |
| micro_bucket_sharded | n3 micro-op | yes | yes |

4 conditions x 7 seeds = 28 runs

### T5: Developmental phases

Does the FW -> buffer transition help?

| Condition | FW matrix | Buffer | Transition |
|-----------|-----------|--------|-----------|
| buffer_only | off | from step 0 | - |
| fw_only | dim 10K | no buffer | never |
| fw_10pct | dim ~150 | from step 150 | 10% childhood |
| fw_30pct | dim ~450 | from step 450 | 30% childhood |

4 conditions x 7 seeds = 28 runs

### T6: Composition (final validation)

| Condition | Config |
|-----------|--------|
| bare_ssm | No memory, no Wernicke |
| current_best | All exp 9/11/13 winners (old architecture) |
| thesis_config | All ablation winners combined |
| transformer | Transformer baseline at same wall time |

4 conditions x 7 seeds = 28 runs

### Total

154 runs / 8 GPUs = ~20 batches x 10 min = **~3.3 hours H100 time**

T2, T3, T4, T5 are independent and can run in parallel batches.
T6 depends on T2-T5 results.

## Operational Plan

1. Write all code targeting 8x H100 (local)
2. Spin up 8x H100 pod + network disk (high-availability region)
   -> verify setup -> tear down
3. Spin up CPU-only pod + same network disk
   -> download/prep 130GB prerequisite data -> tear down
4. Spin up 8x H100 -> run T2+T3+T4+T5 (~2.5 hours)
   -> analyze results -> select winners -> tear down
5. Spin up 8x H100 -> run T6 composition (~40 min)
   -> final submission run -> tear down

Total H100 time: ~4 hours for ablations + submission.
