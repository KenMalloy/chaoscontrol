# Experiment 18: SSM Throughput Advantage — Full-Dataset Sweep + Targeted Depth Training

## Status

Design phase. Pending Phase 0 benchmark verification.

## Motivation

### The Competition Throughput Comparison Is Misleading

The standard comparison between our SSM and competition transformers
measures optimizer steps:

| Model | Steps in 600s | Tokens seen |
|---|---|---|
| Transformer (winners) | 21K | 340M (3.4% of 10B) |
| SSM (compiled scan) | 14K | 230M (2.3% of 10B) |

This makes the SSM look 33% slower. But both models run at batch=32,
seq=512. The SSM's sequential scan cost is O(seq_len) regardless of
batch — the 512 steps are a fixed latency, and the batch dimension
parallelizes freely across GPU cores. For the transformer, attention
is O(seq_len² × batch), so scaling batch costs proportional compute.

### The SSM's Actual Throughput Advantage

FLOP analysis for our model (4L, dim=256, SP8192, one GPU):

| Component | FLOPs/seq (fwd) | Notes |
|---|---|---|
| Projections (4 blocks × 5) | 1.34G | dim² matmuls, parallel |
| FeedForward (4 blocks) | 1.07G | parallel |
| Embedding + LM head | 2.15G | parallel |
| Scan (elementwise) | ~0.03G | negligible FLOPs, but 512 serial steps |
| **Total forward** | **~4.6G** | |
| **Total fwd+bwd** | **~13.7G** | |

Scaling by batch (single GPU, 30% utilization of 990 TFLOPS):

| Batch | FLOPs/step | Matmul time | Tokens/step | Tokens/600s |
|---|---|---|---|---|
| 32 | 438G | 1.5ms | 16K | ~6.4B* |
| 256 | 3.5T | 12ms | 131K | ~6.5B |
| 1024 | 14T | 47ms | 524K | ~6.7B |
| 2048 | 28T | 94ms | 1.0M | ~6.4B |

*At batch=32, measured step time is 41ms (scan overhead dominates).
At larger batch, matmul time catches up and the scan overhead becomes
a smaller fraction. Crossover at approximately batch=3000.

With 4 GPUs (data parallel): ~26B tokens in 600s.

**The full dataset is 10B tokens. We can see it 2.5 times over.**

Currently we see 230M tokens — 2.3% of the dataset. We are leaving
approximately **100x throughput** on the table by running at batch=32.

### Why This Matters

Competition winners train on ~340M random tokens (3.4% of 10B) and
hope that random sampling covers the important patterns. No
transformer entry can see the full dataset in 600 seconds.

An SSM can. This enables a training strategy no transformer can
execute:

1. **Fast sweep**: See all 10B tokens in one pass
2. **Targeted depth**: Retrain on the hardest examples

This redefines "depth recurrence" from an architecture trick (repeat
layers) to a data trick (repeat hard examples). Both spend extra
compute where it helps most, but the SSM version leverages a
structural throughput advantage that transformers cannot match.

## Design

### Phase 0: Throughput Benchmark (10 minutes on pod)

Before the experiment, verify the FLOP analysis empirically:

```python
for batch_size in [32, 128, 256, 512, 1024, 2048, 4096]:
    # 20 steps forward+backward, report:
    # - wall time per step (ms)
    # - tokens per second
    # - peak GPU memory (GB)
    # - projected seconds for full 10B token sweep
```

Measure on 1 GPU first, then 4 GPU data parallel.

**Go/no-go:** Full 10B sweep in ≤ 400s at some feasible batch size,
leaving ≥ 200s for targeted training.

### Phase A: Sweep + Target

The model is the existing fast SP8192 + diag SSM with compiled scan.
No architecture changes. The only variable is data strategy.

**Step 1 — Sweep (large batch, full dataset, ~400s)**

- Set batch to the largest size from Phase 0 that fits in VRAM
- Process all 10B tokens in one pass with normal training (gradients
  on, optimizer stepping)
- Record per-sequence loss at end of sweep
- The model is partially trained after this — it has seen everything
  once

Learning rate scaling: follow the linear scaling rule. If base LR is
2e-3 at batch=32, then at batch=1024 use LR = 2e-3 × (1024/32) =
6.4e-2, with proportional warmup.

**Step 2 — Target (normal batch, hard examples, remaining time)**

- Rank all sequences by loss from Step 1
- Filter: keep sequences where loss > threshold (top N%)
- Train on this subset at batch=32 with base LR for remaining time
- Normal training with gradient clipping, same as current recipe

**Conditions (5 conditions × 7 seeds = 35 runs):**

| Condition | Description |
|---|---|
| baseline_b32 | Standard training, batch=32, 600s, random sampling |
| sweep_only | Large batch, full dataset, 600s, no targeting phase |
| sweep_target_top25 | Sweep + retrain on top 25% by loss |
| sweep_target_top10 | Sweep + retrain on top 10% by loss |
| sweep_target_top5 | Sweep + retrain on top 5% by loss |

All conditions: same model, same total wall time (600s), same SP8192
data, 7 seeds.

### What Each Comparison Teaches

| Comparison | Question |
|---|---|
| sweep_only vs baseline_b32 | Does seeing all data once beat random sampling? |
| sweep_target_* vs sweep_only | Does curriculum focus add value beyond coverage? |
| top25 vs top10 vs top5 | How aggressive should targeting be? |
| baseline_b32 vs all others | Is the entire throughput strategy better than standard training? |

### Go / No-Go

1. Any sweep condition beats baseline_b32 by ≥ 0.02 bpb
2. If sweep_only wins but targeting adds nothing: the value is
   coverage, not curriculum. Still a win.
3. If baseline_b32 wins: gradient updates per example matter more
   than data coverage at this model scale. The throughput advantage
   doesn't translate to learning.

### Risks

| Risk | Mitigation |
|---|---|
| Large batch hurts convergence | Linear LR scaling rule; sweep_only condition tests this directly |
| High-loss sequences are noise | Compare targeting percentages; if top5 is worse than top25, aggressive targeting overfits to noise |
| Sweep takes > 400s | Phase 0 benchmark gates this; don't proceed if infeasible |
| LR scaling is wrong for SSMs | Phase 0 can test a few LR values at large batch |
| Memory overflow at large batch | Phase 0 measures VRAM; gradient checkpointing if needed |

### What Exp 16 Told Us About This

Exp 16's finding that attention is extremely concentrated
(effective_connections ~1.2) suggests that most of the prediction
problem is local. If most tokens are easy to predict from recent
context, then the high-loss tokens — the ones where local context
isn't enough — are exactly where the model has the most to learn.
Targeting those is not random curriculum, it's informed by the
structure of language prediction.

## What This Does Not Test

- Architecture changes (no attention sidecar, no depth recurrence)
- Quantization / artifact compression
- Muon optimizer
- TTT (test-time training)

## Relationship to Other Experiments

**Exp 17** (local attention sidecar) asks an architecture question.
**Exp 18** asks a training strategy question. They are independent
and complementary. If both win, the combined play is: fast SSM with
throughput-aware training + local attention sidecar.

**Exp 16** provided the compiled scan that makes Exp 18 possible,
and the probe data suggesting the prediction problem is mostly local
(which motivates targeting the exceptions).

## Why No One Else Has Tried This

The competition is dominated by transformers. Attention's O(seq²)
cost means batch scaling is proportionally expensive — doubling batch
doubles the compute per step. There is no free throughput to exploit.

SSMs have O(seq_len) fixed scan cost plus O(batch × dim²) parallel
matmuls. Below the crossover batch (~3000), the scan dominates and
increasing batch is nearly free. This structural asymmetry is the
SSM's actual competitive advantage, and it has nothing to do with
architecture expressiveness.

## Files

```text
experiments/18_throughput_advantage/
  DESIGN.md -> symlink or copy of this doc
  bench_throughput.py   (Phase 0)
  runner_exp18.py       (Phase A)
  run_exp18.py
  test_exp18.py
```

No changes to `src/chaoscontrol/` required for Phase 0 or Phase A.
The model and training loop are unchanged. Only the data sampling
strategy and batch size schedule are new.
