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

FLOP analysis for our model (4L, dim=256, SP8192, **single GPU**):

| Component | FLOPs/seq (fwd) | Notes |
|---|---|---|
| Projections (4 blocks × 5) | 1.34G | dim² matmuls, parallel |
| FeedForward (4 blocks) | 1.07G | parallel |
| Embedding + LM head | 2.15G | parallel |
| Scan (elementwise) | ~0.03G | negligible FLOPs, but 512 serial steps |
| **Total forward** | **~4.6G** | |
| **Total fwd+bwd** | **~13.7G** | |

Scaling by batch (**single GPU**, 30% utilization of 990 TFLOPS):

| Batch | FLOPs/step | Matmul time | Tokens/step | Tokens/600s |
|---|---|---|---|---|
| 32 | 438G | 1.5ms | 16K | ~6.4B* |
| 256 | 3.5T | 12ms | 131K | ~6.5B |
| 1024 | 14T | 47ms | 524K | ~6.7B |
| 2048 | 28T | 94ms | 1.0M | ~6.4B |

*At batch=32, measured step time is 41ms (scan overhead dominates).
At larger batch, matmul time catches up and the scan overhead becomes
a smaller fraction. Crossover at approximately batch=3000.

### GPU Assumptions

**Phase 0 benchmarks single-GPU throughput.** The current training
stack (`src/chaoscontrol/training.py`) is single-device. Phase A
runs one model per GPU (same as Exp 15/16 orchestrators — round-robin
GPU assignment, independent processes). Data-parallel multi-GPU
training would multiply throughput but requires distributed training
code we do not have. It is explicitly out of scope for Exp 18.

**Single-GPU feasibility (conservative estimate):**

At batch=1024 on 1 GPU: ~6.7B tokens/600s. The full dataset is 10B
tokens. One GPU can cover ~67% in a single sweep, or 100% in ~900s.
To sweep all 10B in ≤ 400s on one GPU, we need batch ≥ ~2500 (depends
on Phase 0 measurements).

If single-GPU sweep of 10B tokens is infeasible within 400s, the
experiment should either:
- Accept partial coverage (e.g., 5B tokens = 50%) and test whether
  even partial-informed targeting beats random sampling, or
- Use a smaller dataset subset as the "full" sweep target.

Currently we see 230M tokens (2.3% of 10B). Even 50% coverage would
be a 20x improvement and enough to test the targeting hypothesis.

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

### Phase 0: Throughput Benchmark + LR Stability Screen (~30 min on pod)

Before the experiment, verify the FLOP analysis and LR regime
empirically. All measurements are **single-GPU**.

**Part 1 — Throughput curve:**

```python
for batch_size in [32, 128, 256, 512, 1024, 2048, 4096]:
    # 20 steps forward+backward, report:
    # - wall time per step (ms)
    # - tokens per second
    # - peak GPU memory (GB)
    # - projected seconds for full 10B token sweep (1 GPU)
```

**Part 2 — LR stability screen:**

At the largest feasible batch size from Part 1, run 200 training
steps with 3 LR candidates:

- Linear-scaled: base_lr × (large_batch / 32)
- Square-root-scaled: base_lr × sqrt(large_batch / 32)
- Fixed: base_lr (no scaling)

Check for: NaN, loss divergence, loss that fails to decrease.
This takes ~3 minutes and prevents wasting a full Phase A run on
a broken LR regime.

**Go/no-go:**
1. Full 10B sweep in ≤ 400s on 1 GPU at some feasible batch size,
   OR partial sweep (≥ 50% coverage) in ≤ 400s with a clear path
   to testing the targeting hypothesis.
2. At least one LR candidate produces stable, decreasing loss at
   the large batch size for 200 steps.

### Phase A: Sweep + Target

The model is the existing fast SP8192 + diag SSM with compiled scan.
No architecture changes. The only variable is data strategy.

**Step 1 — Sweep (large batch, ~400s)**

- Set batch to the largest feasible size from Phase 0
- Use LR regime validated in Phase 0 Part 2
- Process the full dataset (or largest feasible fraction) with
  gradients on, optimizer stepping
- The model is partially trained after this — it has seen everything
  (or most of it) once

**Step 1.5 — Rescore (frozen model, ~30-60s)**

After the sweep, freeze the model and do a fast forward-only pass
over all training sequences at maximum batch size. Record per-sequence
loss using the **end-of-sweep model weights**. This eliminates
time-bias: losses from early-sweep sequences were scored by a weaker
model. Rescoring with the final snapshot ensures all sequences are
ranked by the same model.

The rescore pass is forward-only (no gradients), so it runs faster
than the sweep. At the same batch size, ~50% of sweep time (no
backward pass). Budget accordingly.

**Step 2 — Target (normal batch, hard examples, remaining time)**

- Rank all sequences by rescored loss (Step 1.5)
- Filter: keep sequences where loss > threshold (top N%)
- Train on this subset at batch=32 with base LR for remaining time
- Normal training with gradient clipping, same as current recipe

**Coverage definition:** "See all 10B tokens" means generating
non-overlapping start indices with stride=seq_len (512), producing
~19.5M windows of 512 tokens each. No overlapping windows during
the sweep. Coverage is measured as unique prediction targets scored.

**Conditions (5 conditions × 7 seeds = 35 runs):**

| Condition | Description |
|---|---|
| baseline_b32 | Standard training, batch=32, 600s, random sampling |
| sweep_only | Large batch, full dataset, sweep fills 600s. If sweep finishes early, continue training on the same data in a second pass (no targeting, no idle time). |
| sweep_target_top25 | Sweep + rescore + retrain on top 25% by rescored loss |
| sweep_target_top10 | Sweep + rescore + retrain on top 10% by rescored loss |
| sweep_target_top5 | Sweep + rescore + retrain on top 5% by rescored loss |

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
| Large batch hurts convergence | Phase 0 Part 2 screens 3 LR candidates; sweep_only condition tests this directly |
| High-loss sequences are noise | Compare targeting percentages; if top5 is worse than top25, aggressive targeting overfits to noise |
| Sweep takes > 400s on 1 GPU | Phase 0 benchmark gates this; accept partial coverage if full sweep is infeasible |
| LR scaling is wrong for SSMs | Phase 0 Part 2 tests linear, sqrt, and fixed scaling before committing |
| Memory overflow at large batch | Phase 0 measures VRAM; gradient checkpointing if needed |
| Time-biased loss scoring | Post-sweep rescore pass with frozen model eliminates model-age confound |
| Rescore pass takes too long | Forward-only at max batch; budget ~30-60s; if too slow, reduce rescore coverage to 50% |

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
