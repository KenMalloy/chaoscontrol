# Experiment 14 — Phase A Report: Typed Buffer × Retrieval Mode Ablation

**Date**: 2026-04-10
**Hardware**: 8× NVIDIA H100 80GB HBM3 (AP-IN-1)
**Data**: FineWeb docs_val_raw.txt (145MB), docs_train_raw.txt (45GB)
**Budget**: 600s training per run, batch_size=32, seq_len=256, dim=128
**Total runs**: 98 (14 conditions × 7 seeds), 98/98 completed, 0 failures

## Result: Typed buffer does NOT help. Bare SSM wins.

| Rank | Condition | mean bpb | std | vs bare_ssm |
|------|-----------|----------|-----|-------------|
| 1 | **bare_ssm** | **2.3962** | 0.019 | **reference** |
| 2 | softmax_all_32 | 2.4140 | 0.018 | +0.018 |
| 3 | topk_uncapped_16 | 2.5594 | 0.019 | +0.163 |
| 4 | mean_uncapped | 2.5632 | 0.022 | +0.167 |
| 5 | flat_16 | 2.5637 | 0.042 | +0.168 |
| 6 | topk_uncapped_8 | 2.5827 | 0.032 | +0.187 |
| 7 | topk_uncapped_4 | 2.5850 | 0.041 | +0.189 |
| 8 | flat_64 | 2.5988 | 0.032 | +0.203 |
| 9 | recent_uncapped | 2.6144 | 0.032 | +0.218 |
| 10 | softmax_all_uncapped | 2.6182 | 0.033 | +0.222 |
| 11 | topk_8k_8 | 2.6227 | 0.022 | +0.227 |
| 12 | flat_8 | 2.6310 | 0.035 | +0.235 |
| 13 | hier_8x8 | 2.7458 | 0.053 | +0.350 |
| 14 | hier_8x32 | 2.7836 | 0.060 | +0.387 |

## Key findings

1. **bare_ssm wins outright** (2.396 bpb). Every buffer condition hurts, no exceptions.

2. **softmax_all_32 is closest** (+0.018 bpb) — the only condition within noise of bare_ssm. This is the capped softmax attention over the 32 most recent buffer entries. It's the simplest retrieval mode and the least overhead.

3. **Hierarchical Wernicke is worst** (hier_8x8: +0.350, hier_8x32: +0.387). More routing complexity = worse results. The extra parameters and routing overhead actively harm the model.

4. **Retrieval mode ranking**: softmax_all_32 > topk > mean > flat > recent > softmax_uncapped. Capped retrieval beats uncapped. Simpler beats complex.

5. **Warming curves show buffer benefit exists but is insufficient**: Most buffer conditions show bpb improvement from 0→5000 warmup tokens (e.g., mean_uncapped: 2.812→2.486). The buffer IS learning something during warmup. But the cost of the buffer mechanism (extra parameters, routing overhead) exceeds the benefit.

## Diagnosis

The typed buffer thesis fails because:
- **Wernicke routing is too expensive**: The MoE router discovers byte-level structure online, burning parameters that would be better spent on the SSM backbone.
- **Bucket keys are unstable**: Continuous hidden-state routing produces noisy, non-reproducible bucket assignments. The same byte sequence may route differently across forward passes.
- **No compression prior**: SP8192 competitors get stable reusable units for free. Our Wernicke tries to learn this from scratch in 10 minutes — not enough time.

## Decision

**Phase B: NO-GO.** Typed buffer does not beat bare_ssm. No Phase B launch.

## Next steps

Pivot to Experiment 15 "ChaosPiece": tokenizer-first architecture.
- Add SP8192 tokenizer up front (stable discrete units)
- Test bare SSM on tokenized input (expected ceiling: ~1.1-1.5 bpb)
- If competitive, test token-keyed memory (discrete keys instead of bucket IDs)

Design doc: `docs/plans/2026-04-10-experiment-15-chaospiece-design.md`
