# Exp23 1x H100 RMSNorm / Backend Probe

Date: 2026-04-21

Pod: RunPod official Parameter Golf template, 1x `NVIDIA H100 80GB HBM3`,
PyTorch `2.9.1+cu128`, CUDA runtime `12.8`.

Shape: 4-layer diag SSM, `dim=256`, `vocab=16384`, `batch=1024`,
`seq_len=512`, `lm_head_tile_size=8192`, synthetic `uint16` SP16384-compatible
tokens, eval disabled.

## Findings

The Exp23 runner must force the native scan backend. Before the runner patch,
`compile_full_path=false` still allowed the legacy diag/post-scan Inductor
defaults to spawn compile workers before the timed loop. GPU utilization sat at
0% with ~50GB allocated until the compile path completed.

Native block-level RMSNorm was not a throughput win. It saved about 4GB VRAM
but slowed the hot loop by roughly 5% on this shape, so block `RMSNorm` remains
on the legacy PyTorch math.

The combined final-norm + streaming CE mode was neutral/slightly slower than
the separate fused RMSNorm + streaming CE path at this shape.

## Rows

| Row | Mode | Tokens/s | Step s | Peak VRAM |
| --- | --- | ---: | ---: | ---: |
| current_streaming_v2_a | fused block RMSNorm + streaming_v2 | 2,912,113 | 0.1800 | 40,850 MB |
| norm_streaming_v2 | fused block RMSNorm + fused_norm_streaming_v2 | 2,911,142 | 0.1801 | 40,850 MB |
| old_block_rms_streaming_v2 | legacy block RMSNorm + streaming_v2 | 3,061,202 | 0.1713 | 44,945 MB |
| current_streaming_v2_b | fused block RMSNorm + streaming_v2 | 2,909,998 | 0.1802 | 40,850 MB |
| legacy_block_streaming_v2 | legacy block RMSNorm + streaming_v2 | 3,075,892 | 0.1705 | 44,946 MB |
| legacy_block_norm_streaming_v2 | legacy block RMSNorm + fused_norm_streaming_v2 | 3,073,565 | 0.1706 | 44,946 MB |

