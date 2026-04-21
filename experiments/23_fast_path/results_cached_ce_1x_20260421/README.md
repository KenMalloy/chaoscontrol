# Exp23 Cached Streaming CE Probe

Date: 2026-04-21

Hardware: 1x H100 SXM on the official `runpod/parameter-golf:latest`
template, PyTorch `2.9.1+cu128`, CUDA runtime `12.8`.

Shape: 4-layer diag SSM, `dim=256`, `vocab=16384`, `batch=1024`,
`seq_len=512`, `lm_head_tile_size=8192`, synthetic SP16384-compatible tokens,
eval disabled.

## Result

The cached-logits CE backend removes the backward logits-recompute GEMM and is
worth promoting to an 8x probe.

| Row | Mode | Tokens/s | Step s | Peak VRAM |
| --- | --- | ---: | ---: | ---: |
| streaming_v2_a | `fused_streaming_v2` | 3,051,951 | 0.1718 | 44,946 MB |
| streaming_cached | `fused_streaming_cached` | 3,214,781 | 0.1631 | 53,138 MB |
| streaming_v2_b | `fused_streaming_v2` | 3,061,693 | 0.1712 | 44,946 MB |

Baseline average: `3,056,822 tok/s`.

Cached speedup: `+5.17%`.

The VRAM increase is about `+8.2GB`, not the full logits tensor size over the
old peak, because `streaming_v2` already carries an 8192-column logits
workspace. Cached mode replaces that single workspace with the two saved
SP16384 tiles.

## Decision

Next step is an 8xH100 smoke with:

```yaml
lm_head_backward_mode: fused_streaming_cached
lm_head_tile_size: 8192
activation_checkpoint: false
grad_allreduce_mode: bulk
```

If the 8x speedup transfers, this should move the current ~23.8M tok/s path to
roughly ~25M tok/s before any larger hot-loop work.

