# Exp23 1x H100 CUDA Graph Probe With Native Scan

Date: 2026-04-21

Shape: 4-layer diag SSM, `dim=256`, `vocab=16384`, `batch=1024`,
`seq_len=512`, `lm_head_backward_mode=fused_streaming_v2`,
`lm_head_tile_size=8192`, synthetic SP16384-compatible tokens, eval disabled.

Runner defaults were the patched Exp23 no-Inductor path:
`CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan` and
`CHAOSCONTROL_POST_SCAN_BACKEND=eager`.

## Result

CUDA graph capture was technically accepted by the gate, but it is not a
production win for the current single-GPU hot path.

| Row | Tokens/s | Step s | Peak VRAM |
| --- | ---: | ---: | ---: |
| graph_native_streaming_v2 | 2,891,373 | 0.1813 aggregate / 0.1739 replay | 45,106 MB |

The replay-only step time (`0.1739s`) is slower than the no-graph confirmation
from `results_rmsnorm_1x_20260421` (`0.1705s`). The gate accepted because its
warmup eager step estimate was pessimistic (`0.2120s`), not because the graph
beat the best measured no-graph steady state.

## Recommendation

Do not prioritize CUDA graph productionization for the current 1x hot path.
Revisit only if the step is restructured enough that graph replay beats a
same-pod no-graph baseline, or if a future 8x path can support DDP/checkpoint
capture without regressing stability.

