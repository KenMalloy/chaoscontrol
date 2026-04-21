# Exp23 Current Fused Path Profile

Date: 2026-04-21

Hardware: 1x H100 SXM, `runpod/parameter-golf:latest`, PyTorch 2.9.1 + CUDA 12.8.

Purpose: refresh the profiler view after the native fused linear+CE path. The
older `profile_h100_current/` trace predates the fused head and overstates the
old `log_softmax` / full-logits traffic problem.

Profile shape:

- `vocab_size=16384`
- `batch_size=1024`
- `seq_len=512`
- `chunk_size=64`
- `model_dim=256`
- `num_layers=4`
- `lm_head_backward_mode=fused`
- `activation_checkpoint=false`
- synthetic uint16 shards, `eval_batches=0`

Training summary under profiler:

```text
steps: 20
aggregate_tokens_per_sec: 2,562,111
peak_vram_mb: 30,865
```

CUDA table read:

- Fused linear+CE is still the largest named training component.
- `_FusedLinearCrossEntropyFn` total CUDA was `9.218s` over 23 calls, with the
  custom backward kernel at `748.6ms` self CUDA and forward kernels around
  `327.6ms` plus `318.7ms`.
- `aten::_to_copy` is still visible, but is no longer the old dominant bucket:
  `411.1ms` CUDA / `345GB` reported traffic across the profiled run.
- CUDA graph is therefore not the main answer; the next meaningful speed work
  is still the fused linear+CE kernel itself, especially eliminating duplicated
  vocab-tiled matmuls or replacing it with a mature fused linear-CE implementation.

Artifacts:

```text
cuda_table.txt
cpu_table.txt
profile_result.json
summary.json
profile_h100_b1024_c64_fused_current.chrome_trace.json.gz
```
