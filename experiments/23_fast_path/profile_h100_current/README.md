# Exp23 H100 Profile Snapshot

Date: 2026-04-21

Pod: 1x H100 SXM, `runpod/parameter-golf:latest`, PyTorch 2.9.1 CUDA 12.8.

This is a short profiler run against the current Exp23 fast path after the
`_ssm_scan` CUDA wheel was installed. The profile uses explicit CUDA
synchronization around coarse sections, so the section timing and reported
tokens/sec are instrumentation-heavy. Use the relative breakdown and Chrome
traces to choose the next optimization, not as a clean throughput number.

## Configs

Both profiles used:

- `vocab_size=16384`
- `model_dim=256`
- `num_layers=4`
- `seq_len=512`
- `batch_size=1024`
- `precision=bf16`
- `optimizer=adamw`
- `activation_checkpoint=false`
- `compile_full_path=false`
- `CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan`

Only `chunk_size` changed: 64 vs 128.

## Coarse Timing

| profile | chunk | avg section sum | profiled tok/s | peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| `profile_h100_b1024_c64` | 64 | 0.2984 s | 1.76M | 42.8 GB |
| `profile_h100_b1024_c128` | 128 | 0.2989 s | 1.75M | 57.2 GB |

The clean smoke timing before profiling was higher; profiler overhead and
per-section synchronizations make these tokens/sec values conservative.

## Main Findings

`_ssm_scan` is not the bottleneck. In the `chunk_size=64` trace, the custom
scan forward is about 26 ms over 40 calls and backward about 32 ms over 40
calls across the active window. The total scan contribution is roughly 2-3%
of CUDA self time.

The largest visible CUDA buckets are:

- Cast/copy traffic: `aten::copy_`, `aten::to`, and `aten::_to_copy` account
  for about 584-589 ms of CUDA time over 10 active steps, moving roughly
  645 GB through `_to_copy` paths.
- Chunked head CE: log-softmax forward/backward is about 645 ms combined
  over 10 active steps at `chunk_size=64`.
- GEMMs: `aten::mm` is about 392 ms over 10 active steps.
- Elementwise kernels: vectorized/unrolled elementwise kernels are another
  large bucket, consistent with eager small-op overhead.
- Batch assembly: `aten::index` is about 216-244 ms CPU total over 10 active
  steps and appears outside the main GPU compute section.

The biggest static-code suspect for that cast/copy bucket is the head loss:

```python
chunk_loss = F.cross_entropy(
    logits_chunk.reshape(-1, vocab).float(),
    tgt_chunk.reshape(-1),
    reduction="sum",
) / total_tokens
```

At `B=1024,T=512,V=16384`, each fp32 full-logits equivalent is about 34 GB.
Chunking keeps peak memory legal, but it still streams enormous fp32 logits
through log-softmax across the step. This is why "just unchunk it" is unlikely
to fit cleanly on 80 GB, and why a fused linear-cross-entropy path is a real
optimization candidate rather than polish.

`chunk_size=128` reduces launch count versus 64, but does not improve the
instrumented step time and costs much more VRAM. At this batch/model shape,
`chunk_size=64` remains the more attractive default until the head path is
changed.

## Code-Level Notes

The Exp23 runner has its own `_run_train_step` in
`experiments/23_fast_path/runner_fast_path.py`. Unlike
`chaoscontrol.train_ssm.train_ssm_step`, it does not currently thread
`compile_full_path` into the encoder call, so Stage A compile sweeps would
not test the compile lever in this runner as written.

The current two-stage backward is still:

```python
hidden = model.encode(inputs)
hidden_for_ce = hidden.detach().requires_grad_(True)
loss = chunked_lm_head_backward(...)
hidden.backward(gradient=hidden_for_ce.grad)
```

That shape protects memory, but it serializes head backward, encoder backward,
and any DDP all-reduce. The trace suggests the next work should focus on
reducing cast/copy traffic and head CE cost before spending on an 8xH100
Stage A sweep.

## Next Optimization Order

1. Wire `compile_full_path` or an Exp23-specific encoder compile path into
   `runner_fast_path.py`, then trace/benchmark the encoder-only compile in
   this exact runner.
2. Hunt the `aten::to` / `_to_copy` source and remove avoidable fp32/bf16
   ping-pong in the hot loop.
3. Add one-step-ahead batch prefetch to hide `batch_from_start_tensor`.
4. Prototype a memory-safe fused linear-cross-entropy or larger-chunk head
   path. Full unchunked logits are likely not viable at the current
   `B=1024,T=512,V=16384` shape without a fused CE implementation.
5. Revisit DDP gradient overlap after the single-GPU hot loop is denser.
