# Exp23 Head Fusion, CUDA Graph, And All-Reduce Probe

Date: 2026-04-21

## Scope

Priority order from the overnight engineering pass:

1. Fused final RMSNorm + linear CE path.
2. Fresh CUDA graph probe after the streaming CE work.
3. Bulk all-reduce vs async per-parameter all-reduce on 8xH100.

All runs used the official `runpod/parameter-golf:latest` template with
PyTorch `2.9.1+cu128`, CUDA runtime `12.8`, and local builds of `_lm_head_loss`
and `_ssm_scan` with `sm_90` cubins.

## 1xH100 Head And Graph Probe

Pod: 1x H100 SXM in DE, `ry7pf71m8qlr1u`, stopped after artifact pull.

Config shape:

- synthetic SP16384 uint16 train/val shards
- `batch_size=1024`, `seq_len=512`, `chunk_size=64`
- 4-layer/256-dim SSM
- `lm_head_tile_size=8192`
- `precision=bf16`, `activation_checkpoint=false`
- 30s measured budget, 5 warmup steps

| lane | head mode | graph | steps | tok/s | step seconds | final loss | peak VRAM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `fused_streaming_v2` | none | 124 | 2,273,729 | 0.2306 | 9.7042 | 44,946.1 MB |
| norm-fused | `fused_norm_streaming_v2` | none | 124 | 2,297,706 | 0.2282 | 9.7038 | 44,945.6 MB |
| graph probe | `fused_norm_streaming_v2` | probe | 132 | 2,428,758 | 0.2159 | 9.7042 | 45,105.6 MB |

CUDA graph summary for the graph probe:

- accepted: `true`
- capture seconds: `0.238`
- warmup seconds: `0.896`
- overhead seconds: `1.134`
- eager step seconds: `0.299`
- graph replay step seconds: `0.205`
- projected total speedup at 30s: `40.2%`
- break-even: `12.1` steps / `3.6s`

Takeaway: the combined norm+CE autograd path is exact and slightly positive on
1xH100, but the larger result is CUDA graph. After the streaming CE/tile work,
graph capture is no longer noise on single GPU. It still does not apply to the
current 8x path because the runner rejects graph mode under DDP.

## 8xH100 All-Reduce Probe

Pod: 8x H100 SXM in IN, `j1sx6nzuey77wl`, stopped immediately after artifact
pull.

Config shape matched the 1x run, except `world_size=8`.

| lane | head mode | grad sync | steps | aggregate tok/s | per-GPU tok/s | step seconds | final loss | peak VRAM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bulk baseline | `fused_streaming_v2` | `bulk` | 164 | 23,862,017 | 2,982,752 | 0.1758 | 9.6358 | 44,946.1 MB |
| bulk norm-fused | `fused_norm_streaming_v2` | `bulk` | 164 | 23,834,416 | 2,979,302 | 0.1760 | 9.6356 | 44,946.2 MB |
| async norm-fused | `fused_norm_streaming_v2` | `async_param` | 160 | 23,810,426 | 2,976,303 | 0.1762 | 9.6356 | 44,946.1 MB |

Takeaway: keep the 8x default on `fused_streaming_v2` + coalesced bulk
all-reduce. The async per-parameter hook path is valid and did not deadlock, but
launch overhead slightly outweighed overlap at this model size. The norm-fused
path also does not improve 8x throughput, so it should remain an optional probe
path rather than the Stage A default.

## 1xH100 Real-Data Probe

Pod: 1x H100 SXM in US, `usxsk0b0r42ng2`, stopped after artifact pull.

This run downloaded a small real SP16384 prefix from
`Natooka/parameter-golf-sp-tokenizers`:

- train shards: `fineweb_train_000000.bin` through `000003.bin`
- validation shard: `fineweb_val_000000.bin`
- tokenizer: `fineweb_16384_bpe.model`

Uncached mmap-cache build times:

| data | tokens | cache bytes | uncached load/cache time |
| --- | ---: | ---: | ---: |
| real train | 400,010,537 | 800,021,074 | 5.67s |
| real val | 42,266,034 | 84,532,068 | included above |
| synthetic train | 64,000,000 | 128,000,000 | 1.26s |
| synthetic val | 1,000,000 | 2,000,000 | included above |

Config shape used `activation_checkpoint=true`, matching the current Stage A
matrix default. Absolute throughput on this pod was much lower than the DE/IN
pods above, so the useful comparison is real vs synthetic on the same machine.

| lane | data | graph | steps | tok/s | elapsed | final loss | peak VRAM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| real fused | real shards | none | 96 | 1,161,468 | 43.33s | 4.7560 | 18,305.1 MB |
| synthetic fused | synthetic shards | none | 96 | 1,165,625 | 43.18s | 9.7044 | 18,305.1 MB |
| real graph | real shards | probe | 96 | 1,154,489 | 43.60s | 4.7748 | 18,369.1 MB |

Takeaway: with the mmap cache already built, real shards did not slow the
steady-state hot loop relative to synthetic shards. CUDA graph capture failed
for the Stage A checkpointed shape with:

`Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture`

That comes from activation checkpointing's RNG-state handling. The runner now
rejects graph capture up front when `activation_checkpoint=true`; graph remains
worth revisiting for the non-checkpointed single-GPU path, but this explains why
the real-data Stage A-shaped graph probe fell back to eager.

## Decisions

- Keep `lm_head_backward_mode=fused_streaming_v2` and `grad_allreduce_mode=bulk`
  as the 8x production defaults.
- Keep `fused_norm_streaming_v2` available for future single-GPU graph work and
  deeper CUDA fusion, but do not promote it for 8x Stage A.
- Keep `async_param` available as a measured negative/neutral arm, but do not
  use it for Stage A/B unless future model sizes make communication heavier.
- CUDA graph deserves a follow-up only if we either run non-checkpointed
  single-GPU experiments or explicitly build a DDP-compatible graph path.
