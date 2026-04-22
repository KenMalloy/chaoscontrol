# Exp23 Cached CE 8x Base-Lock Probe

Run date: 2026-04-21 local / 2026-04-22 UTC.

Hardware: 8x H100 SXM in India (`AP-IN-1`) on the official
`runpod/parameter-golf:latest` template. Pod id was `kn59k28vo3bdji`; it was
stopped after artifacts were harvested.

Environment:

```text
torch 2.9.1+cu128
CUDA runtime 12.8
CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan
CHAOSCONTROL_POST_SCAN_BACKEND=eager
lm_head_backward_mode=fused_streaming_cached
lm_head_tile_size=8192
batch_size=1024 per GPU
seq_len=512
model_dim=256
num_layers=4
activation_checkpoint=false
grad_allreduce_mode=bulk
```

Dataset artifact: `Natooka/parameter-golf-sp-tokenizers` revision
`e9d696d1592d884dbb97e754efb2a7203aca3080`.

Measured token counts:

```text
train_tokens 13,262,831,920
val_tokens      42,266,034
```

## Speed Results

| Run | LR | Budget | Steps | Aggregate tok/s | Full train-token pass estimate | Final train loss | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cached_30s_real` | 0.128 | 30s | 172 | 25,042,244 | 529.6s | 4.4121 | 53,138 MB |
| `cached_90s_real` | 0.128 | 90s | 528 | 25,016,199 | 530.2s | 3.9928 | 53,138 MB |
| `cached_90s_lr0064_real` | 0.064 | 90s | 528 | 24,989,950 | 530.7s | 3.9584 | 53,138 MB |
| `base_random_lr0064_600s` | 0.064 | 600s | 3572 | 25,023,217 | 530.0s | 3.7463 | 53,160 MB |

The cached CE speedup transferred to 8x. Prior 8x `fused_streaming_v2` was about
23.8M tok/s; cached CE is about 25.0M tok/s on real SP16384 shards, a roughly
5% gain with stable memory.

The `0.064` LR probe matched `0.128` throughput but had better short-run loss,
so the first 600s base-lock run used `base_lr=0.064`.

## Quality Slice

The 600s run performed a small eval slice only, not full validation:

```text
eval loss:          3.7272438556
eval BPB:           1.5054478660
eval tokens:        8,388,608
eval scored bytes: 29,963,085
```

This is not a leaderboard score. It is a quick base-lock quality signal and a
saved checkpoint for follow-up evaluation.

## Artifacts

```text
experiments/23_fast_path/results_cached_ce_8x_20260421/
experiments/23_fast_path/results_base_lock_8x_20260421/
experiments/23_fast_path/checkpoints_base_lock_8x_20260421/base_random_lr0064_600s.pt
experiments/23_fast_path/build_8x_cached.log
```

## Decision

Use `fused_streaming_cached`, `tile_size=8192`, and `base_lr=0.064` as the
current fastest base path. The speed target is met with enough headroom to see
the full SP16384 train-token corpus inside a 600s training loop, assuming the
current random-sampling runner is representative of the final training pass.
