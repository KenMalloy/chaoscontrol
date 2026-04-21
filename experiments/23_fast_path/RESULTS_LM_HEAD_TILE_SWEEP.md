# Exp23 LM-Head Tile Sweep

Date: 2026-04-21

Hardware: 1x H100 SXM, resumed `runpod/parameter-golf:latest`, PyTorch 2.9.1
+ CUDA 12.8.

Environment:

```bash
PYTHONPATH=src
CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan
```

Config shape:

- `lm_head_backward_mode=fused_streaming_v2`
- `vocab_size=16384`
- `batch_size=1024`
- `seq_len=512`
- `model_dim=256`
- `num_layers=4`
- `precision=bf16`
- `activation_checkpoint=false`
- `prefetch_batches=true`
- synthetic uint16 token shard, eval disabled

Single-pass sweep:

| tile size | steps | tok/s | step seconds | final loss | peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 40 | 2,768,939 | 0.1893 | 9.7071 | 30,609.1 MB |
| 2048 | 44 | 2,971,005 | 0.1765 | 9.7059 | 32,657.1 MB |
| 4096 | 44 | 3,065,880 | 0.1710 | 9.7056 | 36,753.1 MB |
| 8192 | 48 | 3,097,239 | 0.1693 | 9.7054 | 44,945.1 MB |
| 16384 | 48 | 3,102,686 | 0.1690 | 108.3964 | 61,329.1 MB |

Order-controlled repeat:

| run | tile size | steps | tok/s | step seconds | final loss | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 1 | 1024 | 40 | 2,768,939 | 0.1893 | 9.7071 | 30,609.1 MB |
| candidate 1 | 8192 | 48 | 3,097,239 | 0.1693 | 9.7054 | 44,945.1 MB |
| candidate 2 | 8192 | 68 | 3,079,898 | 0.1702 | 9.7057 | 44,945.1 MB |
| baseline 2 | 1024 | 64 | 2,748,728 | 0.1907 | 9.7071 | 30,609.1 MB |

Mean throughput:

| tile size | mean tok/s | relative |
| ---: | ---: | ---: |
| 1024 | 2,758,833 | baseline |
| 8192 | 3,088,568 | +12.0% |

Takeaway: exposing `lm_head_tile_size` is the largest single-H100 training
speed win after streaming CE. `8192` is the practical winner: it cuts CE tile
count from 16 to 2, keeps loss stable, and still fits with about 45GB peak VRAM.
The full-vocab `16384` tile is not acceptable yet because training loss
exploded despite only a tiny speed gain over `8192`.
