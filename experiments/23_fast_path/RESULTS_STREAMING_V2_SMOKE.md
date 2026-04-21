# Exp23 Streaming V2 CE Smoke

Date: 2026-04-21

Hardware: 1x H100 SXM, resumed `runpod/parameter-golf:latest`, PyTorch 2.9.1
+ CUDA 12.8.

Environment:

```bash
PYTHONPATH=src
CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan
```

Config shape:

- `vocab_size=16384`
- `batch_size=1024`
- `seq_len=512`
- `model_dim=256`
- `num_layers=4`
- `precision=bf16`
- `activation_checkpoint=false`
- `prefetch_batches=true`
- `budget=12s`
- synthetic uint16 token shard, eval disabled

The second pair reverses run order to check for page/cache warmup bias.

| run | mode | steps | tok/s | step seconds | final loss | peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `fused_streaming` | 60 | 2,708,883 | 0.1935 | 9.7063 | 30,865.1 MB |
| 2 | `fused_streaming_v2` | 64 | 2,748,923 | 0.1907 | 9.7064 | 30,609.1 MB |
| 3 | `fused_streaming_v2` | 64 | 2,747,292 | 0.1908 | 9.7065 | 30,609.1 MB |
| 4 | `fused_streaming` | 60 | 2,709,029 | 0.1935 | 9.7060 | 30,865.1 MB |

Mean throughput:

| mode | mean tok/s | relative |
| --- | ---: | ---: |
| `fused_streaming` | 2,708,956 | baseline |
| `fused_streaming_v2` | 2,748,108 | +1.4% |

Takeaway: reusing explicit forward/backward tile workspaces and writing
`grad_weight` directly trims allocator/output-copy overhead. The gain is small
but repeatable, and peak VRAM drops by about 256 MB. This is worth keeping but
does not satisfy the 8x gate by itself.
