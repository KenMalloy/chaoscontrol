# Exp23 Streaming Linear+CE Smoke

Date: 2026-04-21

Hardware: 1x H100 SXM, `runpod/parameter-golf:latest`, PyTorch 2.9.1 + CUDA 12.8.

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
| 1 | `fused` | 60 | 2,571,109 | 0.2039 | 9.7061 | 30,865.1 MB |
| 2 | `fused_streaming` | 60 | 2,709,385 | 0.1935 | 9.7062 | 30,865.1 MB |
| 3 | `fused_streaming` | 60 | 2,707,409 | 0.1936 | 9.7061 | 30,865.1 MB |
| 4 | `fused` | 60 | 2,571,751 | 0.2039 | 9.7064 | 30,865.1 MB |

Mean throughput:

| mode | mean tok/s | relative |
| --- | ---: | ---: |
| `fused` | 2,571,430 | baseline |
| `fused_streaming` | 2,708,397 | +5.3% |

Takeaway: the streaming fused linear+CE forward removes the duplicate forward
vocab-tiled matmul and improves the single-H100 hot-loop by about 5.3% with no
extra peak VRAM. Keep this as the primary path; Liger/Cut-CE remain useful as
external benchmarks or backups, not the first implementation path.
