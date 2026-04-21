# Exp23 Fused Linear+CE Smoke

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
- `prefetch_batches=true`
- `budget=8s`

| mode | steps | tok/s | final loss | peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| `chunked`, `chunk_size=64` | 32 | 2,127,431 | 8.5126 | 42,761.9 MB |
| `fused` linear+CE | 40 | 2,558,276 | 8.3532 | 30,866.1 MB |

Takeaway: the native fused linear+CE path removed the full-logit OOM fallback
at the submission-shaped smoke point and improved 1xH100 throughput by about
20% while reducing peak memory by about 11.9 GB.
