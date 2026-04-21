# Exp23 8xH100 Tile-8192 Smoke

Date: 2026-04-21

Hardware: 8x H100 SXM secure pod in IN, `runpod/parameter-golf:latest`,
PyTorch 2.9.1 + CUDA 12.8. Pod cost was `$21.52/hr`; it was stopped after the
two timing runs.

Environment:

```bash
PYTHONPATH=src
CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan
torch.distributed.run --standalone --nproc_per_node=8
```

Config shape:

- `lm_head_backward_mode=fused_streaming_v2`
- `lm_head_tile_size=8192`
- `vocab_size=16384`
- `batch_size=1024` per GPU
- `seq_len=512`
- `model_dim=256`
- `num_layers=4`
- `precision=bf16`
- `activation_checkpoint=false`
- `prefetch_batches=true`
- synthetic uint16 token shard, eval disabled

| budget | steps | aggregate tok/s | per-GPU tok/s | step seconds | final loss | peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30s | 172 | 23,822,057 | 2,977,757 | 0.1761 | 9.6395 | 44,946.1 MB |
| 90s | 512 | 23,772,765 | 2,971,596 | 0.1764 | 9.5858 | 44,946.3 MB |

Takeaway: the 8x path clears the 21M tokens/s training-throughput threshold with
about 13% headroom on a synthetic shard. The 90s run was stable and matched the
30s smoke closely. Relative to the 1x `8192` tile result (`3.09M tok/s`), 8x
DDP overhead is about 3.8% per GPU, which is acceptable for Stage A/B planning.

This run does not answer quality on the real full corpus. It answers the speed
question: the current fastest path is fast enough to plausibly see an 8B-token
full corpus within 600s before data-loading and setup overhead.
