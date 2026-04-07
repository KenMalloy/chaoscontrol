# Parameter Golf Baseline

Competition: [openai/parameter-golf](https://github.com/openai/parameter-golf)
Track: 10min training, 16MB artifact, FineWeb validation, bits-per-byte metric.

## Files

- `train_gpt.py` — Competition baseline (9L, 512d, 1024 vocab, ~1.2244 bpb)
- `cached_challenge_fineweb.py` — Data downloader (FineWeb shards from HuggingFace)
- `tokenizer_specs.json` — Tokenizer definitions
- `requirements.txt` — Dependencies
- `sota/` — Current SOTA record (1.1147 bpb, 8xH100 required)

## Reference Scores

| Entry | bpb | Hardware | Notes |
|-------|-----|----------|-------|
| Competition baseline | 1.2244 | 1-8x GPU | Runnable on modest hardware |
| SOTA (merged) | 1.1147 | 8xH100 SXM | FA3 required |
| Pending claims | ~1.08 | 8xH100 SXM | Under review |

## Running the Baseline

```bash
# 1. Install deps
pip install sentencepiece huggingface-hub numpy torch

# 2. Download FineWeb data (sp1024 variant, 80 train shards + val)
cd baselines/parameter_golf
python cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# 3. Run baseline (single GPU, 10 min)
python train_gpt.py

# Multi-GPU:
# torchrun --standalone --nproc_per_node=N train_gpt.py
```

## bpb Calculation

The competition uses tokenizer-agnostic bpb:
```
bpb = (mean_ce_nats / ln(2)) * (total_tokens / total_raw_bytes)
    = total_ce_nats / total_raw_bytes / ln(2)
```

This is identical to our `compute_bpb()` in `src/chaoscontrol/evaluation.py`.
