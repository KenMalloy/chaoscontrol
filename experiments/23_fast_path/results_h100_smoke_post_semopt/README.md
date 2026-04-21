# Exp23 H100 Smoke After Semantic Optimizer Merge

Date: 2026-04-21 UTC

Pod:

- `ja1hbjfl0w0ac3`
- `runpod/parameter-golf:latest`
- 1x `NVIDIA H100 80GB HBM3`
- torch `2.9.1+cu128`, CUDA runtime `12.8`

Verification before probes:

- `tests/test_ssm_scan.py`
- `tests/test_optim_semantic.py`
- `tests/test_exp23_fast_path.py`
- `tests/test_exp23_prefetch.py`
- Result: `76 passed in 6.14s`

Synthetic smoke setup:

- Random uint16 train tokens: `/workspace/smoke_data/fineweb_train_000000.bin`
- Random uint16 val tokens: `/workspace/smoke_data/fineweb_val_000000.bin`
- Tokenizer: `Natooka/parameter-golf-sp-tokenizers/fineweb_16384_bpe.model`
- Model: 4-layer SSM, dim 256, vocab 16384, batch 1024, seq 512, bf16,
  fused grad clip, fused Muon, `_ssm_scan`.

Results:

| Condition | Steps | Tokens/sec | Peak VRAM | Notes |
| --- | ---: | ---: | ---: | --- |
| `h100_b1024_c64_noprefetch` | 40 | 2,145,802 | 42.8 GB | Baseline fused Muon path |
| `h100_b1024_c64_prefetch` | 40 | 2,143,835 | 42.8 GB | Prefetch was neutral at this shape |
| `h100_b1024_c128_prefetch` | 40 | 2,165,168 | 57.2 GB | Fastest smoke, higher head-memory pressure |

Compile probe:

- `h100_b1024_c64_prefetch_compile` was launched but aborted after more than
  two minutes in Inductor compile workers with 0 percent GPU utilization.
- No JSON result was produced.
- Recommendation for Stage A: keep `compile_full_path: false` unless we do a
  separate long-run compile amortization test.

Pod was stopped after harvesting results.
