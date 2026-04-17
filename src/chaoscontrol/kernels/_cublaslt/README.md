# `_cublaslt` — bespoke fp8 matmul via cuBLASLt

Phase 1 of replacing `torch._scaled_mm` with a direct cuBLASLt path that
bypasses torch's dispatcher overhead. Exposes a single Python function,
`cublaslt_fp8_matmul`, with `torch._scaled_mm`-compatible semantics.

## Status

- **Scope:** forward only. One fused `cublasLtMatmul` call on E4M3 operands
  with per-tensor fp32 scales and optional bf16 bias fused through
  `CUBLASLT_EPILOGUE_BIAS`. No backward, no deferred amax, no MXFP8.
- **Upstream:** NVIDIA TransformerEngine v2.13
  (commit `287770466f0f4433052260a765db5ff7b8be1320`), Apache 2.0.
  See `NOTICE` for the exact derivation disposition — this extension is a
  clean-room rewrite informed by TE's source, not a byte-copy. `LICENSE_UPSTREAM`
  preserves the upstream Apache 2.0 license text.

## API

```python
from chaoscontrol.kernels._cublaslt import cublaslt_fp8_matmul

y = cublaslt_fp8_matmul(
    a,            # fp8 E4M3, row-major, [M, K]
    b,            # fp8 E4M3, column-major, [K, N]
    scale_a,      # fp32 scalar device tensor — dequant multiplier for a
    scale_b,      # fp32 scalar device tensor — dequant multiplier for b
    bias=None,    # bf16 [N] or None
    out_dtype=torch.bfloat16,
)  # -> [M, N] in out_dtype
```

Contract matches `torch._scaled_mm`. The scales are passed directly to
cuBLASLt's `CUBLASLT_MATMUL_DESC_A/B_SCALE_POINTER` slots; the kernel
computes `(scale_a * a) @ (scale_b * b)` on the accumulator side.

## Build

The extension is a `torch.utils.cpp_extension.CUDAExtension` compiled by
`setup_ext.py`. It is included as part of this package's editable install:

```bash
pip install -e .
```

`setup.py` (in the repo root) delegates to `setup_ext.py` when it detects a
live CUDA toolchain + cublasLt headers; on dev machines without CUDA the
extension is silently skipped and `cublaslt_fp8_matmul` raises a clear
`ImportError` at call time.

## Where to look first for debugging

- `src/cublaslt_fp8_matmul.cpp` — the whole kernel. <300 lines.
- `src/workspace_cache.h` — the per-process workspace + handle holder.
- Parity + throughput smoke: `tests/test_cublaslt_fp8.py`.
