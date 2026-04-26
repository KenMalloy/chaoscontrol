# CPU SSM Controller AMX Benchmark

This is the Phase E4 benchmark harness for the CPU SSM controller. It records
host capabilities, kernel availability, and per-call latency for:

- generic fp32 diagonal recurrence
- AVX-512 diagonal recurrence, when the extension is built with x86 accel flags
- AMX BF16 16x32x16 matmul, when the extension is built with x86 accel flags

Local development machines that do not expose AVX-512/AMX report those modes as
unavailable. That is intentional; unavailable modes are not silently replaced by
generic fallback timings.

## Local Smoke

Run:

```bash
.venv/bin/python experiments/25_controller_pretrain/bench_amx.py \
  --iterations 1000 \
  --warmup 100
```

Expected on Darwin arm64/default builds:

- `cpu_features.is_x86 == false`
- `avx512_recurrence.available == false`
- `amx_bf16_matmul_16x32x16.available == false`

## Sapphire Rapids Run

On the 26-vCPU Sapphire Rapids pod:

```bash
CHAOSCONTROL_CPU_SSM_X86_ACCEL=1 \
  .venv/bin/python src/chaoscontrol/kernels/_cpu_ssm_controller/setup_ext.py build_ext --inplace

.venv/bin/python -m pytest \
  tests/test_cpu_capability_detection.py \
  tests/test_avx512_recurrence.py \
  tests/test_amx_matmul.py -q

.venv/bin/python experiments/25_controller_pretrain/bench_amx.py \
  --iterations 100000 \
  --warmup 1000 \
  --output experiments/25_controller_pretrain/bench_amx_spr.json
```

Commit `bench_amx_spr.json` and replace this section with the measured mean,
median, and p99 latencies for each available mode. Until that pod run exists,
this document should not claim Sapphire Rapids speedups.
