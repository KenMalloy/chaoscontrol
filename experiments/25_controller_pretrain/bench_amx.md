# CPU SSM Controller — AMX / AVX-512 Benchmark

Phase E4 deliverable. Two question this bench answers:

1. **Per-event end-to-end speedup.** Does the on-pod controller hot path
   (`OnlineLearningController::accumulate_backward` — forward matvec +
   diagonal recurrence + backward outer-product axpy + credit attribution
   + history append) run faster with AVX-512 wired vs the scalar
   reference? This is the number that determines whether the controller
   keeps up with the GPU training ranks within the 600s pod budget.
2. **Kernel-level peak.** What is the per-call latency of the
   `_tile_dpbf16ps` AMX kernel and the `_mm512_fmadd_ps`-based AVX-512
   recurrence in isolation? This is the number the brief's "~10×" claim
   is about. It bounds the speedup any caller can hope to realize when
   call-site overhead (vector construction, Python boundary crossings,
   tensor object cost) is amortized to zero.

The bench captures both in one run.

## What the bench measures

**Isolated kernel timings** (per call, tile-aligned synthetic inputs):

- `generic_fp32_recurrence`: scalar `h = decay * h + x` over 512 fp32
  lanes. The reference path; runs on every CPU.
- `avx512_recurrence`: same op, vectorized via `_mm512_fmadd_ps`. Gated
  on `avx512_recurrence_kernel_available() && has_avx512f()`.
- `amx_bf16_matmul_16x32x16`: single AMX tile `C[16, 16] = A[16, 32] @ B[32, 16]`
  in BF16 with FP32 accumulation. The kernel-level perf claim's
  reference shape — what `_tile_dpbf16ps` was designed to chew on.
- `amx_bf16_matmul_64x64x32_tiled`: the tiled kernel doing the actual
  output-tile + K-tile loops. `M=64` × `N=32` requires 4×2=8 dst tiles,
  `K=64` requires 2 K-tiles per dst, so the kernel issues 16
  `_tile_dpbf16ps`. Mirrors a 64-sample minibatch through controller
  pretrain `in_proj` (`fdim=64 -> d_global=32`); validates the tiling
  logic at a representative real-controller shape.

**Per-event end-to-end timings** (`OnlineLearningController.on_replay_outcome`):

- `controller_per_event.scalar`: `set_use_avx512_matops(False)` forces
  the scalar dispatch even on AVX-512 hardware. Measures the reference
  path's per-event latency.
- `controller_per_event.avx512`: `set_use_avx512_matops(True)` forces
  AVX-512. Only run when `avx512_matops_kernel_available() && has_avx512f()`.

The end-to-end timing includes a `record_replay_selection` call per
iteration so the replay outcome has a stateful entry to credit. That
matches the on-pod producer shape (one selection per replay event,
backed by a write-side staging) and ensures `accumulate_backward`
actually runs (not the missing-state skip path).

## Local Smoke (Darwin arm64)

```bash
.venv/bin/python experiments/25_controller_pretrain/bench_amx.py \
  --iterations 1000 --warmup 100
```

Expected on the Darwin arm64 / default build:

- `cpu_features.is_x86 == false`
- `kernel_available.avx512_matops == false` and friends
- `controller_per_event.scalar.uses_avx512_matops == false`
- `controller_per_event.avx512.available == false`

The arm64 numbers exercise the scalar code path end-to-end and confirm
the bench harness itself is correct. Reference numbers from a
representative arm64 run (Apple Silicon, Python 3.14, torch 2.11):

| metric | value |
|---|---|
| `isolated_kernel.generic_fp32_recurrence.mean_us` | ~1.37 |
| `controller_per_event.scalar.mean_us` | ~26.1 |

The controller per-event number is dominated by Python boundary
crossings and the vector-construction in `record_replay_selection`,
not by the matops themselves. The on-pod AVX-512 speedup will move
the matops portion 4-8x but won't speed up the Python boundary —
realized end-to-end speedup is therefore expected to be smaller than
the kernel-level ratio. That's the point of measuring both.

## Sapphire Rapids Run (F3 runbook step)

On the 26-vCPU Sapphire Rapids pod, in the project root:

```bash
CHAOSCONTROL_CPU_SSM_X86_ACCEL=1 \
  .venv/bin/python src/chaoscontrol/kernels/_cpu_ssm_controller/setup_ext.py \
    build_ext --inplace

.venv/bin/python -m pytest \
  tests/test_cpu_capability_detection.py \
  tests/test_avx512_recurrence.py \
  tests/test_avx512_matops.py \
  tests/test_amx_matmul.py \
  tests/test_amx_matmul_vnni_packing.py -q

.venv/bin/python experiments/25_controller_pretrain/bench_amx.py \
  --iterations 100000 --warmup 1000 \
  --output experiments/25_controller_pretrain/bench_amx_spr.json
```

Commit `bench_amx_spr.json` and replace the **Pod-measured numbers**
section below with the harvested values.

## Pod-measured numbers (TO BE FILLED post-F3)

Replace this stub with the numbers from `bench_amx_spr.json` after the
F3 pod run.

### Isolated kernel timings (Sapphire Rapids)

| mode | mean (µs) | median (µs) | p99 (µs) |
|---|---|---|---|
| `generic_fp32_recurrence` | TBD | TBD | TBD |
| `avx512_recurrence` | TBD | TBD | TBD |
| `amx_bf16_matmul_16x32x16` | TBD | TBD | TBD |
| `amx_bf16_matmul_64x64x32_tiled` | TBD | TBD | TBD |

### Per-event end-to-end timings (Sapphire Rapids)

| dispatch | mean (µs) | median (µs) | p99 (µs) |
|---|---|---|---|
| `scalar` | TBD | TBD | TBD |
| `avx512` | TBD | TBD | TBD |

### Speedup interpretation

- **Kernel-level AMX vs generic** (the brief's "~10×" claim): compute
  `generic_fp32_recurrence.mean_us / amx_bf16_matmul_16x32x16.mean_us`,
  but note the comparison crosses ops (recurrence vs matmul) and
  precisions (fp32 vs bf16) — the closer apples-to-apples ratio is
  `at::matmul(fp32) / amx_bf16_matmul(bf16)` which the bench doesn't
  currently measure.
- **Per-event end-to-end** (the on-pod claim): compute
  `controller_per_event.scalar.mean_us / controller_per_event.avx512.mean_us`.
  This is the realized speedup the controller actually sees during
  the 600s training window, with all Python and tensor-object
  overheads still in the timer.

## Out of scope for this bench

- **AMX wired into the controller's per-event path.**
  `accumulate_backward` runs `M=1` per replay event — single-tile AMX
  wastes 15/16 of the tile no matter how cleanly wired. The kernel
  itself (commit `dcda08b`, tiled, arbitrary `M, N, K`) ships and is
  validated locally + ready for SPR parity, but the per-event call
  site doesn't use it. The realistic AMX target is a controller-arch
  change that batches replay events 16-at-a-time before issuing
  `accumulate_backward`. That change is out of scope for E4.
- **AMX wired into `forward_pretrain_model`.** That's the offline
  pretrain pipeline (bootstraps controller weights from heuristic
  traces); speedup there doesn't help the on-pod 600s budget. A
  future commit could swap it once a benchmark proves AMX BF16
  beats `at::matmul`'s MKL fp32 path at controller-pretrain shapes.
