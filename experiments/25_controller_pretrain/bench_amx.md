# CPU SSM Simplex Controller — AMX / AVX-512 Benchmark

Phase E4 + S5 deliverable. The benchmark answers two questions:

1. **Per-query simplex policy end-to-end.** Does the controller's per-event
   hot path (`simplex_forward` → `record_simplex_decision` →
   `on_replay_outcome` → REINFORCE backward → SGD apply) keep up with
   the on-pod producer rate (~6.7K replay outcomes/sec inferred from
   the 640 KB/s wire-event sizing in the design doc)? Per-event budget
   is ~150 µs. Anything materially slower means batching becomes
   structurally necessary.
2. **Kernel-level peak.** What's the per-call latency of the
   `_tile_dpbf16ps` AMX kernel and the AVX-512 recurrence in isolation?
   The simplex `M=16` shape is the design target; the bench includes a
   tiled `64x64x32` shape for follow-up coverage of the tiling logic on
   real-controller pretrain dimensions.

The bench captures both in one run.

## What the bench measures

**Isolated kernel timings** (per call, tile-aligned synthetic inputs):

- `generic_fp32_recurrence` — scalar `h = decay * h + x` over 512 fp32
  lanes; reference path on every CPU.
- `avx512_recurrence` — same op via `_mm512_fmadd_ps`; gated on
  AVX-512 build + runtime.
- `amx_bf16_matmul_16x32x16` — single AMX tile `C[16, 16] = A[16, 32]
  @ B[32, 16]` in BF16 with FP32 accumulation. Kernel-level reference;
  same shape as the simplex policy's Layer 1 GEMM (`(16, K_v) @ (K_v,
  H)` with `K_v=16, H=32`).
- `amx_bf16_matmul_64x64x32_tiled` — exercises the K-tile + dst-tile
  loops on a controller-pretrain shape (64-sample minibatch through
  `in_proj`).

**Per-query simplex end-to-end timings** (`SimplexOnlineLearner`):

- `forward_only` — `simplex_forward(weights, V, E, simplex_features)`
  alone. The read path: scoring the simplex for action selection.
  Producer-side cost on every query.
- `decision_record` — forward + `record_simplex_decision`. The write
  path: action chosen, snapshot saved for credit assignment.
- `full_replay_event` — forward + record + `on_replay_outcome` with
  REINFORCE backward + SGD apply (sgd_interval=1 to capture worst-case
  per-event cost; production cadence amortizes SGD across many
  events). The total simplex-controller cost per replay outcome.

The full-event timing is what matters for "does the controller keep up
with the producer rate." `forward_only` is what matters for "does the
read path stay fast enough that the simplex isn't the query-side
bottleneck."

## Local Smoke (Darwin arm64)

```bash
.venv/bin/python experiments/25_controller_pretrain/bench_amx.py \
  --iterations 1000 --warmup 100
```

Expected on the Darwin arm64 / default build:

- `cpu_features.is_x86 == false`
- `kernel_available.{avx512_recurrence, avx512_matops, amx_bf16_matmul} == false`
- `simplex_per_query.full_replay_event.mean_us` well under the 150 µs
  per-event budget — the simplex hot path is `at::matmul`-bound on
  arm64 and the matmul shapes are tiny (≤ 16x32x16 mostly).

Reference numbers from a representative arm64 run (Apple Silicon,
Python 3.14, torch 2.11, 1000 iters):

| metric | value |
|---|---|
| `isolated_kernel.generic_fp32_recurrence.mean_us` | ~1.4 |
| `simplex_per_query.forward_only.mean_us` | ~16 |
| `simplex_per_query.decision_record.mean_us` | ~24 |
| `simplex_per_query.full_replay_event.mean_us` | ~88 |

Per-event budget is ~150 µs at the design-doc producer rate (6.7K
events/sec). At ~88 µs on arm64 with no AVX-512 / AMX, the simplex
controller has ~1.7× headroom even without any vectorization. Streaming
works; the AMX dispatch is a perf upside, not a streaming requirement.

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
  tests/test_amx_matmul_vnni_packing.py \
  tests/test_simplex_policy.py \
  tests/test_simplex_learner.py -q

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

### Per-query simplex end-to-end timings (Sapphire Rapids)

| mode | mean (µs) | median (µs) | p99 (µs) |
|---|---|---|---|
| `forward_only` | TBD | TBD | TBD |
| `decision_record` | TBD | TBD | TBD |
| `full_replay_event` | TBD | TBD | TBD |

### Speedup interpretation

- **Kernel-level AMX vs generic** (the brief's "~10×" claim):
  `generic_fp32_recurrence.mean_us / amx_bf16_matmul_16x32x16.mean_us`.
  Note this crosses ops (recurrence vs matmul) and precisions (fp32 vs
  bf16); the more apples-to-apples ratio is `at::matmul(fp32) /
  amx_bf16_matmul(bf16)` at the same `(16, 32, 16)` shape, which the
  bench doesn't currently measure.
- **Per-query end-to-end**: simplex_forward and full_replay_event on
  SPR vs the arm64 reference numbers above. Expected: AMX dispatch in
  `simplex_policy.cpp`'s three GEMMs (when wired — currently uses
  `at::matmul`) yields a meaningful drop on the (16, K_v, H) shapes
  that match the AMX tile exactly. The on-pod hot-path number is the
  number that determines streaming margin.

## What survives, what's pending

**Survives** from the per-slot V0 phase:

- Tiled AMX BF16 matmul (commit `dcda08b`). Per-query simplex forward
  fits the tile naturally — `M=16` is exactly one A-tile.
- AVX-512 matvec/axpy (commit `d208cdc`). Available for any future
  per-vertex ops; not currently called from `simplex_policy.cpp`.
- Wire-event ring infrastructure (Phases A + B). QueryEvent schema
  bumped (commit `a1b6e72`) for the simplex candidate set.

**Pending for full perf claim**:

- AMX dispatch from inside `simplex_policy.cpp`. The forward currently
  uses `at::matmul` (commit `369f200`); a follow-up will dispatch to
  `amx_bf16_matmul` when the build includes the kernel and the runtime
  has AMX state. This is the single change that turns "AMX is the
  natural shape" into "AMX is the actual shape." Until it lands,
  per-query timings reflect `at::matmul`'s MKL path.
- Runner-side `simplex_v1` runtime mode in
  `_build_controller_runtime_from_config`. Currently the matrix's
  trained arms (c, d, e) carry `episodic_controller_runtime:
  "simplex_v1"`; the runner will reject this until the dispatch lands.
  Heuristic arms (a, b) run unaffected.

Both are mechanical follow-ups, not architecture work.
