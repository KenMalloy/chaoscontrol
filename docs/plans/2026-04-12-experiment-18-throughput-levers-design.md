# Experiment 18: Throughput Lever Stack

**Date:** 2026-04-12
**Status:** Design approved, implementation pending
**Supersedes:** `2026-04-11-experiment-18-throughput-advantage-design.md` and
`2026-04-11-experiment-18-throughput-advantage-impl.md`

## Role in the experiment sequence

The Apr 11 Exp 18 design framed throughput as a data-coverage problem
("see more of the corpus, pick the hardest windows, retrain on them").
Phase 0 benchmarks falsified the premise: even at maximum single-GPU
throughput (bs=1024, ~98K tok/s, 96% VRAM) scaled to 8×H100 DDP
(~786K tok/s theoretical ceiling), the 600s budget sees only ~4.7% of
the 10B corpus. Targeted subset selection over 4.7% is not a meaningful
axis.

The reframing splits what remains:

- **Exp 18 = training throughput tuning.** Maximize tokens-per-wall-second,
  and learning-signal-per-token, on the training side. Every lever that
  multiplies useful tokens seen in the 600s budget belongs here.
- **Exp 19 = submission tuning.** With the throughput Exp 18 establishes,
  dial in the final architecture choices (depth recurrence, quantization,
  final dim/layers/ff_mult, LR schedule, seeds) that fit 16MB and hit the
  best possible bpb.

Exp 18 is **not** where we make architectural commitments. It's where we
find out how much we can feed the training loop.

## The 10B-token reality check

The submission target is "as much learning as possible on FineWeb in
600s on 8×H100." Raw numbers from Phase 0:

| Scenario | tok/s | 600s total | % of 10B |
|---|---|---|---|
| Current single GPU, bs=1024, AdamW | 98,299 | 59M | 0.59% |
| Naive 8× DDP | ~786K | 471M | 4.7% |
| SOTA transformer (flash-attn 3, seq=2048) | ~760K × 8 | 3.65B | 36.5% |
| Optimistic stack (all Exp 18 wins) | ~5.6M | 3.4B | 34% |

**The SOTA transformer sees 7.8× more tokens per GPU than our current
SSM.** We're not slower because SSMs are slow — we're slower because our
diag scan is going through `torch.compile` fallback while transformers
have hand-tuned flash-attention kernels. Closing this gap is the biggest
single thing Exp 18 can do.

## Core thesis

A pure SSM trained with SOTA-level infrastructure (optimized scan kernel,
large-batch optimizer, maximum batch × seq_len in fixed VRAM, 8×H100 DDP)
can reach throughput comparable to transformers while preserving the
structural advantage that attention can't match: **linear cost in sequence
length.** Within the 600s / 16MB constraints, that translates into either
more tokens seen at fixed model size, or a larger model at fixed tokens
seen — whichever shape Exp 19 benefits from most.

## Lever inventory

Ordered by estimated impact on tokens-of-useful-learning-per-600s.

### Tier 1 — highest leverage

**L1. Mamba scan kernel swap.**
The official `mamba_ssm` package ships a fused CUDA kernel for the
diagonal scan with tensor-core-native reductions and fused softplus/exp.
Reported 1.5-3× faster than compiled-scan implementations. Our current
path goes through `torch.compile`, which fails on some CUDA/Inductor
stacks and falls back to a Python loop. Swapping in `mamba_ssm.ops`
should yield a clean throughput multiplier with identical numerical
semantics.

**Expected impact:** 1.5-3× tokens/sec per GPU. Single biggest lever.

**Risk:** Correctness — need to verify the kernel's diag semantics match
our `ChaosSSMCore._forward_diag_scan` exactly (decay parameterization,
gate application, delta projection). A numerical parity test against our
current implementation gates this.

**L2. Large-batch optimizer (Muon, LAMB, or Sophia).**
AdamW is a bad fit for the large-batch regime we want to operate in.
Competition SOTA uses Muon with momentum 0.97 and reports ~2× efficiency
vs AdamW. LAMB is specifically designed for large-batch training (our
target: global batch 8192+ via bs=1024 × 8 DDP). Sophia is a second-order
method designed for LM training. All three are drop-in replacements.

**Expected impact:** 1.5-2× learning-per-token. At fixed wall time,
equivalent to 1.5-2× more tokens seen.

**Risk:** Implementation correctness and LR scaling. Muon needs Newton-
Schulz matrix orthogonalization in the update step; we'd lift from the
competition SOTA `train_gpt.py` which has a reference implementation.
LAMB is simpler but we'd need to tune separately.

**L3. DDP 8× validation.**
We have measured single-GPU throughput. We have **not** measured
real-world DDP overhead at global batch 8192. At our model size (~13MB
before quantization), all-reduce should be cheap, but gradient variance
at this batch scale might force us to re-tune LR (our current linear-
scaling LR was validated only up to bs=1024 on a single GPU).

**Expected impact:** 7-8× tokens/sec (near-linear DDP scaling, assumed).
Measuring confirms what we're already assuming.

**Risk:** LR stability at large global batch. Our Phase 0 LR screen
tested linear=0.064, sqrt=0.011, fixed=0.002 at bs=1024 single-GPU;
only linear was stable. Global batch 8192 might require either a
smaller LR, longer warmup, or gradient clipping adjustment.

### Tier 2 — meaningful but smaller

**L4. Sequence length scaling.**
We have trained SSMs exclusively at seq_len=512. The SSM recurrence
cost is linear in seq_len, and the SSM state is fixed-size. At fixed
VRAM, going from seq_len=512 to seq_len=2048 should cost the same
memory (SSM state is fixed) and produce 4× more tokens per step — if
step time scales linearly. SOTA uses seq_len=2048. We need to sweep.

**Expected impact:** 1-4× tokens/sec depending on how step time scales
with seq_len at fixed batch. Might also affect per-token learning
efficiency (longer sequences give more context per prediction).

**Risk:** Longer sequences may not deliver proportional learning gain
(tokens at the end of long sequences are more predictable than tokens
at the start). Need to measure both tok/s and per-token loss.

**L5. fp8 mixed precision.**
H100 has native fp8 tensor cores. Training in fp8 would halve memory
(doubling max batch or enabling longer sequences at fixed batch) and
speed up matmul. This is active research territory — training stability
in fp8 is not yet solved for all architectures.

**Expected impact:** 1.5-2× throughput at fixed VRAM, via larger batch
or longer sequences.

**Risk:** Training divergence. fp8 dynamic range is small and SSM
recurrences may accumulate drift. Scope a 2-hour exploration; if it
destabilizes easily, defer.

### Tier 3 — infrastructure

**L6. DDP all-reduce profiling / gradient compression.**
If DDP overhead turns out larger than expected (>10-15% at global batch
8192), gradient compression or bf16 all-reduce could recover it. Only
becomes relevant if L3 measurement shows real overhead.

**L7. Data pipeline instrumentation.**
Confirmed non-bottleneck at single-GPU scale (step time grew 42% from
bs=32 to bs=1024 while batch grew 32×, so compute dominates). At 8×DDP
with workers hitting the same shards, I/O might become a constraint.
Instrument and confirm, don't optimize preemptively.

### What's deliberately NOT in Exp 18

**Depth recurrence / weight sharing.** This is a capacity lever
(trades throughput for effective depth at fixed param count), not a
throughput lever. It belongs in Exp 19 as a submission dial.

**GPTQ int6 quantization.** Post-training quantization, applied only
to the final artifact. Doesn't affect training throughput. Exp 19.

**Architecture search (dim, layers, ff_mult, vocab_size).** Submission
dials, Exp 19.

**Retrieval / attention / memory / Wernicke.** Four prior experiments
show it doesn't help in the strong-baseline regime. Not revisited
unless the research assistant lit search surfaces a specific variant
we haven't tried.

**TTT (test-time training).** Inference-time adaptation. Not a training
throughput concern.

## Test plan

Each test is scoped for a clean measurement in minimum pod time. Tests
run sequentially; later tests use winners from earlier tests as the base
config.

### Test 0: Mamba kernel parity (preflight, local or cheap CUDA)

Before swapping in `mamba_ssm`, verify numerical parity with our current
`_forward_diag_scan`. Build a test that compares output tensors on
random inputs across a range of (batch, seq, dim) shapes. Tolerance:
1e-4 max diff (bf16 numerical noise floor).

**Gate:** if parity fails, drop L1 and investigate upstream — either
our core has a non-standard recurrence that `mamba_ssm` doesn't match,
or we need a custom kernel variant. Either outcome is informative.

### Test 1: Single-GPU throughput with mamba kernel

Re-run Phase 0 `bench_throughput.py` with the mamba kernel active,
matched batch sweep {32, 128, 256, 512, 1024}. Produces a direct
comparison against the existing Phase 0 curve.

**Metric:** tok/s at each batch size, peak VRAM, step time.
**Budget:** ~30 min on single H100.
**Gate:** mamba kernel must beat torch.compile by ≥1.5× at bs=1024 or
L1 is dropped from the stack.

### Test 2: DDP 8× scaling

Run single-condition training at bs=1024 per GPU, seq=512, AdamW, on
8×H100 DDP. Measure global tok/s, all-reduce overhead, per-step wall
time. Run for 200 steps (long enough to get stable timing, short
enough to be cheap).

**Metric:** actual tokens/sec vs 8× single-GPU extrapolation. Gradient
variance at global batch 8192 vs single-GPU bs=1024.
**Budget:** ~20 min on 8×H100.
**Gate:** DDP efficiency must be ≥85% of linear (≥670K tok/s actual
vs 786K ideal) or we need to debug.

### Test 3: LR stability screen at DDP scale

At global batch 8192 (bs=1024 × 8), screen three LR candidates:
linear-scaled from Phase 0, linear/2, linear/4. 200 steps each. Same
NaN/stability checks as Phase 0's `_lr_screen`.

**Metric:** which LR converges stably at global batch 8192.
**Budget:** ~30 min on 8×H100.
**Gate:** at least one LR must be stable. If none, escalate — possibly
need gradient clipping changes or warmup extension.

### Test 4: Sequence length sweep

At winning batch × optimizer from tests 1-3, vary seq_len ∈
{512, 1024, 2048} at a fixed (batch, VRAM) envelope. For SSMs, longer
seq_len at fixed batch should cost similar VRAM but more compute. Key
unknown: does step time grow linearly (1×, 2×, 4×) or sublinearly with
seq_len? And does per-token loss improve with longer context?

**Metric:** tok/s, step time, per-token loss (first 200 steps).
**Budget:** ~1 hour on 8×H100 (3 seq_len × 10 min per run, with some
setup overhead).
**Gate:** pick the seq_len that maximizes tok/s × learning-rate-of-loss
(ad hoc metric: `(bpb_0 - bpb_200) / wall_time`).

### Test 5: Optimizer swap (Muon vs AdamW)

At winning (batch, seq_len, DDP) from tests 1-4, compare Muon against
AdamW for 1000 steps. Check both throughput (Muon's Newton-Schulz has
some overhead) and convergence speed (Muon's main benefit).

**Metric:** loss curve vs steps, loss curve vs wall time. Muon's value
is learning-per-token, so the wall-time curve is what matters.
**Budget:** ~45 min on 8×H100.
**Gate:** Muon must show visibly faster loss decrease vs wall time to
earn inclusion. If loss curves are similar, stick with AdamW (simpler).

### Optional Test 6: fp8 exploration

Time-boxed to 2 hours on 8×H100. Enable fp8 mixed precision (via
`torch.nn.functional.scaled_mm` or equivalent). Attempt same config as
winner. Watch for training divergence.

**Metric:** does it run to completion? What's the new tok/s?
**Gate:** if divergence or corruption in 200 steps, shut down and
defer. No pod time on fp8 debugging.

### Test 7: Composition — stacked winners

Run the winning configuration from all tests (kernel + DDP + optimizer
+ seq_len + optional fp8) for a full 600s training run. Measure final
tok/s, total tokens seen, and bpb on a small held-out slice.

**Metric:** "what would the submission look like if we trained it
today at peak throughput?" This is the handoff to Exp 19.
**Budget:** ~15 min on 8×H100 (600s training + overhead).

## Total budget estimate

| Test | GPUs | Duration | Cost est |
|---|---|---|---|
| 0. Mamba parity | 1 | 5 min | $0.20 |
| 1. Kernel throughput | 1 | 30 min | $1.20 |
| 2. DDP scaling | 8 | 20 min | ~$8 |
| 3. LR screen | 8 | 30 min | ~$12 |
| 4. Seq_len sweep | 8 | 60 min | ~$24 |
| 5. Optimizer swap | 8 | 45 min | ~$18 |
| 6. fp8 (optional) | 8 | 120 min | ~$48 |
| 7. Stacked composition | 8 | 15 min | ~$6 |
| **Total (no fp8)** | | **~3.5 hours** | **~$70** |
| **Total (with fp8)** | | **~5.5 hours** | **~$118** |

Assumes $23/hr for 8×H100 SXM. Conservative — actual tests may run
faster with short warmup windows.

## Go/no-go gates summary

- **L1 (mamba kernel) <1.5× speedup:** drop L1, investigate why.
  Throughput ceiling stays at ~98K/GPU.
- **L2 (Muon) shows no wall-clock convergence advantage:** stick with
  AdamW, save optimizer implementation cost.
- **L3 (DDP) <85% efficiency:** pause before large-batch training,
  profile all-reduce, consider gradient compression.
- **L4 (seq_len) no improvement past 512:** fix at 512, use bench
  results.
- **L5 (fp8) unstable in 200 steps:** defer, revisit post-April 30.
- **Stacked test final tok/s <2× baseline:** something is leaving
  throughput on the floor. Investigate before committing to Exp 19.

## What we'll know at the end of Exp 18

Regardless of which levers pay off:

1. **Realistic peak tok/s for our SSM at 600s / 8×H100.** This is the
   throughput ceiling Exp 19 has to work within. All submission
   decisions downstream depend on this number.
2. **Whether SSM + SOTA infrastructure closes the per-GPU gap against
   transformers.** If our stacked winner is within 2× of SOTA tok/s
   (380K+), the pure-SSM thesis is strong. If not, we need to
   understand why (profiling, probably the scan kernel).
3. **Which optimizer to use in Exp 19.** A single yes/no on Muon.
4. **Which (batch, seq_len) shape to use in Exp 19.** A single best
   point on the throughput curve.
5. **Whether fp8 is a viable submission direction.** A single yes/no.

## Implementation notes

- **Fork point:** Exp 18 work happens on a new branch,
  `exp-18-throughput-levers`, based on `main` after the Exp 17 finalize.
- **Test infrastructure:** Extend `bench_throughput.py` with `--kernel`,
  `--optimizer`, `--seq-len`, `--world-size` flags. Each test is a
  single script invocation with the right flags.
- **Parity test:** New unit test `tests/test_mamba_kernel_parity.py`
  that compares `ChaosSSMCore._forward_diag_scan` against
  `mamba_ssm.ops.selective_scan_fn` with matched parameters. Runs in
  CI as a guard against future regressions.
- **DDP wrapper:** Either use `torch.nn.parallel.DistributedDataParallel`
  directly in a small wrapper around `runner_exp18.py`, or lift the
  DDP setup from the SOTA `train_gpt.py`. The latter is fewer lines of
  code and already validated at the same scale.
- **Muon implementation:** Lift the Newton-Schulz optimizer from SOTA
  `train_gpt.py` (it's ~100 lines, self-contained). Add a
  `--optimizer` flag to the runner.

## Pending: research assistant lit search

A lit search is in flight on whether SSM + retrieval mechanisms have
ever helped at compact (sub-hour) training budgets. The result feeds
Exp 19's architecture choice (pure SSM vs SSM+retrieval variant), not
Exp 18 directly. Exp 18's plan is independent of the RA output.

## How this feeds Exp 19

Exp 19 inherits from Exp 18:
- A validated maximum-throughput training config (kernel, optimizer,
  batch, seq_len, LR, DDP world size).
- A baseline bpb on the stacked-winners composition run (Test 7).
- A clear understanding of which levers matter and which don't, so
  Exp 19 doesn't re-litigate them.

Exp 19 then adds:
- Architecture search (dim, layers, ff_mult, vocab_size).
- Depth recurrence / weight sharing at fixed parameter budget.
- INT6 GPTQ + zstd post-training quantization.
- Final artifact packaging (must fit in 16MB).
- Final 3-seed submission runs.
