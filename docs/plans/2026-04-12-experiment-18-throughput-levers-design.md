# Experiment 18: Throughput Lever Stack

**Date:** 2026-04-12 (rewritten 2026-04-13 after Phase 1 implementation landed)
**Status:** Test 1 complete. Tests 2-10 designed. Phase 1 feature branches ready for review.
**Supersedes:** `2026-04-11-experiment-18-throughput-advantage-design.md` and `-impl.md`

## Scope rule (load-bearing principle)

**Any lever that increases corpus-per-600s is in scope for Exp 18.** That's the single filter. Tokenizer compression (more bytes per token), scan kernel speed, batch size, sequence length, DDP parallelism, and kernel fusion all qualify because they move tokens-per-wall-second or bytes-per-token upward. Optimizer changes are in scope when they increase learning-per-token at fixed wall time.

Out of scope: architectural capacity changes (depth recurrence → Exp 19), retrieval mechanisms (falsified in Exp 09/14/17 + lit review), post-training quantization (Exp 19 submission path), TTT (Exp 20 stub).

## Role in the experiment sequence

- **Exp 18 = training throughput tuning.** Deliverable: a validated training config (kernel, optimizer, batch, seq_len, LR, DDP world size, tokenizer) plus a baseline bpb measured at peak throughput.
- **Exp 19 = submission tuning.** With Exp 18's validated stack, dial in final architecture (depth recurrence, ChaosAttentionBlock control, dim/layers/ff_mult sweep), post-training quantization, final 3-seed submission runs.

Exp 18 is not where we make architectural commitments. It's where we find out how much we can feed the training loop.

## Core thesis

A pure SSM trained with optimized infrastructure (fast scan kernel, large-batch optimizer, maximum batch × seq_len in fixed VRAM, 8×H100 DDP, fused block hot path) can reach throughput comparable to transformers while preserving linear cost in sequence length. Within the 600s / 16MB constraints, that translates into either more tokens seen at fixed model size or a larger model at fixed tokens — whichever shape Exp 19 benefits from most.

## Implementation status (2026-04-13)

Test 1 landed directly on `exp-17-local-attn-sidecar` at commits `713d313` + `69e05d4`. Four Exp 18 feature branches are committed, locally tested, and ready for review. Each branches off `69e05d4` and is independent of the others.

| Branch | Tip | Lever | Tests |
|---|---|---|---|
| `feat/ddp-integration` | `5b080b4` | DDP 8× wrap + rank-aware sharding + rank-0 guards + `metabolic_fork`/`mcts` guard + `runner_exp18.py` entry point | 9 DDP tests + 427 full-suite pass on CPU |
| `feat/muon-optimizer` | `6202154` | Newton-Schulz matrix orthogonalization + AdamW fallback for non-matrix params | parity tests present |
| `feat/lamb-optimizer` | `f6c9bdb` | Per-layer trust ratio, scale-invariant at large batch | 14 CPU tests pass |
| `feat/block-kernel-fusion` | `6a562d7` | Post-scan RMSNorm + residual + FF + residual fused via `torch.compile` | 9 parity tests pass (bit-exact at fp32, <1e-3 at bf16) |

Three additional branches from the same dispatch round are Exp 19 Phase 1 scope (not Exp 18) and are called out here only because the Codex review should see them together:

| Branch | Tip | Exp 19 lever |
|---|---|---|
| `feat/depth-recurrence` | `bc9ebf3` | Weight-tied virtual layers in `ChaosStudentLM`, bit-identical at `count=1` to baseline |
| `feat/gptq-quantization` | `e8d4711` | int6 GPTQ + AR self-gen calibration + LZMA preset=9 packaging |
| `feat/chaos-attention-block` | `3ca1f63` | Pre-norm SDPA block, Exp 19 Phase 2 SSM-vs-attention control |

## Test 1: Chunked scan backend — COMPLETE

**Result:** 763K tok/s single-GPU at bs=512 on chunked backend, 7.77× over torch.compile baseline (and 13-77× at smaller batches). Gradient-parity verified to float32 noise floor against the Python loop reference. Bit-identical loss trajectories under AdamW for 30 steps at fixed seed. See `memory/project_exp18_test1_chunked_scan_2026-04-12.md`.

The chunked scan is the default backend for all subsequent Exp 18 tests. The original L1 "mamba kernel swap" plan was replaced by this simpler chunked cumprod+cumsum implementation — same gate target (≥1.5× at bs=1024), passed by 9-50×.

**Residual open question:** whether the chunked backend plays cleanly with DDP 8× all-reduce under real gradient load. Addressed in Test 4.

## Test 2: Tokenizer revisit — SP8192 vs SP16384 on chunked backend

**Hypothesis (alternative):** at matched 600s wall-clock on the chunked backend, SP16384 produces lower bpb than SP8192.
**Null:** the two tokenizers are statistically indistinguishable.

**Context.** SP16384 won Exp 15 SSM configs (1.959 vs SP8192 1.967) then was shelved after a single OOM at bs=1024 on the old torch.compile backend. That shelving decision was never retested with the chunked scan. Flagged as a missing throughput question in the 2026-04-13 review.

**Conditions:**
1. `bare_fast_ssm_sp8192_d256_L4` on chunked backend
2. `bare_fast_ssm_sp16384_d256_L4` on chunked backend

**Within-test control:** matched seeds (Exp 17 set: 1337, 2674, 4011, 5348, 6685, 8022, 9359), matched dim/layers/LR/step count, paired t-test across 7 seeds.

**Power.** Exp 17 per-condition std ≈0.004 bpb → 7 seeds detects ≥0.008 bpb effect at 80% power. The Exp 15 gap was exactly 0.008 bpb. This test is powered to rediscover it if the new throughput preserves the advantage. Flagged explicitly: effects <0.004 bpb will produce false-null readings at this sample size.

**Gate.** SP16384 beats SP8192 at p<0.05 paired → SP16384 becomes the Exp 19 submission tokenizer. Otherwise SP8192 stays. No goalpost moving, no "marginal" calls at p<0.10.

**Budget:** single spot H100, ~4 GPU-hours, ~$8.

**Branches used:** none new. Uses `exp-17-local-attn-sidecar` chunked backend directly.

## Test 3: Activation checkpointing — push batch ceiling

**Hypothesis:** `torch.utils.checkpoint` on the block stack lets us push from bs=512 (chunked backend VRAM ceiling, bound by fp64 cumprod intermediates) to bs=1024 or higher on single H100.

**Conditions:** (no-ckpt, bs=512), (ckpt, bs=1024), (ckpt, bs=2048).

**Gate:** ckpt/bs=1024 must deliver ≥1.3× tok/s over no-ckpt/bs=512, accounting for the ~30% recompute overhead per step. Below: park the lever.

**Corpus-rule justification:** bigger batch → more tokens per step at fixed wall time.
**SSM-native justification:** architecture-agnostic memory/compute trade. Not a transformer borrow.

**Budget:** ~1 GPU-hour on single spot, ~$2.

**Branches used:** none new — implemented inline during the test.

## Test 4: DDP 8× scaling validation

**Hypothesis:** the chunked backend's 763K tok/s per-GPU scales to 8× DDP with ≥85% efficiency.

**Launch:** `torchrun --standalone --nproc_per_node=8 experiments/18_throughput_levers/runner_exp18.py --config ... --data-path ... --sp-model-path ...`

**Conditions:** bs=1024 per GPU × seq=512 × 8 ranks. Run 200 steps (long enough for stable timing, short enough to be cheap).

**Metric:** actual tokens/sec vs 8× single-GPU extrapolation. Gradient variance at global batch 8192 vs single-GPU bs=1024.

**Gate:** DDP efficiency ≥85% of linear (≥670K per GPU vs 763K single-GPU ideal). Below: debug all-reduce overhead or gradient compression.

**Budget:** ~20 min on grant-funded 8×H100.

**Branches used:** `feat/ddp-integration`. DDP wrap, rank-aware seed + data sharding, rank-0 guards on prints and file writes, `dist.barrier()` around eval, `verify_diag_recurrence` per-rank warmup. Single-device path preserved bit-identically at `world_size=1` (load-bearing regression test passes).

**Known limitation guarded:** `metabolic_gate=True` with `metabolic_mode ∈ {fork, mcts}` is explicitly blocked at function entry because those paths take the raw model and bypass DDP's gradient all-reduce hook. Fails fast with `NotImplementedError`. Monte_carlo mode verified DDP-safe (under `torch.no_grad()`, main forward still goes through `ddp_model`).

## Test 5: LR stability screen at DDP scale

**Conditions:** at global batch 8192, screen three LRs — linear-scaled from single-GPU (0.064), linear/2, linear/4. 200 steps each.

**Gate:** at least one LR stable (no NaN, no divergence). Below: extend warmup or reduce grad clip.

**Budget:** ~30 min on 8×H100.

## Test 6: Sequence length sweep

**Conditions:** at winning (batch, LR) from Tests 4-5, vary `seq_len ∈ {512, 1024, 2048}` at fixed (batch, VRAM) envelope.

**Key unknown:** does step time grow linearly or sub-linearly with seq_len on the chunked backend? The scan is theoretically O(N) but kernel launch overhead and fp64 cumprod allocation can shift the real curve.

**Metric:** tok/s, step time, per-token loss at fixed step count.

**Gate:** pick seq_len maximizing `tok/s × learning-rate-of-loss` (ad-hoc metric: `(bpb_0 - bpb_200) / wall_time`).

**Budget:** ~1 hour on 8×H100.

## Test 7: Optimizer ablation — 3-way AdamW / Muon / LAMB

**Hypothesis:** at global batch 8192+, a large-batch-aware optimizer beats AdamW on learning-per-token for SSMs.

**Conditions:**
- **AdamW** — current baseline.
- **Muon** — Newton-Schulz orthogonalization, competition-proven on transformers. SSM behavior untested; this test measures whether the trick transfers.
- **LAMB** — per-layer trust ratio, first-principles large-batch choice, architecture-agnostic.

**Why 3-way, not Muon-only.** Muon is the leaderboard choice but is borrowed from transformers. LAMB is the first-principles choice for the large-batch regime regardless of architecture. Running both against AdamW is the scientifically honest version of "does optimizer matter here?" — falsifies transformer-borrow reasoning if Muon underperforms, and gives us a fallback if LAMB wins.

**Metric:** loss curve vs steps AND vs wall time. Wall-time curve is what matters (bpb per 600s is the target).

**Gate:** any alternative must show visibly faster loss decrease vs wall time to earn inclusion in Test 9. Otherwise AdamW.

**Budget:** ~1 hour on 8×H100 (3 optimizers × 20 min each).

**Branches used:** `feat/muon-optimizer` and `feat/lamb-optimizer`.

## Test 8: Kernel fusion for SSM block hot path

**Hypothesis:** fusing RMSNorm + residual + FF + residual around the chunked scan saves 5-15% of block latency at small batch, less at large batch. The fusion is launch-overhead reduction, not algorithmic acceleration.

**Conditions:** unfused baseline vs `FusedChaosSSMBlock` at bs ∈ {32, 128, 512}, seq=512.

**Gate:** ≥5% tok/s improvement at any batch config, AND gradient parity to float32 noise at bs=128, 30 steps. Below 5% or any parity drift: park the branch and use unfused.

**Honest caveat from the implementing subagent:** *"this is kernel-launch bookkeeping, not algorithmic acceleration. If the H100 benchmark shows <5% on any batch config, this is worth deprioritizing relative to higher-leverage throughput work."*

**Budget:** ~30 min on single grant-pod GPU.

**Branches used:** `feat/block-kernel-fusion`. Benchmark script at `benchmarks/bench_fused_block.py` ready to run.

## Test 9: Combined integration run

**Run:** full 600s training at winning `(kernel + tokenizer + DDP + optimizer + seq_len + checkpointing + fusion)` configuration. Measure final tok/s, total tokens seen, and bpb on a small held-out slice.

**Deliverable:** "what would the submission look like if we trained it today at peak throughput?" This is the baseline Exp 19 starts from.

**Budget:** ~15 min on 8×H100 (600s training + eval overhead).

## Test 10 (optional): fp8 exploration

Time-boxed to 2 hours. Enable fp8 mixed precision, attempt same config as Test 9 winner. Watch for divergence.

**Gate:** if divergence or corruption in first 200 steps, shut down. No pod time on fp8 debugging. Defer post-deadline if needed.

**Budget:** ~2h on 8×H100, ~$48.

## Runner preflight audit (2026-04-13)

Short audit of `training.py` + `runner_exp17.py` + `data.py` for silent corpus leaks under the 600s budget. **Bottom line: no significant corpus leaks on the bare SSM winning path.**

- **Eval is out-of-budget.** ✓ `runner_exp17.py:398` calls `evaluate_bpb_sp` AFTER `train_chaoscontrol_for_budget` returns.
- **Warmup is pre-training.** ✓ `verify_diag_recurrence(device)` called at `runner_exp17.py:343` before the training timer.
- **No padding in the training loader.** ✓ `batch_from_starts` slices fixed-length windows; every position carries signal.
- **Metabolic / Wernicke / outer_model / CFR / sleep / polyphasic bookkeeping** all gated on features disabled in the bare SSM winning config.
- **Final `maybe_sync_cuda`.** ✓ Runs at loop exit for accurate time accounting.

Minor leaks worth noting, none blocking Test 2 launch:

1. Per-step `float(ce_loss.detach().cpu())` at `training.py:331` forces a GPU→CPU sync every step. ~0.025-0.1% overhead over 14K steps. Could be batched to sync at `spectral_log_interval`.
2. No async data prefetch in `batch_from_starts`. Negligible at current scale; may become measurable at 8×H100 DDP with larger batches (fix: pinned memory + `non_blocking=True`).
3. Partial first-step warmup — only the scan kernel is warmed. RMSNorm/FF/out_proj lazy-init on first step. ~100ms saved with a full forward+backward dummy pass; 0.02%.

## Total budget estimate

| Test | Hardware | Duration | Cost |
|---|---|---|---|
| Test 1 — chunked scan | 1 GPU | **complete** | — |
| Test 2 — tokenizer | 1 spot | 4h | ~$8 |
| Test 3 — activation ckpt | 1 spot | 1h | ~$2 |
| Test 4 — DDP 8× | 8 grant | 20min | ~$8 |
| Test 5 — LR screen | 8 grant | 30min | ~$12 |
| Test 6 — seq_len | 8 grant | 1h | ~$24 |
| Test 7 — optimizer 3-way | 8 grant | 1h | ~$27 |
| Test 8 — fusion | 1 grant | 30min | ~$3 |
| Test 9 — integration | 8 grant | 15min | ~$6 |
| Test 10 — fp8 (optional) | 8 grant | 2h | ~$48 |
| **Total (no fp8)** | | **~8h** | **~$90** |
| **Total (with fp8)** | | **~10h** | **~$138** |

Tests 2, 3, 8 run on cheap single-GPU spot (~$13 total). Tests 4-7, 9, 10 gate on grant-funded pod availability.

## Go/no-go gates summary

| Test | Gate |
|---|---|
| 1 chunked scan | ✓ passed (9-50× over threshold) |
| 2 tokenizer | SP16384 beats SP8192 at p<0.05 paired, or SP8192 stays |
| 3 checkpoint | ≥1.3× tok/s at larger batch, or park |
| 4 DDP 8× | ≥85% scaling efficiency, or debug all-reduce |
| 5 LR screen | at least one stable LR, or extend warmup |
| 6 seq_len | pick winner on tok/s × loss-rate |
| 7 optimizer | alternative must beat AdamW on wall-time curve |
| 8 fusion | ≥5% at any batch config, or park |
| 9 integration | real peak tok/s + baseline bpb for Exp 19 |
| 10 fp8 | no divergence in 200 steps, or shut down |

## What Exp 19 inherits

1. A validated peak-throughput training config `(backend, tokenizer, batch, seq_len, LR, optimizer, DDP, fusion)`.
2. A baseline bpb from Test 9 — this is Exp 19's starting point, replacing the stale Exp 17 single-GPU 1.6277 reference.
3. Infrastructure already lifted and tested: Muon, LAMB, DDP wrapper, fused block, chunked scan.
4. Known limitations: metabolic fork/mcts + DDP guard, kernel fusion parity dependencies on `core.FeedForward` / `RMSNorm`.

## Pending main-session work (post-Codex review)

- Update `memory/project_experiment_plan_2026-04-12.md` with current status (Test 1 complete, Phase 1 branches ready).
- Merge strategy for the 7 feature branches onto `exp-17-local-attn-sidecar` or a new `exp-18-integration` branch (after Codex review lands).
- Per-token loss stratification diagnostic (task #14) — runs between Test 2 and Exp 19 Phase 3.
- Individual Test 2 and Test 3 pre-registration design docs (tasks #6 and #8) — small, run after Codex approves the overall plan.
