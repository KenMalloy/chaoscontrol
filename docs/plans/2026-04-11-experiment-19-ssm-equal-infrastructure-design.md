# Experiment 19: SSM vs Attention Under Equal Infrastructure

**Date:** 2026-04-11 (rewritten 2026-04-12 after code review)
**Status:** Design approved, blocked on Exp 18 completion
**Competition deadline:** 2026-04-30
**Branch:** `exp-19-ssm-submission` (forked from main after Exp 18 merge)

## Note on the 2026-04-12 rewrite

The original design said "fork the SOTA train_gpt.py (~2135 lines, producing
1.075 bpb)" and treated it as the starting point. Code review caught five
errors:

1. **The 1.075 bpb number is from GitHub PR #1529, not from anything in this
   repo.** Our local `baselines/parameter_golf/sota/` is a different record
   at **1.1147 bpb**, with **no TTT** (explicitly dropped), and with **no
   depth recurrence** in its architecture table. The local baseline
   `baselines/parameter_golf/train_gpt.py` is explicitly a starting-off point
   targeting ~1.2 bpb, not SOTA.
2. **Forking train_gpt.py breaks the Exp 18 handoff.** Exp 18 validates a
   ChaosControl-based stack (mamba kernel, Muon, DDP, seq_len, our runner).
   Forking a different codebase for Exp 19 means re-validating that stack
   from scratch in a different tree. The "inherit Exp 18's winners" claim
   is incompatible with "start from train_gpt.py."
3. **The gap analysis was speculative.** Decomposing the 0.55 bpb gap
   additively into architecture + optimizer + compute + quantization treats
   interacting interventions as independent. Quantization is a post-training
   step (pre-quant vs sliding BPB delta is ~0.02 in the sota README), so
   treating ~0.07 bpb as a "training-time gap component" was wrong.
4. **The Exp 17 summary was overstated.** Only score-based topk was
   significantly worse than bare. `topk_random_8` tied bare (p=0.706),
   `topk_random_16` and `local_w32` were worse in mean but not significant
   at p<0.05.
5. **Artifact naming violated competition rules.** The rules require
   submissions to name their entry script `train_gpt.py` (README line 216).
   The original plan called it `train_ssm.py`.

This rewrite commits to **Option B: extend ChaosControl with lifted
components** rather than forking `train_gpt.py`. Rationale for the choice
is in the Method section below.

## Role in the experiment sequence

- **Exp 18** (throughput tuning): validates a ChaosControl-based SSM training
  stack at peak throughput. Delivers (kernel, optimizer, batch, seq_len,
  LR, DDP config) as a handoff, plus a baseline bpb at that throughput.
- **Exp 19** (submission tuning, this doc): takes Exp 18's validated stack
  and dials architecture, quantization, and packaging for the final
  submission.
- **Exp 20** (SSM-specific TTT, stub): only runs if Exp 19 lands close but
  not winning.

## Thesis

**A pure SSM trained with modern competition-tier infrastructure (Muon,
DDP, depth recurrence, GPTQ) can be a competitive Parameter Golf
submission.** We don't know how close to the transformer leaderboard this
gets us — the published gap analysis is unreliable because interventions
interact, and the local sota record (1.1147) is both older than the current
leader and uses a different ingredient mix. What we *do* know:

- Our Exp 16 bare SSM baseline was 1.63 bpb at single-GPU AdamW at bs=32.
- Exp 18's planned levers (mamba kernel, Muon, DDP, seq_len, optional fp8)
  each have independently-published efficiency multipliers, and Exp 18 Test 7
  gives us a stacked throughput baseline before Exp 19 starts.
- Exp 19 adds architecture capacity (depth recurrence, dim/layers search)
  and post-training quantization (GPTQ + compression) on top of that.

The honest framing: **Exp 18 produces our training ceiling. Exp 19 tries to
close whatever gap remains under the 16MB constraint.** We'll discover
how big the gap is when Exp 18 finishes, not by speculating now.

## Why the comparison with attention still matters

Even though we won't fork train_gpt.py, the scientific value of an
SSM-vs-attention comparison *inside our own stack* is real. If we give a
ChaosControl-based attention implementation the same Muon/DDP/seq_len/
quantization treatment we give the SSM, we can measure the per-token
learning-efficiency difference under exactly-matched everything-else. That's
a cleaner comparison than comparing our SSM to a different team's
transformer — no codebase differences, no hidden optimizer tricks, no
data pipeline variation.

This is the "equal infrastructure comparison" that makes the experiment
publishable regardless of whether we win the competition. Exp 19 ships
with both blocks as a scientific deliverable, even if only the SSM goes
into the final submission package.

## Method: extend ChaosControl, don't fork

**What we lift from `baselines/parameter_golf/sota/train_gpt.py`:**

- **Muon optimizer.** ~100 lines, self-contained. Uses Newton-Schulz matrix
  orthogonalization. Drop into `src/chaoscontrol/optimizers/muon.py` as a
  separate module. Add a `--optimizer` flag to the runner.
- **GPTQ int6 quantization pipeline.** ~200 lines of post-training
  quantization. Runs after training is complete, applies to all Linear
  weights. Drop into `src/chaoscontrol/quantization/gptq.py`.
- **LZMA compression + zstd path.** ~20 lines. Final artifact packaging.
- **DDP wrapper patterns.** Lift the `torch.nn.parallel.DistributedDataParallel`
  setup from the sota script as a reference; adapt for our runner's
  per-condition subprocess model.

**What we implement ourselves in ChaosControl:**

- **Depth recurrence.** Weight-tied SSM layers that run their forward pass
  multiple times. ~30 lines in `src/chaoscontrol/model.py`. Parameter
  count stays fixed; compute per token grows linearly with the recurrence
  count. This is a capacity lever, not a throughput lever.
- **AttentionBlock class.** Sits alongside `ChaosSSMBlock` and
  `ChaosSSMHybridBlock` in `src/chaoscontrol/model.py`. Takes the same
  block interface, same residual shape, same projections. Uses
  `torch.nn.functional.scaled_dot_product_attention` (no Flash Attention 3
  dependency — H100 has native support via PyTorch SDPA). This is the
  control arm for the SSM vs attention comparison.
- **Submission entry point.** Named `train_gpt.py` (rules compliance),
  placed at the root of whatever submission folder we generate. It
  imports from `src/chaoscontrol/` so the "counted code" bytes include
  the package sources.

**What we do NOT lift:**

- BigramHash, XSA, SmearGate, VE128, U-Net skips, partial RoPE — all
  transformer-specific ingredients from the sota record. We can revisit
  if the SSM+lifted-stack baseline is close-but-not-winning, but none of
  these are relevant to SSM architectures as primary levers.
- Flash Attention 3. Transformer-only concern. Our AttentionBlock uses
  PyTorch's SDPA which is fast enough at our scale and has no external
  dependency.
- The sota script's transformer-specific forward loop. We keep our own
  training loop (extended with Exp 18 validated components).

## Phase 0: infrastructure validation (NOT "reproduce SOTA bpb")

The reproducibility gate for Exp 19 Phase 0 is **"reproduce Exp 18 Test 7's
stacked throughput baseline within ±2% on tok/s and ±0.01 bpb on the
initial training run."** Not "reproduce 1.075 bpb" — we have no such target
locally. If Exp 18 Test 7 gives us 1.45 bpb at 5M tok/s (for example),
Phase 0 of Exp 19 has to reproduce those numbers *after* the refactoring
(lifted Muon, new file layout, new entry point) before any architectural
changes are made.

Gate: if Phase 0 of Exp 19 can't reproduce Exp 18's baseline, something
broke in the refactor. Stop and debug before going further.

## Phase 1: lifted infrastructure bringup

1. Lift Muon from sota train_gpt.py into `src/chaoscontrol/optimizers/muon.py`.
   Add unit tests matching the sota tests. Add `--optimizer` runner flag.
2. Lift GPTQ pipeline into `src/chaoscontrol/quantization/gptq.py`.
   Post-training-only. Needs a small number of calibration sequences from
   the model itself (the sota record uses AR self-generated calibration,
   which is legal under Issue #1017).
3. Implement depth recurrence in `src/chaoscontrol/model.py` as a config
   flag (`depth_recurrence_shared_layers: [int]` — list of layer indices
   that share weights, and `depth_recurrence_count: int` — how many times
   to run the shared group).
4. Implement `ChaosAttentionBlock` alongside existing SSM blocks, using
   `F.scaled_dot_product_attention`.
5. Write a `train_gpt.py` entry point at `experiments/19_ssm_submission/`
   that invokes the training loop with Exp 18's validated config. This
   becomes the artifact.

**Gate:** all pieces pass unit tests, Phase 0 reproducibility check holds.

## Phase 2: SSM vs Attention controlled comparison

Run both `ChaosSSMBlock` and `ChaosAttentionBlock` through the same
training stack (Exp 18 validated config: kernel, Muon, DDP, seq_len, LR,
budget). 3 seeds each. Single variable: block type.

**Metrics:**
- Steps/sec (throughput at fixed VRAM)
- bpb at 600s wall clock (the metric that matters)
- bpb vs wall-clock curve (the scientific plot)
- Peak VRAM
- Per-condition gate stats for Attention (same instrumentation as Exp 17)

**What this tells us:**
- How much per-step quality the transformer gets in our stack
- How much throughput the SSM gets
- Whether the product (tokens × learning efficiency) favors either
- Whether the SSM thesis "use throughput to close the quality gap" holds
  empirically *inside our own codebase*

## Phase 3: architecture search on the SSM side only

Once Phase 2 establishes the per-block bpb numbers, commit to whichever
block wins. From here, everything is submission tuning on the winning
architecture.

Dials to sweep:
- `num_layers` ∈ {6, 8, 10, 12, 14} at matched artifact size
- `model_dim` ∈ {192, 256, 320, 384} at matched artifact size
- `ff_mult` ∈ {2, 3, 4}
- `depth_recurrence_count` ∈ {1, 2, 3} on a subset of layers
- `vocab_size` ∈ {SP8192, SP16384}

Budget: small N per config (2 seeds), many configs, short eval. Pick the
winner on bpb × artifact size × stability.

## Phase 4: quantization and artifact packaging

Take the winning architecture. Run full 3-seed training. Apply GPTQ int6 +
LZMA compression. Measure pre-quant BPB, post-quant BPB (sliding), and
artifact size. Iterate on calibration seq count and compression level
until artifact fits under 16MB.

**Gate:** 3-seed mean post-quant BPB beats the current local SOTA record
(1.1147 BPB from `baselines/parameter_golf/sota/`) with p<0.01 over 3
seeds, per the competition's submission criteria. If it doesn't, we ship
a non-record "architecturally novel" PR and start Exp 20 (TTT).

## Phase 5: final submission

Package according to competition rules:
- `train_gpt.py` (our entry point, imports chaoscontrol package)
- `README.md` describing the submission
- `submission.json` with metadata
- Train logs from the 3 seeds
- Any other dependencies (our chaoscontrol package bundled in)

Submit PR to `openai/parameter-golf/records/`.

## Timeline (18 days to April 30)

| Phase | Days | Deliverable |
|---|---|---|
| Exp 18 runs to completion | 1-2 | Validated throughput stack, handoff config |
| Phase 0: refactor + reproduce | 1 | Exp 18 config reproducible after refactor |
| Phase 1: infra bringup | 2 | Muon, GPTQ, depth recurrence, AttentionBlock, entry point |
| Phase 2: SSM vs Attention | 2 | The bpb-vs-wall-clock plot, controlled comparison |
| Phase 3: architecture search | 3 | Winning (dim, layers, ff, vocab, depth_rec) config |
| Phase 4: quantization + packaging | 2 | Submission-ready artifact |
| Phase 5: final 3-seed run + submit | 1 | PR to records folder |
| Buffer for bugs and redos | 4 | — |

## Risks and mitigations

- **Muon lift introduces numerical differences from the reference.** Mitigation:
  unit tests against published Muon reference values + training stability
  check on a small model before committing to Exp 19 training.
- **GPTQ calibration without val data is fragile.** The sota record uses AR
  self-generated calibration (legal and compliant). Implement that exact
  approach — not a simpler heuristic that might violate Issue #1017.
- **Depth recurrence on SSMs is untested.** Literature on depth recurrence is
  transformer-specific. Exp 19 Phase 3 will be the first test of SSM depth
  recurrence — if it doesn't help, we still have the flat-depth SSM fallback.
- **AttentionBlock might beat SSMBlock on bpb at our scale.** That's a valid
  scientific outcome (means SSMs are genuinely worse, not just infrastructurally
  under-equipped). We still ship the SSM as the submission if it's close,
  and write up "attention wins in our controlled comparison" as the honest
  finding.

## Prerequisites from Exp 17/18

- **Exp 17** (complete, 2026-04-12): 70 runs across 10 conditions.
  `topk_random_8` ties bare (p=0.706). Score-based topk is significantly
  worse than bare and significantly worse than random-topk at matched k,
  a clean negative result on trainable retrieval paths. Random-topk at
  k>8 is worse in mean but not always significant at p<0.05. Local-window
  conditions: local_w16 significantly worse (p=0.005), local_w32 worse in
  mean but not significant (p=0.108), local_w64 significantly worse
  (p=0.034). Mixed significance pattern; the clean conclusion is "no
  attention variant beats bare, score-based selection is actively
  adversarial, and more attention bandwidth doesn't help." The lit review
  (2026-04-12 deep research report) agrees and notes that published
  hybrid wins require 20B-250B training tokens, 40-500× our available
  budget.
- **Exp 18** (in progress): delivers validated peak throughput config and
  a baseline bpb. Exp 19 starts from that config and adds architecture/
  quantization on top.

## Follow-up: SSM-specific TTT (Exp 20)

Moved out of Exp 19 scope per 2026-04-12 review. SSM state is not static
at eval time — the recurrence state evolves every token — so "apply TTT
on a frozen model" is not a clean drop-in like it is for transformers.
The substrate question (what space do you build the cache over when the
hidden state is itself a cumulative running memory?) deserves its own
experiment. See `2026-04-12-experiment-20-ssm-test-time-training-stub.md`
for the open questions. Exp 20 only runs if Exp 19 lands close-but-not-
winning and we still have time/budget before April 30.

## What we ship

Even in the worst case (SSM doesn't win the competition), Exp 19 produces:
1. The first publicly-documented controlled SSM-vs-attention comparison
   under matched training infrastructure at Parameter Golf scale.
2. A working depth-recurrent SSM implementation (first in the competition).
3. A submission to the `records/` folder that either wins or is a notable
   non-record architecturally-distinct entry.
4. A writeup that uses the Exp 17 (trainable retrieval fails), Exp 18
   (throughput levers), and Exp 19 (architecture under matched infra)
   results as a coherent scientific story.
