# Experiment 19: SSM vs Attention Under Equal Infrastructure

**Date:** 2026-04-11
**Status:** Design approved, blocked on Exp 17/18 results
**Competition deadline:** 2026-04-30 (19 days)
**Branch:** TBD (will fork from main after Exp 17/18 merge)

## Thesis

Given equivalent modern infrastructure (Muon optimizer, 8×H100 DDP, depth
recurrence, GPTQ quantization, SP8192), an SSM backbone converts wall-clock
time to learning signal more efficiently than attention. The throughput
advantage is sufficient to close the per-step quality gap.

**Core claim:** Transformers leave FLOPS on the table. Attention's O(n²) cost
is routing compute, not learning compute. An SSM replaces it with an O(n) scan
whose cost becomes negligible at large batch sizes. On 8×H100, this means every
matmul cycle produces gradient signal — more steps, more tokens, more learning
in the same 600s.

## Why parameter-golf is the right proving ground

The competition controls for everything: fixed hardware (8×H100), fixed time
(600s), fixed artifact size (16MB), public leaderboard, reproducible by anyone.
A result here isn't a benchmark game — it's a hardware-controlled experiment
with a public audit trail.

## Method: One codebase, one variable

Fork the SOTA `train_gpt.py` (~2135 lines, producing 1.075 bpb). It already
includes: Muon optimizer (momentum 0.97), 8×H100 DDP, depth recurrence (layers
3-5 shared → 14 virtual), parallel residuals (GPT-J split lanes), GPTQ int6
quantization + zstd, SP8192, legal TTT (score-before-update), flash attention
3, full data pipeline.

Define a `Block` interface. Provide two implementations:

- **AttentionBlock** — existing flash-attention path (control)
- **SSMBlock** — SSM recurrence, same projections, same residual structure

Everything else stays identical. The only variable is what happens inside the
block.

## Metrics

For each variant, across 3+ seeds:

| Metric | Purpose |
|--------|---------|
| steps/s | Raw throughput — core thesis |
| tokens/s | Effective data coverage |
| bpb at step N | Per-step quality (learning efficiency) |
| bpb at wall-clock T | The metric that matters |
| peak VRAM per GPU | Are we using what's available? |
| FLOP utilization % | Direct measure of the thesis |

**The key deliverable** is a bpb-vs-wall-clock plot. If SSM's curve crosses
attention's curve (starts behind but overtakes), the thesis holds. If not, we
know cleanly.

## SSM Variants

1. **Diagonal scan** (ChaosControl core) — fastest possible, minimal per-step
   overhead. torch.compile'd loop, proven at 1.63 bpb on single-GPU AdamW.
2. **Gated DeltaNet** — delta-rule linear recurrence, proven at 1.209 bpb in
   competition (GDN-Hybrid PR #1553, but with SP1024 and no SOTA techniques).
3. **Mamba2 SSD** (stretch goal) — the household name, structured state-space
   duality algorithm.

Testing multiple variants separates "SSM architecture matters" from
"infrastructure matters."

## Ablations

Each SOTA technique was designed for attention. Do they transfer to SSMs?

- **Depth recurrence** — does sharing SSM layers (like transformer layers 3-5)
  help? SSM state is persistent across steps; sharing changes what that means.
- **Parallel residuals** — do split lanes work when the "attention" lane is SSM?
- **Muon on SSM params** — moved to Exp 18 as a throughput lever (learning
  efficiency per token). Exp 19 inherits whichever optimizer wins Exp 18.

**Test-time training (TTT) is out of scope for Exp 19.** It's been moved to
Exp 20 as its own experiment. Rationale: the SSM's internal state already
evolves during eval, so "apply TTT on frozen weights" isn't a clean drop-in
like it is for transformers. The substrate question — what space do you
build the cache over when the hidden state is itself a cumulative
running memory — is a real research question, not a switch to flip. See
`2026-04-12-experiment-20-ssm-test-time-training-stub.md` for the open
questions. Exp 19 ships a submission with **no** TTT; Exp 20 is the
follow-up lever if we still need more bpb before April 30.

## Competition context

| Entry | Architecture | bpb | Key techniques |
|-------|-------------|-----|----------------|
| SOTA (PR #1529) | Transformer | 1.075 | Full recipe |
| GDN-Hybrid (#1553) | DeltaNet+SWA | 1.209 | SP1024, no Muon, no TTT |
| ChaosControl Exp 16 | Diagonal SSM | 1.630 | SP8192, AdamW, 1 GPU |

**Gap analysis:** The 0.55 bpb gap between our SSM and SOTA is explained by:
- ~0.13 bpb: Architecture (SSM vs attention per-step quality)
- ~0.20 bpb: Optimizer (AdamW vs Muon = 2× efficiency)
- ~0.15 bpb: Compute (1 GPU vs 8×H100 DDP = 8× throughput)
- ~0.07 bpb: Quantization (no GPTQ vs int6 = 2× effective params in 16MB)

If infrastructure accounts for ~0.42 bpb and architecture only ~0.13, the
thesis is viable.

## Prerequisites from Exp 17/18

- **Exp 17** (complete, 2026-04-12): local attention and top-k retrieval
  variants all fail vs bare SSM. `topk_random_8` ties bare (p=0.71), all
  other retrieval variants are significantly worse. Score-based selection
  is actively adversarial (~0.065 bpb worse than random at matched k).
  No signal for including attention in Exp 19 unless the research assistant
  lit search surfaces a specific variant we haven't tested.
- **Exp 18** (redesigned 2026-04-12, see
  `2026-04-12-experiment-18-throughput-levers-design.md`): replaces the
  old "sweep → rescore → targeted retrain" framing with a throughput
  lever stack (mamba scan kernel, DDP validation, seq_len sweep,
  optimizer swap). Exp 18 will hand off a validated peak throughput
  config (kernel, optimizer, batch, seq_len, DDP world size) plus a
  baseline bpb at that throughput. Exp 19 inherits that config as its
  starting point and adds architecture / quantization / submission dials
  on top.

## Follow-up hypothesis: pre-SSM KV source

If Exp 17 topk conditions show topk ≈ topk_random (selection isn't doing real
work), or if topk is comparable to bare_fast_ssm, one compelling explanation
is that the post-SSM features have been too homogenized for the K projection
to distinguish tokens by identity. The SSM blends past information into a
compressed state; once a token passes through the SSM, its K projection
reflects "current blended state at this timestep" rather than "identity of
this original token."

**Hypothesis:** Source K/V from the **pre-SSM residual stream** (the block's
input embedding or mid-stack activations before SSM mixing) instead of
post-SSM features. This gives the attention path un-blended token
representations to retrieve by identity, while the SSM continues to handle
smooth contextual mixing on its own path. The two mechanisms then serve
complementary roles:
- SSM: compressed summary of full context (smooth, positional, blended)
- Attention: exact retrieval of specific high-information past tokens (sharp,
  identity-preserving)

This is a natural variant to test in Exp 19 if the Exp 17 topk results
suggest the selection mechanism is fighting against feature homogenization.
Cost: one new hybrid variant, same infrastructure as the existing forward
path.

**What would falsify it:** If pre-SSM KV also shows no improvement, the
problem isn't feature homogenization — it's that the SSM simply doesn't
benefit from exact token retrieval at this scale.

## Timeline (19 days)

| Phase | Days | Deliverable |
|-------|------|-------------|
| Run Exp 17+18 | 2-3 | Throughput data, local-attn results |
| Phase 0: Fork + scaffold | 2 | train_ssm.py, Block interface, reproduce SOTA bpb |
| Phase 1: Diagonal SSM block | 2 | First SSM bpb + throughput on 8×H100 |
| Phase 2: Key measurement | 1 | bpb-vs-wall-clock plot (THE result) |
| Phase 3: Depth recurrence ablations | 3 | Transfer experiments |
| Phase 4: GDN variant | 2 | Second SSM variant, same infra |
| Phase 5: Quantization + submission | 2 | GPTQ for SSM, artifact <16MB |
| Buffer | 5 | Bugs, additional ablations, writeup |

## Go/no-go gates

- **After Phase 0:** Can we reproduce SOTA bpb (±0.005) with the forked code?
  If not, our fork is broken and results are meaningless.
- **After Phase 1:** Does SSM get >20% more steps/s than attention? If not, the
  throughput thesis is falsified at this scale and we pivot to architecture work.
- **After Phase 2:** Does the bpb-vs-wall-clock curve cross? If SSM never
  overtakes attention, the thesis is falsified. Publish as a negative result
  (still valuable).

## Risk: What if the thesis is wrong?

A clean negative result — "SSMs don't close the gap even with equal
infrastructure, here's exactly why" — is still a notable contribution. The
controlled methodology makes it publishable either way. Mamba earned its name
by showing SSMs *could* compete; if we show they *can't* under this specific
regime, that's equally informative for the field.

## Artifact

The final submission (if competitive) would be:
- `train_ssm.py` — the forked SOTA with SSM blocks
- Compressed model weights (int6 GPTQ + zstd, <16MB)
- `submission.json` with 3-seed results
