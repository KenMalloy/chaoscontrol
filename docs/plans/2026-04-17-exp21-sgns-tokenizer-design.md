# Exp 21 — Semantic Tokenizer via Offline SGNS Embedding Init

**Date:** 2026-04-17
**Status:** Design — pending approval
**Target:** Parameter Golf (FineWeb eval, 16 MB decimal artifact, 600 s train on 8×H100)

## Motivation

Long-standing bet against consensus: a custom tokenizer that embeds semantic meaning can materially help a *raw* SSM. Prior attempts never cleanly isolated the tokenizer variable — they bundled training-time or eval-time machinery (bucket routing, episodic memory, retrieval) that has been falsified across Exp 09/14/17 and ruled out by the SSM-native thesis.

Exp 21 isolates the tokenizer-side variable. Same SP subword tokens as our Exp 18 baseline; only the *initial values* of the embedding table change. Embeddings are pretrained offline via SGNS on FineWeb train, so they carry semantic geometry at step 0 instead of random values.

**Why novel at Parameter Golf:** Four PRs have tried JEPA at Param Golf (#1685 → 1.76 bpb, #1654 → 1.27, #1581 → 1.23, #1556 → 1.44). All used JEPA as the LM *training objective* — the documented latent-MSE-vs-token-CE mismatch cost them ~0.66 bpb in PR #1685 alone. Exp 21 is fully offline: SGNS produces embedding values; the LM still trains with standard token-CE. No objective mismatch, no prior art for this specific approach.

## Hypothesis

Semantic anchor geometry in the embedding table helps raw SSMs specifically, more than it helps transformers, because SSM state compresses tokens through a fixed-size bottleneck (`h_{t+1} = A·h_t + B·embed[x_{t+1}]`) and cannot recover token identity the way attention can.

- **Primary:** SSM with SGNS-init beats SSM with random-init at the **SSM's own LR-tuned optimum** (per-arm Muon LR, see Training protocol).
- **Secondary:** the SSM's gain exceeds the transformer's gain — each arm measured at its own LR-tuned optimum.

Both claims are evaluated *per-architecture*, not under a single fixed LR. The Phase 0 LR sweep produces each arm's best Muon LR; the 4-cell runs compare SGNS-vs-random at those optima. This is the honest version — under a single fixed LR, a transformer handicap could masquerade as SSM-specific benefit.

## Controls: isolating "semantic" from "distributional"

SGNS changes more than neighborhood structure — it also shifts row-norm distribution, anisotropy, and frequency-correlated magnitude. A naïve SGNS-vs-random comparison conflates "semantics helped" with "optimizer conditioning improved by a better-scaled init." Two controls:

1. **Moment-matching before load (mandatory — both flavors run).** Before copying SGNS vectors into `model.embed.weight`, rescale so the embedding table matches the baseline's random-init (Gaussian) statistics. Two flavors, both evaluated:
   - **Mean+std match (primary):** rescale each row so row-norm mean and std match the baseline. Matches what standard Gaussian init assumes.
   - **Full row-covariance match (sanity check):** whiten SGNS then re-color to match the baseline's full row covariance. Stricter; more invasive.
   
   If the two flavors give the same SGNS-vs-random delta within noise, the result is robust to choice of matching scheme. If they diverge, the thesis depends on the specific scheme and warrants follow-up. Adds one SGNS-SSM cell variant (5 extra seeds, ≈ 0.85 GPU-hours).

2. **Shuffled-row SGNS control (1-seed sanity check).** Take the moment-matched SGNS table, shuffle rows randomly, load as init for the SSM arm. Preserves marginal distribution but destroys semantic-ID mapping. If shuffled-SGNS bpb matches real SGNS bpb within noise, the effect is distributional — the thesis falls. One seed on the SSM arm is cheap insurance.

Without both controls, a positive primary result is ambiguous.

## Scope

**In scope:**
- Offline SGNS training on FineWeb train
- Per-subword embedding extraction
- Vocabulary choice (V) + quantization
- Artifact packaging
- 4-cell validation ablation (SSM × transformer × {random, SGNS})

**Out of scope (deferred):**
- Any mutation of the embeddings during the LM's 600 s training (freeze schedules, separate LR profiles) — future post-Exp-19 session
- Eval-time TTT adaptation of embeddings — Exp 20 scope
- Alternative offline SSL objectives (CBOW, tiny-LM CE pretraining, actual JEPA with encoder) — candidates only if SGNS wins cleanly

## Dependencies (assumed from Exp 19 or pre-work)

Exp 21 relies on machinery not in the repo snapshot today:

1. **SP tokenization integrated with the training data path.** Current repo exposes raw-byte data plus a `fixed_stride` tokenizer config (`src/chaoscontrol/config.py:127`, `src/chaoscontrol/runner.py:175`, `src/chaoscontrol/data.py:223`). Exp 21 assumes an SP8192 tokenization path is available — expected from Exp 19 work.
2. **Embedding init-from-file hook.** No current code loads `model.embed.weight` from a saved tensor. Net-new to Exp 21. Insertion point: `src/chaoscontrol/runner.py:build_model` after `model = model.to(device)`; new config field `embed_init_path: str | None = None` in `src/chaoscontrol/config.py`.
3. **Specific modded-NanoGPT variant in the transformer arm.** The runner harness already dispatches on `model_type == "transformer"` (`src/chaoscontrol/runner.py:29`) and a baseline transformer exists at `src/chaoscontrol/baselines.py:9`. What is missing is the specific lean modded-NanoGPT variant documented below (RoPE, RMSNorm, ReLU², Flash Attention + QK-norm, no auxiliary embeddings). Either adapt the existing baseline or register a new variant — not a harness question.

If any prerequisite is absent at execution time, Exp 21 is blocked — verify before Phase 0.

## Components

### SGNS objective

Standard skip-gram with negative sampling.

- **Corpus:** FineWeb train, tokenized with chosen SP vocabulary.
- **Loss:** for center subword `x_c` and true context `x_ctx` in window, maximize `σ(W_in[x_c] · W_out[x_ctx])`; for `k` sampled negatives `x_neg`, minimize `σ(W_in[x_c] · W_out[x_neg])`.
- **Hyperparameters (defaults):**
  - Window size: 5
  - Negatives per positive: 10
  - Subsampling threshold (frequent-word downsampling): 1e-5
  - Embedding dim `d = 256` (matched to SSM `d_model`, drop-in replaceable)
  - Epochs over FineWeb train: 3 (adjust based on loss convergence)
- **Output:** `W_in ∈ R^{V × 256}` per-subword input-embedding table. `W_out` is discarded after training.
- **Precedent:** SP tokenizer training itself is offline training on FineWeb train — SGNS at this scale is kind-compatible. Disclosed in the submission PR.

### Vocabulary and quantization

**Thesis-test default: V = 8192, bf16, untied.** Reasons:
- Matches current Param Golf SOTA choice (PR #1698 and neighbors).
- Fits under 16 MB at bf16 untied (~11.5 MB total), so the thesis ablation does not mix in a quantization variable.
- Cleanest param allocation for the SSM arm (~2.3M non-embedding params, matching our Exp 18 baseline).

**Follow-on (post-thesis, not in Exp 21 core):** V = 16384 + int6. Exp 18 Test 2 found SP16384 beats SP8192 by +0.037 bpb at matched bf16 (n=7 seeds, paired t=7.29, p=3.4e-4; per memory `project_exp18_test2_results_2026-04-14.md` — the earlier design doc `docs/plans/2026-04-12-experiment-18-throughput-levers-design.md` cites a pre-run +0.008 estimate, superseded by the actual Test 2 result). V=16384 bf16 untied overflows 16 MB (~21.5 MB total), so ship-ready V=16384 requires int6 on both sides or tying, **plus GPTQ wiring in the artifact path** (not currently in place — see Artifact packaging). Out of Exp 21 scope; gated on a quant-parity pilot (≤ 0.01 bpb loss vs bf16) for the shipping follow-on.

**Excluded:** SP4096 (Exp 15 showed +0.034 bpb penalty), SP1024 (SOTA's tiny-V choice, no prior work in our repo), int4 (no quant infrastructure in our pipeline).

### Tied vs untied embedding/LM-head

**Decision: untied for Exp 21 thesis test.**

Rationale:
- **Isolates the SGNS input-side effect.** Tying makes `model.embed.weight` serve as the LM head's output projection too. A 2025 analysis (*Weight Tying Biases Token Embeddings Towards the Output Space*) shows the shared parameter is pulled toward output-prediction geometry, eroding input-side semantic structure. For a test specifically about semantic anchor geometry at the input, untying preserves the variable we're measuring.
- **Matches our Exp 18 baseline** (`model.embed` and `model.lm_head` separate per `src/chaoscontrol/model.py:554` and `:680`). Apples-to-apples for the SSM arm.
- Trade-off: untied burns more artifact budget. At V=8192 bf16 the untied model fits (~11.5 MB); at V=16384 bf16 untied it overflows 16 MB. Addressed by keeping the thesis test at V=8192.

Tying is a live question for ship-ready configs (Param Golf baseline default is `TIE_EMBEDDINGS=1`; SOTA PRs follow). A post-thesis follow-on experiment will test whether tying erodes SGNS structure empirically — out of scope for Exp 21.

LM head remains randomly initialized across all cells. SGNS values go only into `model.embed.weight`.

### Artifact packaging

- Embedding weights ship inside the 16 MB decimal artifact (bf16 at V=8192 fits untied without quantization).
- Current artifact serializer (`src/chaoscontrol/artifact.py:52`, invoked at `:172`) applies **symmetric int6** per 2-D tensor as a blanket path — **not GPTQ**. `GPTQQuantizer` exists at `src/chaoscontrol/quantization/gptq.py:256` but is **not wired** into the artifact path today. Any V=16384 int6 ship-ready variant requires that wiring; out of scope for Exp 21 thesis test (which stays bf16 at V=8192).
- SGNS training code is offline — does NOT ship. Only the resulting embedding table is part of the artifact.

## Data flow

```
FineWeb train (bytes)
  │
  ▼
SP tokenization (V ∈ {8192, 16384})
  │
  ▼
Offline SGNS training (3 epochs, NCE loss)
  │  output: W_in ∈ R^{V × 256}
  ▼
Per-subword embedding table
  │
  ▼
Quantization (bf16 or int6 GPTQ)
  │
  ▼
Embedding artifact file (.pt)
  │
  ▼
LM `model.embed.weight` init
  (LM head random, SSM random — only the embedding is SGNS)
```

## Validation methodology

### Intrinsic (offline, minutes)

- **SGNS loss convergence:** NCE loss drops substantially from initialization over FineWeb train epochs. Sanity-check against standard SGNS training curves.
- **Nearest-neighbor sanity:** for 50 common subwords, inspect top-5 nearest neighbors by cosine. Expect semantically coherent clusters.
- **Cluster coherence:** k-means (k=20) on the embedding table; inspect centroid neighborhoods. Expect recognizable categories (punctuation, whitespace, common stems, numerics).

### Extrinsic — 4-cell ablation

|  | Random init | SGNS init |
|---|---|---|
| **Transformer** (modded-NanoGPT, ~10.7M params) | `A` | `B` |
| **Raw SSM** (Exp 18 Test 4b config, ~10.7M params) | `C` | `D` |

**Training protocol, matched across cells (with per-arm LR):**
- 600 s wall-clock budget (matches Param Golf submission constraint).
- **Thesis-test runs on 2× H100 (ws = 2)** — our CUDA-13 pod, matches Exp 18 Test 4b protocol. The Param Golf submission regime is 8× H100; ship-ready re-validation at ws=8 happens post-thesis (separate Exp 19 carry-over, not Exp 21 scope). Within-arm A-vs-B *deltas* are the thesis signal and should carry across ws; absolute bpb at ws=2 will differ from ws=8.
- Muon optimizer, bs = 1024/rank × ws = 2, seq = 512, `activation_checkpoint = true`
- Base LR: **per-arm LR-tuned optimum under Muon.** SSM uses LR=0.064 — the Exp 18 Test 5b-tuned optimum for our config (not a fresh sweep). Transformer uses its own optimum from a Phase 0 sweep (1 seed × 3 LRs ∈ {0.016, 0.032, 0.064}, pick the winner). **Phase 0 is committed, not an open question.** Both "per-arm optimum" and "per-architecture optimum" collapse to this framing: each arm measured at its own known-or-sweept-to Muon best.
- V = 8192, bf16, untied (thesis-test default per Vocabulary section)
- Same FineWeb train stream
- Matched seeds across cells (paired by seed index for the statistical tests)

**SSM architecture (cells C, D):** `src/chaoscontrol/model.py` config — d_model=256, n_layer=4, state_dim=256, chunk_size=64, ff_mult=2, a_mode="diag", untied. ~10.75M params at V=8192 untied.

**Transformer architecture (cells A, B):** modded-NanoGPT lean variant at matched params.
- d_model=256, n_head=4 (head_dim=64), n_layer=8, ffn_mult=4
- RoPE positional encoding, RMSNorm, ReLU² activation, Flash Attention + QK-norm
- No auxiliary embeddings (value, bigram-hash — modded-NanoGPT's speedrun-specific tricks omitted for a clean baseline)
- Untied embed/LM-head (matches SSM arm for protocol parity)

Param-count verification at V=8192 untied: embed 2.10M + LM head 2.10M + attn 2.10M + MLP 4.19M ≈ **10.49M** (within 3% of the SSM's 10.75M).

**Seeds:** 5 per cell (total 20 runs × 10 min ≈ 3.5 GPU-hours).

### Statistical tests

Define effect sizes as **improvements** (positive = SGNS better):

- `Δ_SSM = C - D` (SSM random-init bpb minus SSM SGNS-init bpb)
- `Δ_Trans = A - B` (transformer random-init bpb minus transformer SGNS-init bpb)

**Primary:** paired one-sided t-test on SSM bpb across seeds. `H₁: Δ_SSM > 0`. Reject `H₀` at `p < 0.01`.

**Secondary:** paired one-sided t-test on `Δ_SSM − Δ_Trans` across matched seeds. `H₁: Δ_SSM > Δ_Trans`. Reject `H₀` at `p < 0.01`.

**Power note:** at 5 seeds (df=4, one-sided), critical `t ≈ 3.75`. Given prior seed variance ±0.01 bpb, paired SD ≈ 0.014; minimum detectable effect ≈ 0.025 bpb. If Phase 0 pilot reveals paired SD > 0.014, escalate to 7 seeds (df=6, critical `t ≈ 3.14`, MDE ≈ 0.018 bpb).

### Success criteria

- **Thesis-validating:** primary AND secondary reject `H₀` at `p < 0.01`.
- **Thesis-weak:** primary rejects at `p < 0.01`; secondary does not. Interpretation: SGNS helps LMs generically; SSM-specific benefit is unclear.
- **Ship-worthy:** SSM with SGNS-init reaches `bpb < 1.46` (improvement > 0.03 over Exp 18 Test 4b's 1.493 baseline). Independent of the statistical thesis tests.
- **Null:** no significant result at `p < 0.01`. Conclude the bet does not hold in this form; reconsider offline objective (CBOW, tiny-LM) before abandoning the hypothesis.

## Kill criteria

- **SGNS training (intrinsic):** abort if NCE loss flatlines at near-initialization levels after 1 epoch, or if NN sanity check fails broadly (>50% of 50 spot-checked subwords have incoherent neighbors). Retune hyperparameters (LR, window, negatives) and restart.
- **Quant-parity pilot:** if int6 quantization of SGNS embeddings loses > 0.01 bpb vs bf16 when used as init, fall back to bf16 packaging (may force V = 8192) or mark SP16384+int6 cell as non-viable.
- **Extrinsic runs:** no mid-training kill. All 20 runs complete their full 600 s budget for clean statistical comparison.

## Risks and mitigations

- **SGNS collapse** (embedding norms shrink toward zero): standard NCE regularization and sensible LR; monitor embedding norms during training.
- **V / quant interaction** (larger V quantizes less gracefully): quant-parity pilot is the gate.
- **Transformer baseline param mismatch.** The proposed transformer config is 10.49M — **2.4% below** the SSM's 10.75M, inside the **3% tolerance** we accept for this thesis test. Tightening to 1% would require unconventional `d_model` (e.g., d=260 at n_layer=8 for exact match, which breaks power-of-2 head-dim). The 3% gap is small relative to the minimum detectable effect (~0.025 bpb at 5 seeds); treat as matched.
- **Inverse result** (transformer benefits more than SSM): genuine possibility. Interpreted as thesis failure, not engineering failure. The experiment is honest.
- **Higher-than-expected seed variance:** escalate to 7 seeds; rerun Phase 0 pilot to re-estimate.

## Open questions for redline

- **SGNS hyperparameters.** Defaults are standard (window=5, negatives=10, subsampling=1e-5, d=256, 3 epochs). Per Ken, Exp 19/20 are the real downstream validators; iterate if LM runs expose issues. Out of Exp 21 scope to pre-tune.

## Deferred to future experiments

- Embedding mutation during LM training (freeze schedule, LR profile, biologically-inspired critical-period shape) — post-Exp-19 session.
- Eval-time TTT adaptation of the embedding — Exp 20 scope.
- Alternative offline objectives (CBOW, tiny-LM CE, encoder-based JEPA) — comparison candidates only if SGNS wins cleanly in Exp 21.
