# Learned Tokenizer + Revised Experiment Design

**Date:** 2026-04-07
**Context:** The tokenizer is not a preprocessing step — it's Stage 0 of the composition pipeline. The VQ codebook in the tokenizer and the VQ codebook in Wernicke form a hierarchy of typed representations. Both are learned end-to-end. The compression system, regret tracking, and memory typing all key off this shared hierarchy.

## Core Insight

A "parameter" in ChaosControl is three different kinds of thing:

| Component | Stores | Learned via | Analogous to |
|-----------|--------|-------------|--------------|
| SSM tensors | How to process | Backprop | Procedural memory — skills |
| LTM slots | What happened | Consolidation | Episodic memory — experiences |
| Decompression state | How to remember | Backprop + VQ | The act of recall itself |

The tokenizer adds a fourth: the codebook stores **how to perceive** — the learned segmentation and typing of raw input. This is the foundation everything else builds on.

## Architecture: Tokenizer → Wernicke → SSM

```
raw bytes (256)
  → LearnedTokenizer (Stage 0: bytes → learned tokens)
    → codebook_tok: K entries × token_dim (Level 0 types = morpheme-like)
  → Wernicke (Stage 1: tokens → typed units)
    → codebook_wer: K entries × model_dim (Level 1 types = semantic)
  → Alignment loss between codebook_tok and codebook_wer
  → SSM recurrence (working memory)
  → Memory tiers (episodic → semantic → latent)
```

## LearnedTokenizer Module

Three architectural variants, all: raw bytes in → (token_embeddings, token_ids) out.

### Variant A: Fixed Stride

```python
class FixedStrideTokenizer(nn.Module):
    byte_embed: Embedding(256, byte_dim)
    downsample: Conv1d(byte_dim, token_dim, kernel_size=window, stride=S, causal)
    codebook: Parameter(K, token_dim)  # VQ codebook = vocabulary

    forward(byte_ids):  # (batch, byte_seq)
        x = byte_embed(byte_ids)           # (batch, byte_seq, byte_dim)
        x = causal_conv_downsample(x)      # (batch, byte_seq//S, token_dim)
        token_ids = vq_nearest(x, codebook) # (batch, token_seq)
        token_embeds = straight_through(x, codebook[token_ids])
        return token_embeds, token_ids
```

- Predictable sequence length: `token_seq = byte_seq // S`
- Byte boundaries may not align with meaningful units
- Cheapest compute

### Variant B: Learned Boundaries

```python
class LearnedBoundaryTokenizer(nn.Module):
    byte_embed: Embedding(256, byte_dim)
    boundary_scorer: Conv1d(byte_dim, 1, kernel_size=window, causal) + sigmoid
    pool_encoder: Linear(byte_dim, token_dim)
    codebook: Parameter(K, token_dim)

    forward(byte_ids):
        x = byte_embed(byte_ids)              # (batch, byte_seq, byte_dim)
        scores = boundary_scorer(x).squeeze()  # (batch, byte_seq) in [0,1]
        # Binary boundary decisions (straight-through on threshold)
        boundaries = (scores > 0.5).float() + scores - scores.detach()
        # Pool bytes between boundaries into tokens
        tokens = segment_and_pool(x, boundaries)  # (batch, token_seq, byte_dim)
        tokens = pool_encoder(tokens)              # (batch, token_seq, token_dim)
        token_ids = vq_nearest(tokens, codebook)
        token_embeds = straight_through(tokens, codebook[token_ids])
        return token_embeds, token_ids, boundaries
```

- Variable rate: some tokens cover 2 bytes, some cover 8
- Needs a target token budget (auxiliary loss penalizing deviation from target rate)
- Biologically closest to speech segmentation
- Requires padding/packing for batching

### Variant C: Soft Attention Pooling

```python
class AttnPoolTokenizer(nn.Module):
    byte_embed: Embedding(256, byte_dim)
    window_attn: MultiheadAttention(byte_dim, num_heads=1)  # within-window
    query: Parameter(1, byte_dim)  # learned query per window
    project: Linear(byte_dim, token_dim)
    codebook: Parameter(K, token_dim)

    forward(byte_ids):
        x = byte_embed(byte_ids)            # (batch, byte_seq, byte_dim)
        # Reshape into non-overlapping windows
        windows = x.reshape(batch, n_windows, window_size, byte_dim)
        # Learned attention pooling: one output per window
        pooled = cross_attend(query, windows)  # (batch, n_windows, byte_dim)
        tokens = project(pooled)               # (batch, n_windows, token_dim)
        token_ids = vq_nearest(tokens, codebook)
        token_embeds = straight_through(tokens, codebook[token_ids])
        return token_embeds, token_ids
```

- Predictable length: `token_seq = byte_seq // window_size`
- Context-dependent composition within each window
- More expressive than fixed stride, simpler than learned boundaries
- Window size is the receptive field per token

## Codebook Coupling: Tokenizer ↔ Wernicke

Two separate codebooks, coupled through an alignment loss.

**Tokenizer codebook** (Level 0): `K_tok × token_dim`
- Each entry = a learned byte-pattern type (morpheme-level)
- Standard VQ commitment loss

**Wernicke codebook** (Level 1): `K_wer × model_dim`
- Each entry = a semantic type (phrase-level)
- Standard VQ commitment loss (or MoE routing)

**Projection:** Linear(token_dim → model_dim) bridges the two spaces.

**Four alignment mechanisms to test:**

### None (gradients only)
No explicit coupling loss. The two codebooks are coupled only through the gradient flow: tokenizer output → projection → Wernicke input. The projection layer is the only bridge.

### Soft Contrastive (InfoNCE)
For each Wernicke bucket j, compute the mean of all projected tokenizer codes that were routed to bucket j in this batch. This mean should be closer to Wernicke entry j than to other entries:

```
sim(i,j) = cosine(mean_tok_for_bucket_j, wer_entry_i)
L_align = -mean_j[log(exp(sim(j,j)/τ) / sum_i(exp(sim(i,j)/τ)))]
```

Encourages tokenizer codes to cluster by Wernicke type.

### Diversity (SSIM-style)
Penalize similarity between the two codebooks' entries to encourage complementary information:

```
L_diverse = mean(|cosine_similarity(projected_tok_entries, wer_entries)|)
```

Minimizing this pushes the two codebooks to encode different aspects. Combined with the gradient coupling, this prevents redundancy while maintaining relatability.

### Cosine Distillation
Each Wernicke bucket acts as a teacher. Tokenizer codes that feed into bucket j should predict the Wernicke entry:

```
L_distill = 1 - mean_j[cosine(mean_tok_for_bucket_j, wer_entry_j)]
```

Simpler than contrastive, no temperature parameter, but doesn't explicitly push non-corresponding entries apart.

**All alignment losses are weighted by `align_weight` (config param, default 0.05).**

## FineWeb Data Pipeline

**Source:** FineWeb validation set, same 50k docs as the competition leaderboard.

**Format:** Raw UTF-8 bytes as flat uint8 tensor. No BPE, no external tokenizer.

**Data loader:** `load_fineweb_splits(path)` reads raw text files, converts to byte tensor, splits into train/val matching the competition's document split.

**bpb calculation:**
```python
def compute_bpb(total_ce_nats: float, total_raw_bytes: int) -> float:
    return total_ce_nats / total_raw_bytes / math.log(2.0)
```

The denominator is the raw byte count of the validation text, measured independently of the model. The numerator is the sum of per-token cross-entropy losses across all tokens. For a tokenizer with stride S, each token covers S bytes — the CE per token is higher but there are fewer tokens, so the total CE reflects the same information content.

**This is obviously legal:** raw byte count is a property of the text, CE is a property of the model's predictions. No tokenizer-dependent correction.

## Experiment Matrix

### Layer 0: Tokenizer Architecture (5 configs × 3 seeds = 15 runs)
All: diag A-mode, no gate, no memory, no Wernicke. Isolates the tokenizer.

| Config | Architecture | Vocab | Stride |
|--------|-------------|-------|--------|
| L0_bytes | raw byte embed | 256 | 1 |
| L0_bpe | competition BPE | 1024 | ~4 |
| L0_fixed_s4 | fixed stride conv + VQ | 1024 | 4 |
| L0_learned_boundary | boundary predictor + pool + VQ | 1024 | variable |
| L0_attn_pool | windowed attention pool + VQ | 1024 | window=4 |

### Layer 0.5: Codebook Coupling (4 configs × 3 seeds = 12 runs)
Winner tokenizer. Wernicke enabled (moe_16). Tests alignment mechanism.

| Config | Alignment |
|--------|-----------|
| L05_align_none | gradients only |
| L05_align_contrastive | InfoNCE |
| L05_align_diversity | SSIM-style |
| L05_align_distill | cosine distillation |

### Layer 1: Gate Modes (7 configs × 3 seeds = 21 runs)

| Config | Gate |
|--------|------|
| L1_baseline_ssm | none |
| L1_baseline_tfm | none (transformer) |
| L1_gate_fork_k4 | fork pick-best |
| L1_gate_mc_k4 | monte carlo stats |
| L1_gate_mcts_k4 | micro-MCTS K=4 H=8 |
| L1_gate_mcts_k8 | micro-MCTS K=8 H=8 |
| L1_gate_random_k4 | random timing control |

### Layer 2: Memory (6 configs × 3 seeds = 18 runs)

| Config | Memory | Warmup |
|--------|--------|--------|
| L2_mem_none_cold | none | cold |
| L2_mem_epi_cold | episodic | cold |
| L2_mem_epi_warm | episodic | warmup |
| L2_mem_both_cold | episodic + semantic | cold |
| L2_mem_both_warm | episodic + semantic | warmup |
| L2_mem_both_warm_fullseq | full stack + latent | warmup |

### Layer 3: CFR + Compression (4 configs × 3 seeds = 12 runs)

| Config | CFR | Consequence | Typed Storage |
|--------|-----|-------------|---------------|
| L3_no_cfr | no | no | no |
| L3_cfr | yes | no | yes |
| L3_cfr_consequence | yes | yes | yes |
| L3_cfr_consequence_typed | yes | yes + typed consolidation | yes |

### Layer 3.5: Dark Horses (3 configs × 1 seed = 3 runs)
Cross-layer interactions the layered design might miss.

### Layer 4: Scaling (4 configs × 1 seed = 4 runs)
dim = 128 / 256 / 384 + transformer 384

### Layer 5: Full A-mode (3 configs × 1 seed = 3 runs, 900s budget)

### Layer 6: Inference Adaptation Depth (4 configs × 3 seeds = 12 runs)

| Config | What adapts at eval |
|--------|-------------------|
| L6_wm_only | recurrence only |
| L6_wm_plus_episodic | + surprise-gated episodic writes |
| L6_wm_plus_all | + semantic consolidation + latent reactivation |
| L6_wm_plus_all_seeded | same, LTM starts from training (not cold) |

## Totals

| Layer | Configs | Seeds | Runs | Budget | Time |
|-------|---------|-------|------|--------|------|
| L0 | 5 | 3 | 15 | 150s | 38 min |
| L0.5 | 4 | 3 | 12 | 150s | 30 min |
| L1 | 7 | 3 | 21 | 150s | 53 min |
| L2 | 6 | 3 | 18 | 150s | 45 min |
| L3 | 4 | 3 | 12 | 150s | 30 min |
| L3.5 | 3 | 1 | 3 | 150s | 8 min |
| L4 | 4 | 1 | 4 | 150s | 10 min |
| L5 | 3 | 1 | 3 | 900s | 45 min |
| L6 | 4 | 3 | 12 | 150s | 30 min |
| **Total** | **40** | | **100** | | **~4.8 hours** |

## Implementation Order

1. FineWeb raw bytes data loader
2. LearnedTokenizer module (3 variants: A, B, C)
3. Codebook alignment losses (4 types)
4. Wire tokenizer into ChaosStudentLM and runner
5. Update bpb calculation for variable-stride tokenizers
6. Update experiment 09 configs and runner for new layers
7. Tests for all new modules
8. Deploy and run

## Success Criteria

- Learned tokenizer (any variant) beats both raw bytes and competition BPE
- Alignment mechanism produces measurably different codebook structure than no alignment
- The winning tokenizer-alignment combination improves downstream component performance (L1+ results better than round 1)
- bpb on FineWeb is comparable to leaderboard entries (< 1.5 would be noteworthy for an SSM)
