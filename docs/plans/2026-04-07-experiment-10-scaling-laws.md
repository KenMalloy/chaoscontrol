# Experiment 10: SESSM Scaling Laws

**Goal:** Determine whether the Surprise-gated Episodic SSM (SESSM) scales more parameter-efficiently than a transformer under the 16MB artifact constraint, and whether bio-inspired components (gate, memory, Wernicke, CFR) become more or less valuable as model size increases.

**Hypothesis:** An SSM with recurrence-based working memory should extract more information per parameter than attention at small model sizes, because the O(d) state compresses context more efficiently than O(n²) attention when both are parameter-budget-limited. If the bio components exploit this compressed state effectively, the SESSM scaling curve should stay above the transformer curve at all sizes within the 16MB cap.

**Null hypothesis:** Transformers scale better per-parameter even at 16MB, and bio components are overhead that hurts more as the architecture needs those parameters for the base model.

---

## Design

### Independent Variable: Model Size

| Size | dim | layers | approx params (SSM) | approx params (tfm) | artifact (bf16) |
|------|-----|--------|---------------------|---------------------|-----------------|
| XS   | 64  | 2      | ~50K                | ~50K                | ~100KB          |
| S    | 128 | 4      | ~300K               | ~300K               | ~600KB          |
| M    | 256 | 6      | ~1.5M               | ~1.5M               | ~3MB            |
| L    | 384 | 8      | ~4M                 | ~4M                 | ~8MB            |
| XL   | 512 | 10     | ~8M                 | ~8M                 | ~16MB           |

Parameter counts matched by adjusting transformer layer count / ff_mult. Exact counts recorded per run.

### Dependent Variables

1. **bpb** — bits-per-byte on FineWeb validation (tokenizer-agnostic)
2. **bpb per parameter** — bpb / param_count (efficiency)
3. **bpb per FLOP** — bpb / estimated training FLOPs (isoFLOP curve)
4. **training throughput** — steps/second at each size
5. **component delta** — bpb(bare SSM) - bpb(full stack SSM) at each size

### Conditions (at each size)

| Config | Architecture | Components |
|--------|-------------|------------|
| `bare_ssm` | SSM (diag) | None — pure SSM baseline |
| `full_ssm` | SSM (diag) | Winner stack from exp 09 |
| `our_tfm` | SimpleTransformerLM | None — our transformer |
| `comp_tfm` | Competition baseline | None — parameter-golf train_gpt.py |
| `mamba2_ssm` | Mamba2LM (mamba-ssm) | None — Mamba-2 SSM baseline via `mamba_ssm.Mamba2` |

`full_ssm` uses the winning component configuration from experiment 09 (gate mode, memory tier, Wernicke routing, CFR — whatever won each layer).

`comp_tfm` only runs at XL size (512d) where it matches the competition baseline's architecture. At smaller sizes, `our_tfm` suffices for the scaling curve.

`mamba2_ssm` uses the `Mamba2LM` wrapper in `baselines.py`. Requires `pip install mamba-ssm>=2.3.0` (CUDA required). Runs at all sizes for a direct SSM-vs-SSM scaling comparison against both our custom SSM and the transformer.

### Training Protocol

- **Dataset:** FineWeb (same as competition)
- **Budget:** 600s per run (10 min, matching competition rules)
- **Seeds:** 3 per config
- **Total runs:** 5 sizes × 5 configs × 3 seeds = 75 runs (minus comp_tfm at XS-L = 63 runs)
- **Estimated wall time:** ~10.5 hours sequential, ~2.5 hours with 4-way parallelism

### Compute Budget Matching

The throughput confound from experiment 09 round 1: larger models and more complex components train fewer steps in the same wall time. To disentangle model capacity from training steps:

1. **Wall-time matched** (primary): all configs get 600s. This matches competition rules. If the SSM trains more steps, that's a real advantage of the architecture.

2. **Step-count matched** (secondary): pick a fixed step count (e.g., the XL transformer's step count at 600s) and let smaller/faster models stop early. This isolates per-step learning efficiency.

3. **FLOP-matched** (analysis): estimate FLOPs per step for each config, compute total FLOPs = steps × FLOPs/step. Plot bpb vs total FLOPs. This is the cleanest comparison but requires accurate FLOP estimation.

### FLOPs Estimation

Per-step FLOPs (approximate, batch_size × seq_len × model_ops):

| Component | FLOPs per token |
|-----------|----------------|
| SSM recurrence (diag) | ~6d² |
| Transformer attention | ~4d² + 2nd |
| Transformer FFN | ~8d² × ff_mult |
| SSM FFN | ~8d² × ff_mult |
| Wernicke VQ | ~2dK |
| Memory read/write | ~2d × max_slots |
| Metabolic fork (k=4) | ~4 × model_cost |

Log FLOPs per step for each config. Compute from actual step counts.

---

## Analysis

### Primary: Scaling Curves

Fit power-law to each architecture:
```
bpb(N) = A × N^(-α) + bpb_irreducible
```
where N = parameter count. Compare α (scaling exponent) between SSM and transformer. Higher α = better scaling.

### Secondary: Component ROI

```
component_delta(N) = bpb_bare_ssm(N) - bpb_full_ssm(N)
```

Plot component_delta vs N. Three possible outcomes:
- **Delta grows with N:** Components scale — the bio stack is worth more at larger sizes
- **Delta constant:** Components add a fixed benefit regardless of size
- **Delta shrinks with N:** Components are overhead — base model benefits more from those parameters

### Tertiary: isoFLOP Comparison

Plot bpb vs total_FLOPs for all configs. The architecture that achieves lower bpb at the same FLOP budget is more compute-efficient. This controls for the throughput confound.

### Competition Reference

Plot the competition baseline (1.2244 bpb) and SOTA (1.1147 bpb) as horizontal lines on the scaling curve. If the SESSM curve crosses the baseline line at any size within 16MB, we have a competition-viable entry. If it crosses SOTA, we have a record.

---

## Extended Metrics

### Efficiency (the inference advantage story)

| Metric | How to measure | Why it matters |
|--------|---------------|----------------|
| **Inference latency per token** | Time `model.step()` for SSM vs full forward for transformer (single token, batch=1). Measure at each model size. | SSM recurrence is O(d) per step vs O(nd) for attention. If bpb is competitive, faster inference is a publishable advantage. |
| **Memory footprint at inference** | SSM: fixed state size = `num_layers × d`. Transformer: KV cache grows with sequence. Report state bytes vs KV cache bytes at seq_len=1024,4096,16384. | Fixed-size state vs growing KV cache matters for the 16MB artifact story and real deployment. |
| **Training throughput** | steps/sec and tokens/sec, already logged. Plot alongside bpb for visual isoFLOP argument. | Makes the compute-efficiency claim concrete and visual. |

### Representation Quality

| Metric | How to measure | Why it matters |
|--------|---------------|----------------|
| **Codebook utilization** | Fraction of VQ entries assigned at least once in eval. Measure for both tokenizer codebook and Wernicke codebook at each scale. Dead entries = wasted parameters. | If utilization increases with scale, the model learns to use more of its vocabulary. If it decreases, larger codebooks waste capacity. |
| **Spectral structure** | FFT snapshots of A-matrix eigenvalues (already collected every 50 steps). Plot eigenvalue distribution at each model size. | Showing how learned dynamics change with scale (richer temporal structure at larger d?) is novel for SSMs. |

### Component-Specific

| Metric | How to measure | Why it matters |
|--------|---------------|----------------|
| **Gate fire rate vs bpb delta** | From step logs: `gate_fired` rate and per-firing bpb improvement. Plot both vs model size. | Does the gate fire more or less at scale? Does each firing contribute more? |
| **Memory slot utilization** | Unique slot buckets used / max_slots. Diversity of stored episodes. | At larger d, does episodic memory get used more effectively (more diverse slots) or does the larger recurrence make it redundant? |
| **CFR regret convergence** | Regret table entropy over training steps. Lower entropy = more converged strategy. Plot convergence rate vs model size. | Faster convergence at scale = the larger model has a more stable "policy." |

### Robustness

| Metric | How to measure | Why it matters |
|--------|---------------|----------------|
| **Seed variance** | std(bpb) across 3 seeds at each size. Compare SSM vs transformer. | Lower variance = more stable optimization landscape. If SESSM has lower seed variance, the recurrence provides regularization. |
| **Quantization degradation** | bpb(bf16) vs bpb(int8) vs bpb(int6) at each size. Delta = quantization tax. | Both architectures get quantized to fit 16MB. Which loses less? SSM weights may be more quantization-friendly than attention matrices. |

### Paper Thesis

The strongest angle: **"SESSM matches transformer on bpb, beats it on inference efficiency and quantization robustness, and the bio components' ROI grows with scale."** Three independent axes of comparison:

1. **Scaling efficiency** — bpb per parameter / per FLOP
2. **Inference advantage** — O(d) latency, fixed memory footprint
3. **Component value at scale** — bio stack ROI curve slopes upward

Any two of these holding would be publishable. All three would be a strong contribution.

---

## Kill Criteria

- If bare SSM at XL (512d, 600s) scores worse than 2.0 bpb on FineWeb → architecture is not competitive, investigate before continuing
- If transformer α > SSM α by >0.1 → transformers scale better here, bio components can't compensate
- If component_delta < 0 at XL → full stack hurts at scale, strip back to bare SSM for competition entry
- If seed variance(SSM) > 2× seed variance(transformer) → optimization landscape is too noisy

## Success Criteria

- SSM scaling exponent α ≥ transformer α (SSM scales at least as well)
- component_delta > 0 at all sizes (bio stack always helps)
- SESSM at XL beats competition baseline (< 1.2244 bpb)
- Inference latency < 50% of transformer at matched bpb
- Quantization degradation(SSM) ≤ quantization degradation(transformer)
- Publishable scaling law plot with confidence intervals

---

## Dependency on Experiment 09

Experiment 10 REQUIRES experiment 09 results to determine:
1. Winning tokenizer (L0/L0.5) → used for all experiment 10 configs
2. Winning gate mode (L1) → used for `full_ssm`
3. Winning memory tier (L2) → used for `full_ssm`
4. Winning Wernicke/CFR config (L3) → used for `full_ssm`

Run experiment 09 first. Experiment 10 uses the winners.

## Implementation

The experiment runner is a new file: `experiments/10_scaling_laws/run_scaling.py`. It:
1. Reads experiment 09 results to determine winning configs
2. Generates size-matched SSM and transformer configs at each scale
3. Trains each config for 600s
4. Logs per run:
   - bpb, param_count, steps, wall_time, tokens/sec, steps/sec
   - FLOPs_estimate (per step and total)
   - Gate fire rate, memory slot utilization, codebook utilization
   - Spectral snapshots (A-matrix eigenvalues)
   - CFR regret table entropy (if applicable)
   - Seed index for variance computation
5. Post-training per run:
   - Inference latency benchmark: `model.step()` × 1000 tokens, batch=1 (SSM) vs equivalent transformer forward
   - Memory footprint: SSM state bytes vs transformer KV cache bytes at seq_len=[1024, 4096, 16384]
   - Quantization sweep: bf16 → int8 → int6, measure bpb at each

The analyzer (`experiments/10_scaling_laws/analyze_scaling.py`) produces:
1. **Scaling law plot** — bpb vs params with fitted power-law curves + competition reference lines
2. **isoFLOP plot** — bpb vs total FLOPs
3. **Component ROI plot** — component_delta vs model size
4. **Inference efficiency plot** — latency per token and memory footprint vs model size
5. **Quantization robustness plot** — bpb delta from quantization vs model size
6. **Codebook utilization plot** — fraction of active entries vs model size (tokenizer + Wernicke)
7. **Spectral evolution plot** — A-matrix eigenvalue distributions at each scale
8. **Seed variance table** — std(bpb) per config per size
9. **Summary table** — all metrics at each size, LaTeX-ready

The competition baseline (`baselines/parameter_golf/train_gpt.py`) runs separately with its own data pipeline. Its bpb is plotted as a reference line.
