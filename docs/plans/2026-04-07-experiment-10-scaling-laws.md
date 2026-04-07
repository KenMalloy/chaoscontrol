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

`full_ssm` uses the winning component configuration from experiment 09 (gate mode, memory tier, Wernicke routing, CFR — whatever won each layer).

`comp_tfm` only runs at XL size (512d) where it matches the competition baseline's architecture. At smaller sizes, `our_tfm` suffices for the scaling curve.

### Training Protocol

- **Dataset:** FineWeb (same as competition)
- **Budget:** 600s per run (10 min, matching competition rules)
- **Seeds:** 3 per config
- **Total runs:** 5 sizes × 4 configs × 3 seeds = 60 runs (minus comp_tfm at XS-L = 48 runs)
- **Estimated wall time:** ~8 hours sequential, ~2 hours with 4-way parallelism

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

## Kill Criteria

- If bare SSM at XL (512d, 600s) scores worse than 2.0 bpb on FineWeb → architecture is not competitive, investigate before continuing
- If transformer α > SSM α by >0.1 → transformers scale better here, bio components can't compensate
- If component_delta < 0 at XL → full stack hurts at scale, strip back to bare SSM for competition entry

## Success Criteria

- SSM scaling exponent α ≥ transformer α (SSM scales at least as well)
- component_delta > 0 at all sizes (bio stack always helps)
- SESSM at XL beats competition baseline (< 1.2244 bpb)
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
4. Logs: bpb, param_count, steps, wall_time, FLOPs_estimate
5. Produces scaling law plots (matplotlib) and a summary table

The competition baseline (`baselines/parameter_golf/train_gpt.py`) runs separately with its own data pipeline. Its bpb is plotted as a reference line.
