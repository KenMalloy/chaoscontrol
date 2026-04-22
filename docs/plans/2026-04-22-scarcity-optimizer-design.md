# Scarcity-Aware Optimizer — Design

**Date:** 2026-04-22
**Status:** Design plus initial fallback implementation. Not on the Exp24 first-wave critical path. Target: Exp25+.
**Supersedes:** SemanticOptimizer (killed 2026-04-22 after matched-step + LR-sweep probes landed it behind Muon at every LR tested) and several earlier iterations of this design that used parameter statistics as scarcity proxies.

## Thesis

*On our SSM at ~10M parameters, on FineWeb as loaded through our fast path, we expect a scarcity-aware optimizer to outperform Muon by compensating in the backward pass for the forward-pass smoothing bias inherent to recurrent state.*

SSMs integrate state across time — common patterns accumulate, rare events get smoothed into the average. Generic optimizers (Adam, Muon) also treat all gradient directions similarly in the backward pass. An SSM trained with a generic optimizer has rare signals under-served by both halves of the training loop. Attention-based models have a different forward-pass behavior (selection rather than integration); this story may not transfer to them. We make no architectural-law claim; extending the thesis to transformers requires a matched-param transformer control, out of scope for Exp25.

## Core mechanism

The optimizer starts from *rare events in the current batch*, traces their credit backward into parameters, and preserves the rare gradient direction that common-token training would otherwise erase.

Three principles, each of which kills a specific weaker version:

1. **Event-conditioned, not statistical.** Scarcity is measured from the current batch's actual rare events, not from static proxies (token frequency alone, parameter gradient history). "Quiet is not rare; quiet is often dead." Rare-but-mastered tokens stop getting preferential treatment automatically.
2. **Credit-assigned, not parameter-local.** Per-token pressure is routed to specific parameters through gradient attribution, not applied uniformly to parameters that "look like" they might carry rare signal.
3. **Geometric, not magnitudal.** Scarcity reshapes the update direction through a two-sided metric on matrices, not a scalar magnitude knob. Post-hoc scalar scaling is equivalent to per-group learning-rate tweaking and does not capture the thesis.

## Update rule

```
Per step:
    Step 1.  Forward:
                - Per-token CE (unreduced): ce ∈ R^{B×T}, graph retained
                - Per-layer activation hooks populated: h_L ∈ R^{B×T×C}
    Step 2.  Pressure is a constant weight on ce — compute detached:
                with torch.no_grad():
                    rarity[t]   = token_frequency_weight(target[t])
                    baseline[t] = calibrated_baseline(context[t], target[t])
                    pressure[t] = clamp_positive(rarity[t] · (ce[t] - baseline[t]))
                total_loss = ce.mean()
                rare_loss  = (ce · pressure).sum() / pressure.sum().clamp_min(1)
             rare_loss backprops through ce only; pressure carries no grad.
    Step 3.  Gradients via autograd.grad — do NOT call .backward() (it frees the graph before the split pass):
                g = autograd.grad(total_loss, params, retain_graph=True)
             Every N steps (N default 4; the "split step"):
                rare_grads = autograd.grad(rare_loss, params, retain_graph=True)
                for each hooked SSM layer L:
                    rare_channel_pressure[L, c] = autograd.grad(rare_loss, h_L, retain_graph=True)[...,c].abs().mean()
                Release graph on the last autograd.grad call of the step (retain_graph=False).
             Off-split steps: EMA of rare_grads and rare_channel_pressure; no second traversal.
    Step 4.  DDP all-reduce of every scarcity-derived quantity used downstream
             (g, rare_grads, rare_channel_pressure, row_pressure_ema) — see Engineering.
    Step 5.  Rare-orthogonal preservation against Muon's actual common direction
             (Nesterov momentum, matching muon.py:139 — projecting against raw g would
              confound scarcity with a different baseline optimizer):
                buf[p].mul_(momentum).add_(g)
                common = g + momentum · buf               # Nesterov lookahead
                r_parallel = ((r · common).sum() / (common.square().sum() + ε)) · common
                r_orth     = r - r_parallel
                direction  = common + β · r_orth          # β default 1.0
    Step 6.  Apply geometry by parameter role (see Components).
    Step 7.  W -= lr_t · update
```

Step 5 is the load-bearing move: "preserve the rare-gradient direction that's orthogonal to the common-gradient direction." When rare and common gradients point similarly, `r_orth` is small and the update is close to standard Muon. When they conflict, `r_orth` preserves the rare direction through what would otherwise be a common-token update. Projecting against `common` (Muon's actual baseline direction) rather than raw `g` keeps the ablation clean — Run 2d's "full mechanism" differs from 3a's "Muon baseline" only in the rare/common/geometry terms, not in the underlying momentum behavior.

## Components

### Event-conditioned pressure

Per token `t` in the batch:

```
rarity[t]   = 1 / log(1 + freq(target[t]))      # precomputed from training corpus
baseline[t] = calibrated_baseline(context[t], target[t])
excess[t]   = max(0, ce[t] - baseline[t])       # only positive excess counts
pressure[t] = rarity[t] · excess[t]             # element-wise in [0, ∞)
```

**`rarity`** is a static property of the target token — same precomputed unigram table as earlier designs, ~64KB at V=16384.

**`baseline[t]`** is the predicted expected CE for this token. The design target is an attention-head-predicted baseline given local context: a small offline-trained attention head (1-2 layers, dim 128, ~200-500K params) produces `baseline[t]` from a context window around position `t`, is frozen during the 600s training run, and is used only to compute `excess`.

Current implementation note (2026-04-22): because the community/legal question on a pretrained helper artifact is still unanswered, the runner defaults to `FrequencyBucketBaseline`, an in-run per-frequency-bucket EMA. That keeps ScOpt self-contained and avoids placing pretrained baseline shards on HF, but it is a weaker calibration signal. The `pressure_stats` telemetry is therefore mandatory: the fallback is allowed to miss the 5-25% positive-pressure target without implying the rest of the mechanism failed.

Why attention is still the preferred calibration upgrade: this is where attention's strength (retrieval-style lookup from local context) is genuinely the right tool. SSM state would also work but would conflate the baseline with the model's own dynamics. A frozen, context-aware baseline gives a cleaner "excess" signal than a per-frequency-bucket running mean.

**`excess`** is clipped at zero — tokens where the model is better than baseline contribute no pressure. This is what makes the system self-correcting: a rare token the model masters drops out of the pressure signal until new rare tokens appear.

### Gradient decomposition (rare/common split)

Every `N` steps (default `N = 4`; amortizes the extra backward pass over 4 steps of EMA):

```
rare_loss = (ce · pressure).sum() / pressure.sum().clamp_min(1)
rare_grads[p] = autograd.grad(rare_loss, p, retain_graph=True)
```

In between:

```
rare_grad_ema[p] = 0.9 · rare_grad_ema[p] + 0.1 · rare_grads[p]
```

The rare/common split is applied at Step 5 of the update rule. Parameters whose rare-gradient EMA is nearly parallel to their common gradient receive a near-standard update. Parameters where the rare direction is orthogonal to the common receive a perturbation that preserves the rare component.

**Why this replaces L-BFGS.** L-BFGS approximates the Hessian; the rare/common split directly isolates the signal we care about. L-BFGS has indefinite-Hessian, secant-instability, DDP-scalar, warmup-schedule, and history-semantics failure modes. The rare/common split has none of those. It's also more thesis-faithful: the Hessian doesn't know about rare events; the rare/common decomposition does.

### Credit assignment to per-channel scarcity

Per-token pressure needs to be routed to the specific channels that processed rare tokens. Use autograd.grad with respect to each layer's activations:

```
# For each SSM layer L with activation tensor h_L[B, T, C]:
dh_rare = autograd.grad(rare_loss, h_L, retain_graph=True)[0]
rare_channel_pressure[L, c] = dh_rare[:, :, c].abs().mean()
```

This is credit assignment by construction — gradient of `rare_loss` w.r.t. the layer's activations tells you which channels carried rare-event gradient. Aggregate per-channel (mean of abs across batch and time), and you have a per-channel pressure signal.

Attention-based attribution is a defensible alternative but not necessary; autograd.grad is cheaper and already principled. Keep attention-attribution in reserve if autograd's signals turn out noisy in practice.

### Per-role geometric integration

Different parameter roles get different geometric treatment. All three forms change the update's *direction*, not just its magnitude.

**Body matrices (diagonal two-sided metric).** For `W ∈ R^{out×in}` with `direction ∈ R^{out×in}` from Step 5:

```
# Per-output-channel and per-input-channel scarcity, derived from rare_channel_pressure
# on the activations flowing into/out of W.
out_scarcity[o] = tanh(rare_channel_pressure[layer_of_W, o] / τ_out) + 1    # ∈ [1, 2]
in_scarcity[i]  = tanh(rare_channel_pressure[prev_layer,  i] / τ_in)  + 1   # ∈ [1, 2]

L = out_scarcity.sqrt()[:, None]     # [out, 1]
R = in_scarcity.sqrt()[None, :]      # [1, in]

G_metric = L · direction · R
update   = NS(G_metric)
```

Each side (L, R) carries scarcity for the channels feeding into and out of W. The orthogonalization happens in the scarcity-weighted metric, producing directions aligned with rare input/output channels. This is Shampoo-shaped (left/right preconditioners) but with scarcity as the source instead of gradient covariance. Per-matrix scalar scarcity would have been too blunt for matrices where rare directions live in specific row/column subspaces.

**Token-indexed rows (`embed.weight`, `lm_head.weight`).** Per-token pressure maps directly to per-row scarcity:

```
row_scarcity[token] = tanh(pressure_ema[token] / τ_row) + 1    # ∈ [1, 2]
update = NS(diag(row_scarcity^α) · direction)                  # α default 0.5
```

Pre-scale scarce rows UP before NS, no post-unscale. Scarce rows dominate the SVD of the pre-scaled input; NS finds directions aligned with scarce-token structure. Applied to **both `embed.weight[token]` and `lm_head.weight[token]`** — the model has untied input and output embeddings, and rare-token logits need protection just as rare-token inputs do.

**Recurrence parameters (`log_a`, `delta_proj`).** Per-channel half-life times per-channel rare pressure:

```
a_base[c]   = sigmoid(log_a[c])
decay[c]    = exp(-mean_softplus_delta · a_base[c])   # running mean of delta per channel
half_life[c] = log(0.5) / log(decay[c].clamp(max=0.999))

timescale_factor[c] = normalize(half_life[c])         # ∈ [0, 1] across channels
update_scale[c] = 1 + β_rec · rare_channel_pressure[c] · timescale_factor[c]

update = direction · update_scale                     # per-channel multiplier
```

Long-memory channels (large half-life) implicated in rare-token errors get amplified updates. Short-memory channels or ones not implicated in rare events get closer-to-standard updates. This directly says: "this recurrence is smoothing rare events too much; update the timescale machinery."

### Threshold floors and warm-start

All `τ_*` thresholds use `τ = max(c · running_std, τ_floor)` with explicit floors to avoid divide-by-zero at init. Defaults:

```
τ_out_floor = τ_in_floor = τ_row_floor = 1e-4
c_* = 0.5      # Gerber-style standard
```

**Warm-start.** For the first 200 training steps, all scarcity factors are forced to 1.0 (no modulation). Running statistics stabilize during this phase; then scarcity engages smoothly. Same principle applies to `rare_grad_ema` — undefined during warm-start, skipped until sufficient history accumulates.

## Engineering integrations

- **Unreduced, weighted CE with retained graph.** The current fast path computes total CE inside the fused LM-head helper (`src/chaoscontrol/kernels/_lm_head_loss/__init__.py:552`, reductions `"mean"` / `"sum"` only) and the runner immediately calls `.backward()` on it (`experiments/23_fast_path/runner_fast_path.py:278`). ScOpt needs an alternate entry that returns per-token CE with the graph retained, so both `total_loss` and `rare_loss` can be differentiated via `autograd.grad`. Add `reduction="none"` + a `backward=False` flag; route the optimizer through the new path while the existing runners stay on the fused .backward() fast path. This is a prerequisite — cannot be deferred to implementation-time.
- **`autograd.grad` over `.backward()`.** Two reasons: (1) `.backward()` frees the graph before rare_grads can be computed; (2) `autograd.grad` gives us the control needed to retain_graph across sub-calls and release on the last. Cost of the second traversal is bounded by sharing activations across calls (same hooks, same `h_L` tensors).
- **Activation hooks.** Register once at model build time, one hook per SSM layer, capturing `h_L` into a per-step dict. No per-step overhead beyond the grad call itself.
- **DDP all-reduce contract.** Every scarcity-derived quantity used by the optimizer must be synchronized before it affects the update, or DDP replicas diverge. Per split step:
  - `g` (common gradient) — normal gradient all-reduce.
  - `rare_grads` — second all-reduce after the rare autograd.grad call.
  - `rare_channel_pressure[L, c]` — all-reduce + mean across ranks before it feeds `out_scarcity` / `in_scarcity` / recurrence `update_scale`.
  - `row_pressure_ema[token]` — all-reduce + mean across ranks before it feeds `row_scarcity`.
  Cost: 2 extra collectives per split step (rare_grads already flat-bucketed; scarcity tensors are small). Off-split steps: EMA advances locally, but must be all-reduced again before first use after the next split step (cheap; already-synchronized EMA inputs mean EMA outputs are only locally drifted by clamps/no-ops).
- **Baseline provider.** The implemented default is `FrequencyBucketBaseline`, so no external weights are loaded. The optional attention baseline head remains a future calibration upgrade: frozen during training, loaded from disk once per pod setup, ~200-500K params × fp32 = ~2MB, training-time-only, and not in the shipped artifact.
- **Frobenius vs spectral normalization.** Current Muon (`muon.py:41`) uses Frobenius norm `X / max(‖X‖_F, ε)`. Two-sided metric inputs `L · G · R` have a different condition number than raw `G`; verify NS convergence empirically on scarcity-scaled inputs (Tier 0 probe 0.2) before committing.
- **CUDA graph compatibility.** The variable-cadence split step (every N steps, not every step) and the autograd.grad call pattern will break naive CUDA graph capture. Expect to run the ScOpt path without CUDA graphs initially; revisit only if Tier -1 shows the graph-less path losing >20% throughput to Muon.
- **Step-skip on gradient clip.** If the raw gradient norm exceeded the clip threshold, skip the rare_grad_ema update for that step. Clipped gradients are noisy; don't pollute the rare/common decomposition.

## Artifact budget and legality

- **Precomputed unigram frequency table** at V=16384: ~64KB. Shipped.
- **Default frequency-bucket baseline**: in-run EMA state only. Training-time-only; never evaluated or exported.
- **Optional offline-trained attention baseline**: ~2MB weights if approved. Training-time-only; used to compute `excess[t]` during the 600s training run. Never evaluated. Treated as optimizer state, not as part of the shipped artifact.
- **Rare-grad EMA buffers**: ~40MB at bf16 for 10M params. Training-time state.
- **Optimizer code**: ~few hundred lines of Python. Shipped. Fits easily in the 16MB artifact cap.

**Legality.** The attention baseline is novel relative to current Param Golf submissions. It's offline-trained on FineWeb training data (same precedent class as the SP tokenizer); its output is used only during training (no TTT on it, no inference-time role). Until the discussion/clarification resolves, do not rely on a pretrained helper artifact. The current implementation uses the frequency-bucket fallback as the conservative path.

Before reintroducing the attention baseline: run `gh pr list -R openai/parameter-golf` to check for analogous precedent. If none exists, file a clarifying issue with the exact mechanism described, then decide whether the helper weights live as training-only artifacts on HF or stay private to the pod setup.

## Test sequence

Five tiers. Tier -1 and Tier 0 gate in order; only after both pass do we spend on quality runs. Each quality arm gets its own LR sweep (3-5 LRs around Muon's 0.064 baseline) before head-to-head comparisons.

### Tier -1 — Overhead gate (no quality claim)

Before any quality run. The "~1.25× Muon" estimate in an earlier draft omitted baseline-head inference, unreduced-CE plumbing, activation-grad traversal, the extra collectives, retained-graph memory, and CUDA-graph loss. Measure first.

Run on 1×H100 and 8×H100, 50 steps each, warm-cache:

- **-1.1 Peak VRAM.** Full ScOpt path vs Muon. Threshold: ScOpt peak ≤ 1.5× Muon peak at matched bs/seq, or the chunked-LM-backward + activation-hook stack needs rework before continuing.
- **-1.2 Throughput (tokens/sec).** Full ScOpt path vs Muon. Report split-step and off-split-step separately — off-split cost is what dominates at N=4. Threshold: end-to-end tokens/sec ≥ 0.75× Muon, or the mechanism can't earn its keep inside the 600s budget.
- **-1.3 Collective time breakdown.** Profile DDP: rare_grads all-reduce + rare_channel_pressure + row_pressure_ema. If collectives are >10% of step time at 8×H100, batch them.
- **-1.4 Scaling sanity.** Confirm 8×H100 throughput is within 10% of `8 × 1×H100 throughput` on the ScOpt path. If it diverges, something in the optimizer is serializing.

**Gate:** -1.1 and -1.2 are hard — if either fails, either shrink the mechanism (drop to scalar scarcity, drop to embed-only) or park ScOpt for this scale. -1.3 and -1.4 inform engineering fixes but do not gate.

### Tier 0 — Pre-run sanity probes (single seed, cheap)

Run on 1×H100, 60s each:

- **0.1 Signal distribution check.** Confirm `rare_channel_pressure` has non-degenerate distribution across layers after ~1000 training steps. If all channels show similar pressure, the signal can't differentiate.
- **0.2 NS convergence on scarcity-scaled inputs.** Run NS on `diag(s^α) · p` and `L · G · R` with observed ranges of scarcity and gradient. Verify convergence residual is comparable to standard NS (within 2×).
- **0.3 Rare/common gradient alignment.** Measure mean cosine between `r` and `common` (the Nesterov direction from Step 5, not raw `g`) across parameters. If they're nearly parallel everywhere (cos > 0.9), `r_orth` is too small for the mechanism to matter. If they're nearly orthogonal everywhere (cos < 0.1), the mechanism will dominate and likely destabilize training.
- **0.4 Baseline head calibration — not just MSE.** Two-part probe, both required:
    - (a) MSE vs frequency-bucket baseline on held-out FineWeb tokens.
    - (b) *Pressure sparsity over training time.* Measure `fraction(pressure[t] > 0)` at steps 100, 1k, 10k, and at the N=4 split cadence. If sparsity is dense early (>40% positive) or vanishes (<1% positive after warm-start), the baseline's calibration is wrong — too weak and it degenerates to "weighted ordinary CE"; too strong and pressure signal disappears. Target range: 5-25% positive across most of training.

**Gate:** if any sanity probe fails, stop and fix before spending on full tiers.

### Tier 1 — Placebo

- **Run 1a:** Muon + real scarcity (full v4 mechanism).
- **Run 1b:** Muon + shuffled-pressure placebo (permute `pressure[t]` across positions so it no longer corresponds to real rare events; preserves mean/variance, destroys structure).

If 1a doesn't beat 1b under the noise-aware decision gate, the mechanism is acting as noise/regularization, not picking up rare-event structure. Cleanest falsifier.

### Tier 2 — Ablation of the load-bearing moves

- **Run 2a:** Scalar scarcity replaces the two-sided metric (post-hoc `· s` applied to whole matrix). Tests whether per-channel geometry matters.
- **Run 2b:** `β = 0` in Step 5 (no rare-direction preservation; scarcity only affects row/column scaling). Tests whether the rare/common split is load-bearing.
- **Run 2c:** `r_parallel = r` (no orthogonal decomposition; pure rare-gradient boost). Tests whether the orthogonal-direction preservation specifically matters vs just boosting rare gradient.
- **Run 2d:** Full v4 mechanism.

Reveals which specific moves are carrying the result.

### Tier 3 — Full comparison

- **Run 3a:** Muon baseline.
- **Run 3b:** Full v4 mechanism.
- **Run 3c:** Full v4 mechanism, with running-mean baseline instead of attention baseline (fallback scenario).

All at 600s × 3 seeds × 8×H100.

## Decision gates (noise-aware)

1-seed BPB variance at 600s is 0.005-0.01 from our memory; with 3 seeds and paired-seed comparison, SE on mean difference is roughly 0.003-0.006. Define:

- **Minimum meaningful effect:** `δ_bpb = 0.005`.
- **Wins definition:** A "beats" B iff `mean(bpb_A) ≤ mean(bpb_B) - δ_bpb` **and** paired-seed 95% CI on `bpb_A - bpb_B` excludes zero.
- **Tied:** neither arm wins. Mean difference within `±δ_bpb` or CI includes zero.

Gates:
- **Tier 0 fails any probe:** stop and fix.
- **1a ties or loses to 1b:** thesis is wrong. Park.
- **1a beats 1b, but 2a beats 2d:** per-channel geometry isn't carrying the result. Ship the simpler scalar form.
- **2b beats 2d:** rare/common split isn't carrying the result. Ship without it.
- **2c beats 2d:** orthogonal decomposition isn't carrying; pure rare boost suffices. Simpler form wins.
- **3b beats 3a:** mechanism works. Ship.
- **3c ≈ 3b:** attention baseline isn't uniquely load-bearing; running-mean fallback is acceptable.

## Diagnostics and runtime telemetry

Every run logs scalars every 100 steps. Required for postmortem if a run fails:

- **Pressure distribution:** min, median, 95th percentile, max of `pressure[t]` across the batch; fraction `pressure[t] > 0` (sparsity track from Tier 0 probe 0.4, continued).
- **Rare/common alignment:** mean cosine between `r` and `common` (the Nesterov direction used in Step 5) per parameter group — not between `r` and raw `g`. The ablation's load-bearing quantity is `r_orth` measured against `common`.
- **`r_orth` magnitude:** `‖r_orth‖ / ‖common‖` distribution across parameters. If median ~0, the rare mechanism isn't doing anything.
- **Two-sided metric ratios:** min/max singular value ratios of `L · G · R` vs `G`. Extreme values signal the metric is destabilizing NS.
- **Post-NS scarce-row/column energy enrichment.** NS normalizes away much of the diagonal amplitude we inject via `L` and `R`, so NS-input energy is not a proxy for NS-output energy. After NS, measure:
    - `row_energy[i] = ‖update[i, :]‖² / ‖update‖²_F`; report ratio `mean(row_energy[scarce rows]) / mean(row_energy[common rows])`.
    - Same for columns.
  Ratio > 1 means scarce rows/columns did get disproportionate energy in the update. Ratio ≈ 1 means the two-sided metric isn't translating to a differential update — a silent ablation failure that NS residual alone wouldn't catch.
- **NS convergence residual** per-step, median across matrices.
- **Scarcity factor values:** min/median/max of `out_scarcity`, `in_scarcity`, `row_scarcity`, `update_scale`.
- **Step clip rate:** fraction of steps where grad clip activated.
- **Per-layer gradient norms:** check for layer-wise explosions or collapses.
- **Half-life distribution across channels** per SSM layer over time.

These make the difference between "null result, can't explain" and "null result, here's exactly what broke."

## Open questions

1. **`N` (rare/common decomposition cadence).** Default 4. Lower = more responsive, more compute. Higher = smoother EMA, cheaper. Worth sweeping once core mechanism is stable.
2. **`β` (rare-orthogonal weight in Step 5).** Default 1.0. Too high destabilizes; too low makes the mechanism inert.
3. **`α` (row-scarcity exponent for token-indexed weights).** Default 0.5. Narrow sweep (α ∈ {0.25, 0.5, 1.0}) after Tier 2.
4. **Attention baseline architecture.** 1-2 layers, dim 128 is a guess. Smaller might be fine; larger eats training-time budget. Resolve during baseline training, not during the main experiment.
5. **Credit-assignment aggregation.** `abs().mean()` across batch and time is the simplest choice. Alternatives: pressure-weighted mean (emphasize rare-token positions), max over time (peak attribution). Worth a small ablation.

## Scope discipline

- **Not** an Exp24 first-wave arm. First-wave matrix (fast_slow, spectral, predictive_aux, scheduled dreamworld, and dreamworld event_sleep) ships first.
- **Not** a revival of SemanticOptimizer. Matched-step + LR-sweep probes landed that design behind Muon at every LR tested on 2026-04-22.
- **Not** an architectural-law claim. Scoped to "our SSM at this scale on this benchmark"; transformer comparison is a follow-up experiment, not baked into this evaluation plan.
- Implementation plan comes after first-wave results land and the optimizer-research slot reopens.

## Naming

Working name: **ScOpt** (Scarcity-aware Optimizer). Placeholder; revisit before the implementation plan lands.

## References

- Gerber et al. (2022), *The Gerber Statistic: A Robust Co-movement Measure for Portfolio Optimization*, Journal of Portfolio Management 48(3): 87-102. Source of the threshold-gated signal-gating shape (c · std with floors).
- Gupta et al. (2018), Shampoo. Inspiration for the two-sided diagonal metric on matrix parameters.
- Jordan et al. (2024), Muon optimizer (Newton-Schulz orthogonalization). Baseline and the orthogonalization primitive we reuse.
- Internal: `project_scarcity_optimizer_thesis.md`, `project_criticality_status_2026-04-12.md`, `project_pgolf_real_sota_2026-04-17.md`, `reference_param_golf_rules.md`, `feedback_precedent_over_permission.md`.
- Code anchors: `src/chaoscontrol/core.py:347,402` (a_base and decay definition), `src/chaoscontrol/optim/muon.py:41` (Frobenius normalization), `src/chaoscontrol/model.py:632,758` (untied embed/lm_head).
