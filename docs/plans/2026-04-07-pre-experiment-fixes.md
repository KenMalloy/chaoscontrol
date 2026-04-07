# Pre-Experiment Fixes + Phase 1 Tokenizer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 9 live methodology bugs in the existing code, then implement Phase 1 of the learned tokenizer (fixed-stride only, FineWeb raw bytes, codebook alignment). Defer learned boundaries and tokenizer-step() integration to Phase 2.

**Architecture:** Package A fixes wiring bugs that would produce invalid experiment results. Package B adds the fixed-stride tokenizer as a new input processing stage, with a clear causal target (predict next VQ token ID) and reconstruction loss to prevent lossy collapse.

**Tech Stack:** PyTorch. No new dependencies.

---

## Package A: Fix Live Code Issues

### A1: Eval respects metabolic_mode

**Files:** `src/chaoscontrol/evaluation.py`, `src/chaoscontrol/runner.py`

**Problem:** Eval hardcodes `metabolic_fork()` for gated eval. Should use the model's configured mode (fork, MC, or MCTS).

**Fix:** Pass `metabolic_mode` through to eval. In eval, dispatch to the correct function:
```python
if metabolic_gate:
    if metabolic_mode == "mcts":
        from chaoscontrol.metabolic import micro_mcts
        gated_out = micro_mcts(model, inputs, n_rollouts=metabolic_k, ...)
    elif metabolic_mode == "monte_carlo":
        from chaoscontrol.metabolic import metabolic_monte_carlo
        gated_out = metabolic_monte_carlo(model, inputs, k=metabolic_k, ...)
    else:
        gated_out = metabolic_fork(model, inputs, k=metabolic_k, ...)
```

Also update `pick_winner()` in `run_layered.py` to use `bpb_gated` when available for gated configs.

**Test:** Add test that eval with `metabolic_mode="mcts"` produces different `bpb_gated` than `metabolic_mode="fork"`.

---

### A2: Document step() as intentionally simpler world model

**Files:** `src/chaoscontrol/model.py`, `tests/test_model.py`

**Problem:** step() skips Wernicke, memory reads, semantic bias. Tests pretend step matches forward. maxdiff = 2.6 on featureful models.

**Fix:** This is intentionally correct — mental simulation (System 2) operates on simplified internal models. But:
1. Update step() docstring to document this explicitly
2. Change `test_step_sequence_matches_forward` to only test on bare models (no Wernicke/memory)
3. Add a separate test `test_step_diverges_from_forward_with_features` that asserts step and forward produce DIFFERENT outputs when Wernicke/memory are enabled
4. Add a note in micro_mcts docstring: "Rollouts use the simplified world model (SSM recurrence only, no memory/Wernicke reads) — analogous to System 2 planning on a compressed internal model"

---

### A3: Wire CFR get_strategy() into candidate selection

**Files:** `src/chaoscontrol/training.py`

**Problem:** RegretTable accumulates regret but `get_strategy()` is never called to bias exploration.

**Fix:** When `cfr_enabled` and `regret_table` exists, use the strategy distribution to bias which candidate the gate selects. In the fork/MCTS path, before UCB selection:
```python
if regret_table is not None and dominant_bucket is not None:
    strategy = regret_table.get_strategy(dominant_bucket % regret_table.n_buckets)
    # Bias candidate selection: multiply prior probabilities by strategy weights
    # This is passed into metabolic_fork/micro_mcts as prior_bias
```

This requires adding a `prior_bias` parameter to `metabolic_fork` and `micro_mcts` that weights candidate selection.

---

### A4: Fix CFR value scale mismatch

**Files:** `src/chaoscontrol/training.py`

**Problem:** `actual_value = -ce_val` (~-5.8) while counterfactual values are softmax confidence (~0.015). Regret is always positive and meaningless.

**Fix:** Use the same metric for both. Negative CE for both:
```python
actual_value = -ce_val
# Counterfactual: run 2-step lookahead, compute CE of the continuation
for a in range(k_actions):
    # ... step forward 2 tokens ...
    cf_ce = F.cross_entropy(cf_logits, cf_targets)
    counterfactual_values.append(-cf_ce.item())
```

This requires having targets available during the lookahead. Use the actual next tokens from the training batch as targets for the first lookahead step, then greedy prediction for step 2.

---

### A5: Layered runner handles baseline-wins-L1

**Files:** `experiments/09_revised_architecture/run_layered.py`

**Problem:** If `L1_baseline_ssm` wins, `extract_gate_settings()` returns empty dict. L2+ configs inherit no gate. CFR in L3 needs the gate.

**Fix:** When baseline wins L1, explicitly set `metabolic_gate: false` in the gate settings dict. L3 configs that need CFR should force `metabolic_gate: true` regardless, because CFR is about tracking what WOULD have happened — it needs the gate to fire to generate counterfactuals. Add a note in L3 config generation:
```python
# CFR configs always enable the gate (CFR needs counterfactual generation)
if "cfr" in name:
    cfg["metabolic_gate"] = True
    if "metabolic_mode" not in cfg:
        cfg["metabolic_mode"] = "fork"
        cfg["metabolic_k"] = 4
        cfg["metabolic_threshold"] = 0.1
```

---

### A6: Fix memory tier config dependencies

**Files:** `experiments/09_revised_architecture/run_layered.py`

**Problem:** `L2_mem_both_warm_fullseq` enables `latent_persistence` and `typed_consolidation` without Wernicke or `typed_storage`. These mechanisms need `dominant_bucket` which requires both.

**Fix:** Either:
- Add `wernicke_enabled: true` and `typed_storage: true` to fullseq config
- OR make latent_persistence and typed_consolidation work without bucket IDs (fall back to untyped behavior)

The second option is more robust. In training.py, `latent_persistence` should work with `bucket_id=None` (try reactivating any latent trace when surprise is high, not just matching bucket). And `typed_consolidation` without bucket IDs should fall back to untyped consolidation.

---

### A7: Layer 6 eval exercises all tiers

**Files:** `src/chaoscontrol/evaluation.py`, `experiments/09_revised_architecture/run_layered.py`

**Problem:** Eval warmup only calls `consolidation_step(hidden_last, bucket_id=None)`. Never exercises full_sequence write, latent reactivation, or typed consolidation.

**Fix:** Add config params to eval warmup that mirror training: `warmup_write_mode` ("last" or "full_sequence"), `warmup_latent_persistence`, `warmup_typed_consolidation`. The eval function respects these when warmup=True.

Also fix L6 configs: `L6_wm_plus_all_seeded` needs to differ from `L6_wm_plus_all`. The "seeded" version starts from training memory state (slots survive from training). The non-seeded version clears slots before eval. Add `warmup_cold_start: true/false` to control whether memory is wiped before eval begins.

---

### A8: Warmup isolation covers trigger state

**Files:** `src/chaoscontrol/evaluation.py`

**Problem:** Save/restore around warmup misses `_spike_seen`, `_steps_since_spike`, `_pre_spike_loss` trigger fields.

**Fix:** Add trigger state to the save/restore:
```python
if warmup and getattr(model, "outer_model", None) is not None:
    saved_outer_state["_spike_seen"] = model.outer_model._spike_seen
    saved_outer_state["_steps_since_spike"] = model.outer_model._steps_since_spike
    saved_outer_state["_pre_spike_loss"] = model.outer_model._pre_spike_loss
    if hasattr(model.outer_model, "_retrieval_weights"):
        saved_outer_state["_retrieval_weights"] = model.outer_model._retrieval_weights
```

And restore all of them in the finally block.

---

### A9: Compute-match the random gate control

**Files:** `src/chaoscontrol/training.py`

**Problem:** Random gate fires with Bernoulli probability, but the surprise-gated version fires based on loss dynamics. They may fire at very different rates.

**Fix:** Track the empirical fire rate of the surprise gate and use it to calibrate the random gate. Add a running estimate:
```python
if metabolic_threshold_mode == "random":
    # Match the surprise gate's empirical fire rate
    if steps == 0:
        random_fire_rate = metabolic_threshold  # initial estimate
    use_fork = metabolic_gate and (rng.random() < random_fire_rate)
elif metabolic_threshold_mode == "fixed":
    use_fork = metabolic_gate and surprise_ratio > current_threshold

# Track empirical rate for random mode calibration (logged for analysis)
if metabolic_gate:
    step_record["gate_fired"] = use_fork
```

Actually, the cleaner approach: run the surprise gate AND the random gate in the random config. The surprise gate computes whether it WOULD fire (for rate tracking), but the actual fork decision uses random. This way the random config's fire rate naturally matches because it uses the same threshold parameter — the difference is only WHEN it fires, not HOW OFTEN.

Better fix:
```python
surprise_would_fire = surprise_ratio > current_threshold
if metabolic_threshold_mode == "random":
    use_fork = metabolic_gate and (rng.random() < metabolic_threshold)
else:
    use_fork = metabolic_gate and surprise_would_fire
```

The threshold parameter (0.1) means both fire ~10% of the time. The difference is surprise-timing vs random-timing. That's the controlled comparison.

---

## Package B: Phase 1 Tokenizer (Fixed-Stride Only)

### B1: FineWeb raw bytes data loader

Same as original plan Task 1, but:
- **Additive migration:** Keep `enwik8_path` in config. Add `data_path` and `data_format` as NEW fields with defaults that preserve existing behavior. Runner checks `data_format` first; if "enwik8", uses `enwik8_path` as before.
- **No rename of existing field.**

---

### B2: VQ utilities module

Same as original plan Task 2. No changes needed.

---

### B3: FixedStrideTokenizer ONLY

Same as original plan Task 3, but:
- **Only implement FixedStrideTokenizer.** Defer LearnedBoundaryTokenizer and AttnPoolTokenizer to Phase 2.
- **Add reconstruction loss:** The tokenizer must be able to decode tokens back to bytes. Add a small decoder (transposed conv) that reconstructs bytes from token embeddings. Reconstruction loss prevents lossy collapse:
```python
class FixedStrideTokenizer(nn.Module):
    # ... existing encoder ...
    self.decoder = nn.ConvTranspose1d(token_dim, 256, kernel_size=stride*2, stride=stride)

    def forward(self, byte_ids):
        # ... encode to tokens ...
        # Reconstruction: can we recover the bytes?
        reconstructed_logits = self.decoder(token_embeds.transpose(1,2)).transpose(1,2)
        recon_loss = F.cross_entropy(
            reconstructed_logits[:, :byte_ids.size(1), :].reshape(-1, 256),
            byte_ids.reshape(-1),
        )
        return token_embeds, token_ids, commit_loss, recon_loss
```
- **Clear causal target:** The model predicts the next token ID (from the VQ assignment). Training loss = CE over VQ token vocabulary. bpb = total_ce_nats / raw_byte_count / ln(2).
- **MCTS/CFR disabled for L0:** Layer 0 configs should not enable metabolic_gate or cfr_enabled. These are tested in L1+ after the tokenizer is fixed.
- **L0 test for tokenizer includes:** fixed stride K=512, fixed stride K=1024. Defer attn_pool and learned_boundary to Phase 2 experiment configs.

---

### B4: Codebook alignment losses

Same as original plan Task 4. No changes needed.

---

### B5: bpb calculation + causal target

Same as original plan Task 5, plus:
- **Explicit causal target mapping:** When using a tokenizer with stride S, the training target is the NEXT token in the VQ-assigned sequence. `batch_from_starts` returns `(inputs, targets)` where both are byte tensors. With a tokenizer, both go through the tokenizer, and the target is the tokenizer's VQ assignment for the next position.
- **Reconstruction loss added to total loss in training loop.**
- **Document that bpb = total_ce / raw_bytes / ln(2) is faithful because:** (a) the reconstruction loss prevents the tokenizer from collapsing distinctions, (b) the VQ commitment loss keeps tokens close to codebook entries, (c) the CE is over the full VQ vocabulary so the model must distinguish all learned tokens.

---

### B6: Wire into model + training

Same as original plan Task 6, plus:
- **Tokenizer is a separate module, not part of ChaosStudentLM.** The runner creates it and passes tokenized sequences to the model. The model's vocab_size = tokenizer codebook_size.
- **step() does NOT go through the tokenizer.** MCTS rollouts operate in token space (post-tokenizer). The tokenizer runs once on the input, producing token IDs. MCTS steps through tokens, not bytes.
- **Transformer baseline in L0 and L4 inherits the winning tokenizer.** The tokenizer feeds BOTH the SSM and the transformer.

---

### B7: Update experiment configs + runner

Same as original plan Task 7, plus:
- **L0 has 4 configs, not 5:** bytes, BPE, fixed_stride_k512, fixed_stride_k1024. Defer attn_pool and learned_boundary.
- **L0.5 alignment configs only run when L0 winner is a learned tokenizer.** If bytes or BPE wins L0, skip L0.5 (alignment is N/A).
- **Transformer baseline in L0:** Add `L0_bytes_tfm` — transformer on raw bytes, for comparison.

---

## Dependency Graph

```
A1-A9 (fixes, independent) ──── commit all ──┐
                                              ├── B6 (wire) ── B7 (configs) ── Deploy
B1 (FineWeb loader, independent) ─────────────┤
B2 (VQ utils, independent) ── B3 (tokenizer) ─┤
B4 (alignment, depends on B2) ────────────────┤
B5 (bpb, independent) ────────────────────────┘
```

A1-A9 can be dispatched in parallel. B1, B2, B5 can be dispatched in parallel. B3 depends on B2. B4 depends on B2. B6 integrates everything. B7 generates configs.

## Phase 2 (deferred)

- LearnedBoundaryTokenizer (variable-length, needs masks everywhere)
- AttnPoolTokenizer
- Tokenizer-aware step() API for MCTS rollouts in token space
- Attention masks threaded through Wernicke, memory, CE, eval
