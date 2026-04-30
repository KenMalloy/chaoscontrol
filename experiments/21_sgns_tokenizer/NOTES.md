# Exp 21 NOTES

## Task 10 — Transformer baseline audit (2026-04-17)

### Existing `SimpleTransformerLM` at a glance

Source: `src/chaoscontrol/baselines.py:9-79` (class + `TransformerBlock` at :47 + `CausalSelfAttention` at :61).

- **Positional encoding:** NONE. No RoPE, no learned position embedding, no ALiBi. `forward` applies `self.embed(input_ids)` and feeds straight into layers (`baselines.py:31-33`).
- **Norm:** RMSNorm via `chaoscontrol.core.RMSNorm` (`core.py:210-218`). Pre-norm residual shape: `x = x + attn(norm(x)); x = x + ff(norm(x))` (`baselines.py:55-58`). Matches design.
- **Activation:** SiLU via `core.FeedForward` (`core.py:221-229` — hard-codes `F.silu(self.fc(x))`). Design wants ReLU².
- **Attention variant:** `F.scaled_dot_product_attention(q, k, v, is_causal=True)` (`baselines.py:77`). This is the PyTorch SDPA entry point — on H100 + bf16 + supported shapes it dispatches to FlashAttention-2 internally. No QK-norm on Q or K before SDPA.
- **Tied embed/LM-head:** NO — `nn.Embedding(V,d)` and `nn.Linear(d,V,bias=False)` are separate parameters (`baselines.py:15,20`). Matches design.
- **Config knobs that could let us match spec without forking:** none of the missing pieces (RoPE, ReLU², QK-norm) are gated by a flag. The constructor accepts only `(vocab_size, dim, num_layers, num_heads, ff_mult)`. Adding flags would require touching `TransformerBlock`, `CausalSelfAttention`, and `FeedForward` simultaneously — `FeedForward` is shared with the SSM arm (`model.py` imports it), so a `nonlinearity` flag there has cross-arm blast radius.

### Existing primitives in the repo

- **RMSNorm:** `src/chaoscontrol/core.py:210` — reusable as-is.
- **RoPE:** `src/chaoscontrol/model.py:326` `_ensure_rope_cache` + `:355` `apply_rope` (staticmethod). Lives on `AttentionControlBlock`. The math is extracted as a static method, so a new transformer block can call `AttentionControlBlock.apply_rope(q, cos, sin)` directly without instantiating the attention block — but the cos/sin cache construction is a private instance method. Cleanest path: lift the ~10 lines of cache-building into a small helper, or duplicate it inline in the new block (the full frequency-table build is ~5 lines).
- **ReLU²:** NOT FOUND. Grep for `ReLU2|relu_squared|square.*relu` returned no matches. Must be implemented — trivial: `F.relu(x).square()`.
- **Flash-Attention (SDPA):** `F.scaled_dot_product_attention` already used at `baselines.py:77` and `model.py:416`. No `flash_attn` import anywhere — we ride PyTorch's SDPA backend selector. Reusable as-is.
- **QK-norm:** NOT FOUND. No `QKNorm`, `qk_norm`, or equivalent. Must be implemented — trivial: two `RMSNorm(head_dim)` applied to Q and K after RoPE and before SDPA. (Per modded-NanoGPT convention, RMSNorm per-head on the head_dim axis.)

### Gap vs Exp 21 design

`SimpleTransformerLM` gets the norm choice (RMSNorm), the tying choice (untied), and the attention backend (SDPA → Flash) right, but misses three load-bearing pieces of the modded-NanoGPT lean spec: no RoPE, SiLU instead of ReLU², and no QK-norm. The FF module is shared with the SSM arm via `core.FeedForward`, which means swapping in ReLU² either needs a new FF module or a parametrization on the existing one — the former is cheaper and more honest. The attention block also can't host QK-norm or RoPE without new plumbing. So roughly 60% of what's needed is primitives we already own (RMSNorm, untied embed/head, SDPA, RoPE math as a static method), and 40% is new glue code (ReLU² FF, QK-norm module, a new block that sews RoPE + QK-norm onto SDPA).

### Recommendation: **New variant**

Register a new class `NanoGPTLeanLM` (with a `build_nanogpt_lean(...)` factory) in `baselines.py` rather than adapter-forking the existing `SimpleTransformerLM`. Rationale: (1) Three of the five differences (RoPE, ReLU², QK-norm) each need a new submodule; an "adapter" would have to rewrite `CausalSelfAttention` and `FeedForward` anyway, so we'd be adding a shadow class under the guise of a flag. (2) `core.FeedForward` is shared with the SSM arm — changing its activation via a flag bleeds into Exp 18/19 behavior and invites silent regressions. (3) The existing `SimpleTransformerLM` has real callers (duck-typed to `CareStudentLM` via `outer_model`/`wernicke`/`semantic_tier` attrs at `baselines.py:22-25`) and we should not perturb them. Keeping the new variant separate preserves both contracts.

### Plan for Task 11

- [ ] Add `class ReLUSquaredFeedForward(nn.Module)` in `baselines.py` — fc → `F.relu(x).square()` → proj, bias=False, matching `core.FeedForward`'s shape contract. (~10 LOC)
- [ ] Add `class QKNorm(nn.Module)` in `baselines.py` — wraps one `RMSNorm(head_dim)` applied to the per-head last-axis of Q and K separately (two instances held by the attention block). (~10 LOC)
- [ ] Add `class NanoGPTLeanAttention(nn.Module)` in `baselines.py` — fused QKV, RoPE cos/sin cache (lift the ~5-line freq-table build from `model.py:339-352`; call `AttentionControlBlock.apply_rope` as a staticmethod for the rotation), QK-norm, `F.scaled_dot_product_attention(is_causal=True)`, out_proj, bias=False throughout. (~50 LOC)
- [ ] Add `class NanoGPTLeanBlock(nn.Module)` — pre-norm + residual pattern identical to `TransformerBlock`, but wiring `NanoGPTLeanAttention` and `ReLUSquaredFeedForward`. (~15 LOC)
- [ ] Add `class NanoGPTLeanLM(nn.Module)` — `Embedding(V, d)`, 8× `NanoGPTLeanBlock`, final `RMSNorm`, untied `Linear(d, V, bias=False)`, same forward signature as `SimpleTransformerLM` (returns `{"logits", "hidden", optionally "jacobian_stats"}`), same duck-typing attrs (`outer_model=None`, `wernicke=None`, `wernicke_balance_weight=0.0`, `semantic_tier=None`), same `artifact_bytes` method. (~40 LOC)
- [ ] Add factory `def build_nanogpt_lean(vocab_size=8192, d_model=256, n_head=4, n_layer=8, ffn_mult=4) -> NanoGPTLeanLM` exposing the design hyperparams. (~5 LOC)
- [ ] Write `tests/unit/test_nanogpt_lean_config.py` per plan Task 11 Step 1: param count ∈ [10.30M, 10.70M] at V=8192, forward shape `(2, 64, 8192)`. Add a third test that asserts `model.embed.weight is not model.lm_head.weight` (untied sanity) and one that asserts the three new submodules are actually wired (`isinstance(model.layers[0].ff, ReLUSquaredFeedForward)` etc. — wiring vs existence, per project memory).
- [ ] Register `transformer_nanogpt_lean` in the runner's `arch` dispatch (will show up in Phase 5 config).
