# Experiment 17: Local Attention Sidecar on Fast SP-SSM

## Status

Design approved. Pending implementation.

## Motivation

Experiment 16 proved three things:

1. The fast SP-SSM backbone is viable (1.63 bpb, ~14K steps in 600s
   with torch.compile on the diag recurrence).
2. Attention over recent tokens is extremely concentrated
   (effective_connections ~1.2 — almost all mass on 1-2 positions).
3. The residual stream carries the retrieval signal, not the
   recurrence state (x_only matches x_state, p=0.017; state_only
   is significantly worse, p=0.009).

The remaining gap to the matched transformer (1.59 bpb) may be
closeable with a small, cheap local attention path — not a second
backbone, but a surgical retrieval sidecar attached to the SSM's
strongest features.

### Framing

The SSM is the base model. It handles cheap sequential compression
of the token stream. Attention is the exception path — a tiny
retrieval module for the small fraction of cases where the
compressed state loses precision. The analogy is SSM-as-SLOT,
attention-as-LoRA: the trunk does most of the work, the sidecar
fills in what it can't.

The design question is not "how much transformer can we graft onto
the SSM?" but "how little attention can we add before the gap
closes?"

The hard constraint is preserving the SSM's advantages:

- High step throughput
- Small runtime state
- Small artifact
- Simple recurrent inference story

If the attention sidecar gets too big, we've rebuilt a worse
transformer. If it stays tiny and still helps, that's interesting.

## Architecture

4 blocks total. Blocks 1-3 are pure `ChaosSSMBlock`. Block 4 is
`ChaosSSMHybridBlock`:

```python
self.layers = nn.ModuleList([
    ChaosSSMBlock(...) for _ in range(num_layers - 1)
] + [
    ChaosSSMHybridBlock(
        ...,
        local_attn_window=64,
        local_attn_heads=1,
        local_attn_dim=64,
    )
])
```

### Hybrid Block Internals

```python
# Inside ChaosSSMHybridBlock.step():
normed = input_norm(x)
ssm_y, new_state = core.step(normed, state)
x_ssm = x + ssm_y

attn_out = local_attn(query=x_ssm, kv=kv_cache.last(w))
gate = sigmoid(gate_proj(x_ssm) + gate_bias)  # low-init

x_hybrid = x_ssm + gate * attn_out
x_out = x_hybrid + ff(ff_norm(x_hybrid))

kv_cache.write(k_proj(x_ssm), v_proj(x_ssm))
```

Attention is placed **after** the SSM update and **before** the FF.
The block is still fundamentally an SSM block. Attention is a sidecar
attached to the strongest features (post-SSM residual stream), not a
new stage sitting on top of the model.

### Why Top Block Only

- Matches Exp 16 result: useful retrieval signal lives in higher-level
  token features, so the top layer is the natural place to spend
  retrieval budget.
- Keeps the experiment clean: one hybrid block means one K/V writer,
  one query path, one gate, one small attention module.
- Preserves the "SSM is the base model" story.
- Avoids confounding retrieval capability with depth/parameter
  increases.

### Why Not a Separate Post-Stack Block

Adding a post-stack retrieval block (option B) confounds three things:
more depth, more parameters, and retrieval capability. If it wins,
you don't know which factor helped. Replacing the top SSM block with
a hybrid keeps depth and parameter count matched.

The question "should retrieval be a separate post-stack block?" is
valid but belongs in Exp 18, after Exp 17 establishes whether
retrieval helps at all.

## Design Rules

- Only 1 attention sidecar layer (top block)
- Dense local window first (no sparse/routing in Phase A)
- Window sizes w in {16, 32, 64}
- 1 attention head, attn_dim=64
- Query from x_ssm (not state_only — Exp 16 proved this is wrong)
- Low-init gate: `gate_bias` initialized to -4 so sigmoid(gate_bias)
  ~ 0.018, meaning the sidecar starts near-zero. Bare SSM is
  recoverable if attention hurts.
- Shared K/V projections from x_ssm, stored in a simple rolling buffer
- Measure bpb, steps/s, and artifact bytes together

## Parameter Budget

The sidecar adds per hybrid block:

| Component | Parameters |
|---|---|
| q_proj (dim -> attn_dim) | 256 × 64 = 16,384 |
| k_proj (dim -> attn_dim) | 16,384 |
| v_proj (dim -> attn_dim) | 16,384 |
| out_proj (attn_dim -> dim) | 16,384 |
| gate_proj (dim -> 1) | 256 |
| gate_bias | 1 |
| **Total** | **~65K params** |

Against the 6.5M baseline, this is 1% overhead. Well within
the 16MB artifact budget.

## Phase A Conditions

| Condition | window | attention | attn_dim | heads |
|---|---|---|---|---|
| bare_fast_ssm | — | none | — | — |
| local_w16 | 16 | dense | 64 | 1 |
| local_w32 | 32 | dense | 64 | 1 |
| local_w64 | 64 | dense | 64 | 1 |

All conditions:

- vocab_size=8192 (SP8192)
- model_dim=256
- num_layers=4
- a_mode=diag
- ff_mult=2
- seq_len=512
- batch_size=32
- budget=600s
- 7 seeds per condition (28 runs total)

`bare_fast_ssm` is the control — same model as Exp 15/16 but with the
compiled diag scan. This gives the honest ~14K-step baseline bpb.

## Go / No-Go

Phase A gates:

1. Any local_w condition beats bare_fast_ssm by >= 0.02 bpb
   (statistically significant improvement from retrieval).
2. The winning condition's steps/s >= 50% of bare_fast_ssm steps/s
   (attention overhead is acceptable within the 600s budget).
3. Artifact bytes stays under 16MB.

All three must pass for Phase B.

## Phase B (Contingent on Phase A Pass)

Only if dense local window helps >= 0.02 bpb:

- Sparse top-k within window (k=4, k=8) using x_ssm features
- Compare dense vs sparse at the winning window size
- Measure throughput: sparse should be cheaper than dense if
  window is large

## What This Does Not Test

- Multi-layer attention (Exp 18 question)
- Placement: top-2, every layer, post-stack (Exp 18 question)
- Recurrence-state routing (falsified in Exp 16)
- Non-local retrieval (beyond the window)

## Research Sequence

```
Exp 17: Does tiny local retrieval help?
Exp 18: Where should tiny local retrieval go?
```

Exp 18 only runs if Exp 17 passes. Then it compares:

- Top-block hybrid (Exp 17 winner)
- Top-2 hybrid
- Post-stack retrieval block

## Files

```text
experiments/17_local_attn_sidecar/
  DESIGN.md -> symlink or copy of this doc
  runner_exp17.py
  run_exp17.py
  test_exp17.py
```

The `ChaosSSMHybridBlock` lives in `src/chaoscontrol/model.py`
alongside `ChaosSSMBlock`. It inherits or wraps `ChaosSSMBlock`
and adds the attention sidecar.
