# Codex Architecture Review Part 2: Controller as Distillation Target

## Core Conceptual Correction

The controller is NOT unnecessary. It should be tightly involved in the model path. But it must not define the utility label.

### The Clean Split

```
rank-3 oracle:
    asks: "If memory were forcibly injected here, would it help this token?"
    bypasses controller
    computes dense signed utility

controller:
    asks: "Can I cheaply predict that utility before seeing the target?"
    gates memory injection into the semantic stream

training ranks:
    use oracle utility as:
        1. token loss weights (mild, positive-only initially)
        2. controller distillation targets
```

## Critical Correction: encode() memory_mode

The exact teacher should compare semantic stream with memory disabled vs memory forcibly injected at the REAL injection point. NOT just `hidden + memory_delta` after the fact.

### New encode() API

```python
def encode(self, input_ids, *, memory_mode="controller", teacher_gate=None,
           return_controller_logits=False, return_memory_meta=False):
    """
    memory_mode:
        "off"          -> no memory read/injection
        "force_on"     -> read memory and inject with gate = 1
        "controller"   -> controller decides gate
        "teacher_gate" -> use externally supplied gate
    """
```

## Controller Design

Continuous gate, not binary:

```python
p_help = torch.sigmoid(controller_logits)
gate = torch.clamp((p_help - 0.5) * 2.0, 0.0, 1.0)
x = x + gate[..., None] * mem_delta
```

- p_help <= 0.50 → no memory injection
- p_help = 0.75 → half-strength
- p_help = 1.00 → full injection
- Never use negative gate (anti-memory is a different mechanism)

## Rank-3 Teacher Scoring

```python
@torch.inference_mode()
def score_memory_teacher(model, input_ids, valid_mask, *, head_chunk_size=128):
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    
    h_off = model.encode(x, memory_mode="off")
    h_mem = model.encode(x, memory_mode="force_on")
    
    nll_off = chunked_nll(model, h_off, y, head_chunk_size)
    nll_mem = chunked_nll(model, h_mem, y, head_chunk_size)
    
    utility = (nll_off - nll_mem) * mask.float()
    
    return {
        "loss_weights": utility_to_loss_weight(utility, mask),
        "controller_targets": sigmoid(utility / tau),  # dense calibrated signal
        "utility": utility,
    }
```

## Controller Target Transform

```python
def utility_to_controller_target(utility, *, tau=0.10, floor=0.05, ceil=0.95):
    target = torch.sigmoid(utility.float() / tau)
    return target.clamp(floor, ceil).detach()
```

Scale: utility +0.20 nats → target ≈ 0.88, 0.00 → 0.50, -0.20 → 0.12

## Training-Rank Loss

```python
lm_loss = weighted_chunked_lm_loss(hidden, targets, token_weights=loss_weights)
ctrl_loss = BCE(controller_logits, controller_targets)
loss = lm_loss + lambda_controller * ctrl_loss  # lambda = 0.01
```

## KEY INSIGHT: Don't Downweight Hard Tokens

Negative utility means "memory hurts here", NOT "this token is unimportant."

Use negative utility to CLOSE THE GATE, not to ignore the token.

First version: positive-only LM weighting:
```python
def positive_only_lm_weight(utility, valid_mask, *, tau=0.10, strength=0.15, w_max=1.20):
    pos = torch.relu(torch.tanh(utility.float() / tau))
    w = 1.0 + strength * pos
    w = w.clamp(1.0, w_max) * valid_mask.float()
    mean_w = w.sum().clamp_min(1e-6) / valid_mask.sum().clamp_min(1.0)
    return (w / mean_w.detach()).detach()
```

## Staged Implementation

### Stage 0: Shadow Mode
Rank 3 computes utility but ranks 0-2 ignore it. Measure: step time impact, utility distribution sanity, correlation with existing ce_delta_raw.

### Stage 1: Controller Distillation Only
`loss = lm_loss + 0.01 * ctrl_loss`. No LM reweighting. Test: can controller learn memory-help/hurt boundary?

### Stage 2: Controller-Gated Memory
Online model uses `gate = sigmoid(controller_logits)` to inject memory. Oracle still bypasses controller for labels.

### Stage 3: Mild Positive-Only LM Weighting
`alpha = 0.05` initially, maybe up to 0.15.

### Stage 4: Symmetric Weighting (only if proven)

## Critical Warnings

1. **Causal leakage**: Score batch BEFORE writing it to cache. Wrong order will look amazing and fail eval.
2. **Don't let controller define teacher labels**: force_on vs off, NOT controller-gated vs off.
3. **Stale teacher is OK for controller supervision, less OK for aggressive LM reweighting.**
4. **Rank 3 is opportunistic. Late labels are dropped. Training never waits.**

## One-Sentence Summary

The rank-3 oracle idea is wise; using it as controller supervision is the clean win; using it to aggressively reshape LM loss is the part that could send you into lala land.
