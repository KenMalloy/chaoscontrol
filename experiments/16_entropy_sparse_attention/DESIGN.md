# Experiment 16: Entropy-Guided Sparse Attention via SSM Oracle

## Status

Design and scaffold phase. Phase A probe runner implemented.

## Motivation

Experiment 15 established the decision boundary for this pivot:

- SP8192 + SSM is much better than byte-level SSM.
- SP8192 + SSM is still materially behind the matched transformer.
- The repo already recorded the recommended next move as a depth-recurrence
  or attention-hybrid pivot.

The goal of Experiment 16 is to test whether the SSM state can cheaply
identify a very small set of useful past tokens, so that we can spend exact
attention only on those candidates instead of scanning all prior positions.

The goal is not novelty for novelty's sake. The practical goal is to learn
whether recurrent state can make transformers less necessary for this regime:
if an SSM can cheaply recover most of the benefit of attention with only a
very small number of exact lookups, then the transformer's full dense
attention path becomes less compelling as the default answer.

## Core Hypothesis

If the dense attention distribution over a bounded rolling buffer is naturally
concentrated, then:

```text
effective_connections = exp(H(attn_row))
```

may stay near-constant even as the buffer grows.

If a small oracle fed by the SSM state can recover most of the dense
attention mass with only `k` selected positions, then a sparse
SSM-attention hybrid may close some of the SSM-transformer gap without
paying full quadratic attention cost.

## Literature-Grounded Framing

This direction is no longer "empty literature space."

Several adjacent lines already support parts of the hypothesis:

- Top-k attention mass capture is often strong in practice.
- Cheap candidate generation followed by exact rescoring is a recurring
  winning pattern.
- Entropy is useful as a diagnostic and sometimes as a routing signal.
- SSM-attention hybrids now exist, including entropy-gated ones.

So the question is not:

```text
Is anyone doing anything vaguely like this?
```

The useful question is:

```text
Does recurrent state add selector value beyond simpler sparse-selection
baselines such as recency, heavy hitters, or low-rank attention-side scoring?
```

That is the standard this experiment should be held to.

## Important Design Correction

The first draft proposed a linear head from SSM state to buffer slot logits:

```python
scores = h_t @ W_pos  # (D -> buffer_size)
```

This experiment does **not** use that design in the scaffold.

Why:

- It learns slot-index priors more easily than content retrieval.
- It does not generalize cleanly across buffer sizes.
- It makes the selector mostly about "how old is the useful token?"
  instead of "which past representation matches the current state?"

The scaffold instead uses a **content-addressed selector**:

```python
z_t = concat(query_features_t, state_t)
q_sel = W_q(z_t)
k_mem = W_k(memory_vectors)
scores = q_sel @ k_mem.T / sqrt(d_k)
top_idx = topk(scores, k)
```

This keeps the oracle tied to content similarity rather than slot identity.

## What We Should Learn From Adjacent Successes

The experiment should explicitly absorb the most useful lessons from nearby
work instead of treating them only as citation context.

### 1. Sparse retrieval is probably the easy part

Many adjacent results suggest that attention mass is often concentrated and
that top-k retrieval can work surprisingly well. This means the risky part of
the experiment is not "is sparsity real?" but rather:

```text
Can SSM-derived features identify the right sparse set better or cheaper than
simpler selectors?
```

### 2. The selector is the experiment

Sparse kernels and routing tricks only matter if the candidate set is good.
This is why the scaffold starts with a frozen probe rather than end-to-end
integration. If the selector cannot recover useful mass in a cheap probe,
there is no reason to spend more engineering effort on the full architecture.

### 3. Cheap candidate generation plus exact rescoring is the right shape

The selector should narrow the field, not replace attention entirely.

```python
candidates = cheap_selector(query_like_features, memory)
top_idx = topk(candidates, k_probe)
attn_out = exact_attention(query, memory[top_idx])
```

This is a much more realistic design target than asking a tiny selector head
to solve the entire retrieval problem alone.

### 4. Entropy is mostly a measurement and gating tool

Entropy should help answer:

- Is the dense proxy concentrated?
- Is attention worth invoking at all?
- Does a sparse policy recover most of the useful mass?

It should not be treated as a magical objective that makes sparsity work by
itself.

### 5. Overhead is the real enemy

The most common failure mode in sparse-attention work is selector overhead:
the model saves compute in the attention path and then gives it back in the
routing path. Exp 16 only succeeds if the selector is genuinely cheaper than
the attention it replaces.

## Phase A Scope in the Scaffold

The current runner implements a **frozen probe** version of Phase A:

1. Train the Exp 15-style SP8192 SSM backbone.
2. Freeze the backbone.
3. Replay validation windows token-by-token.
4. Collect per-position oracle examples from a chosen layer.
5. Define a dense proxy attention distribution over a rolling buffer using
   dot-product similarity in the collected feature space.
6. Train a small content-addressed selector to recover that dense proxy.

This is intentionally narrower than the full end-to-end sparse-attention
 architecture. It is the cheapest way to answer:

- Is the dense proxy concentrated?
- Does the recurrent state preserve enough information to recover
  high-mass targets?
- Does the selector beat a trivial recent-token baseline?

The scaffold does **not** yet report `bpb_dense` or `bpb_oracle`, because
that requires integrating learned attention outputs back into the LM path.

That is deliberate. A failed probe is still useful if it clearly tells us
which selector families dominate and whether recurrent state is actually
helpful.

## Oracle Features

The probe is configurable, but the default and recommended feature source is:

```python
query_source = "x_state"
write_source = "x_state"
```

where:

- `x` is the post-SSM residual stream at the chosen layer.
- `state` is the recurrence state returned by `ChaosSSMCore.step()`.

Using only `state` is a plausible ablation, but not the default. If the
failure mode is that the SSM compressed away lexical detail, then the oracle
should not be forced to use only the most compressed representation.

## Probe Metrics

The scaffold logs:

- `selector_recall_at_k`
- `selector_mass_capture_at_k`
- `recent_recall_at_k`
- `recent_mass_capture_at_k`
- `token_keyed_recall_at_k`
- `token_keyed_mass_capture_at_k`
- `full_attn_entropy`
- `effective_connections`
- `top1_mass`
- `oracle_entropy`
- `bare_eval_bpb`

`mass_capture_at_k` is as important as recall. Recovering the same indices is
less meaningful if they carry very little probability mass.

## Adjacent Baselines and Failure Tests

Phase A should not stop at "beats recent-k." That is too weak a bar.

The point of this phase is to discover where the selector value actually
lives. If the SSM-state oracle fails but an attention-side cheap selector
wins, that is still a strong research result because it narrows the real
source of leverage.

The baseline set should evolve toward:

| Baseline / variant | What it teaches |
|---|---|
| `recent_k` | Whether the task is mostly local recency |
| `token_keyed` | Whether stable token-identity lookup is enough (Exp 14 rehab) |
| `heavy_hitter` | Whether simple persistent-salience heuristics are enough |
| `low_rank_key_ranker` | Whether cheap attention-side ranking already solves the problem |
| `x_only` | Whether post-SSM features are enough without recurrence state |
| `state_only` | Whether the recurrent state alone carries enough retrieval signal |
| `x_state` | Whether combining residual features with state is best |

The `token_keyed` baseline is implemented in the probe scaffold. It selects
buffer positions whose stored token ID matches the current query token. If
more than k matches exist, the most recent k are used. If zero matches
exist, recall and mass capture are both 0. This is the simplest possible
retrieval mechanism: pure identity lookup with no learned parameters.

This baseline re-examines Experiment 14's typed-buffer hypothesis. Exp 14
tried to route byte-level inputs into semantic buckets via Wernicke MoE,
producing unstable keys that hurt performance. With SP8192, the tokenizer
provides stable lexical identities for free. If `token_keyed` captures
significant mass, it means Exp 14's idea was sound but its keys were wrong.

Interpretation:

- If `recent_k` wins, the oracle is unnecessary.
- If `token_keyed` wins, stable identity is enough and the learned oracle
  adds no value beyond what the tokenizer already provides.
- If `low_rank_key_ranker` wins, the SSM is not adding selector value.
- If `x_only` wins, this is really an attention-side selector problem.
- If `state_only` wins, that is the strongest version of the original
  hypothesis.
- If `x_state` wins, the best story is that recurrence and residual features
  are complementary.

This makes the phase informative even if the initial hypothesis only partly
survives contact with data.

## Phase A Conditions

The matrix launcher runs six conditions: four sweep buffer size × k at the
default `x_state` feature source, and two ablate the feature source at the
largest operating point to answer the central question.

| Condition | buffer_size | k | query/write source |
|---|---:|---:|---|
| `oracle_buf64_k4` | 64 | 4 | x_state |
| `oracle_buf64_k8` | 64 | 8 | x_state |
| `oracle_buf128_k4` | 128 | 4 | x_state |
| `oracle_buf128_k8` | 128 | 8 | x_state |
| `oracle_buf128_k8_xonly` | 128 | 8 | x (post-SSM residual only) |
| `oracle_buf128_k8_stateonly` | 128 | 8 | state (recurrence only) |

The `x_only` vs `state_only` vs `x_state` comparison at matched buf/k
directly answers whether recurrent state adds selector value beyond
non-SSM features.

All conditions inherit the Exp 15 winner backbone:

- `model_type=ssm`
- `vocab_size=8192`
- `model_dim=256`
- `num_layers=4`
- `ff_mult=2`
- `a_mode=diag`
- `crit_target_coupling=0.92`

## Go / No-Go for the Scaffolded Probe

Because the current runner measures oracle quality rather than LM lift, the
Phase A scaffold uses probe-level criteria:

1. `selector_mass_capture_at_k >= 0.60` for at least one condition
2. `selector_mass_capture_at_k > recent_mass_capture_at_k` (selector beats
   recency baseline)
3. `selector_mass_capture_at_k > token_keyed_mass_capture_at_k` (selector
   beats token-identity baseline)
4. `effective_connections <= 2 * k` for at least one promising condition

The summary logic reports per-seed paired deltas (`selector - recent_k`,
`selector - token_keyed`) with significance tests, not just raw mass
capture rankings across conditions.

These are provisional gate criteria for the probe stage. Once the sparse
attention path is integrated into the model, we can reinstate the original
LM-level gates involving `bpb_dense` and `bpb_oracle`.

The stronger long-term criterion is not merely "works at all," but:

```text
Does the recurrent-state selector beat the strongest cheap non-SSM selector?
```

That is the threshold for making transformers less interesting rather than
just adding another complicated hybrid.

## Planned Phase B

If the probe is promising, the next implementation step is:

1. Add a per-layer rolling buffer inside a dedicated sparse-attention block.
2. Use the same content-addressed selector online during training.
3. Compute exact attention over the selected top-k memory vectors.
4. Mix sparse attention output back into the residual stream with a low-init,
   learned gate.

The expected block shape is:

```python
normed = input_norm(x_t)
ssm_y, new_state = core.step(normed, state)
x_ssm = x_t + ssm_y

z_t = build_query_features(normed, x_ssm, new_state)
idx = selector.topk(z_t, buffer.keys, k)
attn_out = exact_attention(z_t, buffer[idx])
gate = sigmoid(gate_proj(z_t) + gate_bias)

x = x_ssm + gate * attn_out
x = x + ff(ff_norm(x))
buffer.write(current_memory_vector)
```

## Files

```text
experiments/16_entropy_sparse_attention/
  DESIGN.md
  runner_exp16.py
  run_exp16.py
  test_exp16.py
```

No shared `src/` changes are required for the scaffold phase.
