# Experiment 16 Review Brief

## Title

**Entropy-Guided Sparse Attention via SSM Oracle**

## Purpose

This brief is meant for independent research and engineering review.

The goal is to evaluate whether this experiment is:

1. scientifically well-posed,
2. differentiated from adjacent work in a defensible way,
3. likely to teach us something useful even if it fails, and
4. structured so that throughput engineering improvements are not confused
   with architectural gains.

## Executive Summary

Experiment 15 established a clear bottleneck:

- SP8192 + SSM is materially better than byte-level SSM.
- SP8192 + SSM is still materially behind a matched transformer.
- The repo's explicit follow-on recommendation was a depth-recurrence or
  attention-hybrid pivot.

Experiment 16 tests one specific version of that pivot:

> Can the recurrent state of an SSM serve as a cheap, stable,
> content-addressed oracle that identifies a very small set of useful past
> tokens, allowing exact attention to run only on those candidates instead of
> across the full buffer?

The experiment should **not** be read as a claim that "nobody is doing
anything like this." Adjacent and increasingly close work already exists,
including:

- dynamic sparse attention,
- top-k attention / heavy-hitter retention,
- cheap candidate generation plus exact rescoring,
- SSM-attention hybrids with entropy-based routing.

The scientific question is narrower and more defensible:

> Does recurrent state add selector value beyond simpler sparse-selection
> baselines such as recency, heavy hitters, or low-rank attention-side
> ranking?

That is the standard this experiment should be judged against.

## Current Repo Status

Experiment 16 currently has a self-contained scaffold:

- [DESIGN.md](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/experiments/16_entropy_sparse_attention/DESIGN.md)
- [runner_exp16.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/experiments/16_entropy_sparse_attention/runner_exp16.py)
- [run_exp16.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/experiments/16_entropy_sparse_attention/run_exp16.py)
- [test_exp16.py](/Users/kennethmalloy/Local%20Documents/Developer/chaoscontrol/experiments/16_entropy_sparse_attention/test_exp16.py)

The current implementation is a **Phase A frozen probe**, not an end-to-end
sparse-attention model:

1. train the Exp 15 style SP8192 SSM backbone,
2. freeze it,
3. replay validation windows token-by-token,
4. define a dense proxy attention distribution over a bounded rolling buffer,
5. train a small selector to recover high-mass targets from recurrent
   features.

This is intentionally conservative. It is designed to answer the selector
question before the project spends engineering effort on online sparse
attention.

## Why This Experiment Exists

The underlying practical ambition is not novelty. It is to make dense
transformer attention less necessary in this regime.

If the SSM can:

- maintain a strong fast recurrent path,
- cheaply identify a tiny set of useful retrieval candidates, and
- recover most of the benefit of dense attention with exact attention on only
  those few candidates,

then the transformer's full O(T^2) attention path becomes less compelling as
the default solution for this scale and budget.

## What Is New Here, Narrowly Defined

The narrow contribution under evaluation is not "SSM + attention" in general.

The narrow bet is:

1. use recurrent state plus lightweight features as the selector input,
2. use a content-addressed selector rather than slot-index logits,
3. judge success by mass recovery over a bounded buffer,
4. require the selector to beat cheap non-SSM baselines,
5. keep the initial phase probe-based and falsifiable.

The project should avoid broader novelty claims than that.

## Closest Adjacent Work

The following adjacent lessons matter more than novelty language:

### 1. Top-k mass capture is often real

This supports the premise that sparse retrieval can work at all. It does
**not** prove that recurrent state is the right selector.

### 2. Cheap candidate generation plus exact rescoring is a recurring pattern

This supports the architectural shape:

```python
candidates = cheap_selector(query_like_features, memory)
top_idx = topk(candidates, k)
attn_out = exact_attention(query, memory[top_idx])
```

### 3. Entropy is useful as a diagnostic and sometimes a routing signal

Entropy is worth using for:

- concentration diagnostics,
- "invoke attention or not" gating,
- go/no-go criteria for sparse retrieval.

Entropy alone should not be treated as proof that sparse attention will work.

### 4. SSM-attention hybrids already exist

This weakens any broad novelty claim and raises the bar: the experiment must
show why this particular selector hypothesis is useful relative to simpler or
already-known alternatives.

## Scientific Question

The central scientific question is:

> Does recurrent state provide incremental retrieval signal beyond cheap
> attention-side baselines?

This yields a more specific set of testable sub-questions:

1. Is the dense proxy attention over a bounded buffer naturally concentrated?
2. Can a recurrent-state-driven selector recover most of the dense mass?
3. Does it beat trivial and cheap baselines?
4. If it wins, is the win large enough to justify online integration?

## Baselines Required for a Meaningful Result

A result is not convincing if it only beats `recent_k`.

The intended baseline family is:

| Baseline / variant | Why it matters |
|---|---|
| `recent_k` | Tests whether the task is mostly local recency |
| `token_keyed` | Tests whether stable token-identity lookup is sufficient (Exp 14 rehab) |
| `heavy_hitter` | Tests whether persistent-salience heuristics are enough |
| `low_rank_key_ranker` | Tests whether cheap attention-side ranking already solves the problem |
| `x_only` | Tests whether post-SSM residual features are already enough |
| `state_only` | Tests whether recurrence state alone carries the signal |
| `x_state` | Tests whether recurrence and residual features are complementary |

The `token_keyed` baseline is implemented in the current scaffold. It selects
buffer entries matching the current token ID — no learned parameters, no
routing, pure identity. This directly re-tests Experiment 14's typed-buffer
hypothesis with stable SP8192 keys instead of noisy byte-level Wernicke
routing. If token identity alone captures significant attention mass, it
narrows the retrieval problem considerably.

Interpretation:

- If `recent_k` wins, the oracle is unnecessary.
- If `token_keyed` wins, the tokenizer already provides the retrieval signal
  and the learned selector adds no value.
- If `heavy_hitter` wins, simple heuristics may dominate.
- If `low_rank_key_ranker` wins, the SSM is not adding selector value.
- If `x_only` wins, this is mostly an attention-side selector problem.
- If `state_only` wins, that is the strongest version of the original
  hypothesis.
- If `x_state` wins, the best story is that recurrent state helps, but not in
  isolation.

This structure ensures that even a "failure" teaches us something actionable.

## Phase Structure

### Phase A: Frozen Probe

Goal:

- determine whether the selector hypothesis is worth online integration.

Current metrics:

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

What Phase A does **not** claim:

- no end-to-end sparse-attention lift yet,
- no `bpb_dense`,
- no `bpb_oracle`.

### Phase B: Online Sparse Attention

Contingent on a strong Phase A result.

Expected architecture:

```python
normed = input_norm(x_t)
ssm_y, new_state = core.step(normed, state)
x_ssm = x_t + ssm_y

selector_features = build_features(normed, x_ssm, new_state)
idx = selector.topk(selector_features, buffer.keys, k)
attn_out = exact_attention(selector_features, buffer[idx])
gate = sigmoid(gate_proj(selector_features) + gate_bias)

x = x_ssm + gate * attn_out
x = x + ff(ff_norm(x))
buffer.write(write_features)
```

### Phase C: Scaling and Concentration

Only after Phase B demonstrates value.

Goal:

- determine whether effective connections remain near-constant as the buffer
  grows, and whether the mechanism behaves like true sparse retrieval rather
  than a small-buffer artifact.

## Engineering Track vs Science Track

This experiment intersects a second question:

> Can the in-house `diag` SSM be executed much faster via a scan-style backend
> so that the same model gets more optimizer steps within a fixed time budget?

This must be treated as a **separate engineering track**.

### Engineering Track

Potential work:

- add a `diag_scan` backend for the existing `a_mode="diag"` recurrence,
- prove forward/backward equivalence to the current sequential implementation,
- measure throughput improvement in `steps/s` and `tokens/s`.

What this would mean scientifically:

- more training steps for the **same model** under the same time budget,
- not evidence of a better architecture.

### Science Track

The sparse-attention oracle hypothesis remains a distinct architectural
question. It should not be mixed with scan/throughput work in the same claim.

The clean reporting standard is:

- "execution refactor" for scan/backend work,
- "architectural improvement" only for selector/attention changes that beat
  matched baselines.

## Go / No-Go Standards

### For the current probe stage

Provisional criteria:

1. `selector_mass_capture_at_k >= 0.60` for at least one condition,
2. `selector_mass_capture_at_k > recent_mass_capture_at_k` (beats recency),
3. `selector_mass_capture_at_k > token_keyed_mass_capture_at_k` (beats
   token identity),
4. `effective_connections <= 2 * k`.

The summary reports paired per-seed deltas (`selector - recent_k`,
`selector - token_keyed`) with significance tests. The feature-source
ablation (`x_state` vs `x_only` vs `state_only` at buf128_k8) directly
tests whether recurrent state adds selector value.

### For the later online architecture

The stronger standard is:

> Does the recurrent-state selector beat the strongest cheap non-SSM selector
> strongly enough to justify the added architectural complexity?

That is the real threshold for making transformers less interesting rather
than just building another hybrid.

## Main Risks

| Risk | Meaning if observed |
|---|---|
| Selector loses to `recent_k` | Sparse retrieval may be mostly local |
| Selector loses to `token_keyed` | Token identity is sufficient; learned oracle adds no value |
| Selector loses to `low_rank_key_ranker` | The SSM adds little selector value |
| `state_only` underperforms badly | Recurrent state may be too lossy |
| Probe wins but online model regresses | Optimization/integration overhead dominates |
| Sparse path helps only at tiny buffers | Likely a small-buffer artifact |
| Scan backend changes quality at fixed steps | Execution refactor is not semantically equivalent |

## What An Independent Team Should Review

The most useful review questions are:

1. Is the scientific question stated narrowly and honestly enough?
2. Are the baseline families sufficient, or is a critical control missing?
3. Is the frozen-probe Phase A a good filter, or too disconnected from the
   eventual online architecture?
4. Is the `x_only` / `state_only` / `x_state` split the right way to localize
   where selector value comes from?
5. Is the engineering/science separation strong enough to prevent misleading
   conclusions?
6. If the goal is to make dense transformers less necessary, what empirical
   threshold would count as genuinely meaningful?

## Bottom Line

This is a good experiment if it is treated as:

- a narrow selector hypothesis,
- a baseline-heavy falsification exercise,
- a probe-first architecture decision,
- and a separate engineering/science split.

It is a weak experiment if it is treated as:

- a novelty claim,
- a sparse-attention victory lap before hard baselines,
- or a throughput refactor presented as architectural progress.

The current scaffold is intentionally pointed toward the stronger version.
