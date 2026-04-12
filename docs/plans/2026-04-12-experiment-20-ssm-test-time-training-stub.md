# Experiment 20 (stub): SSM-specific Test-Time Training

**Date:** 2026-04-12
**Status:** Stub — open-question capture only. Not yet a committed design.
**Precedes:** Exp 18 (throughput), Exp 19 (submission tuning). Exp 20 only
runs if Exp 19 ships a competitive but not yet winning submission and we
still have budget/time before April 30.

## Why this is its own experiment

The competition SOTA uses Legal TTT (test-time training) as a non-trivial
lever — on the order of 0.002-0.044 bpb depending on how aggressive it is
and whether it's legal under Issue #1017's "score-before-update" rule. It's
the final squeeze lever once training is maxed out.

Exp 19 originally listed "Legal TTT for SSM" as a generic ablation. While
scoping Exp 18/19 (2026-04-12), we realized TTT on an SSM is materially
different from TTT on a transformer, and it deserves its own experiment:

1. **SSM state is not static at eval.** Unlike a transformer (where weights
   and per-position hidden states are both static at eval), the SSM's
   recurrence state evolves every token. "Freezing the model and adding
   TTT on top" is not a clean concept — the model is *already* updating
   its own memory during the forward pass.

2. **Cumulative hidden states are bad retrieval keys.** Exp 16 already
   measured this empirically: `state_only` had significantly worse
   selector performance than `x_only` because the recurrence state is a
   cumulative summary of all past tokens, not a local per-position
   embedding. Two positions near each other have overlapping histories
   and therefore similar states, so "nearest neighbor in state space"
   mostly returns "most recent position." That's not useful as a retrieval
   signal.

3. **The right TTT substrate for an SSM is not obvious.** For a transformer,
   kNN-LM builds a cache over per-position hidden states and interpolates
   predictions. For an SSM, you need to pick a substrate that is:
   - Local enough that different positions give distinct keys
   - Semantically meaningful (not just raw token embeddings)
   - Stable across the eval pass (doesn't drift as the state evolves)
   - Legal under the competition's score-before-update rule

The right substrate is probably **residual stream features** (the post-SSM
output `x_ssm = x + ssm_out`), not the recurrence state itself. Exp 16
validated residual stream features as good retrieval keys. But it's a
design question that needs its own ablation.

## Open design questions

1. **Substrate:** What does the cache key space look like?
   - Residual stream at a specific layer (`x_ssm[layer_k]`)?
   - Concatenated per-layer residuals?
   - A dedicated learned TTT projection trained during the main run as a
     lightweight head?
   - The current token embedding directly (simplest, no context)?

2. **Mechanism:** How does the cache contribute to predictions?
   - kNN-LM interpolation: `p_final = λ·p_ssm + (1-λ)·p_knn`
   - Logit addition from retrieved targets
   - State initialization from retrieved past position
   - Continuous cache (Grave 2017) — bias toward recently-stored next tokens

3. **Gating:** When do you *use* the cache?
   - Always-on, trusted interpolation
   - Gated by current prediction entropy (high entropy → consult)
   - Gated by retrieval confidence (high similarity → trust the cache)
   - Combination

4. **Storage:** What do you put *in* the cache?
   - Every position (dense, large cache)
   - Only surprising positions (B'MOJO-style innovation selection)
   - Sliding window of recent positions

5. **SSM-state interaction:** Does the external cache help or fight with
   the SSM's own evolving memory?
   - If the state is already integrating the stream, the cache might be
     redundant. Test by comparing "state-only baseline" to "state + cache."
   - If they interact adversarially (the cache perturbs predictions in a
     way the state can't compensate for), we need to isolate why.

6. **Legality:** Does the mechanism satisfy Issue #1017's score-before-update?
   - The cache should only contain pairs from *prior* positions, never the
     current or future. Naturally respected by a one-way streaming cache.
   - No weight updates during eval. Trivially satisfied for nonparametric
     caches.

## What Exp 20 would NOT test

- Any *trainable* retrieval path. That's what Exp 09/14/17 already killed,
  backed by the 2026-04-12 lit review. Exp 20 is explicitly about
  nonparametric or minimally-parametric inference-time mechanisms.
- Weight-update TTT (fine-tuning SSM parameters on the val stream). This
  is the expensive, risky version of TTT that transformers get away with
  because attention weights don't need to be internally consistent across
  timesteps. For an SSM with ~14K training steps and a tiny val budget,
  the risk/reward is bad. Not worth testing.

## Dependency on Exp 19 outcome

Exp 20 only runs if:
1. Exp 19 ships a pure-SSM submission that is competitive (bpb <= 1.3,
   roughly — the point at which TTT's marginal gain could matter).
2. We still have non-critical pod budget and calendar time before April 30.
3. The test plan in Exp 20 can be scoped in <1 day of design + <1 day of
   implementation + <$100 of 8×H100 time.

If Exp 19 lands far from competitive, TTT is not the right lever and Exp 20
doesn't run — we commit whatever Exp 19 gives us and focus on the writeup.

## Relation to the 2026-04-12 research assistant report

The deep research report on retrieval/TTT literature (saved at
`project_retrieval_dead_2026-04-12.md`) noted that the *literature-supported*
"one last try" was specifically a nonparametric entropy-gated cache. That
matches what Exp 20 would explore — but applied at inference time (on the
fully-trained model) rather than during training (where the RA said it
wouldn't help). The RA's conclusion was "commit to pure SSM training"; it
didn't rule out the question of whether the same nonparametric mechanism
could earn its keep at TTT time on a model whose representations have
already converged. That's the Exp 20 question.

## What to write before Exp 20 runs

This stub is deliberately not a full design. Before Exp 20 actually
launches, we'd need:
- A concrete test plan with budget and gates (like Exp 18's revised design)
- A specific substrate choice (probably residual stream features, probably
  from the top block or a mid-stack layer)
- An interpolation rule and a method to tune the interpolation weight
- Measurement strategy (per-token loss on rare/tail tokens, not just mean bpb)
- Legal TTT compliance check: confirm it passes Issue #1017's score-before-update

For now, this stub exists only to preserve the open questions and prevent
them from getting rediscovered as "new ideas" later. Nothing to implement
yet.
