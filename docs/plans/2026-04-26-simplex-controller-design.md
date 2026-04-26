# Simplex CPU SSM Controller — Design

## Thesis

The cache is the global memory substrate. Each query induces a local 16-vertex
simplex from the top-K most relevant cache slots. The CPU SSM controller is
not a per-slot regressor — it is a barycentric policy over the simplex.
Replay credit updates the policy in policy-gradient form, moving probability
mass toward vertices that paid CE-delta dividends and away from ones that
didn't.

> *Operational simplex.* Top-16 slots are not guaranteed affinely independent;
> we don't enforce geometry, we exploit it. Vertex/edge/simplex features are
> what give the metaphor mechanism.

## Mechanism

Per query `q`:

1. Heuristic retrieval returns 16 candidate slot ids.
2. Build the candidate matrix `V[16, K_v]` from per-slot vertex features and
   the edge tensor `E[16, 16]` from pairwise cosine.
3. Forward: `V, E, simplex_features → logits[16] → p = softmax(logits / T)`.
4. Action: argmax for greedy, sample from `p` for exploration. Store
   `chosen_idx`, `p[chosen]`, and the full `V, E` snapshot.
5. On replay outcome: `advantage = ce_delta_raw - bucket_baseline`. Loss is
   `-advantage * log p[chosen]`. SGD updates the policy parameters; fast/slow
   EMA blend stays.

The shift from per-slot regression to choice-set policy is what makes the
AMX `M=16` shape natural: 16 vertices = one A-tile = one `_tile_dpbf16ps`
per layer.

## Feature inventory

**Vertex features** (per slot, K_v ≈ 16):
- `utility` — slot.last_seen_utility (scalar)
- `age_steps` — global_step - slot.write_step
- `bucket_id_embed` — small embedding lookup over coarse bucket id (8-dim)
- `cosine_q_slot` — cosine(query_residual, slot.key_rep)
- `pressure_at_write` — recorded at admission
- `slot_norm` — slot.key_rep.norm() / sqrt(D)
- `replay_count` — times this slot has been replayed (saturating)
- `recency_decay` — γ^(global_step - slot.write_step)

**Edge features** (per pair, K_e = 1):
- `cosine_ij` — cosine(slot_i.key_rep, slot_j.key_rep)
- *(diagonal i=i set to 1; symmetric matrix)*

**Simplex features** (global, K_s ≈ 4):
- `top1_utility`
- `mean_utility`
- `cosine_q_top1` — sharpness of best match
- `cosine_q_spread` — max - min over candidate cosines (degeneracy proxy)

Edge and simplex features fall out of the heuristic retrieval pass for free —
the cosine matrix is computed during top-K selection anyway.

## Forward architecture

Three layers, `H = 32`:

```
# Layer 1 — vertex projection (per-vertex linear, no mixing)
vertex_h = V @ W_vp + b_vp      # [16, K_v] @ [K_v, H] -> [16, H]
vertex_h = gelu(vertex_h)

# Layer 2 — edge-aware mixing (set-level)
attn_logits[i, j] = (vertex_h[i] · vertex_h[j]) / sqrt(H) + alpha * E[i, j]
attn[i, j] = softmax_j(attn_logits[i, j])
mixed_h[i] = sum_j(attn[i, j] * vertex_h[j])
mixed_h = mixed_h + vertex_h    # residual

# Layer 3 — logit head (with simplex bias)
logits = mixed_h @ W_lh + b_lh                          # [16, H] @ [H, 1] -> [16]
simplex_bias = simplex_features @ W_sb                  # [K_s] @ [K_s, 1] -> [1]
p = softmax((logits + simplex_bias) / T)                # [16]
```

The single learned scalar `alpha` mixes content (`vertex_h dot product`) with
geometry (`E[i, j]`) in the attention scores. This is the simplest spec that
preserves both the SSM-native vertex transformation and the set-level
information flow that per-slot scoring lacks.

`T` is a fixed temperature (no per-step learned schedule for V1). `simplex_bias`
is a scalar added to all logits — affects entropy, not ranking. Both can graduate
to per-vertex once V1 is bake-validated.

## AMX shape utilization

Per-query forward in three GEMMs:

1. `V @ W_vp`: `(16, K_v) @ (K_v, H) = (16, 32)` — one A-tile, K=16 needs one
   K-tile, N=32 needs two N-tiles → 2 `_tile_dpbf16ps`.
2. `vertex_h @ vertex_h.T`: `(16, H) @ (H, 16) = (16, 16)` — single tile fit.
3. `mixed_h @ W_lh`: `(16, H) @ (H, 1) = (16, 1)` — N=1 wastes the N axis;
   `_tile_dpbf16ps` still works but underused. Acceptable; the head is small.

Total: ~4-5 `_tile_dpbf16ps` per query. The tile is *used* — none of
this is M=1 simulating M=16.

## Backward — REINFORCE

```
# At decision time, store:
snapshot = {
    "V": [16, K_v], "E": [16, 16], "simplex_features": [K_s],
    "chosen_idx": int, "p_chosen_decision": p[chosen],
    "step": gpu_step, "slot_id": chosen_slot_id,
}

# At replay outcome (matched on chosen_slot_id + step proximity):
advantage = ce_delta_raw - bucket_baseline                   # scalar
# Optional importance ratio for off-policy correction:
p_chosen_now = run_forward_again(snapshot)[chosen_idx]
ratio = clip(p_chosen_now / p_chosen_decision, 0.5, 2.0)     # PPO-style clip

loss = -advantage * ratio * log p_chosen_now
# Standard backprop through the three layers.
```

For V1 we run REINFORCE without the importance ratio — the policy doesn't
shift much in 600s and the advantage signal is dominant. The ratio plumbing
is reserved for V2.

The advantage shaping (Gerber concordance, recency decay) carries over from
the per-slot design. Same wire signal, same baseline math; what changes is
the consumer: previously a scalar regression target, now a policy-gradient
multiplier.

## What survives, what dies

**Survives** (no change or pure adaptation):

- Tiled AMX BF16 matmul (commit `dcda08b`). Per-query forward fits the tile.
- AVX-512 matvec/axpy (commit `d208cdc`). Used for the small head/bias ops.
- Wire-event ring infrastructure (Phases A + B). Producers stay; the
  payload schema gets new fields.
- Heuristic candidate generator. Becomes the simplex-vertex selector,
  not the final scorer.
- Bucket baseline, recency decay, Gerber concordance — repurposed as
  advantage shaping (multiplicative on the policy gradient) rather than
  credit attribution to a single slot.
- `episodic_writer` admission path. The cache write side is unchanged.

**Dies or rewrites**:

- `OnlineLearningController::accumulate_backward` (commit `40094bb`).
  Per-slot regression → REINFORCE over the simplex. Forward graph is
  different; backward is policy-gradient.
- `OnlineLearningWeights` shape. Per-slot `(global, slot, feature)`
  projections → simplex `(W_vp, W_lh, W_sb, alpha, T, embeddings)` head.
- `record_replay_selection` payload. Single-slot snapshot →
  whole-simplex snapshot.
- `ActionHistoryEntry` schema. New fields: `V`, `E`, `simplex_features`,
  `chosen_idx`, `p_chosen_decision`.
- F1 matrix arms. Frozen-vs-online stays. Cold-vs-warm is more meaningful
  with simplex (warm cache = richer simplex).

## Wire-event schema delta

The QueryEvent currently carries `query_event_id`, the residual fingerprint,
and one selected slot. The simplex needs the candidate set. Two options:

**Option A — extend QueryEvent.** Add `candidate_slot_ids[16]` (16×u64 = 128B)
and the chosen one is the first id (or recorded separately). Backward-compatible
to producers that emit one slot (zero-pad the rest with sentinel).

**Option B — new SimplexQueryEvent.** Clean separation. Old QueryEvent stays
for Pass C heuristic-only diagnostics; SimplexQueryEvent is the policy path.

V1 uses Option A — adding fields to QueryEvent under the existing
`#pragma pack` keeps the wire path simple and lets the controller dispatch
on whether `candidate_slot_ids[1]` is sentinel (heuristic-only) or populated
(simplex). Net struct growth ~128B; QueryEvent goes from 544B to 672B.

ReplayOutcome stays unchanged — it carries the slot id of the chosen vertex
and the CE-delta reward; the controller looks up the matching candidate
snapshot by `(slot_id, gpu_step)`.

## Pretrain pipeline

Behavior cloning from heuristic decisions. The heuristic argmax is the
target; the controller is trained to softmax-match it given
(V, E, simplex_features).

Loss: cross-entropy `target = heuristic_argmax_idx`, `pred = log p`.

This bootstraps the policy near the heuristic so on-pod online learning
starts from a reasonable prior rather than uniform-random over 16 vertices.

Value head from the per-slot pretrain is dropped — the simplex policy
doesn't need a value baseline (advantage = reward - bucket_baseline,
which is per-event, not per-state).

## F1 matrix arms (revised)

| arm | episodic | controller | TTT | candidate scorer |
|---|---|---|---|---|
| `arm_a_control` | off | n/a | n/a | n/a |
| `arm_b_heuristic` | on | heuristic-only | n/a | argmax(utility) |
| `arm_c_simplex_frozen` | on | trained-simplex | off | `softmax(simplex_logits)` |
| `arm_d_simplex_online` | on | trained-simplex | on | `softmax(simplex_logits)` |
| `arm_e_simplex_warm_online` | on | trained-simplex (ckpt-warm) | on | `softmax(simplex_logits)` |

Five arms × 3 seeds = 15 cells. The cold-vs-warm pair becomes a real
contrast because the simplex's edge structure changes when the cache is
warm-loaded vs fresh.

## Open questions deferred to V2

- Importance ratio in the policy gradient (off-policy correction). V1 is
  on-policy REINFORCE because the policy moves slowly within a 600s
  window.
- Diversity loss term. Useful when the cache surface is degenerate;
  simplex_features carries the signal but V1 doesn't penalize it.
- Per-vertex temperature head. V1 uses a fixed `T`; learning a
  per-simplex `T` from `simplex_features` is V2.
- Multi-head attention in Layer 2. Single `alpha`-blend is the V1
  simplest spec; multi-head would let the controller learn separate
  content/geometry weightings.

## Artifact budget

Total controller parameter count, V1:

| param | shape | floats |
|---|---|---|
| W_vp | (K_v=16, H=32) | 512 |
| b_vp | (32,) | 32 |
| W_lh | (32, 1) | 32 |
| b_lh | (1,) | 1 |
| W_sb | (K_s=4, 1) | 4 |
| alpha | (1,) | 1 |
| bucket_embed | (8 buckets, 8 dim) | 64 |
| **total** | | **~650 fp16 = ~1.3 KB** |

Negligible against the 16 MB artifact cap. The cache dominates.

## SSM-native claim

The vertex transformation in Layer 1 is per-vertex linear + GeLU — the same
shape as one SSM block applied independently to each of the 16 "tokens"
(treating the simplex as a length-16 sequence with diagonal A reduced to
identity since there's no temporal axis within a single query).

Layer 2 (edge-aware mixing) is the cross-token interaction step that pure
SSMs lack and pure attention provides. With N=16 the attention is cheap
(`O(N²) = 256` dot products), but more importantly the `alpha * E[i, j]`
term injects the geometric prior the heuristic retrieval already computed.
This is "SSM trunk + lightweight set-level attention" — the same shape
appearing in modern hybrids (Jamba, Mamba-Hybrid), now applied to a
16-token simplex instead of a long sequence.

The thesis: an SSM controller with explicit choice-set features over a
small candidate simplex is the natural shape for cache routing. Per-slot
scoring is strictly weaker because it can't represent "vertices 2 and 3
are near-duplicates, weight one not both."
