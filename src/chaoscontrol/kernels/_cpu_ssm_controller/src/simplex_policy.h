#pragma once

// Phase S1 — simplex policy forward kernel. Per-query forward over a 16-vertex
// memory-cache simplex (top-K candidate slots induced by retrieval). Computes
// `softmax(logits / T)` and saves the intermediate activations the S2
// REINFORCE backward will read. AMX dispatch is deliberately deferred — this
// translation unit calls `at::matmul` for the three GEMMs so correctness can
// be pinned on arm64 first; an SPR-only `chaoscontrol::amx::amx_bf16_matmul`
// fast-path lands in a follow-up commit once shapes/build flags align.

#include <cstdint>
#include <vector>

namespace chaoscontrol::simplex {

// Learned weights for one controller instance. All matrices are stored
// row-major as flat std::vector<float> so the same struct can serialize
// to/from CSWG without a torch dependency at the C++ layer. The shape
// metadata (K_v, K_e, K_s, H, N) drives every allocation; mismatched
// payload sizes are TORCH_CHECK'd at simplex_forward entry.
struct SimplexWeights {
  uint32_t K_v = 0;     // vertex feature dim (V1: 16)
  uint32_t K_e = 0;     // edge feature dim per pair (V1: 1)
  uint32_t K_s = 0;     // simplex feature dim (V1: 4)
  uint32_t H = 0;       // hidden dim (V1: 32)
  uint32_t N = 0;       // simplex size (V1: 16)
  uint32_t n_heads = 0; // optional HxH residual heads over the simplex

  std::vector<float> W_vp;          // (K_v, H) row-major — Layer 1 projection
  std::vector<float> b_vp;          // (H,)               — Layer 1 bias
  std::vector<float> W_lh;          // (H,)               — Layer 3 single-output head
  float b_lh = 0.0f;                //                    — Layer 3 scalar bias
  std::vector<float> W_sb;          // (K_s,)             — simplex_bias projection
  float alpha = 0.0f;               //                    — edge-vs-content mixing scalar
  float temperature = 1.0f;         //                    — softmax temperature
  std::vector<float> bucket_embed;  // (n_buckets, embed_dim) — stored, unused in V1 forward
  float lambda_hxh = 0.0f;          // residual scale for the HxH branch
  std::vector<float> W_q;           // (n_heads, H, H) — HxH query projections
  std::vector<float> W_k;           // (n_heads, H, H) — HxH key projections
  std::vector<float> W_v;           // (n_heads, H, H) — HxH value projections
  std::vector<float> W_o;           // (n_heads, H)    — per-head logit projections
  std::vector<float> W_e;           // (n_heads, K_e)  — per-head edge-feature bias
};

// Saved-for-backward intermediates. S2's REINFORCE backward consumes:
//   vertex_h  — post-GeLU output of Layer 1 (vertex projection)
//   mixed_h   — post-residual output of Layer 2 (edge-aware mixing)
//   attn      — softmax_j(attn_logits) used in Layer 2's row-wise mix
struct SimplexForwardOutput {
  std::vector<float> logits;     // (N,)
  std::vector<float> p;          // (N,) softmax((logits + simplex_bias) / T)
  std::vector<float> vertex_h;   // (N, H)
  std::vector<float> mixed_h;    // (N, H) — post-residual
  std::vector<float> attn;       // (N, N) — row-wise softmax weights
  std::vector<float> hxh_q;      // (n_heads, N, H) — optional branch
  std::vector<float> hxh_k;      // (n_heads, N, H)
  std::vector<float> hxh_v;      // (n_heads, N, H)
  std::vector<float> hxh_mixed;  // (n_heads, N, H)
  std::vector<float> hxh_attn;   // (n_heads, N, N)
  std::vector<float> logits_hxh; // (N,) pre-lambda branch contribution
};

// Single-query forward pass. V is row-major (N, K_v); E is row-major (N, N)
// pairwise edge features (V1: cosine of slot key reps, diagonal = 1);
// simplex_features is the (K_s,) global vector. Output shapes are pinned by
// SimplexWeights (N, H, K_s) — see the `Forward architecture` section of
// docs/plans/2026-04-26-simplex-controller-design.md for the math, copied
// verbatim into the implementation.
SimplexForwardOutput simplex_forward(
    const SimplexWeights& weights,
    const std::vector<float>& V,
    const std::vector<float>& E,
    const std::vector<float>& simplex_features);

}  // namespace chaoscontrol::simplex
