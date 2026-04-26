#include "simplex_policy.h"

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// Phase S1 — see simplex_policy.h for surface contract and the design doc
// (docs/plans/2026-04-26-simplex-controller-design.md, "Forward architecture")
// for the math. Three layers: vertex projection (Layer 1), edge-aware mixing
// (Layer 2), logit head + simplex bias (Layer 3). All GEMMs route through
// at::matmul; the AMX fast-path lands in a follow-up commit once correctness
// is pinned on arm64.

namespace chaoscontrol::simplex {

namespace {

constexpr double kInvSqrt2 = 0.70710678118654752440;

// Exact GeLU matching the existing extension (cpu_ssm_controller.cpp):
//   0.5 * x * (1 + erf(x / sqrt(2)))
// Applied element-wise on a flat float buffer so the saved-for-backward
// `vertex_h` doesn't pay the tensor-roundtrip cost.
void apply_exact_gelu(float* data, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const float x = data[i];
    const float ge = 0.5f * x *
        (1.0f + static_cast<float>(std::erf(static_cast<double>(x) * kInvSqrt2)));
    data[i] = ge;
  }
}

// Numerically stable row-wise softmax over a (rows, cols) row-major buffer,
// in place. Subtracts the per-row max before exp(); used both for Layer 2's
// `softmax_j(attn_logits[i])` and the final `softmax(logits + bias)`. The
// final softmax is just a 1-row case (rows=1, cols=N).
void rowwise_softmax_inplace(float* data, std::size_t rows, std::size_t cols) {
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = data + r * cols;
    float maxv = -std::numeric_limits<float>::infinity();
    for (std::size_t c = 0; c < cols; ++c) {
      maxv = std::max(maxv, row[c]);
    }
    float sum = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) {
      const float e = std::exp(row[c] - maxv);
      row[c] = e;
      sum += e;
    }
    // sum > 0 always; if every input is -inf, exp(-inf - -inf) = exp(nan)
    // NaN-poisons here — but we never feed -inf into a stable softmax in
    // this kernel, so a zero-divide guard would mask a real bug rather
    // than mitigate it.
    const float inv = 1.0f / sum;
    for (std::size_t c = 0; c < cols; ++c) {
      row[c] *= inv;
    }
  }
}

// Wrap a non-owning view over an existing flat buffer as a CPU float tensor.
// The vector must outlive every tensor returned from this helper.
at::Tensor view_2d(const std::vector<float>& v, int64_t rows, int64_t cols) {
  return at::from_blob(
      const_cast<float*>(v.data()),
      {rows, cols},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

at::Tensor view_1d(const std::vector<float>& v, int64_t n) {
  return at::from_blob(
      const_cast<float*>(v.data()),
      {n},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

// at::matmul into a contiguous fp32 result; copy out to a new vector. The
// result is materialized into the output struct so SimplexForwardOutput is a
// pure-C++ POD that doesn't pin a torch reference past return. Cost: one
// extra copy per GEMM (tens to hundreds of bytes for these shapes — the
// AMX-tile-natural N=16 case is dominated by the GEMM itself, not the copy).
std::vector<float> matmul_to_vec(const at::Tensor& a, const at::Tensor& b) {
  at::Tensor out = at::matmul(a, b).contiguous();
  std::vector<float> result(static_cast<std::size_t>(out.numel()));
  std::memcpy(
      result.data(),
      out.data_ptr<float>(),
      result.size() * sizeof(float));
  return result;
}

}  // namespace

SimplexForwardOutput simplex_forward(
    const SimplexWeights& weights,
    const std::vector<float>& V,
    const std::vector<float>& E,
    const std::vector<float>& simplex_features) {
  const uint32_t N = weights.N;
  const uint32_t K_v = weights.K_v;
  const uint32_t K_s = weights.K_s;
  const uint32_t H = weights.H;

  TORCH_CHECK(N > 0, "SimplexWeights.N must be positive");
  TORCH_CHECK(K_v > 0, "SimplexWeights.K_v must be positive");
  TORCH_CHECK(H > 0, "SimplexWeights.H must be positive");
  TORCH_CHECK(K_s > 0, "SimplexWeights.K_s must be positive");
  TORCH_CHECK(weights.temperature > 0.0f,
              "SimplexWeights.temperature must be positive");
  TORCH_CHECK(
      weights.W_vp.size() == static_cast<std::size_t>(K_v) * H,
      "SimplexWeights.W_vp must have ", K_v * H, " elements (K_v*H), got ",
      weights.W_vp.size());
  TORCH_CHECK(weights.b_vp.size() == H,
              "SimplexWeights.b_vp must have H elements, got ",
              weights.b_vp.size());
  TORCH_CHECK(weights.W_lh.size() == H,
              "SimplexWeights.W_lh must have H elements, got ",
              weights.W_lh.size());
  TORCH_CHECK(weights.W_sb.size() == K_s,
              "SimplexWeights.W_sb must have K_s elements, got ",
              weights.W_sb.size());
  TORCH_CHECK(V.size() == static_cast<std::size_t>(N) * K_v,
              "V must have N*K_v = ", N * K_v, " elements, got ", V.size());
  TORCH_CHECK(E.size() == static_cast<std::size_t>(N) * N,
              "E must have N*N = ", N * N, " elements, got ", E.size());
  TORCH_CHECK(simplex_features.size() == K_s,
              "simplex_features must have K_s = ", K_s, " elements, got ",
              simplex_features.size());

  SimplexForwardOutput out;
  out.logits.resize(N);
  out.p.resize(N);
  out.vertex_h.resize(static_cast<std::size_t>(N) * H);
  out.mixed_h.resize(static_cast<std::size_t>(N) * H);
  out.attn.resize(static_cast<std::size_t>(N) * N);

  // ---- Layer 1: vertex projection ----
  // vertex_h = gelu(V @ W_vp + b_vp)        [N, K_v] @ [K_v, H] -> [N, H]
  {
    at::Tensor V_t = view_2d(V, N, K_v);
    at::Tensor Wvp_t = view_2d(weights.W_vp, K_v, H);
    std::vector<float> proj = matmul_to_vec(V_t, Wvp_t);
    // Broadcast-add bias and GeLU in place on the flat buffer; this is the
    // post-GeLU activation we save for backward.
    const float* b = weights.b_vp.data();
    for (uint32_t i = 0; i < N; ++i) {
      float* row = proj.data() + static_cast<std::size_t>(i) * H;
      for (uint32_t j = 0; j < H; ++j) {
        row[j] += b[j];
      }
    }
    apply_exact_gelu(proj.data(), proj.size());
    out.vertex_h = std::move(proj);
  }

  // ---- Layer 2: edge-aware mixing ----
  // attn_logits[i, j] = (vertex_h[i] · vertex_h[j]) / sqrt(H) + alpha * E[i, j]
  // attn[i, j]        = softmax_j(attn_logits[i, j])
  // mixed_h[i]        = sum_j attn[i, j] * vertex_h[j]   then + vertex_h (residual)
  {
    at::Tensor vh_t = view_2d(out.vertex_h, N, H);
    // vh_t @ vh_t.T -> [N, N] of dot products. at::matmul handles the
    // transpose-then-multiply via the second operand's stride; .contiguous()
    // ensures the materialized result is row-major before we read it.
    at::Tensor dots_t = at::matmul(vh_t, vh_t.t()).contiguous();
    const float* dots = dots_t.data_ptr<float>();
    const float scale = 1.0f / std::sqrt(static_cast<float>(H));
    const float alpha = weights.alpha;

    // Build attn_logits in out.attn (we'll overwrite with softmax in place).
    for (uint32_t i = 0; i < N; ++i) {
      float* row = out.attn.data() + static_cast<std::size_t>(i) * N;
      const float* drow = dots + static_cast<std::size_t>(i) * N;
      const float* erow = E.data() + static_cast<std::size_t>(i) * N;
      for (uint32_t j = 0; j < N; ++j) {
        row[j] = drow[j] * scale + alpha * erow[j];
      }
    }
    rowwise_softmax_inplace(out.attn.data(), N, N);

    // mixed_h = attn @ vertex_h, then add residual (vertex_h).
    at::Tensor attn_t = view_2d(out.attn, N, N);
    std::vector<float> mixed = matmul_to_vec(attn_t, vh_t);
    // Residual: post-residual is what S2's head backward expects to read
    // from out.mixed_h (the design doc's "saved for backward" line).
    for (std::size_t k = 0; k < mixed.size(); ++k) {
      mixed[k] += out.vertex_h[k];
    }
    out.mixed_h = std::move(mixed);
  }

  // ---- Layer 3: logit head + simplex bias + temperature softmax ----
  // logits = mixed_h @ W_lh + b_lh                  [N, H] @ [H] -> [N]
  // simplex_bias = simplex_features · W_sb          scalar broadcasts
  // p = softmax((logits + simplex_bias) / T)
  {
    at::Tensor mh_t = view_2d(out.mixed_h, N, H);
    at::Tensor wlh_t = view_1d(weights.W_lh, H);
    // matmul of (N, H) @ (H,) returns shape (N,).
    at::Tensor logits_t = at::matmul(mh_t, wlh_t).contiguous();
    std::memcpy(
        out.logits.data(),
        logits_t.data_ptr<float>(),
        static_cast<std::size_t>(N) * sizeof(float));

    float sb = 0.0f;
    for (uint32_t k = 0; k < K_s; ++k) {
      sb += simplex_features[k] * weights.W_sb[k];
    }
    const float inv_T = 1.0f / weights.temperature;
    for (uint32_t i = 0; i < N; ++i) {
      out.logits[i] += weights.b_lh;
      // Write into out.p so out.logits keeps the pre-softmax-bias form
      // (the design doc specifies `logits = mixed_h @ W_lh + b_lh` and
      // separately `p = softmax((logits + simplex_bias) / T)`).
      out.p[i] = (out.logits[i] + sb) * inv_T;
    }
    rowwise_softmax_inplace(out.p.data(), 1, N);
  }

  return out;
}

}  // namespace chaoscontrol::simplex
