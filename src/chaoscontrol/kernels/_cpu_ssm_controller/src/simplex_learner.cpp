#include "simplex_learner.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace chaoscontrol::simplex {
namespace {

// Wraps a contiguous std::vector<float> as a 2-D at::Tensor view (no
// copy) so the backward arithmetic can lean on at::matmul. The vector
// must outlive any tensor returned here — every call site below uses
// the view inline within the same scope.
at::Tensor view_2d(std::vector<float>& buf, int64_t rows, int64_t cols) {
  return at::from_blob(
      buf.data(),
      {rows, cols},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

at::Tensor view_2d(const std::vector<float>& buf, int64_t rows, int64_t cols) {
  return at::from_blob(
      const_cast<float*>(buf.data()),
      {rows, cols},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

at::Tensor view_1d(std::vector<float>& buf, int64_t n) {
  return at::from_blob(
      buf.data(),
      {n},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

// dgelu(x)/dx = 0.5 * (1 + erf(x / sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
// Matches simplex_policy.cpp's exact (non-approximate) GeLU.
at::Tensor gelu_grad(const at::Tensor& pre_gelu) {
  const double inv_sqrt2 = 0.70710678118654752440;
  const double inv_sqrt_2pi = 0.39894228040143267794;
  return 0.5 * (1.0 + at::erf(pre_gelu * inv_sqrt2))
       + pre_gelu * at::exp(-0.5 * pre_gelu * pre_gelu) * inv_sqrt_2pi;
}

void zero_simplex_weights(SimplexWeights& w) {
  std::fill(w.W_vp.begin(), w.W_vp.end(), 0.0f);
  std::fill(w.b_vp.begin(), w.b_vp.end(), 0.0f);
  std::fill(w.W_lh.begin(), w.W_lh.end(), 0.0f);
  w.b_lh = 0.0f;
  std::fill(w.W_sb.begin(), w.W_sb.end(), 0.0f);
  w.alpha = 0.0f;
  // temperature and bucket_embed are not trainable in V1 — leave alone.
}

void copy_simplex_shape(const SimplexWeights& src, SimplexWeights& dst) {
  dst.K_v = src.K_v;
  dst.K_e = src.K_e;
  dst.K_s = src.K_s;
  dst.H = src.H;
  dst.N = src.N;
  dst.W_vp.assign(src.W_vp.size(), 0.0f);
  dst.b_vp.assign(src.b_vp.size(), 0.0f);
  dst.W_lh.assign(src.W_lh.size(), 0.0f);
  dst.b_lh = 0.0f;
  dst.W_sb.assign(src.W_sb.size(), 0.0f);
  dst.alpha = 0.0f;
  dst.temperature = src.temperature;
  dst.bucket_embed.assign(src.bucket_embed.size(), 0.0f);
}

}  // namespace

SimplexOnlineLearner::SimplexOnlineLearner(
    uint32_t num_slots,
    uint32_t max_entries_per_slot,
    float gamma,
    float learning_rate,
    uint32_t sgd_interval,
    float ema_alpha,
    uint64_t ema_interval)
    : history_(num_slots, max_entries_per_slot),
      fast_slow_(ema_alpha, ema_interval),
      sgd_(learning_rate),
      gamma_(gamma),
      sgd_interval_(sgd_interval) {
  if (sgd_interval == 0) {
    throw std::invalid_argument(
        "SimplexOnlineLearner sgd_interval must be > 0");
  }
}

void SimplexOnlineLearner::initialize_simplex_weights(SimplexWeights weights) {
  fast_weights_ = weights;
  slow_weights_ = weights;            // slow starts identical to fast
  copy_simplex_shape(weights, grad_weights_);
  weights_initialized_ = true;
}

void SimplexOnlineLearner::record_simplex_decision(
    uint64_t chosen_slot_id,
    uint64_t gpu_step,
    uint32_t policy_version,
    uint32_t chosen_idx,
    float p_chosen_decision,
    std::vector<float> V,
    std::vector<float> E,
    std::vector<float> simplex_features) {
  ActionHistoryEntry entry;
  entry.action_type = 2;  // V1 simplex selection (V0 used 1)
  entry.gpu_step = gpu_step;
  entry.policy_version = policy_version;
  entry.chosen_idx = chosen_idx;
  entry.p_chosen_decision = p_chosen_decision;
  entry.V = std::move(V);
  entry.E = std::move(E);
  entry.simplex_features = std::move(simplex_features);
  // PerSlotActionHistory keys on uint32_t slot_id; the chosen_slot_id
  // is the cache slot the simplex point landed on. Truncate to u32 for
  // the history key (cache capacity is bounded well below 2^32).
  history_.append(static_cast<uint32_t>(chosen_slot_id), std::move(entry));
  ++telemetry_.history_appends;
}

void SimplexOnlineLearner::on_replay_outcome(const ReplayOutcome& ev) {
  ++telemetry_.replay_outcomes;

  if (ev.outcome_status != 0) {
    return;  // Skip non-OK outcomes; reward signal is conditional on success.
  }

  const uint32_t slot_id_u32 = static_cast<uint32_t>(ev.slot_id);
  if (slot_id_u32 >= history_.num_slots()) {
    ++telemetry_.invalid_slot_skips;
    return;
  }
  if (!weights_initialized_) {
    ++telemetry_.backward_skipped_missing_weights;
    return;
  }

  // Find the most recent simplex decision (action_type == 2) for this
  // slot whose gpu_step matches the replay's selection_step. If none is
  // found, the controller wasn't yet trained or the history was GC'd —
  // skip with a counter rather than fabricating a gradient.
  const auto& slot_history = history_.history(slot_id_u32);
  const ActionHistoryEntry* match = nullptr;
  for (auto it = slot_history.rbegin(); it != slot_history.rend(); ++it) {
    if (it->action_type == 2 && it->gpu_step == ev.selection_step) {
      match = &(*it);
      break;
    }
  }
  if (match == nullptr || match->V.empty()) {
    ++telemetry_.backward_skipped_missing_state;
    return;
  }

  ++telemetry_.credited_actions;

  // Recompute the forward on the fast weights so the gradient reflects
  // the policy at update time, not at decision time. (Off-policy ratio
  // p_chosen_now / p_chosen_decision is V2 — V1 is on-policy REINFORCE.)
  SimplexForwardOutput fwd =
      simplex_forward(fast_weights_, match->V, match->E, match->simplex_features);

  ++telemetry_.backward_ready_actions;

  // Advantage: ce_delta_raw is "improvement after replay" (positive =
  // good); bucket_baseline is the rolling average for this admission
  // bucket; advantage centers the reward to reduce variance. Recency
  // decay shrinks credit for stale decisions, matching the existing
  // scalar reward contract.
  float advantage = ev.ce_delta_raw - ev.bucket_baseline;
  if (!std::isfinite(advantage)) {
    advantage = 0.0f;
  }
  const uint64_t step_gap =
      ev.gpu_step >= match->gpu_step ? (ev.gpu_step - match->gpu_step) : 0;
  advantage *= std::pow(gamma_, static_cast<float>(step_gap));
  last_advantage_ = advantage;

  if (advantage == 0.0f) {
    return;  // Zero gradient; skip the rest of the work.
  }

  simplex_backward(*match, fwd, advantage);
  ++actions_since_sgd_;

  maybe_apply_sgd();
  maybe_blend_slow();
}

void SimplexOnlineLearner::simplex_backward(
    const ActionHistoryEntry& entry,
    const SimplexForwardOutput& fwd,
    float advantage) {
  const int64_t N = static_cast<int64_t>(fast_weights_.N);
  const int64_t H = static_cast<int64_t>(fast_weights_.H);
  const int64_t K_v = static_cast<int64_t>(fast_weights_.K_v);
  const int64_t K_s = static_cast<int64_t>(fast_weights_.K_s);
  const float T = fast_weights_.temperature;
  const uint32_t chosen = entry.chosen_idx;

  // ---- g_logits = advantage * (p - one_hot(chosen)) / T --------------
  std::vector<float> g_logits_buf(static_cast<size_t>(N), 0.0f);
  for (int64_t i = 0; i < N; ++i) {
    const float indicator = (static_cast<uint32_t>(i) == chosen) ? 1.0f : 0.0f;
    g_logits_buf[i] = advantage * (fwd.p[i] - indicator) / T;
  }
  at::Tensor g_logits = view_1d(g_logits_buf, N);

  // ---- Layer 3 — logit head: logits = mixed_h @ W_lh + b_lh ----------
  // mixed_h is [N, H], W_lh is [H], logits is [N], b_lh is scalar.
  at::Tensor mixed_h = view_2d(fwd.mixed_h, N, H);

  // dL/dW_lh[h] = sum_i g_logits[i] * mixed_h[i, h]
  at::Tensor g_W_lh = at::matmul(g_logits, mixed_h);  // [H]
  at::Tensor W_lh_acc = view_1d(grad_weights_.W_lh, H);
  W_lh_acc.add_(g_W_lh);

  // dL/db_lh = sum_i g_logits[i]
  grad_weights_.b_lh += g_logits.sum().item<float>();

  // dL/dW_sb[k] = simplex_features[k] * sum_i g_logits[i]
  const float g_logits_sum = g_logits.sum().item<float>();
  for (int64_t k = 0; k < K_s; ++k) {
    grad_weights_.W_sb[k] += entry.simplex_features[k] * g_logits_sum;
  }

  // ---- residual: g_mixed flows to attn-output AND to vertex_h --------
  // g_mixed[i, h] = g_logits[i] * W_lh[h]
  at::Tensor W_lh = view_1d(fast_weights_.W_lh, H);
  at::Tensor g_mixed = g_logits.unsqueeze(1) * W_lh.unsqueeze(0);  // [N, H]

  // The residual mixed_h = attn_out + vertex_h means dL/dattn_out and
  // dL/dvertex_h_residual_branch BOTH equal g_mixed. We carry the
  // residual contribution into g_vertex_h_total below.

  // ---- Layer 2 — edge-aware mixing -----------------------------------
  at::Tensor vertex_h = view_2d(fwd.vertex_h, N, H);
  at::Tensor attn = view_2d(fwd.attn, N, N);
  at::Tensor E_t = view_2d(entry.E, N, N);

  // attn_out[i, h] = sum_j attn[i, j] * vertex_h[j, h]
  // dL/dattn[i, j] = sum_h g_mixed[i, h] * vertex_h[j, h]
  at::Tensor g_attn =
      at::matmul(g_mixed, vertex_h.transpose(0, 1).contiguous());  // [N, N]

  // g_attn_logits[i, k] = attn[i, k] * (g_attn[i, k]
  //                       - sum_j attn[i, j] * g_attn[i, j])
  // Standard softmax backward (row-wise softmax over j).
  at::Tensor row_dot = (attn * g_attn).sum(1, /*keepdim=*/true);  // [N, 1]
  at::Tensor g_attn_logits = attn * (g_attn - row_dot);           // [N, N]

  // dL/dalpha = sum_{i,j} g_attn_logits[i, j] * E[i, j]
  grad_weights_.alpha += (g_attn_logits * E_t).sum().item<float>();

  // ---- backward through bilinear: vh[i] dot vh[j] / sqrt(H) ----------
  // dL/dvh[i, h] (attn-bilinear, first index)
  //     = sum_j g_attn_logits[i, j] * vh[j, h] / sqrt(H)
  // dL/dvh[i, h] (attn-bilinear, second index, by symmetry j <-> i)
  //     = sum_k g_attn_logits[k, i] * vh[k, h] / sqrt(H)
  const float inv_sqrt_H = 1.0f / std::sqrt(static_cast<float>(H));
  at::Tensor g_vh_attn_first =
      at::matmul(g_attn_logits, vertex_h) * inv_sqrt_H;            // [N, H]
  at::Tensor g_vh_attn_second =
      at::matmul(g_attn_logits.transpose(0, 1).contiguous(), vertex_h)
      * inv_sqrt_H;                                                // [N, H]

  // ---- backward through attn @ vh (mixed branch, vertex_h side) ------
  // (attn @ vh)[k, h] = sum_i attn[k, i] * vh[i, h]
  // dL/dvh[i, h] (mixed branch)
  //     = sum_k g_mixed[k, h] * attn[k, i]
  at::Tensor g_vh_mixed =
      at::matmul(attn.transpose(0, 1).contiguous(), g_mixed);      // [N, H]

  // ---- total g_vertex_h: attn-bilinear + mixed + residual ------------
  at::Tensor g_vertex_h =
      g_vh_attn_first + g_vh_attn_second + g_vh_mixed + g_mixed;   // [N, H]

  // ---- Layer 1 — vertex projection backward --------------------------
  // pre_gelu = V @ W_vp + b_vp; vertex_h = gelu(pre_gelu)
  at::Tensor V_t = view_2d(entry.V, N, K_v);
  at::Tensor W_vp = view_2d(fast_weights_.W_vp, K_v, H);
  at::Tensor b_vp = view_1d(fast_weights_.b_vp, H);
  at::Tensor pre_gelu = at::matmul(V_t, W_vp) + b_vp;              // [N, H]

  at::Tensor g_pre_gelu = g_vertex_h * gelu_grad(pre_gelu);        // [N, H]

  // dL/dW_vp[k, h] = sum_i V[i, k] * g_pre_gelu[i, h]
  at::Tensor g_W_vp =
      at::matmul(V_t.transpose(0, 1).contiguous(), g_pre_gelu);    // [K_v, H]
  at::Tensor W_vp_acc = view_2d(grad_weights_.W_vp, K_v, H);
  W_vp_acc.add_(g_W_vp);

  // dL/db_vp[h] = sum_i g_pre_gelu[i, h]
  at::Tensor g_b_vp = g_pre_gelu.sum(0);                           // [H]
  at::Tensor b_vp_acc = view_1d(grad_weights_.b_vp, H);
  b_vp_acc.add_(g_b_vp);
}

void SimplexOnlineLearner::maybe_apply_sgd() {
  if (actions_since_sgd_ < sgd_interval_) {
    return;
  }
  // SGD on flat float vectors; alpha and b_lh are scalars.
  sgd_.apply(fast_weights_.W_vp.data(), grad_weights_.W_vp.data(),
             fast_weights_.W_vp.size());
  sgd_.apply(fast_weights_.b_vp.data(), grad_weights_.b_vp.data(),
             fast_weights_.b_vp.size());
  sgd_.apply(fast_weights_.W_lh.data(), grad_weights_.W_lh.data(),
             fast_weights_.W_lh.size());
  sgd_.apply(&fast_weights_.b_lh, &grad_weights_.b_lh, 1);
  sgd_.apply(fast_weights_.W_sb.data(), grad_weights_.W_sb.data(),
             fast_weights_.W_sb.size());
  sgd_.apply(&fast_weights_.alpha, &grad_weights_.alpha, 1);
  zero_grad();
  actions_since_sgd_ = 0;
  ++telemetry_.sgd_steps;
}

void SimplexOnlineLearner::maybe_blend_slow() {
  fast_slow_.tick_event();
  if (!fast_slow_.should_blend()) {
    return;
  }
  if (!weights_initialized_) {
    return;
  }
  fast_slow_.blend(slow_weights_.W_vp.data(), fast_weights_.W_vp.data(),
                   slow_weights_.W_vp.size());
  fast_slow_.blend(slow_weights_.b_vp.data(), fast_weights_.b_vp.data(),
                   slow_weights_.b_vp.size());
  fast_slow_.blend(slow_weights_.W_lh.data(), fast_weights_.W_lh.data(),
                   slow_weights_.W_lh.size());
  fast_slow_.blend(&slow_weights_.b_lh, &fast_weights_.b_lh, 1);
  fast_slow_.blend(slow_weights_.W_sb.data(), fast_weights_.W_sb.data(),
                   slow_weights_.W_sb.size());
  fast_slow_.blend(&slow_weights_.alpha, &fast_weights_.alpha, 1);
  ++telemetry_.ema_blends;
}

void SimplexOnlineLearner::zero_grad() {
  zero_simplex_weights(grad_weights_);
}

const std::vector<ActionHistoryEntry>& SimplexOnlineLearner::history(
    uint32_t slot_id) const {
  return history_.history(slot_id);
}

const SimplexLearnerTelemetry& SimplexOnlineLearner::telemetry() const {
  return telemetry_;
}

const SimplexWeights& SimplexOnlineLearner::fast_weights() const {
  return fast_weights_;
}

const SimplexWeights& SimplexOnlineLearner::slow_weights() const {
  return slow_weights_;
}

bool SimplexOnlineLearner::weights_initialized() const {
  return weights_initialized_;
}

float SimplexOnlineLearner::last_advantage() const {
  return last_advantage_;
}

}  // namespace chaoscontrol::simplex
