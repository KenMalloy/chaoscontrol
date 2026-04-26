#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "action_history.h"
#include "credit.h"
#include "optimizer.h"
#include "simplex_policy.h"
#include "wire_events.h"

namespace chaoscontrol::simplex {

struct SimplexLearnerTelemetry {
  uint64_t replay_outcomes = 0;
  uint64_t history_appends = 0;
  uint64_t credited_actions = 0;
  uint64_t backward_ready_actions = 0;
  uint64_t backward_skipped_missing_state = 0;
  uint64_t backward_skipped_missing_weights = 0;
  uint64_t invalid_slot_skips = 0;
  uint64_t sgd_steps = 0;
  uint64_t ema_blends = 0;
};

// V1 online learner: REINFORCE over the simplex policy. Keeps fast / slow /
// grad copies of SimplexWeights and applies policy-gradient updates on
// each replay outcome that has a matching record_simplex_decision in the
// per-slot history.
//
// Intended call sequence per query:
//   1. Producer (heuristic top-K) constructs the candidate simplex.
//   2. Caller computes simplex_forward(fast_weights, V, E, simplex_features).
//   3. Caller samples / argmaxes p[16] -> chosen_idx, p_chosen_decision.
//   4. Caller invokes record_simplex_decision(...) with the chosen_slot_id
//      and the full V/E/simplex_features snapshot.
//
// Per replay outcome:
//   1. on_replay_outcome(ev) finds the matching ActionHistoryEntry by
//      chosen_slot_id and pulls the snapshot.
//   2. Re-runs the forward to get current p[16] (cheap; one query worth).
//   3. Computes advantage = ce_delta_raw - bucket_baseline, optionally
//      scaled by recency decay and Gerber concordance.
//   4. Backprops -advantage * log p[chosen_idx] through the simplex
//      forward graph and accumulates gradients in grad_weights_.
//   5. Bumps actions_since_sgd_; on the boundary applies SGD and
//      optionally blends slow := alpha * fast + (1-alpha) * slow.
class SimplexOnlineLearner {
 public:
  SimplexOnlineLearner(
      uint32_t num_slots = 4096,
      uint32_t max_entries_per_slot = 64,
      float gamma = 0.995f,
      float learning_rate = 1.0e-3f,
      uint32_t sgd_interval = 256,
      float ema_alpha = 0.25f,
      uint64_t ema_interval = 64);

  void initialize_simplex_weights(SimplexWeights weights);
  void record_simplex_decision(
      uint64_t chosen_slot_id,
      uint64_t gpu_step,
      uint32_t policy_version,
      uint32_t chosen_idx,
      float p_chosen_decision,
      std::vector<float> V,
      std::vector<float> E,
      std::vector<float> simplex_features);
  void on_replay_outcome(const ReplayOutcome& ev);

  const std::vector<ActionHistoryEntry>& history(uint32_t slot_id) const;
  const SimplexLearnerTelemetry& telemetry() const;
  const SimplexWeights& fast_weights() const;
  const SimplexWeights& slow_weights() const;
  bool weights_initialized() const;
  float last_advantage() const;

 private:
  // Load-bearing: in-place backward through the simplex_forward graph.
  // Computes gradients into grad_weights_ for the saved (V, E,
  // simplex_features) and the chosen_idx; uses the precomputed
  // SimplexForwardOutput's saved buffers (vertex_h, mixed_h, attn) to
  // avoid recomputing them. Caller passes the scalar advantage; this
  // function does NOT apply recency / Gerber shaping (caller does).
  void simplex_backward(
      const ActionHistoryEntry& entry,
      const SimplexForwardOutput& fwd,
      float advantage);

  void maybe_apply_sgd();
  void maybe_blend_slow();
  void zero_grad();

  PerSlotActionHistory history_;
  FastSlowEma fast_slow_;
  SgdStep sgd_;
  float gamma_;
  uint32_t sgd_interval_;
  uint32_t actions_since_sgd_ = 0;
  bool weights_initialized_ = false;
  SimplexWeights fast_weights_;
  SimplexWeights slow_weights_;
  SimplexWeights grad_weights_;
  SimplexLearnerTelemetry telemetry_;
  float last_advantage_ = 0.0f;
};

}  // namespace chaoscontrol::simplex
