#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "action_history.h"
#include "credit.h"
#include "optimizer.h"
#include "wire_events.h"

struct OnlineLearningTelemetry {
  uint64_t replay_outcomes = 0;
  uint64_t history_appends = 0;
  uint64_t credited_actions = 0;
  uint64_t nonzero_credit_actions = 0;
  uint64_t backward_ready_actions = 0;
  uint64_t backward_skipped_missing_state = 0;
  uint64_t backward_skipped_missing_weights = 0;
  uint64_t backward_skipped_bad_shape = 0;
  uint64_t invalid_slot_skips = 0;
  uint64_t sgd_steps = 0;
  uint64_t ema_blends = 0;
};

struct OnlineLearningWeights {
  uint32_t feature_dim = 0;
  uint32_t global_dim = 0;
  uint32_t slot_dim = 0;
  std::vector<float> w_global_in;
  std::vector<float> w_slot_in;
  std::vector<float> decay_global;
  std::vector<float> decay_slot;
  std::vector<float> w_global_out;
  std::vector<float> w_slot_out;
  float bias = 0.0f;
};

class OnlineLearningController {
 public:
  OnlineLearningController(
      uint32_t num_slots = 4096,
      uint32_t max_entries_per_slot = 64,
      float gamma = 0.995f,
      float gerber_c = 0.5f,
      float learning_rate = 1.0e-3f,
      uint32_t sgd_interval = 256,
      float ema_alpha = 0.25f,
      uint64_t ema_interval = 64);

  void on_write(const WriteEvent& ev);
  void on_query(const QueryEvent& ev);
  void on_replay_outcome(const ReplayOutcome& ev);
  void initialize_weights(
      uint32_t feature_dim,
      uint32_t global_dim,
      uint32_t slot_dim,
      std::vector<float> w_global_in,
      std::vector<float> w_slot_in,
      std::vector<float> decay_global,
      std::vector<float> decay_slot,
      std::vector<float> w_global_out,
      std::vector<float> w_slot_out,
      float bias);
  void record_replay_selection(
      uint32_t slot_id,
      uint64_t gpu_step,
      uint32_t policy_version,
      float output_logit,
      uint8_t selected_rank,
      std::vector<float> features,
      std::vector<float> global_state,
      std::vector<float> slot_state);

  const std::vector<ActionHistoryEntry>& history(uint32_t slot_id) const;
  const OnlineLearningTelemetry& telemetry() const;
  float last_credit_sum() const;
  const OnlineLearningWeights& fast_weights() const;
  const OnlineLearningWeights& slow_weights() const;
  bool weights_initialized() const;

 private:
  std::vector<float> sigma_by_action_type() const;
  void append_replay_selection(const ReplayOutcome& ev);
  bool ensure_default_weights_for_entry(const ActionHistoryEntry& entry);
  bool weights_match_entry(const ActionHistoryEntry& entry) const;
  void accumulate_backward(const ActionHistoryEntry& entry, float credit);
  void maybe_apply_sgd();
  void maybe_blend_slow();

  PerSlotActionHistory history_;
  std::array<RollingStddev, 4> sigma_by_type_;
  FastSlowEma fast_slow_;
  SgdStep sgd_;
  float gamma_;
  float gerber_c_;
  uint32_t sgd_interval_;
  uint32_t actions_since_sgd_ = 0;
  bool weights_initialized_ = false;
  // Cached at construction so the per-event hot path skips the CPUID/XCR0
  // dance per replay. Resolves to true on Sapphire Rapids when the build
  // included the AVX-512 matops kernel and the OS has AVX-512 state
  // enabled; false on arm64 / non-AVX-512 hosts. Determines whether
  // accumulate_backward routes through the AVX-512 raw kernels or the
  // scalar reference.
  bool use_avx512_matops_ = false;
  OnlineLearningWeights fast_weights_;
  OnlineLearningWeights slow_weights_;
  OnlineLearningWeights grad_weights_;
  OnlineLearningTelemetry telemetry_;
  float last_credit_sum_ = 0.0f;
};
