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
  uint64_t backward_skipped_missing_state = 0;
  uint64_t invalid_slot_skips = 0;
  uint64_t sgd_steps = 0;
  uint64_t ema_blends = 0;
};

class OnlineLearningController {
 public:
  OnlineLearningController(
      uint32_t num_slots = 4096,
      uint32_t max_entries_per_slot = 64,
      float gamma = 0.995f,
      float gerber_c = 0.5f);

  void on_write(const WriteEvent& ev);
  void on_query(const QueryEvent& ev);
  void on_replay_outcome(const ReplayOutcome& ev);

  const std::vector<ActionHistoryEntry>& history(uint32_t slot_id) const;
  const OnlineLearningTelemetry& telemetry() const;
  float last_credit_sum() const;

 private:
  std::vector<float> sigma_by_action_type() const;
  void append_replay_selection(const ReplayOutcome& ev);

  PerSlotActionHistory history_;
  std::array<RollingStddev, 4> sigma_by_type_;
  FastSlowEma fast_slow_;
  float gamma_;
  float gerber_c_;
  OnlineLearningTelemetry telemetry_;
  float last_credit_sum_ = 0.0f;
};
