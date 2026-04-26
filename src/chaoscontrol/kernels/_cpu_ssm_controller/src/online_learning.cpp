#include "online_learning.h"

#include <cmath>
#include <utility>

OnlineLearningController::OnlineLearningController(
    uint32_t num_slots,
    uint32_t max_entries_per_slot,
    float gamma,
    float gerber_c)
    : history_(num_slots, max_entries_per_slot),
      sigma_by_type_{RollingStddev(), RollingStddev(), RollingStddev(),
                     RollingStddev()},
      fast_slow_(),
      gamma_(gamma),
      gerber_c_(gerber_c) {}

void OnlineLearningController::on_write(const WriteEvent&) {
  // WRITE_EVENT has candidate_id but no slot_id in the current wire schema.
  // Keep this as a no-op until admission events can be mapped to cache slots.
}

void OnlineLearningController::on_query(const QueryEvent&) {
  // QUERY_EVENT represents a controller query, not a selected replay action.
  // It has no selected slot/rank/logit yet, so C10 leaves it as metadata-only.
}

void OnlineLearningController::on_replay_outcome(const ReplayOutcome& ev) {
  ++telemetry_.replay_outcomes;
  last_credit_sum_ = 0.0f;

  if (ev.outcome_status != 0) {
    return;
  }
  if (ev.slot_id >= history_.num_slots()) {
    ++telemetry_.invalid_slot_skips;
    return;
  }

  const std::vector<CreditedAction> credited = attribute_credit(
      ev.gpu_step,
      ev.reward_shaped,
      ev.controller_logit,
      history_.history(ev.slot_id),
      sigma_by_action_type(),
      gamma_,
      gerber_c_);

  for (const CreditedAction& action : credited) {
    ++telemetry_.credited_actions;
    if (action.credit == 0.0f || std::isnan(action.credit)) {
      continue;
    }
    ++telemetry_.nonzero_credit_actions;
    last_credit_sum_ += action.credit;
    if (action.entry.features.empty() || action.entry.global_state.empty() ||
        action.entry.slot_state.empty()) {
      // Backward needs the exact decision-time input and recurrent state.
      // Legacy replay selections still arrive without those checkpoints, so
      // keep the skip visible rather than pretending SGD happened.
      ++telemetry_.backward_skipped_missing_state;
      continue;
    }
    ++telemetry_.backward_ready_actions;
  }

  append_replay_selection(ev);
  sigma_by_type_[1].update(ev.controller_logit);
  fast_slow_.tick_event();
  if (fast_slow_.should_blend()) {
    ++telemetry_.ema_blends;
  }
}

const std::vector<ActionHistoryEntry>& OnlineLearningController::history(
    uint32_t slot_id) const {
  return history_.history(slot_id);
}

const OnlineLearningTelemetry& OnlineLearningController::telemetry() const {
  return telemetry_;
}

float OnlineLearningController::last_credit_sum() const {
  return last_credit_sum_;
}

void OnlineLearningController::record_replay_selection(
    uint32_t slot_id,
    uint64_t gpu_step,
    uint32_t policy_version,
    float output_logit,
    uint8_t selected_rank,
    std::vector<float> features,
    std::vector<float> global_state,
    std::vector<float> slot_state) {
  ActionHistoryEntry entry;
  entry.action_type = 1;
  entry.gpu_step = gpu_step;
  entry.policy_version = policy_version;
  entry.output_logit = output_logit;
  entry.selected_rank = selected_rank;
  entry.neighbor_slot = 0;
  entry.features = std::move(features);
  entry.global_state = std::move(global_state);
  entry.slot_state = std::move(slot_state);
  history_.append(slot_id, std::move(entry));
  ++telemetry_.history_appends;
}

std::vector<float> OnlineLearningController::sigma_by_action_type() const {
  std::vector<float> sigma;
  sigma.reserve(sigma_by_type_.size());
  for (const RollingStddev& tracker : sigma_by_type_) {
    sigma.push_back(tracker.stddev());
  }
  return sigma;
}

void OnlineLearningController::append_replay_selection(const ReplayOutcome& ev) {
  ActionHistoryEntry entry;
  entry.action_type = 1;
  entry.gpu_step = ev.selection_step == 0 ? ev.gpu_step : ev.selection_step;
  entry.policy_version = ev.policy_version;
  entry.output_logit = ev.controller_logit;
  entry.selected_rank = ev.selected_rank;
  entry.neighbor_slot = 0;
  history_.append(ev.slot_id, entry);
  ++telemetry_.history_appends;
}
