#include "online_learning.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

std::size_t checked_matrix_size(
    uint32_t rows,
    uint32_t cols,
    const char* name) {
  const std::size_t r = static_cast<std::size_t>(rows);
  const std::size_t c = static_cast<std::size_t>(cols);
  if (r != 0 && c > static_cast<std::size_t>(-1) / r) {
    throw std::invalid_argument(std::string(name) + " size overflow");
  }
  return r * c;
}

void require_size(
    const std::vector<float>& values,
    std::size_t expected,
    const char* name) {
  if (values.size() != expected) {
    throw std::invalid_argument(
        std::string(name) + " has " + std::to_string(values.size()) +
        " value(s), expected " + std::to_string(expected));
  }
}

void apply_sgd_to_vector(
    const SgdStep& sgd,
    std::vector<float>& weights,
    std::vector<float>& gradients) {
  if (weights.empty()) {
    return;
  }
  sgd.apply(weights.data(), gradients.data(), weights.size());
  std::fill(gradients.begin(), gradients.end(), 0.0f);
}

void blend_vector(
    const FastSlowEma& ema,
    std::vector<float>& slow,
    const std::vector<float>& fast) {
  if (slow.empty()) {
    return;
  }
  ema.blend(slow.data(), fast.data(), slow.size());
}

}  // namespace

OnlineLearningController::OnlineLearningController(
    uint32_t num_slots,
    uint32_t max_entries_per_slot,
    float gamma,
    float gerber_c,
    float learning_rate,
    uint32_t sgd_interval,
    float ema_alpha,
    uint64_t ema_interval)
    : history_(num_slots, max_entries_per_slot),
      sigma_by_type_{RollingStddev(), RollingStddev(), RollingStddev(),
                     RollingStddev()},
      fast_slow_(ema_alpha, ema_interval),
      sgd_(learning_rate),
      gamma_(gamma),
      gerber_c_(gerber_c),
      sgd_interval_(sgd_interval) {
  if (sgd_interval == 0) {
    throw std::invalid_argument(
        "OnlineLearningController sgd_interval must be > 0");
  }
}

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
    if (!ensure_default_weights_for_entry(action.entry)) {
      ++telemetry_.backward_skipped_missing_weights;
      continue;
    }
    if (!weights_match_entry(action.entry)) {
      ++telemetry_.backward_skipped_bad_shape;
      continue;
    }
    ++telemetry_.backward_ready_actions;
    accumulate_backward(action.entry, action.credit);
    ++actions_since_sgd_;
    maybe_apply_sgd();
  }

  append_replay_selection(ev);
  sigma_by_type_[1].update(ev.controller_logit);
  fast_slow_.tick_event();
  maybe_blend_slow();
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

void OnlineLearningController::initialize_weights(
    uint32_t feature_dim,
    uint32_t global_dim,
    uint32_t slot_dim,
    std::vector<float> w_global_in,
    std::vector<float> w_slot_in,
    std::vector<float> decay_global,
    std::vector<float> decay_slot,
    std::vector<float> w_global_out,
    std::vector<float> w_slot_out,
    float bias) {
  if (feature_dim == 0 || global_dim == 0 || slot_dim == 0) {
    throw std::invalid_argument("controller weight dims must be positive");
  }
  require_size(
      w_global_in,
      checked_matrix_size(global_dim, feature_dim, "w_global_in"),
      "w_global_in");
  require_size(
      w_slot_in,
      checked_matrix_size(slot_dim, feature_dim, "w_slot_in"),
      "w_slot_in");
  require_size(decay_global, global_dim, "decay_global");
  require_size(decay_slot, slot_dim, "decay_slot");
  require_size(w_global_out, global_dim, "w_global_out");
  require_size(w_slot_out, slot_dim, "w_slot_out");

  fast_weights_.feature_dim = feature_dim;
  fast_weights_.global_dim = global_dim;
  fast_weights_.slot_dim = slot_dim;
  fast_weights_.w_global_in = std::move(w_global_in);
  fast_weights_.w_slot_in = std::move(w_slot_in);
  fast_weights_.decay_global = std::move(decay_global);
  fast_weights_.decay_slot = std::move(decay_slot);
  fast_weights_.w_global_out = std::move(w_global_out);
  fast_weights_.w_slot_out = std::move(w_slot_out);
  fast_weights_.bias = bias;

  slow_weights_ = fast_weights_;
  grad_weights_ = OnlineLearningWeights{};
  grad_weights_.feature_dim = feature_dim;
  grad_weights_.global_dim = global_dim;
  grad_weights_.slot_dim = slot_dim;
  grad_weights_.w_global_in.assign(fast_weights_.w_global_in.size(), 0.0f);
  grad_weights_.w_slot_in.assign(fast_weights_.w_slot_in.size(), 0.0f);
  grad_weights_.decay_global.assign(fast_weights_.decay_global.size(), 0.0f);
  grad_weights_.decay_slot.assign(fast_weights_.decay_slot.size(), 0.0f);
  grad_weights_.w_global_out.assign(fast_weights_.w_global_out.size(), 0.0f);
  grad_weights_.w_slot_out.assign(fast_weights_.w_slot_out.size(), 0.0f);
  grad_weights_.bias = 0.0f;
  weights_initialized_ = true;
  actions_since_sgd_ = 0;
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

const OnlineLearningWeights& OnlineLearningController::fast_weights() const {
  return fast_weights_;
}

const OnlineLearningWeights& OnlineLearningController::slow_weights() const {
  return slow_weights_;
}

bool OnlineLearningController::weights_initialized() const {
  return weights_initialized_;
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

bool OnlineLearningController::ensure_default_weights_for_entry(
    const ActionHistoryEntry& entry) {
  if (weights_initialized_) {
    return true;
  }
  if (entry.features.empty() || entry.global_state.empty() ||
      entry.slot_state.empty()) {
    return false;
  }
  const uint32_t feature_dim = static_cast<uint32_t>(entry.features.size());
  const uint32_t global_dim = static_cast<uint32_t>(entry.global_state.size());
  const uint32_t slot_dim = static_cast<uint32_t>(entry.slot_state.size());
  const float out_scale = 1.0f / std::sqrt(
      static_cast<float>(global_dim + slot_dim));
  initialize_weights(
      feature_dim,
      global_dim,
      slot_dim,
      std::vector<float>(
          checked_matrix_size(global_dim, feature_dim, "w_global_in"),
          0.0f),
      std::vector<float>(
          checked_matrix_size(slot_dim, feature_dim, "w_slot_in"),
          0.0f),
      std::vector<float>(global_dim, 0.0f),
      std::vector<float>(slot_dim, 0.0f),
      std::vector<float>(global_dim, out_scale),
      std::vector<float>(slot_dim, out_scale),
      0.0f);
  return true;
}

bool OnlineLearningController::weights_match_entry(
    const ActionHistoryEntry& entry) const {
  return weights_initialized_ &&
      entry.features.size() == fast_weights_.feature_dim &&
      entry.global_state.size() == fast_weights_.global_dim &&
      entry.slot_state.size() == fast_weights_.slot_dim;
}

void OnlineLearningController::accumulate_backward(
    const ActionHistoryEntry& entry,
    float credit) {
  const uint32_t fdim = fast_weights_.feature_dim;
  const uint32_t gdim = fast_weights_.global_dim;
  const uint32_t sdim = fast_weights_.slot_dim;
  std::vector<float> out_global(gdim, 0.0f);
  std::vector<float> out_slot(sdim, 0.0f);

  for (uint32_t i = 0; i < gdim; ++i) {
    float out = fast_weights_.decay_global[i] * entry.global_state[i];
    const std::size_t row = static_cast<std::size_t>(i) * fdim;
    for (uint32_t j = 0; j < fdim; ++j) {
      out += fast_weights_.w_global_in[row + j] * entry.features[j];
    }
    out_global[i] = out;
  }

  for (uint32_t i = 0; i < sdim; ++i) {
    float out = fast_weights_.decay_slot[i] * entry.slot_state[i];
    const std::size_t row = static_cast<std::size_t>(i) * fdim;
    for (uint32_t j = 0; j < fdim; ++j) {
      out += fast_weights_.w_slot_in[row + j] * entry.features[j];
    }
    out_slot[i] = out;
  }

  // Positive credit should increase the future selection logit; SGD minimizes,
  // so the local scalar objective is L = -credit * logit.
  const float upstream = -credit;
  grad_weights_.bias += upstream;

  for (uint32_t i = 0; i < gdim; ++i) {
    const float grad_out = upstream * fast_weights_.w_global_out[i];
    grad_weights_.w_global_out[i] += upstream * out_global[i];
    grad_weights_.decay_global[i] += grad_out * entry.global_state[i];
    const std::size_t row = static_cast<std::size_t>(i) * fdim;
    for (uint32_t j = 0; j < fdim; ++j) {
      grad_weights_.w_global_in[row + j] += grad_out * entry.features[j];
    }
  }

  for (uint32_t i = 0; i < sdim; ++i) {
    const float grad_out = upstream * fast_weights_.w_slot_out[i];
    grad_weights_.w_slot_out[i] += upstream * out_slot[i];
    grad_weights_.decay_slot[i] += grad_out * entry.slot_state[i];
    const std::size_t row = static_cast<std::size_t>(i) * fdim;
    for (uint32_t j = 0; j < fdim; ++j) {
      grad_weights_.w_slot_in[row + j] += grad_out * entry.features[j];
    }
  }
}

void OnlineLearningController::maybe_apply_sgd() {
  if (!weights_initialized_ || actions_since_sgd_ < sgd_interval_) {
    return;
  }
  apply_sgd_to_vector(
      sgd_, fast_weights_.w_global_in, grad_weights_.w_global_in);
  apply_sgd_to_vector(sgd_, fast_weights_.w_slot_in, grad_weights_.w_slot_in);
  apply_sgd_to_vector(
      sgd_, fast_weights_.decay_global, grad_weights_.decay_global);
  apply_sgd_to_vector(sgd_, fast_weights_.decay_slot, grad_weights_.decay_slot);
  apply_sgd_to_vector(
      sgd_, fast_weights_.w_global_out, grad_weights_.w_global_out);
  apply_sgd_to_vector(sgd_, fast_weights_.w_slot_out, grad_weights_.w_slot_out);
  sgd_.apply(&fast_weights_.bias, &grad_weights_.bias, 1);
  grad_weights_.bias = 0.0f;
  actions_since_sgd_ = 0;
  ++telemetry_.sgd_steps;
}

void OnlineLearningController::maybe_blend_slow() {
  if (!fast_slow_.should_blend()) {
    return;
  }
  if (!weights_initialized_) {
    return;
  }
  blend_vector(fast_slow_, slow_weights_.w_global_in, fast_weights_.w_global_in);
  blend_vector(fast_slow_, slow_weights_.w_slot_in, fast_weights_.w_slot_in);
  blend_vector(
      fast_slow_, slow_weights_.decay_global, fast_weights_.decay_global);
  blend_vector(fast_slow_, slow_weights_.decay_slot, fast_weights_.decay_slot);
  blend_vector(
      fast_slow_, slow_weights_.w_global_out, fast_weights_.w_global_out);
  blend_vector(fast_slow_, slow_weights_.w_slot_out, fast_weights_.w_slot_out);
  fast_slow_.blend(&slow_weights_.bias, &fast_weights_.bias, 1);
  ++telemetry_.ema_blends;
}
