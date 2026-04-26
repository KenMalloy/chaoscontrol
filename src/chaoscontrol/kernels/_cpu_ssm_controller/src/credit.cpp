#include "credit.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

float recency_decay(float reward, uint64_t T, uint64_t P, float gamma) {
  if (T < P) {
    return reward;
  }
  const uint64_t lag = T - P;
  const double factor = std::pow(static_cast<double>(gamma), static_cast<double>(lag));
  return static_cast<float>(static_cast<double>(reward) * factor);
}

float gerber_weight(float L_v, float L_current, float H) {
  const float threshold = std::max(H, 0.0f);
  if (std::fabs(L_v) <= threshold || std::fabs(L_current) <= threshold) {
    return 0.0f;
  }
  if ((L_v > 0.0f) != (L_current > 0.0f)) {
    return 0.0f;
  }
  return 1.0f;
}

std::vector<CreditedAction> attribute_credit(
    uint64_t outcome_gpu_step,
    float reward,
    float current_logit,
    const std::vector<ActionHistoryEntry>& history,
    const std::vector<float>& sigma_by_action_type,
    float gamma,
    float gerber_c) {
  std::vector<CreditedAction> credited;
  credited.reserve(history.size());

  for (const ActionHistoryEntry& entry : history) {
    const std::size_t action_type =
        static_cast<std::size_t>(entry.action_type);
    if (action_type >= sigma_by_action_type.size()) {
      throw std::invalid_argument(
          "sigma_by_action_type missing action_type " +
          std::to_string(action_type));
    }

    const float H = gerber_c * sigma_by_action_type[action_type];
    const float recency_adjusted =
        recency_decay(reward, outcome_gpu_step, entry.gpu_step, gamma);
    const float gerber =
        gerber_weight(entry.output_logit, current_logit, H);
    const float rank_factor = entry.action_type == 1
        ? 1.0f / static_cast<float>(entry.selected_rank + 1)
        : 1.0f;

    credited.push_back(CreditedAction{
        entry,
        recency_adjusted * gerber * rank_factor,
    });
  }

  return credited;
}

RollingStddev::RollingStddev(float decay)
    : decay_(static_cast<double>(decay)),
      ema_(0.0),
      ema_sq_(0.0),
      count_(0) {
  if (decay < 0.0f || decay >= 1.0f) {
    throw std::invalid_argument("RollingStddev decay must be in [0, 1)");
  }
}

void RollingStddev::update(float x) {
  const double value = static_cast<double>(x);
  const double alpha = 1.0 - decay_;
  ema_ = decay_ * ema_ + alpha * value;
  ema_sq_ = decay_ * ema_sq_ + alpha * value * value;
  ++count_;
}

float RollingStddev::mean() const {
  if (count_ == 0) {
    return 0.0f;
  }
  const double correction = 1.0 - std::pow(decay_, static_cast<double>(count_));
  if (correction <= 0.0) {
    return 0.0f;
  }
  return static_cast<float>(ema_ / correction);
}

float RollingStddev::stddev() const {
  if (count_ == 0) {
    return 0.0f;
  }
  const double correction = 1.0 - std::pow(decay_, static_cast<double>(count_));
  if (correction <= 0.0) {
    return 0.0f;
  }
  const double mean = ema_ / correction;
  const double second_moment = ema_sq_ / correction;
  const double variance = std::max(0.0, second_moment - mean * mean);
  return static_cast<float>(std::sqrt(variance));
}

uint64_t RollingStddev::count() const {
  return count_;
}
