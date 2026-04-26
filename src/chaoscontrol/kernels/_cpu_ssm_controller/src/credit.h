#pragma once

#include <cstdint>
#include <vector>

#include "action_history.h"

float recency_decay(float reward, uint64_t T, uint64_t P, float gamma = 0.995f);

float gerber_weight(float L_v, float L_current, float H);

struct CreditedAction {
  ActionHistoryEntry entry;
  float credit = 0.0f;
};

std::vector<CreditedAction> attribute_credit(
    uint64_t outcome_gpu_step,
    float reward,
    float current_logit,
    const std::vector<ActionHistoryEntry>& history,
    const std::vector<float>& sigma_by_action_type,
    float gamma = 0.995f,
    float gerber_c = 0.5f);

class RollingStddev {
 public:
  explicit RollingStddev(float decay = 0.99f);

  void update(float x);
  float mean() const;
  float stddev() const;
  uint64_t count() const;

 private:
  double decay_;
  double ema_;
  double ema_sq_;
  uint64_t count_;
};
