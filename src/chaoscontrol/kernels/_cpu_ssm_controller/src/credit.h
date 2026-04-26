#pragma once

#include <cstdint>

float recency_decay(float reward, uint64_t T, uint64_t P, float gamma = 0.995f);

float gerber_weight(float L_v, float L_current, float H);

class RollingStddev {
 public:
  explicit RollingStddev(float decay = 0.99f);

  void update(float x);
  float stddev() const;
  uint64_t count() const;

 private:
  double decay_;
  double ema_;
  double ema_sq_;
  uint64_t count_;
};
