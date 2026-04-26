#include "credit.h"

#include <cmath>

float recency_decay(float reward, uint64_t T, uint64_t P, float gamma) {
  if (T < P) {
    return reward;
  }
  const uint64_t lag = T - P;
  const double factor = std::pow(static_cast<double>(gamma), static_cast<double>(lag));
  return static_cast<float>(static_cast<double>(reward) * factor);
}
