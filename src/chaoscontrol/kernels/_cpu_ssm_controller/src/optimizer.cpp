#include "optimizer.h"

SgdStep::SgdStep(float lr) : lr_(lr) {}

void SgdStep::apply(
    float* weights,
    const float* gradients,
    std::size_t n) const {
  for (std::size_t i = 0; i < n; ++i) {
    weights[i] -= lr_ * gradients[i];
  }
}

float SgdStep::lr() const {
  return lr_;
}
