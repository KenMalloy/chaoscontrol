#include "optimizer.h"

#include <stdexcept>

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

FastSlowEma::FastSlowEma(float alpha, uint64_t interval)
    : alpha_(alpha), interval_(interval), event_count_(0) {
  if (alpha < 0.0f || alpha > 1.0f) {
    throw std::invalid_argument("FastSlowEma alpha must be in [0, 1]");
  }
  if (interval == 0) {
    throw std::invalid_argument("FastSlowEma interval must be > 0");
  }
}

void FastSlowEma::tick_event() {
  ++event_count_;
}

bool FastSlowEma::should_blend() const {
  return event_count_ > 0 && event_count_ % interval_ == 0;
}

void FastSlowEma::blend(
    float* slow,
    const float* fast,
    std::size_t n) const {
  const float keep = 1.0f - alpha_;
  for (std::size_t i = 0; i < n; ++i) {
    slow[i] = keep * slow[i] + alpha_ * fast[i];
  }
}

uint64_t FastSlowEma::event_count() const {
  return event_count_;
}

float FastSlowEma::alpha() const {
  return alpha_;
}

void FastSlowEma::set_alpha(float alpha) {
  if (alpha < 0.0f || alpha > 1.0f) {
    throw std::invalid_argument("FastSlowEma alpha must be in [0, 1]");
  }
  alpha_ = alpha;
}

uint64_t FastSlowEma::interval() const {
  return interval_;
}
