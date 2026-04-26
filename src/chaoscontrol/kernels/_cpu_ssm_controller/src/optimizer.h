#pragma once

#include <cstddef>
#include <cstdint>

class SgdStep {
 public:
  explicit SgdStep(float lr);

  void apply(float* weights, const float* gradients, std::size_t n) const;

  float lr() const;

 private:
  float lr_;
};

class FastSlowEma {
 public:
  FastSlowEma(float alpha = 0.25f, uint64_t interval = 64);

  void tick_event();
  bool should_blend() const;
  void blend(float* slow, const float* fast, std::size_t n) const;
  uint64_t event_count() const;
  float alpha() const;
  uint64_t interval() const;

 private:
  float alpha_;
  uint64_t interval_;
  uint64_t event_count_;
};
