#pragma once

#include <cstddef>

class SgdStep {
 public:
  explicit SgdStep(float lr);

  void apply(float* weights, const float* gradients, std::size_t n) const;

  float lr() const;

 private:
  float lr_;
};
