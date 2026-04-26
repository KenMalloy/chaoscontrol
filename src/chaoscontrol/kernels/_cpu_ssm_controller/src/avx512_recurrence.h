#pragma once

#include <torch/extension.h>

namespace chaoscontrol::avx512 {

bool avx512_recurrence_kernel_available();

void avx512_diagonal_recurrence(
    const at::Tensor& decay,
    const at::Tensor& x,
    at::Tensor h);

}  // namespace chaoscontrol::avx512
