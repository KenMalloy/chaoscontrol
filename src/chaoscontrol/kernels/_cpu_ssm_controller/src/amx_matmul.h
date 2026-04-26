#pragma once

#include <ATen/ATen.h>

namespace chaoscontrol::amx {

bool amx_bf16_kernel_available();

at::Tensor amx_bf16_matmul(const at::Tensor& a, const at::Tensor& b);

}  // namespace chaoscontrol::amx
