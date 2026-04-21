#pragma once

#include <ATen/ATen.h>

namespace cc_lm_head_loss {

void launch_rms_norm_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    at::Tensor& out,
    at::Tensor& inv_rms,
    double eps);

void launch_rms_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& inv_rms,
    at::Tensor& grad_x,
    at::Tensor& grad_weight);

}  // namespace cc_lm_head_loss
