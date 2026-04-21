#pragma once

#include <ATen/ATen.h>

namespace cc_lm_head_loss {

void launch_linear_ce_update_max_and_target(
    const at::Tensor& logits,
    const at::Tensor& targets,
    at::Tensor& row_max,
    at::Tensor& target_logits,
    int64_t tile_start);

void launch_linear_ce_accum_sum(
    const at::Tensor& logits,
    const at::Tensor& row_max,
    at::Tensor& row_sum);

void launch_linear_ce_update_online(
    const at::Tensor& logits,
    const at::Tensor& targets,
    at::Tensor& row_max,
    at::Tensor& row_sum,
    at::Tensor& target_logits,
    int64_t tile_start);

void launch_linear_ce_fill_grad_logits(
    const at::Tensor& logits,
    const at::Tensor& targets,
    const at::Tensor& lse,
    const at::Tensor& grad_loss,
    at::Tensor& grad_logits,
    int64_t tile_start,
    double divisor);

}  // namespace cc_lm_head_loss
