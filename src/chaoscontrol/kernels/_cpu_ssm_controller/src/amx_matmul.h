#pragma once

#include <ATen/ATen.h>

namespace chaoscontrol::amx {

bool amx_bf16_kernel_available();

at::Tensor amx_bf16_matmul(const at::Tensor& a, const at::Tensor& b);

// VNNI-rearrange B (K x N bf16) into the (K/2 x 2N bf16) layout that
// _tile_dpbf16ps consumes. Compiled on every platform so the packing
// can be unit-tested locally on arm64 (the AMX execution itself is
// gated, but the buffer layout logic is portable).
at::Tensor pack_b_vnni(const at::Tensor& b);

}  // namespace chaoscontrol::amx
