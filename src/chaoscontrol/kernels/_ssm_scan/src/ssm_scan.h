// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Host-side launch declarations for ssm_scan_fwd.cu and ssm_scan_bwd.cu.
#pragma once

#include <c10/core/ScalarType.h>
#include <cuda_runtime.h>

namespace cc_ssm_scan {

// Launch the forward scan kernel. Host-allocated `out` must have the
// same shape (B, T, D) and match the `update` dtype. `state_fp32` must
// also have shape (B, T, D), always fp32 regardless of input dtype;
// it's the exact register accumulator snapshot per step and is what
// the backward kernel consumes (the output tensor is lossy for bf16/
// fp16 and would quantize the gradient surrogate otherwise). Fp32
// accumulator is always used internally.
void launch_ssm_scan_fwd(
    const void* decay,
    const void* update,
    void* out,
    float* state_fp32,
    int B, int T, int D,
    at::ScalarType decay_dtype,
    at::ScalarType update_dtype,
    cudaStream_t stream);

// Launch the backward scan kernel.
//
// Inputs:
//   grad_state: (B, T, D) upstream grad w.r.t. state; dtype == update dtype
//   decay:      (B, T, D) forward decay; dtype == decay dtype
//   state_fp32: (B, T, D) forward fp32 state (saved); ALWAYS fp32
//
// Outputs (pre-allocated by caller):
//   grad_decay:  (B, T, D) dtype matches decay dtype
//   grad_update: (B, T, D) dtype matches update dtype
//
// Accumulator is fp32 internally regardless of input dtype. The fp32
// state is required (not the lossy `out` of forward) so backward
// differentiates through the true recurrence, not a quantized surrogate.
void launch_ssm_scan_bwd(
    const void* grad_state,
    const void* decay,
    const float* state_fp32,
    void* grad_decay,
    void* grad_update,
    int B, int T, int D,
    at::ScalarType decay_dtype,
    at::ScalarType update_dtype,
    cudaStream_t stream);

}  // namespace cc_ssm_scan
