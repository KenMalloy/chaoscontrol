// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Host-side launch declarations for ssm_scan_fwd.cu and ssm_scan_bwd.cu.
#pragma once

#include <c10/core/ScalarType.h>
#include <cuda_runtime.h>

namespace cc_ssm_scan {

// Launch the forward scan kernel. Host-allocated `out` must have the
// same shape (B, T, D) and match the `update` dtype. Accumulator is
// fp32 internally.
void launch_ssm_scan_fwd(
    const void* decay,
    const void* update,
    void* out,
    int B, int T, int D,
    at::ScalarType decay_dtype,
    at::ScalarType update_dtype,
    cudaStream_t stream);

// Launch the backward scan kernel.
//
// Inputs:
//   grad_state: (B, T, D) upstream grad w.r.t. state; dtype == update dtype
//   decay:      (B, T, D) forward decay; dtype == decay dtype
//   state:      (B, T, D) forward output (saved); dtype == update dtype
//
// Outputs (pre-allocated by caller):
//   grad_decay:  (B, T, D) dtype matches decay dtype
//   grad_update: (B, T, D) dtype matches update dtype
//
// Accumulator is fp32 internally regardless of input dtype.
void launch_ssm_scan_bwd(
    const void* grad_state,
    const void* decay,
    const void* state,
    void* grad_decay,
    void* grad_update,
    int B, int T, int D,
    at::ScalarType decay_dtype,
    at::ScalarType update_dtype,
    cudaStream_t stream);

}  // namespace cc_ssm_scan
