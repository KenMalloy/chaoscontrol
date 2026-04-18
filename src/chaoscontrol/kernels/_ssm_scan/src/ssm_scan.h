// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Host-side launch declarations for ssm_scan_fwd.cu.
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

}  // namespace cc_ssm_scan
