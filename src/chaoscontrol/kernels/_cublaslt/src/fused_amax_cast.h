// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Fused amax-update + bf16→fp8 cast CUDA kernel declarations (host-visible).
// Implementations live in fused_amax_cast.cu.
//
// Phase 3 of the TE fork: the Python-orchestrated
// ``torch.maximum(pending, x.abs().amax()) + (x / scale).to(e4m3)`` chain
// burns ~4 kernel launches + dispatcher overhead per Linear call. This
// file declares a single kernel that does both in one pass and one
// launch. The cast uses the CURRENT scale (passed in, stale at call
// time — refreshed by flush_amax_history once per optimizer step);
// the pending buffer is atomically updated with the new amax so that
// the NEXT flush can roll it into the ring-buffer history.
//
// Atomic on fp32: we atomicMax the integer bit-pattern. For non-negative
// IEEE 754 floats the ordering matches int ordering, so atomicMax on
// (int*)(pending) is monotonic and correct for amax accumulation.

#pragma once

#include <c10/core/ScalarType.h>
#include <cuda_runtime.h>

namespace cc_cublaslt {

// Launches the fused amax+cast kernel on ``stream``.
//
// Inputs:
//   x_bf16    : device pointer to [n_elem] bf16 data (contiguous in memory).
//   scale_ptr : device pointer to fp32 scalar. Cast uses this scale:
//               out_fp8 = (x / scale).to(out_dtype).
//   pending   : device pointer to fp32 scalar. atomicMax-updated in place
//               with the amax of |x_bf16|.
//   out_fp8   : device pointer to [n_elem] fp8 data (E4M3 or E5M2
//               depending on out_dtype). Pre-allocated by caller.
//   out_dtype : at::kFloat8_e4m3fn or at::kFloat8_e5m2.
//   n_elem    : number of elements.
//   stream    : CUDA stream to launch on.
//
// No host sync, no host-side allocations.
void launch_fused_amax_cast_bf16(
    const void* x_bf16,
    const void* scale_ptr,
    void* pending,
    void* out_fp8,
    c10::ScalarType out_dtype,
    int64_t n_elem,
    cudaStream_t stream);

// One-shot flush: roll ``history`` left by one (drop history[0], append
// ``pending`` at history[-1]), compute ``max(history)``, write
// ``max_history / max_rep`` into ``scale`` (or 1.0 if max_history == 0),
// and zero ``pending``. All in a single kernel launch — replaces 6
// Python-level ops per tensor. ``history_len`` must be <= 1024.
void launch_fused_flush_amax(
    float* history,     // device [history_len]
    float* pending,     // device [1]
    float* scale,       // device [1] (output)
    int history_len,
    float max_rep,      // 448.0 for E4M3, 57344.0 for E5M2
    cudaStream_t stream);

// Gx-pending variant: just roll + zero; no scale produced (diagnostic
// path). history_len <= 1024.
void launch_fused_flush_amax_diagnostic(
    float* history,
    float* pending,
    int history_len,
    cudaStream_t stream);

// Variant that reads from a strided bf16 tensor treated as a 2D view
// (rows × cols with stride (col_stride, 1)) but produces a COLUMN-MAJOR
// fp8 output with stride (1, rows). Used for the bwd-w path where the
// grad_y must be materialized as [N, M] row-major (i.e. transposed
// from [M, N]) AND the x operand must be [M, K] column-major (i.e. the
// transpose of the [M, K] row-major saved fp8 tensor).
//
// For the M×N → N×M transpose case, pass rows=N, cols=M, row_stride_in=
// 1, col_stride_in=N (i.e. reading the M×N tensor column-by-column in
// the transposed access pattern). The kernel writes out_fp8 in row-major
// [N, M] layout — i.e. contiguous rows of length M.
void launch_fused_amax_cast_transpose_bf16(
    const void* x_bf16,
    const void* scale_ptr,
    void* pending,
    void* out_fp8,
    c10::ScalarType out_dtype,
    int64_t rows_out,         // output rows (logical [rows_out, cols_out])
    int64_t cols_out,         // output cols
    int64_t in_row_stride,    // stride in elements from [i,j] to [i+1,j] in INPUT
    int64_t in_col_stride,    // stride in elements from [i,j] to [i,j+1] in INPUT
    cudaStream_t stream);

}  // namespace cc_cublaslt
