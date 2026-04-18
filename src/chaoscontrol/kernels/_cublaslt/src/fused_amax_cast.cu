// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Fused amax-update + bf16→fp8 cast kernels.
//
// See ``fused_amax_cast.h`` for the wire contract. One kernel launch per
// tensor — reads bf16, writes fp8, atomically updates a scalar pending
// amax buffer. Block-level warp-shuffle reduction; single grid-level
// atomicMax on the integer bit-pattern of the non-negative fp32 amax.
//
// Design constraints held from Phase 2:
//   * No host sync; callers are free to chain into the cuBLASLt matmul
//     that follows.
//   * Cast uses the CURRENT scale (stale by one optimizer step — refreshed
//     by flush_amax_history). This matches TE's delayed-scaling recipe.
//   * Bit-pattern atomicMax: for non-negative IEEE 754 floats the int
//     ordering matches the float ordering, so __int_as_float /
//     __float_as_int round-trip an atomicMax into a correct amax
//     accumulation — no CAS loop needed.
//
// Reference-only: TE v2.13 common/util/cast.cu for the fused cast+amax
// pattern. No code was copied; our kernel authoring is clean-room.

#include "fused_amax_cast.h"

#include <c10/core/ScalarType.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cc_cublaslt {

namespace {

// Warp-level reduction of ``val`` (fp32, non-negative) into lane 0.
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// Block reduction via shared memory — up to 1024 threads supported.
template <int kBlockThreads>
__device__ __forceinline__ float block_reduce_max(float val) {
    static __shared__ float smem[kBlockThreads / 32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        const int num_warps = kBlockThreads / 32;
        val = (threadIdx.x < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_max(val);
    }
    return val;
}

// atomicMax on fp32 value via integer bit-pattern. Valid for non-negative
// inputs (amax). The int representation of a positive fp32 is monotonically
// ordered the same as the float value. NaN is treated as large-positive
// which is safe here — upstream amax already cleansed by torch's own
// producers, and a NaN amax would correctly propagate into the scale.
__device__ __forceinline__ void atomic_max_fp32_bits(float* addr, float v) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    atomicMax(addr_as_int, __float_as_int(v));
}

// bf16 → fp32 helper.
__device__ __forceinline__ float bf16_to_fp32(const __nv_bfloat16& v) {
    return __bfloat162float(v);
}

// Quantize fp32 → fp8. Selected at compile time via TagE4M3 / TagE5M2.
struct TagE4M3 {};
struct TagE5M2 {};

__device__ __forceinline__ __nv_fp8_storage_t quantize_to_fp8(
    float x, TagE4M3) {
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ __nv_fp8_storage_t quantize_to_fp8(
    float x, TagE5M2) {
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2);
}

// Core kernel: 1-D element-wise traversal. Each thread processes a single
// element per grid stride, accumulates a thread-local amax, casts the
// bf16 value to fp8 using the given dequant scale, stores the fp8 byte
// at the matching index. Block-reduce the thread-local amax; a single
// thread from each block performs the grid-level atomicMax.
template <typename Tag, int kBlockThreads>
__global__ void fused_amax_cast_1d_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    const float* __restrict__ scale_ptr,
    float* __restrict__ pending,
    __nv_fp8_storage_t* __restrict__ out_fp8,
    int64_t n_elem) {
    const float inv_scale = 1.0f / (*scale_ptr);

    float thread_max = 0.0f;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_elem; i += stride) {
        float v = bf16_to_fp32(x_bf16[i]);
        float a = fabsf(v);
        if (a > thread_max) thread_max = a;
        out_fp8[i] = quantize_to_fp8(v * inv_scale, Tag{});
    }
    float block_max = block_reduce_max<kBlockThreads>(thread_max);
    if (threadIdx.x == 0 && block_max > 0.0f && pending != nullptr) {
        atomic_max_fp32_bits(pending, block_max);
    }
}

// Transpose variant: reads a 2-D input with arbitrary strides (to support
// a transposed view) and writes out_fp8 as row-major [rows_out, cols_out].
// Used for grad_y.t() (bwd-w) and for x [M,K] col-major (bwd-w b operand).
//
// Optional column-sum side-output (``bias_grad_fp32``, nullable): when
// non-null, each thread does ``atomicAdd(&bias_grad_fp32[r], v)`` in
// addition to the fp8 cast, producing the bf16-reduced-sum-over-cols
// needed for the fp8 E5M2×E4M3 bias-grad path (cf. task 25 —
// CUBLASLT_EPILOGUE_BGRADB unsupported by NVIDIA for this dtype pair).
// Caller must pre-zero the buffer; we only accumulate.
//
// Atomic contention: in the submission regime (rows_out=N, cols_out=M,
// M >> blockDim), each block's threads mostly hit the same ``r``, so
// up to blockDim threads contend on one address. H100's L2 atomics
// absorb that at a few-µs-per-block cost — small vs the ~50-100 µs
// savings we get from eliminating the Python grad_y.sum(0) reduction.
// A block-local smem reduction to drop contention to one atomicAdd
// per block per ``r`` is a straightforward optimization if the bench
// shows it as a hot lane.
template <typename Tag, int kBlockThreads>
__global__ void fused_amax_cast_transpose_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    const float* __restrict__ scale_ptr,
    float* __restrict__ pending,
    float* __restrict__ bias_grad_fp32,
    __nv_fp8_storage_t* __restrict__ out_fp8,
    int64_t rows_out,
    int64_t cols_out,
    int64_t in_row_stride,
    int64_t in_col_stride) {
    const float inv_scale = 1.0f / (*scale_ptr);
    const int64_t n_total = rows_out * cols_out;

    float thread_max = 0.0f;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t k = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         k < n_total; k += stride) {
        const int64_t r = k / cols_out;
        const int64_t c = k - r * cols_out;
        const int64_t in_idx = r * in_row_stride + c * in_col_stride;
        const float v = bf16_to_fp32(x_bf16[in_idx]);
        const float a = fabsf(v);
        if (a > thread_max) thread_max = a;
        out_fp8[k] = quantize_to_fp8(v * inv_scale, Tag{});
        if (bias_grad_fp32 != nullptr) {
            atomicAdd(&bias_grad_fp32[r], v);
        }
    }
    float block_max = block_reduce_max<kBlockThreads>(thread_max);
    if (threadIdx.x == 0 && block_max > 0.0f && pending != nullptr) {
        atomic_max_fp32_bits(pending, block_max);
    }
}

constexpr int kThreadsPerBlock = 256;

int grid_for_elements(int64_t n_elem) {
    const int64_t blocks = (n_elem + kThreadsPerBlock - 1) / kThreadsPerBlock;
    // Cap at a reasonable upper bound; kernel is grid-strided.
    const int64_t cap = 4096;
    return static_cast<int>(blocks < cap ? blocks : cap);
}

}  // namespace

void launch_fused_amax_cast_bf16(
    const void* x_bf16,
    const void* scale_ptr,
    void* pending,
    void* out_fp8,
    c10::ScalarType out_dtype,
    int64_t n_elem,
    cudaStream_t stream) {
    if (n_elem == 0) return;
    const int grid = grid_for_elements(n_elem);
    const auto* x_ptr = static_cast<const __nv_bfloat16*>(x_bf16);
    const auto* s_ptr = static_cast<const float*>(scale_ptr);
    auto* p_ptr = static_cast<float*>(pending);
    auto* o_ptr = static_cast<__nv_fp8_storage_t*>(out_fp8);
    if (out_dtype == at::kFloat8_e4m3fn) {
        fused_amax_cast_1d_kernel<TagE4M3, kThreadsPerBlock>
            <<<grid, kThreadsPerBlock, 0, stream>>>(
                x_ptr, s_ptr, p_ptr, o_ptr, n_elem);
    } else if (out_dtype == at::kFloat8_e5m2) {
        fused_amax_cast_1d_kernel<TagE5M2, kThreadsPerBlock>
            <<<grid, kThreadsPerBlock, 0, stream>>>(
                x_ptr, s_ptr, p_ptr, o_ptr, n_elem);
    } else {
        // Caller checked; this is a programmer error.
        return;
    }
}

// --------------------------------------------------------------------------
// One-shot flush kernels. Single block, up to 1024 threads (history_len).
// Replaces the Python-orchestrated torch.roll + indexing + .max() + .where()
// + .copy_() + .zero_() chain (6-7 kernels per tensor) with one launch.
// --------------------------------------------------------------------------

__global__ void fused_flush_amax_kernel(
    float* __restrict__ history,
    float* __restrict__ pending,
    float* __restrict__ scale,
    int history_len,
    float max_rep) {
    // Each thread owns one history slot. Read the shifted-in value
    // (history[tid+1], or pending for the last slot), keep it in
    // register, synchronize so all reads finish before any write, then
    // write history[tid]. This is race-free because each thread writes
    // a distinct slot.
    const int tid = threadIdx.x;
    const float pending_val = *pending;

    float v = 0.0f;
    if (tid < history_len - 1) {
        v = history[tid + 1];   // shifted value
    } else if (tid == history_len - 1) {
        v = pending_val;
    }
    // Sync: all threads must finish reading old history[] before any
    // thread writes.
    __syncthreads();
    if (tid < history_len) {
        history[tid] = v;
    }

    // Block-reduce max over [0, history_len).
    float reduced = (tid < history_len) ? v : 0.0f;
    reduced = warp_reduce_max(reduced);
    // For history_len <= 32 we're done at the warp level. For larger
    // lengths, collapse via smem.
    if (history_len > 32) {
        __shared__ float warp_maxes[32];
        const int lane = tid & 31;
        const int warp = tid >> 5;
        if (lane == 0) warp_maxes[warp] = reduced;
        __syncthreads();
        if (warp == 0) {
            const int n_warps = (history_len + 31) / 32;
            reduced = (lane < n_warps) ? warp_maxes[lane] : 0.0f;
            reduced = warp_reduce_max(reduced);
        }
    }

    if (tid == 0) {
        const float m = reduced;
        const float s = (m > 0.0f) ? (m / max_rep) : 1.0f;
        *scale = s;
        *pending = 0.0f;
    }
}

__global__ void fused_flush_amax_diagnostic_kernel(
    float* __restrict__ history,
    float* __restrict__ pending,
    int history_len) {
    const int tid = threadIdx.x;
    const float pending_val = *pending;

    float v = 0.0f;
    if (tid < history_len - 1) {
        v = history[tid + 1];
    } else if (tid == history_len - 1) {
        v = pending_val;
    }
    if (tid < history_len) {
        history[tid] = v;
    }
    if (tid == 0) {
        *pending = 0.0f;
    }
}

void launch_fused_flush_amax(
    float* history, float* pending, float* scale,
    int history_len, float max_rep, cudaStream_t stream) {
    if (history_len <= 0 || history_len > 1024) return;
    // Round block up to next warp for clean shfl.
    int block = ((history_len + 31) / 32) * 32;
    if (block < 32) block = 32;
    fused_flush_amax_kernel<<<1, block, 0, stream>>>(
        history, pending, scale, history_len, max_rep);
}

void launch_fused_flush_amax_diagnostic(
    float* history, float* pending,
    int history_len, cudaStream_t stream) {
    if (history_len <= 0 || history_len > 1024) return;
    int block = ((history_len + 31) / 32) * 32;
    if (block < 32) block = 32;
    fused_flush_amax_diagnostic_kernel<<<1, block, 0, stream>>>(
        history, pending, history_len);
}

void launch_fused_amax_cast_transpose_bf16(
    const void* x_bf16,
    const void* scale_ptr,
    void* pending,
    void* bias_grad_fp32,
    void* out_fp8,
    c10::ScalarType out_dtype,
    int64_t rows_out,
    int64_t cols_out,
    int64_t in_row_stride,
    int64_t in_col_stride,
    cudaStream_t stream) {
    const int64_t n_elem = rows_out * cols_out;
    if (n_elem == 0) return;
    const int grid = grid_for_elements(n_elem);
    const auto* x_ptr = static_cast<const __nv_bfloat16*>(x_bf16);
    const auto* s_ptr = static_cast<const float*>(scale_ptr);
    auto* p_ptr = static_cast<float*>(pending);
    auto* bg_ptr = static_cast<float*>(bias_grad_fp32);
    auto* o_ptr = static_cast<__nv_fp8_storage_t*>(out_fp8);
    if (out_dtype == at::kFloat8_e4m3fn) {
        fused_amax_cast_transpose_kernel<TagE4M3, kThreadsPerBlock>
            <<<grid, kThreadsPerBlock, 0, stream>>>(
                x_ptr, s_ptr, p_ptr, bg_ptr, o_ptr,
                rows_out, cols_out, in_row_stride, in_col_stride);
    } else if (out_dtype == at::kFloat8_e5m2) {
        fused_amax_cast_transpose_kernel<TagE5M2, kThreadsPerBlock>
            <<<grid, kThreadsPerBlock, 0, stream>>>(
                x_ptr, s_ptr, p_ptr, bg_ptr, o_ptr,
                rows_out, cols_out, in_row_stride, in_col_stride);
    } else {
        return;
    }
}

}  // namespace cc_cublaslt
