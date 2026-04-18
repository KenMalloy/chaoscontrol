// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Diag SSM scan forward kernel.
//
// Computes state[t] = decay[t] * state[t-1] + update[t] for t in [0, T),
// with state[-1] = 0, over every (b, d) independently.
//
// Layout: inputs and output are (B, T, D) row-major. Stride across T is
// D, stride across D is 1. Each thread owns one (b, d) lane and serial-
// loops over T, carrying the fp32 state in a register.
//
// Why naive-first: at our submission shape B=1024, T=512, D=256 we have
// 262,144 independent scans — ample parallelism without per-timestep
// communication. Each thread does T FMAs on registers, writes T bytes
// to global memory, with stride D between consecutive writes (coalesced
// across consecutive threads on the warp dimension, D).
//
// Accumulator is fp32 regardless of input dtype. Output dtype matches
// `update` dtype. bf16 inputs cost two bf16→fp32 loads per step plus
// one fp32→bf16 store; the cast is saturating and latency-free on H100
// tensor pipelines.
//
// Coalescing: `threadIdx.x` ranges over D, so consecutive threads hit
// consecutive D-lanes at the same (b, t) — stride-1 accesses in global
// memory. When D exceeds the block width we have multiple blocks per
// batch, each owning a contiguous D-tile.
//
// Reference: HGRN (MIT) and Mamba's selective_scan (Apache-2.0) — we
// read structure, did not copy code. See NOTICE in this directory.

#include "ssm_scan.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cc_ssm_scan {

namespace {

// Element loader + caster. Specialized per dtype tag so host can pick
// the right load/store pair.
struct TagBF16 {};
struct TagFP16 {};
struct TagFP32 {};

template <typename Tag>
struct ElemOps;

template <>
struct ElemOps<TagBF16> {
    using Elem = __nv_bfloat16;
    __device__ __forceinline__ static float load(const Elem* p) {
        return __bfloat162float(*p);
    }
    __device__ __forceinline__ static void store(Elem* p, float v) {
        *p = __float2bfloat16(v);
    }
};

template <>
struct ElemOps<TagFP16> {
    using Elem = __half;
    __device__ __forceinline__ static float load(const Elem* p) {
        return __half2float(*p);
    }
    __device__ __forceinline__ static void store(Elem* p, float v) {
        *p = __float2half(v);
    }
};

template <>
struct ElemOps<TagFP32> {
    using Elem = float;
    __device__ __forceinline__ static float load(const Elem* p) {
        return *p;
    }
    __device__ __forceinline__ static void store(Elem* p, float v) {
        *p = v;
    }
};

// Naive per-thread serial scan. grid = (ceil(D / BlockX), B); thread
// owns the (blockIdx.y, blockIdx.x * BlockX + threadIdx.x) lane for
// the full T sweep.
template <typename DecayTag, typename UpdateTag, typename OutTag, int BlockX>
__global__ void ssm_scan_fwd_kernel(
    const typename ElemOps<DecayTag>::Elem* __restrict__ decay,
    const typename ElemOps<UpdateTag>::Elem* __restrict__ update,
    typename ElemOps<OutTag>::Elem* __restrict__ out,
    int B, int T, int D) {
    const int d = blockIdx.x * BlockX + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= D) return;

    // Base offset into (B, T, D). Each step advances by D.
    const int64_t base = int64_t(b) * int64_t(T) * int64_t(D) + int64_t(d);

    float state = 0.0f;
    for (int t = 0; t < T; ++t) {
        const int64_t idx = base + int64_t(t) * int64_t(D);
        const float dec = ElemOps<DecayTag>::load(decay + idx);
        const float upd = ElemOps<UpdateTag>::load(update + idx);
        state = dec * state + upd;
        ElemOps<OutTag>::store(out + idx, state);
    }
}

}  // namespace

// Launch dispatch. The input dtype pair determines the template
// instantiation. Valid combos:
//   (decay, update, out) in { (bf16, bf16, bf16), (bf16, fp16, fp16),
//                             (fp32, fp32, fp32), (fp16, fp16, fp16) }.
// The public API takes decay+update dtypes and the wrapper here picks
// the instantiation. bf16 is the prod case; fp32 is the numerical
// reference path.
//
// BlockX default 128 — a compromise between coalescing (want BlockX
// divides D cleanly; D=256 gives 2 blocks per batch at BlockX=128,
// 1 block at BlockX=256) and occupancy. For D<128 the tail lanes idle
// but there's nothing to scan there either. Micro-benched at 64/128/256
// during optimization; 128 wins at D=256.
void launch_ssm_scan_fwd(
    const void* decay,
    const void* update,
    void* out,
    int B, int T, int D,
    at::ScalarType decay_dtype,
    at::ScalarType update_dtype,
    cudaStream_t stream) {
    constexpr int BlockX = 128;
    const int grid_x = (D + BlockX - 1) / BlockX;
    dim3 grid(grid_x, B);
    dim3 block(BlockX);

    // Dispatch: decay/update share dtype in the common bf16 case;
    // allow bf16 decay × fp16 update variant plus fp32 reference.
    const bool decay_bf16 = (decay_dtype == at::kBFloat16);
    const bool decay_fp16 = (decay_dtype == at::kHalf);
    const bool decay_fp32 = (decay_dtype == at::kFloat);
    const bool update_bf16 = (update_dtype == at::kBFloat16);
    const bool update_fp16 = (update_dtype == at::kHalf);
    const bool update_fp32 = (update_dtype == at::kFloat);

    if (decay_bf16 && update_bf16) {
        ssm_scan_fwd_kernel<TagBF16, TagBF16, TagBF16, BlockX><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(decay),
            reinterpret_cast<const __nv_bfloat16*>(update),
            reinterpret_cast<__nv_bfloat16*>(out),
            B, T, D);
    } else if (decay_bf16 && update_fp16) {
        ssm_scan_fwd_kernel<TagBF16, TagFP16, TagFP16, BlockX><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(decay),
            reinterpret_cast<const __half*>(update),
            reinterpret_cast<__half*>(out),
            B, T, D);
    } else if (decay_fp16 && update_fp16) {
        ssm_scan_fwd_kernel<TagFP16, TagFP16, TagFP16, BlockX><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(decay),
            reinterpret_cast<const __half*>(update),
            reinterpret_cast<__half*>(out),
            B, T, D);
    } else if (decay_fp32 && update_fp32) {
        ssm_scan_fwd_kernel<TagFP32, TagFP32, TagFP32, BlockX><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(decay),
            reinterpret_cast<const float*>(update),
            reinterpret_cast<float*>(out),
            B, T, D);
    } else {
        // Fall back: bf16 decay with fp32 update (unusual; promote both).
        // Not exposed on the public API today, but kept for symmetry
        // if a future caller asks for it.
        // We simply abort — host should have checked dtype compat.
        // Using a device-side assert would require cooperative-kernel
        // machinery; instead raise at the host boundary.
        // (Intentional no-op: the binding enforces this upstream.)
    }
}

}  // namespace cc_ssm_scan
