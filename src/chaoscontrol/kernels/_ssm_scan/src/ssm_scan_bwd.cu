// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Diag SSM scan backward kernel.
//
// Given upstream grad_state (dL/ds, shape (B, T, D), dtype = update dtype),
// the saved decay and forward state tensors, produce grad_decay and
// grad_update (both (B, T, D)).
//
// Recurrence (in reverse time):
//   carry = 0
//   for t = T-1 .. 0:
//       G_t            = grad_state[t] + carry           (full dL/ds[t] through downstream)
//       grad_decay[t]  = G_t * state[t-1]                (state[-1] := 0)
//       grad_update[t] = G_t
//       carry          = decay[t] * G_t                  (propagated to t-1)
//
// Layout: same (B, T, D) row-major per-thread lane as the forward kernel.
// Each thread owns one (b, d) lane and serial-loops over T in reverse,
// carrying the fp32 accumulator in a register.
//
// State[t-1] is read from the `state_fp32` input tensor (always fp32,
// written by the forward kernel alongside the lossy `out`) with a t>0
// guard — the t=0 write of grad_decay is a literal 0. Never issue an
// out-of-bounds load at (t-1)=-1.
//
// Fp32 accumulator regardless of input dtype; outputs cast back to their
// storage dtype at the store. Same dtype dispatch as forward.
//
// Reading fp32 state (not the lossy bf16/fp16 output) means backward
// differentiates through the TRUE fp32 recurrence, not a quantized
// surrogate. At bf16 this closes a ~3e-3 pessimistic drift that the
// old backward inherited from the output-dtype roundtrip.
//
// Traffic per (b, d, t): 3 loads (grad_state, decay, state[t-1]) + 2
// stores (grad_decay, grad_update). Forward does 2 loads + 1 store. So
// bwd is ~1.6× fwd bytes; since fwd was HBM-bound at ~0.35 ms, bwd's
// lower bound is ~0.56 ms for the same shape. Still well under compile's
// ~4-8 ms full-step overhead.

#include "ssm_scan.h"

#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cc_ssm_scan {

namespace {

// Mirrors the dtype tag-dispatch in ssm_scan_fwd.cu so both kernels
// share the same load/store surface. Defined in an anonymous namespace
// so the linker doesn't try to merge them with the fwd copy; the two
// translation units are independent.
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

// Naive per-thread reverse serial scan. grid = (ceil(D / BlockX), B);
// thread owns the (blockIdx.y, blockIdx.x * BlockX + threadIdx.x) lane
// for the full reverse-T sweep.
//
// `state_fp32` is always fp32 (the forward kernel writes it verbatim
// from the register accumulator). Reading it directly preserves the
// fp32 state exactly — no template on UpdateTag for this load.
template <typename DecayTag, typename UpdateTag, int BlockX>
__global__ void ssm_scan_bwd_kernel(
    const typename ElemOps<UpdateTag>::Elem* __restrict__ grad_state,
    const typename ElemOps<DecayTag>::Elem* __restrict__ decay,
    const float* __restrict__ state_fp32,
    typename ElemOps<DecayTag>::Elem* __restrict__ grad_decay,
    typename ElemOps<UpdateTag>::Elem* __restrict__ grad_update,
    int B, int T, int D) {
    const int d = blockIdx.x * BlockX + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= D) return;

    // Base offset into (B, T, D). Each step advances by D.
    const int64_t base = int64_t(b) * int64_t(T) * int64_t(D) + int64_t(d);

    float carry = 0.0f;
    // Reverse time loop. t=T-1 .. 0.
    for (int t = T - 1; t >= 0; --t) {
        const int64_t idx = base + int64_t(t) * int64_t(D);
        const float gs = ElemOps<UpdateTag>::load(grad_state + idx);
        const float dec = ElemOps<DecayTag>::load(decay + idx);
        const float G = gs + carry;

        // grad_update[t] = G.
        ElemOps<UpdateTag>::store(grad_update + idx, G);

        // grad_decay[t] = G * state[t-1]; state[-1] := 0.
        float s_prev = 0.0f;
        if (t > 0) {
            const int64_t idx_prev = idx - int64_t(D);
            s_prev = state_fp32[idx_prev];
        }
        ElemOps<DecayTag>::store(grad_decay + idx, G * s_prev);

        // carry propagates backward through the decay chain.
        carry = dec * G;
    }
}

}  // namespace

// Launch dispatch. Dtype combos mirror the forward kernel:
//   (decay, update) in { (bf16, bf16), (bf16, fp16), (fp16, fp16),
//                         (fp32, fp32), (fp32, bf16), (fp32, fp16) }.
// grad_state shares update_dtype; grad_update matches update_dtype;
// grad_decay matches decay_dtype. `state_fp32` is always fp32 regardless
// of the (decay, update) tuple. The (fp32, bf16) combo is the autocast
// bf16 production path — see ssm_scan_fwd.cu for the full rationale.
void launch_ssm_scan_bwd(
    const void* grad_state,
    const void* decay,
    const float* state_fp32,
    void* grad_decay,
    void* grad_update,
    int B, int T, int D,
    at::ScalarType decay_dtype,
    at::ScalarType update_dtype,
    cudaStream_t stream) {
    constexpr int BlockX = 128;
    const int grid_x = (D + BlockX - 1) / BlockX;
    dim3 grid(grid_x, B);
    dim3 block(BlockX);

    const bool decay_bf16 = (decay_dtype == at::kBFloat16);
    const bool decay_fp16 = (decay_dtype == at::kHalf);
    const bool decay_fp32 = (decay_dtype == at::kFloat);
    const bool update_bf16 = (update_dtype == at::kBFloat16);
    const bool update_fp16 = (update_dtype == at::kHalf);
    const bool update_fp32 = (update_dtype == at::kFloat);

    if (decay_bf16 && update_bf16) {
        ssm_scan_bwd_kernel<TagBF16, TagBF16, BlockX><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_state),
            reinterpret_cast<const __nv_bfloat16*>(decay),
            state_fp32,
            reinterpret_cast<__nv_bfloat16*>(grad_decay),
            reinterpret_cast<__nv_bfloat16*>(grad_update),
            B, T, D);
    } else if (decay_bf16 && update_fp16) {
        ssm_scan_bwd_kernel<TagBF16, TagFP16, BlockX><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_state),
            reinterpret_cast<const __nv_bfloat16*>(decay),
            state_fp32,
            reinterpret_cast<__nv_bfloat16*>(grad_decay),
            reinterpret_cast<__half*>(grad_update),
            B, T, D);
    } else if (decay_fp16 && update_fp16) {
        ssm_scan_bwd_kernel<TagFP16, TagFP16, BlockX><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_state),
            reinterpret_cast<const __half*>(decay),
            state_fp32,
            reinterpret_cast<__half*>(grad_decay),
            reinterpret_cast<__half*>(grad_update),
            B, T, D);
    } else if (decay_fp32 && update_fp32) {
        ssm_scan_bwd_kernel<TagFP32, TagFP32, BlockX><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(grad_state),
            reinterpret_cast<const float*>(decay),
            state_fp32,
            reinterpret_cast<float*>(grad_decay),
            reinterpret_cast<float*>(grad_update),
            B, T, D);
    } else if (decay_fp32 && update_bf16) {
        // Autocast bf16 production path. grad_state/grad_update are
        // bf16; grad_decay is fp32 (matches decay). See ssm_scan_fwd.cu
        // for the full dtype-flow rationale.
        ssm_scan_bwd_kernel<TagFP32, TagBF16, BlockX><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_state),
            reinterpret_cast<const float*>(decay),
            state_fp32,
            reinterpret_cast<float*>(grad_decay),
            reinterpret_cast<__nv_bfloat16*>(grad_update),
            B, T, D);
    } else if (decay_fp32 && update_fp16) {
        // fp16 analogue — grad_state/grad_update are fp16; grad_decay
        // is fp32.
        ssm_scan_bwd_kernel<TagFP32, TagFP16, BlockX><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_state),
            reinterpret_cast<const float*>(decay),
            state_fp32,
            reinterpret_cast<float*>(grad_decay),
            reinterpret_cast<__half*>(grad_update),
            B, T, D);
    } else {
        // Unreachable under the binding-side whitelist; host-side
        // TORCH_CHECK mirrors ssm_scan_fwd.cu's contract. See that file
        // for the ScalarType stringify workaround.
        TORCH_CHECK(false,
                    "cc_ssm_scan bwd: unsupported (decay, update) dtype combo "
                    "reached the CUDA dispatcher; binding should have "
                    "rejected it upstream. Got decay=",
                    static_cast<int>(decay_dtype),
                    " update=", static_cast<int>(update_dtype),
                    " (see c10/core/ScalarType.h for enum mapping)");
    }
}

}  // namespace cc_ssm_scan
