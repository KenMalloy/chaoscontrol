// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// See descriptor_cache.h for the contract.

#include "descriptor_cache.h"

#include "workspace_cache.h"   // check_cublas + kWorkspaceBytes

#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>

namespace cc_cublaslt {

namespace {

// Local copies of the dtype translation helpers used by fp8_matmul_impl.
// We duplicate them here rather than expose them from the main .cpp file
// because the symbols are in an anonymous namespace there; keeping them
// internal to this translation unit is cleaner than teaching the cache
// to peek into the impl.
cudaDataType_t fp8_cuda_dtype(at::ScalarType s) {
    if (s == at::kFloat8_e4m3fn) return CUDA_R_8F_E4M3;
    if (s == at::kFloat8_e5m2) return CUDA_R_8F_E5M2;
    throw std::runtime_error(
        "descriptor_cache: operand dtype must be fp8 e4m3fn or e5m2");
}

cudaDataType_t out_cuda_dtype(at::ScalarType s) {
    if (s == at::kBFloat16) return CUDA_R_16BF;
    if (s == at::kHalf) return CUDA_R_16F;
    throw std::runtime_error(
        "descriptor_cache: out_dtype must be bfloat16 or float16");
}

// Build the plan for this key. Runs heuristic selection once and caches
// the chosen algo alongside the four layouts. Pref is created only for
// the heuristic call and destroyed immediately after — it's not needed
// at matmul time.
//
// Alignment is part of the cache key. Fresh allocator-backed tensors
// are typically 256B aligned, but sliced-yet-layout-valid CUDA views can
// lower the effective alignment while keeping the same (M, N, K, dtype)
// tuple. Reusing a 256B-planned heuristic for a 2B-aligned view is
// avoidable risk, so we rebuild the preference attrs for the actual
// runtime operand alignment.
std::unique_ptr<CachedGemmPlan> build_plan(
    const DescriptorKey& k,
    cublasLtHandle_t handle,
    size_t workspace_bytes) {
    auto plan = std::make_unique<CachedGemmPlan>();

    const cudaDataType_t dtype_a = fp8_cuda_dtype(k.a_dtype);
    const cudaDataType_t dtype_b = fp8_cuda_dtype(k.b_dtype);
    const cudaDataType_t dtype_d = out_cuda_dtype(k.out_dtype);

    // Layouts mirror cublaslt_fp8_matmul.cpp's convention: we swap
    // torch (a, b) into cuBLAS (B, A) so row-major D[M,N] emerges as
    // col-major D[N,M] with ld=N. A storage is [K,N] col-major ld=K;
    // B storage is [K,M] col-major ld=K; D storage is [N,M] col-major
    // ld=N. C reuses D's layout since beta=0 and no accumulator read.
    check_cublas(cublasLtMatrixLayoutCreate(&plan->A_layout, dtype_b, k.K, k.N, k.K),
                 "layout A");
    check_cublas(cublasLtMatrixLayoutCreate(&plan->B_layout, dtype_a, k.K, k.M, k.K),
                 "layout B");
    check_cublas(cublasLtMatrixLayoutCreate(&plan->D_layout, dtype_d, k.N, k.M, k.N),
                 "layout D");
    check_cublas(cublasLtMatrixLayoutCreate(&plan->C_layout, dtype_d, k.N, k.M, k.N),
                 "layout C");

    // Heuristic needs a transient op_desc with the fixed attrs set.
    // We build it here, run the heuristic, and destroy it — the same
    // shape will be built fresh at each matmul call. That's fine
    // because the algo (the expensive output) is what gets cached.
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    auto cleanup = [&]() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    };
    try {
        check_cublas(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                     "op_desc for heuristic");

        const cublasOperation_t opA = CUBLAS_OP_T;
        const cublasOperation_t opB = CUBLAS_OP_N;
        check_cublas(cublasLtMatmulDescSetAttribute(
                         op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)),
                     "TRANSA");
        check_cublas(cublasLtMatmulDescSetAttribute(
                         op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
                     "TRANSB");
        const int8_t fa = k.fast_accum ? 1 : 0;
        check_cublas(cublasLtMatmulDescSetAttribute(
                         op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fa, sizeof(fa)),
                     "FAST_ACCUM");

        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        if (k.bias_mode == BiasMode::ForwardBias) epilogue = CUBLASLT_EPILOGUE_BIAS;
        else if (k.bias_mode == BiasMode::BGradB) epilogue = CUBLASLT_EPILOGUE_BGRADB;
        check_cublas(cublasLtMatmulDescSetAttribute(
                         op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                         &epilogue, sizeof(epilogue)),
                     "EPILOGUE");

        if (k.bias_mode != BiasMode::None) {
            const cudaDataType_t bias_dtype = dtype_d;
            check_cublas(cublasLtMatmulDescSetAttribute(
                             op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                             &bias_dtype, sizeof(bias_dtype)),
                         "BIAS_DATA_TYPE");
        }

        check_cublas(cublasLtMatmulPreferenceCreate(&pref), "pref create");
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                         &workspace_bytes, sizeof(workspace_bytes)),
                     "pref workspace");
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                         &k.align_a_bytes, sizeof(k.align_a_bytes)),
                     "pref align A");
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                         &k.align_b_bytes, sizeof(k.align_b_bytes)),
                     "pref align B");
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                         &k.align_c_bytes, sizeof(k.align_c_bytes)),
                     "pref align C");
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                         &k.align_d_bytes, sizeof(k.align_d_bytes)),
                     "pref align D");

        int returned = 0;
        check_cublas(cublasLtMatmulAlgoGetHeuristic(
                         handle, op_desc,
                         plan->A_layout, plan->B_layout,
                         plan->C_layout, plan->D_layout,
                         pref, 1, &plan->heuristic, &returned),
                     "heuristic");
        if (returned == 0) {
            throw std::runtime_error(
                "descriptor_cache: no suitable algo for shape");
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();

    return plan;
}

}  // namespace

CachedGemmPlan::~CachedGemmPlan() {
    // Destroy in reverse of creation order. We accept that if the
    // driver is already torn down during interpreter shutdown (the
    // typical path for a process-lifetime singleton), these calls may
    // fail silently — swallowing is the standard pattern here.
    if (D_layout) cublasLtMatrixLayoutDestroy(D_layout);
    if (C_layout) cublasLtMatrixLayoutDestroy(C_layout);
    if (B_layout) cublasLtMatrixLayoutDestroy(B_layout);
    if (A_layout) cublasLtMatrixLayoutDestroy(A_layout);
}

DescriptorCache& DescriptorCache::instance() {
    static DescriptorCache c;
    return c;
}

const CachedGemmPlan& DescriptorCache::get(
    const DescriptorKey& key,
    cublasLtHandle_t handle,
    size_t workspace_bytes) {
    // Fast path: shared lock, hashmap hit, return.
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto it = cache_.find(key);
        if (it != cache_.end()) return *it->second;
    }
    // Slow path: build and insert under exclusive lock. Double-check
    // since another thread could have built it between our shared
    // unlock and exclusive acquire.
    auto built = build_plan(key, handle, workspace_bytes);
    std::unique_lock<std::shared_mutex> lock(mu_);
    auto it = cache_.find(key);
    if (it != cache_.end()) return *it->second;
    auto [inserted, _] = cache_.emplace(key, std::move(built));
    return *inserted->second;
}

}  // namespace cc_cublaslt
