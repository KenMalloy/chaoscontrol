// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// cublaslt_fp8_matmul — direct cuBLASLt fp8 matmul for bespoke fp8 Linear.
//
// Bypasses `torch._scaled_mm` and its dispatcher overhead. Semantics are
// intentionally identical: a row-major fp8 [M,K] operand, a column-major
// fp8 [K,N] operand, two fp32 scalar scales (dequant multipliers applied
// on the accumulator side), an optional bf16 [N] bias fused into the
// epilogue, and a bf16 [M,N] output.
//
// Why custom: at dim=256 batch=1024 on H100, `torch._scaled_mm` spends
// measurable time in the torch dispatcher + aten kernel boundary that we
// can elide by calling cublasLtMatmul ourselves with a cached handle and
// workspace.
//
// Layout translation (cuBLASLt is column-major BLAS; torch is row-major):
//
//   We want D_rm[M,N] = A_rm[M,K] @ B_cm[K,N] with fp8 scales.
//
//   Expressing this as a column-major D_cm[N,M] = B_cm^T[N,K] @ A_rm^T[K,M]:
//     * A_cublas := torch `b` (stored as col-major [K,N] ld=K). Use op=T
//       so cuBLASLt treats it as N×K.
//     * B_cublas := torch `a` (row-major [M,K] = col-major [K,M] ld=K).
//       Use op=N so cuBLASLt treats it as K×M.
//     * D_cublas shape N×M col-major ld=N — which in bytes is identical
//       to a row-major [M,N] tensor stride (N, 1). That matches what
//       `torch._scaled_mm` produces.
//
// Scale convention: the two fp32 scales passed in are dequant multipliers
// wired to CUBLASLT_MATMUL_DESC_A_SCALE_POINTER / B_SCALE_POINTER. cuBLASLt
// computes (scale_a * A) @ (scale_b * B) on the accumulator side, same as
// `torch._scaled_mm`'s own convention.
//
// Fast-accum: enabled (CUBLASLT_MATMUL_DESC_FAST_ACCUM=1) — tolerable for
// forward-only Linear at our dim range; TE does the same for forward.

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "workspace_cache.h"

namespace cc_cublaslt {

namespace {

inline cudaDataType_t fp8_cuda_dtype(at::ScalarType s) {
    if (s == at::kFloat8_e4m3fn) return CUDA_R_8F_E4M3;
    if (s == at::kFloat8_e5m2) return CUDA_R_8F_E5M2;
    throw std::runtime_error("cublaslt_fp8_matmul: operand dtype must be fp8 e4m3fn or e5m2");
}

inline cudaDataType_t out_cuda_dtype(at::ScalarType s) {
    if (s == at::kBFloat16) return CUDA_R_16BF;
    if (s == at::kHalf) return CUDA_R_16F;
    throw std::runtime_error("cublaslt_fp8_matmul: out_dtype must be bfloat16 or float16");
}

// Alignment probe (cuBLASLt heuristic wants alignment in bytes).
inline uint32_t ptr_alignment(const void* p) {
    uintptr_t v = reinterpret_cast<uintptr_t>(p);
    uint32_t a = 256;
    while (a > 1 && (v % a) != 0) a /= 2;
    return a;
}

}  // namespace

at::Tensor cublaslt_fp8_matmul(
    const at::Tensor& a,             // fp8 E4M3/E5M2, row-major [M, K]
    const at::Tensor& b,             // fp8 E4M3/E5M2, column-major [K, N]
    const at::Tensor& scale_a,       // fp32 scalar, device
    const at::Tensor& scale_b,       // fp32 scalar, device
    const c10::optional<at::Tensor>& bias_opt,  // bf16 [N], device, or None
    at::ScalarType out_dtype) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must live on CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda(), "scales must live on CUDA");
    TORCH_CHECK(scale_a.scalar_type() == at::kFloat && scale_a.numel() == 1,
                "scale_a must be a fp32 scalar");
    TORCH_CHECK(scale_b.scalar_type() == at::kFloat && scale_b.numel() == 1,
                "scale_b must be a fp32 scalar");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2-D");

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t K_b = b.size(0);
    const int64_t N = b.size(1);
    TORCH_CHECK(K == K_b, "inner dims must match, got a[", M, ",", K, "] b[", K_b, ",", N, "]");

    // Layout checks: a must be row-major M×K (stride (K, 1)); b must be
    // column-major K×N (stride (1, K)). _scaled_mm imposes the same.
    TORCH_CHECK(a.stride(0) == K && a.stride(1) == 1,
                "a must be row-major [M,K] with stride (K, 1); got stride (",
                a.stride(0), ", ", a.stride(1), ")");
    TORCH_CHECK(b.stride(0) == 1 && b.stride(1) == K,
                "b must be column-major [K,N] with stride (1, K); got stride (",
                b.stride(0), ", ", b.stride(1), ")");

    const cudaDataType_t dtype_a = fp8_cuda_dtype(a.scalar_type());
    const cudaDataType_t dtype_b = fp8_cuda_dtype(b.scalar_type());
    const cudaDataType_t dtype_d = out_cuda_dtype(out_dtype);

    // Allocate row-major [M,N] output with stride (N, 1). In column-major
    // terms that's [N,M] with ld=N — what we ask cuBLASLt to produce.
    auto d = at::empty({M, N}, a.options().dtype(out_dtype));

    const at::Tensor* bias = nullptr;
    if (bias_opt.has_value() && bias_opt->defined()) {
        TORCH_CHECK(bias_opt->is_cuda(), "bias must live on CUDA");
        TORCH_CHECK(bias_opt->scalar_type() == out_dtype,
                    "bias dtype must match out_dtype");
        TORCH_CHECK(bias_opt->dim() == 1 && bias_opt->size(0) == N,
                    "bias must be 1-D with size N=", N);
        bias = &bias_opt.value();
    }

    auto& cache = WorkspaceCache::instance();
    cublasLtHandle_t handle = cache.handle();

    // Descriptor & layouts.
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t A_layout = nullptr;
    cublasLtMatrixLayout_t B_layout = nullptr;
    cublasLtMatrixLayout_t C_layout = nullptr;
    cublasLtMatrixLayout_t D_layout = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    // Always destroy on scope exit, including on throw.
    auto cleanup = [&]() {
        if (pref)     cublasLtMatmulPreferenceDestroy(pref);
        if (D_layout) cublasLtMatrixLayoutDestroy(D_layout);
        if (C_layout) cublasLtMatrixLayoutDestroy(C_layout);
        if (B_layout) cublasLtMatrixLayoutDestroy(B_layout);
        if (A_layout) cublasLtMatrixLayoutDestroy(A_layout);
        if (op_desc)  cublasLtMatmulDescDestroy(op_desc);
    };

    try {
        // Compute type: 32-bit accumulator for fp8 matmul.
        check_cublas(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                     "cublasLtMatmulDescCreate");

        // A_cublas = torch `b`, col-major [K, N] ld=K. Use op=T to get N×K.
        // B_cublas = torch `a`, col-major [K, M] ld=K (from row-major [M,K]).
        //   Use op=N to get K×M.
        // Product: N×K · K×M = N×M column-major with ld=N, i.e. row-major [M,N].
        const cublasOperation_t opA = CUBLAS_OP_T;
        const cublasOperation_t opB = CUBLAS_OP_N;
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &opA, sizeof(opA)),
                     "set TRANSA");
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &opB, sizeof(opB)),
                     "set TRANSB");

        // Fp8 fast-accumulation (forward-safe at our dim range).
        const int8_t fast_accum = 1;
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                                    &fast_accum, sizeof(fast_accum)),
                     "set FAST_ACCUM");

        // Scale pointers. In cuBLASLt's nomenclature A_SCALE corresponds to
        // our cuBLAS-A (torch `b`), B_SCALE to cuBLAS-B (torch `a`). Pass
        // scale_b into A_SCALE and scale_a into B_SCALE to keep semantics
        // the same as `torch._scaled_mm(a, b, scale_a, scale_b)`.
        const void* A_scale_ptr = scale_b.data_ptr();  // cuBLAS-A <- torch b
        const void* B_scale_ptr = scale_a.data_ptr();  // cuBLAS-B <- torch a
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc,
                                                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                    &A_scale_ptr, sizeof(A_scale_ptr)),
                     "set A_SCALE_POINTER");
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc,
                                                    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                    &B_scale_ptr, sizeof(B_scale_ptr)),
                     "set B_SCALE_POINTER");

        // Bias epilogue. cuBLASLt's bias is added to the output rows, which
        // in our column-major D (N×M ld=N) means the bias must be length N
        // and indexed along the leading dim — exactly the per-output-feature
        // bias semantics of nn.Linear.
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        if (bias != nullptr) {
            epilogue = CUBLASLT_EPILOGUE_BIAS;
            const void* bias_ptr = bias->data_ptr();
            check_cublas(cublasLtMatmulDescSetAttribute(op_desc,
                                                        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                        &bias_ptr, sizeof(bias_ptr)),
                         "set BIAS_POINTER");
            const cudaDataType_t bias_dtype = dtype_d;
            check_cublas(cublasLtMatmulDescSetAttribute(op_desc,
                                                        CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                        &bias_dtype, sizeof(bias_dtype)),
                         "set BIAS_DATA_TYPE");
        }
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &epilogue, sizeof(epilogue)),
                     "set EPILOGUE");

        // Matrix layouts. Shapes passed to cublasLtMatrixLayoutCreate are
        // ALWAYS the physical (storage) dims, not the op-transformed dims.
        // A storage: [K, N] col-major ld=K  (torch `b`).
        // B storage: [K, M] col-major ld=K  (torch `a`, from row-major view).
        // D storage: [N, M] col-major ld=N.
        check_cublas(cublasLtMatrixLayoutCreate(&A_layout, dtype_b, K, N, K),
                     "layout A");
        check_cublas(cublasLtMatrixLayoutCreate(&B_layout, dtype_a, K, M, K),
                     "layout B");
        check_cublas(cublasLtMatrixLayoutCreate(&D_layout, dtype_d, N, M, N),
                     "layout D");
        // For fp8 output cuBLASLt requires a separate C layout that matches
        // the bias/accumulator path. For our bf16/fp16 output, reusing the
        // D layout as C is fine (beta=0 so C isn't read).
        check_cublas(cublasLtMatrixLayoutCreate(&C_layout, dtype_d, N, M, N),
                     "layout C");

        // Heuristic selection.
        check_cublas(cublasLtMatmulPreferenceCreate(&pref), "preference create");
        size_t ws_bytes = cache.workspace_bytes();
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                         &ws_bytes, sizeof(ws_bytes)),
                     "pref workspace");

        const uint32_t a_align = ptr_alignment(b.data_ptr());
        const uint32_t b_align = ptr_alignment(a.data_ptr());
        const uint32_t d_align = ptr_alignment(d.data_ptr());
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                         &a_align, sizeof(a_align)),
                     "pref align A");
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                         &b_align, sizeof(b_align)),
                     "pref align B");
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                         &d_align, sizeof(d_align)),
                     "pref align C");
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                         &d_align, sizeof(d_align)),
                     "pref align D");

        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned = 0;
        check_cublas(cublasLtMatmulAlgoGetHeuristic(handle, op_desc,
                                                    A_layout, B_layout, C_layout, D_layout,
                                                    pref, 1, &heuristic, &returned),
                     "heuristic");
        if (returned == 0) {
            throw std::runtime_error("cublaslt_fp8_matmul: no suitable algo for shape");
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        check_cublas(cublasLtMatmul(handle, op_desc,
                                    &alpha,
                                    b.data_ptr(), A_layout,      // A_cublas = torch b
                                    a.data_ptr(), B_layout,      // B_cublas = torch a
                                    &beta,
                                    d.data_ptr(), C_layout,      // C unused (beta=0)
                                    d.data_ptr(), D_layout,      // D
                                    &heuristic.algo,
                                    cache.workspace(), cache.workspace_bytes(),
                                    stream),
                     "cublasLtMatmul");
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    return d;
}

}  // namespace cc_cublaslt


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cublaslt_fp8_matmul", &cc_cublaslt::cublaslt_fp8_matmul,
          py::arg("a"),
          py::arg("b"),
          py::arg("scale_a"),
          py::arg("scale_b"),
          py::arg("bias") = py::none(),
          py::arg("out_dtype") = at::kBFloat16,
          R"doc(
Fused cuBLASLt fp8 matmul.

Args:
  a: fp8 row-major [M, K].
  b: fp8 column-major [K, N].
  scale_a: fp32 scalar device tensor, dequant multiplier for a.
  scale_b: fp32 scalar device tensor, dequant multiplier for b.
  bias: optional bf16/fp16 [N] bias, fused in cuBLASLt epilogue.
  out_dtype: torch dtype of output (bfloat16 or float16).

Returns:
  Tensor of shape [M, N] in out_dtype.

Semantics match ``torch._scaled_mm(a, b, scale_a=..., scale_b=..., bias=...,
out_dtype=...)`` — the scales are dequant multipliers applied on the
accumulator side: output = (scale_a * a) @ (scale_b * b) + bias.
)doc");
}
