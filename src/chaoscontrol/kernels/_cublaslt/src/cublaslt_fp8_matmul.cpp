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
// Task 1B-3 additions: backward GEMM entry points for grad_x and grad_w,
// the latter with optional BGRADB bias-gradient fusion. Backward GEMMs
// disable CUBLASLT_MATMUL_DESC_FAST_ACCUM (matches TE's `grad`/
// `use_split_accumulator` convention — fast-accum introduces error that
// compounds across the backward pass).
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
// BGRADB (grad_w entry point): bias-grad of the matmul's "B" operand in
// cuBLAS terms — which is torch `a`, i.e. `grad_y` in the grad_w path.
// BGRADB reduces along k (here: along batch M) and emits a vector sized
// to D_cublas's column count = N = out_features. That is exactly the
// shape of `bias.grad` for an `nn.Linear(in_features, out_features)`.

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

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

// --------------------------------------------------------------------------
// Shared fp8 GEMM driver.
//
// Takes torch's (a, b) convention and wires it into cuBLASLt via the
// swap described at the top of the file. Parameterized over:
//   * bias_mode: NONE (no epilogue), FORWARD_BIAS (bias added to D,
//     CUBLASLT_EPILOGUE_BIAS), BGRADB (bias-grad output, reduce B over k
//     and write length-N vector into bias_ptr).
//   * fast_accum: 1 for forward, 0 for backward. Matches TE's
//     use_split_accumulator pattern for bwd GEMMs.
//
// Output tensor D is allocated here and returned, with optional bias-grad
// allocated alongside when bias_mode == BGRADB.
// --------------------------------------------------------------------------
enum class BiasMode { None, ForwardBias, BGradB };

struct GemmResult {
    at::Tensor d;
    std::optional<at::Tensor> bias_grad;
};

GemmResult fp8_matmul_impl(
    const at::Tensor& a,             // fp8 E4M3/E5M2, row-major [M, K]
    const at::Tensor& b,             // fp8 E4M3/E5M2, column-major [K, N]
    const at::Tensor& scale_a,       // fp32 scalar, device
    const at::Tensor& scale_b,       // fp32 scalar, device
    const c10::optional<at::Tensor>& bias_in_opt,   // forward-only bias input
    at::ScalarType out_dtype,
    BiasMode bias_mode,
    bool fast_accum) {
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

    // Forward-bias input tensor (consumed, read-only).
    const at::Tensor* fwd_bias = nullptr;
    if (bias_mode == BiasMode::ForwardBias) {
        TORCH_CHECK(bias_in_opt.has_value() && bias_in_opt->defined(),
                    "ForwardBias mode requires a bias tensor");
        TORCH_CHECK(bias_in_opt->is_cuda(), "bias must live on CUDA");
        TORCH_CHECK(bias_in_opt->scalar_type() == out_dtype,
                    "bias dtype must match out_dtype");
        TORCH_CHECK(bias_in_opt->dim() == 1 && bias_in_opt->size(0) == N,
                    "bias must be 1-D with size N=", N);
        fwd_bias = &bias_in_opt.value();
    }

    // BGradB: allocate the bias-grad output. BGRADB docs say "the bias
    // size corresponds to the number of columns of the matrix D", and
    // "the reduction happens over the GEMM's k dimension". In our swap
    // layout cuBLAS D is [N_func, M_func] col-major with ld=N_func, so
    // its "number of columns" is M_func — the first dim of torch `a`.
    // For the grad_w call (torch a = grad_y.t() [N_torch, M_torch]),
    // M_func == N_torch == out_features. bf16 to match the bias dtype.
    std::optional<at::Tensor> bias_grad;
    if (bias_mode == BiasMode::BGradB) {
        bias_grad = at::empty({M}, a.options().dtype(out_dtype));
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

        // Fast-accumulation: ON for forward, OFF for backward. TE disables
        // fast-accum on bwd GEMMs to avoid fp8 error compounding; we follow.
        const int8_t fast_accum_flag = fast_accum ? 1 : 0;
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                                    &fast_accum_flag,
                                                    sizeof(fast_accum_flag)),
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

        // Epilogue selection.
        //   None:         DEFAULT, no bias read/write.
        //   ForwardBias:  BIAS, bias tensor is ADDED to the output; bf16
        //                 vector of length N = D's column count.
        //   BGradB:       BGRADB, bias tensor is the REDUCTION output —
        //                 dbias = sum_k(B_cublas)_n, which in torch terms is
        //                 sum over batch of the B-operand (torch `a` in bwd
        //                 grad_w call = grad_y.t()). Output length = N.
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        const void* bias_ptr_for_desc = nullptr;
        if (bias_mode == BiasMode::ForwardBias) {
            epilogue = CUBLASLT_EPILOGUE_BIAS;
            bias_ptr_for_desc = fwd_bias->data_ptr();
        } else if (bias_mode == BiasMode::BGradB) {
            epilogue = CUBLASLT_EPILOGUE_BGRADB;
            bias_ptr_for_desc = bias_grad->data_ptr();
        }
        if (bias_ptr_for_desc != nullptr) {
            check_cublas(cublasLtMatmulDescSetAttribute(op_desc,
                                                        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                        &bias_ptr_for_desc,
                                                        sizeof(bias_ptr_for_desc)),
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
    return GemmResult{std::move(d), std::move(bias_grad)};
}

}  // namespace

// --------------------------------------------------------------------------
// Forward fp8 matmul. Public signature unchanged since 1B-3.
// Fast-accum ON; optional CUBLASLT_EPILOGUE_BIAS fusion.
// --------------------------------------------------------------------------
at::Tensor cublaslt_fp8_matmul(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const c10::optional<at::Tensor>& bias_opt,
    at::ScalarType out_dtype) {
    const BiasMode mode = (bias_opt.has_value() && bias_opt->defined())
                              ? BiasMode::ForwardBias
                              : BiasMode::None;
    auto result = fp8_matmul_impl(a, b, scale_a, scale_b, bias_opt,
                                  out_dtype, mode, /*fast_accum=*/true);
    return std::move(result.d);
}

// --------------------------------------------------------------------------
// Backward GEMM: grad_x = grad_y @ W.
//
// Callers pass:
//   grad_y_fp8 : E5M2, row-major [M, N]        (torch `a`)
//   weight_fp8 : E4M3, col-major [N, K]        (torch `b`)
//   scale_gy   : fp32 scalar, dequant multiplier for grad_y
//   scale_w    : fp32 scalar, dequant multiplier for weight
//   out_dtype  : bf16 (fp16 also accepted)
//
// Produces row-major [M, K] bf16 tensor = grad_x.
// Fast-accum OFF (matches TE's split-accumulator convention for bwd).
// --------------------------------------------------------------------------
at::Tensor cublaslt_fp8_matmul_grad_x(
    const at::Tensor& grad_y_fp8,
    const at::Tensor& weight_fp8,
    const at::Tensor& scale_gy,
    const at::Tensor& scale_w,
    at::ScalarType out_dtype) {
    auto result = fp8_matmul_impl(grad_y_fp8, weight_fp8, scale_gy, scale_w,
                                  /*bias_in=*/c10::nullopt, out_dtype,
                                  BiasMode::None, /*fast_accum=*/false);
    return std::move(result.d);
}

// --------------------------------------------------------------------------
// Backward GEMM: grad_w = grad_y.t() @ x, with optional BGRADB bias-grad.
//
// Callers pass:
//   grad_y_fp8_t : E5M2, row-major [N, M]      (torch `a`; already transposed)
//   x_fp8        : E4M3, col-major [M, K]      (torch `b`)
//   scale_gy     : fp32 scalar
//   scale_x      : fp32 scalar
//   out_dtype    : bf16
//   compute_bias_grad : if true, attempt BGRADB epilogue fusion. On
//     hardware/driver combos where BGRADB is unsupported for fp8 GEMMs
//     (observed: cuBLAS 12.8.4 rejects both BGRADA and BGRADB with
//     CUBLAS_STATUS_NOT_SUPPORTED for E5M2×E4M3), the kernel falls back
//     to the DEFAULT epilogue and returns nullopt; the caller should
//     then compute the bias-gradient eagerly (``grad_y.sum(0)``). The
//     Python wrapper handles this fallback transparently.
//
// Produces:
//   grad_w    : row-major [N, K] bf16
//   grad_bias : length-[N] bf16 (filled by BGRADB) or nullopt if the
//     epilogue fused path isn't supported and the caller must sum eagerly.
//
// BGRADB semantics walk-through:
//   Torch view:  grad_w[N,K] = grad_y.t()[N,M] @ x[M,K]
//   Our swap:    cuBLAS A = torch b = x;         cuBLAS B = torch a = grad_y.t()
//   CuBLAS GEMM: D[m,n] = A^op[m,k] @ B^op[k,n]
//                  with m = N_func (first dim of torch a after op=T) = N_torch
//                       n = M_func? No — follow the code: we set transa=T,
//                       transb=N and the cublas m/n fall out as:
//                         m = A_cublas rows after op=T = N_arg = K_torch
//                         n = B_cublas cols after op=N = M_arg = N_torch
//                         k = K_arg = M_torch
//   So cuBLAS D is [m, n] col-major = [K_torch, N_torch] col-major. Its
//   "cols of D" (n) = N_torch = out_features. BGRADB writes a length-n
//   vector = length N_torch. Exactly our dbias shape. The reduction is
//   over cuBLAS k = M_torch, i.e. sum over batch of grad_y — correct.
// --------------------------------------------------------------------------
std::tuple<at::Tensor, std::optional<at::Tensor>> cublaslt_fp8_matmul_grad_w(
    const at::Tensor& grad_y_fp8_t,
    const at::Tensor& x_fp8,
    const at::Tensor& scale_gy,
    const at::Tensor& scale_x,
    at::ScalarType out_dtype,
    bool compute_bias_grad) {
    if (compute_bias_grad) {
        // Try BGRADB first. If the driver rejects it (observed on
        // cuBLAS 12.8.4 with E5M2×E4M3), fall back to DEFAULT and let
        // the Python side compute the bias-grad eagerly. The fallback
        // is behavior-preserving at the API boundary.
        try {
            auto result = fp8_matmul_impl(grad_y_fp8_t, x_fp8, scale_gy, scale_x,
                                          /*bias_in=*/c10::nullopt, out_dtype,
                                          BiasMode::BGradB, /*fast_accum=*/false);
            return std::make_tuple(std::move(result.d),
                                   std::move(result.bias_grad));
        } catch (const std::runtime_error& e) {
            const std::string what = e.what();
            if (what.find("heuristic") == std::string::npos) {
                throw;   // Not a heuristic/support issue — re-raise.
            }
            // Fall through to DEFAULT path.
        }
    }
    auto result = fp8_matmul_impl(grad_y_fp8_t, x_fp8, scale_gy, scale_x,
                                  /*bias_in=*/c10::nullopt, out_dtype,
                                  BiasMode::None, /*fast_accum=*/false);
    return std::make_tuple(std::move(result.d), std::optional<at::Tensor>{});
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
Fused cuBLASLt fp8 matmul (forward).

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
Fast-accum ON; callers wanting split-accumulator semantics should use
the grad_x / grad_w entry points instead.
)doc");

    m.def("cublaslt_fp8_matmul_grad_x", &cc_cublaslt::cublaslt_fp8_matmul_grad_x,
          py::arg("grad_y_fp8"),
          py::arg("weight_fp8"),
          py::arg("scale_gy"),
          py::arg("scale_w"),
          py::arg("out_dtype") = at::kBFloat16,
          R"doc(
Backward fp8 matmul for grad_x = grad_y @ W.

Args:
  grad_y_fp8: fp8 E5M2 row-major [M, N].
  weight_fp8: fp8 E4M3 column-major [N, K].
  scale_gy: fp32 scalar device tensor.
  scale_w: fp32 scalar device tensor.
  out_dtype: bf16 by default.

Returns:
  Tensor of shape [M, K] in out_dtype. Fast-accum OFF.
)doc");

    m.def("cublaslt_fp8_matmul_grad_w", &cc_cublaslt::cublaslt_fp8_matmul_grad_w,
          py::arg("grad_y_fp8_t"),
          py::arg("x_fp8"),
          py::arg("scale_gy"),
          py::arg("scale_x"),
          py::arg("out_dtype") = at::kBFloat16,
          py::arg("compute_bias_grad") = false,
          R"doc(
Backward fp8 matmul for grad_w = grad_y.t() @ x, with optional fused bias_grad.

Args:
  grad_y_fp8_t: fp8 E5M2 row-major [N, M] (transpose already materialized).
  x_fp8: fp8 E4M3 column-major [M, K].
  scale_gy: fp32 scalar.
  scale_x: fp32 scalar.
  out_dtype: bf16.
  compute_bias_grad: if True, fuse BGRADB epilogue and return dbias [N]
    alongside grad_w.

Returns:
  Tuple (grad_w [N, K], grad_bias [N] or None). Fast-accum OFF.
)doc");
}
