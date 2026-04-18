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

#include "descriptor_cache.h"
#include "fused_amax_cast.h"
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

// (Per-call alignment probe removed alongside the shift to DescriptorCache
// in 2026-04-18: alignment is now pinned at 256 bytes in the cached plan
// on the assumption that torch's CUDA caching allocator always hands back
// at-least-256-byte-aligned pointers. If that ever breaks we'd surface it
// here as a validation and key the cache on alignment too.)

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
//
// ``BiasMode`` is defined in descriptor_cache.h (in namespace
// cc_cublaslt, not anonymous) so the cache key can reference it.
// --------------------------------------------------------------------------

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

    // Validate operand dtypes — the cache stores layouts that embed the
    // dtype, but these also guard against an invalid-dtype input path.
    // The only local use of dtype_d is below, for BIAS_DATA_TYPE; the
    // descriptor cache owns the other two.
    (void)fp8_cuda_dtype(a.scalar_type());
    (void)fp8_cuda_dtype(b.scalar_type());
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

    auto& ws = WorkspaceCache::instance();
    cublasLtHandle_t handle = ws.handle();

    // Layouts + heuristic algo come from the shape/dtype-keyed cache.
    // First call for a key builds + inserts (~50-100 µs heuristic cost);
    // every subsequent call is a hashmap hit. The cache stores ONLY the
    // immutable planning state — the op_desc itself is rebuilt per call
    // below, cheap (~5 SetAttribute) and avoids any SetAttribute race
    // on shared state.
    const DescriptorKey key{
        M, N, K,
        bias_mode, fast_accum,
        a.scalar_type(), b.scalar_type(), out_dtype,
    };
    const CachedGemmPlan& plan = DescriptorCache::instance().get(
        key, handle, ws.workspace_bytes());

    // Build the per-call op_desc. The "fixed" attrs (TRANSA/B, FAST_ACCUM,
    // EPILOGUE, BIAS_DATA_TYPE) set here duplicate what the heuristic
    // saw during plan construction — cuBLASLt requires them on the desc
    // passed to cublasLtMatmul regardless of caching.
    cublasLtMatmulDesc_t op_desc = nullptr;
    auto cleanup = [&]() {
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    };

    try {
        check_cublas(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                     "cublasLtMatmulDescCreate");

        const cublasOperation_t opA = CUBLAS_OP_T;
        const cublasOperation_t opB = CUBLAS_OP_N;
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &opA, sizeof(opA)),
                     "set TRANSA");
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &opB, sizeof(opB)),
                     "set TRANSB");

        const int8_t fast_accum_flag = fast_accum ? 1 : 0;
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                                    &fast_accum_flag,
                                                    sizeof(fast_accum_flag)),
                     "set FAST_ACCUM");

        // Per-call mutable attrs: scale pointers (addresses of scale_a /
        // scale_b tensors, which change every step under deferred amax).
        // A_SCALE corresponds to cuBLAS-A (torch `b`); B_SCALE to
        // cuBLAS-B (torch `a`) — matches ``torch._scaled_mm``'s contract.
        const void* A_scale_ptr = scale_b.data_ptr();
        const void* B_scale_ptr = scale_a.data_ptr();
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc,
                                                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                    &A_scale_ptr, sizeof(A_scale_ptr)),
                     "set A_SCALE_POINTER");
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc,
                                                    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                    &B_scale_ptr, sizeof(B_scale_ptr)),
                     "set B_SCALE_POINTER");

        // Epilogue + bias pointer. Epilogue is part of the cache key
        // (the heuristic saw the same value), so we only need to set it
        // here and attach the per-call bias pointer.
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        const void* bias_ptr_for_desc = nullptr;
        if (bias_mode == BiasMode::ForwardBias) {
            epilogue = CUBLASLT_EPILOGUE_BIAS;
            bias_ptr_for_desc = fwd_bias->data_ptr();
        } else if (bias_mode == BiasMode::BGradB) {
            epilogue = CUBLASLT_EPILOGUE_BGRADB;
            bias_ptr_for_desc = bias_grad->data_ptr();
        }
        check_cublas(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                    &epilogue, sizeof(epilogue)),
                     "set EPILOGUE");
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

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        check_cublas(cublasLtMatmul(handle, op_desc,
                                    &alpha,
                                    b.data_ptr(), plan.A_layout,   // A_cublas = torch b
                                    a.data_ptr(), plan.B_layout,   // B_cublas = torch a
                                    &beta,
                                    d.data_ptr(), plan.C_layout,   // C unused (beta=0)
                                    d.data_ptr(), plan.D_layout,   // D
                                    &plan.heuristic.algo,
                                    ws.workspace(), ws.workspace_bytes(),
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
        // Try BGRADB first. Empirically (cuBLAS 12.8.4 and 13.4.0.1,
        // probed 2026-04-17 and 2026-04-18) BGRADB is rejected with
        // CUBLAS_STATUS_NOT_SUPPORTED (code 15) for fp8 E5M2×E4M3 at
        // every shape we tested. The fallback path below returns nullopt
        // so the caller can reduce ``grad_y`` eagerly — this is what TE
        // itself does at the same dim range, so we're not leaving
        // throughput on the table by a measurable amount. Keep the path
        // in place until a future cuBLAS release lights up the epilogue.
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

// --------------------------------------------------------------------------
// Phase 3 fused entry points: fold amax-update + bf16→fp8 cast + GEMM into
// a single C++ call. Python still reads amax scales from separate device
// buffers refreshed by flush_amax_history() — we only collapse the per-call
// orchestration, not the deferred-amax lifecycle.
//
// Each entry point takes bf16 operands + a scale buffer + a pending buffer,
// allocates the fp8 staging tensors, launches the fused kernel, and then
// delegates to the existing fp8_matmul_impl for the cuBLASLt matmul. The
// fp8 staging tensors go out of scope at the end of each call — PyTorch's
// caching allocator keeps the slab warm so there is no per-iteration
// cudaMalloc.
// --------------------------------------------------------------------------

namespace {

// Fused amax + bf16→fp8 cast producing a ROW-MAJOR [M, K] fp8 output
// from a row-major [M, K] bf16 input. Used for:
//   * forward x (E4M3)
//   * backward grad_y @ grad_x path (E5M2)
static at::Tensor fused_amax_cast_rowmajor(
    const at::Tensor& x_bf16,    // [M, K] row-major bf16 (or a contiguous view)
    const at::Tensor& scale,     // fp32 scalar, device
    at::Tensor pending,          // fp32 scalar, device — updated atomically
    at::ScalarType fp8_dtype) {  // kFloat8_e4m3fn or kFloat8_e5m2
    TORCH_CHECK(x_bf16.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x_bf16.scalar_type() == at::kBFloat16,
                "input must be bfloat16");
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == at::kFloat
                    && scale.numel() == 1,
                "scale must be fp32 scalar device tensor");
    TORCH_CHECK(pending.is_cuda() && pending.scalar_type() == at::kFloat
                    && pending.numel() == 1,
                "pending must be fp32 scalar device tensor");
    TORCH_CHECK(fp8_dtype == at::kFloat8_e4m3fn || fp8_dtype == at::kFloat8_e5m2,
                "fp8_dtype must be e4m3fn or e5m2");
    TORCH_CHECK(x_bf16.is_contiguous(), "input must be contiguous row-major");
    auto out = at::empty(x_bf16.sizes(), x_bf16.options().dtype(fp8_dtype));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_fused_amax_cast_bf16(
        x_bf16.data_ptr(),
        scale.data_ptr(),
        pending.data_ptr(),
        out.data_ptr(),
        fp8_dtype,
        x_bf16.numel(),
        stream);
    return out;
}

// Fused amax + cast producing a COLUMN-MAJOR [K, N] fp8 output from a
// row-major [N, K] bf16 input (i.e. casts weight.t()). weight_bf16 is
// [N, K] contiguous row-major with strides (K, 1). The output is the
// transpose: [K, N] with strides (1, K) — exactly what the cuBLASLt
// forward kernel expects as its ``b`` operand.
//
// Implementation: we write the output in a logical [K, N] row-major
// layout which, since N fits the physical col-stride, produces the
// col-major [K, N] layout we need. The kernel treats the bf16 input
// via a transposed indexing pattern.
static at::Tensor fused_amax_cast_transpose(
    const at::Tensor& weight_bf16,  // [N, K] row-major bf16
    const at::Tensor& scale,
    at::Tensor pending,
    at::ScalarType fp8_dtype) {
    TORCH_CHECK(weight_bf16.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(weight_bf16.scalar_type() == at::kBFloat16,
                "weight must be bfloat16");
    TORCH_CHECK(weight_bf16.dim() == 2, "weight must be 2-D");
    TORCH_CHECK(weight_bf16.is_contiguous(),
                "weight must be contiguous [N, K] row-major");
    const int64_t N = weight_bf16.size(0);
    const int64_t K = weight_bf16.size(1);

    // Allocate as [K, N] with stride (1, K) — column-major over [K, N].
    // We do this by allocating a contiguous [N, K] tensor and taking
    // .t(); the .t() flips strides to (1, K). The cast kernel writes
    // out in row-major [N, K] which, after .t(), presents as col-major
    // [K, N] — exactly the b-operand layout cuBLASLt wants.
    auto staging = at::empty({N, K}, weight_bf16.options().dtype(fp8_dtype));
    // We will write staging as row-major [N, K] with the SAME indexing
    // as weight_bf16 — i.e. no transpose kernel needed here because the
    // byte layout of staging.t() is already column-major [K, N].
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_fused_amax_cast_bf16(
        weight_bf16.data_ptr(),
        scale.data_ptr(),
        pending.data_ptr(),
        staging.data_ptr(),
        fp8_dtype,
        weight_bf16.numel(),
        stream);
    // .t() returns a view with flipped strides; no data movement.
    return staging.t();
}

// Fused amax + cast for the grad_w path's two operands.
//
// (a) grad_y.t() [N, M] row-major E5M2 from grad_y [M, N] row-major bf16.
//     The kernel writes [N, M] row-major fp8 directly; transpose is
//     implemented by an index permutation in the kernel, not by a
//     .contiguous() copy.
static at::Tensor fused_amax_cast_grad_y_transposed(
    const at::Tensor& grad_y_bf16,  // [M, N] row-major bf16
    const at::Tensor& scale,
    at::Tensor pending,
    at::ScalarType fp8_dtype) {
    TORCH_CHECK(grad_y_bf16.is_cuda()
                    && grad_y_bf16.scalar_type() == at::kBFloat16
                    && grad_y_bf16.dim() == 2
                    && grad_y_bf16.is_contiguous(),
                "grad_y must be contiguous 2-D bf16 CUDA tensor");
    const int64_t M = grad_y_bf16.size(0);
    const int64_t N = grad_y_bf16.size(1);

    auto out = at::empty({N, M}, grad_y_bf16.options().dtype(fp8_dtype));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // Output [N, M] row-major — for output cell (r=n, c=m), read input
    // [m, n] = (n * 1 + m * N) in row-major [M, N] flattening. With
    // in_row_stride = 1, in_col_stride = N.
    launch_fused_amax_cast_transpose_bf16(
        grad_y_bf16.data_ptr(),
        scale.data_ptr(),
        pending.data_ptr(),
        out.data_ptr(),
        fp8_dtype,
        /*rows_out=*/N,
        /*cols_out=*/M,
        /*in_row_stride=*/1,
        /*in_col_stride=*/N,
        stream);
    return out;
}

// (b) x [M, K] column-major fp8 from x [M, K] row-major bf16.
//     Physical layout needed: strides (1, M) — the transpose of an [M, K]
//     contiguous tensor. Implementation: allocate a contiguous [K, M] fp8
//     tensor, write rows=K × cols=M where each cell (r=k, c=m) reads
//     input[m, k] = m*K + k (in_row_stride=K for row=k, in_col_stride=1
//     for col=m  — OR just swap axes: we want to write dst[k, m] from
//     src[m, k]). Then return staging.t() as the [M, K] col-major view.
static at::Tensor fused_amax_cast_x_transposed_then_transpose_back(
    const at::Tensor& x_bf16,   // [M, K] row-major bf16
    const at::Tensor& scale,
    at::Tensor pending,
    at::ScalarType fp8_dtype) {
    TORCH_CHECK(x_bf16.is_cuda()
                    && x_bf16.scalar_type() == at::kBFloat16
                    && x_bf16.dim() == 2
                    && x_bf16.is_contiguous(),
                "x must be contiguous 2-D bf16 CUDA tensor");
    const int64_t M = x_bf16.size(0);
    const int64_t K = x_bf16.size(1);

    auto staging = at::empty({K, M}, x_bf16.options().dtype(fp8_dtype));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // Output [K, M] row-major; for (r=k, c=m) read src[m, k] → index
    // m*K + k. So in_row_stride = 1 (row index k contributes +1),
    // in_col_stride = K (col index m contributes +K).
    launch_fused_amax_cast_transpose_bf16(
        x_bf16.data_ptr(),
        scale.data_ptr(),
        pending.data_ptr(),
        staging.data_ptr(),
        fp8_dtype,
        /*rows_out=*/K,
        /*cols_out=*/M,
        /*in_row_stride=*/1,
        /*in_col_stride=*/K,
        stream);
    // staging.t() is [M, K] with strides (1, K) — col-major, what
    // cublaslt_fp8_matmul_grad_w wants as its b operand.
    return staging.t();
}

}  // namespace

// Forward: amax + E4M3 cast of x and w, then GEMM. Bias optional.
at::Tensor cublaslt_fp8_linear_fwd(
    const at::Tensor& a_bf16,        // [M, K] row-major bf16
    const at::Tensor& b_bf16,        // [N, K] row-major bf16 (i.e. weight);
                                     // we transpose to [K, N] col-major inside.
    const at::Tensor& x_scale,
    const at::Tensor& w_scale,
    at::Tensor x_pending,
    at::Tensor w_pending,
    const c10::optional<at::Tensor>& bias,
    at::ScalarType out_dtype) {
    at::Tensor a_fp8 = fused_amax_cast_rowmajor(
        a_bf16, x_scale, x_pending, at::kFloat8_e4m3fn);
    at::Tensor b_fp8 = fused_amax_cast_transpose(
        b_bf16, w_scale, w_pending, at::kFloat8_e4m3fn);

    const BiasMode mode = (bias.has_value() && bias->defined())
                              ? BiasMode::ForwardBias
                              : BiasMode::None;
    auto result = fp8_matmul_impl(a_fp8, b_fp8, x_scale, w_scale, bias,
                                  out_dtype, mode, /*fast_accum=*/true);
    return std::move(result.d);
}

// Backward grad_x: fused cast of grad_y (E5M2) and weight.t() (E4M3 col-major),
// then GEMM.
at::Tensor cublaslt_fp8_linear_bwd_x(
    const at::Tensor& grad_y_bf16,  // [M, N] row-major bf16
    const at::Tensor& weight_bf16,  // [N, K] row-major bf16
    const at::Tensor& gy_scale,
    const at::Tensor& w_scale,
    at::Tensor gy_pending,
    at::Tensor gx_pending,            // diagnostic only; we don't touch it here
    at::ScalarType out_dtype) {
    (void)gx_pending;   // diagnostic grad_x amax folded by Python after return

    at::Tensor gy_fp8 = fused_amax_cast_rowmajor(
        grad_y_bf16, gy_scale, gy_pending, at::kFloat8_e5m2);
    // Weight: we want [N, K] col-major E4M3 — strides (1, N). That means
    // a contiguous [K, N] cast with a transposed read pattern, then .t().
    //
    // Input layout: weight_bf16 is [N, K] contiguous row-major (strides
    //   (K, 1)). Output (staging) layout: [K, N] contiguous row-major.
    //   Cell (r=k, c=n) in staging reads input[n, k] = n*K + k.
    //   So in_row_stride = 1, in_col_stride = K.
    TORCH_CHECK(weight_bf16.is_contiguous(), "weight must be contiguous");
    const int64_t N = weight_bf16.size(0);
    const int64_t K = weight_bf16.size(1);
    auto staging_w = at::empty({K, N}, weight_bf16.options().dtype(at::kFloat8_e4m3fn));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // Pass nullptr for the pending buffer — the weight's amax was
    // already accumulated in w_pending during forward, and re-
    // accumulating the same tensor here would not change the max
    // (idempotent) but would cost an atomic round-trip. Skipping the
    // atomic write entirely is cleaner.
    launch_fused_amax_cast_transpose_bf16(
        weight_bf16.data_ptr(),
        w_scale.data_ptr(),
        /*pending=*/nullptr,
        staging_w.data_ptr(),
        at::kFloat8_e4m3fn,
        /*rows_out=*/K,
        /*cols_out=*/N,
        /*in_row_stride=*/1,
        /*in_col_stride=*/K,
        stream);
    at::Tensor w_fp8 = staging_w.t();   // [N, K] col-major

    auto result = fp8_matmul_impl(gy_fp8, w_fp8, gy_scale, w_scale,
                                  /*bias_in=*/c10::nullopt,
                                  out_dtype, BiasMode::None,
                                  /*fast_accum=*/false);
    return std::move(result.d);
}

// One-shot flush of a (history, pending) pair → (new_scale), all in
// a single kernel launch. Replaces ``_push_amax_and_rescale`` in
// fp8_linear.py. Works in-place on the passed tensors — no return
// value since both history and scale are updated on device.
void cublaslt_fp8_flush_amax(
    at::Tensor history,   // fp32 [H], device
    at::Tensor pending,   // fp32 [1], device
    at::Tensor scale,     // fp32 [1], device
    double max_rep) {
    TORCH_CHECK(history.is_cuda() && history.scalar_type() == at::kFloat,
                "history must be fp32 CUDA");
    TORCH_CHECK(pending.is_cuda() && pending.scalar_type() == at::kFloat
                    && pending.numel() == 1,
                "pending must be fp32 scalar CUDA");
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == at::kFloat
                    && scale.numel() == 1,
                "scale must be fp32 scalar CUDA");
    TORCH_CHECK(history.is_contiguous(), "history must be contiguous");
    TORCH_CHECK(history.numel() <= 1024, "history too long (max 1024)");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_fused_flush_amax(
        static_cast<float*>(history.data_ptr()),
        static_cast<float*>(pending.data_ptr()),
        static_cast<float*>(scale.data_ptr()),
        static_cast<int>(history.numel()),
        static_cast<float>(max_rep),
        stream);
}

// Diagnostic flush (no scale): roll + zero. Used for gx_amax_history.
void cublaslt_fp8_flush_amax_diagnostic(
    at::Tensor history,
    at::Tensor pending) {
    TORCH_CHECK(history.is_cuda() && history.scalar_type() == at::kFloat,
                "history must be fp32 CUDA");
    TORCH_CHECK(pending.is_cuda() && pending.scalar_type() == at::kFloat
                    && pending.numel() == 1,
                "pending must be fp32 scalar CUDA");
    TORCH_CHECK(history.is_contiguous(), "history must be contiguous");
    TORCH_CHECK(history.numel() <= 1024, "history too long (max 1024)");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_fused_flush_amax_diagnostic(
        static_cast<float*>(history.data_ptr()),
        static_cast<float*>(pending.data_ptr()),
        static_cast<int>(history.numel()),
        stream);
}

// Backward grad_w + optional bias_grad: fused transposed-cast of grad_y
// and x, then GEMM with optional BGRADB epilogue.
std::tuple<at::Tensor, std::optional<at::Tensor>> cublaslt_fp8_linear_bwd_w(
    const at::Tensor& grad_y_bf16,   // [M, N] row-major bf16
    const at::Tensor& x_bf16,        // [M, K] row-major bf16
    const at::Tensor& gy_scale,
    const at::Tensor& x_scale,
    at::Tensor gy_pending,
    at::Tensor x_pending,            // scratch — reuse is safe (see forward)
    at::ScalarType out_dtype,
    bool compute_bias_grad) {
    // grad_y.t() → [N, M] E5M2 row-major
    at::Tensor gy_t_fp8 = fused_amax_cast_grad_y_transposed(
        grad_y_bf16, gy_scale, gy_pending, at::kFloat8_e5m2);
    // x → [M, K] E4M3 column-major (transpose-and-back trick). The
    // forward already accumulated the x amax into x_pending at the
    // current step; re-running the atomicMax here would be idempotent
    // but wasted work. The helper below honors pending=nullptr to skip
    // the atomic write. Allocate staging and call through the kernel
    // directly so we can pass nullptr.
    (void)x_pending;
    TORCH_CHECK(x_bf16.is_contiguous(), "x must be contiguous");
    const int64_t xM = x_bf16.size(0);
    const int64_t xK = x_bf16.size(1);
    auto staging_x = at::empty({xK, xM},
                               x_bf16.options().dtype(at::kFloat8_e4m3fn));
    cudaStream_t stream_bwdw = c10::cuda::getCurrentCUDAStream();
    launch_fused_amax_cast_transpose_bf16(
        x_bf16.data_ptr(),
        x_scale.data_ptr(),
        /*pending=*/nullptr,
        staging_x.data_ptr(),
        at::kFloat8_e4m3fn,
        /*rows_out=*/xK,
        /*cols_out=*/xM,
        /*in_row_stride=*/1,
        /*in_col_stride=*/xK,
        stream_bwdw);
    at::Tensor x_col_fp8 = staging_x.t();

    if (compute_bias_grad) {
        try {
            auto result = fp8_matmul_impl(gy_t_fp8, x_col_fp8,
                                          gy_scale, x_scale,
                                          /*bias_in=*/c10::nullopt,
                                          out_dtype, BiasMode::BGradB,
                                          /*fast_accum=*/false);
            return std::make_tuple(std::move(result.d),
                                   std::move(result.bias_grad));
        } catch (const std::runtime_error& e) {
            const std::string what = e.what();
            if (what.find("heuristic") == std::string::npos) throw;
            // Fall through to DEFAULT. See cublaslt_fp8_matmul_grad_w
            // above — BGRADB is unsupported for fp8 E5M2×E4M3 in every
            // cuBLAS we've probed (12.8.4, 13.4.0.1).
        }
    }
    auto result = fp8_matmul_impl(gy_t_fp8, x_col_fp8,
                                  gy_scale, x_scale,
                                  /*bias_in=*/c10::nullopt,
                                  out_dtype, BiasMode::None,
                                  /*fast_accum=*/false);
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

    // -- Phase 3 fused entry points --------------------------------------
    m.def("cublaslt_fp8_linear_fwd", &cc_cublaslt::cublaslt_fp8_linear_fwd,
          py::arg("a_bf16"),
          py::arg("b_bf16"),
          py::arg("x_scale"),
          py::arg("w_scale"),
          py::arg("x_pending"),
          py::arg("w_pending"),
          py::arg("bias") = py::none(),
          py::arg("out_dtype") = at::kBFloat16,
          R"doc(
Fused amax-update + fp8 cast + forward cuBLASLt matmul.

Args:
  a_bf16: bf16 [M, K] row-major activations.
  b_bf16: bf16 [N, K] row-major weight (same layout as nn.Linear.weight).
  x_scale, w_scale: fp32 scalar dequant multipliers (READ; refreshed by
    flush_amax_history between optimizer steps).
  x_pending, w_pending: fp32 scalar pending-amax buffers (atomically
    UPDATED with max(|a_bf16|), max(|b_bf16|)).
  bias: optional bf16 [N] bias fused via CUBLASLT_EPILOGUE_BIAS.
  out_dtype: bfloat16 default.

Returns:
  bf16 [M, N] output. Fast-accum ON.
)doc");

    m.def("cublaslt_fp8_linear_bwd_x", &cc_cublaslt::cublaslt_fp8_linear_bwd_x,
          py::arg("grad_y_bf16"),
          py::arg("weight_bf16"),
          py::arg("gy_scale"),
          py::arg("w_scale"),
          py::arg("gy_pending"),
          py::arg("gx_pending"),
          py::arg("out_dtype") = at::kBFloat16,
          R"doc(
Fused amax-update + fp8 cast + backward cuBLASLt matmul for grad_x.

Args:
  grad_y_bf16: bf16 [M, N] row-major.
  weight_bf16: bf16 [N, K] row-major.
  gy_scale, w_scale: fp32 scalar dequant multipliers.
  gy_pending: fp32 scalar pending — atomically updated with amax(grad_y).
  gx_pending: reserved diagnostic buffer; not touched inside.
  out_dtype: bfloat16.

Returns:
  bf16 [M, K] grad_x. Fast-accum OFF.
)doc");

    m.def("cublaslt_fp8_linear_bwd_w", &cc_cublaslt::cublaslt_fp8_linear_bwd_w,
          py::arg("grad_y_bf16"),
          py::arg("x_bf16"),
          py::arg("gy_scale"),
          py::arg("x_scale"),
          py::arg("gy_pending"),
          py::arg("x_pending"),
          py::arg("out_dtype") = at::kBFloat16,
          py::arg("compute_bias_grad") = false,
          R"doc(
Fused amax-update + fp8 cast + backward cuBLASLt matmul for grad_w (+ dbias).

Args:
  grad_y_bf16: bf16 [M, N] row-major.
  x_bf16: bf16 [M, K] row-major (typically re-materialized from the fp8
    saved_for_backward; callers can pass the bf16 input if the hot path
    keeps it live).
  gy_scale, x_scale: fp32 scalar dequant multipliers.
  gy_pending, x_pending: fp32 scalar pending — gy_pending is updated;
    x_pending is currently ignored (forward already updated it for the
    same tensor/step; passing it again would double-count).
  out_dtype: bfloat16.
  compute_bias_grad: if true, fuse BGRADB epilogue.

Returns:
  Tuple (grad_w [N, K], grad_bias [N] or None). Fast-accum OFF.
)doc");

    m.def("cublaslt_fp8_flush_amax", &cc_cublaslt::cublaslt_fp8_flush_amax,
          py::arg("history"),
          py::arg("pending"),
          py::arg("scale"),
          py::arg("max_rep"),
          R"doc(
One-shot amax flush: roll history left, append pending at history[-1],
compute max(history), write scale = max_history / max_rep (or 1.0 if
zero), zero pending. Single kernel launch. max_rep is 448.0 for E4M3,
57344.0 for E5M2.
)doc");

    m.def("cublaslt_fp8_flush_amax_diagnostic",
          &cc_cublaslt::cublaslt_fp8_flush_amax_diagnostic,
          py::arg("history"),
          py::arg("pending"),
          R"doc(
Diagnostic flush variant: roll history left + append pending + zero
pending. No scale output.
)doc");
}
