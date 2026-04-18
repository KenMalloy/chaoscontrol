// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Python binding for the diag SSM scan kernel.
//
// Public surface (one call):
//   ssm_scan_forward(decay, update) -> state
//
// decay: (B, T, D) bf16 | fp16 | fp32, contiguous row-major
// update: (B, T, D) same dtype as decay (or fp16 update with bf16 decay;
//         see dispatch table in ssm_scan_fwd.cu)
// Returns: (B, T, D) same dtype as update, fp32 accumulator internal.
//
// Semantics: state[0] = decay[0] * 0 + update[0]; state[t] = decay[t] *
// state[t-1] + update[t]. Bit-identical (fp32 accumulator) to the
// Python reference at ``chaoscontrol.core._diag_recurrence_inner`` when
// that reference is also run in fp32.

#include "ssm_scan.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <stdexcept>

namespace cc_ssm_scan {

at::Tensor ssm_scan_forward(const at::Tensor& decay,
                            const at::Tensor& update) {
    TORCH_CHECK(decay.is_cuda() && update.is_cuda(),
                "decay and update must live on CUDA");
    TORCH_CHECK(decay.dim() == 3 && update.dim() == 3,
                "decay and update must be 3-D (B, T, D)");
    TORCH_CHECK(decay.sizes() == update.sizes(),
                "decay and update must share shape (B, T, D)");
    TORCH_CHECK(decay.is_contiguous() && update.is_contiguous(),
                "decay and update must be contiguous row-major (B, T, D)");
    TORCH_CHECK(decay.device() == update.device(),
                "decay and update must live on the same CUDA device");

    const auto decay_dtype = decay.scalar_type();
    const auto update_dtype = update.scalar_type();

    // Allowed dtype combos. Extend the dispatcher in ssm_scan_fwd.cu
    // if a caller needs more.
    const bool bf16_bf16 = (decay_dtype == at::kBFloat16 && update_dtype == at::kBFloat16);
    const bool bf16_fp16 = (decay_dtype == at::kBFloat16 && update_dtype == at::kHalf);
    const bool fp16_fp16 = (decay_dtype == at::kHalf && update_dtype == at::kHalf);
    const bool fp32_fp32 = (decay_dtype == at::kFloat && update_dtype == at::kFloat);
    TORCH_CHECK(bf16_bf16 || bf16_fp16 || fp16_fp16 || fp32_fp32,
                "unsupported (decay, update) dtype combo: (",
                toString(decay_dtype), ", ", toString(update_dtype),
                "); want bf16/bf16, bf16/fp16, fp16/fp16, or fp32/fp32");

    const int64_t B = decay.size(0);
    const int64_t T = decay.size(1);
    const int64_t D = decay.size(2);
    TORCH_CHECK(B > 0 && T > 0 && D > 0,
                "decay shape (", B, ", ", T, ", ", D, ") must be positive");
    TORCH_CHECK(B <= std::numeric_limits<int>::max(),
                "B exceeds int32 (kernel uses int indexing on blockIdx.y)");
    TORCH_CHECK(D <= std::numeric_limits<int>::max(),
                "D exceeds int32");
    TORCH_CHECK(T <= std::numeric_limits<int>::max(),
                "T exceeds int32");

    // Output matches update's dtype — state lives on the update side
    // of the recurrence, so downstream post-scan ops (gate, out_proj)
    // chain off whichever dtype update uses.
    auto out = at::empty_like(update);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_ssm_scan_fwd(
        decay.data_ptr(),
        update.data_ptr(),
        out.data_ptr(),
        static_cast<int>(B),
        static_cast<int>(T),
        static_cast<int>(D),
        decay_dtype,
        update_dtype,
        stream);
    // Propagate any async launch error. Without this, a mis-shaped
    // input could reach the caller as a silent NaN/garbage tensor.
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cc_ssm_scan: launch failed: ") +
            cudaGetErrorString(err));
    }
    return out;
}

}  // namespace cc_ssm_scan


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ssm_scan_forward", &cc_ssm_scan::ssm_scan_forward,
          py::arg("decay"),
          py::arg("update"),
          R"doc(
Forward diag SSM scan.

Computes state[t] = decay[t] * state[t-1] + update[t] with state[-1] = 0,
over every (batch, channel) lane independently.

Args:
  decay: (B, T, D) bf16/fp16/fp32, contiguous, row-major.
  update: (B, T, D) matching dtype (see dispatcher in ssm_scan_fwd.cu
    for allowed combos).

Returns:
  (B, T, D) state tensor; dtype matches ``update``. Accumulator is fp32
  inside the kernel regardless of input dtype.
)doc");
}
