// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Python binding for the diag SSM scan kernel.
//
// Public surface:
//   ssm_scan_forward(decay, update) -> (out, state_fp32)
//   ssm_scan_backward(grad_state, decay, state_fp32) -> (grad_decay, grad_update)
//
// Forward:
//   decay: (B, T, D) bf16 | fp16 | fp32, contiguous row-major
//   update: (B, T, D) same dtype as decay (or fp16 update with bf16 decay;
//           see dispatch table in ssm_scan_fwd.cu)
//   Returns:
//     out        — (B, T, D) same dtype as update
//     state_fp32 — (B, T, D) ALWAYS fp32, the true register-level state
//                  snapshot; consumed by backward (the `out` tensor is
//                  lossy for bf16/fp16 so it cannot be used for grad).
//
// Backward:
//   grad_state: (B, T, D) upstream grad; dtype == update_dtype
//   decay:      (B, T, D) saved from forward; dtype == decay_dtype
//   state_fp32: (B, T, D) forward fp32 state (saved); ALWAYS fp32
//   Returns:    (grad_decay: decay_dtype, grad_update: update_dtype), both (B, T, D).
//
// Semantics: state[0] = decay[0] * 0 + update[0]; state[t] = decay[t] *
// state[t-1] + update[t]. Bit-identical (fp32 accumulator) to the
// Python reference at ``chaoscontrol.core._diag_recurrence_inner`` when
// that reference is also run in fp32. Backward derivatives follow the
// analytical reverse recurrence; matches autograd of the fp32 Python
// reference to fp32 bit-level.

#include "ssm_scan.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <stdexcept>
#include <tuple>

namespace cc_ssm_scan {

std::tuple<at::Tensor, at::Tensor> ssm_scan_forward(
    const at::Tensor& decay,
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
    // chain off whichever dtype update uses. `state_fp32` is always
    // fp32 (allocated here so the kernel never sees dtype ambiguity on
    // the state buffer). Memory cost at submission regime (B=1024,
    // T=512, D=256): ~512MB. Acceptable — we save autograd from having
    // to stash `update` separately, and backward reads fp32-exact.
    auto out = at::empty_like(update);
    auto state_fp32 = at::empty(
        decay.sizes(),
        update.options().dtype(at::kFloat));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_ssm_scan_fwd(
        decay.data_ptr(),
        update.data_ptr(),
        out.data_ptr(),
        state_fp32.data_ptr<float>(),
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
    return std::make_tuple(out, state_fp32);
}

std::tuple<at::Tensor, at::Tensor> ssm_scan_backward(
    const at::Tensor& grad_state,
    const at::Tensor& decay,
    const at::Tensor& state_fp32) {
    TORCH_CHECK(grad_state.is_cuda() && decay.is_cuda() && state_fp32.is_cuda(),
                "grad_state, decay, state_fp32 must all live on CUDA");
    TORCH_CHECK(grad_state.dim() == 3 && decay.dim() == 3 && state_fp32.dim() == 3,
                "grad_state, decay, state_fp32 must all be 3-D (B, T, D)");
    TORCH_CHECK(grad_state.sizes() == decay.sizes() &&
                grad_state.sizes() == state_fp32.sizes(),
                "grad_state, decay, state_fp32 must share shape (B, T, D)");
    TORCH_CHECK(grad_state.is_contiguous() && decay.is_contiguous() &&
                state_fp32.is_contiguous(),
                "grad_state, decay, state_fp32 must be contiguous row-major");
    TORCH_CHECK(grad_state.device() == decay.device() &&
                grad_state.device() == state_fp32.device(),
                "grad_state, decay, state_fp32 must live on the same CUDA device");
    TORCH_CHECK(state_fp32.scalar_type() == at::kFloat,
                "state_fp32 must be fp32 (got ", toString(state_fp32.scalar_type()),
                "); this is the fp32 register snapshot from the forward kernel, "
                "not the storage-dtype output tensor.");

    const auto decay_dtype = decay.scalar_type();
    const auto update_dtype = grad_state.scalar_type();

    // Dispatch table mirrors the forward. Any additional combos must be
    // added to BOTH dispatchers in lockstep.
    const bool bf16_bf16 = (decay_dtype == at::kBFloat16 && update_dtype == at::kBFloat16);
    const bool bf16_fp16 = (decay_dtype == at::kBFloat16 && update_dtype == at::kHalf);
    const bool fp16_fp16 = (decay_dtype == at::kHalf && update_dtype == at::kHalf);
    const bool fp32_fp32 = (decay_dtype == at::kFloat && update_dtype == at::kFloat);
    TORCH_CHECK(bf16_bf16 || bf16_fp16 || fp16_fp16 || fp32_fp32,
                "unsupported (decay, update) dtype combo for backward: (",
                toString(decay_dtype), ", ", toString(update_dtype),
                "); want bf16/bf16, bf16/fp16, fp16/fp16, or fp32/fp32");

    const int64_t B = decay.size(0);
    const int64_t T = decay.size(1);
    const int64_t D = decay.size(2);
    TORCH_CHECK(B > 0 && T > 0 && D > 0,
                "decay shape (", B, ", ", T, ", ", D, ") must be positive");
    TORCH_CHECK(B <= std::numeric_limits<int>::max() &&
                T <= std::numeric_limits<int>::max() &&
                D <= std::numeric_limits<int>::max(),
                "B/T/D exceeds int32");

    // grad_decay matches decay's dtype; grad_update matches update's dtype.
    auto grad_decay = at::empty_like(decay);
    auto grad_update = at::empty_like(grad_state);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_ssm_scan_bwd(
        grad_state.data_ptr(),
        decay.data_ptr(),
        state_fp32.data_ptr<float>(),
        grad_decay.data_ptr(),
        grad_update.data_ptr(),
        static_cast<int>(B),
        static_cast<int>(T),
        static_cast<int>(D),
        decay_dtype,
        update_dtype,
        stream);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cc_ssm_scan backward: launch failed: ") +
            cudaGetErrorString(err));
    }
    return std::make_tuple(grad_decay, grad_update);
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
  Tuple (out, state_fp32):
    out        — (B, T, D) state tensor; dtype matches ``update``.
    state_fp32 — (B, T, D) fp32 register-level state snapshot; always
                 fp32 regardless of input dtype. Required by backward
                 so gradients differentiate through the true fp32
                 recurrence, not a bf16/fp16-quantized surrogate.

Accumulator is fp32 inside the kernel regardless of input dtype.
)doc");

    m.def("ssm_scan_backward", &cc_ssm_scan::ssm_scan_backward,
          py::arg("grad_state"),
          py::arg("decay"),
          py::arg("state_fp32"),
          R"doc(
Backward diag SSM scan.

Given grad_state (upstream dL/ds), the saved decay, and the saved fp32
state snapshot from forward, compute grad_decay and grad_update per the
analytical reverse recurrence. Each (batch, channel) lane is independent
and computed in-kernel by a per-thread reverse-time serial scan with fp32
accumulator.

Args:
  grad_state: (B, T, D) upstream grad; dtype must match update dtype
    from the original forward call.
  decay:      (B, T, D) saved from forward; decay dtype.
  state_fp32: (B, T, D) saved fp32 register-level state from forward;
    must be fp32.

Returns:
  Tuple (grad_decay, grad_update). grad_decay dtype matches decay;
  grad_update dtype matches grad_state. Both (B, T, D), row-major
  contiguous.
)doc");
}
