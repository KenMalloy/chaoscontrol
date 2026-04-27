#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "wire_events.h"

namespace {

constexpr uint64_t kFingerprintModulus = (uint64_t{1} << 61) - 1;
constexpr uint64_t kFingerprintBase = 1000003;

__device__ __forceinline__ uint64_t add_mod_mersenne(uint64_t a, uint64_t b) {
  uint64_t sum = a + b;
  if (sum >= kFingerprintModulus) {
    sum -= kFingerprintModulus;
  }
  return sum;
}

__device__ uint64_t mul_base_mod_mersenne(uint64_t h) {
  uint64_t out = 0;
  uint64_t a = h;
  uint64_t b = kFingerprintBase;
  while (b != 0) {
    if ((b & 1U) != 0) {
      out = add_mod_mersenne(out, a);
    }
    a = add_mod_mersenne(a, a);
    b >>= 1U;
  }
  return out;
}

__device__ uint64_t fingerprint_window_i64(
    const int64_t* input_ids,
    int64_t B,
    int64_t T,
    int64_t b,
    int64_t t,
    int64_t W) {
  uint64_t h = 0;
  const int64_t row = b * T;
  for (int64_t i = t - W; i < t; ++i) {
    const uint64_t tok = static_cast<uint64_t>(input_ids[row + i]);
    h = add_mod_mersenne(
        mul_base_mod_mersenne(h),
        (tok + 1U) % kFingerprintModulus);
  }
  return h;
}

template <typename scalar_t>
__device__ __forceinline__ float to_float_device(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float_device<c10::Half>(c10::Half value) {
  const __half raw = *reinterpret_cast<const __half*>(&value);
  return __half2float(raw);
}

template <>
__device__ __forceinline__ float to_float_device<c10::BFloat16>(
    c10::BFloat16 value) {
  const __nv_bfloat16 raw = *reinterpret_cast<const __nv_bfloat16*>(&value);
  return __bfloat162float(raw);
}

__device__ __forceinline__ uint16_t fp32_to_fp16_bits(float value) {
  const __half h = __float2half_rn(value);
  return __half_as_ushort(h);
}

template <typename rep_t>
__global__ void pack_write_events_kernel(
    uint8_t* out_bytes,
    const int64_t* input_ids,
    const int64_t* target_ids,
    const rep_t* key_rep,
    const float* pressure,
    const float* per_token_ce,
    const int64_t* positions,
    const int64_t* candidate_base,
    int64_t M,
    int64_t B,
    int64_t T,
    int64_t D,
    uint64_t gpu_step,
    uint8_t source_rank,
    uint8_t write_bucket,
    int64_t fingerprint_window,
    int64_t span_length,
    int64_t key_rep_dim) {
  const int64_t m = static_cast<int64_t>(blockIdx.x);
  if (m >= M) {
    return;
  }
  auto* ev = reinterpret_cast<WriteEvent*>(
      out_bytes + static_cast<std::size_t>(m) * sizeof(WriteEvent));
  auto* ev_bytes = reinterpret_cast<uint8_t*>(ev);
  auto* key_rep_wire = reinterpret_cast<uint16_t*>(
      ev_bytes + offsetof(WriteEvent, key_rep));
  auto* value_tok_wire = reinterpret_cast<uint16_t*>(
      ev_bytes + offsetof(WriteEvent, value_tok_ids));
  for (int64_t idx = threadIdx.x; idx < static_cast<int64_t>(sizeof(WriteEvent));
       idx += blockDim.x) {
    ev_bytes[idx] = 0;
  }
  __syncthreads();

  const int64_t b = positions[m * 2 + 0];
  const int64_t t = positions[m * 2 + 1];
  const bool valid =
      b >= 0 && b < B && t >= fingerprint_window && t + span_length <= T &&
      key_rep_dim > 0 && key_rep_dim <= KEY_REP_DIM_DEFAULT &&
      span_length > 0 && span_length <= SPAN_LENGTH_DEFAULT &&
      D >= key_rep_dim;
  if (!valid) {
    return;
  }

  if (threadIdx.x == 0) {
    const uint64_t low = static_cast<uint64_t>(*candidate_base) +
        static_cast<uint64_t>(m);
    ev->event_type = 1;
    ev->source_rank = source_rank;
    ev->write_bucket = write_bucket;
    ev->candidate_id =
        (static_cast<uint64_t>(source_rank) << 56) | (low & ((uint64_t{1} << 56) - 1));
    ev->gpu_step = gpu_step;
    ev->key_fp = fingerprint_window_i64(
        input_ids, B, T, b, t, fingerprint_window);
    ev->value_anchor_id = static_cast<uint32_t>(target_ids[b * T + t]);
    ev->pressure_at_write = pressure == nullptr ? 1.0f : pressure[b * T + t];
    ev->pre_write_ce = per_token_ce[b * T + t];
  }
  for (int64_t j = threadIdx.x; j < key_rep_dim; j += blockDim.x) {
    const float value = to_float_device<rep_t>(
        key_rep[(b * T + t) * D + j]);
    key_rep_wire[j] = fp32_to_fp16_bits(value);
  }
  for (int64_t j = threadIdx.x; j < span_length; j += blockDim.x) {
    value_tok_wire[j] = static_cast<uint16_t>(target_ids[b * T + t + j]);
  }
}

void check_cuda_tensor(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void pack_write_events_cuda_(
    at::Tensor out,
    const at::Tensor& input_ids,
    const at::Tensor& target_ids,
    const at::Tensor& key_rep,
    const at::Tensor& pressure,
    const at::Tensor& per_token_ce,
    const at::Tensor& positions,
    const at::Tensor& candidate_base,
    int64_t gpu_step,
    int64_t source_rank,
    int64_t write_bucket,
    int64_t fingerprint_window,
    int64_t span_length,
    int64_t key_rep_dim) {
  check_cuda_tensor(out, "out");
  check_cuda_tensor(input_ids, "input_ids");
  check_cuda_tensor(target_ids, "target_ids");
  check_cuda_tensor(key_rep, "key_rep");
  check_cuda_tensor(per_token_ce, "per_token_ce");
  check_cuda_tensor(positions, "positions");
  check_cuda_tensor(candidate_base, "candidate_base");
  TORCH_CHECK(
      input_ids.scalar_type() == at::ScalarType::Long,
      "input_ids must be int64");
  TORCH_CHECK(
      target_ids.scalar_type() == at::ScalarType::Long,
      "target_ids must be int64");
  TORCH_CHECK(
      per_token_ce.scalar_type() == at::ScalarType::Float,
      "per_token_ce must be float32");
  TORCH_CHECK(
      pressure.numel() == 0 ||
          (pressure.is_cuda() && pressure.is_contiguous() &&
           pressure.scalar_type() == at::ScalarType::Float),
      "pressure must be empty or a contiguous CUDA float32 tensor");
  TORCH_CHECK(
      positions.scalar_type() == at::ScalarType::Long,
      "positions must be int64");
  TORCH_CHECK(
      candidate_base.scalar_type() == at::ScalarType::Long &&
          candidate_base.numel() == 1,
      "candidate_base must be a scalar int64 tensor");
  TORCH_CHECK(
      out.scalar_type() == at::ScalarType::Byte,
      "out must have dtype=torch.uint8");
  TORCH_CHECK(
      out.dim() == 2 && out.size(1) == static_cast<int64_t>(sizeof(WriteEvent)),
      "out must have shape [M, sizeof(WriteEvent)]");
  TORCH_CHECK(input_ids.dim() == 2, "input_ids must be [B, T]");
  TORCH_CHECK(target_ids.sizes() == input_ids.sizes(), "target_ids shape mismatch");
  TORCH_CHECK(per_token_ce.sizes() == input_ids.sizes(), "per_token_ce shape mismatch");
  TORCH_CHECK(
      pressure.numel() == 0 || pressure.sizes() == input_ids.sizes(),
      "pressure shape mismatch");
  TORCH_CHECK(key_rep.dim() == 3, "key_rep must be [B, T, D]");
  TORCH_CHECK(
      key_rep.size(0) == input_ids.size(0) && key_rep.size(1) == input_ids.size(1),
      "key_rep [B, T] shape mismatch");
  TORCH_CHECK(
      positions.dim() == 2 && positions.size(1) == 2 &&
          positions.size(0) == out.size(0),
      "positions must be [M, 2] matching out");
  TORCH_CHECK(source_rank >= 0 && source_rank < 256, "source_rank must fit u8");
  TORCH_CHECK(write_bucket >= 0 && write_bucket < 256, "write_bucket must fit u8");
  const int64_t M = out.size(0);
  if (M == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(M));
  const dim3 block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  const float* pressure_ptr =
      pressure.numel() == 0 ? nullptr : pressure.data_ptr<float>();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      key_rep.scalar_type(),
      "pack_write_events_cuda",
      [&] {
        pack_write_events_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<uint8_t>(),
            input_ids.data_ptr<int64_t>(),
            target_ids.data_ptr<int64_t>(),
            key_rep.data_ptr<scalar_t>(),
            pressure_ptr,
            per_token_ce.data_ptr<float>(),
            positions.data_ptr<int64_t>(),
            candidate_base.data_ptr<int64_t>(),
            M,
            input_ids.size(0),
            input_ids.size(1),
            key_rep.size(2),
            static_cast<uint64_t>(gpu_step),
            static_cast<uint8_t>(source_rank),
            static_cast<uint8_t>(write_bucket),
            fingerprint_window,
            span_length,
            key_rep_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
