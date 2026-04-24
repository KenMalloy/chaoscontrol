#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <limits>

#include "linear_ce.h"

namespace cc_lm_head_loss {
namespace {

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p) {
    return static_cast<float>(*p);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T* p, float v) {
    *p = static_cast<T>(v);
}

template <typename scalar_t, int Block>
__global__ void linear_ce_update_max_and_target_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ row_max,
    float* __restrict__ target_logits,
    int64_t rows,
    int tile_cols,
    int64_t tile_start) {
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    const int64_t base = row * static_cast<int64_t>(tile_cols);

    float local_max = -INFINITY;
    for (int col = tid; col < tile_cols; col += Block) {
        local_max = fmaxf(local_max, load_as_float(logits + base + col));
    }

    __shared__ float scratch[Block];
    scratch[tid] = local_max;
    __syncthreads();
    for (int stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_max[row] = fmaxf(row_max[row], scratch[0]);
        const int64_t target = targets[row];
        const int64_t offset = target - tile_start;
        if (offset >= 0 && offset < tile_cols) {
            target_logits[row] = load_as_float(logits + base + offset);
        }
    }
}

template <typename scalar_t, int Block>
__global__ void linear_ce_accum_sum_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ row_max,
    float* __restrict__ row_sum,
    int64_t rows,
    int tile_cols) {
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    const int64_t base = row * static_cast<int64_t>(tile_cols);
    const float max_v = row_max[row];

    float local_sum = 0.0f;
    for (int col = tid; col < tile_cols; col += Block) {
        const float logit = load_as_float(logits + base + col);
        local_sum += expf(logit - max_v);
    }

    __shared__ float scratch[Block];
    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_sum[row] += scratch[0];
    }
}

template <typename logits_t, typename grad_t>
__global__ void linear_ce_fill_grad_logits_kernel(
    const logits_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ lse,
    const float* __restrict__ grad_loss,
    grad_t* __restrict__ grad_logits,
    int64_t rows,
    int tile_cols,
    int64_t tile_start,
    float inv_divisor) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = rows * static_cast<int64_t>(tile_cols);
    if (idx >= total) {
        return;
    }

    const int64_t row = idx / tile_cols;
    const int col = static_cast<int>(idx - row * static_cast<int64_t>(tile_cols));
    float grad = expf(load_as_float(logits + idx) - lse[row]);
    if (targets[row] == tile_start + col) {
        grad -= 1.0f;
    }
    grad *= load_as_float(grad_loss) * inv_divisor;
    store_from_float(grad_logits + idx, grad);
}

template <typename logits_t, typename grad_t>
__global__ void linear_ce_fill_grad_logits_weighted_kernel(
    const logits_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ lse,
    const float* __restrict__ grad_loss,
    const float* __restrict__ row_weight,
    const float* __restrict__ normalizer,
    grad_t* __restrict__ grad_logits,
    int64_t rows,
    int tile_cols,
    int64_t tile_start) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = rows * static_cast<int64_t>(tile_cols);
    if (idx >= total) {
        return;
    }

    const int64_t row = idx / tile_cols;
    const int col = static_cast<int>(idx - row * static_cast<int64_t>(tile_cols));
    float grad = expf(load_as_float(logits + idx) - lse[row]);
    if (targets[row] == tile_start + col) {
        grad -= 1.0f;
    }
    grad *= load_as_float(grad_loss) * row_weight[row] / fmaxf(normalizer[0], 1.0f);
    store_from_float(grad_logits + idx, grad);
}

template <typename scalar_t, int Block>
__global__ void linear_ce_update_online_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    float* __restrict__ target_logits,
    int64_t rows,
    int tile_cols,
    int64_t tile_start) {
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    const int64_t base = row * static_cast<int64_t>(tile_cols);

    float local_max = -INFINITY;
    for (int col = tid; col < tile_cols; col += Block) {
        local_max = fmaxf(local_max, load_as_float(logits + base + col));
    }

    __shared__ float scratch[Block];
    scratch[tid] = local_max;
    __syncthreads();
    for (int stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }

    const float tile_max = scratch[0];
    const float old_max = row_max[row];
    const float new_max = fmaxf(old_max, tile_max);

    float local_sum = 0.0f;
    const int64_t target = targets[row];
    const int64_t target_offset = target - tile_start;
    for (int col = tid; col < tile_cols; col += Block) {
        const float logit = load_as_float(logits + base + col);
        local_sum += expf(logit - new_max);
        if (target_offset == col) {
            target_logits[row] = logit;
        }
    }

    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float scaled_old_sum = isfinite(old_max)
            ? row_sum[row] * expf(old_max - new_max)
            : 0.0f;
        row_sum[row] = scaled_old_sum + scratch[0];
        row_max[row] = new_max;
    }
}

template <typename scalar_t>
void launch_update_max_and_target_typed(
    const at::Tensor& logits,
    const at::Tensor& targets,
    at::Tensor& row_max,
    at::Tensor& target_logits,
    int64_t tile_start) {
    constexpr int Block = 256;
    const auto rows = logits.size(0);
    const auto tile_cols = static_cast<int>(logits.size(1));
    const dim3 grid(rows);
    const dim3 block(Block);
    auto stream = at::cuda::getCurrentCUDAStream();
    linear_ce_update_max_and_target_kernel<scalar_t, Block><<<grid, block, 0, stream>>>(
        logits.data_ptr<scalar_t>(),
        targets.data_ptr<int64_t>(),
        row_max.data_ptr<float>(),
        target_logits.data_ptr<float>(),
        rows,
        tile_cols,
        tile_start);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_accum_sum_typed(
    const at::Tensor& logits,
    const at::Tensor& row_max,
    at::Tensor& row_sum) {
    constexpr int Block = 256;
    const auto rows = logits.size(0);
    const auto tile_cols = static_cast<int>(logits.size(1));
    const dim3 grid(rows);
    const dim3 block(Block);
    auto stream = at::cuda::getCurrentCUDAStream();
    linear_ce_accum_sum_kernel<scalar_t, Block><<<grid, block, 0, stream>>>(
        logits.data_ptr<scalar_t>(),
        row_max.data_ptr<float>(),
        row_sum.data_ptr<float>(),
        rows,
        tile_cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_update_online_typed(
    const at::Tensor& logits,
    const at::Tensor& targets,
    at::Tensor& row_max,
    at::Tensor& row_sum,
    at::Tensor& target_logits,
    int64_t tile_start) {
    constexpr int Block = 256;
    const auto rows = logits.size(0);
    const auto tile_cols = static_cast<int>(logits.size(1));
    const dim3 grid(rows);
    const dim3 block(Block);
    auto stream = at::cuda::getCurrentCUDAStream();
    linear_ce_update_online_kernel<scalar_t, Block><<<grid, block, 0, stream>>>(
        logits.data_ptr<scalar_t>(),
        targets.data_ptr<int64_t>(),
        row_max.data_ptr<float>(),
        row_sum.data_ptr<float>(),
        target_logits.data_ptr<float>(),
        rows,
        tile_cols,
        tile_start);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename logits_t, typename grad_t>
void launch_fill_grad_logits_typed(
    const at::Tensor& logits,
    const at::Tensor& targets,
    const at::Tensor& lse,
    const at::Tensor& grad_loss,
    at::Tensor& grad_logits,
    int64_t tile_start,
    double divisor) {
    constexpr int Block = 256;
    const auto rows = logits.size(0);
    const auto tile_cols = static_cast<int>(logits.size(1));
    const int64_t total = rows * static_cast<int64_t>(tile_cols);
    const dim3 grid((total + Block - 1) / Block);
    const dim3 block(Block);
    const float inv_divisor = 1.0f / static_cast<float>(divisor);
    auto stream = at::cuda::getCurrentCUDAStream();
    linear_ce_fill_grad_logits_kernel<logits_t, grad_t><<<grid, block, 0, stream>>>(
        logits.data_ptr<logits_t>(),
        targets.data_ptr<int64_t>(),
        lse.data_ptr<float>(),
        grad_loss.data_ptr<float>(),
        grad_logits.data_ptr<grad_t>(),
        rows,
        tile_cols,
        tile_start,
        inv_divisor);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename logits_t, typename grad_t>
void launch_fill_grad_logits_weighted_typed(
    const at::Tensor& logits,
    const at::Tensor& targets,
    const at::Tensor& lse,
    const at::Tensor& grad_loss,
    const at::Tensor& row_weight,
    const at::Tensor& normalizer,
    at::Tensor& grad_logits,
    int64_t tile_start) {
    constexpr int Block = 256;
    const auto rows = logits.size(0);
    const auto tile_cols = static_cast<int>(logits.size(1));
    const int64_t total = rows * static_cast<int64_t>(tile_cols);
    const dim3 grid((total + Block - 1) / Block);
    const dim3 block(Block);
    auto stream = at::cuda::getCurrentCUDAStream();
    linear_ce_fill_grad_logits_weighted_kernel<logits_t, grad_t><<<grid, block, 0, stream>>>(
        logits.data_ptr<logits_t>(),
        targets.data_ptr<int64_t>(),
        lse.data_ptr<float>(),
        grad_loss.data_ptr<float>(),
        row_weight.data_ptr<float>(),
        normalizer.data_ptr<float>(),
        grad_logits.data_ptr<grad_t>(),
        rows,
        tile_cols,
        tile_start);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void launch_linear_ce_update_max_and_target(
    const at::Tensor& logits,
    const at::Tensor& targets,
    at::Tensor& row_max,
    at::Tensor& target_logits,
    int64_t tile_start) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(),
        "cc_lm_head_loss_linear_ce_update_max_and_target",
        [&] {
            launch_update_max_and_target_typed<scalar_t>(
                logits, targets, row_max, target_logits, tile_start);
        });
}

void launch_linear_ce_accum_sum(
    const at::Tensor& logits,
    const at::Tensor& row_max,
    at::Tensor& row_sum) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(),
        "cc_lm_head_loss_linear_ce_accum_sum",
        [&] {
            launch_accum_sum_typed<scalar_t>(logits, row_max, row_sum);
        });
}

void launch_linear_ce_update_online(
    const at::Tensor& logits,
    const at::Tensor& targets,
    at::Tensor& row_max,
    at::Tensor& row_sum,
    at::Tensor& target_logits,
    int64_t tile_start) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(),
        "cc_lm_head_loss_linear_ce_update_online",
        [&] {
            launch_update_online_typed<scalar_t>(
                logits, targets, row_max, row_sum, target_logits, tile_start);
        });
}

void launch_linear_ce_fill_grad_logits(
    const at::Tensor& logits,
    const at::Tensor& targets,
    const at::Tensor& lse,
    const at::Tensor& grad_loss,
    at::Tensor& grad_logits,
    int64_t tile_start,
    double divisor) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(),
        "cc_lm_head_loss_linear_ce_fill_grad_logits_logits",
        [&] {
            using logits_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, grad_logits.scalar_type(),
                "cc_lm_head_loss_linear_ce_fill_grad_logits_grad",
                [&] {
                    launch_fill_grad_logits_typed<logits_t, scalar_t>(
                        logits, targets, lse, grad_loss, grad_logits, tile_start, divisor);
                });
        });
}

void launch_linear_ce_fill_grad_logits_weighted(
    const at::Tensor& logits,
    const at::Tensor& targets,
    const at::Tensor& lse,
    const at::Tensor& grad_loss,
    const at::Tensor& row_weight,
    const at::Tensor& normalizer,
    at::Tensor& grad_logits,
    int64_t tile_start) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, logits.scalar_type(),
        "cc_lm_head_loss_linear_ce_fill_grad_logits_weighted_logits",
        [&] {
            using logits_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf, at::kBFloat16, grad_logits.scalar_type(),
                "cc_lm_head_loss_linear_ce_fill_grad_logits_weighted_grad",
                [&] {
                    launch_fill_grad_logits_weighted_typed<logits_t, scalar_t>(
                        logits,
                        targets,
                        lse,
                        grad_loss,
                        row_weight,
                        normalizer,
                        grad_logits,
                        tile_start);
                });
        });
}

}  // namespace cc_lm_head_loss
