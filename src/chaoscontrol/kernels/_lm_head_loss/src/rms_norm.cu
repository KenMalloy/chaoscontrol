#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>

#include "rms_norm.h"

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
__global__ void rms_norm_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    float* __restrict__ inv_rms,
    int64_t rows,
    int dim,
    float eps) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(row) * dim;

    float ss = 0.0f;
    for (int col = tid; col < dim; col += Block) {
        const float xv = load_as_float(x + base + col);
        ss += xv * xv;
    }

    __shared__ float scratch[Block];
    scratch[tid] = ss;
    __syncthreads();
    for (int stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float inv = rsqrtf(scratch[0] / static_cast<float>(dim) + eps);
    if (tid == 0) {
        inv_rms[row] = inv;
    }
    for (int col = tid; col < dim; col += Block) {
        const float xv = load_as_float(x + base + col);
        const float wv = load_as_float(weight + col);
        store_from_float(out + base + col, xv * inv * wv);
    }
}

template <typename scalar_t, int Block>
__global__ void rms_norm_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const float* __restrict__ inv_rms,
    scalar_t* __restrict__ grad_x,
    float* __restrict__ grad_weight,
    int64_t rows,
    int dim) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(row) * dim;
    const float inv = inv_rms[row];

    float dot = 0.0f;
    for (int col = tid; col < dim; col += Block) {
        const float go = load_as_float(grad_out + base + col);
        const float xv = load_as_float(x + base + col);
        const float wv = load_as_float(weight + col);
        dot += go * wv * xv;
    }

    __shared__ float scratch[Block];
    scratch[tid] = dot;
    __syncthreads();
    for (int stride = Block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float coeff = scratch[0] * inv * inv * inv / static_cast<float>(dim);
    for (int col = tid; col < dim; col += Block) {
        const float go = load_as_float(grad_out + base + col);
        const float xv = load_as_float(x + base + col);
        const float wv = load_as_float(weight + col);
        const float gx = go * wv * inv - xv * coeff;
        const float gw = go * xv * inv;
        store_from_float(grad_x + base + col, gx);
        atomicAdd(grad_weight + col, gw);
    }
}

template <typename scalar_t>
void launch_forward_typed(
    const at::Tensor& x,
    const at::Tensor& weight,
    at::Tensor& out,
    at::Tensor& inv_rms,
    double eps) {
    constexpr int Block = 256;
    const auto rows = x.numel() / x.size(-1);
    const auto dim = static_cast<int>(x.size(-1));
    const dim3 grid(rows);
    const dim3 block(Block);
    auto stream = at::cuda::getCurrentCUDAStream();
    rms_norm_fwd_kernel<scalar_t, Block><<<grid, block, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        inv_rms.data_ptr<float>(),
        rows,
        dim,
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_backward_typed(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& inv_rms,
    at::Tensor& grad_x,
    at::Tensor& grad_weight) {
    constexpr int Block = 256;
    const auto rows = x.numel() / x.size(-1);
    const auto dim = static_cast<int>(x.size(-1));
    const dim3 grid(rows);
    const dim3 block(Block);
    auto stream = at::cuda::getCurrentCUDAStream();
    rms_norm_bwd_kernel<scalar_t, Block><<<grid, block, 0, stream>>>(
        grad_out.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        inv_rms.data_ptr<float>(),
        grad_x.data_ptr<scalar_t>(),
        grad_weight.data_ptr<float>(),
        rows,
        dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void launch_rms_norm_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    at::Tensor& out,
    at::Tensor& inv_rms,
    double eps) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(),
        "cc_lm_head_loss_rms_norm_forward",
        [&] {
            launch_forward_typed<scalar_t>(x, weight, out, inv_rms, eps);
        });
}

void launch_rms_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& inv_rms,
    at::Tensor& grad_x,
    at::Tensor& grad_weight) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(),
        "cc_lm_head_loss_rms_norm_backward",
        [&] {
            launch_backward_typed<scalar_t>(
                grad_out, x, weight, inv_rms, grad_x, grad_weight);
        });
}

}  // namespace cc_lm_head_loss
