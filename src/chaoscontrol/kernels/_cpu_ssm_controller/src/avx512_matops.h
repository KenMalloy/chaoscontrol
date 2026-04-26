#pragma once

#include <cstddef>

#include <torch/extension.h>

namespace chaoscontrol::avx512 {

bool avx512_matops_kernel_available();

// Raw-pointer C++ overloads. The at::Tensor variants below validate +
// dispatch into these; internal C++ callers (e.g., OnlineLearningController's
// per-event hot path operating on std::vector<float> buffers) call these
// directly to skip tensor-creation overhead per row. The runtime AVX-512
// availability check is left to the caller — the OnlineLearningController
// caches the dispatch decision once at construction.
//
// out[i] = decay[i] * state[i] + sum_j(w[i, j] * x[j]) for i in [0, N).
// Shapes: w is fp32 row-major contiguous of size N * K; decay, state are
// of length N; x is of length K; out is of length N.
void avx512_matvec_fma_with_decay_raw(
    const float* w,
    const float* decay,
    const float* state,
    const float* x,
    float* out,
    std::size_t n,
    std::size_t k);

// y[j] += alpha * x[j] for j in [0, k). x and y are fp32 of length k.
void avx512_axpy_fma_raw(
    float alpha, const float* x, float* y, std::size_t k);

// out[i] = decay[i] * state[i] + sum_j(w[i, j] * x[j]) for i in [0, N).
//
// Shapes: w is [N, K] fp32 row-major contiguous; decay, state are [N];
// x is [K]; out is [N]. All fp32. K need not be a multiple of 16 (the
// inner loop handles the K % 16 tail with a scalar fallback).
void avx512_matvec_fma_with_decay(
    const at::Tensor& w,
    const at::Tensor& decay,
    const at::Tensor& state,
    const at::Tensor& x,
    at::Tensor out);

// y[j] += alpha * x[j] for j in [0, K). Updates y in place.
//
// Shapes: x and y are 1-D fp32 contiguous tensors of equal length. K
// need not be a multiple of 16 (tail handled with a scalar fallback).
void avx512_axpy_fma(float alpha, const at::Tensor& x, at::Tensor y);

}  // namespace chaoscontrol::avx512
