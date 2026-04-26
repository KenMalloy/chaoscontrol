#include "avx512_matops.h"

#include <cstddef>

#include "cpu_features.h"

#if defined(__x86_64__) && defined(__AVX512F__) && \
    defined(CHAOSCONTROL_CPU_SSM_AVX512_KERNEL)
#define CHAOSCONTROL_AVX512_MATOPS_KERNEL_COMPILED 1
#include <immintrin.h>
#endif

namespace chaoscontrol::avx512 {
namespace {

[[maybe_unused]] void check_matvec_inputs(
    const at::Tensor& w,
    const at::Tensor& decay,
    const at::Tensor& state,
    const at::Tensor& x,
    const at::Tensor& out) {
  TORCH_CHECK(
      w.device().is_cpu(),
      "avx512_matvec_fma_with_decay: w must be a CPU tensor");
  TORCH_CHECK(
      decay.device().is_cpu(),
      "avx512_matvec_fma_with_decay: decay must be a CPU tensor");
  TORCH_CHECK(
      state.device().is_cpu(),
      "avx512_matvec_fma_with_decay: state must be a CPU tensor");
  TORCH_CHECK(
      x.device().is_cpu(),
      "avx512_matvec_fma_with_decay: x must be a CPU tensor");
  TORCH_CHECK(
      out.device().is_cpu(),
      "avx512_matvec_fma_with_decay: out must be a CPU tensor");

  TORCH_CHECK(
      w.scalar_type() == at::kFloat,
      "avx512_matvec_fma_with_decay: w must be float32");
  TORCH_CHECK(
      decay.scalar_type() == at::kFloat,
      "avx512_matvec_fma_with_decay: decay must be float32");
  TORCH_CHECK(
      state.scalar_type() == at::kFloat,
      "avx512_matvec_fma_with_decay: state must be float32");
  TORCH_CHECK(
      x.scalar_type() == at::kFloat,
      "avx512_matvec_fma_with_decay: x must be float32");
  TORCH_CHECK(
      out.scalar_type() == at::kFloat,
      "avx512_matvec_fma_with_decay: out must be float32");

  TORCH_CHECK(
      w.dim() == 2,
      "avx512_matvec_fma_with_decay: w must be 2-dimensional [N, K]");
  TORCH_CHECK(
      decay.dim() == 1,
      "avx512_matvec_fma_with_decay: decay must be 1-dimensional");
  TORCH_CHECK(
      state.dim() == 1,
      "avx512_matvec_fma_with_decay: state must be 1-dimensional");
  TORCH_CHECK(
      x.dim() == 1, "avx512_matvec_fma_with_decay: x must be 1-dimensional");
  TORCH_CHECK(
      out.dim() == 1,
      "avx512_matvec_fma_with_decay: out must be 1-dimensional");

  const auto n = w.size(0);
  const auto k = w.size(1);
  TORCH_CHECK(
      decay.size(0) == n,
      "avx512_matvec_fma_with_decay: decay shape must match w.size(0)");
  TORCH_CHECK(
      state.size(0) == n,
      "avx512_matvec_fma_with_decay: state shape must match w.size(0)");
  TORCH_CHECK(
      out.size(0) == n,
      "avx512_matvec_fma_with_decay: out shape must match w.size(0)");
  TORCH_CHECK(
      x.size(0) == k,
      "avx512_matvec_fma_with_decay: x shape must match w.size(1)");

  TORCH_CHECK(
      w.is_contiguous(),
      "avx512_matvec_fma_with_decay: w must be contiguous");
  TORCH_CHECK(
      decay.is_contiguous(),
      "avx512_matvec_fma_with_decay: decay must be contiguous");
  TORCH_CHECK(
      state.is_contiguous(),
      "avx512_matvec_fma_with_decay: state must be contiguous");
  TORCH_CHECK(
      x.is_contiguous(),
      "avx512_matvec_fma_with_decay: x must be contiguous");
  TORCH_CHECK(
      out.is_contiguous(),
      "avx512_matvec_fma_with_decay: out must be contiguous");
}

[[maybe_unused]] void check_axpy_inputs(
    const at::Tensor& x,
    const at::Tensor& y) {
  TORCH_CHECK(x.device().is_cpu(), "avx512_axpy_fma: x must be a CPU tensor");
  TORCH_CHECK(y.device().is_cpu(), "avx512_axpy_fma: y must be a CPU tensor");
  TORCH_CHECK(
      x.scalar_type() == at::kFloat, "avx512_axpy_fma: x must be float32");
  TORCH_CHECK(
      y.scalar_type() == at::kFloat, "avx512_axpy_fma: y must be float32");
  TORCH_CHECK(x.dim() == 1, "avx512_axpy_fma: x must be 1-dimensional");
  TORCH_CHECK(y.dim() == 1, "avx512_axpy_fma: y must be 1-dimensional");
  TORCH_CHECK(
      x.sizes().equals(y.sizes()),
      "avx512_axpy_fma: x and y must have the same shape");
  TORCH_CHECK(x.is_contiguous(), "avx512_axpy_fma: x must be contiguous");
  TORCH_CHECK(y.is_contiguous(), "avx512_axpy_fma: y must be contiguous");
}

[[noreturn]] void raise_kernel_unavailable() {
  TORCH_CHECK(
      false,
      "AVX-512 matops kernel unavailable: extension was not compiled with "
      "AVX-512 kernel support. Rebuild on x86_64/amd64 with "
      "CHAOSCONTROL_CPU_SSM_X86_ACCEL=1 to enable it.");
}

}  // namespace

bool avx512_matops_kernel_available() {
#if defined(CHAOSCONTROL_AVX512_MATOPS_KERNEL_COMPILED)
  return true;
#else
  return false;
#endif
}

void avx512_matvec_fma_with_decay(
    const at::Tensor& w,
    const at::Tensor& decay,
    const at::Tensor& state,
    const at::Tensor& x,
    at::Tensor out) {
#if defined(CHAOSCONTROL_AVX512_MATOPS_KERNEL_COMPILED)
  check_matvec_inputs(w, decay, state, x, out);
  TORCH_CHECK(
      chaoscontrol::cpu_features::runtime_has_avx512f(),
      "AVX-512 matops kernel unavailable at runtime: hardware or OS "
      "AVX-512 state is not enabled");

  const auto n = static_cast<std::size_t>(w.size(0));
  const auto k = static_cast<std::size_t>(w.size(1));
  const float* w_ptr = w.data_ptr<float>();
  const float* decay_ptr = decay.data_ptr<float>();
  const float* state_ptr = state.data_ptr<float>();
  const float* x_ptr = x.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();

  const std::size_t k_aligned = k - (k % 16);

  for (std::size_t i = 0; i < n; ++i) {
    const float* w_row = w_ptr + i * k;

    // Seed the accumulator's first lane with decay[i] * state[i] so the
    // horizontal reduce returns out[i] in one go (saves a scalar add).
    __m512 acc = _mm512_setzero_ps();
    {
      // Lane 0 carries the seed; remaining lanes stay zero. The
      // _mm512_mask_set1_ps pattern would also work, but a plain set with
      // a single populated lane keeps the intent obvious.
      const float seed = decay_ptr[i] * state_ptr[i];
      acc = _mm512_set_ps(
          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, seed);
    }

    std::size_t j = 0;
    for (; j < k_aligned; j += 16) {
      // loadu over load: torch's allocator returns 64B-aligned storage for
      // contiguous tensors, but sliced/strided callers can land on
      // unaligned offsets. Modern Intel/AMD uarch has no penalty for
      // unaligned loads on aligned pointers, so loadu is correctness-
      // equivalent and faster in the unaligned case.
      const __m512 wv = _mm512_loadu_ps(w_row + j);
      const __m512 xv = _mm512_loadu_ps(x_ptr + j);
      // Explicit FMA: under -O3 clang/gcc would fuse add(mul(...)) into
      // vfmadd231ps anyway, but spelling it out removes any ambiguity
      // about rounding mode and survives -fno-fma builds.
      acc = _mm512_fmadd_ps(wv, xv, acc);
    }

    float dot = _mm512_reduce_add_ps(acc);
    for (; j < k; ++j) {
      dot += w_row[j] * x_ptr[j];
    }
    out_ptr[i] = dot;
  }
#else
  (void)w;
  (void)decay;
  (void)state;
  (void)x;
  (void)out;
  raise_kernel_unavailable();
#endif
}

void avx512_axpy_fma(float alpha, const at::Tensor& x, at::Tensor y) {
#if defined(CHAOSCONTROL_AVX512_MATOPS_KERNEL_COMPILED)
  check_axpy_inputs(x, y);
  TORCH_CHECK(
      chaoscontrol::cpu_features::runtime_has_avx512f(),
      "AVX-512 matops kernel unavailable at runtime: hardware or OS "
      "AVX-512 state is not enabled");

  const auto k = static_cast<std::size_t>(y.numel());
  const float* x_ptr = x.data_ptr<float>();
  float* y_ptr = y.data_ptr<float>();

  const __m512 alpha_v = _mm512_set1_ps(alpha);
  std::size_t j = 0;
  for (; j + 16 <= k; j += 16) {
    const __m512 xv = _mm512_loadu_ps(x_ptr + j);
    const __m512 yv = _mm512_loadu_ps(y_ptr + j);
    _mm512_storeu_ps(y_ptr + j, _mm512_fmadd_ps(alpha_v, xv, yv));
  }
  for (; j < k; ++j) {
    y_ptr[j] += alpha * x_ptr[j];
  }
#else
  (void)alpha;
  (void)x;
  (void)y;
  raise_kernel_unavailable();
#endif
}

}  // namespace chaoscontrol::avx512
