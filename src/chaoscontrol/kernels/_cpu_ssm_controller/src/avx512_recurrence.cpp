#include "avx512_recurrence.h"

#include <cstddef>

#include "cpu_features.h"

#if defined(__x86_64__) && defined(__AVX512F__) && \
    defined(CHAOSCONTROL_CPU_SSM_AVX512_KERNEL)
#define CHAOSCONTROL_AVX512_RECURRENCE_KERNEL_COMPILED 1
#include <immintrin.h>
#endif

namespace chaoscontrol::avx512 {
namespace {

[[maybe_unused]] void check_recurrence_inputs(
    const at::Tensor& decay,
    const at::Tensor& x,
    const at::Tensor& h) {
  TORCH_CHECK(
      decay.device().is_cpu(),
      "avx512_diagonal_recurrence: decay must be a CPU tensor");
  TORCH_CHECK(
      x.device().is_cpu(),
      "avx512_diagonal_recurrence: x must be a CPU tensor");
  TORCH_CHECK(
      h.device().is_cpu(),
      "avx512_diagonal_recurrence: h must be a CPU tensor");
  TORCH_CHECK(
      decay.scalar_type() == at::kFloat,
      "avx512_diagonal_recurrence: decay must be float32");
  TORCH_CHECK(
      x.scalar_type() == at::kFloat,
      "avx512_diagonal_recurrence: x must be float32");
  TORCH_CHECK(
      h.scalar_type() == at::kFloat,
      "avx512_diagonal_recurrence: h must be float32");
  TORCH_CHECK(
      decay.dim() == 1,
      "avx512_diagonal_recurrence: decay must be 1-dimensional");
  TORCH_CHECK(
      x.dim() == 1,
      "avx512_diagonal_recurrence: x must be 1-dimensional");
  TORCH_CHECK(
      h.dim() == 1,
      "avx512_diagonal_recurrence: h must be 1-dimensional");
  TORCH_CHECK(
      decay.sizes().equals(x.sizes()) && decay.sizes().equals(h.sizes()),
      "avx512_diagonal_recurrence: decay, x, and h must have the same shape");
  TORCH_CHECK(
      decay.is_contiguous(),
      "avx512_diagonal_recurrence: decay must be contiguous");
  TORCH_CHECK(
      x.is_contiguous(), "avx512_diagonal_recurrence: x must be contiguous");
  TORCH_CHECK(
      h.is_contiguous(), "avx512_diagonal_recurrence: h must be contiguous");
}

[[noreturn]] void raise_kernel_unavailable() {
  TORCH_CHECK(
      false,
      "AVX-512 diagonal recurrence kernel unavailable: extension was not "
      "compiled with AVX-512 kernel support. Rebuild on x86_64/amd64 with "
      "CHAOSCONTROL_CPU_SSM_X86_ACCEL=1 to enable it.");
}

}  // namespace

bool avx512_recurrence_kernel_available() {
#if defined(CHAOSCONTROL_AVX512_RECURRENCE_KERNEL_COMPILED)
  return true;
#else
  return false;
#endif
}

void avx512_diagonal_recurrence(
    const at::Tensor& decay,
    const at::Tensor& x,
    at::Tensor h) {
#if defined(CHAOSCONTROL_AVX512_RECURRENCE_KERNEL_COMPILED)
  check_recurrence_inputs(decay, x, h);
  TORCH_CHECK(
      chaoscontrol::cpu_features::runtime_has_avx512f(),
      "AVX-512 diagonal recurrence kernel unavailable at runtime: hardware "
      "or OS AVX-512 state is not enabled");

  const auto n = static_cast<std::size_t>(h.numel());
  const float* decay_ptr = decay.data_ptr<float>();
  const float* x_ptr = x.data_ptr<float>();
  float* h_ptr = h.data_ptr<float>();

  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m512 d = _mm512_loadu_ps(decay_ptr + i);
    const __m512 xi = _mm512_loadu_ps(x_ptr + i);
    const __m512 hi = _mm512_loadu_ps(h_ptr + i);
    _mm512_storeu_ps(h_ptr + i, _mm512_add_ps(_mm512_mul_ps(d, hi), xi));
  }
  for (; i < n; ++i) {
    h_ptr[i] = decay_ptr[i] * h_ptr[i] + x_ptr[i];
  }
#else
  (void)decay;
  (void)x;
  (void)h;
  raise_kernel_unavailable();
#endif
}

}  // namespace chaoscontrol::avx512
