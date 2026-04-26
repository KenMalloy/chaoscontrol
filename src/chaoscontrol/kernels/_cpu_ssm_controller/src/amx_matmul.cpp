#include "amx_matmul.h"

#include <torch/extension.h>

#include <cstdint>
#include <cstring>

#include "cpu_features.h"

#if defined(__x86_64__) && defined(__AMX_BF16__) && \
    defined(CHAOSCONTROL_CPU_SSM_AMX_BF16_KERNEL)
#define CHAOSCONTROL_AMX_BF16_KERNEL_COMPILED 1
#include <immintrin.h>
#endif

namespace chaoscontrol::amx {
namespace {

[[maybe_unused]] void check_bf16_matmul_inputs(
    const at::Tensor& a,
    const at::Tensor& b) {
  TORCH_CHECK(a.device().is_cpu(), "amx_bf16_matmul: a must be a CPU tensor");
  TORCH_CHECK(b.device().is_cpu(), "amx_bf16_matmul: b must be a CPU tensor");
  TORCH_CHECK(a.dim() == 2, "amx_bf16_matmul: a must be 2-dimensional");
  TORCH_CHECK(b.dim() == 2, "amx_bf16_matmul: b must be 2-dimensional");
  TORCH_CHECK(
      a.scalar_type() == at::kBFloat16,
      "amx_bf16_matmul: a must have dtype torch.bfloat16");
  TORCH_CHECK(
      b.scalar_type() == at::kBFloat16,
      "amx_bf16_matmul: b must have dtype torch.bfloat16");

  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t b_k = b.size(0);
  const int64_t n = b.size(1);
  TORCH_CHECK(
      k == b_k,
      "amx_bf16_matmul: shape mismatch, a.shape[1] (", k,
      ") must equal b.shape[0] (", b_k, ")");

  // E2 exposes one AMX tile block: C[M,N] += A[M,K] x B[K,N],
  // with B packed transposed internally. Larger controller kernels can tile
  // over this primitive once pod-side validation lands.
  TORCH_CHECK(m > 0 && m <= 16,
              "amx_bf16_matmul: M dimension must be in [1, 16], got ", m);
  TORCH_CHECK(n > 0 && n <= 16,
              "amx_bf16_matmul: N dimension must be in [1, 16], got ", n);
  TORCH_CHECK(k > 0 && k <= 32,
              "amx_bf16_matmul: K dimension must be in [1, 32], got ", k);
  TORCH_CHECK((k % 2) == 0,
              "amx_bf16_matmul: K dimension must be even for BF16 dot pairs");
}

#if defined(CHAOSCONTROL_AMX_BF16_KERNEL_COMPILED)
struct alignas(64) TileConfig {
  uint8_t palette_id = 0;
  uint8_t start_row = 0;
  uint8_t reserved[14] = {};
  uint16_t colsb[16] = {};
  uint8_t rows[16] = {};
};

void configure_single_block_tiles(int64_t m, int64_t n, int64_t k) {
  TileConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.palette_id = 1;

  cfg.colsb[0] = static_cast<uint16_t>(n * sizeof(float));
  cfg.rows[0] = static_cast<uint8_t>(m);

  cfg.colsb[1] = static_cast<uint16_t>(k * sizeof(c10::BFloat16));
  cfg.rows[1] = static_cast<uint8_t>(m);

  // TDPBF16PS consumes the right-hand matrix in transposed BF16 tile layout:
  // one tile row per output column, K BF16 values per row.
  cfg.colsb[2] = static_cast<uint16_t>(k * sizeof(c10::BFloat16));
  cfg.rows[2] = static_cast<uint8_t>(n);

  _tile_loadconfig(&cfg);
}
#endif

[[noreturn]] void raise_kernel_unavailable() {
  TORCH_CHECK(
      false,
      "AMX BF16 matmul kernel unavailable: extension was not compiled with "
      "AMX BF16 kernel support. Rebuild on x86_64/amd64 with "
      "CHAOSCONTROL_CPU_SSM_X86_ACCEL=1 to enable it.");
}

}  // namespace

bool amx_bf16_kernel_available() {
#if defined(CHAOSCONTROL_AMX_BF16_KERNEL_COMPILED)
  return true;
#else
  return false;
#endif
}

at::Tensor amx_bf16_matmul(const at::Tensor& a, const at::Tensor& b) {
#if defined(CHAOSCONTROL_AMX_BF16_KERNEL_COMPILED)
  check_bf16_matmul_inputs(a, b);
  TORCH_CHECK(
      chaoscontrol::cpu_features::runtime_has_amx_bf16(),
      "AMX BF16 matmul kernel unavailable at runtime: hardware or OS AMX "
      "state is not enabled");

  const at::Tensor a_contig = a.contiguous();
  const at::Tensor b_transposed = b.transpose(0, 1).contiguous();
  at::Tensor out =
      at::empty({a.size(0), b.size(1)}, a.options().dtype(at::kFloat));
  out.zero_();

  configure_single_block_tiles(a.size(0), b.size(1), a.size(1));
  _tile_zero(0);
  _tile_loadd(1, a_contig.data_ptr<c10::BFloat16>(),
              static_cast<int>(a.size(1) * sizeof(c10::BFloat16)));
  _tile_loadd(2, b_transposed.data_ptr<c10::BFloat16>(),
              static_cast<int>(a.size(1) * sizeof(c10::BFloat16)));
  _tile_dpbf16ps(0, 1, 2);
  _tile_stored(0, out.data_ptr<float>(),
               static_cast<int>(b.size(1) * sizeof(float)));
  _tile_release();

  return out;
#else
  (void)a;
  (void)b;
  raise_kernel_unavailable();
#endif
}

}  // namespace chaoscontrol::amx
