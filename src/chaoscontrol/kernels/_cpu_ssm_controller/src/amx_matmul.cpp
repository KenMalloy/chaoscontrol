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
  // with B packed in VNNI (K/2 x 2N bf16) internally. Larger controller
  // kernels can tile over this primitive once pod-side validation lands.
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

  // tile0 (dst): M rows of N fp32.
  cfg.colsb[0] = static_cast<uint16_t>(n * sizeof(float));
  cfg.rows[0] = static_cast<uint8_t>(m);

  // tile1 (src1 = A): M rows of K bf16.
  cfg.colsb[1] = static_cast<uint16_t>(k * sizeof(c10::BFloat16));
  cfg.rows[1] = static_cast<uint8_t>(m);

  // tile2 (src2 = B in VNNI): TDPBF16PS reads K/2 rows, each row holding
  // N pairs of bf16 (=2*N bf16 = 4*N bytes per row). Per-cell layout:
  //   tile2[r, 2*n + 0] = B_logical[2*r,     n]
  //   tile2[r, 2*n + 1] = B_logical[2*r + 1, n]
  // The VNNI interleaving lets a single dot-product instruction consume
  // adjacent K-pairs per output column; previous "transpose-contiguous"
  // packing fed the wrong byte layout — see
  // tests/test_amx_matmul_vnni_packing.py for the SDM-derived proof.
  cfg.colsb[2] = static_cast<uint16_t>(2 * n * sizeof(c10::BFloat16));
  cfg.rows[2] = static_cast<uint8_t>(k / 2);

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

// Rearrange B (K x N bf16, contiguous) into the VNNI layout TDPBF16PS
// expects (K/2 x 2N bf16). Compiled on every platform (intrinsic-free)
// so the rearrangement logic can be unit-tested on arm64 without
// booking pod time. The hardware-gated parity test in test_amx_matmul.py
// validates pack + tile-load + dot together; this gives us the pack
// half locally.
at::Tensor pack_b_vnni(const at::Tensor& b) {
  TORCH_CHECK(b.dim() == 2, "pack_b_vnni: b must be 2-dimensional");
  TORCH_CHECK(
      b.scalar_type() == at::kBFloat16,
      "pack_b_vnni: b must have dtype torch.bfloat16");
  const int64_t k = b.size(0);
  const int64_t n = b.size(1);
  TORCH_CHECK((k % 2) == 0, "pack_b_vnni: K must be even for bf16 VNNI");
  at::Tensor packed = at::empty(
      {k / 2, 2 * n}, b.options().memory_format(at::MemoryFormat::Contiguous));
  const at::Tensor b_contig = b.contiguous();
  const c10::BFloat16* src = b_contig.data_ptr<c10::BFloat16>();
  c10::BFloat16* dst = packed.data_ptr<c10::BFloat16>();
  for (int64_t r = 0; r < k / 2; ++r) {
    for (int64_t col = 0; col < n; ++col) {
      dst[r * (2 * n) + 2 * col + 0] = src[(2 * r + 0) * n + col];
      dst[r * (2 * n) + 2 * col + 1] = src[(2 * r + 1) * n + col];
    }
  }
  return packed;
}

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
  const at::Tensor b_vnni = pack_b_vnni(b);
  at::Tensor out =
      at::empty({a.size(0), b.size(1)}, a.options().dtype(at::kFloat));
  out.zero_();

  configure_single_block_tiles(a.size(0), b.size(1), a.size(1));
  _tile_zero(0);
  _tile_loadd(1, a_contig.data_ptr<c10::BFloat16>(),
              static_cast<int>(a.size(1) * sizeof(c10::BFloat16)));
  // tile2 stride matches its row width: 2*N bf16 = 4*N bytes per row.
  _tile_loadd(2, b_vnni.data_ptr<c10::BFloat16>(),
              static_cast<int>(2 * b.size(1) * sizeof(c10::BFloat16)));
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
