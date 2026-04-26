#include "amx_matmul.h"

#include <torch/extension.h>

#include <algorithm>
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

  // E2 tiled kernel: C[M,N] += A[M,K] x B[K,N], with B VNNI-packed once
  // into (K/2 x 2N bf16) and the (M, N) output covered by 16x16 dst
  // tiles (with edge tails). K is consumed in 32-bf16 chunks per
  // _tile_dpbf16ps. Arbitrary (M, N, K) supported; only constraint
  // beyond positivity is that K is even (BF16 dot-pair requirement).
  TORCH_CHECK(m > 0,
              "amx_bf16_matmul: M dimension must be positive, got ", m);
  TORCH_CHECK(n > 0,
              "amx_bf16_matmul: N dimension must be positive, got ", n);
  TORCH_CHECK(k > 0,
              "amx_bf16_matmul: K dimension must be positive, got ", k);
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

// Configure tiles 0/1/2 for one output sub-tile of size
// (m_block, n_block) with K-dimension chunk k_block.
//
// tile0 (dst): m_block rows of n_block fp32.
// tile1 (src1 = A): m_block rows of k_block bf16.
// tile2 (src2 = B in VNNI): TDPBF16PS reads (k_block/2) rows, each row
//   holding n_block pairs of bf16 (= 2*n_block bf16 = 4*n_block bytes).
//   Per-cell layout:
//     tile2[r, 2*n + 0] = B_logical[k0 + 2*r,     n0 + n]
//     tile2[r, 2*n + 1] = B_logical[k0 + 2*r + 1, n0 + n]
//   The VNNI interleaving lets one dot-product instruction consume
//   adjacent K-pairs per output column; see
//   tests/test_amx_matmul_vnni_packing.py for the SDM-derived proof.
void configure_block_tiles(int64_t m_block, int64_t n_block, int64_t k_block) {
  TileConfig cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.palette_id = 1;

  cfg.colsb[0] = static_cast<uint16_t>(n_block * sizeof(float));
  cfg.rows[0] = static_cast<uint8_t>(m_block);

  cfg.colsb[1] = static_cast<uint16_t>(k_block * sizeof(c10::BFloat16));
  cfg.rows[1] = static_cast<uint8_t>(m_block);

  cfg.colsb[2] = static_cast<uint16_t>(2 * n_block * sizeof(c10::BFloat16));
  cfg.rows[2] = static_cast<uint8_t>(k_block / 2);

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

  constexpr int64_t M_TILE = 16;
  constexpr int64_t N_TILE = 16;
  constexpr int64_t K_TILE = 32;

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(1);

  // K-tile policy: zero-pad K up to a multiple of K_TILE so every
  // K-iter inside a single output tile uses the same tile config. The
  // E2 brief originally specified "reconfigure for the K-tail mid
  // K-loop", but per the Intel AMX Programming Reference §3.1
  // (TILELOADCONFIG / LDTILECFG zeros all tile data registers),
  // mid-loop reconfig would clobber the dst accumulator and produce
  // wrong results. Padded entries contribute x*0=0 to the dot product
  // so the fp32 result matches the simulator's tail-block math
  // exactly (proof: tests/test_amx_matmul_vnni_packing.py
  // ::test_simulate_tiled_amx_matmul_matches_at_matmul). The padded
  // buffers are also small — for the worst-case shape (13, 17, 30) we
  // allocate (13, 32) bf16 + (16, 34) bf16 ≈ 1.9KB total.
  const int64_t K_pad = (K <= K_TILE) ? K : ((K + K_TILE - 1) / K_TILE) * K_TILE;
  const bool needs_k_pad = (K_pad != K);

  const at::Tensor a_contig = a.contiguous();

  // Build padded A (and the source for B-VNNI packing) only when K
  // doesn't already land on a K-tile boundary.
  at::Tensor a_for_kernel;
  at::Tensor b_for_pack;
  if (needs_k_pad) {
    a_for_kernel = at::zeros(
        {M, K_pad}, a.options().memory_format(at::MemoryFormat::Contiguous));
    a_for_kernel.narrow(/*dim=*/1, /*start=*/0, /*length=*/K).copy_(a_contig);
    b_for_pack = at::zeros(
        {K_pad, N}, b.options().memory_format(at::MemoryFormat::Contiguous));
    b_for_pack.narrow(/*dim=*/0, /*start=*/0, /*length=*/K).copy_(b);
  } else {
    a_for_kernel = a_contig;
    b_for_pack = b;
  }

  const at::Tensor b_vnni = pack_b_vnni(b_for_pack);
  at::Tensor out =
      at::empty({M, N}, a.options().dtype(at::kFloat));

  const c10::BFloat16* a_ptr = a_for_kernel.data_ptr<c10::BFloat16>();
  const c10::BFloat16* b_vnni_ptr = b_vnni.data_ptr<c10::BFloat16>();
  float* out_ptr = out.data_ptr<float>();

  // Strides in bytes for tile load/store. A is (M, K_pad) bf16
  // row-major; B-VNNI is (K_pad/2, 2N) bf16 row-major; out is (M, N)
  // fp32 row-major.
  const int a_stride_bytes = static_cast<int>(K_pad * sizeof(c10::BFloat16));
  const int b_vnni_stride_bytes =
      static_cast<int>(2 * N * sizeof(c10::BFloat16));
  const int out_stride_bytes = static_cast<int>(N * sizeof(float));

  // Per-K-iter tile shape. After padding, every K iter except possibly
  // the K<=K_TILE case has k_block == K_TILE; in that small case
  // k_block == K_pad == K (already even by input check).
  const int64_t k_block = std::min<int64_t>(K_TILE, K_pad);

  // Track the current tile config so we only reload it when the
  // (m_block, n_block) output shape changes. _tile_loadconfig is a
  // few-hundred-cycle instruction; on uniform-shape M/N grids this
  // collapses to a single configure for the main loop plus at most
  // one reconfigure each for the M-tail and N-tail.
  int64_t cur_m = -1, cur_n = -1, cur_k = -1;
  auto ensure_config =
      [&](int64_t m_block, int64_t n_block, int64_t kb) {
        if (m_block != cur_m || n_block != cur_n || kb != cur_k) {
          configure_block_tiles(m_block, n_block, kb);
          cur_m = m_block;
          cur_n = n_block;
          cur_k = kb;
        }
      };

  for (int64_t m0 = 0; m0 < M; m0 += M_TILE) {
    const int64_t m_block = std::min<int64_t>(M_TILE, M - m0);
    for (int64_t n0 = 0; n0 < N; n0 += N_TILE) {
      const int64_t n_block = std::min<int64_t>(N_TILE, N - n0);

      ensure_config(m_block, n_block, k_block);
      _tile_zero(0);

      for (int64_t k0 = 0; k0 < K_pad; k0 += K_TILE) {
        // A sub-tile pointer: row m0, col k0; stride = full padded A row.
        const c10::BFloat16* a_sub = a_ptr + m0 * K_pad + k0;
        // B-VNNI sub-tile pointer: row (k0/2), col (2*n0); stride =
        // full B-VNNI row width = 2N bf16.
        const c10::BFloat16* b_sub =
            b_vnni_ptr + (k0 / 2) * (2 * N) + 2 * n0;

        _tile_loadd(1, a_sub, a_stride_bytes);
        _tile_loadd(2, b_sub, b_vnni_stride_bytes);
        _tile_dpbf16ps(0, 1, 2);
      }

      // Store dst sub-tile back into out.
      float* out_sub = out_ptr + m0 * N + n0;
      _tile_stored(0, out_sub, out_stride_bytes);
    }
  }

  _tile_release();

  return out;
#else
  (void)a;
  (void)b;
  raise_kernel_unavailable();
#endif
}

}  // namespace chaoscontrol::amx
