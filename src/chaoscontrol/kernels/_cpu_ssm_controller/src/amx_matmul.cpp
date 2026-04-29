#include "amx_matmul.h"

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

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
  TileConfig cfg{};
  cfg.palette_id = 1;

  cfg.colsb[0] = static_cast<uint16_t>(n_block * sizeof(float));
  cfg.rows[0] = static_cast<uint8_t>(m_block);

  cfg.colsb[1] = static_cast<uint16_t>(k_block * sizeof(c10::BFloat16));
  cfg.rows[1] = static_cast<uint8_t>(m_block);

  cfg.colsb[2] = static_cast<uint16_t>(2 * n_block * sizeof(c10::BFloat16));
  cfg.rows[2] = static_cast<uint8_t>(k_block / 2);

  _tile_loadconfig(&cfg);
}

inline __mmask16 low_lane_mask(int64_t count) {
  if (count >= 16) {
    return static_cast<__mmask16>(0xFFFF);
  }
  return static_cast<__mmask16>((1u << count) - 1u);
}

inline __m512 exp512_approx_ps(__m512 x) {
  // Fast exp approximation adapted from the classic cephes/SSE polynomial,
  // widened to AVX-512. The NLL scorer only feeds shifted logits
  // (logit - row_max), so values live in [-inf, 0]; clamping keeps the
  // polynomial in the useful range and avoids denorm churn on dead lanes.
  const __m512 max_x = _mm512_set1_ps(0.0f);
  const __m512 min_x = _mm512_set1_ps(-80.0f);
  x = _mm512_min_ps(x, max_x);
  x = _mm512_max_ps(x, min_x);

  const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);
  const __m512 half = _mm512_set1_ps(0.5f);
  __m512 fx = _mm512_add_ps(_mm512_mul_ps(x, log2ef), half);
  fx = _mm512_roundscale_ps(
      fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

  const __m512 c1 = _mm512_set1_ps(0.693359375f);
  const __m512 c2 = _mm512_set1_ps(-2.12194440e-4f);
  x = _mm512_sub_ps(x, _mm512_mul_ps(fx, c1));
  x = _mm512_sub_ps(x, _mm512_mul_ps(fx, c2));

  const __m512 z = _mm512_mul_ps(x, x);
  __m512 y = _mm512_set1_ps(1.9875691500e-4f);
  y = _mm512_add_ps(_mm512_mul_ps(y, x), _mm512_set1_ps(1.3981999507e-3f));
  y = _mm512_add_ps(_mm512_mul_ps(y, x), _mm512_set1_ps(8.3334519073e-3f));
  y = _mm512_add_ps(_mm512_mul_ps(y, x), _mm512_set1_ps(4.1665795894e-2f));
  y = _mm512_add_ps(_mm512_mul_ps(y, x), _mm512_set1_ps(1.6666665459e-1f));
  y = _mm512_add_ps(_mm512_mul_ps(y, x), _mm512_set1_ps(5.0000001201e-1f));
  y = _mm512_mul_ps(y, z);
  y = _mm512_add_ps(y, x);
  y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));

  const __m512i exponent =
      _mm512_slli_epi32(
          _mm512_add_epi32(
              _mm512_cvttps_epi32(fx), _mm512_set1_epi32(127)),
          23);
  return _mm512_mul_ps(y, _mm512_castsi512_ps(exponent));
}

inline float reduce_max16(const float* values, int64_t count) {
  const __mmask16 mask = low_lane_mask(count);
  const __m512 v = _mm512_mask_loadu_ps(
      _mm512_set1_ps(-std::numeric_limits<float>::infinity()),
      mask,
      values);
  return _mm512_reduce_max_ps(v);
}

inline float reduce_exp_sum16(const float* values, int64_t count, float row_max) {
  const __mmask16 mask = low_lane_mask(count);
  const __m512 inactive = _mm512_set1_ps(row_max - 80.0f);
  const __m512 v = _mm512_mask_loadu_ps(inactive, mask, values);
  const __m512 shifted = _mm512_sub_ps(v, _mm512_set1_ps(row_max));
  const __m512 exp_v = _mm512_maskz_mov_ps(mask, exp512_approx_ps(shifted));
  return _mm512_reduce_add_ps(exp_v);
}
#endif

[[maybe_unused]] bool amx_scorer_thread_pinning_enabled() {
  const char* raw = std::getenv("CHAOSCONTROL_AMX_SCORER_PIN_THREADS");
  if (raw == nullptr) {
    return true;
  }
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
      std::strcmp(raw, "False") != 0 && std::strcmp(raw, "off") != 0;
}

[[maybe_unused]] std::vector<int> allowed_cpu_ids() {
  std::vector<int> out;
#if defined(__linux__)
  if (!amx_scorer_thread_pinning_enabled()) {
    return out;
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
    return out;
  }
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &mask)) {
      out.push_back(cpu);
    }
  }
#endif
  return out;
}

[[maybe_unused]] void pin_current_thread_to_cpu(int cpu_id) {
#if defined(__linux__)
  if (cpu_id < 0 || !amx_scorer_thread_pinning_enabled()) {
    return;
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu_id, &mask);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
#else
  (void)cpu_id;
#endif
}

[[noreturn, maybe_unused]] void raise_kernel_unavailable() {
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

at::Tensor amx_bf16_nll(
    const at::Tensor& hidden_states,
    const at::Tensor& targets,
    const at::Tensor& norm_weight,
    const at::Tensor& lm_head_vnni,
    double eps,
    int64_t row_chunk_size,
    int64_t lanes) {
#if defined(CHAOSCONTROL_AMX_BF16_KERNEL_COMPILED)
  TORCH_CHECK(
      chaoscontrol::cpu_features::runtime_has_amx_bf16(),
      "AMX BF16 NLL scorer unavailable at runtime: hardware or OS AMX "
      "state is not enabled");
  TORCH_CHECK(
      hidden_states.device().is_cpu(),
      "amx_bf16_nll: hidden_states must be a CPU tensor");
  TORCH_CHECK(
      targets.device().is_cpu(),
      "amx_bf16_nll: targets must be a CPU tensor");
  TORCH_CHECK(
      norm_weight.device().is_cpu(),
      "amx_bf16_nll: norm_weight must be a CPU tensor");
  TORCH_CHECK(
      lm_head_vnni.device().is_cpu(),
      "amx_bf16_nll: lm_head_vnni must be a CPU tensor");
  TORCH_CHECK(
      hidden_states.dim() == 3,
      "amx_bf16_nll: hidden_states must have shape [B,T,D]");
  TORCH_CHECK(
      targets.dim() == 2,
      "amx_bf16_nll: targets must have shape [B,T]");
  TORCH_CHECK(
      norm_weight.dim() == 1,
      "amx_bf16_nll: norm_weight must be 1D");
  TORCH_CHECK(
      lm_head_vnni.dim() == 2,
      "amx_bf16_nll: lm_head_vnni must be 2D [K/2,2V]");
  TORCH_CHECK(
      hidden_states.scalar_type() == at::kFloat,
      "amx_bf16_nll: hidden_states must be float32");
  TORCH_CHECK(
      norm_weight.scalar_type() == at::kFloat,
      "amx_bf16_nll: norm_weight must be float32");
  TORCH_CHECK(
      lm_head_vnni.scalar_type() == at::kBFloat16,
      "amx_bf16_nll: lm_head_vnni must be bfloat16");
  TORCH_CHECK(
      targets.scalar_type() == at::kLong,
      "amx_bf16_nll: targets must be int64");

  constexpr int64_t M_TILE = 16;
  constexpr int64_t N_TILE = 16;
  constexpr int64_t K_TILE = 32;

  const int64_t batch = hidden_states.size(0);
  const int64_t seq = hidden_states.size(1);
  const int64_t dim = hidden_states.size(2);
  TORCH_CHECK(
      targets.size(0) == batch && targets.size(1) == seq,
      "amx_bf16_nll: targets shape must match hidden_states[:2]");
  TORCH_CHECK(
      norm_weight.size(0) == dim,
      "amx_bf16_nll: norm_weight length must match hidden dim");
  TORCH_CHECK(dim > 0, "amx_bf16_nll: hidden dim must be positive");
  TORCH_CHECK(
      lm_head_vnni.size(1) % 2 == 0,
      "amx_bf16_nll: lm_head_vnni.shape[1] must be 2*V");

  const int64_t rows_total = batch * seq;
  const int64_t vocab = lm_head_vnni.size(1) / 2;
  const int64_t k_pad = lm_head_vnni.size(0) * 2;
  TORCH_CHECK(vocab > 0, "amx_bf16_nll: vocab must be positive");
  TORCH_CHECK(
      (k_pad % K_TILE) == 0,
      "amx_bf16_nll: packed K must be a multiple of 32");
  TORCH_CHECK(
      dim <= k_pad,
      "amx_bf16_nll: hidden dim exceeds packed K");
  TORCH_CHECK(
      vocab <= std::numeric_limits<int>::max(),
      "amx_bf16_nll: vocab too large");

  const at::Tensor h_contig = hidden_states.contiguous();
  const at::Tensor t_contig = targets.contiguous();
  const at::Tensor w_contig = norm_weight.contiguous();
  const at::Tensor b_contig = lm_head_vnni.contiguous();
  at::Tensor out = at::empty({batch, seq}, hidden_states.options().dtype(at::kFloat));

  const float* h_ptr = h_contig.data_ptr<float>();
  const int64_t* target_ptr = t_contig.data_ptr<int64_t>();
  const float* norm_ptr = w_contig.data_ptr<float>();
  const c10::BFloat16* b_ptr = b_contig.data_ptr<c10::BFloat16>();
  float* out_ptr = out.data_ptr<float>();
  if (rows_total == 0) {
    return out;
  }
  for (int64_t row = 0; row < rows_total; ++row) {
    const int64_t target = target_ptr[row];
    TORCH_CHECK(
        target >= 0 && target < vocab,
        "amx_bf16_nll: target id out of range");
  }

  const int b_vnni_stride_bytes =
      static_cast<int>(2 * vocab * sizeof(c10::BFloat16));
  const int a_stride_bytes =
      static_cast<int>(k_pad * sizeof(c10::BFloat16));
  const int scratch_stride_bytes =
      static_cast<int>(N_TILE * sizeof(float));
  const double eps_d = eps;
  const int64_t requested_lanes = std::max<int64_t>(1, lanes);
  const int64_t lane_count =
      std::min<int64_t>(requested_lanes, std::max<int64_t>(1, rows_total / M_TILE));
  const int64_t rows_per_lane =
      ((rows_total + lane_count - 1) / lane_count + M_TILE - 1) / M_TILE * M_TILE;
  const std::vector<int> lane_cpus = allowed_cpu_ids();

  auto score_range = [&](int64_t row_begin, int64_t row_end, int cpu_id) {
    pin_current_thread_to_cpu(cpu_id);
    std::vector<c10::BFloat16> a_buf(
        static_cast<size_t>(M_TILE * k_pad), c10::BFloat16(0.0f));
    alignas(64) float scratch[M_TILE * N_TILE];
    float row_max[M_TILE];
    float target_logits[M_TILE];
    float row_sum[M_TILE];

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

    for (int64_t row0 = row_begin; row0 < row_end; row0 += M_TILE) {
      const int64_t m_block = std::min<int64_t>(M_TILE, row_end - row0);
      std::fill(a_buf.begin(), a_buf.end(), c10::BFloat16(0.0f));
      for (int64_t m = 0; m < m_block; ++m) {
        const int64_t row = row0 + m;
        const float* h_row = h_ptr + row * dim;
        double mean_sq = 0.0;
        for (int64_t d = 0; d < dim; ++d) {
          const double x = static_cast<double>(h_row[d]);
          mean_sq += x * x;
        }
        const double inv_rms = 1.0 / std::sqrt(mean_sq / static_cast<double>(dim) + eps_d);
        c10::BFloat16* a_row = a_buf.data() + m * k_pad;
        for (int64_t d = 0; d < dim; ++d) {
          a_row[d] = c10::BFloat16(
              static_cast<float>(static_cast<double>(h_row[d]) * inv_rms *
                                 static_cast<double>(norm_ptr[d])));
        }
        row_max[m] = -std::numeric_limits<float>::infinity();
        target_logits[m] = -std::numeric_limits<float>::infinity();
        row_sum[m] = 0.0f;
      }

      for (int64_t n0 = 0; n0 < vocab; n0 += N_TILE) {
        const int64_t n_block = std::min<int64_t>(N_TILE, vocab - n0);
        ensure_config(m_block, n_block, K_TILE);
        _tile_zero(0);
        for (int64_t k0 = 0; k0 < k_pad; k0 += K_TILE) {
          const c10::BFloat16* a_sub = a_buf.data() + k0;
          const c10::BFloat16* b_sub = b_ptr + (k0 / 2) * (2 * vocab) + 2 * n0;
          _tile_loadd(1, a_sub, a_stride_bytes);
          _tile_loadd(2, b_sub, b_vnni_stride_bytes);
          _tile_dpbf16ps(0, 1, 2);
        }
        _tile_stored(0, scratch, scratch_stride_bytes);
        for (int64_t m = 0; m < m_block; ++m) {
          const int64_t target = target_ptr[row0 + m];
          const float* logits = scratch + m * N_TILE;
          row_max[m] = std::max(row_max[m], reduce_max16(logits, n_block));
          if (target >= n0 && target < n0 + n_block) {
            target_logits[m] = logits[target - n0];
          }
        }
      }

      for (int64_t n0 = 0; n0 < vocab; n0 += N_TILE) {
        const int64_t n_block = std::min<int64_t>(N_TILE, vocab - n0);
        ensure_config(m_block, n_block, K_TILE);
        _tile_zero(0);
        for (int64_t k0 = 0; k0 < k_pad; k0 += K_TILE) {
          const c10::BFloat16* a_sub = a_buf.data() + k0;
          const c10::BFloat16* b_sub = b_ptr + (k0 / 2) * (2 * vocab) + 2 * n0;
          _tile_loadd(1, a_sub, a_stride_bytes);
          _tile_loadd(2, b_sub, b_vnni_stride_bytes);
          _tile_dpbf16ps(0, 1, 2);
        }
        _tile_stored(0, scratch, scratch_stride_bytes);
        for (int64_t m = 0; m < m_block; ++m) {
          const float* logits = scratch + m * N_TILE;
          row_sum[m] += reduce_exp_sum16(logits, n_block, row_max[m]);
        }
      }

      for (int64_t m = 0; m < m_block; ++m) {
        out_ptr[row0 + m] = row_max[m] +
            std::log(std::max(row_sum[m], 1.0e-30f)) - target_logits[m];
      }
    }
    _tile_release();
  };

  if (lane_count <= 1 || rows_total < row_chunk_size) {
    score_range(0, rows_total, /*cpu_id=*/-1);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(lane_count));
    for (int64_t lane = 0; lane < lane_count; ++lane) {
      const int64_t begin = lane * rows_per_lane;
      if (begin >= rows_total) {
        break;
      }
      const int64_t end = std::min<int64_t>(rows_total, begin + rows_per_lane);
      const int cpu_id = lane_cpus.empty()
          ? -1
          : lane_cpus[static_cast<size_t>(lane) % lane_cpus.size()];
      workers.emplace_back(
          [&, begin, end, cpu_id]() { score_range(begin, end, cpu_id); });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  }

  return out;
#else
  (void)hidden_states;
  (void)targets;
  (void)norm_weight;
  (void)lm_head_vnni;
  (void)eps;
  (void)row_chunk_size;
  (void)lanes;
  raise_kernel_unavailable();
#endif
}

}  // namespace chaoscontrol::amx
