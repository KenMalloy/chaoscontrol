#include "cpu_features.h"

#include <cstdint>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#define CHAOSCONTROL_CPU_X86 1
#endif

#if defined(CHAOSCONTROL_CPU_X86) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

namespace chaoscontrol::cpu_features {
namespace {

#if defined(CHAOSCONTROL_CPU_X86) && (defined(__GNUC__) || defined(__clang__))
constexpr uint32_t kLeaf1 = 1;
constexpr uint32_t kLeaf7 = 7;
constexpr uint32_t kOsxsaveBit = 1u << 27;
constexpr uint32_t kAvx512fBit = 1u << 16;
constexpr uint32_t kAmxBf16Bit = 1u << 22;
constexpr uint32_t kAmxTileBit = 1u << 24;

constexpr uint64_t kXcr0Sse = 1ull << 1;
constexpr uint64_t kXcr0YmmHi = 1ull << 2;
constexpr uint64_t kXcr0OpMask = 1ull << 5;
constexpr uint64_t kXcr0ZmmHi256 = 1ull << 6;
constexpr uint64_t kXcr0Hi16Zmm = 1ull << 7;
constexpr uint64_t kXcr0TileCfg = 1ull << 17;
constexpr uint64_t kXcr0TileData = 1ull << 18;
constexpr uint64_t kAvx512Xcr0Mask =
    kXcr0Sse | kXcr0YmmHi | kXcr0OpMask | kXcr0ZmmHi256 | kXcr0Hi16Zmm;
constexpr uint64_t kAmxXcr0Mask = kXcr0TileCfg | kXcr0TileData;

struct CpuidRegs {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
};

bool cpuid(uint32_t leaf, uint32_t subleaf, CpuidRegs& out) {
  const uint32_t max_leaf = __get_cpuid_max(0, nullptr);
  if (leaf > max_leaf) {
    return false;
  }
  __cpuid_count(leaf, subleaf, out.eax, out.ebx, out.ecx, out.edx);
  return true;
}

uint64_t read_xcr0() {
  uint32_t eax = 0;
  uint32_t edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<uint64_t>(edx) << 32) | eax;
}

bool xcr0_has(uint64_t xcr0, uint64_t mask) {
  return (xcr0 & mask) == mask;
}
#endif

}  // namespace

CpuFeatures detect_cpu_features() {
#if defined(CHAOSCONTROL_CPU_X86) && (defined(__GNUC__) || defined(__clang__))
  CpuFeatures features;
  features.is_x86 = true;

  CpuidRegs leaf1;
  const bool has_leaf1 = cpuid(kLeaf1, 0, leaf1);
  const bool osxsave = has_leaf1 && ((leaf1.ecx & kOsxsaveBit) != 0);
  const uint64_t xcr0 = osxsave ? read_xcr0() : 0;
  features.os_avx512_enabled = osxsave && xcr0_has(xcr0, kAvx512Xcr0Mask);
  features.os_amx_enabled = osxsave && xcr0_has(xcr0, kAmxXcr0Mask);

  CpuidRegs leaf7;
  if (cpuid(kLeaf7, 0, leaf7)) {
    features.has_avx512f = (leaf7.ebx & kAvx512fBit) != 0;
    features.has_amx_tile = (leaf7.edx & kAmxTileBit) != 0;
    features.has_amx_bf16 = (leaf7.edx & kAmxBf16Bit) != 0;
  }
  return features;
#else
  return CpuFeatures{};
#endif
}

bool runtime_has_avx512f() {
  const CpuFeatures features = detect_cpu_features();
  return features.has_avx512f && features.os_avx512_enabled;
}

bool runtime_has_amx_bf16() {
  const CpuFeatures features = detect_cpu_features();
  return features.has_amx_tile && features.has_amx_bf16 &&
      features.os_amx_enabled;
}

}  // namespace chaoscontrol::cpu_features
