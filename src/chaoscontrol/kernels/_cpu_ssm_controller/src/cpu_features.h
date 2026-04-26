#pragma once

namespace chaoscontrol::cpu_features {

struct CpuFeatures {
  bool is_x86 = false;
  bool has_avx512f = false;
  bool has_amx_tile = false;
  bool has_amx_bf16 = false;
  bool os_avx512_enabled = false;
  bool os_amx_enabled = false;
};

CpuFeatures detect_cpu_features();

bool runtime_has_avx512f();
bool runtime_has_amx_bf16();

}  // namespace chaoscontrol::cpu_features
