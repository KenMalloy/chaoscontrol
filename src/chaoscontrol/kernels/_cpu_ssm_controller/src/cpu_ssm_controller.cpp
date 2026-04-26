#include <torch/extension.h>

#include <tuple>

namespace {

void check_vec(const at::Tensor& t, const char* name, int64_t n) {
  TORCH_CHECK(!t.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(t.dim() == 1 && t.size(0) == n,
              name, " must have shape [", n, "]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_mat(
    const at::Tensor& t,
    const char* name,
    int64_t rows,
    int64_t cols) {
  TORCH_CHECK(!t.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == cols,
              name, " must have shape [", rows, ", ", cols, "]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> forward_step(
    const at::Tensor& features,
    const at::Tensor& global_state,
    const at::Tensor& slot_state,
    const at::Tensor& w_global_in,
    const at::Tensor& w_slot_in,
    const at::Tensor& decay_global,
    const at::Tensor& decay_slot,
    const at::Tensor& w_global_out,
    const at::Tensor& w_slot_out,
    double bias) {
  TORCH_CHECK(features.dim() == 1, "features must be 1-D");
  TORCH_CHECK(global_state.dim() == 1, "global_state must be 1-D");
  TORCH_CHECK(slot_state.dim() == 1, "slot_state must be 1-D");
  const int64_t feature_dim = features.size(0);
  const int64_t global_dim = global_state.size(0);
  const int64_t slot_dim = slot_state.size(0);

  check_vec(features, "features", feature_dim);
  check_vec(global_state, "global_state", global_dim);
  check_vec(slot_state, "slot_state", slot_dim);
  check_mat(w_global_in, "w_global_in", global_dim, feature_dim);
  check_mat(w_slot_in, "w_slot_in", slot_dim, feature_dim);
  check_vec(decay_global, "decay_global", global_dim);
  check_vec(decay_slot, "decay_slot", slot_dim);
  check_vec(w_global_out, "w_global_out", global_dim);
  check_vec(w_slot_out, "w_slot_out", slot_dim);

  auto out_global = at::empty_like(global_state);
  auto out_slot = at::empty_like(slot_state);

  const float* f = features.data_ptr<float>();
  const float* g = global_state.data_ptr<float>();
  const float* s = slot_state.data_ptr<float>();
  const float* wg = w_global_in.data_ptr<float>();
  const float* ws = w_slot_in.data_ptr<float>();
  const float* dg = decay_global.data_ptr<float>();
  const float* ds = decay_slot.data_ptr<float>();
  float* og = out_global.data_ptr<float>();
  float* os = out_slot.data_ptr<float>();

  for (int64_t i = 0; i < global_dim; ++i) {
    float acc = 0.0f;
    const float* row = wg + i * feature_dim;
    for (int64_t j = 0; j < feature_dim; ++j) {
      acc += row[j] * f[j];
    }
    og[i] = dg[i] * g[i] + acc;
  }

  for (int64_t i = 0; i < slot_dim; ++i) {
    float acc = 0.0f;
    const float* row = ws + i * feature_dim;
    for (int64_t j = 0; j < feature_dim; ++j) {
      acc += row[j] * f[j];
    }
    os[i] = ds[i] * s[i] + acc;
  }

  const float* wgo = w_global_out.data_ptr<float>();
  const float* wso = w_slot_out.data_ptr<float>();
  float logit = static_cast<float>(bias);
  for (int64_t i = 0; i < global_dim; ++i) {
    logit += og[i] * wgo[i];
  }
  for (int64_t i = 0; i < slot_dim; ++i) {
    logit += os[i] * wso[i];
  }

  auto logit_tensor = at::empty({}, features.options());
  *logit_tensor.data_ptr<float>() = logit;
  return std::make_tuple(out_global, out_slot, logit_tensor);
}

bool has_amx_bf16() {
#if defined(__AMX_BF16__)
  return true;
#else
  return false;
#endif
}

const char* backend_name() {
#if defined(__AMX_BF16__)
  return "amx_bf16_compile_available";
#else
  return "reference";
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_step", &forward_step, "CPU SSM controller reference step");
  m.def("has_amx_bf16", &has_amx_bf16, "Whether built with AMX BF16 support");
  m.def("backend_name", &backend_name, "Compiled backend name");
}
