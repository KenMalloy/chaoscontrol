#include <torch/extension.h>

// pybind11/stl.h pulls in the std::optional<T> caster used by
// SpscRing::pop()'s binding (returns Python None when empty). Without
// this include the binding compiles but pop() returns an opaque
// std::optional object instead of None / int.
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <tuple>
#include <vector>

#include "action_history.h"
#include "controller_main.h"
#include "posix_shm.h"
#include "shm_ring.h"
#include "spsc_ring.h"
#include "wire_events.h"

namespace {

constexpr uint32_t kCswgVersion = 1;
constexpr uint32_t kCswgDtypeFp16 = 1;
constexpr std::size_t kCswgHeaderBytes = 28;
constexpr double kInvSqrt2 = 0.70710678118654752440;

struct CswgHeader {
  uint32_t version;
  uint32_t n_layers;
  uint32_t d_global;
  uint32_t d_slot;
  uint32_t feature_dim;
  uint32_t dtype;
};

uint32_t read_le_u32(const unsigned char* p) {
  return static_cast<uint32_t>(p[0]) |
      (static_cast<uint32_t>(p[1]) << 8) |
      (static_cast<uint32_t>(p[2]) << 16) |
      (static_cast<uint32_t>(p[3]) << 24);
}

CswgHeader parse_cswg_header(const std::vector<unsigned char>& bytes) {
  TORCH_CHECK(bytes.size() == kCswgHeaderBytes, "CSWG internal header size bug");
  TORCH_CHECK(std::memcmp(bytes.data(), "CSWG", 4) == 0, "CSWG bad magic");
  return CswgHeader{
      read_le_u32(bytes.data() + 4),
      read_le_u32(bytes.data() + 8),
      read_le_u32(bytes.data() + 12),
      read_le_u32(bytes.data() + 16),
      read_le_u32(bytes.data() + 20),
      read_le_u32(bytes.data() + 24),
  };
}

uint64_t checked_mul_u64(uint64_t a, uint64_t b, const char* name) {
  TORCH_CHECK(
      a == 0 || b <= std::numeric_limits<uint64_t>::max() / a,
      "CSWG element count overflow while sizing ", name);
  return a * b;
}

uint64_t checked_add_u64(uint64_t a, uint64_t b, const char* name) {
  TORCH_CHECK(
      b <= std::numeric_limits<uint64_t>::max() - a,
      "CSWG element count overflow while adding ", name);
  return a + b;
}

float half_bits_to_float(uint16_t bits) {
  c10::Half value;
  std::memcpy(&value, &bits, sizeof(bits));
  return static_cast<float>(value);
}

at::Tensor read_fp16_tensor(
    const std::vector<uint16_t>& payload,
    std::size_t& offset,
    const std::vector<int64_t>& shape,
    const char* name) {
  int64_t count_i64 = 1;
  for (int64_t dim : shape) {
    TORCH_CHECK(dim >= 0, name, " has negative dimension");
    TORCH_CHECK(
        count_i64 <= std::numeric_limits<int64_t>::max() / std::max<int64_t>(dim, 1),
        name, " element count overflows int64");
    count_i64 *= dim;
  }
  const std::size_t count = static_cast<std::size_t>(count_i64);
  TORCH_CHECK(
      offset <= payload.size() && count <= payload.size() - offset,
      "CSWG payload ended while reading tensor ", name);
  auto out = at::empty(
      at::IntArrayRef(shape),
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  float* dst = out.data_ptr<float>();
  for (std::size_t i = 0; i < count; ++i) {
    dst[i] = half_bits_to_float(payload[offset + i]);
  }
  offset += count;
  return out;
}

at::Tensor dict_tensor(const pybind11::dict& weights, const char* key) {
  pybind11::str py_key(key);
  TORCH_CHECK(weights.contains(py_key), "missing CSWG tensor ", key);
  at::Tensor value = pybind11::cast<at::Tensor>(weights[py_key]);
  TORCH_CHECK(!value.is_cuda(), key, " must be a CPU tensor");
  TORCH_CHECK(value.scalar_type() == at::kFloat, key, " must be float32");
  return value.contiguous();
}

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

void check_batch(const at::Tensor& t, const char* name, int64_t cols) {
  TORCH_CHECK(!t.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(t.dim() == 2 && t.size(1) == cols,
              name, " must have shape [B, ", cols, "]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_tensor3(
    const at::Tensor& t,
    const char* name,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2) {
  TORCH_CHECK(!t.is_cuda(), name, " must be a CPU tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(
      t.dim() == 3 && t.size(0) == dim0 && t.size(1) == dim1 &&
          t.size(2) == dim2,
      name, " must have shape [", dim0, ", ", dim1, ", ", dim2, "]");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor exact_gelu(const at::Tensor& x) {
  return 0.5 * x * (1.0 + at::erf(x * kInvSqrt2));
}

}  // namespace

pybind11::dict load_weights_from_path(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  TORCH_CHECK(in.good(), "failed to open CSWG weight file: ", path);

  std::vector<unsigned char> header_bytes(kCswgHeaderBytes);
  in.read(
      reinterpret_cast<char*>(header_bytes.data()),
      static_cast<std::streamsize>(header_bytes.size()));
  TORCH_CHECK(in.gcount() == static_cast<std::streamsize>(header_bytes.size()),
              "CSWG file is too short to contain a header: ", path);
  const CswgHeader header = parse_cswg_header(header_bytes);
  TORCH_CHECK(header.version == kCswgVersion,
              "CSWG unsupported version ", header.version);
  TORCH_CHECK(header.dtype == kCswgDtypeFp16,
              "CSWG unsupported dtype enum ", header.dtype);
  TORCH_CHECK(header.n_layers > 0, "CSWG n_layers must be positive");
  TORCH_CHECK(header.d_global > 0, "CSWG d_global must be positive");
  TORCH_CHECK(header.d_slot > 0, "CSWG d_slot must be positive");
  TORCH_CHECK(header.feature_dim > 0, "CSWG feature_dim must be positive");

  in.seekg(0, std::ios::end);
  const std::streamoff file_size = in.tellg();
  TORCH_CHECK(file_size >= static_cast<std::streamoff>(kCswgHeaderBytes),
              "CSWG file size underflow: ", path);
  const std::streamoff payload_bytes =
      file_size - static_cast<std::streamoff>(kCswgHeaderBytes);
  TORCH_CHECK(payload_bytes % 2 == 0,
              "CSWG fp16 payload byte count must be even");
  in.seekg(static_cast<std::streamoff>(kCswgHeaderBytes), std::ios::beg);

  std::vector<uint16_t> payload(static_cast<std::size_t>(payload_bytes / 2));
  in.read(
      reinterpret_cast<char*>(payload.data()),
      static_cast<std::streamsize>(payload_bytes));
  TORCH_CHECK(in.gcount() == payload_bytes,
              "failed to read full CSWG payload from ", path);

  const uint64_t n_layers = header.n_layers;
  const uint64_t d_global = header.d_global;
  const uint64_t d_slot = header.d_slot;
  const uint64_t feature_dim = header.feature_dim;
  const uint64_t total = payload.size();
  uint64_t expected = 0;
  const uint64_t d_global_sq =
      checked_mul_u64(d_global, d_global, "D_global squared");
  auto add = [&](uint64_t term, const char* name) {
    expected = checked_add_u64(expected, term, name);
  };
  add(checked_mul_u64(d_global, feature_dim, "trunk.in_proj.weight"),
      "trunk.in_proj.weight");
  add(d_global, "trunk.in_proj.bias");
  add(checked_mul_u64(n_layers, d_global, "trunk.decay"), "trunk.decay");
  add(checked_mul_u64(n_layers, d_global_sq, "trunk.w_in"), "trunk.w_in");
  add(checked_mul_u64(n_layers, d_global_sq, "trunk.w_out"), "trunk.w_out");
  add(checked_mul_u64(n_layers, d_global, "trunk.bias"), "trunk.bias");
  add(checked_mul_u64(d_slot, d_global, "policy_head.weight"),
      "policy_head.weight");
  add(d_slot, "policy_head.bias");
  add(d_global, "value_head.weight");
  add(1, "value_head.bias");
  TORCH_CHECK(
      total == expected,
      "CSWG payload element count mismatch: header expects ", expected,
      " fp16 values but file has ", total);
  TORCH_CHECK(
      feature_dim <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "CSWG feature_dim overflows int64");

  const int64_t l = static_cast<int64_t>(n_layers);
  const int64_t dg = static_cast<int64_t>(d_global);
  const int64_t ds = static_cast<int64_t>(d_slot);
  const int64_t f = static_cast<int64_t>(feature_dim);
  std::size_t offset = 0;
  pybind11::dict out;
  out["trunk.in_proj.weight"] = read_fp16_tensor(
      payload, offset, {dg, f}, "trunk.in_proj.weight");
  out["trunk.in_proj.bias"] = read_fp16_tensor(
      payload, offset, {dg}, "trunk.in_proj.bias");
  out["trunk.decay"] = read_fp16_tensor(
      payload, offset, {l, dg}, "trunk.decay");
  out["trunk.w_in"] = read_fp16_tensor(
      payload, offset, {l, dg, dg}, "trunk.w_in");
  out["trunk.w_out"] = read_fp16_tensor(
      payload, offset, {l, dg, dg}, "trunk.w_out");
  out["trunk.bias"] = read_fp16_tensor(
      payload, offset, {l, dg}, "trunk.bias");
  out["policy_head.weight"] = read_fp16_tensor(
      payload, offset, {ds, dg}, "policy_head.weight");
  out["policy_head.bias"] = read_fp16_tensor(
      payload, offset, {ds}, "policy_head.bias");
  out["value_head.weight"] = read_fp16_tensor(
      payload, offset, {1, dg}, "value_head.weight");
  out["value_head.bias"] = read_fp16_tensor(
      payload, offset, {1}, "value_head.bias");
  TORCH_CHECK(offset == payload.size(),
              "CSWG loader left unread payload elements");
  return out;
}

std::tuple<at::Tensor, at::Tensor> forward_pretrain_model(
    const at::Tensor& features,
    const pybind11::dict& weights) {
  at::Tensor in_proj_weight = dict_tensor(weights, "trunk.in_proj.weight");
  at::Tensor in_proj_bias = dict_tensor(weights, "trunk.in_proj.bias");
  at::Tensor decay = dict_tensor(weights, "trunk.decay");
  at::Tensor w_in = dict_tensor(weights, "trunk.w_in");
  at::Tensor w_out = dict_tensor(weights, "trunk.w_out");
  at::Tensor trunk_bias = dict_tensor(weights, "trunk.bias");
  at::Tensor policy_weight = dict_tensor(weights, "policy_head.weight");
  at::Tensor policy_bias = dict_tensor(weights, "policy_head.bias");
  at::Tensor value_weight = dict_tensor(weights, "value_head.weight");
  at::Tensor value_bias = dict_tensor(weights, "value_head.bias");

  TORCH_CHECK(in_proj_weight.dim() == 2,
              "trunk.in_proj.weight must have shape [D_global, F]");
  const int64_t d_global = in_proj_weight.size(0);
  const int64_t feature_dim = in_proj_weight.size(1);
  check_vec(in_proj_bias, "trunk.in_proj.bias", d_global);
  TORCH_CHECK(decay.dim() == 2 && decay.size(1) == d_global,
              "trunk.decay must have shape [n_layers, D_global]");
  const int64_t n_layers = decay.size(0);
  check_tensor3(w_in, "trunk.w_in", n_layers, d_global, d_global);
  check_tensor3(w_out, "trunk.w_out", n_layers, d_global, d_global);
  check_mat(trunk_bias, "trunk.bias", n_layers, d_global);
  TORCH_CHECK(policy_weight.dim() == 2 && policy_weight.size(1) == d_global,
              "policy_head.weight must have shape [D_slot, D_global]");
  check_vec(policy_bias, "policy_head.bias", policy_weight.size(0));
  check_mat(value_weight, "value_head.weight", 1, d_global);
  check_vec(value_bias, "value_head.bias", 1);

  at::Tensor x = features.contiguous();
  check_batch(x, "features", feature_dim);

  at::Tensor h = at::matmul(x, in_proj_weight.t()) + in_proj_bias;
  for (int64_t layer = 0; layer < n_layers; ++layer) {
    h = h * decay.select(0, layer) +
        at::matmul(h, w_in.select(0, layer)) +
        trunk_bias.select(0, layer);
    h = exact_gelu(h);
    h = at::matmul(h, w_out.select(0, layer));
  }
  at::Tensor logits = at::matmul(h, policy_weight.t()) + policy_bias;
  at::Tensor value =
      (at::matmul(h, value_weight.t()) + value_bias).squeeze(-1);
  return std::make_tuple(logits.contiguous(), value.contiguous());
}

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

// Wire-event introspection (Phase A1). Reports the byte sizes pinned by
// `static_assert` in wire_events.h, the natural alignment of the largest
// member (`alignof(uint64_t)` — note that `alignof(struct)` is 1 under
// `#pragma pack(push, 1)` and is therefore not what callers want), and
// the compile-time constants that drive the array fields.
pybind11::dict wire_event_sizes() {
  pybind11::dict d;
  d["WriteEvent"] = static_cast<int64_t>(sizeof(WriteEvent));
  d["QueryEvent"] = static_cast<int64_t>(sizeof(QueryEvent));
  d["ReplayOutcome"] = static_cast<int64_t>(sizeof(ReplayOutcome));
  return d;
}

pybind11::int_ wire_event_min_slot_alignment() {
  // Largest member alignment across all three wire events, not
  // `alignof(struct)` (= 1 under pack(1)). All three structs share the
  // same requirement (`alignof(uint64_t)` = 8 on every targeted platform)
  // because they all carry u64 fields and nothing wider; a single int
  // matches the structural reality better than a dict-of-three identical
  // values. ShmRing slot strides (Phase A4) must satisfy this value so a
  // u64 load from any field lands on a natural boundary.
  return static_cast<int64_t>(alignof(uint64_t));
}

pybind11::dict wire_event_constants() {
  pybind11::dict d;
  d["KEY_REP_DIM_DEFAULT"] = static_cast<int64_t>(KEY_REP_DIM_DEFAULT);
  d["SPAN_LENGTH_DEFAULT"] = static_cast<int64_t>(SPAN_LENGTH_DEFAULT);
  return d;
}

// Concrete SpscRing instantiation used only by the Phase A2 Python test
// (tests/test_spsc_ring.py). The 1024-slot u64 ring is a stand-in — the
// real wire-event rings (WriteEvent / QueryEvent / ReplayOutcome over
// shared memory) are instantiated in Phase A4's ShmRing wrapper. u64
// payload is wide enough to surface byte-tearing if memory ordering is
// wrong but small enough that a 1024-slot ring is ~8KB (two
// cacheline-padded indices + 8KB slot array) and stays well under any
// pybind11 / heap pressure.
using TestRing = SpscRing<uint64_t, 1024>;

// Concrete ShmRing instantiation used only by the Phase A4 Python test
// (tests/test_shm_ring.py). Same TestRing payload/capacity as A2 but
// with the SPSC state placed in POSIX shared memory so a fork()'d
// child can attach by name. Real wire-event ring instantiations
// (below, A5) ride per-rank rings allocated by the runner in B4.
using TestShmRing = ShmRing<uint64_t, 1024>;

// === Phase A5: real wire-event ShmRing instantiations ===
//
// Capacities are chosen per the design doc's per-rank throughput
// estimates: 16384 slots gives ~9.5MB region for write_ring (5.7MB/s
// traffic at 2M tok/s), 16384 for query_ring (~9MB), and 8192 for
// replay_outcome_ring (~770KB at 640KB/s replay traffic). All powers
// of 2 to satisfy SpscRing's mask-based-modulo static_assert.
using ShmRingWriteEventT = ShmRing<WriteEvent, 16384>;
using ShmRingQueryEventT = ShmRing<QueryEvent, 16384>;
using ShmRingReplayOutcomeT = ShmRing<ReplayOutcome, 8192>;

namespace {

// Strict-key dict validator. Raises Python KeyError on extra or missing
// keys — callers can't accidentally rely on a zero-initialized field
// by omitting it, and a typo in a field name fails fast rather than
// silently overwriting an unrelated field. Single helper for all three
// event types (the key set is the only difference).
void check_dict_keys(const pybind11::dict& d,
                     const char* const* keys,
                     std::size_t key_count,
                     const char* event_name) {
  // Missing key check.
  for (std::size_t i = 0; i < key_count; ++i) {
    if (!d.contains(keys[i])) {
      throw pybind11::key_error(
          std::string(event_name) + " dict missing required key '" +
          keys[i] + "'");
    }
  }
  // Extra key check — len(d) == key_count combined with the missing
  // check above implies the key sets match exactly.
  if (d.size() != key_count) {
    // Find the offending key for a clear error message.
    for (auto item : d) {
      const std::string k = pybind11::str(item.first);
      bool found = false;
      for (std::size_t i = 0; i < key_count; ++i) {
        if (k == keys[i]) { found = true; break; }
      }
      if (!found) {
        throw pybind11::key_error(
            std::string(event_name) + " dict has unexpected key '" +
            k + "'");
      }
    }
  }
}

// Copy a Python sequence of length `n` into a uint16_t array. Used by
// WriteEvent.key_rep / value_tok_ids and QueryEvent.query_rep. Raises
// ValueError on length mismatch (programmer error, not silent
// truncation) and propagates pybind11's TypeError on non-integer items.
void copy_u16_array(const pybind11::handle& seq,
                    uint16_t* dst,
                    std::size_t n,
                    const char* field_name) {
  auto pyseq = pybind11::reinterpret_borrow<pybind11::sequence>(seq);
  if (static_cast<std::size_t>(pybind11::len(pyseq)) != n) {
    throw pybind11::value_error(
        std::string("field '") + field_name + "' must have length " +
        std::to_string(n) + ", got " + std::to_string(pybind11::len(pyseq)));
  }
  for (std::size_t i = 0; i < n; ++i) {
    dst[i] = pyseq[i].cast<uint16_t>();
  }
}

// Reverse direction — uint16_t array → Python list of ints.
pybind11::list u16_array_to_list(const uint16_t* src, std::size_t n) {
  pybind11::list out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = pybind11::int_(src[i]);
  }
  return out;
}

// --- WriteEvent dict <-> struct ---
constexpr const char* kWriteEventKeys[] = {
    "event_type", "source_rank", "write_bucket",
    "candidate_id", "gpu_step", "key_fp",
    "key_rep", "value_tok_ids", "value_anchor_id",
    "pressure_at_write", "pre_write_ce",
};
constexpr std::size_t kWriteEventKeyCount =
    sizeof(kWriteEventKeys) / sizeof(kWriteEventKeys[0]);

WriteEvent dict_to_write_event(const pybind11::dict& d) {
  check_dict_keys(d, kWriteEventKeys, kWriteEventKeyCount, "WriteEvent");
  WriteEvent ev{};
  ev.event_type        = d["event_type"].cast<uint8_t>();
  ev.source_rank       = d["source_rank"].cast<uint8_t>();
  ev.write_bucket      = d["write_bucket"].cast<uint8_t>();
  ev.candidate_id      = d["candidate_id"].cast<uint64_t>();
  ev.gpu_step          = d["gpu_step"].cast<uint64_t>();
  ev.key_fp            = d["key_fp"].cast<uint64_t>();
  copy_u16_array(d["key_rep"], ev.key_rep, KEY_REP_DIM_DEFAULT, "key_rep");
  copy_u16_array(d["value_tok_ids"], ev.value_tok_ids,
                 SPAN_LENGTH_DEFAULT, "value_tok_ids");
  ev.value_anchor_id   = d["value_anchor_id"].cast<uint32_t>();
  ev.pressure_at_write = d["pressure_at_write"].cast<float>();
  ev.pre_write_ce      = d["pre_write_ce"].cast<float>();
  return ev;
}

pybind11::dict write_event_to_dict(const WriteEvent& ev) {
  pybind11::dict d;
  d["event_type"]        = pybind11::int_(ev.event_type);
  d["source_rank"]       = pybind11::int_(ev.source_rank);
  d["write_bucket"]      = pybind11::int_(ev.write_bucket);
  d["candidate_id"]      = pybind11::int_(ev.candidate_id);
  d["gpu_step"]          = pybind11::int_(ev.gpu_step);
  d["key_fp"]            = pybind11::int_(ev.key_fp);
  d["key_rep"]           = u16_array_to_list(ev.key_rep, KEY_REP_DIM_DEFAULT);
  d["value_tok_ids"]     = u16_array_to_list(ev.value_tok_ids, SPAN_LENGTH_DEFAULT);
  d["value_anchor_id"]   = pybind11::int_(ev.value_anchor_id);
  d["pressure_at_write"] = pybind11::float_(ev.pressure_at_write);
  d["pre_write_ce"]      = pybind11::float_(ev.pre_write_ce);
  return d;
}

// --- QueryEvent dict <-> struct ---
constexpr const char* kQueryEventKeys[] = {
    "event_type", "source_rank", "bucket",
    "query_id", "gpu_step", "query_rep",
    "pressure", "pre_query_ce",
};
constexpr std::size_t kQueryEventKeyCount =
    sizeof(kQueryEventKeys) / sizeof(kQueryEventKeys[0]);

QueryEvent dict_to_query_event(const pybind11::dict& d) {
  check_dict_keys(d, kQueryEventKeys, kQueryEventKeyCount, "QueryEvent");
  QueryEvent ev{};
  ev.event_type   = d["event_type"].cast<uint8_t>();
  ev.source_rank  = d["source_rank"].cast<uint8_t>();
  ev.bucket       = d["bucket"].cast<uint8_t>();
  ev.query_id     = d["query_id"].cast<uint64_t>();
  ev.gpu_step     = d["gpu_step"].cast<uint64_t>();
  copy_u16_array(d["query_rep"], ev.query_rep, KEY_REP_DIM_DEFAULT, "query_rep");
  ev.pressure     = d["pressure"].cast<float>();
  ev.pre_query_ce = d["pre_query_ce"].cast<float>();
  return ev;
}

pybind11::dict query_event_to_dict(const QueryEvent& ev) {
  pybind11::dict d;
  d["event_type"]   = pybind11::int_(ev.event_type);
  d["source_rank"]  = pybind11::int_(ev.source_rank);
  d["bucket"]       = pybind11::int_(ev.bucket);
  d["query_id"]     = pybind11::int_(ev.query_id);
  d["gpu_step"]     = pybind11::int_(ev.gpu_step);
  d["query_rep"]    = u16_array_to_list(ev.query_rep, KEY_REP_DIM_DEFAULT);
  d["pressure"]     = pybind11::float_(ev.pressure);
  d["pre_query_ce"] = pybind11::float_(ev.pre_query_ce);
  return d;
}

// --- ReplayOutcome dict <-> struct ---
constexpr const char* kReplayOutcomeKeys[] = {
    "event_type", "selected_rank", "outcome_status",
    "replay_id", "gpu_step", "query_event_id", "source_write_id",
    "slot_id", "policy_version", "selection_step",
    "teacher_score", "controller_logit",
    "ce_before_replay", "ce_after_replay", "ce_delta_raw",
    "bucket_baseline", "reward_shaped",
    "grad_cos_rare", "grad_cos_total",
    "flags",
};
constexpr std::size_t kReplayOutcomeKeyCount =
    sizeof(kReplayOutcomeKeys) / sizeof(kReplayOutcomeKeys[0]);

ReplayOutcome dict_to_replay_outcome(const pybind11::dict& d) {
  check_dict_keys(d, kReplayOutcomeKeys, kReplayOutcomeKeyCount,
                  "ReplayOutcome");
  ReplayOutcome ev{};
  ev.event_type       = d["event_type"].cast<uint8_t>();
  ev.selected_rank    = d["selected_rank"].cast<uint8_t>();
  ev.outcome_status   = d["outcome_status"].cast<uint8_t>();
  ev.replay_id        = d["replay_id"].cast<uint64_t>();
  ev.gpu_step         = d["gpu_step"].cast<uint64_t>();
  ev.query_event_id   = d["query_event_id"].cast<uint64_t>();
  ev.source_write_id  = d["source_write_id"].cast<uint64_t>();
  ev.slot_id          = d["slot_id"].cast<uint32_t>();
  ev.policy_version   = d["policy_version"].cast<uint32_t>();
  ev.selection_step   = d["selection_step"].cast<uint64_t>();
  ev.teacher_score    = d["teacher_score"].cast<float>();
  ev.controller_logit = d["controller_logit"].cast<float>();
  ev.ce_before_replay = d["ce_before_replay"].cast<float>();
  ev.ce_after_replay  = d["ce_after_replay"].cast<float>();
  ev.ce_delta_raw     = d["ce_delta_raw"].cast<float>();
  ev.bucket_baseline  = d["bucket_baseline"].cast<float>();
  ev.reward_shaped    = d["reward_shaped"].cast<float>();
  ev.grad_cos_rare    = d["grad_cos_rare"].cast<float>();
  ev.grad_cos_total   = d["grad_cos_total"].cast<float>();
  ev.flags            = d["flags"].cast<uint16_t>();
  return ev;
}

pybind11::dict replay_outcome_to_dict(const ReplayOutcome& ev) {
  pybind11::dict d;
  d["event_type"]       = pybind11::int_(ev.event_type);
  d["selected_rank"]    = pybind11::int_(ev.selected_rank);
  d["outcome_status"]   = pybind11::int_(ev.outcome_status);
  d["replay_id"]        = pybind11::int_(ev.replay_id);
  d["gpu_step"]         = pybind11::int_(ev.gpu_step);
  d["query_event_id"]   = pybind11::int_(ev.query_event_id);
  d["source_write_id"]  = pybind11::int_(ev.source_write_id);
  d["slot_id"]          = pybind11::int_(ev.slot_id);
  d["policy_version"]   = pybind11::int_(ev.policy_version);
  d["selection_step"]   = pybind11::int_(ev.selection_step);
  d["teacher_score"]    = pybind11::float_(ev.teacher_score);
  d["controller_logit"] = pybind11::float_(ev.controller_logit);
  d["ce_before_replay"] = pybind11::float_(ev.ce_before_replay);
  d["ce_after_replay"]  = pybind11::float_(ev.ce_after_replay);
  d["ce_delta_raw"]     = pybind11::float_(ev.ce_delta_raw);
  d["bucket_baseline"]  = pybind11::float_(ev.bucket_baseline);
  d["reward_shaped"]    = pybind11::float_(ev.reward_shaped);
  d["grad_cos_rare"]    = pybind11::float_(ev.grad_cos_rare);
  d["grad_cos_total"]   = pybind11::float_(ev.grad_cos_total);
  d["flags"]            = pybind11::int_(ev.flags);
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_step", &forward_step, "CPU SSM controller reference step");
  m.def("load_weights_from_path", &load_weights_from_path,
        "Load a CSWG controller pretrain weight dump");
  m.def("forward_pretrain_model", &forward_pretrain_model,
        "Run the D4 controller pretrain model from a CSWG weight dict");
  m.def("has_amx_bf16", &has_amx_bf16, "Whether built with AMX BF16 support");
  m.def("backend_name", &backend_name, "Compiled backend name");
  m.def("wire_event_sizes", &wire_event_sizes,
        "Byte sizes of WriteEvent / QueryEvent / ReplayOutcome wire structs");
  m.def("wire_event_min_slot_alignment", &wire_event_min_slot_alignment,
        "Single int — minimum ShmRing slot alignment shared by all three wire events");
  m.def("wire_event_constants", &wire_event_constants,
        "Compile-time constants driving wire-event array dimensions");
  m.def("controller_main",
        [](const std::vector<std::string>& write_ring_names,
           const std::string& query_ring_name,
           const std::string& replay_ring_name,
           const std::string& exit_flag_shm_name,
           uint32_t idle_sleep_ns) {
          pybind11::gil_scoped_release release;
          return controller_main(
              write_ring_names,
              query_ring_name,
              replay_ring_name,
              exit_flag_shm_name,
              idle_sleep_ns);
        },
        pybind11::arg("write_ring_names"),
        pybind11::arg("query_ring_name"),
        pybind11::arg("replay_ring_name"),
        pybind11::arg("exit_flag_shm_name"),
        pybind11::arg("idle_sleep_ns") = 100,
        "Poll controller shm rings until the 1-byte exit flag is nonzero. "
        "C1 stub handlers count records and return the total processed.");

  pybind11::class_<ActionHistoryEntry>(m, "ActionHistoryEntry")
      .def(pybind11::init<>())
      .def_readwrite("action_type", &ActionHistoryEntry::action_type)
      .def_readwrite("gpu_step", &ActionHistoryEntry::gpu_step)
      .def_readwrite("policy_version", &ActionHistoryEntry::policy_version)
      .def_readwrite("output_logit", &ActionHistoryEntry::output_logit)
      .def_readwrite("selected_rank", &ActionHistoryEntry::selected_rank)
      .def_readwrite("neighbor_slot", &ActionHistoryEntry::neighbor_slot)
      .def_readwrite("global_state", &ActionHistoryEntry::global_state)
      .def_readwrite("slot_state", &ActionHistoryEntry::slot_state);

  pybind11::class_<PerSlotActionHistory>(m, "PerSlotActionHistory")
      .def(pybind11::init<uint32_t, uint32_t>(),
           pybind11::arg("num_slots"),
           pybind11::arg("max_entries_per_slot"))
      .def("append", &PerSlotActionHistory::append,
           pybind11::arg("slot_id"), pybind11::arg("entry"))
      .def("history", &PerSlotActionHistory::history,
           pybind11::arg("slot_id"),
           pybind11::return_value_policy::reference_internal)
      .def("mark_evicted", &PerSlotActionHistory::mark_evicted,
           pybind11::arg("slot_id"), pybind11::arg("current_event_id"))
      .def("gc", &PerSlotActionHistory::gc,
           pybind11::arg("current_event_id"), pybind11::arg("gc_lookahead"))
      .def("size", &PerSlotActionHistory::size, pybind11::arg("slot_id"))
      .def("is_evicted", &PerSlotActionHistory::is_evicted,
           pybind11::arg("slot_id"))
      .def_property_readonly("num_slots", &PerSlotActionHistory::num_slots)
      .def_property_readonly("max_entries_per_slot",
                             &PerSlotActionHistory::max_entries_per_slot);

  // Phase A2 test fixture — see tests/test_spsc_ring.py. `capacity` is
  // exposed as a static class property (not a method) because the
  // template parameter is compile-time; `pop` returns Optional[int]
  // via the std::optional caster from pybind11/stl.h.
  pybind11::class_<TestRing>(m, "SpscRingU64x1024")
      .def(pybind11::init<>())
      .def("push", &TestRing::push, "Push u64; returns False if full")
      .def("pop", &TestRing::pop, "Pop u64; returns None if empty")
      .def("size", &TestRing::size, "Approximate occupied-slot count")
      .def_property_readonly_static(
          "capacity",
          [](pybind11::object) { return TestRing::capacity(); },
          "Compile-time capacity (1024)");

  // Phase A3 binding — see tests/test_posix_shm.py. `write_bytes` /
  // `read_bytes` exist for the test only; production callers will go
  // through Phase A4's ShmRing rather than touching raw bytes. The
  // bytes accessors validate (offset, length) against the region size
  // here in C++ so a Python bug can't smash arbitrary memory.
  pybind11::class_<PosixShm>(m, "PosixShm")
      .def(pybind11::init<const std::string&, std::size_t, bool>(),
           pybind11::arg("name"), pybind11::arg("size"), pybind11::arg("create"),
           "Open or create a POSIX shm region. name must start with '/'; "
           "size is bytes (ignored when create=False — the existing region's "
           "size is recovered via fstat).")
      .def("size", &PosixShm::size, "Region size in bytes")
      .def("name", &PosixShm::name, "Region name (the '/'-prefixed POSIX name)")
      .def_static("unlink", &PosixShm::unlink, pybind11::arg("name"),
                  "Remove the name from the kernel namespace. ENOENT is a no-op "
                  "so re-runs after clean teardown don't error.")
      .def("write_bytes",
           [](PosixShm& self, std::size_t offset, pybind11::bytes data) {
             const std::string s = data;  // pybind11::bytes → std::string copy
             if (offset + s.size() > self.size()) {
               throw std::out_of_range(
                   "PosixShm.write_bytes: offset(" + std::to_string(offset) +
                   ") + len(" + std::to_string(s.size()) + ") > size(" +
                   std::to_string(self.size()) + ")");
             }
             std::memcpy(static_cast<char*>(self.ptr()) + offset, s.data(), s.size());
           },
           pybind11::arg("offset"), pybind11::arg("data"),
           "Test-only: copy bytes into the region at offset.")
      .def("read_bytes",
           [](const PosixShm& self, std::size_t offset, std::size_t length) {
             if (offset + length > self.size()) {
               throw std::out_of_range(
                   "PosixShm.read_bytes: offset(" + std::to_string(offset) +
                   ") + length(" + std::to_string(length) + ") > size(" +
                   std::to_string(self.size()) + ")");
             }
             return pybind11::bytes(
                 static_cast<const char*>(self.ptr()) + offset, length);
           },
           pybind11::arg("offset"), pybind11::arg("length"),
           "Test-only: read `length` bytes from the region at offset.");

  // Phase A4 binding — see tests/test_shm_ring.py. Move-only class
  // (owns a PosixShm) so we expose factory methods via def_static and
  // omit the default `def(py::init<>())`. pybind11's default
  // unique_ptr holder handles move-only types returned by value from
  // factories. `capacity` and `REGION_BYTES` are exposed as static
  // class properties (compile-time template parameters), matching the
  // A2 pattern for `capacity`.
  pybind11::class_<TestShmRing>(m, "ShmRingU64x1024")
      .def_static("create", &TestShmRing::create, pybind11::arg("name"),
                  "Creator-side factory: shm_open(O_CREAT|O_RDWR), "
                  "ftruncate, mmap, then placement-new the SpscRing into "
                  "the region. Returns the owning ShmRing handle.")
      .def_static("attach", &TestShmRing::attach, pybind11::arg("name"),
                  "Attacher-side factory: shm_open(O_RDWR), mmap an "
                  "existing region. Does NOT re-construct the ring; the "
                  "SPSC state is already there from the creator.")
      .def("push", &TestShmRing::push, pybind11::arg("item"),
           "Push u64; returns False if full. Producer-side only.")
      .def("pop", &TestShmRing::pop,
           "Pop u64; returns None if empty. Consumer-side only.")
      .def("size", &TestShmRing::size,
           "Approximate occupied-slot count.")
      .def("name", &TestShmRing::name,
           "POSIX shm name (the '/'-prefixed region name).")
      .def_static("unlink", &TestShmRing::unlink, pybind11::arg("name"),
                  "Remove the name from the kernel namespace. Forwards "
                  "to PosixShm::unlink, which treats ENOENT as a no-op.")
      .def_property_readonly_static(
          "capacity",
          [](pybind11::object) { return TestShmRing::capacity(); },
          "Compile-time capacity (1024).")
      .def_property_readonly_static(
          "REGION_BYTES",
          [](pybind11::object) { return TestShmRing::REGION_BYTES; },
          "Static byte size of the underlying SpscRing<T, Capacity> — "
          "the shm region size required by `create` / `attach`.");

  // === Phase A5 — real wire-event ShmRing instantiations ===
  //
  // Each binding mirrors the A4 ShmRingU64x1024 surface but accepts /
  // returns Python dicts whose keys match the non-pad fields of the
  // corresponding wire-event struct. dict_to_*_event / *_to_dict
  // helpers above do the field-by-field copy and validate the key set
  // so a typo or omission fails fast with KeyError.

  pybind11::class_<ShmRingWriteEventT>(m, "ShmRingWriteEvent")
      .def_static("create", &ShmRingWriteEventT::create, pybind11::arg("name"),
                  "Creator-side factory — allocates an ~9.5MB shm region "
                  "(REGION_BYTES) and placement-news the SpscRing<WriteEvent, "
                  "16384> into it.")
      .def_static("attach", &ShmRingWriteEventT::attach, pybind11::arg("name"),
                  "Attacher-side factory — mmap an existing region created "
                  "by ShmRingWriteEvent.create.")
      .def("push",
           [](ShmRingWriteEventT& self, const pybind11::dict& d) {
             return self.push(dict_to_write_event(d));
           },
           pybind11::arg("event"),
           "Push a WriteEvent dict; returns False if the ring is full. "
           "Raises KeyError on missing/extra keys.")
      .def("pop",
           [](ShmRingWriteEventT& self) -> pybind11::object {
             auto opt = self.pop();
             if (!opt.has_value()) {
               return pybind11::none();
             }
             return write_event_to_dict(*opt);
           },
           "Pop a WriteEvent; returns a dict, or None if the ring is empty.")
      .def("size", &ShmRingWriteEventT::size,
           "Approximate occupied-slot count.")
      .def("name", &ShmRingWriteEventT::name,
           "POSIX shm name (the '/'-prefixed region name).")
      .def_static("unlink", &ShmRingWriteEventT::unlink, pybind11::arg("name"),
                  "Remove the name from the kernel namespace.")
      .def_property_readonly_static(
          "capacity",
          [](pybind11::object) { return ShmRingWriteEventT::capacity(); },
          "Compile-time capacity (16384).")
      .def_property_readonly_static(
          "REGION_BYTES",
          [](pybind11::object) { return ShmRingWriteEventT::REGION_BYTES; },
          "Static byte size of SpscRing<WriteEvent, 16384>.");

  pybind11::class_<ShmRingQueryEventT>(m, "ShmRingQueryEvent")
      .def_static("create", &ShmRingQueryEventT::create, pybind11::arg("name"),
                  "Creator-side factory — allocates an ~9MB shm region "
                  "(REGION_BYTES) and placement-news the SpscRing<QueryEvent, "
                  "16384> into it.")
      .def_static("attach", &ShmRingQueryEventT::attach, pybind11::arg("name"),
                  "Attacher-side factory — mmap an existing region.")
      .def("push",
           [](ShmRingQueryEventT& self, const pybind11::dict& d) {
             return self.push(dict_to_query_event(d));
           },
           pybind11::arg("event"),
           "Push a QueryEvent dict; returns False if full. "
           "Raises KeyError on missing/extra keys.")
      .def("pop",
           [](ShmRingQueryEventT& self) -> pybind11::object {
             auto opt = self.pop();
             if (!opt.has_value()) {
               return pybind11::none();
             }
             return query_event_to_dict(*opt);
           },
           "Pop a QueryEvent; returns a dict, or None if empty.")
      .def("size", &ShmRingQueryEventT::size,
           "Approximate occupied-slot count.")
      .def("name", &ShmRingQueryEventT::name,
           "POSIX shm name.")
      .def_static("unlink", &ShmRingQueryEventT::unlink, pybind11::arg("name"),
                  "Remove the name from the kernel namespace.")
      .def_property_readonly_static(
          "capacity",
          [](pybind11::object) { return ShmRingQueryEventT::capacity(); },
          "Compile-time capacity (16384).")
      .def_property_readonly_static(
          "REGION_BYTES",
          [](pybind11::object) { return ShmRingQueryEventT::REGION_BYTES; },
          "Static byte size of SpscRing<QueryEvent, 16384>.");

  pybind11::class_<ShmRingReplayOutcomeT>(m, "ShmRingReplayOutcome")
      .def_static("create", &ShmRingReplayOutcomeT::create, pybind11::arg("name"),
                  "Creator-side factory — allocates a ~770KB shm region "
                  "(REGION_BYTES) and placement-news the SpscRing<ReplayOutcome, "
                  "8192> into it.")
      .def_static("attach", &ShmRingReplayOutcomeT::attach, pybind11::arg("name"),
                  "Attacher-side factory — mmap an existing region.")
      .def("push",
           [](ShmRingReplayOutcomeT& self, const pybind11::dict& d) {
             return self.push(dict_to_replay_outcome(d));
           },
           pybind11::arg("event"),
           "Push a ReplayOutcome dict; returns False if full. "
           "Raises KeyError on missing/extra keys.")
      .def("pop",
           [](ShmRingReplayOutcomeT& self) -> pybind11::object {
             auto opt = self.pop();
             if (!opt.has_value()) {
               return pybind11::none();
             }
             return replay_outcome_to_dict(*opt);
           },
           "Pop a ReplayOutcome; returns a dict, or None if empty.")
      .def("size", &ShmRingReplayOutcomeT::size,
           "Approximate occupied-slot count.")
      .def("name", &ShmRingReplayOutcomeT::name,
           "POSIX shm name.")
      .def_static("unlink", &ShmRingReplayOutcomeT::unlink, pybind11::arg("name"),
                  "Remove the name from the kernel namespace.")
      .def_property_readonly_static(
          "capacity",
          [](pybind11::object) { return ShmRingReplayOutcomeT::capacity(); },
          "Compile-time capacity (8192).")
      .def_property_readonly_static(
          "REGION_BYTES",
          [](pybind11::object) { return ShmRingReplayOutcomeT::REGION_BYTES; },
          "Static byte size of SpscRing<ReplayOutcome, 8192>.");
}
