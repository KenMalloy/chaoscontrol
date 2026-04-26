#include <torch/extension.h>

// pybind11/stl.h pulls in the std::optional<T> caster used by
// SpscRing::pop()'s binding (returns Python None when empty). Without
// this include the binding compiles but pop() returns an opaque
// std::optional object instead of None / int.
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <tuple>

#include "posix_shm.h"
#include "shm_ring.h"
#include "spsc_ring.h"
#include "wire_events.h"

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
// (ShmRing<WriteEvent, ...>, ShmRing<QueryEvent, ...>,
// ShmRing<ReplayOutcome, ...>) land in B4 when the per-rank lifecycle
// goes into the runner.
using TestShmRing = ShmRing<uint64_t, 1024>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_step", &forward_step, "CPU SSM controller reference step");
  m.def("has_amx_bf16", &has_amx_bf16, "Whether built with AMX BF16 support");
  m.def("backend_name", &backend_name, "Compiled backend name");
  m.def("wire_event_sizes", &wire_event_sizes,
        "Byte sizes of WriteEvent / QueryEvent / ReplayOutcome wire structs");
  m.def("wire_event_min_slot_alignment", &wire_event_min_slot_alignment,
        "Single int — minimum ShmRing slot alignment shared by all three wire events");
  m.def("wire_event_constants", &wire_event_constants,
        "Compile-time constants driving wire-event array dimensions");

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
}
