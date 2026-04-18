// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Process-wide cache of cuBLASLt matrix layouts + chosen algo, keyed on
// (shape, bias_mode, fast_accum, dtype-triple). First call for a key pays
// the ~50-100 µs heuristic lookup + four layout creations; every
// subsequent call for the same key is a hashmap hit + pointer grab.
//
// Scope decision: we cache the IMMUTABLE planning state (layouts +
// algo). The op_desc itself is rebuilt per call — it's cheap (~5
// SetAttribute calls) and rebuilding avoids thread-safety concerns when
// SetAttribute mutates shared state while another caller is mid-matmul.
// If a future multi-stream setup appears we'd want to either lock the
// desc or make it per-stream; the layouts + algo are read-only after
// construction and safe to share.
//
// A training step at L=4 SSM layers does 3 matmuls per layer (fwd,
// bwd_x, bwd_w) × 4 layers = 12 calls. With the cache, only the very
// first step pays heuristic cost; steady-state is purely hashmap hits.

#pragma once

#include <cublasLt.h>
#include <c10/core/ScalarType.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

namespace cc_cublaslt {

// Epilogue selection. Mirrors the three cuBLASLt epilogues we use.
// Promoted out of the anonymous namespace in cublaslt_fp8_matmul.cpp so
// the cache key can reference it.
enum class BiasMode {
    None,         // CUBLASLT_EPILOGUE_DEFAULT
    ForwardBias,  // CUBLASLT_EPILOGUE_BIAS — bias added to D
    BGradB,       // CUBLASLT_EPILOGUE_BGRADB — reduce B over k, emit bias grad
};

struct DescriptorKey {
    int64_t M;
    int64_t N;
    int64_t K;
    uint32_t align_a_bytes;
    uint32_t align_b_bytes;
    uint32_t align_c_bytes;
    uint32_t align_d_bytes;
    BiasMode bias_mode;
    bool fast_accum;
    at::ScalarType a_dtype;
    at::ScalarType b_dtype;
    at::ScalarType out_dtype;

    bool operator==(const DescriptorKey& o) const noexcept {
        return M == o.M && N == o.N && K == o.K
               && align_a_bytes == o.align_a_bytes
               && align_b_bytes == o.align_b_bytes
               && align_c_bytes == o.align_c_bytes
               && align_d_bytes == o.align_d_bytes
               && bias_mode == o.bias_mode && fast_accum == o.fast_accum
               && a_dtype == o.a_dtype && b_dtype == o.b_dtype
               && out_dtype == o.out_dtype;
    }
};

struct DescriptorKeyHash {
    size_t operator()(const DescriptorKey& k) const noexcept {
        // Boost-style hash combine. The spread on (M, N, K) matters
        // most because (bias_mode, fast_accum, dtypes) are small enums.
        auto mix = [](size_t seed, size_t v) {
            return seed ^ (v + 0x9e3779b9ull + (seed << 6) + (seed >> 2));
        };
        size_t h = std::hash<int64_t>{}(k.M);
        h = mix(h, std::hash<int64_t>{}(k.N));
        h = mix(h, std::hash<int64_t>{}(k.K));
        h = mix(h, std::hash<uint32_t>{}(k.align_a_bytes));
        h = mix(h, std::hash<uint32_t>{}(k.align_b_bytes));
        h = mix(h, std::hash<uint32_t>{}(k.align_c_bytes));
        h = mix(h, std::hash<uint32_t>{}(k.align_d_bytes));
        h = mix(h, static_cast<size_t>(k.bias_mode));
        h = mix(h, static_cast<size_t>(k.fast_accum));
        h = mix(h, static_cast<size_t>(k.a_dtype));
        h = mix(h, static_cast<size_t>(k.b_dtype));
        h = mix(h, static_cast<size_t>(k.out_dtype));
        return h;
    }
};

// Cached planning state. Layouts + algo only — op_desc is per-call so
// we don't need to manage its mutation races here.
struct CachedGemmPlan {
    cublasLtMatrixLayout_t A_layout = nullptr;
    cublasLtMatrixLayout_t B_layout = nullptr;
    cublasLtMatrixLayout_t C_layout = nullptr;
    cublasLtMatrixLayout_t D_layout = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};

    CachedGemmPlan() = default;
    ~CachedGemmPlan();

    CachedGemmPlan(const CachedGemmPlan&) = delete;
    CachedGemmPlan& operator=(const CachedGemmPlan&) = delete;
};

class DescriptorCache {
public:
    static DescriptorCache& instance();

    // Returns a reference valid for the process lifetime. The cache
    // owns every plan and never evicts — Param Golf's shape set is
    // finite (a handful of distinct (M,N,K,dtype) tuples per model)
    // so unbounded growth isn't a practical concern.
    //
    // On first insertion for a key, builds the plan by running the
    // cuBLASLt heuristic and allocating the four matrix layouts.
    // Subsequent calls with the same key are read-only cache hits.
    const CachedGemmPlan& get(const DescriptorKey& key,
                              cublasLtHandle_t handle,
                              size_t workspace_bytes);

    DescriptorCache(const DescriptorCache&) = delete;
    DescriptorCache& operator=(const DescriptorCache&) = delete;

private:
    DescriptorCache() = default;

    std::shared_mutex mu_;
    std::unordered_map<DescriptorKey,
                       std::unique_ptr<CachedGemmPlan>,
                       DescriptorKeyHash> cache_;
};

}  // namespace cc_cublaslt
