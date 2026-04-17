// Copyright (c) 2026 Kenneth Malloy. Licensed under the project LICENSE.
//
// Per-process cublasLtHandle + workspace holder. cuBLASLt wants a ~32 MiB
// workspace scratchpad for heuristic selection and split-k staging; we
// allocate it once and reuse across calls. The handle itself is cheap to
// create but there is no reason to recreate it per call.
//
// Not thread-safe across multiple CUDA streams on different host threads;
// good enough for single-process training where all calls come from the
// same python thread on the default stream. A later revision can key the
// workspace on a `(device, stream)` pair.

#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace cc_cublaslt {

inline constexpr size_t kWorkspaceBytes = 32ull * 1024ull * 1024ull;  // 32 MiB

// Minimal cuBLAS error check — throws std::runtime_error with a readable
// code so pybind11 surfaces a clean Python exception.
inline void check_cublas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cublasLt error in ") + what +
                                 ": code " + std::to_string(static_cast<int>(s)));
    }
}

inline void check_cuda(cudaError_t s, const char* what) {
    if (s != cudaSuccess) {
        throw std::runtime_error(std::string("cuda error in ") + what + ": " +
                                 cudaGetErrorString(s));
    }
}

class WorkspaceCache {
public:
    static WorkspaceCache& instance() {
        static WorkspaceCache cache;
        return cache;
    }

    cublasLtHandle_t handle() { return handle_; }
    void* workspace() { return workspace_; }
    size_t workspace_bytes() const { return kWorkspaceBytes; }

    WorkspaceCache(const WorkspaceCache&) = delete;
    WorkspaceCache& operator=(const WorkspaceCache&) = delete;

private:
    WorkspaceCache() {
        check_cublas(cublasLtCreate(&handle_), "cublasLtCreate");
        check_cuda(cudaMalloc(&workspace_, kWorkspaceBytes), "cudaMalloc(workspace)");
    }

    ~WorkspaceCache() {
        // Intentionally not destroying in production (process-lifetime
        // singleton). If someone tries to tear the cache down during
        // interpreter shutdown, the driver may already be gone; swallow.
        if (workspace_ != nullptr) {
            cudaFree(workspace_);
        }
        if (handle_ != nullptr) {
            cublasLtDestroy(handle_);
        }
    }

    cublasLtHandle_t handle_ = nullptr;
    void* workspace_ = nullptr;
};

}  // namespace cc_cublaslt
