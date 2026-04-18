"""Build hook for the SSM diag-scan extension.

Invoked from the repo-root ``setup.py``. Sibling of
``_cublaslt/setup_ext.py`` — same shape, different sources. The kernel
has no cuBLAS dependency; it's a pure CUDA kernel compiled by nvcc.
"""
from __future__ import annotations

from pathlib import Path


def build_ext_modules() -> list:
    """Return the list of extension objects to pass to ``setuptools.setup``.

    Returns an empty list on environments where nvcc / the CUDA toolchain
    is not available (dev macs). The Python wrapper in ``__init__.py``
    handles the case where the extension wasn't compiled by raising at
    call time with a clear ImportError.

    Pod environment expectation: CUDA 13 toolchain with nvcc in PATH and
    a sm_90 H100 available at runtime. No cuBLAS linkage required — this
    extension is a standalone CUDA kernel + pybind11 binding.
    """
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError:
        return []

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[3]
    cpp_rel = this_dir.relative_to(repo_root) / "src" / "ssm_scan_binding.cpp"
    cu_rel = this_dir.relative_to(repo_root) / "src" / "ssm_scan_fwd.cu"
    sources = [str(cpp_rel), str(cu_rel)]

    src_dir = this_dir / "src"
    include_dirs = [str(src_dir)]

    # nvcc must be available for .cu compilation — if the host machine
    # doesn't have it, CUDAExtension construction will throw deep in
    # setup, which we want to surface only on pods (dev mac falls
    # through the ImportError path above via `torch.utils.cpp_extension`
    # still importing but nvcc absent). Handle that explicitly.
    import shutil
    if shutil.which("nvcc") is None:
        return []

    cxx_args = ["-O3", "-std=c++17"]
    nvcc_args = [
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_90,code=sm_90",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        # Use fast-math for fp32 FMAs inside the scan — scan is
        # accumulator-dominated and we want the bf16-cast path to hit
        # the H100 tensor pipeline's fp32 FMA throughput. Disable
        # auto-contract that could reorder operations across the
        # recurrence (we do not want `decay * state + update` reduced
        # to an imprecise pattern).
        "-use_fast_math",
        # Silence the CCCL compatibility check: our cu13 pod has nvcc
        # 13.2 + cudart 13.0, which mismatch the minor version but are
        # still ABI-compatible. Observed 2026-04-18: the `cuda_toolkit.h`
        # sanity check in ``nvidia/cu13/include/cccl/cuda/std/__cccl/``
        # rejects this out of the box. We hit it because including
        # ``c10/core/ScalarType.h`` transitively pulls in cccl's
        # thrust/complex headers. Defining this macro bypasses the
        # check; the upstream header's own comment calls out that
        # users on newer CTKs should do exactly this. We are on the
        # "newer nvcc, older cudart" branch rather than the reverse,
        # but the macro has the same effect — skip the version gate,
        # let the host code link at runtime.
        "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
    ]
    extra_compile_args = {"cxx": cxx_args, "nvcc": nvcc_args}

    ext = CUDAExtension(
        name="chaoscontrol.kernels._ssm_scan._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )
    return [ext]


def cmdclass_with_build_ext() -> dict:
    """Return a cmdclass dict with torch's BuildExtension pre-wired."""
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError:
        return {}
    return {"build_ext": BuildExtension}


if __name__ == "__main__":
    # Allow manual invocation: `python setup_ext.py build_ext --inplace`
    # from this directory.
    from setuptools import setup

    exts = build_ext_modules()
    if not exts:
        raise SystemExit(
            "no extension to build — nvcc / CUDA toolchain missing"
        )
    setup(
        name="chaoscontrol_ssm_scan_ext",
        ext_modules=exts,
        cmdclass=cmdclass_with_build_ext(),
    )
