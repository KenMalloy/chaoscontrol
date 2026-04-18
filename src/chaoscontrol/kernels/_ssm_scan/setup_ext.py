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
    cu_fwd_rel = this_dir.relative_to(repo_root) / "src" / "ssm_scan_fwd.cu"
    cu_bwd_rel = this_dir.relative_to(repo_root) / "src" / "ssm_scan_bwd.cu"
    sources = [str(cpp_rel), str(cu_fwd_rel), str(cu_bwd_rel)]

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
        # Use fast-math. `-use_fast_math` bundles several nvcc flags:
        #   * `-fmad=true`          — fuse `a*b + c` into a single
        #                             fp32 FMA. This is the main win
        #                             for us. No ordering hazard: FMA
        #                             collapses two consecutive ops
        #                             into one, it does not reorder
        #                             across the t→t+1 boundary (the
        #                             loop is serial inside each
        #                             thread).
        #   * `-ftz=true`           — flush denormals to zero. Our
        #                             decay values are bounded below
        #                             by ~exp(-delta*a) which stays
        #                             well above 1e-30; update values
        #                             are clamped by tanh(·)*sigmoid(·)
        #                             inside _diag_terms. We never
        #                             produce denormal intermediates,
        #                             so FTZ is a latency win with no
        #                             numerical cost.
        #   * `-prec-div=false`     — use approximate fp32 div. We
        #                             don't use fp32 div in the scan
        #                             kernels at all (only multiply
        #                             and add). No-op for us.
        #   * `-prec-sqrt=false`    — use approximate fp32 sqrt. We
        #                             don't use fp32 sqrt in the scan
        #                             kernels. No-op.
        #
        # Net effect for this kernel: FMA fusion, other flags are
        # dead code. Kept as `-use_fast_math` for consistency with
        # the sibling `_cublaslt` extension.
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
