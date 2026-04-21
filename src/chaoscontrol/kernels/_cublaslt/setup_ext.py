"""Build hook for the cuBLASLt fp8 extension.

Invoked from the repo-root ``setup.py`` (which exists purely so
``pip install -e .`` can run an ext_modules build on top of the
otherwise pure-pyproject package). Isolated here so the wire-up
stays readable and easy to re-invoke manually during kernel work
(e.g. ``python -m chaoscontrol.kernels._cublaslt.setup_ext build_ext
--inplace`` from the repo root on a pod).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List


def _nvcc_gencode_args(default_arch_list: str = "9.0") -> list[str]:
    """Return nvcc ``-gencode`` flags from the requested CUDA arch list.

    Production wheels default to H100-only (``sm_90``), but scratch CUDA
    13 build pods can be cheaper Ada machines. ``TORCH_CUDA_ARCH_LIST``
    or ``CHAOSCONTROL_CUDA_ARCH_LIST`` can request both, e.g. ``8.9;9.0``.
    """
    raw = (
        os.environ.get("CHAOSCONTROL_CUDA_ARCH_LIST")
        or os.environ.get("TORCH_CUDA_ARCH_LIST")
        or default_arch_list
    )
    args: List[str] = []
    for arch in raw.replace(",", ";").split(";"):
        arch = arch.strip()
        if not arch:
            continue
        emit_ptx = arch.upper().endswith("+PTX")
        if emit_ptx:
            arch = arch[:-4].strip()
        digits = arch.replace(".", "")
        if not digits.isdigit():
            continue
        args.append(f"-gencode=arch=compute_{digits},code=sm_{digits}")
        if emit_ptx:
            args.append(f"-gencode=arch=compute_{digits},code=compute_{digits}")
    if args:
        return args
    return ["-gencode=arch=compute_90,code=sm_90"]


def build_ext_modules() -> list:
    """Return the list of extension objects to pass to ``setuptools.setup``.

    Returns an empty list on environments where the cuBLASLt headers + a
    working CUDA toolchain aren't available (dev macs). The Python
    wrapper in ``__init__.py`` handles the case where the extension
    wasn't compiled by raising at call time with a clear message.

    Pod environment expectation (CUDA 13 / torch cu130):
        * Headers: ``/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include``
          (cublasLt.h lives here, matching the runtime libs torch is
          loading via ``LD_LIBRARY_PATH`` / ldconfig).
        * Libs:    ``/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib``
          (libcublasLt.so.13).
        * Compiler: system ``g++``. No nvcc needed — the extension has
          no CUDA kernels, only host-side cuBLASLt calls.
    """
    try:
        from torch.utils.cpp_extension import CppExtension, CUDAExtension
    except ImportError:
        # torch not installed — no editable build here. The pyproject
        # declares torch>=2.0 so this should only hit truly broken envs.
        return []

    this_dir = Path(__file__).resolve().parent
    # setuptools rejects absolute paths for `sources` in the final
    # editable-wheel packaging step, so emit them relative to the
    # project root (which is this_dir.parents[3]: repo/src/chaoscontrol/
    # kernels/_cublaslt -> repo). `include_dirs` tolerates absolute paths
    # — only `sources` is policed.
    repo_root = this_dir.parents[3]
    cpp_rel = this_dir.relative_to(repo_root) / "src" / "cublaslt_fp8_matmul.cpp"
    desc_cache_rel = this_dir.relative_to(repo_root) / "src" / "descriptor_cache.cpp"
    cu_rel = this_dir.relative_to(repo_root) / "src" / "fused_amax_cast.cu"
    sources = [str(cpp_rel), str(desc_cache_rel), str(cu_rel)]

    src_dir = this_dir / "src"
    include_dirs = [str(src_dir)]
    library_dirs: List[str] = []
    libraries: List[str] = []

    # CUDA layout discovery — we support two layouts:
    #   1. cu13 wheel: /usr/local/lib/python3.12/dist-packages/nvidia/cu13/
    #      (single umbrella wheel, ships .so.13 versioned libs)
    #   2. cu12 wheels (the default): per-component wheels under
    #      /usr/local/lib/python3.12/dist-packages/nvidia/<component>/
    #      (cublas/, cuda_runtime/, ...) shipping .so.12 libs
    # We probe for cu13 first to match a cu13 torch build; fall back to
    # cu12 wheels; fall back to /usr/local/cuda for a classic CUDA toolkit
    # install. An explicit CC_CUBLAS_PREFIX env var overrides detection.
    cu13_umbrella = [
        "/usr/local/lib/python3.12/dist-packages/nvidia/cu13",
        "/usr/local/lib/python3.10/dist-packages/nvidia/cu13",
    ]
    cu_prefix = None
    cu_version = None  # 13 or 12 — selects the -l:libfoo.so.NN suffix.
    for c in cu13_umbrella:
        if os.path.exists(os.path.join(c, "include", "cublasLt.h")):
            cu_prefix = c
            cu_version = 13
            break
    if cu_prefix is None:
        # cu12 per-component layout: cublas/include/cublasLt.h lives in the
        # cublas wheel; cuda_runtime is a separate wheel.
        for dist in ("/usr/local/lib/python3.12/dist-packages/nvidia",
                     "/usr/local/lib/python3.10/dist-packages/nvidia"):
            cublas_inc = os.path.join(dist, "cublas", "include", "cublasLt.h")
            if os.path.exists(cublas_inc):
                # "prefix" here is synthetic — we aggregate include + lib
                # dirs from multiple component wheels below.
                cu_prefix = dist
                cu_version = 12
                break
    if os.environ.get("CC_CUBLAS_PREFIX"):
        cu_prefix = os.environ["CC_CUBLAS_PREFIX"]
        # If the user overrides, assume cu13 layout unless
        # CC_CUDA_VERSION is set.
        cu_version = int(os.environ.get("CC_CUDA_VERSION", "13"))
    if cu_prefix is None and os.path.exists("/usr/local/cuda/include/cublasLt.h"):
        cu_prefix = "/usr/local/cuda"
        # /usr/local/cuda is usually cu12.x on these pods; pin to 12
        # unless overridden.
        cu_version = int(os.environ.get("CC_CUDA_VERSION", "12"))

    if cu_prefix is None:
        # No cuBLASLt headers — skip the extension. Call sites get a
        # clean ImportError from __init__.py.
        return []

    if cu_version == 13:
        include_dirs.append(os.path.join(cu_prefix, "include"))
        for lib_sub in ("lib", "lib64"):
            lib_dir = os.path.join(cu_prefix, lib_sub)
            if os.path.isdir(lib_dir):
                library_dirs.append(lib_dir)
    elif cu_version == 12 and cu_prefix.endswith("/nvidia"):
        # cu12 per-component wheel layout.
        for component, headers in (
            ("cublas", True),
            ("cuda_runtime", True),
            ("cuda_cupti", False),
        ):
            comp_dir = os.path.join(cu_prefix, component)
            if headers:
                inc_dir = os.path.join(comp_dir, "include")
                if os.path.isdir(inc_dir):
                    include_dirs.append(inc_dir)
            lib_dir = os.path.join(comp_dir, "lib")
            if os.path.isdir(lib_dir):
                library_dirs.append(lib_dir)
    else:
        # Classic /usr/local/cuda toolkit layout.
        include_dirs.append(os.path.join(cu_prefix, "include"))
        for lib_sub in ("lib64", "lib"):
            lib_dir = os.path.join(cu_prefix, lib_sub)
            if os.path.isdir(lib_dir):
                library_dirs.append(lib_dir)

    # Cross-wheel crt/host_defines.h shim — the cu13 wheel ships
    # driver_types.h that includes "crt/host_defines.h" but the wheel
    # lacks the crt/ subdir. Adding /usr/local/cuda/include as a
    # SECONDARY include path satisfies that unconditional include.
    # Harmless on cu12 paths (they ship crt/ in-place).
    for fallback in ("/usr/local/cuda/include",
                     "/usr/local/cuda-12.8/targets/x86_64-linux/include"):
        if os.path.exists(os.path.join(fallback, "crt", "host_defines.h")):
            if fallback not in include_dirs:
                include_dirs.append(fallback)
            break

    # `runtime_library_dirs` bakes the path into the .so's rpath so the
    # dynamic loader finds libcublasLt.so.<N> without needing
    # LD_LIBRARY_PATH preset at import time.
    runtime_library_dirs = list(library_dirs)

    # The extension now includes a .cu kernel (``fused_amax_cast.cu``) so
    # we need nvcc in the loop. Split extra_compile_args by compiler.
    cxx_args = ["-O3", "-std=c++17"]
    # nvcc must match the host C++ standard. The production default is
    # H100 sm_90; scratch build pods can override via CUDA arch list.
    nvcc_args = [
        "-O3",
        "-std=c++17",
        *_nvcc_gencode_args(),
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
    ]
    extra_compile_args = {"cxx": cxx_args, "nvcc": nvcc_args}

    # Link against versioned libs by filename — the nvidia-*-cuNN wheels
    # only ship versioned .so.NN files (no unversioned symlinks). The
    # ``-l:filename`` form is a GCC/GNU-ld extension that asks for a
    # specific file rather than the libfoo.so search.
    suffix = "so.13" if cu_version == 13 else "so.12"
    extra_link_args: List[str] = [
        f"-l:libcublasLt.{suffix}",
        f"-l:libcublas.{suffix}",
        f"-l:libcudart.{suffix}",
    ]

    # Tell PyTorch's BUILD_ABI stays consistent with how torch was built.
    # CUDAExtension injects nvcc into the build and wires up CUDA include
    # dirs; we still need cublasLt headers in include_dirs (nvidia wheel
    # layout).
    ext = CUDAExtension(
        name="chaoscontrol.kernels._cublaslt._C",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
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
    # from this directory. Places the .so next to __init__.py so a plain
    # `python -c "from chaoscontrol.kernels._cublaslt import _C"` finds it.
    from setuptools import setup

    exts = build_ext_modules()
    if not exts:
        raise SystemExit(
            "no extension to build — CUDA toolchain / cublasLt headers missing"
        )
    setup(
        name="chaoscontrol_cublaslt_ext",
        ext_modules=exts,
        cmdclass=cmdclass_with_build_ext(),
    )
