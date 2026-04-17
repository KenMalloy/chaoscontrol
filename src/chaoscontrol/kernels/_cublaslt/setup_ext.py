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
        from torch.utils.cpp_extension import CppExtension
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
    sources = [str(cpp_rel)]

    src_dir = this_dir / "src"
    include_dirs = [str(src_dir)]
    library_dirs: List[str] = []
    # We link against versioned libs directly (``-l:libcublasLt.so.13``)
    # rather than unversioned ``-lcublasLt`` because the nvidia-cublas-cu13
    # wheel ships only the versioned ``.so.13`` files — no unversioned
    # symlinks. The ``-l:filename`` form is a GCC/GNU-ld extension that
    # skips the ``lib<name>.so`` search and asks for the specific file
    # directly. Libraries list is populated below once we know the CUDA
    # layout (cu13 wheel vs. classic /usr/local/cuda).
    libraries: List[str] = []

    # Find cuBLASLt headers. Prefer the cu13 wheel install (which matches
    # torch cu130 ABI) over /usr/local/cuda (which on this pod is cu12.8).
    cu13_candidates = [
        "/usr/local/lib/python3.12/dist-packages/nvidia/cu13",
        "/usr/local/lib/python3.10/dist-packages/nvidia/cu13",
    ]
    cu_prefix = None
    for c in cu13_candidates:
        if os.path.exists(os.path.join(c, "include", "cublasLt.h")):
            cu_prefix = c
            break
    # Honor a manual override (useful if the cu-version wheel changes).
    if os.environ.get("CC_CUBLAS_PREFIX"):
        cu_prefix = os.environ["CC_CUBLAS_PREFIX"]
    if cu_prefix is None and os.path.exists("/usr/local/cuda/include/cublasLt.h"):
        cu_prefix = "/usr/local/cuda"

    if cu_prefix is None:
        # No cuBLASLt headers — skip the extension. Call sites get a
        # clean ImportError from __init__.py.
        return []

    include_dirs.append(os.path.join(cu_prefix, "include"))
    # Library path: cu13 wheel layout is lib/, classic CUDA layout is lib64/.
    for lib_sub in ("lib", "lib64"):
        lib_dir = os.path.join(cu_prefix, lib_sub)
        if os.path.isdir(lib_dir):
            library_dirs.append(lib_dir)

    # The nvidia-*-cu13 wheel (/usr/local/lib/python3.12/dist-packages/nvidia/cu13)
    # ships driver_types.h, which unconditionally does
    #   #include "crt/host_defines.h"
    # — but the wheel is missing the crt/ subdirectory. cu13's host_defines.h
    # lives at the root of include/ instead. Add /usr/local/cuda/include
    # (CUDA toolkit, always cu12.x on these pods) as a SECONDARY include
    # path: cu13 takes precedence for everything except the crt/ shim,
    # which is API-stable across minor CUDA versions. Dropping this line
    # makes the build fail with "crt/host_defines.h: No such file".
    for fallback in ("/usr/local/cuda/include",
                     "/usr/local/cuda-12.8/targets/x86_64-linux/include"):
        if os.path.exists(os.path.join(fallback, "crt", "host_defines.h")):
            include_dirs.append(fallback)
            break

    # `runtime_library_dirs` bakes the path into the .so's rpath so the
    # dynamic loader finds libcublasLt.so.13 without needing LD_LIBRARY_PATH
    # preset at import time.
    runtime_library_dirs = list(library_dirs)

    extra_compile_args = ["-O3", "-std=c++17"]
    # Link directly against versioned libs. On cu13 the wheel is the only
    # thing that ships libcublasLt.so.13, so we pin by filename. If someone
    # later runs this on a box with stock /usr/local/cuda 12.x we'd need
    # to flip to the unversioned ``-lcublasLt``, but that case isn't a
    # pod we currently own.
    extra_link_args: List[str] = [
        "-l:libcublasLt.so.13",
        "-l:libcublas.so.13",
        "-l:libcudart.so.13",
    ]

    # Tell PyTorch's BUILD_ABI stays consistent with how torch was built.
    # CppExtension already appends torch's include dirs + ABI flag.
    ext = CppExtension(
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
