"""Build hook for the Exp23 LM-head/loss native helper extension."""
from __future__ import annotations

from pathlib import Path


def _nvcc_gencode_args(default_arch_list: str = "9.0") -> list[str]:
    import os

    raw = (
        os.environ.get("CHAOSCONTROL_CUDA_ARCH_LIST")
        or os.environ.get("TORCH_CUDA_ARCH_LIST")
        or default_arch_list
    )
    args: list[str] = []
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
    try:
        from torch.utils.cpp_extension import CUDAExtension
    except ImportError:
        return []

    import shutil

    if shutil.which("nvcc") is None:
        return []

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[3]
    cpp_rel = this_dir.relative_to(repo_root) / "src" / "rms_norm_binding.cpp"
    cu_rel = this_dir.relative_to(repo_root) / "src" / "rms_norm.cu"
    sources = [str(cpp_rel), str(cu_rel)]

    src_dir = this_dir / "src"
    cxx_args = ["-O3", "-std=c++17"]
    nvcc_args = [
        "-O3",
        "-std=c++17",
        *_nvcc_gencode_args(),
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        "-use_fast_math",
        "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
    ]

    return [
        CUDAExtension(
            name="chaoscontrol.kernels._lm_head_loss._C",
            sources=sources,
            include_dirs=[str(src_dir)],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        )
    ]


def cmdclass_with_build_ext() -> dict:
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError:
        return {}
    return {"build_ext": BuildExtension}


if __name__ == "__main__":
    from setuptools import setup

    exts = build_ext_modules()
    if not exts:
        raise SystemExit("no extension to build - nvcc / CUDA toolchain missing")
    setup(
        name="chaoscontrol_lm_head_loss_ext",
        ext_modules=exts,
        cmdclass=cmdclass_with_build_ext(),
    )
