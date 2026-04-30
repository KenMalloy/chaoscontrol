"""Build hook for the CPU SSM controller reference extension."""
from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path


def _x86_accel_compile_args() -> list[str]:
    # KNOWN-DEBT: distutils' Extension applies extra_compile_args at the
    # extension level, not per-source. The -m flags below are visible to
    # every translation unit in this extension, which means the compiler
    # is free to auto-vectorize unrelated TUs (action_history.cpp,
    # optimizer.cpp, etc.) using AVX-512 / AMX instructions. Practical
    # fallout: the resulting .so will SIGILL on x86 hosts that lack
    # AVX-512 even when the kernel paths themselves are gated off at
    # runtime. The right fix is either (a) split into a "core" extension
    # (no -m flags) and an "accel" extension (with), linking the latter
    # as a static lib; or (b) replace the flags with #pragma GCC target
    # scoped inside the AVX-512/AMX kernel files. We deploy only on
    # Sapphire Rapids today, so this is theoretical for now — but anyone
    # adding a non-AVX-512 x86 deployment target must address it first.
    machine = platform.machine().lower()
    requested = os.environ.get("CHAOSCONTROL_CPU_SSM_X86_ACCEL") == "1"
    if not requested or machine not in {"x86_64", "amd64"}:
        return []
    return [
        "-DCHAOSCONTROL_CPU_SSM_AMX_BF16_KERNEL=1",
        "-DCHAOSCONTROL_CPU_SSM_AVX512_KERNEL=1",
        "-mamx-tile",
        "-mamx-bf16",
        "-mavx512f",
    ]


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cuda_major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    match = re.search(r"(\d+)\.(\d+)", str(version))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _torch_cuda_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return getattr(getattr(torch, "version", None), "cuda", None)


def _nvcc_cuda_version(cuda_home: str | os.PathLike[str]) -> str | None:
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    if not nvcc.exists():
        return None
    try:
        out = subprocess.run(
            [str(nvcc), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    text = f"{out.stdout}\n{out.stderr}"
    match = re.search(r"release\s+(\d+\.\d+)", text)
    if match is not None:
        return match.group(1)
    return None


def _cuda_toolkit_matches_torch(cuda_home: str | os.PathLike[str]) -> bool:
    torch_cuda = _cuda_major_minor(_torch_cuda_version())
    nvcc_cuda = _cuda_major_minor(_nvcc_cuda_version(cuda_home))
    return torch_cuda is not None and nvcc_cuda is not None and torch_cuda == nvcc_cuda


def _cuda_write_event_enabled() -> bool:
    requested = os.environ.get("CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT")
    if requested is not None:
        requested_enabled = _env_bool(requested)
        if not requested_enabled:
            return False
    else:
        requested_enabled = False
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        return False
    if CUDA_HOME is None:
        if requested_enabled:
            raise RuntimeError(
                "CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT=1 was requested, but "
                "torch.utils.cpp_extension.CUDA_HOME is not set."
            )
        return False
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc.exists():
        if requested_enabled:
            raise RuntimeError(
                "CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT=1 was requested, but "
                f"nvcc was not found under CUDA_HOME={CUDA_HOME!r}."
            )
        return False
    if not _cuda_toolkit_matches_torch(CUDA_HOME):
        if requested_enabled:
            raise RuntimeError(
                "CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT=1 was requested, but "
                f"CUDA_HOME={CUDA_HOME!r} does not match torch.version.cuda="
                f"{_torch_cuda_version()!r}; refusing to build the CUDA "
                "WriteEvent packer against a mismatched toolkit."
            )
        return False
    return True


def build_ext_modules() -> list:
    try:
        from torch.utils.cpp_extension import CppExtension
    except ImportError:
        return []

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[3]
    cpp_rel = this_dir.relative_to(repo_root) / "src" / "cpu_ssm_controller.cpp"
    controller_main_rel = (
        this_dir.relative_to(repo_root) / "src" / "controller_main.cpp"
    )
    event_handlers_rel = (
        this_dir.relative_to(repo_root) / "src" / "event_handlers.cpp"
    )
    action_history_rel = (
        this_dir.relative_to(repo_root) / "src" / "action_history.cpp"
    )
    credit_rel = this_dir.relative_to(repo_root) / "src" / "credit.cpp"
    cpu_features_rel = (
        this_dir.relative_to(repo_root) / "src" / "cpu_features.cpp"
    )
    amx_matmul_rel = this_dir.relative_to(repo_root) / "src" / "amx_matmul.cpp"
    avx512_recurrence_rel = (
        this_dir.relative_to(repo_root) / "src" / "avx512_recurrence.cpp"
    )
    avx512_matops_rel = (
        this_dir.relative_to(repo_root) / "src" / "avx512_matops.cpp"
    )
    optimizer_rel = this_dir.relative_to(repo_root) / "src" / "optimizer.cpp"
    online_learning_rel = (
        this_dir.relative_to(repo_root) / "src" / "online_learning.cpp"
    )
    simplex_policy_rel = (
        this_dir.relative_to(repo_root) / "src" / "simplex_policy.cpp"
    )
    simplex_learner_rel = (
        this_dir.relative_to(repo_root) / "src" / "simplex_learner.cpp"
    )
    write_event_pack_cuda_rel = (
        this_dir.relative_to(repo_root) / "src" / "write_event_pack.cu"
    )
    sources = [
        str(cpp_rel),
        str(controller_main_rel),
        str(event_handlers_rel),
        str(action_history_rel),
        str(credit_rel),
        str(cpu_features_rel),
        str(amx_matmul_rel),
        str(avx512_recurrence_rel),
        str(avx512_matops_rel),
        str(optimizer_rel),
        str(online_learning_rel),
        str(simplex_policy_rel),
        str(simplex_learner_rel),
    ]
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17", *_x86_accel_compile_args()],
    }
    extension_cls = CppExtension
    if _cuda_write_event_enabled():
        try:
            from torch.utils.cpp_extension import CUDAExtension
        except ImportError:
            return []
        extension_cls = CUDAExtension
        sources.append(str(write_event_pack_cuda_rel))
        extra_compile_args["cxx"].append(
            "-DCHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT_KERNEL=1"
        )
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT_KERNEL=1",
        ]
    return [
        extension_cls(
            name="chaoscontrol.kernels._cpu_ssm_controller._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]


def cmdclass_with_build_ext() -> dict:
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError:
        return {}
    return {"build_ext": BuildExtension}


if __name__ == "__main__":  # pragma: no cover - manual build path
    from setuptools import setup

    setup(
        name="chaoscontrol_cpu_ssm_controller_ext",
        ext_modules=build_ext_modules(),
        cmdclass=cmdclass_with_build_ext(),
    )
