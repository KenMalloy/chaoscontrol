"""Build hook for the CPU SSM controller reference extension."""
from __future__ import annotations

import os
import platform
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
    optimizer_rel = this_dir.relative_to(repo_root) / "src" / "optimizer.cpp"
    online_learning_rel = (
        this_dir.relative_to(repo_root) / "src" / "online_learning.cpp"
    )
    return [
        CppExtension(
            name="chaoscontrol.kernels._cpu_ssm_controller._C",
            sources=[
                str(cpp_rel),
                str(controller_main_rel),
                str(event_handlers_rel),
                str(action_history_rel),
                str(credit_rel),
                str(cpu_features_rel),
                str(amx_matmul_rel),
                str(avx512_recurrence_rel),
                str(optimizer_rel),
                str(online_learning_rel),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", *_x86_accel_compile_args()],
            },
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
