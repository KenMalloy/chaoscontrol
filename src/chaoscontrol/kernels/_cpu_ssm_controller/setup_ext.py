"""Build hook for the CPU SSM controller reference extension."""
from __future__ import annotations

from pathlib import Path


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
    return [
        CppExtension(
            name="chaoscontrol.kernels._cpu_ssm_controller._C",
            sources=[
                str(cpp_rel),
                str(controller_main_rel),
                str(event_handlers_rel),
                str(action_history_rel),
                str(credit_rel),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
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
