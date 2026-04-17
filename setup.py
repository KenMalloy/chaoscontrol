"""Setuptools entry point — exists purely so ``pip install -e .`` can build
the cuBLASLt fp8 C++ extension alongside the pure-Python package defined
in ``pyproject.toml``.

Metadata (name, version, dependencies) lives in ``pyproject.toml`` and is
intentionally NOT duplicated here. Setuptools merges the two: fields from
``pyproject.toml``'s ``[project]`` table win, and this file only adds
``ext_modules`` + ``cmdclass``.

If the cuBLASLt extension cannot be built (dev mac without CUDA, missing
nvidia-cu13 headers, etc.) ``build_ext_modules()`` returns an empty list
and the install proceeds as pure-Python. The kernel module's
``__init__.py`` handles the missing-extension case at call time with a
clear ImportError.
"""
from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup

# Make the in-tree kernel's setup_ext importable without installing
# anything first (classic chicken-and-egg for editable installs).
_KERNEL_DIR = Path(__file__).resolve().parent / "src" / "chaoscontrol" / "kernels" / "_cublaslt"
sys.path.insert(0, str(_KERNEL_DIR))

from setup_ext import build_ext_modules, cmdclass_with_build_ext  # noqa: E402

setup(
    ext_modules=build_ext_modules(),
    cmdclass=cmdclass_with_build_ext(),
)
