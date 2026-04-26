"""Setuptools entry point — exists purely so ``pip install -e .`` can build
our in-tree C++/CUDA extensions alongside the pure-Python package defined
in ``pyproject.toml``.

Metadata (name, version, dependencies) lives in ``pyproject.toml`` and is
intentionally NOT duplicated here. Setuptools merges the two: fields from
``pyproject.toml``'s ``[project]`` table win, and this file only adds
``ext_modules`` + ``cmdclass``.

Extensions built here (concatenated in the order listed):
  * ``chaoscontrol.kernels._cublaslt._C`` — bespoke cuBLASLt fp8 matmul
    (build hook at ``src/chaoscontrol/kernels/_cublaslt/setup_ext.py``).
  * ``chaoscontrol.kernels._lm_head_loss._C`` — native LM-head/loss
    helper kernels (build hook at
    ``src/chaoscontrol/kernels/_lm_head_loss/setup_ext.py``).
  * ``chaoscontrol.kernels._ssm_scan._C`` — diag SSM scan kernel
    (build hook at ``src/chaoscontrol/kernels/_ssm_scan/setup_ext.py``).
  * ``chaoscontrol.kernels._cpu_ssm_controller._C`` — CPU reference runtime
    for the learned episodic controller.

If an extension's toolchain prerequisites aren't met (dev mac without
CUDA, missing nvidia-cu13 headers, missing nvcc, etc.) its
``build_ext_modules()`` returns an empty list and the install proceeds
without that extension. Each kernel module's ``__init__.py`` handles
the missing-extension case at call time with a clear ImportError.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from setuptools import setup

_KERNELS_DIR = Path(__file__).resolve().parent / "src" / "chaoscontrol" / "kernels"


def _load_build_hook(name: str):
    """Load ``kernels/<name>/setup_ext.py`` as an anonymous module.

    The two ``setup_ext.py`` modules share identical names. A plain
    ``sys.path.insert`` trick only loads whichever we add first, so we
    import each under a private alias with ``importlib``.
    """
    path = _KERNELS_DIR / name / "setup_ext.py"
    spec = importlib.util.spec_from_file_location(
        f"_chaoscontrol_build_hook_{name}", path
    )
    assert spec is not None and spec.loader is not None, (
        f"failed to locate build hook at {path}"
    )
    module = importlib.util.module_from_spec(spec)
    # Register so relative imports inside the hook (none today, but
    # defensive) resolve cleanly.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_cublaslt_hook = _load_build_hook("_cublaslt")
_lm_head_loss_hook = _load_build_hook("_lm_head_loss")
_ssm_scan_hook = _load_build_hook("_ssm_scan")
_cpu_ssm_controller_hook = _load_build_hook("_cpu_ssm_controller")

_ext_modules = (
    _cublaslt_hook.build_ext_modules()
    + _lm_head_loss_hook.build_ext_modules()
    + _ssm_scan_hook.build_ext_modules()
    + _cpu_ssm_controller_hook.build_ext_modules()
)
# Both hooks return the same cmdclass (torch's BuildExtension); pick one.
_cmdclass = (
    _cublaslt_hook.cmdclass_with_build_ext()
    or _lm_head_loss_hook.cmdclass_with_build_ext()
    or _ssm_scan_hook.cmdclass_with_build_ext()
    or _cpu_ssm_controller_hook.cmdclass_with_build_ext()
)

setup(
    ext_modules=_ext_modules,
    cmdclass=_cmdclass,
)
