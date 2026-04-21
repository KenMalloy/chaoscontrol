from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_setup_ext(rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem + "_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ssm_scan_build_hook_honors_cuda_arch_env(monkeypatch):
    module = _load_setup_ext("src/chaoscontrol/kernels/_ssm_scan/setup_ext.py")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9;9.0")

    assert module._nvcc_gencode_args() == [
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]


def test_cublaslt_build_hook_honors_cuda_arch_env(monkeypatch):
    module = _load_setup_ext("src/chaoscontrol/kernels/_cublaslt/setup_ext.py")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9;9.0")

    assert module._nvcc_gencode_args() == [
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]


def test_cuda_arch_env_supports_ptx_suffix(monkeypatch):
    module = _load_setup_ext("src/chaoscontrol/kernels/_ssm_scan/setup_ext.py")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0+PTX")

    assert module._nvcc_gencode_args() == [
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_90,code=compute_90",
    ]
