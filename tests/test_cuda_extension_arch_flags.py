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


def test_cpu_ssm_controller_cuda_write_event_auto_enables_with_nvcc(
    monkeypatch, tmp_path
):
    module = _load_setup_ext(
        "src/chaoscontrol/kernels/_cpu_ssm_controller/setup_ext.py"
    )
    cuda_home = tmp_path / "cuda"
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n")
    monkeypatch.delenv("CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT", raising=False)

    import torch.utils.cpp_extension as cpp_extension

    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(cuda_home))

    assert module._cuda_write_event_enabled() is True


def test_cpu_ssm_controller_cuda_write_event_env_can_disable(
    monkeypatch, tmp_path
):
    module = _load_setup_ext(
        "src/chaoscontrol/kernels/_cpu_ssm_controller/setup_ext.py"
    )
    monkeypatch.setenv("CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT", "0")

    assert module._cuda_write_event_enabled() is False
