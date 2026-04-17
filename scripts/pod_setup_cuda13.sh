#!/usr/bin/env bash
# CUDA 13 + TransformerEngine 2.13 environment for 8×H100 Parameter Golf pods.
#
# Reproduces the pod state used for Exp 18 Test 10 (2026-04-17):
#   - torch 2.11.0+cu130 (from https://download.pytorch.org/whl/cu130)
#   - transformer_engine[pytorch] 2.13.0 (from PyPI + pypi.nvidia.com)
#   - nvidia-cublas pinned to ==13.4.0.1 so that the cuBLASLt symbols
#     TE 2.13 was built against are actually present. torch 2.11's
#     transitive dep resolves to 13.1.0.3 which is missing
#     `cublasLtGroupedMatrixLayoutInit_internal@@libcublasLt.so.13`
#     (added in a later 13.x cublas release). Observed on 2026-04-17:
#     TE import crashes with "undefined symbol" on 13.1.0.3, works on 13.4.0.1.
#   - sentencepiece, numpy, pytest, chaoscontrol (editable)
#
# Design notes:
#   - --only-binary=:all: on every install: the training pod has no CUDA
#     toolkit / cmake / cc1plus, so source builds must fail loud.
#   - --extra-index-url https://pypi.nvidia.com: nvidia-*-cu13 binary wheels
#     live on NVIDIA's index, not default PyPI. Without this, pip either
#     source-builds (we block that) or falls back to the generic package.
#   - nvidia-cublas is pinned EXPLICITLY to 13.4.0.1. The `nvidia-cublas-cu13`
#     package on pypi.nvidia.com is a 0.0.1 STUB (no library); the real cuBLAS
#     lives in `nvidia-cublas`. Transitive resolution via torch 2.11 picks
#     13.1.0.3, which doesn't yet expose every symbol TE 2.13 references at
#     import time. Pinning 13.4.0.1 is the ABI-matching answer.
#   - Idempotent: if EVERY required dep imports cleanly, skip the whole
#     reinstall. Checking only TE was not enough — a pod with working TE
#     but missing sentencepiece / pytest / editable chaoscontrol install
#     would hit the fast-path, declare "Pod ready," and then immediately
#     fail the first test import. The probe below covers every dep the
#     full install below produces.
#
# Dead-ends we tried that did NOT work — do not copy from shell history:
#   - transformer-engine==1.13 / 1.14 / 2.1 / 2.12 (source builds need
#     cmake + CUDA toolkit headers; pod image has neither)
#   - transformer-engine-cu12 alone (does not provide the 'pytorch' extra)
#   - transformer-engine-cu12 + transformer-engine-torch (torch bundle
#     pulls cu13 libs; fails at runtime on cu12 image)
#   - nvidia-{cusolver,cusparse,curand,cuda-runtime,cuda-nvrtc}-cu13 on
#     PyPI (deprecated, force source build)
#   - --index-url https://download.pytorch.org/whl/cu130 for EVERYTHING
#     (replaces default, nvidia-cublas gets upgraded to the generic
#     version, breaks TE's cublasLt symbol dependency)
#
# Usage:
#   bash scripts/pod_setup_cuda13.sh
#   REPO_ROOT=/workspace/chaoscontrol bash scripts/pod_setup_cuda13.sh

set -euo pipefail

PY_SITEPKG=${PY_SITEPKG:-/usr/local/lib/python3.12/dist-packages}
CU13_LIB=$PY_SITEPKG/nvidia/cu13/lib
REPO_ROOT=${REPO_ROOT:-/workspace/chaoscontrol}

PIP_FLAGS=(--break-system-packages --only-binary=:all:)
NVIDIA_INDEX="--extra-index-url https://pypi.nvidia.com"
PYTORCH_CU130="--extra-index-url https://download.pytorch.org/whl/cu130"

echo "==> checking whether every required dep already works (skip reinstall if so)"
# Probe every dep the full install below produces — torch, TE (with a
# Linear construction smoke to catch broken cublas linking), sentencepiece,
# numpy, pytest, and the editable chaoscontrol package. If ANY fails,
# fall through to the full install. Fast-path declaring "Pod ready" with
# missing deps would only be caught at first test import, wasting pod
# time and obscuring which dep actually went missing.
if python3 - <<'PROBE' 2>/dev/null
import torch
import transformer_engine.pytorch as te
import sentencepiece  # noqa: F401
import numpy  # noqa: F401
import pytest  # noqa: F401
import chaoscontrol  # noqa: F401 — editable install present
# Bespoke cuBLASLt fp8 extension must be built. If missing the C-side
# compiled module, the editable install needs to re-run so setup.py
# builds ext_modules. Import the compiled _C directly rather than the
# wrapper — the wrapper falls back gracefully to an ImportError at
# call time, but here we want the probe to fire.
from chaoscontrol.kernels._cublaslt import _C  # noqa: F401
_ = te.Linear(16, 16, device='cuda' if torch.cuda.is_available() else 'cpu')
PROBE
then
    echo "    torch + TE + sentencepiece + numpy + pytest + chaoscontrol all import;"
    echo "    cuBLASLt fp8 extension importable; TE Linear constructs — skipping reinstall."
    echo "    (force-reinstall by removing one of the above imports from the pod.)"
    echo ""
    echo "Pod ready."
    exit 0
fi
echo "    one or more deps missing/broken; proceeding with full install."

echo "==> 1/5 upgrading pip"
pip install --break-system-packages --upgrade pip

echo "==> 2/5 installing PyTorch 2.11.0 against CUDA 13"
pip install "${PIP_FLAGS[@]}" $PYTORCH_CU130 \
    torch==2.11.0

echo "==> 3/5 pinning nvidia-cublas to the TE-compatible version from NVIDIA index"
# nvidia-cublas must be >=13.4.0.1 so libcublasLt.so.13 has
# cublasLtGroupedMatrixLayoutInit_internal (TE 2.13 imports it at module load).
# The older 13.1.0.3 pulled transitively by torch 2.11 is missing that symbol.
# The cu13-suffixed package name on pypi.nvidia.com is a 0.0.1 stub, NOT the
# library — don't be fooled by its pypi.nvidia.com presence.
pip install "${PIP_FLAGS[@]}" $NVIDIA_INDEX \
    'nvidia-cublas==13.4.0.1' \
    nvidia-cudnn-cu13 \
    nvidia-cusparselt-cu13 \
    nvidia-nccl-cu13 \
    nvidia-nvshmem-cu13

echo "==> 4/5 installing TransformerEngine 2.13.0 (with NVIDIA index for cu13 deps)"
# transformer_engine_torch is an sdist-only pybind11 extension that wraps
# libtransformer_engine (which lives in the prebuilt transformer_engine_cu13
# wheel). No .cu files, no nvcc required — just g++ against TE's C headers.
# Exempt that one package from --only-binary=:all: so the sdist is accepted.
# MAX_JOBS uses every available vCPU for the C++ extension build; the default
# is 1 (minutes per file × N files) and we have no reason to serialize it.
MAX_JOBS=$(nproc) pip install --break-system-packages --only-binary=:all: \
    --no-binary=transformer_engine_torch \
    $NVIDIA_INDEX \
    'transformer-engine[pytorch]==2.13.0'

echo "==> 5/5 registering CUDA 13 lib path for the dynamic loader"
# Without this, libcublas.so.13 and friends aren't found at TE import time.
echo "$CU13_LIB" > /etc/ld.so.conf.d/cuda13.conf
ldconfig

echo "==> installing remaining deps and chaoscontrol (editable)"
pip install "${PIP_FLAGS[@]}" sentencepiece numpy pytest
if [ -f "$REPO_ROOT/pyproject.toml" ] || [ -f "$REPO_ROOT/setup.py" ]; then
    cd "$REPO_ROOT"
    pip install --break-system-packages \
        -e . --no-deps --no-build-isolation
else
    echo "    (skipping: no pyproject.toml/setup.py at $REPO_ROOT)"
fi

echo ""
echo "==> smoke check"
python3 - <<'PY'
import torch
import transformer_engine.pytorch as te
print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  "
      f"GPUs {torch.cuda.device_count()}")

if torch.cuda.is_available():
    from transformer_engine.pytorch import Linear, fp8_autocast
    lin = Linear(16, 16, device="cuda")
    with fp8_autocast():
        x = torch.randn(4, 16, device="cuda")
        y = lin(x)
    print(f"fp8 smoke OK: y.shape={tuple(y.shape)}")

    # Bespoke cuBLASLt fp8 extension smoke — confirms the ext_modules
    # build happened during pip install -e . and the .so loads against
    # the cu13 libcublasLt the training binary will use at runtime.
    from chaoscontrol.kernels._cublaslt import cublaslt_fp8_matmul
    a = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
    scale = torch.tensor(1.0 / 448.0, device="cuda")
    y = cublaslt_fp8_matmul(a, b, scale, scale, None, torch.bfloat16)
    print(f"cublaslt fp8 smoke OK: y.shape={tuple(y.shape)} dtype={y.dtype}")
else:
    print("no CUDA device visible; fp8 smoke skipped")
PY

echo ""
echo "Pod ready. CUDA 13 + TE 2.13 installed."
echo "  tests:        pytest tests/test_train_ssm.py"
echo "  persistent:   python experiments/19_prereqs/run_persistent_launcher.py ..."
