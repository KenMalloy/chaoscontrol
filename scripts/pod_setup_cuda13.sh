#!/usr/bin/env bash
# CUDA 13 + TransformerEngine 2.13 environment for 8×H100 Parameter Golf pods.
#
# Reproduces the pod state used for Exp 18 Test 10 (2026-04-17):
#   - torch 2.11.0+cu130 (from https://download.pytorch.org/whl/cu130)
#   - transformer_engine[pytorch] 2.13.0 (from PyPI + pypi.nvidia.com)
#   - nvidia-cublas-cu13 pinned to avoid the generic nvidia-cublas package
#     (which has a newer cublasLt ABI that's missing symbols TE was built
#     against — observed `cublasLtGroupedMatrixLayoutInit_internal` undefined
#     when pip resolved to the generic)
#   - sentencepiece, numpy, pytest, chaoscontrol (editable)
#
# Design notes:
#   - --only-binary=:all: on every install: the training pod has no CUDA
#     toolkit / cmake / cc1plus, so source builds must fail loud.
#   - --extra-index-url https://pypi.nvidia.com: nvidia-*-cu13 binary wheels
#     live on NVIDIA's index, not default PyPI. Without this, pip either
#     source-builds (we block that) or falls back to the generic package.
#   - nvidia-cublas-cu13 is pinned EXPLICITLY because TE's pre-built
#     libtransformer_engine.so is compiled against its specific cublasLt
#     ABI. Letting pip resolve nvidia-cublas (generic, newer) breaks TE.
#   - Idempotent: if TE already imports cleanly, skip the whole reinstall.
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

echo "==> checking whether TE already works (skip reinstall if so)"
if python3 -c "import transformer_engine.pytorch as te; import torch; \
    lin = te.Linear(16, 16, device='cuda' if torch.cuda.is_available() else 'cpu'); \
    print('TE OK')" 2>/dev/null | grep -q "TE OK"; then
    echo "    TE imports and constructs a Linear — skipping reinstall."
    echo "    (delete the import or force-reinstall if you need a clean rebuild.)"
    echo ""
    echo "Pod ready."
    exit 0
fi
echo "    TE not working; proceeding with install."

echo "==> 1/5 upgrading pip"
pip install --break-system-packages --upgrade pip

echo "==> 2/5 installing PyTorch 2.11.0 against CUDA 13"
pip install "${PIP_FLAGS[@]}" $PYTORCH_CU130 \
    torch==2.11.0

echo "==> 3/5 pinning nvidia-cublas-cu13 from NVIDIA index (TE ABI requirement)"
# Must be pinned explicitly so pip doesn't upgrade to nvidia-cublas (generic,
# newer, missing cublasLtGroupedMatrixLayoutInit_internal that TE expects).
pip install "${PIP_FLAGS[@]}" $NVIDIA_INDEX \
    nvidia-cublas-cu13 \
    nvidia-cudnn-cu13 \
    nvidia-cusparselt-cu13 \
    nvidia-nccl-cu13 \
    nvidia-nvshmem-cu13

echo "==> 4/5 installing TransformerEngine 2.13.0 (with NVIDIA index for cu13 deps)"
pip install "${PIP_FLAGS[@]}" $NVIDIA_INDEX \
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
else:
    print("no CUDA device visible; fp8 smoke skipped")
PY

echo ""
echo "Pod ready. CUDA 13 + TE 2.13 installed."
echo "  tests:        pytest tests/test_train_ssm.py"
echo "  persistent:   python experiments/19_prereqs/run_persistent_launcher.py ..."
