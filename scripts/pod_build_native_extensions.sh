#!/usr/bin/env bash
# Build ChaosControl native extensions on a GPU pod.
#
# This is the narrow, repeatable version of the manual H100 final-suite setup:
# use the pod image's CUDA-compatible torch, put nvcc on PATH, build the
# extension entry points from the repo root, then smoke-import the native paths.
#
# Usage:
#   cd /workspace/chaoscontrol
#   bash scripts/pod_build_native_extensions.sh
#
# Useful overrides:
#   REPO_ROOT=/workspace/chaoscontrol
#   WORKSPACE_VENV=/workspace/venv
#   CUDA_HOME=/usr/local/cuda-12.8
#   CHAOSCONTROL_CUDA_ARCH_LIST=9.0
#   CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT=0|1
#
# The CPU SSM controller auto-enables write_event_pack.cu only when nvcc's
# CUDA major.minor matches torch.version.cuda. Leave
# CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT unset for the normal pod path: matched
# toolkits build the pack, mismatched toolkits build the explicit CPU-only
# controller. Set it to 1 only when you want a fail-fast assertion that the
# toolkit matches PyTorch.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/chaoscontrol}
WORKSPACE_VENV=${WORKSPACE_VENV:-/workspace/venv}

if [ ! -d "$REPO_ROOT" ]; then
    echo "ERROR: REPO_ROOT does not exist: $REPO_ROOT" >&2
    exit 1
fi

if [ ! -f "$WORKSPACE_VENV/bin/python" ]; then
    echo "==> creating venv at $WORKSPACE_VENV with system site packages"
    python3 -m venv --system-site-packages "$WORKSPACE_VENV"
fi

# shellcheck source=/dev/null
source "$WORKSPACE_VENV/bin/activate"

cd "$REPO_ROOT"

echo "==> ensuring build/test Python deps"
python -m pip install -q --upgrade pip
python -m pip install -q pytest numpy pyyaml sentencepiece ninja

TORCH_CUDA=$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)

if [ -z "${CUDA_HOME:-}" ]; then
    if [[ "$TORCH_CUDA" == 13* ]]; then
        PY_SITEPKG=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
        CUDA_HOME="$PY_SITEPKG/nvidia/cu13"
    elif [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
        CUDA_HOME=/usr/local/cuda-12.8
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDA_HOME=/usr/local/cuda
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
    else
        echo "ERROR: nvcc not found. Set CUDA_HOME to a CUDA toolkit root." >&2
        exit 1
    fi
fi

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: nvcc not executable at $CUDA_HOME/bin/nvcc" >&2
    exit 1
fi

export CUDA_HOME
export PATH="$WORKSPACE_VENV/bin:$CUDA_HOME/bin:$PATH"

if [ -z "${CHAOSCONTROL_CUDA_ARCH_LIST:-}" ]; then
    CHAOSCONTROL_CUDA_ARCH_LIST=$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)
fi
export CHAOSCONTROL_CUDA_ARCH_LIST
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$CHAOSCONTROL_CUDA_ARCH_LIST}

if [ -z "${CHAOSCONTROL_CPU_SSM_X86_ACCEL:-}" ]; then
    if grep -qw avx512f /proc/cpuinfo 2>/dev/null; then
        export CHAOSCONTROL_CPU_SSM_X86_ACCEL=1
    else
        export CHAOSCONTROL_CPU_SSM_X86_ACCEL=0
    fi
fi

echo "==> native extension build config"
echo "    repo:        $REPO_ROOT"
echo "    python:      $(python --version 2>&1)"
echo "    torch cuda:  $TORCH_CUDA"
echo "    CUDA_HOME:   $CUDA_HOME"
echo "    arch list:   $CHAOSCONTROL_CUDA_ARCH_LIST"
echo "    x86 accel:   $CHAOSCONTROL_CPU_SSM_X86_ACCEL"
echo "    write pack:  ${CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT:-auto}"
python - <<'PY'
import torch
print(f"    cuda ok:     {torch.cuda.is_available()} ({torch.cuda.device_count()} device(s))")
if torch.cuda.is_available():
    print(f"    gpu[0]:      {torch.cuda.get_device_name(0)}")
PY

echo "==> building _lm_head_loss"
python src/chaoscontrol/kernels/_lm_head_loss/setup_ext.py build_ext --inplace

echo "==> building _cpu_ssm_controller"
python src/chaoscontrol/kernels/_cpu_ssm_controller/setup_ext.py build_ext --inplace

echo "==> building _ssm_scan"
python src/chaoscontrol/kernels/_ssm_scan/setup_ext.py build_ext --inplace

echo "==> native extension smoke check"
python - <<'PY'
import torch
from chaoscontrol.kernels._lm_head_loss import _C as lm_head_C  # noqa: F401
from chaoscontrol.kernels._ssm_scan import _C as ssm_scan_C  # noqa: F401
from chaoscontrol.kernels import _cpu_ssm_controller as cpu_ext

assert torch.cuda.is_available(), "CUDA is not visible after extension build"
print("    _lm_head_loss:         OK")
print("    _ssm_scan:             OK")
print("    cpu_ssm_controller:    OK")
print(f"    cuda write-event pack: {cpu_ext.write_event_cuda_pack_available()}")
print(f"    AMX BF16 available:    {getattr(cpu_ext, 'has_amx_bf16', lambda: False)()}")
PY

echo "Native extensions ready."
