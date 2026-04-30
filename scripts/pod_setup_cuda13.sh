#!/usr/bin/env bash
# CUDA 13 + TransformerEngine 2.13 environment for 8×H100 Parameter Golf pods.
#
# Reproduces the pod state used for Exp 18 Test 10 (2026-04-17) plus the
# 2026-04-18 cu13 upgrade for the bespoke cuBLASLt fp8 extension
# (exp19-phase1a, task 23):
#   - torch 2.11.0+cu130 (from https://download.pytorch.org/whl/cu130)
#   - transformer_engine[pytorch] 2.13.0 (from PyPI + pypi.nvidia.com)
#   - nvidia-cublas pinned to ==13.4.0.1 so that the cuBLASLt symbols
#     TE 2.13 was built against are actually present. torch 2.11's
#     transitive dep resolves to 13.1.0.3 which is missing
#     `cublasLtGroupedMatrixLayoutInit_internal@@libcublasLt.so.13`
#     (added in a later 13.x cublas release). Observed 2026-04-17 and
#     reproduced 2026-04-18: TE import crashes with "undefined symbol"
#     on 13.1.0.3, works on 13.4.0.1.
#     NOTE: BGRADB for fp8 E5M2×E4M3 is STILL rejected by the cublasLt
#     heuristic on 13.4.0.1 (probed at M={64,128,512,1024,2048}) — the
#     Python fallback in src/chaoscontrol/kernels/fp8_linear.py stays.
#   - nvcc/nvvm/cccl on cu13 (installed below) — needed so the bespoke
#     cuBLASLt fp8 extension can build its .cu kernel. The three wheels
#     MUST agree at the same minor version (cicc emits PTX at its own
#     version, ptxas must accept it, cccl headers must match the host
#     API). Observed mismatch: nvcc 13.0 ptxas rejects nvvm 13.2 PTX
#     ("Unsupported .version 9.2; current version is '9.0'"). We install
#     all three at 13.2.x.
#   - libcudart.so symlink inside the cu13 lib dir. The nvidia-cu13
#     wheel ships libcudart.so.13 without the unversioned symlink, and
#     torch.utils.cpp_extension's link step passes both "-lcudart" and
#     "-l:libcudart.so.13". Without the symlink, linking fails with
#     "cannot find -lcudart". The symlink is created idempotently below.
#   - sentencepiece, numpy, pytest, chaoscontrol (editable)
#
# Design notes:
#   - --only-binary=:all: on every install EXCEPT transformer_engine_torch:
#     the training pod has the manylinux toolchain (g++, cmake) but no
#     nvcc, so the TE bindings sdist is safe; everything else should be a
#     prebuilt wheel.
#   - --extra-index-url https://pypi.nvidia.com: nvidia-*-cu13 binary wheels
#     live on NVIDIA's index, not default PyPI. Without this, pip either
#     source-builds (we block that) or falls back to the generic package.
#   - nvidia-cublas is pinned EXPLICITLY to 13.4.0.1. The `nvidia-cublas-cu13`
#     package on pypi.nvidia.com is a 0.0.1 STUB (no library); the real
#     cuBLAS lives in ``nvidia-cublas``. Transitive resolution via torch
#     2.11 → cuda-toolkit 13.0.2 → nvidia-cublas==13.1.0.3.* always wins if
#     we pin before step 4. The pin therefore lives AFTER the TE install:
#     once TE is on disk, no later install drags the 13.1 version back.
#   - Idempotent: if EVERY required dep imports cleanly, skip the whole
#     reinstall. Checking only TE was not enough — a pod with working TE
#     but missing sentencepiece / pytest / editable chaoscontrol install
#     would hit the fast-path, declare "Pod ready," and then immediately
#     fail the first test import. The probe below covers every dep the
#     full install below produces. The CPU SSM controller's CUDA
#     write-event pack is deliberately optional here: it is built only when
#     nvcc's CUDA major.minor matches torch.version.cuda. A mismatched pod
#     should be ready with the CPU-only controller path, not fail setup.
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

# --- venv on /workspace so pip packages survive pod stop/start. ---
# RunPod stop/start wipes the container disk — dist-packages under /usr
# is gone on every restart, which means a fresh pod_setup installing
# into system Python pays the full torch+TE download+compile (~10 min)
# every time. The volume at /workspace IS persistent, so we put pip
# packages there via a venv. Activating an existing venv is O(seconds);
# creating one on first use still pays the full install, but only once.
WORKSPACE_VENV=${WORKSPACE_VENV:-/workspace/venv}
if [ -d "$WORKSPACE_VENV" ]; then
    echo "==> activating existing venv at $WORKSPACE_VENV"
    # shellcheck source=/dev/null
    source "$WORKSPACE_VENV/bin/activate"
else
    echo "==> creating venv at $WORKSPACE_VENV (first-run; persists across restarts)"
    python3 -m venv "$WORKSPACE_VENV"
    # shellcheck source=/dev/null
    source "$WORKSPACE_VENV/bin/activate"
fi

PY_SITEPKG=${PY_SITEPKG:-$WORKSPACE_VENV/lib/python3.12/site-packages}
CU13_LIB=$PY_SITEPKG/nvidia/cu13/lib
REPO_ROOT=${REPO_ROOT:-/workspace/chaoscontrol}

# --break-system-packages is not needed inside a venv — the venv owns
# its own site-packages — but --only-binary is still the right default
# to prevent surprise source builds of deps that aren't transformer_engine_torch.
PIP_FLAGS=(--only-binary=:all:)
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
# builds ext_modules. Import the compiled _C modules directly rather than
# wrappers that may fall back gracefully; readiness should fail if either
# extension is absent.
from chaoscontrol.kernels._cublaslt import _C  # noqa: F401
from chaoscontrol.kernels._ssm_scan import _C as _ssm_scan_C  # noqa: F401
# CPU SSM controller (AMX BF16 backend) — load the C extension. The
# CUDA write-event pack availability is reported in the smoke check below;
# setup_ext.py refuses mismatched nvcc/torch CUDA builds and cleanly builds
# the CPU-only controller in that case.
from chaoscontrol.kernels import _cpu_ssm_controller as _cpu_ext
_ = te.Linear(16, 16, device='cuda' if torch.cuda.is_available() else 'cpu')
PROBE
then
    echo "    torch + TE + sentencepiece + numpy + pytest + chaoscontrol all import;"
    echo "    cuBLASLt fp8 + SSM scan extensions importable; CPU SSM controller loads;"
    echo "    TE Linear constructs — skipping reinstall."
    echo "    (force-reinstall by removing one of the above imports from the pod.)"
    echo ""
    echo "Pod ready."
    exit 0
fi
echo "    one or more deps missing/broken; proceeding with full install."

echo "==> 1/5 upgrading pip"
pip install --upgrade pip

echo "==> 2/5 installing PyTorch 2.11.0 against CUDA 13"
pip install "${PIP_FLAGS[@]}" $PYTORCH_CU130 \
    torch==2.11.0

echo "==> 3/5 installing nvidia cu13 runtime deps (except cublas — pinned in 4b)"
# cudnn / cusparselt / nccl / nvshmem are stable deps; cublas is skipped
# here and force-reinstalled after TE (step 4b) because TE's transitive
# pins drag cublas back to 13.1.0.3 if we set it before the TE install.
pip install "${PIP_FLAGS[@]}" $NVIDIA_INDEX \
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
MAX_JOBS=$(nproc) pip install --only-binary=:all: \
    --no-binary=transformer_engine_torch \
    $NVIDIA_INDEX \
    'transformer-engine[pytorch]==2.13.0'

echo "==> 4b/5 force-pinning nvidia-cublas==13.4.0.1 (post-TE)"
# TE's transitive cuda-toolkit==13.0.2 re-pins nvidia-cublas==13.1.0.3.*,
# which lacks cublasLtGroupedMatrixLayoutInit_internal — TE import crashes.
# Force-reinstall with --no-deps AFTER TE is on disk: no later install
# drags it back. --no-deps is required so we don't trigger another
# cuda-toolkit resolution pass.
pip install "${PIP_FLAGS[@]}" $NVIDIA_INDEX --force-reinstall --no-deps \
    'nvidia-cublas==13.4.0.1'

echo "==> 4c/5 installing nvcc + nvvm + cccl for bespoke ext build"
# The bespoke cuBLASLt fp8 extension ships a .cu kernel (fused_amax_cast),
# so `python setup.py build_ext` needs nvcc and its tool chain. Keep the
# three wheels at the SAME minor version (13.2.x): cicc (from nvvm) emits
# PTX at its own version, ptxas (from nvcc) must accept it, and the cccl
# headers (nv/target, cub/) must match the nvcc host API. Observed on
# 2026-04-18 with nvidia-cuda-nvcc==13.0.88 + nvidia-nvvm==13.2.78 (the
# latter pulled transitively): ptxas rejected "Unsupported .version 9.2".
pip install "${PIP_FLAGS[@]}" $NVIDIA_INDEX \
    'nvidia-cuda-nvcc==13.2.78' \
    'nvidia-nvvm==13.2.78' \
    'nvidia-cuda-cccl==13.2.27'

echo "==> 4e/5 force-pinning runtime stack to match nvcc/cccl 13.2 (post-TE)"
# The cccl headers `cuda/std/__cccl/cuda_toolkit.h` compare the compiler
# major.minor against `CUDART_VERSION` (from cuda_runtime_api.h). Default
# torch 2.11+cu130 pulls nvidia-cuda-runtime==13.0.96 / nvrtc==13.0.88 /
# cupti==13.0.85 transitively. With nvcc/cccl at 13.2.x the version check
# fails and any .cu compile aborts with
#   "CUDA compiler and CUDA toolkit headers are incompatible".
# Force-reinstall these three wheels to the 13.2 line, post-TE so TE's
# transitive resolver can't drag them back. --no-deps avoids retriggering
# any cuda-toolkit umbrella resolution. Observed 2026-04-27 on
# pvapfq2vsyvh0o: write_event_pack.cu silently dropped from the build
# without these pins, producing the $30 cuda_stream_enabled=False
# regression (see docs/reports/2026-04-27-step3-results.md).
pip install "${PIP_FLAGS[@]}" $NVIDIA_INDEX --force-reinstall --no-deps \
    'nvidia-cuda-runtime==13.2.75' \
    'nvidia-cuda-nvrtc==13.2.78' \
    'nvidia-cuda-cupti==13.2.75'

echo "==> 4d/5 creating libcudart.so symlink for torch's link step"
# torch.utils.cpp_extension's link step emits both "-lcudart" (needs
# unversioned .so) and "-l:libcudart.so.13" (versioned, redundant with
# -lcudart). The nvidia-cu13 wheel ships libcudart.so.13 only, so the
# unversioned -lcudart fails with "cannot find -lcudart" unless we add
# the symlink. Idempotent: -sf re-creates if the target changes.
ln -sf libcudart.so.13 "$CU13_LIB/libcudart.so"

echo "==> 5/5 registering CUDA 13 lib path for the dynamic loader"
# Without this, libcublas.so.13 and friends aren't found at TE import time.
echo "$CU13_LIB" > /etc/ld.so.conf.d/cuda13.conf
ldconfig

echo "==> installing remaining deps and chaoscontrol (editable)"
pip install "${PIP_FLAGS[@]}" sentencepiece numpy pytest
if [ -f "$REPO_ROOT/pyproject.toml" ] || [ -f "$REPO_ROOT/setup.py" ]; then
    cd "$REPO_ROOT"
    # The editable install triggers setup.py build_ext for the bespoke
    # cuBLASLt extension, which shells out to nvcc. Prepend the cu13
    # nvcc bin dir to PATH and point CUDA_HOME at the cu13 umbrella so
    # nvcc finds cccl headers and its own ptxas. MAX_JOBS fans the 9 TUs
    # out across ninja workers instead of compiling them serially.
    # The CPU SSM controller build auto-enables write_event_pack.cu only
    # when nvcc's CUDA major.minor matches torch.version.cuda. Do not force
    # CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT=1 here: PyTorch cu130 plus a
    # 13.2 toolkit is a valid setup, but forcing the pack would correctly
    # fail because the CUDA ABIs do not match.
    # NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK is a
    # belt-and-suspenders against the cccl header version check —
    # step 4e's runtime pins should make it unnecessary, but the bypass
    # is documented in cccl's own header as the supported escape hatch.
    MAX_JOBS=$(nproc) \
    PATH="$CU13_LIB/../bin:$PATH" \
    CUDA_HOME="$CU13_LIB/.." \
    NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK \
        pip install \
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
    # TE's FP8 shape assertion needs prod(shape[:-1]) % 8 == 0 and
    # shape[-1] % 16 == 0, so pick 16x32 to clear both bars.
    lin = Linear(32, 32, device="cuda")
    with fp8_autocast():
        x = torch.randn(16, 32, device="cuda")
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

    from chaoscontrol.kernels._ssm_scan import ssm_scan_forward
    decay = torch.full((2, 4, 8), 0.9, device="cuda", dtype=torch.float32)
    update = torch.randn(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    y = ssm_scan_forward(decay, update)
    print(f"ssm scan smoke OK: y.shape={tuple(y.shape)} dtype={y.dtype}")

    from chaoscontrol.kernels import _cpu_ssm_controller as cpu_ext
    print(
        "cpu ssm controller smoke OK: "
        f"cuda_write_event_pack={cpu_ext.write_event_cuda_pack_available()}"
    )
else:
    print("no CUDA device visible; fp8 smoke skipped")
PY

echo ""
echo "Pod ready. CUDA 13 + TE 2.13 installed."
echo "  tests:        pytest tests/test_train_ssm.py"
echo "  persistent:   python experiments/19_prereqs/run_persistent_launcher.py ..."
