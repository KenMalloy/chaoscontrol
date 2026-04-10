#!/usr/bin/env bash
# =============================================================================
# pod_bootstrap.sh — Bootstrap a RunPod GPU instance for ChaosControl experiments
#
# Run ON the pod after rsyncing the repo:
#   bash /workspace/chaoscontrol/tools/pod_bootstrap.sh
#
# Idempotent: safe to run multiple times. Skips steps that are already done.
# Expects: NVIDIA GPU(s), Ubuntu-based RunPod image, network volume at /workspace
# =============================================================================
set -euo pipefail

REPO="/workspace/chaoscontrol"
DATA="/workspace/data"
ENWIK8_URL="http://mattmahoney.net/dc/enwik8.zip"

echo "============================================"
echo "ChaosControl pod bootstrap"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. GPU / CUDA check
# ---------------------------------------------------------------------------
echo ""
echo "=== GPU / CUDA info ==="
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is this a GPU pod?"
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Driver version: $CUDA_VERSION"

# Check available CUDA toolkit version
if command -v nvcc &>/dev/null; then
    echo "nvcc version: $(nvcc --version | grep release | awk '{print $NF}')"
fi

# ---------------------------------------------------------------------------
# 2. Python environment (system Python, no venv — RunPod images come with pip)
# ---------------------------------------------------------------------------
echo ""
echo "=== Python environment ==="

PYTHON="python3"
PIP="pip3"

echo "Python: $($PYTHON --version 2>&1)"

# ---------------------------------------------------------------------------
# 3. Install PyTorch — auto-detect CUDA version, code is CUDA-agnostic
# ---------------------------------------------------------------------------
echo ""
echo "=== Installing PyTorch ==="

# Check if torch is already installed with CUDA support
TORCH_OK=false
if $PYTHON -c "import torch; assert torch.cuda.is_available(); print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null; then
    echo "PyTorch with CUDA already installed, skipping"
    TORCH_OK=true
fi

if [ "$TORCH_OK" = false ]; then
    # Let pip pick the right CUDA build for whatever the pod has
    echo "Installing PyTorch (auto-detect CUDA)..."
    $PIP install torch
fi

# ---------------------------------------------------------------------------
# 4. Install mamba-ssm (needs causal-conv1d first)
# ---------------------------------------------------------------------------
echo ""
echo "=== Installing mamba-ssm ==="

if $PYTHON -c "import mamba_ssm; print(f'mamba-ssm {mamba_ssm.__version__}')" 2>/dev/null; then
    echo "mamba-ssm already installed, skipping"
else
    echo "Installing causal-conv1d (mamba-ssm dependency)..."
    $PIP install causal-conv1d
    echo "Installing mamba-ssm..."
    $PIP install mamba-ssm
fi

# ---------------------------------------------------------------------------
# 5. Install project dependencies + chaoscontrol in dev mode
# ---------------------------------------------------------------------------
echo ""
echo "=== Installing project dependencies ==="

$PIP install pyyaml numpy

if [ -d "$REPO" ]; then
    echo "Installing chaoscontrol in dev mode..."
    $PIP install -e "$REPO"
else
    echo "WARNING: $REPO not found. rsync the repo first."
    echo "  python tools/runpod.py deploy <pod_id>"
fi

# ---------------------------------------------------------------------------
# 6. Download enwik8 if not present
# ---------------------------------------------------------------------------
echo ""
echo "=== Data: enwik8 ==="

mkdir -p "$DATA"

if [ -f "$DATA/enwik8" ]; then
    SIZE=$(stat -c%s "$DATA/enwik8" 2>/dev/null || stat -f%z "$DATA/enwik8" 2>/dev/null)
    echo "enwik8 already present ($SIZE bytes), skipping download"
else
    echo "Downloading enwik8..."
    cd "$DATA"
    wget -q --show-progress "$ENWIK8_URL" -O enwik8.zip
    unzip -o enwik8.zip
    rm -f enwik8.zip
    SIZE=$(stat -c%s "$DATA/enwik8" 2>/dev/null || stat -f%z "$DATA/enwik8" 2>/dev/null)
    echo "Downloaded enwik8: $SIZE bytes"
    cd "$REPO"
fi

# Validate enwik8 size (should be exactly 100,000,000 bytes)
EXPECTED_SIZE=100000000
ACTUAL_SIZE=$(stat -c%s "$DATA/enwik8" 2>/dev/null || stat -f%z "$DATA/enwik8" 2>/dev/null)
if [ "$ACTUAL_SIZE" -ne "$EXPECTED_SIZE" ]; then
    echo "WARNING: enwik8 size mismatch. Expected $EXPECTED_SIZE, got $ACTUAL_SIZE"
else
    echo "enwik8 validated: $ACTUAL_SIZE bytes (OK)"
fi

# ---------------------------------------------------------------------------
# 7. Create directory structure
# ---------------------------------------------------------------------------
echo ""
echo "=== Directory structure ==="

mkdir -p /workspace/results
mkdir -p /workspace/checkpoints
echo "  /workspace/chaoscontrol/  (code)"
echo "  /workspace/data/          (datasets)"
echo "  /workspace/results/       (experiment outputs)"
echo "  /workspace/checkpoints/   (model checkpoints)"

# ---------------------------------------------------------------------------
# 8. Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "=== Smoke test ==="

$PYTHON -c "
import torch
import chaoscontrol

gpu_count = torch.cuda.device_count()
print(f'{gpu_count} GPU(s) detected')
for i in range(gpu_count):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')

print()
print(f'PyTorch {torch.__version__}')
print(f'CUDA {torch.version.cuda}')
print(f'chaoscontrol package: OK')
"

# Test that mamba import works (optional dependency)
if $PYTHON -c "import mamba_ssm; print(f'mamba-ssm {mamba_ssm.__version__}: OK')" 2>/dev/null; then
    true
else
    echo "WARNING: mamba-ssm import failed (optional — needed only for mamba2 model_type)"
fi

# ---------------------------------------------------------------------------
# 9. Batch size recommendation
# ---------------------------------------------------------------------------
echo ""
echo "=== Batch size recommendation ==="

$PYTHON -c "
import torch

gpu_count = torch.cuda.device_count()
if gpu_count == 0:
    print('No GPUs detected, cannot recommend batch size')
else:
    mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    # Heuristic: ~1GB per 64 batch size for dim=128, seq_len=256
    # Scale linearly with VRAM, conservatively
    recommended = int((mem_gb / 1.5) * 64)
    # Round down to nearest power of 2 for efficiency
    recommended = 2 ** int(recommended).bit_length() // 2
    recommended = max(32, min(recommended, 512))
    print(f'GPU VRAM: {mem_gb:.1f} GB x {gpu_count} GPU(s)')
    print(f'Recommended batch_size: {recommended} (per GPU, for dim=128 seq_len=256)')
    print(f'For larger models (dim=384+), halve the batch size.')
"

# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Bootstrap complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"
echo ""
echo "Data path:  $DATA/enwik8"
echo "Repo path:  $REPO"
echo ""
echo "Run experiments with:"
echo "  cd $REPO"
echo "  python experiments/09_revised_architecture/run_layered.py --data-path $DATA/enwik8 --budget 600 --num-gpus \$(nvidia-smi -L | wc -l)"
echo ""
