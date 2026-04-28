#!/usr/bin/env bash
# =============================================================================
# pod_bootstrap.sh — Bootstrap a RunPod GPU instance for ChaosControl experiments
#
# Legacy broad bootstrap path. It mutates the Python environment, may install
# Torch/Mamba dependencies, and runs extra benchmarks. Prefer the active
# one-command setup in scripts/pod_bootstrap.sh, or the native-extension-only
# helper in scripts/pod_build_native_extensions.sh for test pods.
#
# Run ON the pod only when you intentionally want the broad path:
#   CHAOSCONTROL_ALLOW_BROAD_BOOTSTRAP=1 bash /workspace/chaoscontrol/tools/pod_bootstrap.sh
#
# Idempotent: safe to run multiple times. Skips steps that are already done.
# Expects: NVIDIA GPU(s), Ubuntu-based RunPod image, network volume at /workspace
# =============================================================================
set -euo pipefail

if [ "${CHAOSCONTROL_ALLOW_BROAD_BOOTSTRAP:-}" != "1" ]; then
    cat >&2 <<'EOF'
ERROR: tools/pod_bootstrap.sh is the legacy broad bootstrap path.

It can create/modify the venv, install or replace Torch, install optional
Mamba dependencies, and run extra benchmarks. For Exp23 / Parameter Golf pods,
use scripts/pod_bootstrap.sh instead. For already-synced test pods that only
need native kernels, use scripts/pod_build_native_extensions.sh.

If you intentionally want the broad mutating path, rerun with:
  CHAOSCONTROL_ALLOW_BROAD_BOOTSTRAP=1 bash /workspace/chaoscontrol/tools/pod_bootstrap.sh
EOF
    exit 2
fi

REPO="/workspace/chaoscontrol"
DATA="/workspace/data"
FINEWEB_DIR="/workspace/fineweb_data"

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

# Create venv if it doesn't exist (avoids PEP 668 "externally-managed" errors)
if [ ! -f "$REPO/.venv/bin/python3" ]; then
    echo "Creating venv at $REPO/.venv..."
    $PYTHON -m venv "$REPO/.venv"
fi
export PATH="$REPO/.venv/bin:$PATH"
PYTHON="$REPO/.venv/bin/python3"
PIP="$REPO/.venv/bin/pip3"

# Redirect HF/pip caches to network volume to avoid filling container disk
export HF_HOME="/workspace/.cache"
export PIP_CACHE_DIR="/workspace/.pip_cache"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

echo "Python: $($PYTHON --version 2>&1)"
echo "Venv: $REPO/.venv"

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

$PIP install pyyaml numpy sentencepiece

if [ -d "$REPO" ]; then
    echo "Installing chaoscontrol in dev mode..."
    $PIP install -e "$REPO"
else
    echo "WARNING: $REPO not found. Clone it first, on the pod:"
    echo "  git clone git@github.com:KenMalloy/chaoscontrol.git $REPO"
fi

# ---------------------------------------------------------------------------
# 6. Verify FineWeb data (should already be on the network disk from prep_data.sh)
# ---------------------------------------------------------------------------
echo ""
echo "=== Data: FineWeb ==="

FINEWEB_DIR="/workspace/fineweb_data"
RAW_TEXT=$(find "$FINEWEB_DIR" -name "docs_raw.txt" 2>/dev/null | head -1)

if [ -n "$RAW_TEXT" ] && [ -s "$RAW_TEXT" ]; then
    SIZE=$(stat -c%s "$RAW_TEXT" 2>/dev/null || stat -f%z "$RAW_TEXT" 2>/dev/null)
    echo "FineWeb docs_raw.txt found: $RAW_TEXT ($SIZE bytes)"
    DATA_PATH="$(dirname "$RAW_TEXT")"
else
    echo "WARNING: FineWeb docs_raw.txt not found in $FINEWEB_DIR"
    echo "  Run prep_data.sh on a CPU pod first to download and extract FineWeb."
    echo "  The GPU pod expects the data to already be on the network disk."
    DATA_PATH="$FINEWEB_DIR"
fi

# Check for SP8192 tokenized binary shards (needed by Exp 15+)
# Contract from cached_challenge_fineweb.py --variant sp8192:
#   datasets/fineweb10B_sp8192/fineweb_train_*.bin, fineweb_val_*.bin
#   tokenizers/fineweb_8192_bpe.model
echo ""
echo "=== Data: SP8192 tokenized shards ==="
SP_DATA_DIR="$REPO/baselines/parameter_golf/datasets/fineweb10B_sp8192"
SP_TRAIN_SHARD=$(find "$SP_DATA_DIR" -name "fineweb_train_*.bin" 2>/dev/null | head -1)
SP_VAL_SHARD=$(find "$SP_DATA_DIR" -name "fineweb_val_*.bin" 2>/dev/null | head -1)
SP_MODEL="$REPO/baselines/parameter_golf/tokenizers/fineweb_8192_bpe.model"

if [ -n "$SP_TRAIN_SHARD" ] && [ -n "$SP_VAL_SHARD" ]; then
    echo "SP8192 shards found at $SP_DATA_DIR"
    echo "  train: $SP_TRAIN_SHARD"
    echo "  val:   $SP_VAL_SHARD"
else
    echo "WARNING: SP8192 tokenized shards not found in $SP_DATA_DIR"
    echo "  Exp 17/18 require pre-tokenized SP8192 .bin shards."
    echo "  Run: cd $REPO/baselines/parameter_golf && python cached_challenge_fineweb.py --variant sp8192 --train-shards 80"
fi

if [ -f "$SP_MODEL" ]; then
    echo "SP8192 model: $SP_MODEL"
else
    echo "WARNING: fineweb_8192_bpe.model not found at $SP_MODEL"
    echo "  Required for Exp 17/18 bpb scoring."
    echo "  Run: cd $REPO/baselines/parameter_golf && python cached_challenge_fineweb.py --variant sp8192 --train-shards 80"
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
# 9. Batch size benchmark
# ---------------------------------------------------------------------------
echo ""
echo "=== Batch size benchmark ==="

BENCH_JSON="/workspace/results/exp14_batch_bench.json"
if [ -d "$REPO" ]; then
    if $PYTHON "$REPO/tools/benchmark_batch.py" --output-json "$BENCH_JSON"; then
        $PYTHON -c "
import json
with open('$BENCH_JSON') as f:
    payload = json.load(f)
print(f\"Recommended batch_size: {payload['recommended_batch_size']}\")
"
    else
        echo "WARNING: benchmark_batch.py failed, falling back to heuristic recommendation"
        $PYTHON -c "
import torch
gpu_count = torch.cuda.device_count()
if gpu_count == 0:
    print('No GPUs detected, cannot recommend batch size')
else:
    mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    recommended = int((mem_gb / 1.5) * 64)
    recommended = 2 ** int(recommended).bit_length() // 2
    recommended = max(32, min(recommended, 512))
    print(f'Recommended batch_size (heuristic fallback): {recommended}')
"
    fi
fi

# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Bootstrap complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"
echo ""
echo "Data path:  $DATA_PATH"
echo "Repo path:  $REPO"
echo ""
echo "Run experiments with:"
echo "  cd $REPO"
echo "  # Exp 17 (local attention sidecar):"
echo "  python experiments/17_local_attn_sidecar/run_exp17.py --data-path $SP_DATA_DIR --sp-model-path \$SP_MODEL --budget 600 --num-gpus \$(nvidia-smi -L | wc -l)"
echo "  # Exp 18 (throughput advantage):"
echo "  python experiments/18_throughput_advantage/run_exp18.py --phase0-summary <path> --budget 600 --num-gpus \$(nvidia-smi -L | wc -l)"
echo ""
