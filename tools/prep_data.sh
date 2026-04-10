#!/usr/bin/env bash
# =============================================================================
# prep_data.sh — Prepare datasets on a CPU-only RunPod network disk
#
# Run ON the pod (CPU-only is fine):
#   bash /workspace/chaoscontrol/tools/prep_data.sh
#
# Sets up the full data and environment on the network disk so that GPU pods
# can start experiments immediately by mounting the same volume.
#
# Idempotent: safe to run multiple times. Skips steps that are already done.
#
# Disk budget estimate:
#   Python + PyTorch CPU:      ~3 GB
#   enwik8:                    100 MB (100,000,000 bytes)
#   enwik9:                    1 GB   (1,000,000,000 bytes)
#   PG-19 (subset):            ~11 GB (optional — for long-context eval)
#   Experiment checkpoints:    ~1 GB  (184 runs x ~5 MB each)
#   Results/logs:              ~500 MB
#   Margin:                    ~3 GB
#   -----------------------------------------
#   Total:                     ~20 GB (without PG-19: ~8 GB)
#
# Note: Ken's original 130GB estimate included CUDA PyTorch (~8GB) and room
# for FineWeb data. With enwik8/enwik9 as the primary datasets, actual
# requirements are much smaller. A 50GB network volume is more than sufficient.
# The 130GB figure is only needed if also storing FineWeb shards (~30GB) or
# full PG-19 (~30GB).
# =============================================================================
set -euo pipefail

DATA="/workspace/data"
RESULTS="/workspace/results"
CHECKPOINTS="/workspace/checkpoints"
ENWIK8_URL="http://mattmahoney.net/dc/enwik8.zip"
ENWIK9_URL="http://mattmahoney.net/dc/enwik9.bz2"
# PG-19 is hosted on Google Cloud Storage
PG19_BASE="https://storage.googleapis.com/deepmind-gutenberg"

echo "============================================"
echo "ChaosControl data preparation"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Directory structure
# ---------------------------------------------------------------------------
echo ""
echo "=== Creating directory structure ==="

mkdir -p "$DATA"
mkdir -p "$RESULTS"
mkdir -p "$CHECKPOINTS"
mkdir -p /workspace/chaoscontrol  # placeholder if code not yet rsynced

echo "  /workspace/chaoscontrol/   (code — rsync from local)"
echo "  /workspace/data/           (datasets)"
echo "  /workspace/results/        (experiment outputs)"
echo "  /workspace/checkpoints/    (model checkpoints)"

# ---------------------------------------------------------------------------
# 2. Download enwik8 (100 MB compressed, 100 MB uncompressed)
# ---------------------------------------------------------------------------
echo ""
echo "=== Dataset: enwik8 ==="

ENWIK8_EXPECTED=100000000

if [ -f "$DATA/enwik8" ]; then
    ACTUAL=$(stat -c%s "$DATA/enwik8" 2>/dev/null || stat -f%z "$DATA/enwik8" 2>/dev/null)
    if [ "$ACTUAL" -eq "$ENWIK8_EXPECTED" ]; then
        echo "enwik8 already present and valid ($ACTUAL bytes), skipping"
    else
        echo "enwik8 exists but wrong size ($ACTUAL != $ENWIK8_EXPECTED), re-downloading"
        rm -f "$DATA/enwik8"
    fi
fi

if [ ! -f "$DATA/enwik8" ]; then
    echo "Downloading enwik8..."
    # Ensure unzip is available
    if ! command -v unzip &>/dev/null; then
        echo "  Installing unzip..."
        apt-get update -qq
        apt-get install -y -qq unzip
    fi
    cd "$DATA"
    wget -q --show-progress "$ENWIK8_URL" -O enwik8.zip
    unzip -o enwik8.zip
    rm -f enwik8.zip
    ACTUAL=$(stat -c%s "$DATA/enwik8" 2>/dev/null || stat -f%z "$DATA/enwik8" 2>/dev/null)
    echo "enwik8 downloaded: $ACTUAL bytes"
    if [ "$ACTUAL" -ne "$ENWIK8_EXPECTED" ]; then
        echo "ERROR: enwik8 size mismatch! Expected $ENWIK8_EXPECTED, got $ACTUAL"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 3. Download enwik9 (1 GB compressed, 1 GB uncompressed)
# ---------------------------------------------------------------------------
echo ""
echo "=== Dataset: enwik9 ==="

ENWIK9_EXPECTED=1000000000

if [ -f "$DATA/enwik9" ]; then
    ACTUAL=$(stat -c%s "$DATA/enwik9" 2>/dev/null || stat -f%z "$DATA/enwik9" 2>/dev/null)
    if [ "$ACTUAL" -eq "$ENWIK9_EXPECTED" ]; then
        echo "enwik9 already present and valid ($ACTUAL bytes), skipping"
    else
        echo "enwik9 exists but wrong size ($ACTUAL != $ENWIK9_EXPECTED), re-downloading"
        rm -f "$DATA/enwik9"
    fi
fi

if [ ! -f "$DATA/enwik9" ]; then
    echo "Downloading enwik9..."
    # Ensure bzip2 is available
    if ! command -v bzip2 &>/dev/null; then
        echo "  Installing bzip2..."
        apt-get update -qq
        apt-get install -y -qq bzip2
    fi
    cd "$DATA"
    wget -q --show-progress "$ENWIK9_URL" -O enwik9.bz2
    bzip2 -d enwik9.bz2
    ACTUAL=$(stat -c%s "$DATA/enwik9" 2>/dev/null || stat -f%z "$DATA/enwik9" 2>/dev/null)
    echo "enwik9 downloaded: $ACTUAL bytes"
    if [ "$ACTUAL" -ne "$ENWIK9_EXPECTED" ]; then
        echo "ERROR: enwik9 size mismatch! Expected $ENWIK9_EXPECTED, got $ACTUAL"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 4. Validate data files
# ---------------------------------------------------------------------------
echo ""
echo "=== Validating datasets ==="

# enwik8: first bytes should be XML (starts with "<mediawiki")
HEADER=$(head -c 12 "$DATA/enwik8")
if [[ "$HEADER" == "<mediawiki"* ]]; then
    echo "enwik8: valid XML content (OK)"
else
    echo "WARNING: enwik8 does not start with expected XML header"
    echo "  Got: $(head -c 40 "$DATA/enwik8" | cat -v)"
fi

# enwik9: same format, just larger
HEADER9=$(head -c 12 "$DATA/enwik9")
if [[ "$HEADER9" == "<mediawiki"* ]]; then
    echo "enwik9: valid XML content (OK)"
else
    echo "WARNING: enwik9 does not start with expected XML header"
    echo "  Got: $(head -c 40 "$DATA/enwik9" | cat -v)"
fi

# Quick byte distribution check (should see ASCII + some UTF-8 multibyte)
python3 -c "
import os

for name in ['enwik8', 'enwik9']:
    path = os.path.join('$DATA', name)
    if not os.path.exists(path):
        continue
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        sample = f.read(min(100000, size))
    ascii_frac = sum(32 <= b < 127 for b in sample) / len(sample)
    zero_frac = sum(b == 0 for b in sample) / len(sample)
    unique = len(set(sample))
    print(f'{name}: {size:,} bytes, {unique} unique byte values, '
          f'{ascii_frac:.1%} printable ASCII, {zero_frac:.1%} null bytes')
    if zero_frac > 0.01:
        print(f'  WARNING: high null byte fraction — file may be corrupt')
" 2>/dev/null || echo "  (python3 not available for validation, skipping)"

# ---------------------------------------------------------------------------
# 5. PG-19 download (optional — for long-context evaluation)
# ---------------------------------------------------------------------------
echo ""
echo "=== Dataset: PG-19 (optional, long-context eval) ==="

PG19_DIR="$DATA/pg19"

if [ -d "$PG19_DIR" ] && [ "$(ls -1 "$PG19_DIR"/*.txt 2>/dev/null | wc -l)" -gt 0 ]; then
    COUNT=$(ls -1 "$PG19_DIR"/*.txt 2>/dev/null | wc -l)
    echo "PG-19 already present ($COUNT files), skipping"
else
    echo "PG-19 download skipped (not required for primary experiments)."
    echo ""
    echo "To download PG-19 later, run:"
    echo "  mkdir -p $PG19_DIR"
    echo "  pip install datasets"
    echo "  python3 -c \""
    echo "    from datasets import load_dataset"
    echo "    ds = load_dataset('deepmind/pg19', split='train')"
    echo "    import os; os.makedirs('$PG19_DIR', exist_ok=True)"
    echo "    for i, row in enumerate(ds):"
    echo "        with open(f'$PG19_DIR/{i:05d}.txt', 'w') as f:"
    echo "            f.write(row['text'])"
    echo "    print(f'Wrote {i+1} PG-19 books')"
    echo "  \""
    echo ""
    echo "  PG-19 is ~11GB for the training split."
fi

# ---------------------------------------------------------------------------
# 6. Python environment setup (CPU-only PyTorch for preprocessing)
# ---------------------------------------------------------------------------
echo ""
echo "=== Python environment ==="

PYTHON="python3"

if $PYTHON -c "import torch; import yaml; import numpy; print('Dependencies present')" 2>/dev/null; then
    echo "Required Python packages already installed, skipping"
else
    echo "Installing CPU-only Python packages for data preprocessing..."
    pip3 install torch --index-url https://download.pytorch.org/whl/cpu
    pip3 install pyyaml numpy
    echo "Python packages installed"
fi

# Install chaoscontrol if repo is present
REPO="/workspace/chaoscontrol"
if [ -f "$REPO/pyproject.toml" ]; then
    echo "Installing chaoscontrol in dev mode..."
    pip3 install -e "$REPO"
fi

# ---------------------------------------------------------------------------
# 7. Pre-validate data compatibility with chaoscontrol
# ---------------------------------------------------------------------------
echo ""
echo "=== Data compatibility check ==="

python3 -c "
import os
import sys

data_dir = '$DATA'
enwik8 = os.path.join(data_dir, 'enwik8')
enwik9 = os.path.join(data_dir, 'enwik9')

# ChaosControl loads enwik8 as raw bytes via mmap (uint8 tensor)
# Verify the file is a clean byte stream
for name, path in [('enwik8', enwik8), ('enwik9', enwik9)]:
    if not os.path.exists(path):
        print(f'{name}: NOT FOUND')
        continue
    size = os.path.getsize(path)
    # Read last few bytes to ensure file isn't truncated
    with open(path, 'rb') as f:
        f.seek(max(0, size - 100))
        tail = f.read()
    print(f'{name}: {size:,} bytes, tail OK ({len(tail)} bytes readable)')

print()
print('Data is ready for chaoscontrol byte-level modeling.')
print('Usage: --data-path $DATA/enwik8')
" 2>/dev/null || echo "  (python3 validation skipped)"

# ---------------------------------------------------------------------------
# 8. Disk usage summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Disk usage ==="

echo "Dataset files:"
ls -lh "$DATA"/ 2>/dev/null || echo "  (no files)"
echo ""

if command -v du &>/dev/null; then
    echo "Directory sizes:"
    du -sh "$DATA" 2>/dev/null || true
    du -sh "$RESULTS" 2>/dev/null || true
    du -sh "$CHECKPOINTS" 2>/dev/null || true
    du -sh /workspace/chaoscontrol 2>/dev/null || true
    echo ""
    du -sh /workspace 2>/dev/null || true
fi

echo ""
echo "Network volume free space:"
df -h /workspace 2>/dev/null || true

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Data preparation complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"
echo ""
echo "Datasets:"
echo "  enwik8:  $DATA/enwik8  (100 MB — primary benchmark)"
echo "  enwik9:  $DATA/enwik9  (1 GB — scaling tests)"
echo "  PG-19:   (not downloaded — run instructions printed above)"
echo ""
echo "Next steps:"
echo "  1. Start a GPU pod with this network volume mounted"
echo "  2. rsync code: python tools/runpod.py deploy <pod_id>"
echo "  3. Bootstrap GPU env: bash /workspace/chaoscontrol/tools/pod_bootstrap.sh"
echo "  4. Run experiments: python experiments/09_revised_architecture/run_layered.py \\"
echo "       --data-path $DATA/enwik8 --budget 600 --num-gpus 3"
echo ""
