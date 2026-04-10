#!/usr/bin/env bash
# =============================================================================
# prep_data.sh — Prepare FineWeb data on a CPU-only RunPod network disk
#
# Run ON the pod (CPU-only is fine — this is the cheap part):
#   bash /workspace/chaoscontrol/tools/prep_data.sh
#
# Downloads FineWeb, extracts raw text via jq (NOT Python json.loads),
# and leaves docs_raw.txt ready to mmap on the GPU pod.
#
# Disk budget: ~130GB (FineWeb shards + raw text + environment)
# =============================================================================
set -euo pipefail

REPO="/workspace/chaoscontrol"
FINEWEB_DIR="/workspace/fineweb_data"
RESULTS="/workspace/results"
CHECKPOINTS="/workspace/checkpoints"

echo "============================================"
echo "ChaosControl data preparation (FineWeb)"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Directory structure
# ---------------------------------------------------------------------------
echo ""
echo "=== Creating directory structure ==="
mkdir -p "$FINEWEB_DIR"
mkdir -p "$RESULTS"
mkdir -p "$CHECKPOINTS"

# ---------------------------------------------------------------------------
# 2. Dependencies
# ---------------------------------------------------------------------------
echo ""
echo "=== Installing dependencies ==="

# Use venv pip if available (RunPod images block bare pip3 via PEP 668)
if [ -f "$REPO/.venv/bin/pip3" ]; then
    export PATH="$REPO/.venv/bin:$PATH"
    echo "Using venv at $REPO/.venv"
fi
pip3 install --quiet huggingface-hub pyyaml numpy

# Redirect HF cache to network volume to avoid filling container disk
export HF_HOME="/workspace/.cache"
mkdir -p "$HF_HOME"

# jq is critical — stream extraction, not Python json.loads
if ! command -v jq &>/dev/null; then
    echo "Installing jq..."
    apt-get update -qq
    apt-get install -y -qq jq
fi
echo "jq: $(jq --version)"

# ---------------------------------------------------------------------------
# 3. Download FineWeb with docs
# ---------------------------------------------------------------------------
echo ""
echo "=== Downloading FineWeb (byte260 variant with docs) ==="

DOWNLOADER="$REPO/baselines/parameter_golf/cached_challenge_fineweb.py"

if [ ! -f "$DOWNLOADER" ]; then
    echo "ERROR: cached_challenge_fineweb.py not found at $DOWNLOADER"
    echo "  rsync the repo first: python tools/runpod.py deploy <pod_id>"
    exit 1
fi

# Download byte260 variant with document text
cd "$FINEWEB_DIR"
python3 "$DOWNLOADER" --variant byte260 --with-docs

echo "FineWeb download complete"

# ---------------------------------------------------------------------------
# 4. Extract raw text from JSONL — document-aware split
# ---------------------------------------------------------------------------
echo ""
echo "=== Extracting raw text from JSONL (document-aware split) ==="

# Competition defines validation as "the fixed first-50k-document set."
# Each line in docs_selected.jsonl is one document. First 50k = val, rest = train.
VAL_DOCS=50000

JSONL=$(find "$FINEWEB_DIR" -name "docs_selected.jsonl" 2>/dev/null | head -1)
if [ -z "$JSONL" ]; then
    # Fallback: any jsonl
    JSONL=$(find "$FINEWEB_DIR" -name "*.jsonl" 2>/dev/null | head -1)
fi
DATA_DIR="$(dirname "$JSONL")"
VAL_RAW="$DATA_DIR/docs_val_raw.txt"
TRAIN_RAW="$DATA_DIR/docs_train_raw.txt"

if [ -z "$JSONL" ]; then
    echo "ERROR: No JSONL file found in $FINEWEB_DIR"
    echo "  Check that --with-docs produced docs_selected.jsonl"
    exit 1
fi

if [ -f "$VAL_RAW" ] && [ -s "$VAL_RAW" ] && [ -f "$TRAIN_RAW" ] && [ -s "$TRAIN_RAW" ]; then
    VAL_SIZE=$(stat -c%s "$VAL_RAW" 2>/dev/null || stat -f%z "$VAL_RAW" 2>/dev/null)
    TRAIN_SIZE=$(stat -c%s "$TRAIN_RAW" 2>/dev/null || stat -f%z "$TRAIN_RAW" 2>/dev/null)
    echo "Split files already exist (val: $VAL_SIZE bytes, train: $TRAIN_SIZE bytes), skipping"
    RAW_TEXT="$VAL_RAW"
else
    echo "Extracting from: $JSONL"
    echo "First $VAL_DOCS docs -> $VAL_RAW (validation)"
    echo "Remaining docs -> $TRAIN_RAW (training)"
    echo "Using jq (streaming, NOT Python json.loads)"

    # Split at document boundary using head/tail + jq
    # head -N gives first N lines (= first N documents)
    head -"$VAL_DOCS" "$JSONL" | jq -r '.text' > "$VAL_RAW"
    tail -n +"$((VAL_DOCS + 1))" "$JSONL" | jq -r '.text' > "$TRAIN_RAW"

    VAL_SIZE=$(stat -c%s "$VAL_RAW" 2>/dev/null || stat -f%z "$VAL_RAW" 2>/dev/null)
    TRAIN_SIZE=$(stat -c%s "$TRAIN_RAW" 2>/dev/null || stat -f%z "$TRAIN_RAW" 2>/dev/null)
    echo "Extraction complete: val=$VAL_SIZE bytes, train=$TRAIN_SIZE bytes"
    RAW_TEXT="$VAL_RAW"
fi

# ---------------------------------------------------------------------------
# 5. Validate
# ---------------------------------------------------------------------------
echo ""
echo "=== Validation ==="

python3 -c "
import os
for label, path in [('val', '$VAL_RAW'), ('train', '$TRAIN_RAW')]:
    if not os.path.exists(path):
        print(f'{label}: MISSING at {path}')
        continue
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(200)
    print(f'{label}: {size:,} bytes ({size/1e9:.2f} GB)')
    print(f'  Head: {head[:80]}...')
    unique = len(set(head))
    print(f'  Unique byte values in head: {unique}')
    if size < 1_000_000:
        print(f'  WARNING: {label} file seems too small')
"

# ---------------------------------------------------------------------------
# 6. Disk usage
# ---------------------------------------------------------------------------
echo ""
echo "=== Disk usage ==="
du -sh "$FINEWEB_DIR" 2>/dev/null || true
du -sh "$RESULTS" 2>/dev/null || true
du -sh "$CHECKPOINTS" 2>/dev/null || true
echo ""
df -h /workspace 2>/dev/null || true

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Data preparation complete"
echo "============================================"
echo ""
echo "FineWeb val:   $VAL_RAW"
echo "FineWeb train: $TRAIN_RAW"
echo "Data dir:      $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Start GPU pod with this network volume"
echo "  2. rsync code: python tools/runpod.py deploy <pod_id>"
echo "  3. Bootstrap: bash /workspace/chaoscontrol/tools/pod_bootstrap.sh"
echo "  4. Run exp 14:"
echo "     python experiments/14_vram_buffer/run_exp14.py \\"
echo "       --data-path $DATA_DIR --budget 600 --num-gpus 8 --phase A"
echo ""
