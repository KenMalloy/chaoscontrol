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

pip3 install --quiet huggingface-hub pyyaml numpy

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
# 4. Extract raw text from JSONL using jq (fast, streaming)
# ---------------------------------------------------------------------------
echo ""
echo "=== Extracting raw text from JSONL ==="

# Find the docs JSONL — typically under datasets/fineweb10B_byte260/
JSONL=$(find "$FINEWEB_DIR" -name "docs_selected.jsonl" -o -name "*.jsonl" | head -1)
RAW_TEXT="$(dirname "$JSONL")/docs_raw.txt"

if [ -z "$JSONL" ]; then
    echo "ERROR: No JSONL file found in $FINEWEB_DIR"
    echo "  Check that --with-docs produced docs_selected.jsonl"
    exit 1
fi

if [ -f "$RAW_TEXT" ] && [ -s "$RAW_TEXT" ]; then
    RAW_SIZE=$(stat -c%s "$RAW_TEXT" 2>/dev/null || stat -f%z "$RAW_TEXT" 2>/dev/null)
    echo "docs_raw.txt already exists ($RAW_SIZE bytes), skipping extraction"
else
    echo "Extracting from: $JSONL"
    echo "Writing to: $RAW_TEXT"
    echo "Using jq (streaming, NOT Python json.loads)"

    # Stream extract: jq reads line-by-line, outputs raw text
    # This is orders of magnitude faster than Python json.loads on large files
    jq -r '.text' "$JSONL" > "$RAW_TEXT"

    RAW_SIZE=$(stat -c%s "$RAW_TEXT" 2>/dev/null || stat -f%z "$RAW_TEXT" 2>/dev/null)
    echo "Extraction complete: $RAW_SIZE bytes ($(echo "scale=2; $RAW_SIZE / 1073741824" | bc) GB)"
fi

# ---------------------------------------------------------------------------
# 5. Validate
# ---------------------------------------------------------------------------
echo ""
echo "=== Validation ==="

python3 -c "
import os
path = '$RAW_TEXT'
size = os.path.getsize(path)
with open(path, 'rb') as f:
    head = f.read(200)
    f.seek(max(0, size - 200))
    tail = f.read()
print(f'docs_raw.txt: {size:,} bytes ({size/1e9:.2f} GB)')
print(f'  Head: {head[:80]}...')
print(f'  Tail readable: {len(tail)} bytes')
unique = len(set(head + tail))
print(f'  Unique byte values in sample: {unique}')
if size < 1_000_000:
    print('  WARNING: file seems too small for FineWeb')
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
echo "FineWeb raw text: $RAW_TEXT"
echo ""
echo "Next steps:"
echo "  1. Start GPU pod with this network volume"
echo "  2. rsync code: python tools/runpod.py deploy <pod_id>"
echo "  3. Bootstrap: bash /workspace/chaoscontrol/tools/pod_bootstrap.sh"
echo "  4. Run exp 14:"
echo "     python experiments/14_vram_buffer/run_exp14.py \\"
echo "       --data-path $(dirname $RAW_TEXT) --budget 600 --num-gpus 8 --phase A"
echo ""
