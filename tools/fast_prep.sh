#!/usr/bin/env bash
# =============================================================================
# fast_prep.sh — Fastest possible data prep for a fresh pod (no network volume)
#
# Optimized for expensive GPU pods where every minute counts.
# Downloads FineWeb via HF, extracts with parallel jq, splits by document.
#
# Usage: HF_TOKEN=hf_xxx bash tools/fast_prep.sh [JOBS]
#   JOBS = number of parallel jq workers (default: nproc)
#
# Requirements: jq, python3 with huggingface-hub, HF_TOKEN env var
# =============================================================================
set -euo pipefail

REPO="/workspace/chaoscontrol"
FINEWEB_DIR="/workspace/fineweb_data"
CHUNKS_DIR="/workspace/fineweb_data/chunks"
VAL_DOCS=50000
JOBS="${1:-$(nproc)}"

echo "============================================"
echo "ChaosControl fast data prep"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Workers: $JOBS"
echo "============================================"

# Redirect caches to workspace (not container disk)
export HF_HOME="/workspace/.cache"
export PIP_CACHE_DIR="/workspace/.pip_cache"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$FINEWEB_DIR"

# Use venv if available
if [ -f "$REPO/.venv/bin/python3" ]; then
    export PATH="$REPO/.venv/bin:$PATH"
fi

# Check HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export it before running."
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Download ONLY the JSONL (skip tokenized shards — we use raw bytes)
# ---------------------------------------------------------------------------
JSONL="$REPO/baselines/parameter_golf/docs_selected.jsonl"
if [ -f "$JSONL" ]; then
    echo "=== JSONL already present ($(du -sh "$JSONL" | cut -f1)), skipping download ==="
else
    echo ""
    echo "=== Downloading docs_selected.jsonl from HuggingFace ==="
    T0=$(date +%s)
    cd "$REPO/baselines/parameter_golf"
    # Download ONLY the JSONL — no training shards (we don't use sp1024 tokens).
    # --train-shards 0 skips all tokenized bins. --with-docs gets the JSONL.
    python3 cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs
    cd "$REPO"
    T1=$(date +%s)
    echo "Download took $((T1 - T0))s"
fi

# Verify JSONL exists
if [ ! -f "$JSONL" ]; then
    echo "ERROR: JSONL not found at $JSONL after download"
    exit 1
fi
echo "JSONL: $(wc -l < "$JSONL") lines, $(du -sh "$JSONL" | cut -f1)"

# ---------------------------------------------------------------------------
# 2. Parallel extraction — split by document boundary, extract with jq
# ---------------------------------------------------------------------------
VAL_RAW="$FINEWEB_DIR/docs_val_raw.txt"
TRAIN_RAW="$FINEWEB_DIR/docs_train_raw.txt"

if [ -f "$VAL_RAW" ] && [ -s "$VAL_RAW" ] && [ -f "$TRAIN_RAW" ] && [ -s "$TRAIN_RAW" ]; then
    echo "=== Split files already exist, skipping extraction ==="
    echo "  val:   $(du -sh "$VAL_RAW" | cut -f1)"
    echo "  train: $(du -sh "$TRAIN_RAW" | cut -f1)"
else
    echo ""
    echo "=== Extracting val ($VAL_DOCS docs) ==="
    T0=$(date +%s)
    head -"$VAL_DOCS" "$JSONL" | jq -r '.text' > "$VAL_RAW"
    T1=$(date +%s)
    echo "Val extraction: $((T1 - T0))s, $(du -sh "$VAL_RAW" | cut -f1)"

    echo ""
    echo "=== Extracting train (parallel, $JOBS workers) ==="
    T0=$(date +%s)
    mkdir -p "$CHUNKS_DIR"
    # Split train portion into chunks
    tail -n +"$((VAL_DOCS + 1))" "$JSONL" | split -l 500000 -d --additional-suffix=.jsonl - "$CHUNKS_DIR/c_"
    NCHUNKS=$(ls "$CHUNKS_DIR"/c_*.jsonl | wc -l)
    echo "  Split into $NCHUNKS chunks, processing..."
    # Parallel jq extraction
    ls "$CHUNKS_DIR"/c_*.jsonl | xargs -P "$JOBS" -I {} sh -c 'jq -r ".text" "{}" > "{}.txt"'
    # Concatenate in order
    cat "$CHUNKS_DIR"/c_*.jsonl.txt > "$TRAIN_RAW"
    # Cleanup chunks
    rm -rf "$CHUNKS_DIR"
    T1=$(date +%s)
    echo "Train extraction: $((T1 - T0))s, $(du -sh "$TRAIN_RAW" | cut -f1)"
fi

# ---------------------------------------------------------------------------
# 3. Clean up JSONL + HF cache to free disk
# ---------------------------------------------------------------------------
echo ""
echo "=== Cleanup ==="
rm -f "$JSONL" "$REPO/baselines/parameter_golf/docs_selected.source_manifest.json"
rm -rf "$REPO/baselines/parameter_golf/datasets"
rm -rf "/workspace/.cache"
echo "Freed: JSONL + HF cache + tokenized bins"

# ---------------------------------------------------------------------------
# 4. Validate
# ---------------------------------------------------------------------------
echo ""
echo "=== Validation ==="
python3 -c "
import os
for label, path in [('val', '$VAL_RAW'), ('train', '$TRAIN_RAW')]:
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(200)
    print(f'{label}: {size:,} bytes ({size/1e9:.2f} GB)')
    print(f'  Head: {head[:80]}...')
    if size < 1_000_000:
        print(f'  WARNING: {label} seems too small')
"

echo ""
echo "============================================"
echo "Fast prep complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"
echo ""
echo "Data dir: $FINEWEB_DIR"
echo "  val:   $VAL_RAW"
echo "  train: $TRAIN_RAW"
