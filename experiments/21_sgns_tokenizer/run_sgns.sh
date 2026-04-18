#!/usr/bin/env bash
# Offline SGNS training for Exp 21 — pins the canonical hyperparameters.
#
# Usage:
#   bash experiments/21_sgns_tokenizer/run_sgns.sh <data-dir> <out-artifact>
#
# Reads FineWeb SP8192 shards from <data-dir>, trains SGNS for 3 epochs,
# writes (V, D) embedding tensor to <out-artifact>.
set -euo pipefail

DATA_DIR="${1:?data-dir required}"
OUT="${2:-artifacts/sgns_v8192_d256.pt}"

mkdir -p "$(dirname "$OUT")"

python scripts/train_sgns.py \
  --data-dir "$DATA_DIR" \
  --vocab-size 8192 --dim 256 --window 5 --k 10 --epochs 3 \
  --subsample-threshold 1e-5 --batch-size 4096 --lr 0.025 --seed 0 \
  --max-tokens 50000000 \
  --out "$OUT"
