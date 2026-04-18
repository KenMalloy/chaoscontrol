#!/usr/bin/env bash
# Exp 21 quant-penalty launcher — pins fairness-critical args for the
# V=8192 vs V=16384 comparison so the two serial invocations can't
# drift on calibration size, chunk size, seed, or budget.
#
# Usage:
#   bash experiments/21_sgns_tokenizer/run_quantmeasure.sh \
#       <ckpt-path> <sp-model-path> <output-json>
#
# EVAL_JSONL is overridable via env var because pod data layout can
# vary; everything else (BUDGET_SECONDS, CALIBRATION_SEQS,
# CALIBRATION_SEQ_LEN, SEED, CHUNK_SIZE, MAX_DOCS) is hardcoded so
# the V=8192 and V=16384 runs see IDENTICAL settings. Do not change
# these between two arms of one comparison — they're the control.
set -euo pipefail

CKPT="${1:?usage: run_quantmeasure.sh <ckpt-path> <sp-model-path> <output-json>}"
SP_MODEL="${2:?usage: run_quantmeasure.sh <ckpt-path> <sp-model-path> <output-json>}"
OUTPUT_JSON="${3:?usage: run_quantmeasure.sh <ckpt-path> <sp-model-path> <output-json>}"

EVAL_JSONL="${EVAL_JSONL:-/workspace/shards/fineweb_eval.jsonl}"
BUDGET_SECONDS=600
CALIBRATION_SEQS=64
CALIBRATION_SEQ_LEN=2048
SEED=0
DEVICE=auto
CHUNK_SIZE=256
MAX_DOCS=50000

mkdir -p "$(dirname "$OUTPUT_JSON")"

exec python scripts/eval_quant.py \
    --ckpt "$CKPT" \
    --eval-jsonl "$EVAL_JSONL" \
    --sp-model-path "$SP_MODEL" \
    --output-json "$OUTPUT_JSON" \
    --budget-seconds "$BUDGET_SECONDS" \
    --calibration-seqs "$CALIBRATION_SEQS" \
    --calibration-seq-len "$CALIBRATION_SEQ_LEN" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --chunk-size "$CHUNK_SIZE" \
    --max-docs "$MAX_DOCS"
