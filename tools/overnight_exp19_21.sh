#!/usr/bin/env bash
# Overnight Exp19/Exp21 completion runner for a 4xH100 RunPod.
#
# Runs on the pod from /workspace/chaoscontrol. It is intentionally
# idempotent: existing result JSONs are skipped by the experiment launchers,
# and missing data/artifacts are prepared before GPU work starts.
set -euo pipefail

REPO="${REPO:-/workspace/chaoscontrol}"
PY="${PY:-/workspace/venv/bin/python}"
export PY
LOG_DIR="$REPO/experiments/overnight_exp19_21_logs"
STATUS_JSON="$LOG_DIR/status.json"
DATA_BASE="$REPO/baselines/parameter_golf"
DOCS_JSONL="${DOCS_JSONL:-$DATA_BASE/datasets/docs_selected.jsonl}"
DOCS_JSONL_FALLBACK="$DATA_BASE/docs_selected.jsonl"
SP8192_DATA="$DATA_BASE/datasets/fineweb10B_sp8192"
SP16384_DATA="$DATA_BASE/datasets/fineweb10B_sp16384"
SP8192_MODEL="$DATA_BASE/tokenizers/fineweb_8192_bpe.model"
SP16384_MODEL="$DATA_BASE/tokenizers/fineweb_16384_bpe.model"
ARTIFACT_DIR="$REPO/artifacts"
SGNS="$ARTIFACT_DIR/sgns_v8192_d256.pt"
EXP19_MATRIX="$REPO/experiments/19_phase1/matrix_phase1c_fp8_ws4.json"
EXP19_RESULTS="$REPO/experiments/19_phase1/results_phase1c_fp8_ws4"
EXP19_SUMMARY="$REPO/experiments/19_phase1/RESULTS_PHASE1C_FP8_WS4.md"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR" "$EXP19_RESULTS"
cd "$REPO"

# SentencePiece training writes a multi-GB plaintext corpus through
# tempfile.TemporaryDirectory. On RunPod the default /tmp is the small
# container overlay, while /workspace is the attached volume.
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"

write_status() {
    local state="$1"
    local detail="${2:-}"
    "$PY" - "$STATUS_JSON" "$state" "$detail" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, state, detail = sys.argv[1:4]
payload = {
    "state": state,
    "detail": detail,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

on_exit() {
    rc=$?
    if [[ $rc -eq 0 ]]; then
        write_status "completed" "overnight_exp19_21 finished"
    else
        write_status "failed" "overnight_exp19_21 rc=$rc"
    fi
}
trap on_exit EXIT

write_status "running" "starting"
echo "=== overnight_exp19_21 start: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
echo "repo=$REPO"
echo "python=$PY"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
PY

download_variant() {
    local vocab_size="$1"
    local data_dir="$2"
    local tokenizer="$3"
    local log_file="$LOG_DIR/build_sp${vocab_size}.log"
    if compgen -G "$data_dir/fineweb_train_*.bin" >/dev/null \
        && compgen -G "$data_dir/fineweb_val_*.bin" >/dev/null \
        && [[ -f "$tokenizer" ]]; then
        echo "=== data sp${vocab_size}: present ==="
        return 0
    fi

    if [[ ! -f "$DOCS_JSONL" && -f "$DOCS_JSONL_FALLBACK" ]]; then
        DOCS_JSONL="$DOCS_JSONL_FALLBACK"
    fi
    if [[ ! -f "$DOCS_JSONL" ]]; then
        echo "=== docs_selected.jsonl: downloading ==="
        (
            cd "$DATA_BASE"
            "$PY" cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs
        ) 2>&1 | tee "$LOG_DIR/download_docs_selected.log"
    fi
    if [[ ! -f "$DOCS_JSONL" && -f "$DOCS_JSONL_FALLBACK" ]]; then
        DOCS_JSONL="$DOCS_JSONL_FALLBACK"
    fi
    if [[ ! -f "$DOCS_JSONL" ]]; then
        echo "ERROR: docs_selected.jsonl not found at $DOCS_JSONL or $DOCS_JSONL_FALLBACK" >&2
        return 1
    fi

    echo "=== data sp${vocab_size}: building local tokenized shards ==="
    local args=(
        scripts/build_sp_shards.py
        --docs-path "$DOCS_JSONL"
        --vocab-size "$vocab_size"
        --output-dir "$data_dir"
        --tokenizer-dir "$DATA_BASE/tokenizers"
        --num-workers "$(nproc)"
    )
    # Reuse an existing deterministic tokenizer when present; otherwise
    # train it from docs_selected.jsonl before tokenizing. This saves the
    # SP8192 rebuild path where the model is already checked into the repo.
    if [[ -f "$tokenizer" ]]; then
        args+=(--skip-train)
    fi
    # A failed/partial prior build can leave shards without a manifest.
    # Force-clear that exact stale state rather than letting load_fineweb
    # consume partial bins later.
    if compgen -G "$data_dir/fineweb_*.bin" >/dev/null \
        && [[ ! -f "$data_dir/build_manifest.json" ]]; then
        args+=(--force)
    fi
    "$PY" "${args[@]}" 2>&1 | tee "$log_file"
}

ensure_sgns_inits() {
    mkdir -p "$ARTIFACT_DIR"
    if [[ -f "$ARTIFACT_DIR/sgns_init_shuffled.pt" && -f "$ARTIFACT_DIR/sgns_init_zero.pt" ]]; then
        echo "=== SGNS init artifacts: present ==="
        return 0
    fi
    if [[ ! -f "$SGNS" ]]; then
        echo "=== SGNS base tensor missing: training ==="
        bash experiments/21_sgns_tokenizer/run_sgns.sh "$SP8192_DATA" "$SGNS" \
            2>&1 | tee "$LOG_DIR/train_sgns.log"
    fi
    echo "=== deriving SGNS init controls ==="
    "$PY" scripts/prepare_sgns_inits.py \
        --sgns "$SGNS" \
        --out-dir "$ARTIFACT_DIR" \
        --data-dir-for-counts "$SP8192_DATA" \
        --sp-model "$SP8192_MODEL" \
        2>&1 | tee "$LOG_DIR/prepare_sgns_inits.log"
}

run_exp21_controls() {
    echo "=== Exp21 controls start: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
    write_status "running" "exp21_controls"
    "$PY" experiments/21_sgns_tokenizer/runner_controls.py \
        --data-path "$SP8192_DATA" \
        --sp-model-path "$SP8192_MODEL" \
        --budget 600 \
        --num-slots 2 \
        2>&1 | tee "$LOG_DIR/exp21_controls.log"
}

run_exp19_phase1c() {
    echo "=== Exp19 Phase1C start: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
    write_status "running" "exp19_phase1c"
    # The remote data/tokenizer artifacts were absent at session start, so
    # do not trust pre-seeded rows from the prior fp8-bugged run. Delete
    # once, then let persistent-DDP idempotency preserve any fresh rows if
    # this driver is restarted after partial progress.
    local reset_marker="$EXP19_RESULTS/.fresh_tokenizer_reset_done"
    if [[ ! -f "$reset_marker" ]]; then
        rm -f "$EXP19_RESULTS"/*.json "$EXP19_RESULTS"/*.log
        touch "$reset_marker"
    fi
    "$PY" experiments/19_prereqs/run_persistent_launcher.py \
        --data-path "$SP16384_DATA" \
        --sp-model-path "$SP16384_MODEL" \
        --output-dir "$EXP19_RESULTS" \
        --world-size 4 \
        --budget 600 \
        --matrix-json "$EXP19_MATRIX" \
        --rdzv-port 29519 \
        2>&1 | tee "$LOG_DIR/exp19_phase1c.log"

    echo "=== Exp19 summary ==="
    if ! "$PY" experiments/19_phase1/summarize_phase1.py \
        --matrix-json "$EXP19_MATRIX" \
        --runs-dir "$EXP19_RESULTS" \
        --output "$EXP19_SUMMARY" \
        2>&1 | tee "$LOG_DIR/exp19_summarize.log"; then
        echo "strict summary failed; writing lenient summary"
        "$PY" experiments/19_phase1/summarize_phase1.py \
            --matrix-json "$EXP19_MATRIX" \
            --runs-dir "$EXP19_RESULTS" \
            --output "$EXP19_SUMMARY" \
            --lenient \
            2>&1 | tee -a "$LOG_DIR/exp19_summarize.log"
    fi
}

download_variant 8192 "$SP8192_DATA" "$SP8192_MODEL"
download_variant 16384 "$SP16384_DATA" "$SP16384_MODEL"
ensure_sgns_inits
run_exp21_controls
run_exp19_phase1c

echo "=== overnight_exp19_21 done: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
