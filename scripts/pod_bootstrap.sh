#!/usr/bin/env bash
# One-command pod bootstrap. After cloning the repo to /workspace, run:
#
#     bash scripts/pod_bootstrap.sh
#
# By default this builds a scoring-ready pod: native extensions, Natooka
# SP16384 shards, tokenizer, and the Exp27 ValCache verified against the
# prepared validation shard. For a train-only pod, set:
#
#     CHAOSCONTROL_BUILD_VAL_CACHE=0 CHAOSCONTROL_REQUIRE_VAL_CACHE=0 \
#         bash scripts/pod_bootstrap.sh
#
# The script is idempotent — re-running on a partially-set-up pod skips
# anything that's already done.
#
# Steps:
#   1. CUDA 13 + TE 2.13 + chaoscontrol editable install (calls
#      pod_setup_cuda13.sh which is itself idempotent)
#   2. Build/smoke native extensions from the current repo checkout
#      (CUDA write-event pack is optional and depends on toolkit/PyTorch
#      CUDA minor-version agreement; the CPU controller extension itself is
#      required)
#   3. Install huggingface-hub + requests in the venv
#   4. Fetch SP16384 train+val shards + tokenizer (Natooka)
#   5. Stream first 50k docs, build the scorer ValCache, and byte-compare
#      it against Natooka's prepared SP16384 val shard. This is required
#      for Exp27 calc_types and final scoring. It can be explicitly disabled
#      for train-only pods.
#   6. Smoke-import everything to confirm the pod is fire-ready
#
# Total wall on a fresh pod: setup time plus ValCache construction. The
# script writes per-step bootstrap timings so pod bring-up cost stays
# explicit next time.
#
# Reproducibility: every dependency, env var, and download path is
# captured here. The next pod is one bash invocation away from ready.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/chaoscontrol}
WORKSPACE_VENV=${WORKSPACE_VENV:-/workspace/venv}
VAL_CACHE_DIR=${VAL_CACHE_DIR:-"$REPO_ROOT/experiments/27_ttt_headline/val_cache"}
HF_TOKEN_REQUIRED=${HF_TOKEN_REQUIRED:-1}
CHAOSCONTROL_BUILD_VAL_CACHE=${CHAOSCONTROL_BUILD_VAL_CACHE:-1}
CHAOSCONTROL_REQUIRE_VAL_CACHE=${CHAOSCONTROL_REQUIRE_VAL_CACHE:-1}
BOOTSTRAP_TIMING_PATH=${BOOTSTRAP_TIMING_PATH:-"$REPO_ROOT/bootstrap_timing_$(date -u '+%Y%m%dT%H%M%SZ').jsonl"}

if [ ! -d "$REPO_ROOT" ]; then
    echo "ERROR: $REPO_ROOT not found. Clone the repo first:" >&2
    echo "  cd /workspace && git clone https://github.com/KenMalloy/chaoscontrol.git" >&2
    exit 1
fi

cd "$REPO_ROOT"

mkdir -p "$(dirname "$BOOTSTRAP_TIMING_PATH")"
: > "$BOOTSTRAP_TIMING_PATH"

_BOOTSTRAP_STARTED_AT=$(date +%s)
_BOOTSTRAP_STEP_OPEN=0
_BOOTSTRAP_STEP_NAME=""
_BOOTSTRAP_STEP_STARTED_AT=0

_bootstrap_record_timing() {
    local step="$1"
    local status="$2"
    local started_at="$3"
    local ended_at="$4"
    local elapsed=$((ended_at - started_at))
    local total=$((ended_at - _BOOTSTRAP_STARTED_AT))
    printf '[bootstrap-timing] %-28s %-7s %5ss  total=%5ss\n' \
        "$step" "$status" "$elapsed" "$total"
    printf '{"step":"%s","status":"%s","elapsed_s":%s,"total_elapsed_s":%s,"epoch_s":%s}\n' \
        "$step" "$status" "$elapsed" "$total" "$ended_at" >> "$BOOTSTRAP_TIMING_PATH"
}

_bootstrap_step_begin() {
    local step="$1"
    local label="$2"
    echo ""
    echo "==> $label"
    _BOOTSTRAP_STEP_NAME="$step"
    _BOOTSTRAP_STEP_STARTED_AT=$(date +%s)
    _BOOTSTRAP_STEP_OPEN=1
}

_bootstrap_step_end() {
    local ended_at
    ended_at=$(date +%s)
    _bootstrap_record_timing "$_BOOTSTRAP_STEP_NAME" "ok" "$_BOOTSTRAP_STEP_STARTED_AT" "$ended_at"
    _BOOTSTRAP_STEP_OPEN=0
    _BOOTSTRAP_STEP_NAME=""
    _BOOTSTRAP_STEP_STARTED_AT=0
}

_bootstrap_on_exit() {
    local status=$?
    local ended_at
    ended_at=$(date +%s)
    if [ "$_BOOTSTRAP_STEP_OPEN" = "1" ]; then
        _bootstrap_record_timing "$_BOOTSTRAP_STEP_NAME" "failed" "$_BOOTSTRAP_STEP_STARTED_AT" "$ended_at"
    fi
    if [ "$status" -eq 0 ]; then
        _bootstrap_record_timing "total" "ok" "$_BOOTSTRAP_STARTED_AT" "$ended_at"
        echo "Bootstrap timing written to: $BOOTSTRAP_TIMING_PATH"
    else
        _bootstrap_record_timing "total" "failed" "$_BOOTSTRAP_STARTED_AT" "$ended_at"
        echo "Bootstrap timing written to: $BOOTSTRAP_TIMING_PATH" >&2
    fi
}
trap _bootstrap_on_exit EXIT

_bootstrap_step_begin "setup_cuda13" "Step 1/6: CUDA 13 + TE + chaoscontrol editable (idempotent)"
bash scripts/pod_setup_cuda13.sh
# pod_setup_cuda13.sh activates and creates /workspace/venv. Source it
# so subsequent steps see the venv.
# shellcheck source=/dev/null
source "$WORKSPACE_VENV/bin/activate"
_bootstrap_step_end

_bootstrap_step_begin "build_native_extensions" "Step 2/6: build/smoke native extensions"
bash scripts/pod_build_native_extensions.sh
_bootstrap_step_end

_bootstrap_step_begin "install_hf_requests" "Step 3/6: install huggingface-hub + requests in venv"
# Idempotent: pip skips if up-to-date.
pip install --only-binary=:all: huggingface-hub requests
_bootstrap_step_end

_bootstrap_step_begin "fetch_sp16384" "Step 4/6: fetch SP16384 train+val shards + tokenizer"
python "$REPO_ROOT/scripts/fetch_sp16384_dataset.py"
_bootstrap_step_end

_bootstrap_step_begin "exp27_val_cache" "Step 5/6: Exp27 scorer ValCache"
if [ "$CHAOSCONTROL_BUILD_VAL_CACHE" = "1" ]; then
    echo "    CHAOSCONTROL_BUILD_VAL_CACHE=1, streaming docs and building cache"
    if [ -z "${HF_TOKEN:-}" ] && [ "$HF_TOKEN_REQUIRED" = "1" ]; then
        echo "ERROR: HF_TOKEN env var required (willdepueoai/parameter-golf is auth-only)" >&2
        echo "  HF_TOKEN=hf_... bash scripts/pod_bootstrap.sh" >&2
        exit 1
    fi
    python "$REPO_ROOT/scripts/stream_docs_selected.py"

    DOCS_JSONL="$REPO_ROOT/baselines/parameter_golf/datasets/docs_selected.jsonl"
    SP_MODEL="$REPO_ROOT/baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model"
    if [ -f "$VAL_CACHE_DIR/manifest.json" ]; then
        echo "    val cache already built at $VAL_CACHE_DIR — skipping"
    else
        python "$REPO_ROOT/scripts/build_exp20_val_cache.py" \
            --jsonl-path "$DOCS_JSONL" \
            --sp-model-path "$SP_MODEL" \
            --cache-dir "$VAL_CACHE_DIR" \
            --max-docs 50000
    fi
    python "$REPO_ROOT/scripts/verify_sp16384_eval_cache.py" \
        --val-shard "$REPO_ROOT/baselines/parameter_golf/datasets/fineweb10B_sp16384/fineweb_val_000000.bin" \
        --val-cache-dir "$VAL_CACHE_DIR"
else
    echo "    CHAOSCONTROL_BUILD_VAL_CACHE=0, skipping cache build"
    if [ -f "$VAL_CACHE_DIR/manifest.json" ]; then
        echo "    existing val cache found at $VAL_CACHE_DIR — verifying"
        python "$REPO_ROOT/scripts/verify_sp16384_eval_cache.py" \
            --val-shard "$REPO_ROOT/baselines/parameter_golf/datasets/fineweb10B_sp16384/fineweb_val_000000.bin" \
            --val-cache-dir "$VAL_CACHE_DIR"
    elif [ "$CHAOSCONTROL_REQUIRE_VAL_CACHE" = "1" ]; then
        echo "ERROR: Exp27/final scoring ValCache is missing: $VAL_CACHE_DIR" >&2
        echo "  Leave CHAOSCONTROL_BUILD_VAL_CACHE=1, or set CHAOSCONTROL_REQUIRE_VAL_CACHE=0 for train-only pods." >&2
        exit 1
    else
        echo "    train-only setup allowed by CHAOSCONTROL_REQUIRE_VAL_CACHE=0"
    fi
fi
_bootstrap_step_end

_bootstrap_step_begin "smoke_check" "Step 6/6: smoke check"
python - <<'PY'
import torch
from chaoscontrol.kernels import _cpu_ssm_controller as cpu_ext
from chaoscontrol.kernels._lm_head_loss import _C as lm_head_C  # noqa: F401
from chaoscontrol.kernels._cublaslt import _C as cublaslt_C  # noqa: F401
from chaoscontrol.kernels._ssm_scan import _C as ssm_scan_C  # noqa: F401
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  GPUs={torch.cuda.device_count()}")
print(f"cpu_ssm_controller cuda_pack_available={cpu_ext.write_event_cuda_pack_available()}")
PY
_bootstrap_step_end

echo ""
echo "Pod ready. Launch the matrix with:"
echo "  PYTHONPATH=$REPO_ROOT/src \\"
echo "    python $REPO_ROOT/experiments/26_arm/profile_exp26.py \\"
echo "    --world-size 4 --arm both --budget 45 \\"
echo "    --results-dir $REPO_ROOT/experiments/26_arm/validation/profile_4h100_both_45s"
echo ""
echo "For the fixed canary:"
echo "  PYTHONPATH=$REPO_ROOT/src \\"
echo "    python $REPO_ROOT/experiments/26_arm/run_exp26.py --world-size 4 --budget 45"
