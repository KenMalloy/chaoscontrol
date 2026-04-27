#!/usr/bin/env bash
# One-command pod bootstrap. After cloning the repo to /workspace, run:
#
#     HF_TOKEN=hf_... bash scripts/pod_bootstrap.sh
#
# The script is idempotent — re-running on a partially-set-up pod skips
# anything that's already done.
#
# Steps:
#   1. CUDA 13 + TE 2.13 + chaoscontrol editable install (calls
#      pod_setup_cuda13.sh which is itself idempotent)
#   2. Install huggingface-hub + requests in the venv
#   3. Fetch SP16384 train+val shards + tokenizer (Natooka)
#   4. Stream first 50k docs of docs_selected.jsonl (willdepueoai)
#   5. Build val cache for Exp 20 fast scorer
#   6. Smoke-import everything to confirm the pod is fire-ready
#
# Total wall on a fresh pod: ~15 min (most of it pod_setup pip installs).
#
# Reproducibility: every dependency, env var, and download path is
# captured here. The next pod is one bash invocation away from ready.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/chaoscontrol}
WORKSPACE_VENV=${WORKSPACE_VENV:-/workspace/venv}
VAL_CACHE_DIR=${VAL_CACHE_DIR:-/workspace/cache/exp23_val_16384}
HF_TOKEN_REQUIRED=${HF_TOKEN_REQUIRED:-1}

if [ ! -d "$REPO_ROOT" ]; then
    echo "ERROR: $REPO_ROOT not found. Clone the repo first:" >&2
    echo "  cd /workspace && git clone https://github.com/KenMalloy/chaoscontrol.git" >&2
    exit 1
fi

cd "$REPO_ROOT"

echo "==> Step 1/6: CUDA 13 + TE + chaoscontrol editable (idempotent)"
bash scripts/pod_setup_cuda13.sh
# pod_setup_cuda13.sh activates and creates /workspace/venv. Source it
# so subsequent steps see the venv.
# shellcheck source=/dev/null
source "$WORKSPACE_VENV/bin/activate"

echo ""
echo "==> Step 2/6: install huggingface-hub + requests in venv"
# Idempotent: pip skips if up-to-date.
pip install --only-binary=:all: huggingface-hub requests

echo ""
echo "==> Step 3/6: fetch SP16384 train+val shards + tokenizer"
python "$REPO_ROOT/scripts/fetch_sp16384_dataset.py"

echo ""
echo "==> Step 4/6: stream first 50k docs of docs_selected.jsonl"
if [ -z "${HF_TOKEN:-}" ] && [ "$HF_TOKEN_REQUIRED" = "1" ]; then
    echo "ERROR: HF_TOKEN env var required (willdepueoai/parameter-golf is auth-only)" >&2
    echo "  HF_TOKEN=hf_... bash scripts/pod_bootstrap.sh" >&2
    exit 1
fi
python "$REPO_ROOT/scripts/stream_docs_selected.py"

echo ""
echo "==> Step 5/6: build val cache (50k docs, SP16384 tokenizer)"
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

echo ""
echo "==> Step 6/6: smoke check"
python - <<'PY'
import torch
from chaoscontrol.kernels import _cpu_ssm_controller as cpu_ext
from chaoscontrol.kernels._cublaslt import _C as cublaslt_C  # noqa: F401
from chaoscontrol.kernels._ssm_scan import _C as ssm_scan_C  # noqa: F401
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  GPUs={torch.cuda.device_count()}")
print(f"cpu_ssm_controller cuda_pack_available={cpu_ext.write_event_cuda_pack_available()}")
PY

echo ""
echo "Pod ready. Launch the matrix with:"
echo "  EPISODIC_CONTROLLER_V1_WEIGHTS_PATH=$REPO_ROOT/experiments/25_controller_pretrain/simplex_policy_v1.cswg \\"
echo "    python $REPO_ROOT/experiments/24_training_time_bundle/run_exp24.py \\"
echo "    --matrix episodic_controller_v1 --limit 18 \\"
echo "    --full-val-score --val-cache-dir $VAL_CACHE_DIR --val-budget-seconds 600"
echo ""
echo "(scp simplex_policy_v1.cswg from your local Mac before launch — it's"
echo " gitignored and not on HF.)"
