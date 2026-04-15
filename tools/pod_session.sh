#!/bin/bash
# pod_session.sh — laptop-side orchestration for an Exp 18 pod session.
#
# Different layer than tools/pod_bootstrap.sh: that script runs ON the
# pod after the code has been rsynced and installs torch/mamba/etc from
# scratch. This script runs LOCALLY, SSHes into a pod that's already
# been bootstrapped once, and handles the session-start dance I was
# doing manually on 2026-04-14:
#
#   1. runpodctl ssh info -> extract IP/port
#   2. verify SSH reachability
#   3. sync local git HEAD via git archive | scp | tar extract
#      (preserves any files on the pod that aren't tracked by git,
#      e.g. datasets/, tokenizers/, result JSONs from prior tests)
#   4. ensure sentencepiece is installed (the historic "missing dep"
#      that killed the first attempt at running Test 2 yesterday)
#   5. preflight the runner + orchestrator import chain
#   6. verify the SP8192 + SP16384 data + tokenizer files are where
#      the orchestrators expect them
#   7. detect GPU count and print ready-to-paste launch commands for
#      each of Tests 3/4/5/6/7 with the right --num-gpus / --num-slots
#
# Usage:
#   tools/pod_session.sh <pod_id>
#
# Env vars:
#   RUNPOD_SSH_KEY    Path to the RunPod SSH private key. Defaults to
#                     ~/.runpod/ssh/RunPod-Key-Go.
#   DATA_BASE         Override the expected data root on the pod.
#                     Defaults to /workspace/chaoscontrol/baselines/parameter_golf.

set -euo pipefail

POD_ID="${1:?usage: tools/pod_session.sh <pod_id>}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/RunPod-Key-Go}"
DATA_BASE="${DATA_BASE:-/workspace/chaoscontrol/baselines/parameter_golf}"

if [[ ! -f "$SSH_KEY" ]]; then
    echo "ERROR: SSH key not found at $SSH_KEY" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 1. Fetch SSH connection info
# -----------------------------------------------------------------------------
echo ">>> [1/7] Fetching SSH info for pod $POD_ID..."
if ! SSH_INFO=$(runpodctl ssh info "$POD_ID" 2>&1); then
    echo "ERROR: runpodctl ssh info failed:" >&2
    echo "$SSH_INFO" >&2
    exit 1
fi

POD_IP=$(echo "$SSH_INFO" | python3 -c 'import sys, json; print(json.load(sys.stdin)["ip"])')
POD_PORT=$(echo "$SSH_INFO" | python3 -c 'import sys, json; print(json.load(sys.stdin)["port"])')

SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"
SSH="ssh $SSH_OPTS root@$POD_IP -p $POD_PORT"
SCP="scp $SSH_OPTS -P $POD_PORT"

echo "    pod address: $POD_IP:$POD_PORT"

# -----------------------------------------------------------------------------
# 2. Verify SSH reachability
# -----------------------------------------------------------------------------
echo ">>> [2/7] Verifying SSH reachability..."
if ! $SSH 'echo pong' >/dev/null 2>&1; then
    echo "ERROR: cannot SSH into pod. Is it actually RUNNING?" >&2
    echo "       runpodctl get pod $POD_ID" >&2
    exit 1
fi
echo "    ok"

# -----------------------------------------------------------------------------
# 3. Sync code via git archive if pod HEAD doesn't match local HEAD
# -----------------------------------------------------------------------------
echo ">>> [3/7] Checking code state..."
LOCAL_HEAD=$(git rev-parse HEAD)
echo "    local HEAD:  $LOCAL_HEAD"

REMOTE_HEAD=$($SSH 'cat /workspace/chaoscontrol/.cc_head 2>/dev/null || echo none')
echo "    remote HEAD: $REMOTE_HEAD"

if [[ "$REMOTE_HEAD" != "$LOCAL_HEAD" ]]; then
    echo "    remote stale, syncing via git archive..."
    TAR_PATH="/tmp/cc_session_$$.tar.gz"
    git archive HEAD --format=tar.gz -o "$TAR_PATH"
    $SCP "$TAR_PATH" "root@$POD_IP:/tmp/cc_session.tar.gz" >/dev/null
    $SSH "cd /workspace/chaoscontrol && tar -xzf /tmp/cc_session.tar.gz && echo '$LOCAL_HEAD' > .cc_head && rm /tmp/cc_session.tar.gz"
    rm -f "$TAR_PATH"
    echo "    synced to $LOCAL_HEAD"
else
    echo "    up to date"
fi

# -----------------------------------------------------------------------------
# 4. Install missing pip deps
# -----------------------------------------------------------------------------
echo ">>> [4/7] Checking pip deps..."
if $SSH 'python -c "import sentencepiece" 2>/dev/null'; then
    echo "    sentencepiece ok"
else
    echo "    sentencepiece missing, installing..."
    $SSH 'pip install --break-system-packages sentencepiece 2>&1 | tail -3'
fi

# -----------------------------------------------------------------------------
# 5. Preflight runner + orchestrator imports
# -----------------------------------------------------------------------------
echo ">>> [5/7] Preflighting import chain..."
$SSH 'cd /workspace/chaoscontrol && python - <<PY
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "experiments/09_revised_architecture")
sys.path.insert(0, "experiments/17_local_attn_sidecar")
sys.path.insert(0, "experiments/18_throughput_levers")

# runner + training stack
from runner_exp17 import build_model, load_sp_data, evaluate_bpb_sp  # noqa
from chaoscontrol.training import train_chaoscontrol_for_budget  # noqa
from chaoscontrol.optim.muon import Muon  # noqa
from chaoscontrol.optim.lamb import LAMB  # noqa
import sentencepiece  # noqa

# orchestrators + shared helper
import _harness  # noqa
import runner_exp18  # noqa
import run_exp18  # noqa   (Test 2 — tokenizer shootout)
import run_exp18_test3  # noqa
import run_exp18_test4  # noqa
import run_exp18_test5  # noqa
import run_exp18_test6  # noqa
import run_exp18_test7  # noqa

print("    import chain ok")
PY'

# -----------------------------------------------------------------------------
# 6. Verify data + tokenizer paths
# -----------------------------------------------------------------------------
echo ">>> [6/7] Verifying data and tokenizer paths under $DATA_BASE..."
$SSH "
set -e
missing=0
for d in '$DATA_BASE/datasets/fineweb10B_sp8192' '$DATA_BASE/datasets/fineweb10B_sp16384'; do
    if [[ ! -d \"\$d\" ]]; then
        echo \"    MISSING dataset: \$d\"
        missing=1
    fi
done
for f in '$DATA_BASE/tokenizers/fineweb_8192_bpe.model' '$DATA_BASE/tokenizers/fineweb_16384_bpe.model'; do
    if [[ ! -f \"\$f\" ]]; then
        echo \"    MISSING tokenizer: \$f\"
        missing=1
    fi
done
if [[ \$missing -ne 0 ]]; then
    exit 1
fi
echo '    all data + tokenizer files present'
"

# -----------------------------------------------------------------------------
# 7. Detect GPU count and print launch recipes
# -----------------------------------------------------------------------------
echo ">>> [7/7] Detecting GPU count..."
GPU_COUNT=$($SSH 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l' | tr -d ' \r\n')
echo "    $GPU_COUNT GPUs visible"

if (( GPU_COUNT < 2 )); then
    DDP_SLOTS=1
else
    DDP_SLOTS=$(( GPU_COUNT / 2 ))
fi

SP16_DATA="$DATA_BASE/datasets/fineweb10B_sp16384"
SP16_TOK="$DATA_BASE/tokenizers/fineweb_16384_bpe.model"

cat <<EOF

===============================================================================
POD READY FOR EXP 18 TEST SESSION
===============================================================================

Pod:   $POD_ID
Host:  $POD_IP:$POD_PORT
HEAD:  $LOCAL_HEAD
GPUs:  $GPU_COUNT  (ws=2 parallel slots: $DDP_SLOTS)
Data:  $DATA_BASE

SSH:
  ssh -i $SSH_KEY -o StrictHostKeyChecking=no root@$POD_IP -p $POD_PORT

Launch recipes (paste into the SSH session from /workspace/chaoscontrol,
each runs under nohup so the ssh session can close afterward):

  # Test 4 FIRST — anchors the ws=1 baseline that Test 5 cross-checks against
  mkdir -p experiments/18_throughput_levers/results_test4
  nohup python experiments/18_throughput_levers/run_exp18_test4.py \\
      --data-path $SP16_DATA \\
      --sp-model-path $SP16_TOK \\
      --num-gpus $GPU_COUNT --budget 600 \\
      > experiments/18_throughput_levers/results_test4/orchestrator.log 2>&1 &

  # Test 3 — activation checkpointing ceiling push (parallel single-GPU)
  mkdir -p experiments/18_throughput_levers/results_test3
  nohup python experiments/18_throughput_levers/run_exp18_test3.py \\
      --data-path $SP16_DATA \\
      --sp-model-path $SP16_TOK \\
      --num-gpus $GPU_COUNT --budget 600 \\
      > experiments/18_throughput_levers/results_test3/orchestrator.log 2>&1 &

  # Test 5 — LR stability screen at ws=2 DDP (gates on Test 4's ws1)
  mkdir -p experiments/18_throughput_levers/results_test5
  nohup python experiments/18_throughput_levers/run_exp18_test5.py \\
      --data-path $SP16_DATA \\
      --sp-model-path $SP16_TOK \\
      --num-slots $DDP_SLOTS --budget 600 \\
      > experiments/18_throughput_levers/results_test5/orchestrator.log 2>&1 &

  # Test 6 — seq_len sweep at ws=2 DDP (reads Test 5 winning LR)
  mkdir -p experiments/18_throughput_levers/results_test6
  nohup python experiments/18_throughput_levers/run_exp18_test6.py \\
      --data-path $SP16_DATA \\
      --sp-model-path $SP16_TOK \\
      --num-slots $DDP_SLOTS --budget 600 \\
      > experiments/18_throughput_levers/results_test6/orchestrator.log 2>&1 &

  # Test 7 — optimizer ablation at ws=2 DDP (reads Test 5 winning LR)
  mkdir -p experiments/18_throughput_levers/results_test7
  nohup python experiments/18_throughput_levers/run_exp18_test7.py \\
      --data-path $SP16_DATA \\
      --sp-model-path $SP16_TOK \\
      --num-slots $DDP_SLOTS --budget 600 \\
      > experiments/18_throughput_levers/results_test7/orchestrator.log 2>&1 &

Run order:
  Test 4  (must be first — establishes ws=1 baseline)
  Test 3  (independent; can run in parallel with Test 4 if the pod has
           enough GPUs, but Test 4 is the dependency for Test 5)
  Test 5  (reads Test 4's ws=1 results via results_test4/ws1_s*.json)
  Test 6  (reads Test 5's winning LR from results_test5/test5_summary.json;
           refuses to run on a provisional/missing winner unless you pass
           --base-lr explicitly)
  Test 7  (same dependency on Test 5 as Test 6)

EOF
