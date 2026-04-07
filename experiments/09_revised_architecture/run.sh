#!/usr/bin/env bash
set -euo pipefail
DATA="${1:?Usage: run.sh /path/to/fineweb_data [--budget SECONDS] [--l5-budget SECONDS] [--num-gpus N]}"
shift
BUDGET=150
L5_BUDGET=900
NUM_GPUS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget) BUDGET="$2"; shift 2 ;;
    --l5-budget) L5_BUDGET="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python3 "$(dirname "$0")/run_layered.py" \
    --data-path "$DATA" \
    --budget "$BUDGET" \
    --l5-budget "$L5_BUDGET" \
    --num-gpus "$NUM_GPUS"
