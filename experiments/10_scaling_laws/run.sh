#!/usr/bin/env bash
set -euo pipefail
DATA="${1:?Usage: run.sh /path/to/data [--budget SECONDS] [--seeds N] [--sizes XS,S,M,L,XL] [--conditions bare_ssm,full_ssm,our_tfm,mamba2_ssm]}"
shift
BUDGET=600
SEEDS=3
SIZES="XS,S,M,L,XL"
CONDITIONS="bare_ssm,full_ssm,our_tfm,mamba2_ssm"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget) BUDGET="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --conditions) CONDITIONS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python3 "$(dirname "$0")/run_scaling.py" \
    --data-path "$DATA" \
    --budget "$BUDGET" \
    --seeds "$SEEDS" \
    --sizes "$SIZES" \
    --conditions "$CONDITIONS"
