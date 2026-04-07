#!/usr/bin/env bash
set -euo pipefail
ENWIK8="${1:?Usage: run.sh /path/to/enwik8 [--budget SECONDS]}"
shift
BUDGET=300
while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget) BUDGET="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
RESULTS_DIR="$(dirname "$0")/results"
mkdir -p "$RESULTS_DIR"

for cfg in "$(dirname "$0")"/configs/*.yaml; do
    name="$(basename "$cfg" .yaml)"
    echo "  Running $name..."
    .venv/bin/python -m chaoscontrol.runner \
        --config "$cfg" \
        --enwik8-path "$ENWIK8" \
        --budget "$BUDGET" \
        --output-json "$RESULTS_DIR/${name}.json"
done
echo "Done. Results in $RESULTS_DIR"
