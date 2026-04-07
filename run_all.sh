#!/usr/bin/env bash
set -euo pipefail
ENWIK8="${1:?Usage: run_all.sh /path/to/enwik8 [--budget SECONDS]}"
shift
BUDGET=300
while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget) BUDGET="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== ChaosControl Research Suite ==="
echo "enwik8:  $ENWIK8"
echo "budget:  ${BUDGET}s per config"
echo ""

# Phase 1: Independent experiments (01-06)
for exp in "$REPO_ROOT"/experiments/0{1,2,3,4,5,6}_*/; do
    echo "=== $(basename "$exp") ==="
    bash "$exp/run.sh" "$ENWIK8" --budget "$BUDGET"
    echo ""
done

# Phase 2: Select winners, generate 07-08 configs
echo "=== Promoting winners ==="
.venv/bin/python "$REPO_ROOT/analysis/promote_winners.py"
echo ""

# Phase 3: Dependent experiments (07-08)
for exp in "$REPO_ROOT"/experiments/0{7,8}_*/; do
    echo "=== $(basename "$exp") ==="
    bash "$exp/run.sh" "$ENWIK8" --budget "$BUDGET"
    echo ""
done

echo "=== All experiments complete ==="
echo "Total result files:"
find "$REPO_ROOT/experiments" -name "*.json" -path "*/results/*" | wc -l
