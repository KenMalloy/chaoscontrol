#!/usr/bin/env bash
# Resume from experiment 04 onwards (01-03 already complete)
trap 'echo "=== Shutting down pod (trap) ==="; sleep 120; poweroff' EXIT

LOGFILE=/workspace/chaoscontrol/experiment_run_resume.log
exec > >(tee -a "$LOGFILE") 2>&1

REPO_ROOT=/workspace/chaoscontrol
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
ENWIK8=/workspace/enwik8
BUDGET=300

echo "=== ChaosControl RESUME Run ==="
echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Resuming from experiment 04 (01-03 already done)"
echo ""

# Phase 1 remainder: experiments 04-06
for exp in "$REPO_ROOT"/experiments/0{4,5,6}_*/; do
    echo "=== $(basename "$exp") ==="
    bash "$exp/run.sh" "$ENWIK8" --budget "$BUDGET"
    echo ""
done

# Phase 2: Select winners
echo "=== Promoting winners ==="
.venv/bin/python "$REPO_ROOT/analysis/promote_winners.py"
echo ""

# Phase 3: experiments 07-08
for exp in "$REPO_ROOT"/experiments/0{7,8}_*/; do
    echo "=== $(basename "$exp") ==="
    bash "$exp/run.sh" "$ENWIK8" --budget "$BUDGET"
    echo ""
done

echo "=== All experiments complete ==="
echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Total result files:"
find "$REPO_ROOT/experiments" -name "*.json" -path "*/results/*" | wc -l
