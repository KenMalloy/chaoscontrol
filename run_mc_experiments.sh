#!/usr/bin/env bash
# Run only the Monte Carlo metabolic gate experiments
trap 'echo "=== Shutting down pod (trap) ==="; sleep 120; poweroff' EXIT

LOGFILE=/workspace/chaoscontrol/experiment_mc_run.log
exec > >(tee -a "$LOGFILE") 2>&1

REPO_ROOT=/workspace/chaoscontrol
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
ENWIK8=/workspace/enwik8
BUDGET=300

echo "=== Monte Carlo Metabolic Gate Experiments ==="
echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

for cfg in experiments/06_metabolic_gate/configs/mc_*.yaml; do
    name="$(basename "$cfg" .yaml)"
    echo "  Running $name..."
    .venv/bin/python -m chaoscontrol.runner \
        --config "$cfg" \
        --enwik8-path "$ENWIK8" \
        --budget "$BUDGET" \
        --output-json "experiments/06_metabolic_gate/results/${name}.json"
done

echo ""
echo "=== MC experiments complete ==="
echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
