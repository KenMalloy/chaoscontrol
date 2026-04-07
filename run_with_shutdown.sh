#!/usr/bin/env bash
# Guaranteed shutdown: trap ensures poweroff even on crash/error
trap 'echo "=== Shutting down pod (trap) ==="; sleep 60; poweroff' EXIT

LOGFILE=/workspace/chaoscontrol/experiment_run.log
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== ChaosControl Experiment Run ==="
echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

cd /workspace/chaoscontrol
bash run_all.sh /workspace/enwik8 --budget 300

echo ""
echo "=== Run complete ==="
echo "Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
