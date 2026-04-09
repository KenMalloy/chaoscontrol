#!/usr/bin/env bash
# Remote watchdog: SSH to pod every 2 min, check for crashes.
# Alerts via macOS notification if GPU utilization drops to zero
# while the orchestrator is still running (= crash in progress).
set -euo pipefail

POD_IP="63.141.33.5"
POD_PORT="22114"
SSH_KEY="$HOME/.ssh/id_runpod"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"
LOG="$HOME/Local Documents/Developer/chaoscontrol/tools/watchdog.log"
RESULTS_REMOTE="/workspace/chaoscontrol/experiments/11_sleep_cycle/results"

echo "$(date): Remote watchdog started" >> "$LOG"

while true; do
    sleep 120

    # Count results
    JSON_COUNT=$(ssh $SSH_OPTS -i "$SSH_KEY" -p "$POD_PORT" root@"$POD_IP" \
        "ls $RESULTS_REMOTE/*.json 2>/dev/null | grep -v summary | wc -l" 2>/dev/null || echo "?")

    # Check GPU utilization
    GPU_UTIL=$(ssh $SSH_OPTS -i "$SSH_KEY" -p "$POD_PORT" root@"$POD_IP" \
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ',' " 2>/dev/null || echo "?")

    # Check if orchestrator is alive
    ORCH=$(ssh $SSH_OPTS -i "$SSH_KEY" -p "$POD_PORT" root@"$POD_IP" \
        "pgrep -f run_sleep_ablation | wc -l" 2>/dev/null || echo "?")

    echo "$(date): results=$JSON_COUNT gpu=[$GPU_UTIL] orchestrator=$ORCH" >> "$LOG"

    # Alert if orchestrator is dead but we don't have 63 results
    if [ "$ORCH" = "0" ] && [ "$JSON_COUNT" != "?" ] && [ "$JSON_COUNT" -lt 63 ]; then
        echo "$(date): ALERT — orchestrator dead with only $JSON_COUNT/63 results!" >> "$LOG"
        osascript -e "display notification \"Orchestrator died at $JSON_COUNT/63! Check watchdog.log\" with title \"ChaosControl CRASH\"" 2>/dev/null || true
        exit 1
    fi

    # Done?
    if [ "$JSON_COUNT" = "63" ]; then
        echo "$(date): All 63 results complete!" >> "$LOG"
        osascript -e "display notification \"Experiment 11 complete! 63/63.\" with title \"ChaosControl\"" 2>/dev/null || true
        exit 0
    fi
done
