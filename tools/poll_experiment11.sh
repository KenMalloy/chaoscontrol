#!/usr/bin/env bash
# Poll experiment 11 progress on pod asvnf6bwu59pen every 5 minutes.
# Harvests results when done, stops pod, and removes itself from crontab.
set -euo pipefail

POD_ID="asvnf6bwu59pen"
POD_IP="63.141.33.5"
POD_PORT="22114"
SSH_KEY="$HOME/.ssh/id_runpod"
REPO="$HOME/Local Documents/Developer/chaoscontrol"
RESULTS_REMOTE="/workspace/chaoscontrol/experiments/11_sleep_cycle/results/"
RESULTS_LOCAL="$REPO/experiments/11_sleep_cycle/results/"
LOG="$REPO/tools/poll_experiment11.log"
TOTAL_RUNS=63

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"

echo "$(date): polling..." >> "$LOG"

# Count completed result JSONs on pod
DONE=$(ssh $SSH_OPTS -i "$SSH_KEY" -p "$POD_PORT" root@"$POD_IP" \
    "ls /workspace/chaoscontrol/experiments/11_sleep_cycle/results/*.json 2>/dev/null | grep -v summary | wc -l" 2>/dev/null || echo "0")

echo "$(date): $DONE / $TOTAL_RUNS complete" >> "$LOG"

# Sync results locally
mkdir -p "$RESULTS_LOCAL"
rsync -az --progress -e "ssh $SSH_OPTS -i $SSH_KEY -p $POD_PORT" \
    root@"$POD_IP":"$RESULTS_REMOTE" "$RESULTS_LOCAL" >> "$LOG" 2>&1 || true

if [ "$DONE" -ge "$TOTAL_RUNS" ]; then
    echo "$(date): ALL DONE. Harvesting final results and stopping pod." >> "$LOG"

    # Final sync
    rsync -az -e "ssh $SSH_OPTS -i $SSH_KEY -p $POD_PORT" \
        root@"$POD_IP":"$RESULTS_REMOTE" "$RESULTS_LOCAL" >> "$LOG" 2>&1

    # Also grab the full log
    scp $SSH_OPTS -i "$SSH_KEY" -P "$POD_PORT" \
        root@"$POD_IP":/workspace/experiment11.log "$REPO/experiments/11_sleep_cycle/" >> "$LOG" 2>&1 || true

    # Stop the pod
    runpodctl pod stop "$POD_ID" >> "$LOG" 2>&1 || true

    # Remove ourselves from crontab
    crontab -l 2>/dev/null | grep -v poll_experiment11 | crontab - 2>/dev/null || true

    echo "$(date): Pod stopped. Cron removed. Results in $RESULTS_LOCAL" >> "$LOG"

    # Desktop notification
    osascript -e 'display notification "Experiment 11 complete! 63/63 runs done." with title "ChaosControl"' 2>/dev/null || true
fi
