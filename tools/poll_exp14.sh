#!/usr/bin/env bash
# =============================================================================
# poll_exp14.sh — Monitor Experiment 14, sync results, auto-stop on completion.
#
# Usage:
#   bash tools/poll_exp14.sh <pod_id> <expected_results> [phase_label]
#
# Cron (every 5 min):
#   */5 * * * * bash /path/to/chaoscontrol/tools/poll_exp14.sh <pod_id> 91 "Phase A"
#
# The script:
#   1. SSHs into the pod, counts .json and .failed results
#   2. Checks GPU utilization (catches idle-but-running pods)
#   3. Syncs results back via rsync
#   4. On completion: final sync, stops the pod, removes itself from crontab
#   5. On stall (no progress in 3 polls = 15 min): sends alert
#   6. Logs everything to tools/poll_exp14.log
# =============================================================================
set -euo pipefail

POD_ID="${1:?Usage: poll_exp14.sh <pod_id> <expected_results> [phase_label]}"
EXPECTED="${2:?Usage: poll_exp14.sh <pod_id> <expected_results> [phase_label]}"
PHASE_LABEL="${3:-Exp14}"

REPO="$HOME/Local Documents/Developer/chaoscontrol"
LOG="$REPO/tools/poll_exp14.log"
STALL_FILE="$REPO/tools/.exp14_last_count"
REMOTE_RESULTS="/workspace/chaoscontrol/experiments/14_vram_buffer/results/"
LOCAL_RESULTS="$REPO/experiments/14_vram_buffer/results/"

# Get SSH info from runpodctl
SSH_INFO=$(runpodctl ssh info "$POD_ID" -o json 2>/dev/null) || {
    echo "$(date): ERROR: cannot get ssh info for $POD_ID (pod may be stopped)" >> "$LOG"
    exit 0
}

POD_IP=$(echo "$SSH_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['ip'])" 2>/dev/null) || {
    echo "$(date): ERROR: pod not ready yet" >> "$LOG"
    exit 0
}
POD_PORT=$(echo "$SSH_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['port'])")
SSH_KEY=$(echo "$SSH_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['ssh_key']['path'])")

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"

echo "$(date): polling $POD_ID ($PHASE_LABEL, expecting $EXPECTED)..." >> "$LOG"

# --- Count results ---
COUNTS=$(ssh $SSH_OPTS -i "$SSH_KEY" -p "$POD_PORT" root@"$POD_IP" "
    cd $REMOTE_RESULTS 2>/dev/null || { echo '0 0'; exit 0; }
    DONE=\$(ls *.json 2>/dev/null | grep -v summary | wc -l)
    FAIL=\$(ls *.failed 2>/dev/null | wc -l)
    echo \"\$DONE \$FAIL\"
" 2>/dev/null) || {
    echo "$(date): ERROR: SSH failed" >> "$LOG"
    exit 0
}

DONE=$(echo "$COUNTS" | awk '{print $1}')
FAILED=$(echo "$COUNTS" | awk '{print $2}')
TOTAL=$((DONE + FAILED))

echo "$(date): $PHASE_LABEL: $DONE done, $FAILED failed, $TOTAL/$EXPECTED" >> "$LOG"

# --- GPU utilization check ---
GPU_UTIL=$(ssh $SSH_OPTS -i "$SSH_KEY" -p "$POD_PORT" root@"$POD_IP" \
    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1" 2>/dev/null) || GPU_UTIL="?"
echo "$(date): GPU util: ${GPU_UTIL}%" >> "$LOG"

# --- Stall detection ---
LAST_COUNT=0
if [ -f "$STALL_FILE" ]; then
    LAST_COUNT=$(cat "$STALL_FILE")
fi
echo "$TOTAL" > "$STALL_FILE"

if [ "$TOTAL" -eq "$LAST_COUNT" ] && [ "$TOTAL" -gt 0 ] && [ "$TOTAL" -lt "$EXPECTED" ]; then
    # Read stall counter
    STALL_COUNTER_FILE="$REPO/tools/.exp14_stall_count"
    STALL_COUNT=0
    if [ -f "$STALL_COUNTER_FILE" ]; then
        STALL_COUNT=$(cat "$STALL_COUNTER_FILE")
    fi
    STALL_COUNT=$((STALL_COUNT + 1))
    echo "$STALL_COUNT" > "$STALL_COUNTER_FILE"

    if [ "$STALL_COUNT" -ge 3 ]; then
        echo "$(date): STALL DETECTED — no progress in $((STALL_COUNT * 5)) min ($TOTAL/$EXPECTED)" >> "$LOG"
        osascript -e "display notification \"$PHASE_LABEL STALLED at $TOTAL/$EXPECTED — no progress in $((STALL_COUNT * 5)) min\" with title \"ChaosControl\" sound name \"Basso\"" 2>/dev/null || true
    fi
else
    # Reset stall counter on progress
    echo "0" > "$REPO/tools/.exp14_stall_count"
fi

# --- Sync results ---
mkdir -p "$LOCAL_RESULTS"
rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS -i $SSH_KEY -p $POD_PORT" \
    root@"$POD_IP":"$REMOTE_RESULTS" "$LOCAL_RESULTS" >> "$LOG" 2>&1 || true

# --- Completion check ---
if [ "$TOTAL" -ge "$EXPECTED" ]; then
    echo "$(date): $PHASE_LABEL COMPLETE. $DONE done, $FAILED failed." >> "$LOG"

    # Final sync — grab logs too
    rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS -i $SSH_KEY -p $POD_PORT" \
        root@"$POD_IP":"$REMOTE_RESULTS" "$LOCAL_RESULTS" >> "$LOG" 2>&1 || true

    # Stop the pod
    echo "$(date): Stopping pod $POD_ID..." >> "$LOG"
    runpodctl pod stop "$POD_ID" >> "$LOG" 2>&1 || true
    echo "$(date): Pod stopped." >> "$LOG"

    # Clean up lease
    python3 "$REPO/tools/runpod.py" stop "$POD_ID" >> "$LOG" 2>&1 || true

    # Remove from crontab
    crontab -l 2>/dev/null | grep -v "poll_exp14" | crontab - 2>/dev/null || true
    echo "$(date): Removed from crontab." >> "$LOG"

    # Clean up stall tracking files
    rm -f "$STALL_FILE" "$REPO/tools/.exp14_stall_count"

    # Notify
    osascript -e "display notification \"$PHASE_LABEL complete! $DONE/$EXPECTED done, $FAILED failed. Pod stopped.\" with title \"ChaosControl\" sound name \"Glass\"" 2>/dev/null || true

    echo "$(date): === $PHASE_LABEL FINISHED ===" >> "$LOG"
fi
