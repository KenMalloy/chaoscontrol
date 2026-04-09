#!/usr/bin/env bash
# Poll Exp 13 completion and Exp 12 progress; sync results back each cycle.
# Cron: */5 * * * * /path/to/poll_exp13_exp12.sh
set -euo pipefail

POD_IP="63.141.33.5"
POD_PORT="22114"
SSH_KEY="$HOME/.ssh/id_runpod"
REPO="$HOME/Local Documents/Developer/chaoscontrol"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"

EXP13_REMOTE="/workspace/chaoscontrol/experiments/13_constants_validation/results/"
EXP13_LOCAL="$REPO/experiments/13_constants_validation/results/"
EXP12_REMOTE="/workspace/chaoscontrol/experiments/12_polyphasic_sleep/results/"
EXP12_LOCAL="$REPO/experiments/12_polyphasic_sleep/results/"
LOG="$REPO/tools/poll_exp13_exp12.log"
DONE_FILE="$REPO/tools/.exp13_done"

echo "$(date): polling..." >> "$LOG"

# --- Exp 13 ---
if [ ! -f "$DONE_FILE" ]; then
    EXP13_COUNT=$(ssh $SSH_OPTS -i "$SSH_KEY" -p "$POD_PORT" root@"$POD_IP" \
        "ls /workspace/chaoscontrol/experiments/13_constants_validation/results/*.json 2>/dev/null | grep -v summary | wc -l" 2>/dev/null || echo "0")
    echo "$(date): Exp 13: $EXP13_COUNT/182" >> "$LOG"

    mkdir -p "$EXP13_LOCAL"
    rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS -i $SSH_KEY -p $POD_PORT" \
        root@"$POD_IP":"$EXP13_REMOTE" "$EXP13_LOCAL" >> "$LOG" 2>&1 || true

    if [ "$EXP13_COUNT" -ge 182 ]; then
        echo "$(date): Exp 13 COMPLETE. Final sync done." >> "$LOG"
        touch "$DONE_FILE"
        osascript -e 'display notification "Experiment 13 complete! 182/182 runs done." with title "ChaosControl"' 2>/dev/null || true
    fi
fi

# --- Exp 12 ---
EXP12_COUNT=$(ssh $SSH_OPTS -i "$SSH_KEY" -p "$POD_PORT" root@"$POD_IP" \
    "ls /workspace/chaoscontrol/experiments/12_polyphasic_sleep/results/*.json 2>/dev/null | grep -v summary | wc -l" 2>/dev/null || echo "0")
echo "$(date): Exp 12: $EXP12_COUNT/21" >> "$LOG"

mkdir -p "$EXP12_LOCAL"
rsync -az --no-perms --no-owner --no-group -e "ssh $SSH_OPTS -i $SSH_KEY -p $POD_PORT" \
    root@"$POD_IP":"$EXP12_REMOTE" "$EXP12_LOCAL" >> "$LOG" 2>&1 || true

if [ "$EXP12_COUNT" -ge 21 ]; then
    echo "$(date): Exp 12 COMPLETE. Final sync done." >> "$LOG"
    # Also grab the full log
    scp $SSH_OPTS -i "$SSH_KEY" -P "$POD_PORT" \
        root@"$POD_IP":/workspace/experiment12.log "$REPO/experiments/12_polyphasic_sleep/" >> "$LOG" 2>&1 || true
    osascript -e 'display notification "Experiment 12 complete! 21/21 runs done." with title "ChaosControl"' 2>/dev/null || true
    # Remove ourselves from crontab
    crontab -l 2>/dev/null | grep -v poll_exp13_exp12 | crontab - 2>/dev/null || true
    echo "$(date): Removed from crontab." >> "$LOG"
fi
