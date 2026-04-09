#!/usr/bin/env bash
# Watchdog: monitor a running experiment for crashes.
# If new .failed files appear, alert immediately.
# Usage: watchdog_experiment.sh <results_dir> <log_file>
set -euo pipefail

RESULTS="${1:?Usage: watchdog_experiment.sh <results_dir> <log_file>}"
LOG="${2:?Usage: watchdog_experiment.sh <results_dir> <log_file>}"
POLL_SECONDS=120

LAST_FAIL_COUNT=0
if [ -d "$RESULTS" ]; then
    LAST_FAIL_COUNT=$(ls "$RESULTS"/*.failed 2>/dev/null | wc -l | tr -d ' ')
fi

echo "$(date): Watchdog started. Monitoring $RESULTS (initial failures: $LAST_FAIL_COUNT)" >> "$LOG"

while true; do
    sleep "$POLL_SECONDS"

    # Count current failures
    FAIL_COUNT=0
    if [ -d "$RESULTS" ]; then
        FAIL_COUNT=$(ls "$RESULTS"/*.failed 2>/dev/null | wc -l | tr -d ' ')
    fi

    # Count successes
    JSON_COUNT=0
    if [ -d "$RESULTS" ]; then
        JSON_COUNT=$(ls "$RESULTS"/*.json 2>/dev/null | grep -v summary | wc -l | tr -d ' ')
    fi

    if [ "$FAIL_COUNT" -gt "$LAST_FAIL_COUNT" ]; then
        NEW_FAILS=$((FAIL_COUNT - LAST_FAIL_COUNT))
        echo "$(date): ALERT — $NEW_FAILS new failure(s)! ($JSON_COUNT succeeded, $FAIL_COUNT failed)" >> "$LOG"

        # Show the error from the most recent failure
        LATEST=$(ls -t "$RESULTS"/*.failed 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "  Latest failure: $(cat "$LATEST")" >> "$LOG"
        fi

        # Desktop notification
        osascript -e "display notification \"$NEW_FAILS run(s) crashed! Check $LOG\" with title \"ChaosControl CRASH\"" 2>/dev/null || true

        LAST_FAIL_COUNT="$FAIL_COUNT"
    else
        echo "$(date): OK — $JSON_COUNT succeeded, $FAIL_COUNT failed" >> "$LOG"
    fi

    # If no python runners are alive, we're done
    if ! pgrep -f "chaoscontrol.runner" > /dev/null 2>&1; then
        # Check if orchestrator is also gone
        if ! pgrep -f "run_sleep_ablation\|run_mamba2_baseline\|run_polyphasic" > /dev/null 2>&1; then
            echo "$(date): No runners alive. Watchdog exiting." >> "$LOG"
            osascript -e "display notification \"Experiment finished. $JSON_COUNT succeeded, $FAIL_COUNT failed.\" with title \"ChaosControl\"" 2>/dev/null || true
            exit 0
        fi
    fi
done
