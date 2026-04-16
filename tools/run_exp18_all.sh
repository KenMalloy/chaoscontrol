#!/bin/bash
# run_exp18_all.sh — sequential Exp 18 Tests 4/3/5/6/7 on a 4-GPU pod.
#
# Meant to run ON the pod via nohup from tools/pod_session.sh:
#
#   ssh <pod> 'cd /workspace/chaoscontrol && \
#     nohup bash tools/run_exp18_all.sh > /workspace/chaoscontrol/run_exp18_all.log 2>&1 &'
#
# The tests are sequential because each one claims all available GPUs
# for its chosen launch pattern (Tests 4/5/6/7 are DDP; Test 3 is
# parallel-single-GPU across all GPUs). Running them concurrently would
# contend for the same physical devices.
#
# Run order and dependencies:
#
#   Test 4  establishes the ws=1 baseline that Test 5's Stage 2 gate
#           cross-checks against. MUST run first.
#   Test 3  independent; slotted between Test 4 and Test 5 because
#           it can't run concurrently with anything on a 4-GPU pod.
#   Test 5  reads Test 4's results_test4/ws1_s*.json.
#   Test 6  reads Test 5's results_test5/test5_summary.json and
#           refuses to run on a provisional/missing winner.
#   Test 7  same dependency on Test 5 as Test 6.
#
# On failure: set -e aborts the chain at the first non-zero exit so
# downstream tests don't run on contaminated state. Partial results
# from earlier tests are preserved in their results_test* directories
# regardless. Re-run the wrapper and each test's launch_matrix will
# skip already-completed seeds via its existing idempotency check.
#
# Cost estimate on 4xH100 at $8-10/hr:
#   Test 4:  ~70 min  (ws=1 parallel + ws=2 2-slot + ws=4 serial x 4 seeds)
#   Test 3:  ~30 min  (12 runs / 4 parallel = 3 waves)
#   Test 5:  ~60 min  (12 runs / 2 slots = 6 waves)
#   Test 6:  ~60 min
#   Test 7:  ~60 min
#   Total:   ~4.7 h  ~= $40-$50

set -euo pipefail

REPO="${REPO:-/workspace/chaoscontrol}"
cd "$REPO"

DATA="${DATA:-$REPO/baselines/parameter_golf/datasets/fineweb10B_sp16384}"
TOK="${TOK:-$REPO/baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model}"
NUM_GPUS="${NUM_GPUS:-4}"
DDP_SLOTS="${DDP_SLOTS:-2}"
BUDGET="${BUDGET:-600}"

TESTS_DIR="$REPO/experiments/18_throughput_levers"

echo "============================================="
echo "Exp 18 full-matrix run"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Data:    $DATA"
echo "GPUs:    $NUM_GPUS (DDP slots at ws=2: $DDP_SLOTS)"
echo "Budget:  ${BUDGET}s per training run"
echo "============================================="

run_test() {
    local test_num="$1"
    local results_dir="$2"
    shift 2
    local orch_log="$results_dir/orchestrator.log"
    mkdir -p "$results_dir"
    echo
    echo ">>> Starting Test $test_num at $(date -u '+%H:%M:%S UTC')"
    echo "    log: $orch_log"
    if python "$@" > "$orch_log" 2>&1; then
        echo ">>> Test $test_num DONE at $(date -u '+%H:%M:%S UTC')"
        tail -3 "$orch_log" | sed 's/^/    /'
    else
        local ret=$?
        echo ">>> Test $test_num FAILED with exit code $ret at $(date -u '+%H:%M:%S UTC')"
        echo "    last 20 lines of $orch_log:"
        tail -20 "$orch_log" | sed 's/^/        /'
        return $ret
    fi
}

# Test 4 is skipped: ws=1 and ws=2 results from the 2026-04-15 launch
# are already on disk in results_test4/. Test 5 reads ws=1 seed JSONs
# directly via _load_ws1_seed_bpbs. The ws=4 condition was dropped (see
# run_exp18_test4.py CONDITIONS comment for the chunked-CE / DDP NCCL
# deadlock). Test 3 was dropped from this matrix earlier.

run_test 5 "$TESTS_DIR/results_test5" \
    "$TESTS_DIR/run_exp18_test5.py" \
    --data-path "$DATA" \
    --sp-model-path "$TOK" \
    --num-slots "$DDP_SLOTS" \
    --budget "$BUDGET"

run_test 6 "$TESTS_DIR/results_test6" \
    "$TESTS_DIR/run_exp18_test6.py" \
    --data-path "$DATA" \
    --sp-model-path "$TOK" \
    --num-slots "$DDP_SLOTS" \
    --budget "$BUDGET"

run_test 7 "$TESTS_DIR/results_test7" \
    "$TESTS_DIR/run_exp18_test7.py" \
    --data-path "$DATA" \
    --sp-model-path "$TOK" \
    --num-slots "$DDP_SLOTS" \
    --budget "$BUDGET"

echo
echo "============================================="
echo "Exp 18 full matrix COMPLETE"
echo "Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================="
