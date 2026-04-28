# CRCT mailbox full-600 run

**Run:** 2026-04-28 on pod `3mlt2x4k1byzyh` (4x H100). Matrix `crct_v1`, arms `arm_a_fastslow_control` and `arm_b_crct_controller`, seeds `1337`, `4011`, `7331`, 600s train budget.

**Code under test:** `055945c`, `73f2b66`, `a982de3`.

## Why This Run Exists

The collective CRCT teacher transport violated the trunk-throughput contract: even with `async_op=True`, rank 0 could block at collective launch when rank 3 was busy scoring the teacher. The mailbox transport moves request/result exchange to a non-collective `/dev/shm` sidecar so train ranks never rendezvous with the memory rank on the hot path.

## Verification

On the H100 pod, with warnings promoted to errors:

```bash
CHAOSCONTROL_DIAG_SCAN_BACKEND=chunked \
PYTHONPATH=/workspace/chaoscontrol/src \
/workspace/venv/bin/python -m pytest \
  /workspace/chaoscontrol/tests/test_crct_runner_integration.py \
  /workspace/chaoscontrol/tests/test_cache_utility.py \
  /workspace/chaoscontrol/tests/test_model.py \
  /workspace/chaoscontrol/tests/test_exp24_training_bundle.py \
  -W error -q --tb=short
```

Result: `148 passed in 28.46s`.

## Matrix Results

Raw artifacts were harvested locally under:

`experiments/24_training_time_bundle/results/crct_lead_20260428_full600_mailbox/`

| arm | seed | steps | aggregate tok/s | per-train-rank tok/s | final loss |
|---|---:|---:|---:|---:|---:|
| control | 1337 | 3576 | 12,528,826 | 3,132,207 | 3.73924 |
| control | 4011 | 3572 | 12,517,422 | 3,129,355 | 3.74335 |
| control | 7331 | 3568 | 12,497,378 | 3,124,345 | 3.71580 |
| CRCT | 1337 | 3568 | 9,364,562 | 3,121,521 | 3.69555 |
| CRCT | 4011 | 3564 | 9,353,947 | 3,117,982 | 3.72655 |
| CRCT | 7331 | 3564 | 9,352,789 | 3,117,596 | 3.70523 |

Mean control: `3572.0` steps, `3,128,636` per-train-rank tok/s, final loss `3.73279`.

Mean CRCT: `3565.3` steps, `3,119,033` per-train-rank tok/s, final loss `3.70911`.

The headline throughput ratio is `0.9969x` per active train rank. The lower aggregate CRCT tok/s is expected: CRCT uses 3 train ranks plus 1 memory rank, while control trains on all 4 ranks. The trunk no longer pays the 6x slowdown seen with collective slot broadcast.

## Transport Health

For all three CRCT seeds:

| metric | value |
|---|---:|
| `teacher_transport_mode` | `async_rank0_memory_mailbox` |
| `teacher_param_syncs` | 0 |
| `train_rank_slot_reads` | 0 |
| `train_rank_slot_writes` | 0 |
| `teacher_payloads` | 56 |
| `teacher_memory_slots` | 7144 |
| `payload_lag_steps_max` | 4 |
| coordinator transport errors | 0 |
| memory-rank transport errors | 0 |

Teacher payloads are sparse by configuration: `crct_teacher_score_interval_steps=64`, so roughly 56 payloads over 3565 steps is expected. `teacher_fail_open` remains high because most steps intentionally run without a fresh teacher payload; this is now cadence telemetry, not a transport failure signal.

## Read

The mailbox sidecar restores the design contract: the trunk SSM does not read CRCT memory, does not receive slot broadcasts, does not sync teacher params, and does not wait for the rank-3 teacher to finish scoring. The teacher is stale-but-safe and opportunistic.

Training-loss direction is favorable in this run (`3.70911` vs `3.73279` mean final loss), but this is not a held-out BPB claim. The next useful check is the same mailbox path with held-out scoring enabled, then a score-interval sweep if the teacher signal looks promising.
