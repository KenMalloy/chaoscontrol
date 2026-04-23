# Exp 24 Phase 0 Base Lock

Locked config: `experiments/24_training_time_bundle/configs/exp24_base.yaml`

Confirm source:
`experiments/24_training_time_bundle/phase0_confirm_4x_20260423T044539Z/summary.json`

## Confirm BPB Table

| Candidate | Seed | Config | Val BPB | Val docs | Tokens/sec |
| --- | ---: | --- | ---: | ---: | ---: |
| A | 1337 | `fs_i32a025_dw_c16i16_w010` | 1.4800138840 | 50,000 | 11,645,639.6 |
| A | 2674 | `fs_i32a025_dw_c16i16_w010` | 1.4806722329 | 50,000 | 11,701,243.9 |
| A | 4011 | `fs_i32a025_dw_c16i16_w010` | 1.4809748156 | 50,000 | 11,651,801.0 |
| B | 1337 | `fs_i64a025_dw_c16i16_w010` | 1.4810027836 | 50,000 | 11,649,231.0 |
| B | 2674 | `fs_i64a025_dw_c16i16_w010` | 1.4812578753 | 50,000 | 11,674,110.3 |
| B | 4011 | `fs_i64a025_dw_c16i16_w010` | 1.4808310995 | 50,000 | 11,659,950.5 |

## Aggregate

| Candidate | Mean BPB | Sample stddev | Min BPB | Max BPB |
| --- | ---: | ---: | ---: | ---: |
| A: `fs_i32a025_dw_c16i16_w010` | 1.4805536442 | 0.0004913195 | 1.4800138840 | 1.4809748156 |
| B: `fs_i64a025_dw_c16i16_w010` | 1.4810305861 | 0.0002147420 | 1.4808310995 | 1.4812578753 |

## Decision

Lock Candidate B: `fs_i64a025_dw_c16i16_w010`.

Candidate A has the lower 3-seed mean by `0.0004769419` BPB. That gap is
inside the confirm noise scale: it is below Candidate A's sample stddev
(`0.0004913195`) and below the pooled two-candidate sample scale
(`~0.000536`). Per the preregistered tiebreaker, the lock goes to Candidate B
because it has lower seed-to-seed stddev and the simpler, less aggressive
fast/slow cadence (`interval=64` instead of `32`) while keeping the same
`alpha=0.25` and DW settings.

Locked knobs:

- `fast_slow_enabled=true`
- `fast_slow_interval=64`
- `fast_slow_alpha=0.25`
- `fast_slow_eval_copy=slow`
- `dreamworld_enabled=true`
- `dreamworld_cache_interval=16`
- `dreamworld_interval=16`
- `dreamworld_weight=0.10`
- `dreamworld_prefix_tokens=128`
- `dreamworld_replay_tokens=64`
- `dreamworld_replay_batch_size=128`
- `dreamworld_buffer_size=16`
- `dreamworld_min_size=2`
- `dreamworld_max_age_steps=256`

## Caveats

- The Phase 0 DW sweep winner sat on the original grid boundary at interval 16
  and weight 0.10. A future tuning pass could test interval 32 or a lower DW
  weight, but this plan intentionally locks the preregistered base before that
  expansion.
- The confirm candidates are extremely close. The lock is a stability and
  simplicity choice, not a claim that Candidate B has a decisive mean-BPB edge.
- All confirm full-validation summaries scored 50,000 documents and
  `summary.json` reported 0 errors.
