# Exp 24 Phase 0 Base Lock

Locked config: `experiments/24_training_time_bundle/configs/exp24_base.yaml`

Primary lock source:
`experiments/24_training_time_bundle/phase0_fastslow_only_control_4x_20260423T193800Z/summary.json`

Comparison source:
`experiments/24_training_time_bundle/phase0_confirm_4x_20260423T044539Z/summary.json`

## Decision

Lock the Phase 0 base to the matched 4x fast/slow-only control:
`control_fastslow_only_i64a025`.

This supersedes the earlier stacked lock. The matched control showed that, on
the actual 4x H100 / 600s / 50k-doc full-val contract, `fast_slow` alone beats
both confirmed `fast_slow + dreamworld` candidates. Operationally, optimizer
work should ride on the fast/slow-only base. Dreamworld and sleep remain real
mechanism ideas, but they are not part of the locked static training stack.

## Fast/Slow-Only Control

| Seed | Config | Val BPB | Val docs | Tokens/sec |
| ---: | --- | ---: | ---: | ---: |
| 1337 | `control_fastslow_only_i64a025` | 1.4794907444 | 50,000 | 12,493,852.5 |
| 2674 | `control_fastslow_only_i64a025` | 1.4793956205 | 50,000 | 12,470,956.8 |
| 4011 | `control_fastslow_only_i64a025` | 1.4788831134 | 50,000 | 12,480,749.4 |

## Aggregate

| Config | Mean BPB | Sample stddev | Min BPB | Max BPB |
| --- | ---: | ---: | ---: | ---: |
| `control_fastslow_only_i64a025` | 1.4792564928 | 0.0003268352 | 1.4788831134 | 1.4794907444 |
| Confirm A: `fs_i32a025_dw_c16i16_w010` | 1.4805536442 | 0.0004913195 | 1.4800138840 | 1.4809748156 |
| Confirm B: `fs_i64a025_dw_c16i16_w010` | 1.4810305861 | 0.0002147420 | 1.4808310995 | 1.4812578753 |

Mean deltas versus the locked control:

- Confirm A minus control: `+0.0012971514`
- Confirm B minus control: `+0.0017740934`

## Locked Knobs

- `fast_slow_enabled=true`
- `fast_slow_interval=64`
- `fast_slow_alpha=0.25`
- `fast_slow_eval_copy=slow`
- `dreamworld_enabled=false`
- `dreamworld_cache_interval=0`
- `dreamworld_interval=0`
- `dreamworld_weight=0.0`
- `dreamworld_replay_batch_size=0`
- `event_sleep_enabled=false`
- `event_sleep_weight=0.0`

## Interpretation

- Dreamworld is still a real mechanism: it beat the plain baseline in first-wave
  screening.
- The non-synergy here is artifact-specific. `fast_slow` is scored as the slow
  EMA artifact, while Dreamworld and sleep are training-time mechanisms whose
  gains did not survive that artifact contract cleanly enough in the 600-second
  run.
- Treat this as an operational lock for the static training stack, not as a
  final metaphysical claim that Dreamworld or sleep are useless.

## Caveats

- This lock is tuned on 4x H100. If the first 8x optimizer run drifts from the
  expected BPB regime, revisit the base at 8x rather than assuming invariance.
- Historical Phase 0 sweep artifacts remain in-tree for reproducibility. They
  are not deleted, but they no longer define the base config.
- All full-validation summaries cited above scored 50,000 documents and the
  run summaries reported 0 errors.
