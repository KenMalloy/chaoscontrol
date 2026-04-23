# Exp 24 Phase 0 Top-2 Confirm Candidates

Sources:

- `experiments/24_training_time_bundle/phase0_dw_sweep_4x_20260422T224040Z/summary.json`
- `experiments/24_training_time_bundle/phase0_fs_sweep_4x_20260423T005417Z/summary.json`

The merged screen has 27 raw rows and 24 unique configs. Repeated anchor cells
were deduped by config name, keeping the lower BPB.

## Merged Ranking Snapshot

| Rank | Config | Val BPB | Val docs | Tokens/sec |
| ---: | --- | ---: | ---: | ---: |
| 1 | `exp24_phase0_fs_i16a025_dw_c16i16_w010_s1337` | 1.4799531118 | 50,000 | 11,661,111.0 |
| 2 | `exp24_phase0_fs_i32a025_dw_c16i16_w010_s1337` | 1.4800433967 | 50,000 | 11,666,821.2 |
| 3 | `exp24_phase0_fs_i16a025_dw_c16i16_w025_s1337` | 1.4808458640 | 50,000 | 11,656,678.8 |
| 4 | `exp24_phase0_fs_i64a025_dw_c16i16_w010_s1337` | 1.4809925359 | 50,000 | 11,651,253.0 |
| 5 | `exp24_phase0_fs_i32a025_dw_c16i16_w025_s1337` | 1.4811754469 | 50,000 | 11,647,300.9 |
| 6 | `exp24_phase0_fs_i64a025_dw_c16i16_w025_s1337` | 1.4813208403 | 50,000 | 11,644,212.3 |
| 7 | `exp24_phase0_fs_i16a025_dw_c8i8_w010_s1337` | 1.4828674705 | 50,000 | 10,948,014.3 |
| 8 | `exp24_phase0_fs_i32a025_dw_c8i8_w010_s1337` | 1.4828987573 | 50,000 | 10,971,585.1 |
| 9 | `exp24_phase0_fs_i64a050_dw_c16i16_w010_s1337` | 1.4833425878 | 50,000 | 11,669,199.6 |
| 10 | `exp24_phase0_fs_i64a025_dw_c8i8_w010_s1337` | 1.4836206251 | 50,000 | 10,935,843.0 |
| 11 | `exp24_phase0_fs_i32a050_dw_c16i16_w010_s1337` | 1.4841480759 | 50,000 | 11,656,061.6 |
| 12 | `exp24_phase0_fs_i32a050_dw_c16i16_w025_s1337` | 1.4843175148 | 50,000 | 11,644,878.7 |

## Top-2 for Confirmation

The absolute lowest single-seed BPB is `fs_i16a025_dw_c16i16_w010`, but the
top cluster is well inside the plan's `<0.005` BPB near-tie band. Applying the
preregistered tie-break, confirmation carries the less aggressive schedules:
larger fast/slow intervals, same smaller DW weight.

### Candidate A

- Config label: `fs_i32a025_dw_c16i16_w010`
- Screening source: Phase 0 FS sweep
- Screening BPB: `1.4800433967`
- `fast_slow_interval=32`
- `fast_slow_alpha=0.25`
- `fast_slow_eval_copy=slow`
- `dreamworld_cache_interval=16`
- `dreamworld_interval=16`
- `dreamworld_weight=0.10`
- `dreamworld_prefix_tokens=128`
- `dreamworld_replay_tokens=64`
- `dreamworld_replay_batch_size=128`
- `dreamworld_buffer_size=16`
- `dreamworld_min_size=2`
- `dreamworld_max_age_steps=256`

### Candidate B

- Config label: `fs_i64a025_dw_c16i16_w010`
- Screening source: Phase 0 FS sweep
- Screening BPB: `1.4809925359`
- `fast_slow_interval=64`
- `fast_slow_alpha=0.25`
- `fast_slow_eval_copy=slow`
- `dreamworld_cache_interval=16`
- `dreamworld_interval=16`
- `dreamworld_weight=0.10`
- `dreamworld_prefix_tokens=128`
- `dreamworld_replay_tokens=64`
- `dreamworld_replay_batch_size=128`
- `dreamworld_buffer_size=16`
- `dreamworld_min_size=2`
- `dreamworld_max_age_steps=256`

## Tie-Break Rationale

- Candidate A is `0.000090` BPB behind the absolute best while keeping the
  original fast/slow anchor cadence (`interval=32`) and reducing alpha to `0.25`.
- Candidate B is `0.001039` BPB behind the absolute best and is the least
  aggressive `dw_w010`/`alpha=0.25` schedule in the near-tie cluster.
- The `dw_w025` variants are also close, but the plan's tie-break prefers the
  smaller DW weight when BPB differences are below `0.005`.
- `alpha=0.50` variants are not selected: they are within the broad near-tie
  band for some rows, but consistently trail the matching `alpha=0.25` rows.
