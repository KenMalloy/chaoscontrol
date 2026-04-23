# Exp 24 Phase 0 Dreamworld Sweep Winner

Source run:
`experiments/24_training_time_bundle/phase0_dw_sweep_4x_20260422T224040Z/summary.json`

All arms use the Phase 0 fast/slow baseline settings pinned for this sweep:
`fast_slow_interval=32`, `fast_slow_alpha=0.50`, `seed=1337`.

## BPB Ranking

| Rank | Config | DW cache interval | DW interval | DW weight | Val BPB | Val docs | Tokens/sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `exp24_phase0_fs_i32a050_dw_c16i16_w010_s1337` | 16 | 16 | 0.10 | 1.4841480759 | 50,000 | 11,656,061.6 |
| 2 | `exp24_phase0_fs_i32a050_dw_c16i16_w025_s1337` | 16 | 16 | 0.25 | 1.4843243149 | 50,000 | 11,662,024.3 |
| 3 | `exp24_phase0_fs_i32a050_dw_c8i8_w010_s1337` | 8 | 8 | 0.10 | 1.4871007630 | 50,000 | 10,979,809.5 |
| 4 | `exp24_phase0_fs_i32a050_dw_c16i16_w050_s1337` | 16 | 16 | 0.50 | 1.4877538739 | 50,000 | 11,656,038.6 |
| 5 | `exp24_phase0_fs_i32a050_dw_c8i8_w025_s1337` | 8 | 8 | 0.25 | 1.4889585836 | 50,000 | 10,929,375.2 |
| 6 | `exp24_phase0_fs_i32a050_dw_c4i4_w010_s1337` | 4 | 4 | 0.10 | 1.4915933925 | 50,000 | 9,732,857.4 |
| 7 | `exp24_phase0_fs_i32a050_dw_c4i4_w025_s1337` | 4 | 4 | 0.25 | 1.4921096561 | 50,000 | 9,869,289.5 |
| 8 | `exp24_phase0_fs_i32a050_dw_c8i8_w050_s1337` | 8 | 8 | 0.50 | 1.4925685827 | 50,000 | 10,965,006.2 |
| 9 | `exp24_phase0_fs_i32a050_dw_c4i4_w050_s1337` | 4 | 4 | 0.50 | 1.5016964592 | 50,000 | 9,754,389.5 |

## Decision

The single best arm is `exp24_phase0_fs_i32a050_dw_c16i16_w010_s1337`:

- `dreamworld_cache_interval=16`
- `dreamworld_interval=16`
- `dreamworld_weight=0.10`

However, the plan's seed-noise rule says to carry candidates within about `0.015` BPB. The top three arms are within `0.002953` BPB of each other, so Task 7 should carry all three DW candidates into the fast/slow sweep:

| Candidate | DW cache interval | DW interval | DW weight | Screening BPB |
| --- | ---: | ---: | ---: | ---: |
| `dw_c16i16_w010` | 16 | 16 | 0.10 | 1.4841480759 |
| `dw_c16i16_w025` | 16 | 16 | 0.25 | 1.4843243149 |
| `dw_c8i8_w010` | 8 | 8 | 0.10 | 1.4871007630 |

## Sanity Notes

- `summary.json` contains 9 ranked arms and 0 errors.
- Every full-validation summary scored 50,000 documents.
- No BPB exceeds the loose pathological threshold of 2.0; the max is `1.5016964592`.
- Adjacent grid cells are coherent. The largest adjacent weight change is well below 0.3 BPB.
- The single best arm is on the sweep boundary at interval 16 and weight 0.10. A future follow-up could test interval 32 or a lower DW weight, but Phase 0 proceeds with the preregistered grid and carries the top noise-band candidates forward.
