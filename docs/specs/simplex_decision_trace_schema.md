# Simplex Decision Trace Schema

Status: locked for the Phase-S simplex controller instrumentation.

Purpose: make the controller trace directly ingestible by DuckDB without a
post-pod rewrite. Rows are NDJSON at write time; DuckDB readers should cast
using the column contract below and may partition files by `arm` and a coarse
`gpu_step` bucket.

Recommended path layout:

```text
decision_trace/arm=<arm>/gpu_step_bucket=<floor(gpu_step / 10000)>/part-*.ndjson
```

Required columns:

| Column | DuckDB type | Meaning |
| --- | --- | --- |
| `gpu_step` | `BIGINT` | Producer GPU step for the query/decision. |
| `slot_id` | `UBIGINT` | Chosen cache slot id. |
| `arm` | `VARCHAR` | Matrix arm name; empty string when unavailable in-process. |
| `query_event_id` | `UBIGINT` | Rank-prefixed query id. |
| `replay_id` | `UBIGINT` | Replay/action id derived from query id and chosen rank until controller-issued ids land. |
| `source_write_id` | `UBIGINT` | Original WRITE_EVENT/candidate id for the chosen slot. |
| `chosen_idx` | `UTINYINT` | Chosen simplex vertex in the candidate set. |
| `selected_rank` | `UTINYINT` | Alias of `chosen_idx` for wire-event joins. |
| `p_chosen` | `FLOAT` | Behavior-policy probability assigned to the sampled vertex after valid-candidate renormalization. |
| `p_behavior` | `FLOAT[]` | Full behavior distribution over the valid candidates. |
| `entropy` | `FLOAT` | `-sum(p_behavior * log(p_behavior))`. |
| `teacher_score` | `FLOAT` | Heuristic score for the chosen slot, usually cosine times utility. |
| `controller_logit` | `FLOAT` | Controller logit for the chosen vertex. |
| `gerber_weight` | `FLOAT` | Gerber gate applied at replay attribution; NaN on decision-only rows. |
| `advantage_raw` | `FLOAT` | `ce_delta_raw - bucket_baseline` before standardization; NaN on decision-only rows. |
| `advantage_corrected` | `FLOAT` | Standardized, recency-decayed, Gerber-gated advantage; NaN on decision-only rows. |
| `lambda_hxh` | `FLOAT` | Active HxH residual scale after warmup/clipping. |
| `feature_manifest_hash` | `VARCHAR` | Stable hash of the V/E/simplex feature manifest. |

Optional sampled columns:

| Column | DuckDB type | Meaning |
| --- | --- | --- |
| `candidate_slot_ids` | `UBIGINT[]` | Valid candidate slot ids in simplex order. |
| `candidate_scores` | `FLOAT[]` | Heuristic scores in simplex order. |
| `logits` | `FLOAT[]` | Controller logits in simplex order. |
| `V_sample` | `FLOAT[]` | Flattened `(N, K_v)` vertex features, sampled every `simplex_trace_sample_stride` rows. |
| `E_sample` | `FLOAT[]` | Flattened edge features, sampled every `simplex_trace_sample_stride` rows. |

Notes:

- Gerber statistics are maintained per `(bucket, action_type)`. A global
  statistic may be logged for diagnostics, but it is not the correction used
  by the learner.
- `p_behavior` is always over the valid candidates, not the zero-padded
  vertices. This keeps entropy and Gerber log-prob margins comparable across
  short candidate sets.
- `feature_manifest_hash` changes whenever the meaning or ordering of `V`,
  `E`, or `simplex_features` changes; it is the join key for analysis code
  that reconstructs controller inputs.
