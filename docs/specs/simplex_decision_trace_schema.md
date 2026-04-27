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
| `row_type` | `VARCHAR` | `decision`, `credit`, or `skip`. |
| `status` | `VARCHAR` | `ok` / `skipped`; analysis should not infer failure from missing rows. |
| `status_reason` | `VARCHAR` | Empty for normal rows; otherwise `outcome_status`, `invalid_slot`, `missing_weights`, `missing_decision`, `zero_advantage`, or `gerber_rejected`. |
| `gpu_step` | `BIGINT` | Producer GPU step for the query/decision; replay step on credit/skip rows. |
| `slot_id` | `UBIGINT` | Chosen cache slot id. |
| `arm` | `VARCHAR` | Matrix arm name; empty string when unavailable in-process. |
| `query_event_id` | `UBIGINT` | Rank-prefixed query id. |
| `replay_id` | `UBIGINT` | Replay/action id derived from query id and chosen rank until controller-issued ids land. |
| `source_write_id` | `UBIGINT` | Original WRITE_EVENT/candidate id for the chosen slot. |
| `selection_step` | `UBIGINT` | Producer step stamped on the decision; credit rows must match this against history. |
| `policy_version` | `UINTEGER` | Behavior policy version. |
| `outcome_status` | `UTINYINT` | Wire replay outcome status; zero on decision rows. |
| `flags` | `USMALLINT` | Reserved replay flags. |
| `write_bucket` | `TINYINT` | Token-frequency bucket for the admitted slot. |
| `slot_age_steps` | `UBIGINT` | `gpu_step - selection_step` on replay rows. |
| `n_actual` | `UINTEGER` | Number of real candidates before sentinel padding. |
| `chosen_idx` | `UTINYINT` | Chosen simplex vertex in the candidate set. |
| `selected_rank` | `UTINYINT` | Alias of `chosen_idx` for wire-event joins. |
| `p_chosen` | `FLOAT` | Behavior-policy probability assigned to the sampled vertex after valid-candidate renormalization. |
| `p_current_chosen` | `FLOAT` | Current-policy probability at replay attribution time; NaN on decision rows. |
| `p_behavior` | `FLOAT[]` | Full behavior distribution over the valid candidates. |
| `entropy` | `FLOAT` | `-sum(p_behavior * log(p_behavior))`. |
| `temperature` | `FLOAT` | Active simplex softmax temperature. |
| `entropy_beta` | `FLOAT` | Active entropy-bonus weight. |
| `teacher_score` | `FLOAT` | Heuristic score for the chosen slot, usually cosine times utility. |
| `controller_logit` | `FLOAT` | Controller logit for the chosen vertex. |
| `ce_before_replay` | `FLOAT` | Replay CE before the optimizer step; NaN on decision-only rows. |
| `ce_after_replay` | `FLOAT` | Replay CE after the replay forward/step; NaN on decision-only rows. |
| `ce_delta_raw` | `FLOAT` | `ce_before_replay - ce_after_replay`; NaN on decision-only rows. |
| `bucket_baseline` | `FLOAT` | Running bucket EMA baseline. |
| `reward_shaped` | `FLOAT` | Raw CE delta minus bucket baseline from the wire event. |
| `gerber_weight` | `FLOAT` | Gerber gate applied at replay attribution; NaN on decision-only rows. |
| `advantage_raw` | `FLOAT` | `ce_delta_raw - bucket_baseline` before standardization; NaN on decision-only rows. |
| `advantage_standardized` | `FLOAT` | Per-bucket standardized advantage before recency/Gerber. |
| `recency_weight` | `FLOAT` | `gamma ** slot_age_steps`. |
| `advantage_pre_gerber` | `FLOAT` | Standardized and recency-decayed advantage before Gerber. |
| `advantage_final` | `FLOAT` | Advantage actually credited to policy after Gerber. |
| `gerber_threshold` | `FLOAT` | Bucket/type-local Gerber threshold. |
| `behavior_logprob_margin` | `FLOAT` | Behavior-policy categorical margin used by Gerber. |
| `current_logprob_margin` | `FLOAT` | Current-policy categorical margin used by Gerber. |
| `lambda_hxh` | `FLOAT` | Active HxH residual scale after warmup/clipping. |
| `selection_mode` | `VARCHAR` | `argmax`, `sample`, or equivalent mode. |
| `selection_seed` | `BIGINT` | CPU generator seed when stochastic selection is used; `-1` when unavailable. |
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

- Rows are emitted through a bounded async writer. If the writer falls behind,
  rows drop and `SimplexLearnerTelemetry.simplex_trace_drops` increments; the
  controller/replay path must not wait on disk.
- Gerber statistics are maintained per `(bucket, action_type)`. A global
  statistic may be logged for diagnostics, but it is not the correction used
  by the learner.
- `p_behavior` is always over the valid candidates, not the zero-padded
  vertices. This keeps entropy and Gerber log-prob margins comparable across
  short candidate sets.
- `feature_manifest_hash` changes whenever the meaning or ordering of `V`,
  `E`, or `simplex_features` changes; it is the join key for analysis code
  that reconstructs controller inputs.
