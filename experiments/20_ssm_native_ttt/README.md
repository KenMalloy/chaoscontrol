# Exp20 SSM-Native TTT Runs

This directory contains the runnable Exp20 matrix artifacts. The goal is to
run the actual SSM-native ablation, not the old naive `adapt_set=all` envelope.

## Record Eligibility

Parameter Golf record claims must score the full fixed validation split
(`fineweb_val_*`, 50,000 documents) and finish evaluation within the 600 second
8xH100 limit. A run that times out before `docs_scored == 50000` is incomplete,
even if it produced a mean BPB for the scored prefix.

The Exp20 summaries therefore carry explicit interpretation fields:

- `requested_docs_complete`: the configured `max_docs` finished without timeout
  or collapse.
- `full_validation_complete`: the fixed 50k-document validation set completed.
- `record_eligible`: full validation completed within `budget_seconds`.
- `result_status`: one of `record_eligible`, `exploratory_prefix_complete`,
  `incomplete_timeout`, `incomplete_collapsed`, `incomplete_docs`, or
  `full_validation_over_budget`.

Use `max_docs < 50000` only for pilots and label those results exploratory.
Do not compare them as final Parameter Golf scores.

## Pod Lifecycle

The repo has a lease-aware RunPod helper for bringing pods up/down:

```bash
python tools/runpod.py lease-status
python tools/runpod.py create \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --name chaoscontrol-exp20-smoke \
  --lease-minutes 180
# then on the pod: git clone (first time) or git pull (subsequent) in /workspace/chaoscontrol
```

Use a 1xH100 pod for the fp8/env smoke and floor pass if cost matters. Once
the floor is known and the queue dry-run looks right, start or create the 8xH100
fanout pod:

```bash
python tools/runpod.py start <POD_ID> --lease-minutes 480
# or, if creating fresh:
python tools/runpod.py create \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 8 \
  --name chaoscontrol-exp20-first-wave \
  --lease-minutes 480
# then on the pod: git clone (first time) or git pull (subsequent) in /workspace/chaoscontrol
```

After results are written under `experiments/20_ssm_native_ttt/`, harvest and
stop in one step:

```bash
python tools/runpod.py harvest-stop <POD_ID>
```

## Full-Validation Performance Gate

Before running the TTT matrix for paper- or record-facing claims, lock the
performance path by proving the score-only floor completes all 50,000 validation
documents under the 600 second budget on the target 8xH100 shape. If a floor
run reports `timed_out: true`, `full_validation_complete: false`, or
`record_eligible: false`, do not use it as a score floor for confirmatory TTT.

The prior 1xH100 floor pass timed out at roughly 1.3k carry-state docs and
1.6k-1.8k reset docs. That result is useful throughput evidence, not a valid
score floor for record claims.

## Generated Validation Cache

The fast eval path uses a generated cache so tokenization and JSONL parsing stay
outside the timed loop. Build it on the pod or shared volume; do not commit the
cache contents unless there is a specific reproducibility reason.

```bash
python scripts/build_exp20_val_cache.py \
  --jsonl-path /workspace/chaoscontrol/baselines/parameter_golf/docs_selected.jsonl \
  --sp-model-path /workspace/chaoscontrol/baselines/parameter_golf/tokenizers/fineweb_8192_bpe.model \
  --cache-dir /workspace/cache/exp20_val_8192 \
  --max-docs 50000
```

The cache contains `tokens.npy`, `docs.npy`, and `manifest.json`. The manifest
pins the source JSONL path/size/mtime, tokenizer hash, schema version, and
requested doc count so stale caches are rejected unless rebuilt with `--force`.

## Fast Score-Only Floor

Use the cache-backed scorer for record-facing score-only floors. It scores
document-boundary reset semantics by default and includes chunk-boundary
targets, so `tokens_scored` should equal the sum of `token_len - 1` over all
scored documents. Disable that only when comparing against the legacy Exp20
harness:

```bash
CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan \
python scripts/run_exp20_fast_score.py \
  --cache-dir /workspace/cache/exp20_val_8192 \
  --checkpoint-path /workspace/results/quantmeasure/v8192.pt \
  --output-path experiments/20_ssm_native_ttt/results_floor/score_only.jsonl \
  --summary-path experiments/20_ssm_native_ttt/results_floor/score_only_summary.json \
  --chunk-size 256 \
  --doc-batch-size 4096 \
  --max-forward-tokens auto \
  --budget-seconds 600 \
  --device cuda
```

The scorer uses chunk-aware packing by default (`doc_packing:
chunk_count_tail`). It groups docs by recurrent chunk count first, then by tail
bucket and token length, so fixed-size chunk work is denser than raw length sort
alone. Completed full-validation frozen runs are unaffected; timed-out partial
runs are not prefix samples and are labeled with their `doc_packing` value. Use
`--doc-packing source_order` or the legacy `--no-sort-docs-by-length` only when a
prefix-shaped exploratory run matters more than throughput.

Ordering contract: packed scheduling is record-safe only for this runner's
current semantics: frozen score-only eval, `persistence_mode=reset`, and final
BPB computed from the commutative sum of per-doc CE and raw bytes. In that mode,
document schedule cannot optimize over the validation sequence because no state,
weights, or adaptation crosses doc boundaries. The summary records
`record_order_safe: true` and
`record_order_safe_reason: reset_score_only_commutative_ce_reduction`. If carry
state, cross-doc memory, or eval-time adaptation is enabled later, use
`source_order` unless the organizers explicitly bless schedule-only reordering.

Under `torchrun`, non-source packing forms nearby-key batches, sorts those
batches by padded token work, and assigns them to ranks with longest-processing-
time first. This avoids giving one rank all the expensive batches. Rank files
still write records sorted by original document index, and the summary records
`rank_assignment`.

It also stages the validation token cache as one device-resident int64 tensor
before the timer loop. This costs a few hundred MB for the current 50k cache
and avoids per-chunk host-to-device copies plus uint16-to-target dtype churn.

`doc_batch_size` is an upper bound. The runner resolves `--max-forward-tokens`
before the measured scoring loop and caps effective microbatches with
`max_forward_tokens / chunk_size` so packed batches do not put too much full-
width work into one OOM-prone forward. `auto` probes the requested CUDA shape
and backs off to a safe fixed shape if it OOMs; on non-CUDA devices it uses the
requested shape. The deprecated `--max-batch-tokens` spelling is still accepted
as an alias for old launch commands.

Optional warmup / graphing probe:

```bash
python scripts/run_exp20_fast_score.py ... \
  --score-warmup-steps 20 \
  --score-graph-mode cuda
```

Keep `torch_compile_mode`, `score_warmup_steps`, `score_warmup_seconds`,
`pre_eval_setup_seconds`, `score_graph_mode`, `graph_replay_count`,
`graph_fallback_count`, and `elapsed_seconds` from the summary with the result.
The scorer warmup uses synthetic token IDs and restores no validation state; it
exists to prime fixed forward/loss shapes before measured scoring.

`--score-graph-mode cuda` captures only the fixed full-batch full-chunk path
with previous recurrent states. First chunks, reduced prefixes, final partial
batches, and ragged tails fall back to eager. That keeps graph replay limited to
the shape CUDA graphs are good at while preserving eager behavior everywhere
shape or state semantics vary. Compare BPB and token counts against
`--score-graph-mode none` before using graph mode for a floor claim.

A full-shape `torch.compile(reduce-overhead)` probe without this explicit
warmup did not finish its first scoring batch after several minutes of
compilation, so keep compile mode as a separate benchmark axis from CUDA graph
mode until the warmed compile path is separately validated.

For legacy parity checks only:

```bash
python scripts/run_exp20_fast_score.py ... --no-score-boundary-targets
```

On the 1xH100 pod sampled on 2026-04-20, uncapped `4096 x 256` OOMed. With
`max_forward_tokens=524288`, the effective shape was `2048 x 256`. Before
chunk-aware packing landed, a sorted long-document sample scored 10.68M tokens
in 158.3s (~67.5k tok/s), while a fixed source-order sample scored 1.66M tokens
in 79.8s (~20.8k tok/s). CUDA graph mode was parity-checked at `128 x 256`:
2.40M tokens in 57.64s eager vs. 52.03s graph, same BPB/tokens, 43 graph
replays. Confirm the current packing+graph combination with an actual multi-GPU
full-validation run before treating any projection as a record floor.

## Exploratory Floor Pass

Generate score-floor configs first:

```bash
python experiments/20_ssm_native_ttt/build_matrix.py \
  --matrix first_wave \
  --phase floor \
  --config-dir experiments/20_ssm_native_ttt/configs_floor \
  --output-root experiments/20_ssm_native_ttt/results_floor \
  --checkpoint-path /workspace/results/final.pt \
  --sp-model-path /workspace/tokenizers/fineweb_8192.model \
  --jsonl-path /workspace/data/eval.jsonl \
  --seeds 0 1 2 \
  --safety-margin-seconds 20
```

Run them:

```bash
python experiments/20_ssm_native_ttt/run_queue.py \
  --config-dir experiments/20_ssm_native_ttt/configs_floor \
  --gpus 0 1 2 3 4 5 6 7 \
  --resume \
  --log-dir experiments/20_ssm_native_ttt/logs_floor
```

For exploratory matrix design, choose a bounded prefix that completes with
`result_status: exploratory_prefix_complete`. Do not promote an incomplete
timeout into a score floor.

## First-Wave Screens

Generate configs after the performance gate or exploratory prefix is known:

```bash
python experiments/20_ssm_native_ttt/build_matrix.py \
  --matrix first_wave \
  --phase all \
  --config-dir experiments/20_ssm_native_ttt/configs_first_wave \
  --output-root experiments/20_ssm_native_ttt/results_first_wave \
  --checkpoint-path /workspace/results/final.pt \
  --sp-model-path /workspace/tokenizers/fineweb_8192.model \
  --jsonl-path /workspace/data/eval.jsonl \
  --seeds 0 1 2 \
  --score-floor-seconds <MEASURED_SCORE_FLOOR_SECONDS> \
  --safety-margin-seconds 20
```

This emits 57 configs. The `floor` configs still measure with
`score_floor_seconds=0`; non-floor configs use the measured score floor:

- `floor`: 6 configs, `none × {reset, carry_state}`, score-only.
- `axis1`: 36 configs, `{log_a, delta_proj, log_a+delta_proj, B_side, C_side, lm_head} × {0.016, 0.064} × 3 seeds`.
- `axis3`: 15 configs, `{delta_scale 0.5/1/2, log_a_shift -0.5/+0.5} × 3 seeds`.

The generator deliberately excludes `adapt_set=all`.

## Run Queue

Dry-run the queue:

```bash
python experiments/20_ssm_native_ttt/run_queue.py \
  --config-dir experiments/20_ssm_native_ttt/configs_first_wave \
  --gpus 0 1 2 3 4 5 6 7 \
  --resume \
  --dry-run
```

Run across 8 H100s:

```bash
python experiments/20_ssm_native_ttt/run_queue.py \
  --config-dir experiments/20_ssm_native_ttt/configs_first_wave \
  --gpus 0 1 2 3 4 5 6 7 \
  --resume \
  --log-dir experiments/20_ssm_native_ttt/logs_first_wave
```

Each config is a single-GPU `scripts/run_exp20_eval.py` invocation with
`CUDA_VISIBLE_DEVICES=<gpu>` and `LOCAL_RANK=0`. Logs are written per config.

## Phase-Only Regeneration

After the floor pass, it is usually cleaner to regenerate only the screen you
want:

```bash
python experiments/20_ssm_native_ttt/build_matrix.py ... \
  --phase axis1 \
  --score-floor-seconds <MEASURED_SCORE_FLOOR_SECONDS>

python experiments/20_ssm_native_ttt/build_matrix.py ... \
  --phase axis3 \
  --score-floor-seconds <MEASURED_SCORE_FLOOR_SECONDS>
```
