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
python tools/runpod.py deploy <POD_ID>
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
python tools/runpod.py deploy <POD_ID>
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

The scorer length-sorts each rank's document work by default to keep hot-loop
chunk shapes dense. Completed full-validation runs are unaffected; timed-out
partial runs are not prefix samples and are labeled with
`doc_ordering: token_len_desc`. Use `--no-sort-docs-by-length` only when a
prefix-shaped exploratory run matters more than throughput.

It also stages the validation token cache as one device-resident int64 tensor
before the timer loop. This costs a few hundred MB for the current 50k cache
and avoids per-chunk host-to-device copies plus uint16-to-target dtype churn.

`doc_batch_size` is an upper bound. The runner resolves `--max-forward-tokens`
before the measured scoring loop and caps effective microbatches with
`max_forward_tokens / chunk_size` so length-sorted batches do not put all
longest documents into one OOM-prone full-width group. `auto` probes the
requested CUDA shape and backs off to a safe fixed shape if it OOMs; on non-CUDA
devices it uses the requested shape. The deprecated `--max-batch-tokens`
spelling is still accepted as an alias for old launch commands.

Optional warmup / graphing probe:

```bash
python scripts/run_exp20_fast_score.py ... \
  --torch-compile-mode reduce-overhead \
  --score-warmup-steps 20
```

Keep `torch_compile_mode`, `score_warmup_steps`, `score_warmup_seconds`,
`pre_eval_setup_seconds`, and `elapsed_seconds` from the summary with the
result. The scorer warmup uses synthetic token IDs and restores no validation
state; it exists to prime fixed forward/loss shapes before measured scoring. A
full-shape `reduce-overhead` probe without this explicit warmup did not finish
its first scoring batch after several minutes of compilation, so do not use
compile mode for floor claims until the warmed compile path or a CUDA graph path
is separately validated.

For legacy parity checks only:

```bash
python scripts/run_exp20_fast_score.py ... --no-score-boundary-targets
```

On the 1xH100 pod sampled on 2026-04-20, uncapped `4096 x 256` OOMed. With
`max_forward_tokens=524288`, the effective shape was `2048 x 256`. A sorted
long-document sample scored 10.68M tokens in 158.3s (~67.5k tok/s); a fixed
source-order sample scored 1.66M tokens in 79.8s (~20.8k tok/s). The difference
is throughput, not score semantics: reset score-only BPB is order-invariant, and
the tests compare chunked sorted/source-order runs against whole-doc scoring.
Confirm with an actual multi-GPU full-validation run before treating any
projection as a record floor.

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
