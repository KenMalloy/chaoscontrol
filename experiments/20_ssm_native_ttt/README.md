# Exp20 SSM-Native TTT Runs

This directory contains the runnable Exp20 matrix artifacts. The goal is to
run the actual SSM-native ablation, not the old naive `adapt_set=all` envelope.

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

## Floor Pass

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

Choose the score floor from the resulting score-only summaries, then generate
the screens with that measured value.

## First-Wave Screens

Generate configs after the floor is known:

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
