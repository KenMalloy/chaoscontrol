# Exp 19 prereqs — persistent-DDP multi-seed launcher

One `torchrun` per matrix instead of one per (condition, seed). The
workers keep the process / CUDA context / DDP process group /
`torch.compile` cache / FUSE mmap warm across every entry in the
matrix; only the model, optimizer, and LR scheduler are rebuilt
between entries.

## Why

The per-seed spawn pattern in
`experiments/18_throughput_levers/run_exp18_test10.py` launches a
fresh `torchrun` for every (condition, seed). Each spawn pays ~10 min
of overhead (imports, CUDA context, DDP rendezvous, FUSE page-cache
warmup, cold `torch.compile`). Over a 4-seed × 2-condition matrix
that's ~80 min of dead time at $1-2/min on 4×H100.

## Run

    python experiments/19_prereqs/run_persistent_launcher.py \
        --data-path baselines/parameter_golf/datasets/fineweb10B_sp16384 \
        --sp-model-path baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
        --output-dir experiments/19_prereqs/results_smoke/ \
        --world-size 4 --base-lr 0.128 --budget 600

CPU-only dry-run for matrix-shape sanity:

    python experiments/19_prereqs/run_persistent_launcher.py ... --dry-run
    python experiments/19_prereqs/runner_persistent_ddp.py --dry-run \
        --data-path /tmp/x --sp-model-path /tmp/y \
        --config-matrix /tmp/persistent_matrix_*.json \
        --output-dir /tmp/out

## Outputs

Per-entry JSONs at `{output-dir}/{name}_s{seed}.json` with the same
schema as `experiments/18_throughput_levers/runner_exp18_ssm.py`
(fields: `config`, `params`, `train`, `eval`). Error markers for
fp8-skipped or per-entry-failed runs are written as
`{"config": ..., "error": "..."}` — `_harness.result_is_finite()`
returns False on these, matching existing summarizer contracts.

## Persistent vs per-entry state

Carried once (warm): CUDA context, DDP process group, Python imports,
train/val mmap tensors, SP LUTs, `torch.compile` cache (via
`verify_diag_recurrence`), cudnn flags, device/dtype.

Rebuilt per entry: RNGs (`torch`/`cuda`/`numpy`/`random`), model
(`build_model`), fp8 promotion (if `precision == "fp8"`), optimizer,
LR scheduler (implicit in optimizer), train-starts sharding,
eval-start sampling. `torch.cuda.empty_cache()` + `dist.barrier()`
separate entries.

## Diff vs `run_exp18_test10.py`

Test 10 spawns one `torchrun` per entry via
`_harness.run_parallel_ddp_matrix`. Here, one `torchrun` runs every
entry in sequence inside a single persistent process; fp8-unavailable
entries are pre-skipped with error markers in the launcher before
matrix write, and per-entry failures are isolated with a cross-rank
`all_reduce(MAX)` error flag instead of aborting the matrix.

## Sticky skip markers

The fp8 skip markers from a TE-less pod persist on disk and are
honored by the idempotent existing-output check on the next run. If
you move to a pod that has TE and want fp8 to actually run, delete
the fp8 markers first: `rm {output-dir}/fp8_s*.json`.

The launcher's pre-flight now refuses to run when TE is available
AND stale `skipped: transformer_engine unavailable on pod` markers
exist for fp8 entries in the current matrix. Without this guard,
the idempotent skip would classify those markers as benign and the
launcher would return rc=0 with zero fp8 results — a silent
failure for anyone reproducing the matrix on a different pod. The
error names the stale paths and the exact `rm` command to recover.
