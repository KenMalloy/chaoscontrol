# Phase 1A Bench — Results

Throughput, peak VRAM, and final-loss microbench for the three Track 1A
levers (`fused_grad_clip`, `fused_muon`, `compile_full_path`) at the
Exp 18 submission regime (4L × 256d SSM, V=16384, seq=512, bs=1024/rank,
Muon LR=0.064, bf16, activation_checkpoint=True).

Each (lever_combo × seed) is one measurement. Per-lever marginal effects
are reported as paired deltas (on − off) across the 4 settings of the
other two levers, so each lever has up to 4 × N_seeds paired deltas.

## Harness

- Script: `experiments/19_phase1/bench_phase1a.py`
- Hardware: 1× H100 SXM 80GB, CUDA 13, torch 2.11
- CPU pinning: `torch.set_num_threads(4)` + `OMP_NUM_THREADS=4` set
  BEFORE `import torch` (default 28-thread pool inflates variance)
- Warmup: 20 steps (warmup-restore pattern from
  `runner_persistent_ddp.py`), timed region = 200 steps split into
  5 × 40-step blocks for within-measurement tok/s std
- Seeds: `[1337, 2674]`
- RNG caveat: each of the 5 × 40-step blocks runs with an independent
  batch-RNG seed (`seed + 7919 * block_idx`), so `final_loss` is the
  loss of the last 40-step block under an independent draw — not the
  loss of a single 200-step trajectory. The ±0.02 loss gate is a sanity
  check, not a convergence comparison.

## Invocation

```bash
python experiments/19_phase1/bench_phase1a.py \
    --data-path baselines/parameter_golf/datasets/fineweb10B_sp16384 \
    --sp-model-path baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
    --output-dir experiments/19_phase1/results_bench_phase1a/ \
    --world-size 1 --n-timed-steps 200 --warmup-steps 20 \
    --seeds 1337 2674
```

## Success gates (per lever)

A lever is green-lit for default-on in the Phase 1C matrix iff all three
gates pass.

| Gate | Criterion |
|------|-----------|
| Throughput | paired Δ mean ≥ +5% tok/s AND paired Δ std ≤ (paired Δ mean)/2 |
| VRAM | paired Δ mean within ±5% peak VRAM |
| Loss | paired Δ mean of final_loss within ±0.02 |

## Results

Baseline (all levers off), mean across seeds:

- tokens/sec: _TODO: numbers_
- peak VRAM: _TODO: numbers_ MB
- final loss at step 200: _TODO: numbers_

### Lever: `fused_grad_clip`

- tok/s Δ: _TODO: numbers_
- peak VRAM Δ: _TODO: numbers_
- final_loss Δ: _TODO: numbers_
- Gates: _TODO: PASS/FAIL_

### Lever: `fused_muon`

- tok/s Δ: _TODO: numbers_
- peak VRAM Δ: _TODO: numbers_
- final_loss Δ: _TODO: numbers_
- Gates: _TODO: PASS/FAIL_

### Lever: `compile_full_path`

- tok/s Δ: _TODO: numbers_
- peak VRAM Δ: _TODO: numbers_
- final_loss Δ: _TODO: numbers_
- Gates: _TODO: PASS/FAIL_

## Decision

_TODO: which levers proceed as default-on into the Phase 1C matrix,
which are deferred pending follow-up, and why._

## Raw data

`results.jsonl` (untracked per repo convention) in the `--output-dir`
above. Each line is one measurement with the schema:

```json
{
  "seed": 1337,
  "fused_grad_clip": true,
  "fused_muon": false,
  "compile_full_path": true,
  "n_timed_steps": 200,
  "warmup_steps": 20,
  "tokens_per_sec_mean": 0.0,
  "tokens_per_sec_std": 0.0,
  "peak_vram_mb": 0.0,
  "final_loss": 0.0,
  "wall_clock_s": 0.0,
  "config_hash": "abcd1234",
  "block_elapsed_s": [0.0, 0.0, 0.0, 0.0, 0.0],
  "block_steps": [40, 40, 40, 40, 40],
  "world_size": 1,
  "tokens_per_step": 524288
}
```
