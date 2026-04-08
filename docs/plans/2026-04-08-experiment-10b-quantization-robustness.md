# Experiment 10b: Quantization Robustness and Mechanism Isolation

## Two Claims

1. **Architecture claim:** ChaosControl is more quantization-robust than matched baselines.
2. **Mechanism claim:** The robustness comes from latent memory reactivation.

These require separate standards of proof. The mechanism claim is stronger and must be tested with ablations that isolate reactivation from other components.

## Primary Metric

```
delta_bpb(model, q) = bpb_quantized(model, q) - bpb_bf16(model)
```

Same checkpoint, same eval set, same warmup protocol, same tokenizer, same seed. A lower delta_bpb curve means more quantization-robust.

## Factorial Grid

| Dimension | Levels |
|-----------|--------|
| Model variant | bare_ssm, full_ssm, full_ssm_no_reactivation, our_tfm, mamba2_ssm |
| Quantization | bf16, int8, int6, int4 |
| Eval memory state | seeded (training memory preserved), cold-start (memory wiped) |
| Seeds | 5 minimum, 7-10 ideal |

Total eval runs per trained checkpoint: 5 models × 4 quant × 2 eval states = 40.
With 5 seeds: 200 evals. Evals are cheap (forward pass only, no training).

## Key Statistics

### Architecture claim (claim 1)
```
delta_bpb(full_ssm, int8) < delta_bpb(our_tfm, int8)
delta_bpb(full_ssm, int6) < delta_bpb(our_tfm, int6)
```
Report: mean delta_bpb, bootstrap 95% CI, paired effect size.

### Mechanism claim (claim 2)
```
reactivation_gain(q) = delta_bpb(full_ssm_no_reactivation, q) - delta_bpb(full_ssm, q)
```
- If consistently positive: reactivation helps under quantization.
- If near zero: reactivation is not the mechanism.
- If negative: reactivation hurts under quantization.

### Diff-in-diff
```
(full_ssm_int8 - full_ssm_bf16) - (baseline_int8 - baseline_bf16)
```
And the same with latent_persistence on vs off.

## Mediation Story

If the mechanism claim holds, this causal chain should be observable:

1. Quantization increases effective surprise / state corruption
2. Higher surprise triggers more latent reactivation events
3. Reactivation reduces the local performance drop

### Per-run instrumentation needed:
- `latent_reactivation_count` — how many times `try_reactivate` fires during eval
- `reactivation_rate` — reactivation_count / total_eval_steps
- `post_reactivation_loss_delta` — mean loss improvement in the N steps after a reactivation event vs matched non-reactivation windows

### Falsification criteria:
- full_ssm degrades no less than baselines after quantization → claim 1 fails
- full_ssm is better, but benefit persists with latent_persistence off → claim 2 fails
- quantization does not increase reactivation usage → mediation fails
- reactivation fires but doesn't improve local loss → mediation fails
- robustness explained by Wernicke routing or semantic memory alone → mechanism is different

## Model Variants

| Variant | Config changes from full_ssm |
|---------|------------------------------|
| `full_ssm` | experiment 09 winning stack |
| `full_ssm_no_reactivation` | `latent_persistence: false` |
| `full_ssm_no_memory` | `outer_model_dim: 0` |
| `full_ssm_cold_eval` | eval with `warmup_cold_start: true` |
| `bare_ssm` | no components |
| `our_tfm` | SimpleTransformerLM, param-matched |
| `mamba2_ssm` | Mamba-2, same size |

## Implementation

### Runner: `experiments/10b_quantization/run_quantization.py`

```python
for size in sizes:
    # Train once per model variant per seed
    for variant in model_variants:
        for seed in seeds:
            checkpoint = train_and_save(variant, size, seed)
    
    # Eval factorial grid on each checkpoint
    for variant in model_variants:
        for seed in seeds:
            for quant in ["bf16", "int8", "int6", "int4"]:
                for eval_state in ["seeded", "cold"]:
                    result = eval_checkpoint(
                        checkpoint, quant, eval_state,
                        log_reactivation=True,
                    )
                    save(result)
```

### Instrumentation: add to `evaluation.py`

Return `reactivation_count` and `reactivation_steps` from `evaluate_chaoscontrol_bpb` when warmup is enabled.

### Artifact pipeline integration

For each trained checkpoint:
1. `serialize_artifact()` at each quant level
2. `load_artifact()` → eval
3. Report: bpb_bf16, bpb_artifact, bpb_ttt, delta_bpb, ttt_recovery

## Depends On

- Experiment 09 results (winning component stack)
- Experiment 10a results (trained checkpoints at each size)
- Artifact pipeline (`artifact.py` — done)

## Estimated Cost

Training: same as experiment 10a (already done).
Eval: 200 forward passes × ~30s each = ~100 minutes on 3 GPUs.
Total new compute: ~35 minutes with parallelism.
