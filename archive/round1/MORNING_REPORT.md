# ChaosControl Experiment Run — Morning Report

**Run launched:** 2026-04-07 03:37 UTC
**Expected completion:** ~09:30 UTC (~6 hours, 70 configs x 5 min each)
**Pod:** H100 80GB, 213.181.105.235:19321, SSH key `~/.ssh/id_runpod`

---

## What happened tonight

### Three bugs fixed before launch

1. **fp16 → bf16** — All SSM configs produced `bpb=NaN` at step 1. Root cause: model weights cast directly to fp16 without GradScaler, SSM recurrence compounds overflow across 128 timesteps. Fix: changed default dtype to bf16 (same H100 tensor core throughput, fp32 exponent range).

2. **batch_size 4 → 64, seq_len 128 → 256** — The original settings used 1.7GB of 80GB VRAM (2% utilization). 16x larger batch + 2x longer sequences. All configs use the same defaults, so relative comparisons remain valid.

3. **CUDA backends** — Added `cudnn.benchmark=True` and `float32_matmul_precision='high'` (TF32) for H100.

### Six experiment integrity fixes

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | **HIGH** | compression_consequence never called from memory compression | `_compress()` now records quality deltas, training loop feeds them to `WernickeLayer.compression_consequence_update()` |
| 2 | **HIGH** | survival_vs_random config was a stub ("requires code modification") | Added `compression_selection: random` config field, random merge ordering in `_compress()` |
| 3 | MED | Metabolic gate threshold never logged | `current_threshold` and `surprise_ratio` now in step_record; analyze.py reports fork rate + threshold trajectory |
| 4 | MED | Bucket distribution for typed composition never persisted | `bucket_snapshots` logged every 50 steps (counts, active buckets); analyze.py reports utilization |
| 5 | MED | dynamic_crit_per_layer was a stub | Per-layer criticality targets linearly spaced around global target; per-layer loss computation |
| 6 | MED | lambda_max mislabeled as "Lyapunov exponent" | Relabeled to "top log singular value (criticality proxy)" |

### Spectral logging added

Training loop now computes FFT power spectrum of hidden states every 50 steps. Experiment 02 analyzer reports dominant frequency evolution, criticality proxy trajectory, and power spectrum shape (1/f-like = near-critical).

---

## Early results (from smoke tests + first config)

| Config | bpb | Notes |
|--------|-----|-------|
| SSM full (384d) | **2.24** | Beats transformer at matched size |
| SSM medium (256d) | **2.32** | |
| SSM small (128d) | **2.47** | |
| Transformer full (384d) | 2.44 | SSM wins by 0.20 bpb |
| Transformer medium (256d) | 2.52 | |
| Transformer small (128d) | 2.84 | |

**The SSM already beats the transformer at every size.** This is a strong baseline for the thesis.

---

## When you wake up

### If the pod is stopped (expected — auto-shutdown after completion):
```bash
# 1. Start the pod via RunPod dashboard

# 2. Harvest results
rsync -az -e "ssh -i ~/.ssh/id_runpod -p 19321" \
  root@213.181.105.235:/workspace/chaoscontrol/experiments/ \
  experiments/

# 3. Check how many completed
find experiments -name "*.json" -path "*/results/*" | wc -l
# Expected: 70

# 4. Run per-experiment analysis
for exp in experiments/0{1,2,3,4,5,6,7,8}_*/; do
  echo "=== $(basename $exp) ==="
  PYTHONPATH=src python3 "$exp/analyze.py"
  echo ""
done
```

### If the pod is still running:
```bash
ssh -i ~/.ssh/id_runpod -p 19321 root@213.181.105.235 \
  "tail -30 /workspace/chaoscontrol/experiment_run.log"
```

### If the pod was preempted (community GPU risk):
- The disk persists — restart the pod
- Check which results were saved
- Re-run only the missing configs (each experiment's `run.sh` skips existing results... actually no, it overwrites. You'd need to re-run the full remaining experiments)

---

## Safety layers in place

1. **Auto-shutdown script** — `poweroff` runs on clean exit
2. **Watchdog process** — polls tmux session every 5 min; if session dies (crash, error), `poweroff` in 2 min
3. **Persistent disk** — results survive shutdown/preemption at low monthly storage cost

---

## Files changed (not yet committed)

- `src/chaoscontrol/config.py` — bf16 default, batch_size=64, seq_len=256, compression_selection field
- `src/chaoscontrol/training.py` — spectral logging, bucket logging, metabolic threshold logging, compression consequence wiring, per-layer criticality
- `src/chaoscontrol/runner.py` — CUDA backends, compression_selection passthrough
- `src/chaoscontrol/memory.py` — compression consequences tracking, random compression selection
- `src/chaoscontrol/model.py` — compression_selection + per-layer jacobian stats passthrough
- `experiments/02_critical_dynamics/analyze.py` — spectral + criticality analysis
- `experiments/05_typed_composition/analyze.py` — bucket utilization analysis
- `experiments/06_metabolic_gate/analyze.py` — fork rate + threshold analysis
- `experiments/08_gap_analysis/configs/survival_vs_random.yaml` — now uses compression_selection: random
- `experiments/08_gap_analysis/configs/dynamic_crit_per_layer.yaml` — stub comment removed
- `run_with_shutdown.sh` — trap-based guaranteed shutdown
