# Exp24 Semantic Optimizer Overhead Gate

Run date: 2026-04-22.

Hardware: 1x H100 SXM in India (`AP-IN-1`) on the official
`runpod/parameter-golf:latest` template. Pod id was `o0lrljyem78oxx`; it was
stopped after artifacts were harvested.

Environment:

```text
torch 2.9.1+cu128
CUDA runtime/toolkit 12.8
CHAOSCONTROL_DIAG_SCAN_BACKEND=ssm_scan
CHAOSCONTROL_POST_SCAN_BACKEND=eager
lm_head_backward_mode=fused_streaming_cached
lm_head_tile_size=8192
batch_size=1024
seq_len=512
model_dim=256
num_layers=4
world_size=1
budget=90s per arm
```

Dataset artifact: `Natooka/parameter-golf-sp-tokenizers` revision
`e9d696d1592d884dbb97e754efb2a7203aca3080`.

Before the run, the fresh pod built/import-checked the native CUDA extensions
and passed:

```text
tests/test_ssm_scan.py tests/test_lm_head_loss_kernel.py
tests/test_exp24_training_bundle.py::test_run_exp24_cli_semantic_gate_defaults_to_cheap_smoke

55 passed in 6.59s
```

## Result

| Arm | Optimizer | Steps | Tokens/s | Final Train Loss | Peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| `exp24_smoke_semantic_gate_muon_s1337` | Muon | 532 | 3,154,715 | 4.149150 | 53,158.7 MB |
| `exp24_smoke_semantic_gate_semantic_s1337` | SemanticOptimizer | 516 | 3,055,098 | 4.178514 | 53,179.3 MB |

SemanticOptimizer was 3.16% slower than Muon on this 1xH100 smoke. The train
loss values above are telemetry, not a quality comparison: this was matched
wall-clock, not matched steps, and it did not run validation BPB. Muon received
532 optimizer steps while SemanticOptimizer received 516, so the endpoint train
loss folds throughput into the optimizer comparison.

Interpretation: this run is an environment and overhead smoke only. It says
SemanticOptimizer works on the fast path and its measured overhead is modest on
1xH100. It does not answer whether SemanticOptimizer learns better per gradient
step, nor whether its shipped weights score better on validation under a fixed
training budget.

The next honest optimizer comparison should use matched optimizer steps, record
tokens/s separately, and score validation BPB from the resulting weights. The
wall-clock decision can then be computed from both pieces: per-step quality and
achievable steps.

Raw harvested artifacts are local under:

```text
experiments/24_training_time_bundle/results_semantic_gate_1x_20260422T122800Z/
```
