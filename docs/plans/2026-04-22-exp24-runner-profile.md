# Exp 24 Event-Sleep Arm Profile

## Status

Profile harness added at `experiments/24_training_time_bundle/profile_event_sleep_arm.py`.

Local verification was a CPU smoke run because this workstation has no CUDA device:

```text
torch.cuda.is_available() -> False
torch.cuda.device_count() -> 0
runpodctl pod list -> []
```

Do not treat the CPU numbers below as H100 optimization evidence. They only prove the harness runs and emits the host-wall schema that the H100 run will use.

## CPU Smoke

Latest command after the fast-slow foreach change and event-sleep tensor gate:

```bash
'/Users/kennethmalloy/Local Documents/Developer/chaoscontrol/.venv/bin/python' experiments/24_training_time_bundle/profile_event_sleep_arm.py --steps 64 --seconds 10 --device cpu
```

Output JSON: `experiments/24_training_time_bundle/profile_event_sleep_arm_out.json`

Summary:

- Device: `cpu`
- Steps: 64
- Tokens per step: 4096
- Mean inside-budget wall time per step: 5.840 ms

| rank | section | mean wall ms | p50 wall ms | p95 wall ms | share |
|---:|---|---:|---:|---:|---:|
| 1 | backward | 3.419 | 3.153 | 3.824 | 58.54% |
| 2 | logits_and_loss | 1.657 | 1.585 | 1.959 | 28.38% |
| 3 | optimizer_step | 0.400 | 0.388 | 0.688 | 6.85% |
| 4 | dreamworld_replay | 1.510 | 1.576 | 1.625 | 2.42% |
| 5 | encode_forward | 0.140 | 0.129 | 0.183 | 2.39% |
| 6 | event_sleep_gate | 0.057 | 0.055 | 0.077 | 0.97% |
| 7 | fast_slow_ema | 0.026 | 0.001 | 0.002 | 0.45% |

## Initial Verdicts

| candidate | verdict | rationale |
|---|---|---|
| Fast-slow foreach lerp | landed on CPU smoke; H100 delta pending | Replaced the per-parameter slow-weight blend with `torch._foreach_lerp_` for matching dtype/device tensors and retained scalar fallback for mismatches. The latest CPU smoke shows a 0.001 ms p50 for `fast_slow_ema`; the mean/share are dominated by rare host outliers. |
| Event-sleep on-device EMA | landed as CUDA-sync avoidance; H100 delta pending | `LossTriggeredReplayEMA` now keeps EMA/pressure as tensors and the runner resolves the prior step's decision at the next step boundary. CPU host-wall for `event_sleep_gate` increased to 0.057 ms because CPU tensor ops replace scalar math; the intended win is avoiding CUDA `.item()` sync on H100. |
| Spectral regularizer vectorization | negligible-skip unless H100 says otherwise | The active smoke arm did not sample `spectral_reg`; execute only if H100 profile shows >1% host-wall share. |
| Train-step graph-break audit | not applicable locally | Base config has `compile_full_path: false`; run only if the H100 profile/active arm enables the compiled path. |
| Dreamworld replay backward | needs H100 sample | The 10-step CPU smoke did not sample replay; use the longer H100 profile to classify it. |

## H100 Run

Pending. There is no active RunPod pod in this environment, and creating a new paid pod is outside the local code implementation step without explicit operator confirmation.
