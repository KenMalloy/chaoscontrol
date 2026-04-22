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

Command:

```bash
'/Users/kennethmalloy/Local Documents/Developer/chaoscontrol/.venv/bin/python' experiments/24_training_time_bundle/profile_event_sleep_arm.py --steps 10 --seconds 10 --device cpu
```

Output JSON: `experiments/24_training_time_bundle/profile_event_sleep_arm_out.json`

Summary:

- Device: `cpu`
- Steps: 10
- Tokens per step: 4096
- Mean inside-budget wall time per step: 7.224 ms

| rank | section | mean wall ms | p50 wall ms | p95 wall ms | share |
|---:|---|---:|---:|---:|---:|
| 1 | backward | 4.791 | 3.348 | 11.459 | 66.32% |
| 2 | logits_and_loss | 1.908 | 1.784 | 2.695 | 26.41% |
| 3 | optimizer_step | 0.351 | 0.325 | 0.521 | 4.86% |
| 4 | encode_forward | 0.158 | 0.142 | 0.232 | 2.18% |
| 5 | event_sleep_gate | 0.015 | 0.014 | 0.023 | 0.21% |
| 6 | fast_slow_ema | 0.002 | 0.002 | 0.002 | 0.02% |

## Initial Verdicts

| candidate | verdict | rationale |
|---|---|---|
| Fast-slow foreach lerp | landing-target | Task 8 is small, already covered by an equivalence test, and replaces a visible Python loop. Measure host-wall delta after implementation. |
| Event-sleep on-device EMA | needs-audit | Task 9 is not unconditional: the caller immediately consumes Python bools/floats, so a host-wall win must be measured before keeping it. |
| Spectral regularizer vectorization | negligible-skip unless H100 says otherwise | The active smoke arm did not sample `spectral_reg`; execute only if H100 profile shows >1% host-wall share. |
| Train-step graph-break audit | not applicable locally | Base config has `compile_full_path: false`; run only if the H100 profile/active arm enables the compiled path. |
| Dreamworld replay backward | needs H100 sample | The 10-step CPU smoke did not sample replay; use the longer H100 profile to classify it. |

## H100 Run

Pending. There is no active RunPod pod in this environment, and creating a new paid pod is outside the local code implementation step without explicit operator confirmation.
