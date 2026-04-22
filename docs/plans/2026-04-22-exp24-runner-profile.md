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

Latest command after the fast-slow foreach change:

```bash
'/Users/kennethmalloy/Local Documents/Developer/chaoscontrol/.venv/bin/python' experiments/24_training_time_bundle/profile_event_sleep_arm.py --steps 64 --seconds 10 --device cpu
```

Output JSON: `experiments/24_training_time_bundle/profile_event_sleep_arm_out.json`

Summary:

- Device: `cpu`
- Steps: 64
- Tokens per step: 4096
- Mean inside-budget wall time per step: 6.267 ms

| rank | section | mean wall ms | p50 wall ms | p95 wall ms | share |
|---:|---|---:|---:|---:|---:|
| 1 | backward | 3.668 | 3.254 | 4.650 | 58.52% |
| 2 | logits_and_loss | 1.842 | 1.668 | 2.453 | 29.39% |
| 3 | optimizer_step | 0.452 | 0.399 | 0.806 | 7.21% |
| 4 | dreamworld_replay | 1.609 | 1.620 | 1.795 | 2.41% |
| 5 | encode_forward | 0.135 | 0.129 | 0.168 | 2.15% |
| 6 | event_sleep_gate | 0.015 | 0.014 | 0.020 | 0.23% |
| 7 | fast_slow_ema | 0.006 | 0.001 | 0.002 | 0.09% |

## Initial Verdicts

| candidate | verdict | rationale |
|---|---|---|
| Fast-slow foreach lerp | landed on CPU smoke; H100 delta pending | Replaced the per-parameter slow-weight blend with `torch._foreach_lerp_` for matching dtype/device tensors and retained scalar fallback for mismatches. The 64-step CPU smoke shows `fast_slow_ema` at 0.09% host-wall share after landing. |
| Event-sleep on-device EMA | needs-audit | Task 9 is not unconditional: the caller immediately consumes Python bools/floats, so a host-wall win must be measured before keeping it. |
| Spectral regularizer vectorization | negligible-skip unless H100 says otherwise | The active smoke arm did not sample `spectral_reg`; execute only if H100 profile shows >1% host-wall share. |
| Train-step graph-break audit | not applicable locally | Base config has `compile_full_path: false`; run only if the H100 profile/active arm enables the compiled path. |
| Dreamworld replay backward | needs H100 sample | The 10-step CPU smoke did not sample replay; use the longer H100 profile to classify it. |

## H100 Run

Pending. There is no active RunPod pod in this environment, and creating a new paid pod is outside the local code implementation step without explicit operator confirmation.
