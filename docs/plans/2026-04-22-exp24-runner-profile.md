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
- Mean inside-budget wall time per step: 6.289 ms

| rank | section | mean wall ms | p50 wall ms | p95 wall ms | share |
|---:|---|---:|---:|---:|---:|
| 1 | backward | 3.634 | 3.125 | 4.345 | 57.78% |
| 2 | logits_and_loss | 1.769 | 1.557 | 2.225 | 28.12% |
| 3 | optimizer_step | 0.400 | 0.365 | 0.710 | 6.36% |
| 4 | dreamworld_replay | 1.667 | 1.672 | 1.833 | 2.49% |
| 5 | encode_forward | 0.149 | 0.126 | 0.319 | 2.36% |
| 6 | event_sleep_gate | 0.139 | 0.058 | 0.085 | 2.22% |
| 7 | fast_slow_ema | 0.037 | 0.001 | 0.003 | 0.59% |
| 8 | event_sleep_decision_resolve | 0.010 | 0.009 | 0.014 | 0.08% |

## Candidate Verdicts

| candidate | verdict | rationale |
|---|---|---|
| Fast-slow foreach lerp | landed on CPU smoke; H100 delta pending | Replaced the per-parameter slow-weight blend with `torch._foreach_lerp_` for matching dtype/device tensors and retained scalar fallback for mismatches. The latest CPU smoke shows a 0.001 ms p50 for `fast_slow_ema`; the mean/share are dominated by rare host outliers. |
| Event-sleep on-device EMA | landed as CUDA-sync avoidance; H100 delta pending | `LossTriggeredReplayEMA` now keeps EMA/pressure as tensors and the runner resolves the prior step's decision at the next step boundary. CPU host-wall for `event_sleep_gate` increased because CPU tensor ops replace scalar math; explicit decision materialization is now isolated in `event_sleep_decision_resolve` at 0.010 ms mean. The intended win is avoiding CUDA `.item()` sync on H100. |
| Spectral regularizer vectorization | inactive for this arm; deferred | The `fast_slow_dreamworld_event_sleep` arm leaves `spectral_reg_lambda_*` at `0.0`, and the profile records `spectral_reg` with count 0/share 0.0%. No vectorization landed because it cannot move this arm's 600-second budget. |
| Train-step graph-break audit | not applicable for active config | Base Exp 23/24 configs have `compile_full_path: false`; there is no compiled train-step region for `TORCH_LOGS=graph_breaks` to inspect in this arm. |
| Dreamworld replay backward | audited; no code change | Replay already does a state-conditioned teacher-forced forward and LM-head backward from cached detached states. The sampled entry can be old/random, so reusing current-step activations would change the replay contract rather than just remove recomputation. CPU smoke samples replay at 2.49% host-wall share; H100 share still pending. |

## Dreamworld Backward Audit

`dreamworld_replay_backward` subsamples the replay entry if configured, moves cached states and replay tokens to the model device, runs `model.encode(replay_inputs, initial_states=states, return_final_states=False)`, then backpropagates through either `full_lm_head_backward` or the fused LM-head path. It does not reuse main-step activations, and that is intentional: replay entries are detached buffer samples, not guaranteed to be from the current batch or current step. Reusing live activations would couple replay semantics to the current minibatch and break the buffer contract.

## H100 Run

Pending. There is no active RunPod pod in this environment, and creating a new paid pod is outside the local code implementation step without explicit operator confirmation.
