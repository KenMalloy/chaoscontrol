"""CPU SSM controller runtime scaffold.

The C++ runtime starts with a reference diagonal SSM cell. AMX/AVX kernels can
replace the inner loops later as long as this ABI and numerical contract hold.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from chaoscontrol.episodic.cpu_ssm_controller import (
    CpuSsmControllerDecision,
    CpuSsmControllerRuntime,
    CpuSsmControllerState,
    CpuSsmControllerWeights,
    cpp_available,
    forward_step,
)


def _load_runner_module():
    path = (
        Path(__file__).resolve().parent.parent
        / "experiments" / "23_fast_path" / "runner_fast_path.py"
    )
    spec = importlib.util.spec_from_file_location("runner_fast_path", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _weights() -> CpuSsmControllerWeights:
    return CpuSsmControllerWeights(
        w_global_in=torch.tensor(
            [[0.10, 0.20, -0.10], [0.00, 0.30, 0.40]],
            dtype=torch.float32,
        ),
        w_slot_in=torch.tensor([[0.50, -0.20, 0.10]], dtype=torch.float32),
        decay_global=torch.tensor([0.90, 0.50], dtype=torch.float32),
        decay_slot=torch.tensor([0.25], dtype=torch.float32),
        w_global_out=torch.tensor([0.70, -0.30], dtype=torch.float32),
        w_slot_out=torch.tensor([0.20], dtype=torch.float32),
        bias=torch.tensor(0.05, dtype=torch.float32),
    )


def test_forward_step_reference_matches_hand_computation():
    weights = _weights()
    state = CpuSsmControllerState(
        global_state=torch.tensor([1.0, -1.0], dtype=torch.float32),
        slot_state=torch.tensor([2.0], dtype=torch.float32),
    )
    features = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)

    out = forward_step(features, state, weights, prefer_cpp=False)

    expected_global = torch.tensor([
        0.90 * 1.0 + (0.10 * 2.0 + 0.20 * 3.0 - 0.10 * 4.0),
        0.50 * -1.0 + (0.00 * 2.0 + 0.30 * 3.0 + 0.40 * 4.0),
    ])
    expected_slot = torch.tensor([
        0.25 * 2.0 + (0.50 * 2.0 - 0.20 * 3.0 + 0.10 * 4.0)
    ])
    expected_logit = (
        expected_global @ weights.w_global_out
        + expected_slot @ weights.w_slot_out
        + weights.bias
    )

    torch.testing.assert_close(out.global_state, expected_global)
    torch.testing.assert_close(out.slot_state, expected_slot)
    torch.testing.assert_close(out.logit, expected_logit)


def test_weight_dump_round_trip(tmp_path: Path):
    path = tmp_path / "controller_weights.pt"
    weights = _weights()

    weights.save(path)
    restored = CpuSsmControllerWeights.load(path)

    for key, original in weights.to_dict().items():
        loaded = restored.to_dict()[key]
        torch.testing.assert_close(loaded, original, msg=f"{key} diverged")


def test_cpp_runtime_matches_reference_when_built():
    if not cpp_available():
        pytest.skip("cpu_ssm_controller C++ extension not built")

    weights = _weights()
    state = CpuSsmControllerState(
        global_state=torch.tensor([0.25, -0.75], dtype=torch.float32),
        slot_state=torch.tensor([1.5], dtype=torch.float32),
    )
    features = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)

    ref = forward_step(features, state, weights, prefer_cpp=False)
    cpp = forward_step(features, state, weights, prefer_cpp=True)

    torch.testing.assert_close(cpp.global_state, ref.global_state)
    torch.testing.assert_close(cpp.slot_state, ref.slot_state)
    torch.testing.assert_close(cpp.logit, ref.logit)


def test_runtime_tracks_global_and_per_slot_state_independently():
    weights = _weights()
    runtime = CpuSsmControllerRuntime(weights, capacity=3, prefer_cpp=False)

    features_a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    features_b = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    runtime.score_slot(features_a, slot=0)
    slot0_after_first = runtime.slot_state[0].clone()
    slot1_initial = runtime.slot_state[1].clone()
    runtime.score_slot(features_b, slot=1)

    assert not torch.equal(runtime.global_state, torch.zeros_like(runtime.global_state))
    assert torch.equal(runtime.slot_state[0], slot0_after_first)
    assert not torch.equal(runtime.slot_state[1], slot1_initial)


def test_runtime_score_slot_with_snapshot_returns_pre_update_state():
    weights = _weights()
    runtime = CpuSsmControllerRuntime(weights, capacity=2, prefer_cpp=False)
    runtime.global_state.copy_(torch.tensor([1.0, -1.0], dtype=torch.float32))
    runtime.slot_state[1].copy_(torch.tensor([2.0], dtype=torch.float32))
    features = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)

    decision = runtime.score_slot_with_snapshot(features, slot=1)

    assert isinstance(decision, CpuSsmControllerDecision)
    torch.testing.assert_close(decision.features, features)
    torch.testing.assert_close(
        decision.global_state,
        torch.tensor([1.0, -1.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        decision.slot_state,
        torch.tensor([2.0], dtype=torch.float32),
    )

    expected = forward_step(
        features,
        CpuSsmControllerState(
            global_state=decision.global_state,
            slot_state=decision.slot_state,
        ),
        weights,
        prefer_cpp=False,
    )
    torch.testing.assert_close(decision.logit, expected.logit)
    torch.testing.assert_close(runtime.global_state, expected.global_state)
    torch.testing.assert_close(runtime.slot_state[1], expected.slot_state)

    # The snapshot must be detached from the mutable runtime buffers; C10 uses
    # it later for delayed replay credit after the runtime has advanced.
    runtime.global_state.zero_()
    runtime.slot_state[1].zero_()
    torch.testing.assert_close(
        decision.global_state,
        torch.tensor([1.0, -1.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        decision.slot_state,
        torch.tensor([2.0], dtype=torch.float32),
    )


def test_runtime_raises_when_weights_path_missing_for_trained_mode():
    """Trained-runtime modes (cpp_reference / cpu_ssm_reference) must
    refuse to construct from default zero weights — that previously
    produced ``controller_logit ≡ 0`` for every event, indistinguishable
    from "controller-on, no signal." The error message must point at the
    config knob the caller has to set.
    """
    mod = _load_runner_module()
    with pytest.raises(ValueError, match="episodic_controller_weights_path"):
        mod._build_controller_runtime_from_config(
            {
                "episodic_controller_runtime": "cpu_ssm_reference",
                "episodic_controller_weights_path": None,
            },
            capacity=4,
        )


def test_runtime_loads_when_weights_path_provided(tmp_path: Path):
    """Trained runtime constructs successfully when a real weights path
    is supplied. Mirrors the production wiring: dump weights to disk, set
    the config knob, builder loads them. ``prefer_cpp=False`` via the
    ``cpu_ssm_reference`` mode so the test is portable across environments
    that don't have the C++ extension built.
    """
    mod = _load_runner_module()
    weights = _weights()
    weights_path = tmp_path / "controller_weights.pt"
    weights.save(weights_path)

    runtime = mod._build_controller_runtime_from_config(
        {
            "episodic_controller_runtime": "cpu_ssm_reference",
            "episodic_controller_weights_path": str(weights_path),
        },
        capacity=4,
    )
    assert isinstance(runtime, CpuSsmControllerRuntime)
    assert runtime.slot_state.shape == (4, weights.slot_dim)
    assert runtime.global_state.shape == (weights.global_dim,)


def test_runtime_heuristic_mode_does_not_require_weights():
    """Heuristic mode short-circuits to ``None`` and never needs weights.
    Pin this so the I1 raise doesn't accidentally tighten the heuristic
    path — only the trained modes are affected.
    """
    mod = _load_runner_module()
    out = mod._build_controller_runtime_from_config(
        {"episodic_controller_runtime": "heuristic"},
        capacity=4,
    )
    assert out is None
    out_default = mod._build_controller_runtime_from_config({}, capacity=4)
    assert out_default is None
