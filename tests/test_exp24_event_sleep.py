"""Regression coverage for the Exp 24 `event_sleep` training-time mechanism.

The hot-loop implementation lives with the Exp 23 fast-path runner
(`experiments/23_fast_path/runner_fast_path.py`) which is the shared
training substrate. This file owns the Exp 24-specific invariants:
DDP lockstep, one-step delay, seed-anchored trigger trajectory,
warmup-restore bit equality, and bf16 loss handling.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


_BACKEND_ENV_KEYS = (
    "CHAOSCONTROL_DIAG_SCAN_BACKEND",
    "CHAOSCONTROL_POST_SCAN_BACKEND",
)


@pytest.fixture(autouse=True)
def _restore_backend_env_after_test():
    """Restore Exp23 backend env vars after runner module imports."""
    snapshot = {key: os.environ.get(key) for key in _BACKEND_ENV_KEYS}
    try:
        yield
    finally:
        for key, value in snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


_RUNNER_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "23_fast_path"
    / "runner_fast_path.py"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "runner_fast_path_event_sleep", _RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TinyTokenTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(6, 4)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, 6, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embed(inputs)


def _gate_worker(rank, world_size, init_file, per_rank_loss_map, result_path):
    """Spawned-process worker: init gloo, run one gate update, write result."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mod = _load_runner_module()
        gate = mod.LossTriggeredReplayEMA(decay=0.5, warmup_steps=1)
        seed_loss, trigger_loss = per_rank_loss_map[rank]
        gate.update(torch.tensor(seed_loss))
        decision = gate.update(
            torch.tensor(trigger_loss),
            threshold=1.25,
            pressure_threshold=0.05,
            ddp_active=True,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        payload = {
            "triggered": bool(decision.triggered),
            "global_pressure": float(decision.global_pressure),
            "fire_count": int(decision.fire_count),
            "local_fire": bool(decision.local_fire),
            "local_ratio": float(decision.local_ratio),
        }
        torch.save(payload, f"{result_path}.rank{rank}")
    finally:
        dist.destroy_process_group()


def test_gloo_four_rank_lockstep_and_boundary(tmp_path, monkeypatch):
    """All ranks see identical triggered / global_pressure / fire_count."""
    world_size = 4
    per_rank_loss = {
        0: (2.0, 2.75),  # ratio 1.375, above threshold 1.25: fires
        1: (2.0, 2.00),  # ratio 1.00, below: no fire
        2: (2.0, 2.50),  # ratio 1.25, exactly at threshold: no fire
        3: (2.0, 3.00),  # ratio 1.50, above: fires
    }

    result_path = tmp_path / "decisions"
    init_file = tmp_path / "rendezvous"
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        monkeypatch.setenv(
            "GLOO_SOCKET_IFNAME",
            "lo0" if sys.platform == "darwin" else "lo",
        )

    mp.spawn(
        _gate_worker,
        args=(world_size, str(init_file), per_rank_loss, str(result_path)),
        nprocs=world_size,
        join=True,
    )

    decisions = [
        torch.load(f"{result_path}.rank{rank}") for rank in range(world_size)
    ]

    expected_local_fire = [True, False, False, True]
    for rank, decision in enumerate(decisions):
        assert decision["local_fire"] is expected_local_fire[rank], (
            f"rank {rank}: local_fire {decision['local_fire']} != "
            f"{expected_local_fire[rank]}"
        )

    assert decisions[2]["local_ratio"] == pytest.approx(1.25, abs=0.0)
    assert decisions[2]["local_fire"] is False

    fire_count_ref = decisions[0]["fire_count"]
    global_pressure_ref = decisions[0]["global_pressure"]
    triggered_ref = decisions[0]["triggered"]
    assert fire_count_ref == 2, "ranks 0 and 3 fired -> fire_count = 2"
    for decision in decisions:
        assert decision["fire_count"] == fire_count_ref
        assert decision["global_pressure"] == pytest.approx(global_pressure_ref)
        assert decision["triggered"] is triggered_ref


def test_event_sleep_gate_waits_for_warmup_and_threshold():
    mod = _load_runner_module()
    ema = mod.LossTriggeredReplayEMA(decay=0.5, warmup_steps=2)

    assert ema.update(torch.tensor(2.0)) is None
    assert ema.update(torch.tensor(2.2), threshold=1.05) is None

    decision = ema.update(
        torch.tensor(2.31),
        threshold=1.05,
        pressure_threshold=0.20,
        ddp_active=False,
        world_size=1,
        device=torch.device("cpu"),
    )

    assert decision is not None
    assert decision.local_ratio == pytest.approx(2.31 / 2.1)
    assert decision.local_fire is True
    assert decision.global_pressure < 0.20
    assert decision.triggered is False


def test_train_fast_for_budget_runs_event_sleep_replay(monkeypatch):
    mod = _load_runner_module()
    events = []

    class FakeDreamBuffer:
        def __init__(self, **kwargs):
            self.entries = []

        def __len__(self):
            return len(self.entries)

        def add(self, *, step, states, replay_tokens):
            self.entries.append((step, states, replay_tokens))

        def sample(self, *, generator, current_step):
            events.append(("sample", current_step))
            return object()

        def diagnostics(self, *, current_step):
            return {"size": len(self.entries), "current_step": current_step}

    class FakeEventGate:
        def __init__(self, **kwargs):
            pass

        def update(self, loss, **kwargs):
            return mod.LossTriggeredReplayDecision(
                local_loss=float(loss.detach().float().item()),
                ema_loss=1.0,
                local_ratio=1.25,
                local_pressure=0.15,
                global_pressure=0.15,
                fire_count=1,
                local_fire=True,
                triggered=True,
            )

    monkeypatch.setattr(mod, "DreamReplayBuffer", FakeDreamBuffer)
    monkeypatch.setattr(mod, "LossTriggeredReplayEMA", FakeEventGate)
    monkeypatch.setattr(
        mod,
        "capture_dream_entry",
        lambda model, inputs, **kwargs: type(
            "E",
            (),
            {
                "step": kwargs["step"],
                "states": [torch.zeros(inputs.size(0), 4)],
                "replay_tokens": inputs[:, :3],
            },
        )(),
    )

    def fake_replay_backward(model, *, entry, weight, **kwargs):
        events.append(("replay", float(weight)))
        return torch.tensor(0.1)

    monkeypatch.setattr(mod, "dreamworld_replay_backward", fake_replay_backward)

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=6,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=False,
        dreamworld_enabled=True,
        dreamworld_cache_interval=1,
        dreamworld_interval=0,
        dreamworld_weight=0.25,
        dreamworld_prefix_tokens=3,
        dreamworld_replay_tokens=2,
        dreamworld_min_size=1,
        event_sleep_enabled=True,
        event_sleep_min_interval=1,
        event_sleep_weight=0.5,
    )

    assert ("sample", 0) not in events
    assert ("sample", 1) in events
    assert ("replay", 0.5) in events
    event_diag = result["mechanisms"]["event_sleep"]
    assert event_diag["enabled"] is True
    assert event_diag["trigger_count"] == 1
    assert event_diag["replay_count"] == 1
    assert event_diag["decision_count"] == 2
    assert event_diag["mean_global_pressure"] == pytest.approx(0.15)
    assert event_diag["last_decision"]["fire_count"] == 1
    assert event_diag["artifact_impact"] == "artifact_training_only"


def test_event_sleep_replay_lands_one_step_after_trigger(monkeypatch):
    mod = _load_runner_module()

    sampled_steps: list[int] = []

    class InstrumentedBuffer(mod.DreamReplayBuffer):
        def sample(self, *, generator, current_step):
            sampled_steps.append(int(current_step))
            return super().sample(generator=generator, current_step=current_step)

    monkeypatch.setattr(mod, "DreamReplayBuffer", InstrumentedBuffer)

    fire_steps = {5}

    class ScriptedGate:
        def __init__(self, **kwargs):
            self._step = 0

        def update(self, loss, **kwargs):
            step = self._step
            self._step += 1
            if step in fire_steps:
                return mod.LossTriggeredReplayDecision(
                    local_loss=1.0,
                    ema_loss=1.0,
                    local_ratio=1.3,
                    local_pressure=0.2,
                    global_pressure=0.2,
                    fire_count=1,
                    local_fire=True,
                    triggered=True,
                )
            return mod.LossTriggeredReplayDecision(
                local_loss=1.0,
                ema_loss=1.0,
                local_ratio=0.95,
                local_pressure=0.0,
                global_pressure=0.0,
                fire_count=0,
                local_fire=False,
                triggered=False,
            )

    monkeypatch.setattr(mod, "LossTriggeredReplayEMA", ScriptedGate)
    monkeypatch.setattr(
        mod,
        "capture_dream_entry",
        lambda model, inputs, **kwargs: type(
            "E",
            (),
            {
                "step": kwargs["step"],
                "states": [torch.zeros(inputs.size(0), 4)],
                "replay_tokens": inputs[:, :3],
            },
        )(),
    )
    monkeypatch.setattr(
        mod,
        "dreamworld_replay_backward",
        lambda *args, **kwargs: torch.tensor(0.1),
    )

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=6,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=10,
        prefetch_batches=False,
        dreamworld_enabled=True,
        dreamworld_cache_interval=1,
        dreamworld_interval=0,
        dreamworld_weight=0.25,
        dreamworld_prefix_tokens=3,
        dreamworld_replay_tokens=2,
        dreamworld_min_size=1,
        event_sleep_enabled=True,
        event_sleep_min_interval=1,
        event_sleep_weight=0.5,
    )

    assert 5 not in sampled_steps, (
        f"replay leaked into trigger step: {sampled_steps}"
    )
    assert 6 in sampled_steps, (
        f"replay did not land one step after trigger: {sampled_steps}"
    )
    assert sampled_steps == [6], f"replay fired on unexpected steps: {sampled_steps}"


def test_gate_produces_exact_trigger_trajectory():
    mod = _load_runner_module()

    losses = []
    base = 2.5
    for step in range(40):
        base = 0.995 * base
        spike = 0.8 if step in (12, 24, 35) else 0.0
        losses.append(base + spike)

    gate = mod.LossTriggeredReplayEMA(decay=0.9, warmup_steps=5)

    fired_steps: list[int] = []
    for step, loss in enumerate(losses):
        decision = gate.update(
            torch.tensor(loss),
            threshold=1.10,
            pressure_threshold=0.02,
            ddp_active=False,
            world_size=1,
            device=torch.device("cpu"),
        )
        if decision is not None and decision.triggered:
            fired_steps.append(step)

    assert fired_steps == [12, 24, 35], (
        f"trigger trajectory drifted: got {fired_steps}, expected [12, 24, 35]"
    )


def test_warmup_restore_is_bit_equal():
    mod = _load_runner_module()

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    config = {
        "warmup_steps": 4,
        "chunk_size": 2,
        "grad_clip_norm": 0.0,
        "fused_grad_clip": False,
        "precision": "bf16",
        "prefetch_batches": False,
    }

    saved_state = mod._state_dict_clone(model)
    mod._warmup(
        model=model,
        train_tokens=torch.arange(128, dtype=torch.int16) % 6,
        train_num_tokens=128,
        stride=4,
        seq_len=6,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        config=config,
        rank=0,
        world_size=1,
        seed=1337,
        vocab_size=6,
    )

    mid_state = model.state_dict()
    drifted = any(
        not torch.equal(saved_state[name], mid_state[name]) for name in saved_state
    )
    assert drifted, "_warmup did not mutate any params - test is vacuous"

    mod._restore_state_dict(model, saved_state)
    post_state = model.state_dict()

    for name, pre_tensor in saved_state.items():
        post_tensor = post_state[name]
        assert torch.equal(pre_tensor, post_tensor), (
            f"param {name} drifted after warmup-restore: "
            f"max diff {(pre_tensor - post_tensor).abs().max().item()}"
        )


def test_gate_handles_bf16_loss_tensor():
    mod = _load_runner_module()

    losses_fp32 = [2.5, 2.48, 2.46, 2.45, 2.44, 3.20]
    losses_bf16 = [
        torch.tensor(value, dtype=torch.bfloat16).float().item()
        for value in losses_fp32
    ]

    gate_fp32 = mod.LossTriggeredReplayEMA(decay=0.5, warmup_steps=1)
    gate_bf16 = mod.LossTriggeredReplayEMA(decay=0.5, warmup_steps=1)

    fp32_decisions = []
    bf16_decisions = []
    for loss_fp32, loss_bf16 in zip(losses_bf16, losses_bf16, strict=True):
        d_fp32 = gate_fp32.update(
            torch.tensor(loss_fp32, dtype=torch.float32),
            threshold=1.10,
            pressure_threshold=0.05,
        )
        d_bf16 = gate_bf16.update(
            torch.tensor(loss_bf16, dtype=torch.bfloat16),
            threshold=1.10,
            pressure_threshold=0.05,
        )
        fp32_decisions.append(d_fp32)
        bf16_decisions.append(d_bf16)

    for d_fp32, d_bf16 in zip(fp32_decisions, bf16_decisions, strict=True):
        if d_fp32 is None:
            assert d_bf16 is None
            continue
        assert d_bf16 is not None
        assert d_fp32.triggered == d_bf16.triggered
        assert d_fp32.local_fire == d_bf16.local_fire
        assert torch.isfinite(torch.tensor(d_bf16.local_loss))
        assert torch.isfinite(torch.tensor(d_bf16.ema_loss))
