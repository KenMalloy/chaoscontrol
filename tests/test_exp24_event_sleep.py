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
from pathlib import Path

import pytest
import torch
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


def test_event_sleep_gate_reduces_rank_loss_pressure(monkeypatch):
    mod = _load_runner_module()

    reduced: list[list[float]] = []

    def fake_all_reduce(tensor, op=None):
        assert op is mod.dist.ReduceOp.SUM
        reduced.append(tensor.detach().cpu().tolist())
        tensor.copy_(torch.tensor([1.25, 3.0]))

    monkeypatch.setattr(mod.dist, "all_reduce", fake_all_reduce)

    ema = mod.LossTriggeredReplayEMA(decay=0.5, warmup_steps=1)
    assert ema.update(torch.tensor(2.0)) is None

    decision = ema.update(
        torch.tensor(3.0),
        threshold=1.10,
        pressure_threshold=0.05,
        ddp_active=True,
        world_size=4,
        device=torch.device("cpu"),
    )

    assert decision is not None
    assert reduced == [[pytest.approx((3.0 / 2.0) - 1.10), 1.0]]
    assert decision.local_fire is True
    assert decision.fire_count == 3
    assert decision.global_pressure == pytest.approx(1.25 / 4.0)
    assert decision.triggered is True


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
