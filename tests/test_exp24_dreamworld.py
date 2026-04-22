"""Tests for Experiment 24 Dreamworld v0 primitives."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "23_fast_path" / "dreamworld.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("exp24_dreamworld", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _TinyDreamModel(nn.Module):
    """Minimal stand-in model that records encode calls."""

    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 4, dtype=torch.float32)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, vocab_size, bias=False)
        self.encode_calls: list[dict[str, object]] = []

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        initial_states: list[torch.Tensor] | None = None,
        return_final_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        self.encode_calls.append(
            {
                "shape": tuple(input_ids.shape),
                "has_initial_states": initial_states is not None,
            }
        )
        input_ids = input_ids.to(torch.long)
        hidden = self.embed(input_ids)
        if initial_states:
            hidden = hidden + initial_states[0].to(hidden.device).to(hidden.dtype).unsqueeze(1)

        if return_final_states:
            final_state = torch.zeros(
                input_ids.shape[0],
                hidden.size(-1),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            if initial_states is not None:
                final_state = initial_states[0].to(final_state)
            return hidden, [final_state]
        return hidden


def test_dream_replay_buffer_detaches_evicts_and_ages_entries() -> None:
    mod = _load_module()
    buffer = mod.DreamReplayBuffer(max_entries=2, max_age_steps=2)

    states_a = [torch.ones(2, 4, requires_grad=True)]
    tokens_a = torch.tensor([[0, 1, 2]], dtype=torch.long)
    buffer.add(step=0, states=states_a, replay_tokens=tokens_a)
    assert len(buffer) == 1

    buffer.add(
        step=1,
        states=[torch.full((2, 4), 2.0)],
        replay_tokens=torch.tensor([[3, 4, 5]], dtype=torch.long),
    )
    assert len(buffer) == 2

    stored_a = buffer.sample(torch.Generator().manual_seed(0), current_step=1)
    assert stored_a is not None
    assert stored_a.states[0].requires_grad is False
    assert stored_a.replay_tokens.requires_grad is False
    assert stored_a.states[0].data_ptr() != states_a[0].data_ptr()
    assert stored_a.replay_tokens.data_ptr() != tokens_a.data_ptr()

    buffer.add(
        step=2,
        states=[torch.full((2, 4), 3.0)],
        replay_tokens=torch.tensor([[6, 7, 8]], dtype=torch.long),
    )
    diag = buffer.diagnostics(current_step=2)
    assert len(buffer) == 2
    assert diag["size"] == 2
    assert diag["drop_count"] == 1
    assert diag["add_count"] == 3
    assert diag["age_min"] == 0
    assert diag["age_max"] == 1
    assert pytest.approx(diag["age_mean"], rel=0, abs=1e-6) == 0.5

    buffer.add(
        step=5,
        states=[torch.full((2, 4), 4.0)],
        replay_tokens=torch.tensor([[9, 10, 11]], dtype=torch.long),
    )
    diag = buffer.diagnostics(current_step=5)
    assert len(buffer) == 1
    assert diag["size"] == 1
    assert diag["drop_count"] == 3
    assert diag["age_min"] == 0

    empty = mod.DreamReplayBuffer(max_entries=2, max_age_steps=2)
    empty_diag = empty.diagnostics(current_step=99)
    assert empty_diag["age_min"] == 0
    assert empty_diag["age_max"] == 0
    assert empty_diag["age_mean"] == 0.0


def test_build_dream_replay_tokens_slices_seed_plus_targets() -> None:
    mod = _load_module()
    inputs = torch.arange(20).view(2, 10)
    replay = mod.build_dream_replay_tokens(inputs=inputs, prefix_tokens=4, replay_tokens=3)
    expected = torch.tensor([[3, 4, 5, 6], [13, 14, 15, 16]])
    assert torch.equal(replay, expected)


def test_capture_dream_entry_uses_state_before_replay_seed() -> None:
    mod = _load_module()
    model = _TinyDreamModel(vocab_size=16)
    inputs = (torch.arange(20).view(2, 10) % 16)
    entry = mod.capture_dream_entry(
        model,
        inputs,
        step=7,
        prefix_tokens=4,
        replay_tokens=3,
    )

    expected = torch.tensor([[3, 4, 5, 6], [13, 14, 15, 0]])
    assert torch.equal(entry.replay_tokens, expected)
    assert len(model.encode_calls) == 1
    assert model.encode_calls[0] == {"shape": (2, 3), "has_initial_states": False}


def test_dreamworld_replay_backward_uses_cached_initial_state() -> None:
    mod = _load_module()
    model = _TinyDreamModel(vocab_size=8)
    entry = mod.DreamReplayEntry(
        step=11,
        states=[torch.ones(2, 4)],
        replay_tokens=torch.tensor(
            [[1, 2, 3], [4, 5, 6]], dtype=torch.long
        ),
    )

    loss = mod.dreamworld_replay_backward(model, entry, weight=0.5)
    assert loss.requires_grad is False
    assert len(model.encode_calls) == 1
    assert model.encode_calls[0] == {"shape": (2, 2), "has_initial_states": True}
    assert model.embed.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_dreamworld_replay_backward_can_subbatch_and_use_fused_ce(monkeypatch) -> None:
    mod = _load_module()
    model = _TinyDreamModel(vocab_size=8)
    entry = mod.DreamReplayEntry(
        step=11,
        states=[torch.ones(4, 4)],
        replay_tokens=torch.tensor(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7],
            ],
            dtype=torch.long,
        ),
    )
    calls: dict[str, object] = {}

    def fake_backend_for_mode(mode: str) -> str:
        calls["mode"] = mode
        return "streaming_v2"

    def fake_fused_backward(
        hidden: torch.Tensor,
        final_norm: nn.Module,
        lm_head: nn.Linear,
        targets: torch.Tensor,
        *,
        backend: str,
        tile_size: int,
        loss_weight: float,
    ) -> torch.Tensor:
        calls["hidden_shape"] = tuple(hidden.shape)
        calls["target_shape"] = tuple(targets.shape)
        calls["backend"] = backend
        calls["tile_size"] = tile_size
        calls["loss_weight"] = loss_weight
        loss = final_norm(hidden).mean() + 0.01 * lm_head.weight.mean()
        (loss * loss_weight).backward()
        return loss.detach()

    monkeypatch.setattr(mod, "fused_lm_head_backend_for_mode", fake_backend_for_mode)
    monkeypatch.setattr(mod, "fused_lm_head_backward", fake_fused_backward)

    loss = mod.dreamworld_replay_backward(
        model,
        entry,
        weight=0.5,
        lm_head_backward_mode="fused_streaming_v2",
        lm_head_tile_size=8192,
        replay_batch_size=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert loss.requires_grad is False
    assert len(model.encode_calls) == 1
    assert model.encode_calls[0] == {"shape": (2, 3), "has_initial_states": True}
    assert calls == {
        "mode": "fused_streaming_v2",
        "hidden_shape": (2, 3, 4),
        "target_shape": (2, 3),
        "backend": "streaming_v2",
        "tile_size": 8192,
        "loss_weight": 0.5,
    }
    assert model.embed.weight.grad is not None
    assert model.lm_head.weight.grad is not None
