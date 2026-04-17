"""Unit tests for runner_persistent_ddp warmup-restore helper.

The helper must leave the model byte-equivalent to its pre-warmup state
so that reported bpb is free of warmup-phase weight updates — matches
the Parameter Golf submission harness contract (``baselines/parameter_golf/
train_gpt.py`` lines 935-961): warmup primes compile/kernels/allocator,
restore loads initial weights back, timer starts from the restored
state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "19_prereqs"))
sys.path.insert(0, str(REPO / "experiments" / "18_throughput_levers"))
sys.path.insert(0, str(REPO / "experiments" / "17_local_attn_sidecar"))
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.model import ChaosStudentLM  # noqa: E402
from runner_persistent_ddp import _warmup_and_restore  # noqa: E402


def _tiny_model() -> ChaosStudentLM:
    torch.manual_seed(42)
    return ChaosStudentLM(
        vocab_size=32, dim=8, num_layers=1, ff_mult=2, a_mode="diag",
    )


class TestWarmupAndRestore:
    def test_model_weights_roundtrip_exactly(self) -> None:
        model = _tiny_model()
        initial_state = {
            name: tensor.clone() for name, tensor in model.state_dict().items()
        }

        def warmup_call() -> None:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p))

        built_count = [0]

        def build_optimizer() -> torch.optim.Optimizer:
            built_count[0] += 1
            return torch.optim.AdamW(model.parameters(), lr=1e-3)

        new_optimizer = _warmup_and_restore(
            model=model,
            warmup_call_fn=warmup_call,
            build_optimizer_fn=build_optimizer,
            device=torch.device("cpu"),
            ddp_active=False,
        )

        restored_state = model.state_dict()
        for name, initial_tensor in initial_state.items():
            assert torch.equal(restored_state[name], initial_tensor), (
                f"param {name!r} not restored to initial state"
            )
        assert built_count[0] == 1
        assert isinstance(new_optimizer, torch.optim.AdamW)

    def test_warmup_runs_before_optimizer_rebuild(self) -> None:
        model = _tiny_model()
        events: list[str] = []

        def warmup_call() -> None:
            events.append("warmup")
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1.0)

        def build_optimizer() -> torch.optim.Optimizer:
            events.append("build_optimizer")
            return torch.optim.AdamW(model.parameters(), lr=1e-3)

        _warmup_and_restore(
            model=model,
            warmup_call_fn=warmup_call,
            build_optimizer_fn=build_optimizer,
            device=torch.device("cpu"),
            ddp_active=False,
        )
        assert events == ["warmup", "build_optimizer"]

    def test_grads_cleared_after_restore(self) -> None:
        """Matches PG reference at baselines/parameter_golf/train_gpt.py:955.

        Warmup leaves per-param grads populated (optimizer.step consumes them
        but never zeros them; zero_grad fires at the START of the next iter).
        If the helper returns without clearing, the real timed pass starts
        with a full set of grad tensors allocated — perturbing first-step
        timing and peak-memory accounting relative to a true submission run.
        """
        model = _tiny_model()

        def warmup_call() -> None:
            with torch.no_grad():
                for p in model.parameters():
                    p.grad = torch.ones_like(p)

        def build_optimizer() -> torch.optim.Optimizer:
            return torch.optim.AdamW(model.parameters(), lr=1e-3)

        _warmup_and_restore(
            model=model,
            warmup_call_fn=warmup_call,
            build_optimizer_fn=build_optimizer,
            device=torch.device("cpu"),
            ddp_active=False,
        )

        for name, p in model.named_parameters():
            assert p.grad is None, (
                f"param {name!r} still has grads after _warmup_and_restore; "
                "timer-start state diverges from PG reference harness"
            )
