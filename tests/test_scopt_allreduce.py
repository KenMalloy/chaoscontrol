from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from chaoscontrol.optim.scopt import scopt_allreduce_config


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("runner_scopt_allreduce", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scopt_allreduce_uses_train_prescale_sum_for_episodic_topology() -> None:
    cfg = scopt_allreduce_config(world_size=4, all_group=object())

    train_grads = torch.tensor([3.0, 6.0, 9.0])
    train_average = train_grads.mean()
    old_avg_over_all = torch.cat([train_grads, torch.zeros(1)]).mean()
    new_sum_all_group = torch.cat(
        [train_grads * cfg.train_grad_scale, torch.zeros(1)]
    ).sum()

    assert cfg.op == "sum"
    assert cfg.materialize_zeros is True
    assert cfg.train_rank_count == 3
    assert cfg.train_grad_scale == pytest.approx(1.0 / 3.0)
    assert new_sum_all_group.item() == pytest.approx(train_average.item())
    assert old_avg_over_all.item() / new_sum_all_group.item() == pytest.approx(0.75)


def test_scopt_allreduce_uses_sum_prescale_for_symmetric_ddp() -> None:
    cfg = scopt_allreduce_config(world_size=4, all_group=None)

    rank_grads = torch.tensor([2.0, 4.0, 6.0, 8.0])
    new_sum = (rank_grads * cfg.train_grad_scale).sum()

    assert cfg.op == "sum"
    assert cfg.materialize_zeros is False
    assert cfg.train_rank_count == 4
    assert cfg.train_grad_scale == pytest.approx(0.25)
    assert new_sum.item() == pytest.approx(rank_grads.mean().item())


def test_scopt_dense_state_sync_uses_same_sum_all_group_convention(monkeypatch):
    runner = _load_runner()
    group = object()
    calls = []

    def fake_all_reduce(tensor, *, op, group=None):
        calls.append((tensor.detach().clone(), op, group))

    monkeypatch.setattr(runner.dist, "all_reduce", fake_all_reduce)
    dense = torch.tensor([3.0, 6.0, 9.0])

    runner._sync_scopt_dense_tensors_coalesced(
        [dense],
        world_size=4,
        all_group=group,
    )

    assert len(calls) == 1
    reduced, op, seen_group = calls[0]
    assert op == runner.dist.ReduceOp.SUM
    assert seen_group is group
    torch.testing.assert_close(reduced, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(dense, torch.tensor([1.0, 2.0, 3.0]))
