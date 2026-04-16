"""DDP smoke test for ``train_ssm.train_ssm_for_budget``.

Mirrors the shape of ``test_ddp_integration.py`` — spawns two gloo-backend
workers on CPU, runs the lean training loop end-to-end, and asserts the
post-training parameters are identical across ranks. That identity is
the signature of correct gradient all-reduce: if the new
``allreduce_grads`` or ``should_stop_now`` wiring had a bug, ranks
would diverge within a handful of steps.
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from chaoscontrol.core import verify_diag_recurrence
from chaoscontrol.model import ChaosStudentLM


MODEL_KWARGS = dict(
    vocab_size=64,
    dim=16,
    num_layers=2,
    ff_mult=2,
    a_mode="diag",
    rich_b_mode="none",
    outer_model_dim=0,
)

N_TRAIN_TOKENS = 1024
TRAIN_STARTS = list(range(0, 800, 8))

# Pre-warm at import so the first worker doesn't eat its budget on
# torch.compile resolution.
verify_diag_recurrence(torch.device("cpu"))


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_model(seed: int) -> ChaosStudentLM:
    torch.manual_seed(seed)
    return ChaosStudentLM(**MODEL_KWARGS)


def _build_tokens(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, MODEL_KWARGS["vocab_size"], (N_TRAIN_TOKENS,), generator=g)


class _DeterministicClock:
    """Monotonic fake clock so budget-loop semantics don't depend on host speed."""

    def __init__(self, *, start: float = 0.0, step: float = 0.25) -> None:
        self._t = start
        self._step = step

    def __call__(self) -> float:
        current = self._t
        self._t += self._step
        return current


def _ddp_train_ssm_worker(
    rank: int,
    world_size: int,
    master_port: int,
    param_dump_paths: list[str],
    stdout_log_paths: list[str],
    seed: int,
) -> None:
    """Worker: init PG, build model, call train_ssm_for_budget, dump params."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.use_deterministic_algorithms(True, warn_only=True)

    # Route stdout per-rank so pytest capture stays clean.
    with open(stdout_log_paths[rank], "w") as log_file:
        sys.stdout = log_file
        try:
            dist.init_process_group(
                backend="gloo", rank=rank, world_size=world_size,
            )

            from chaoscontrol.train_ssm import train_ssm_for_budget

            model = _build_model(seed)
            tokens = _build_tokens(seed)
            verify_diag_recurrence(torch.device("cpu"))
            clock = _DeterministicClock(step=0.25)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            result = train_ssm_for_budget(
                model=model,
                train_tokens=tokens,
                train_starts=list(TRAIN_STARTS),
                seq_len=16,
                batch_size=4,
                device=torch.device("cpu"),
                optimizer=optimizer,
                budget_seconds=1.1,
                chunk_size=16,
                seed=seed,
                rank=rank,
                world_size=world_size,
                time_fn=clock,
            )
            # Assertions inside the worker show up in the log if they fire;
            # the parent also checks the same invariants post-spawn.
            assert result["rank"] == rank
            assert result["world_size"] == world_size
            assert result["steps"] == 4
            assert result["elapsed_s"] == 1.5

            flat = torch.cat([p.detach().flatten() for p in model.parameters()])
            torch.save(flat, param_dump_paths[rank])
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            sys.stdout = sys.__stdout__


class TestTrainSSMDDPSmoke(unittest.TestCase):
    """Two-rank smoke test: params must match across ranks after training."""

    def test_ws2_params_match_across_ranks(self) -> None:
        seed = 7777
        world_size = 2

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            param_dump_paths = [str(tmpdir / f"params_rank{r}.pt") for r in range(world_size)]
            stdout_log_paths = [str(tmpdir / f"stdout_rank{r}.log") for r in range(world_size)]
            master_port = _pick_free_port()

            mp.spawn(
                _ddp_train_ssm_worker,
                args=(
                    world_size,
                    master_port,
                    param_dump_paths,
                    stdout_log_paths,
                    seed,
                ),
                nprocs=world_size,
                join=True,
            )

            # Every rank must have written its param dump — if a worker
            # failed inside the spawn it'll show up as a missing file.
            for p in param_dump_paths:
                self.assertTrue(
                    Path(p).exists(),
                    msg=f"missing param dump at {p}; check stdout logs",
                )

            dumps = [torch.load(p, weights_only=True) for p in param_dump_paths]

            # The defining property of correct all-reduce: identical
            # parameters across ranks after each optimizer step. Any
            # drift here proves the gradient sync is broken.
            for r in range(1, world_size):
                max_diff = (dumps[0] - dumps[r]).abs().max().item()
                self.assertEqual(
                    max_diff,
                    0.0,
                    msg=(
                        f"rank 0 vs rank {r} parameter drift {max_diff} — "
                        f"all-reduce failed to keep params in lockstep. "
                        f"check stdout log: {stdout_log_paths[r]}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
