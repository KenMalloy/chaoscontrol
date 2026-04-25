"""mp.spawn / gloo coverage for the Phase 1 Task 1.3 3+1 skip-main flow.

Phase 1 Task 1.3 of the memory-aware optimizer plan: the runner's
canonical train step must support an asymmetric 3+1 topology where
rank ``world_size - 1`` skips the main forward+backward and the train
ranks combine their gradients via a single SUM all-reduce that the
episodic rank participates in with all-zero contribution. The test
exercises the collective pattern directly — train ranks run a real
forward+backward against a tiny in-memory model, the episodic rank
skips that work entirely, and all 4 ranks then call

    allreduce_grads(model, world_size, group=all_group,
                    op=ReduceOp.SUM, materialize_zeros=True)

with the train ranks pre-scaled by ``1 / (world_size - 1)``. After the
collective every rank must hold the train-rank average of grads,
bit-identical across ranks.

Test:

* ``test_3plus1_skip_main_yields_main_avg_only`` — the 4-rank smoke.
  Per-rank seed picks distinct grad contributions; the assertion is
  (a) rank 0 grads equal the independently-computed train-rank
  average, and (b) all 4 ranks hold bit-identical grads. The kwarg-
  level coverage of ``allreduce_grads`` itself lives in
  ``test_distributed_allreduce_grads.py``; this file pins the runner-
  specific call shape and the cross-rank consistency invariant.

Runs on CPU with the ``gloo`` backend so no GPU is required. Follows
the file-backed result + ``dist.barrier`` + ``finally:
destroy_process_group`` pattern from
``test_distributed_allreduce_grads.py`` because mp.spawn rendezvous
on local CPU is racy on macOS (see ``feedback_mp_spawn_flake.md``); a
fresh ephemeral port per spawn keeps the flakes off this file.
"""
from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from chaoscontrol.distributed import allreduce_grads


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Ask the OS for an ephemeral TCP port and immediately release it.

    Same idiom as ``test_distributed_allreduce_grads._pick_free_port``.
    A fresh port per ``mp.spawn`` invocation prevents collisions when
    multiple multi-process tests run back-to-back in the same pytest
    session.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_multi_param_model() -> nn.Module:
    """Tiny multi-layer model with multiple parameter groups.

    Two Linear layers + a LayerNorm so the flatten path concatenates
    weights, biases, and norm scales/offsets — exercises
    ``materialize_zeros`` over heterogeneous tensor types and shapes,
    not just one Linear's weight. The total count stays well under 1K
    params so gloo's CPU collective is cheap.
    """
    return nn.Sequential(
        nn.Linear(4, 8, bias=True),
        nn.LayerNorm(8),
        nn.Linear(8, 2, bias=True),
    )


def _train_rank_input(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-rank synthetic batch with a distinct seed.

    A different seed per train rank guarantees the per-rank gradient
    contributions are distinct, which makes the post-allreduce mean a
    non-trivial function of every train rank's contribution — a bug
    where one rank's grad is silently dropped (e.g. wrong group, wrong
    op) would still pass an all-equal grad smoke test, but cannot pass
    this distinct-grad average test.
    """
    torch.manual_seed(100 + rank)
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    return x, y


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _worker_3plus1_skip_main(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
) -> None:
    """4-rank gloo: 3 train + 1 episodic, then verify 3+1 SUM collective.

    Train ranks 0/1/2: build the model, draw their per-rank batch,
    run a real forward+backward, then pre-scale all grads by
    1/(world_size - 1) before the SUM all-reduce.

    Episodic rank 3: build the model with the same init seed (so the
    parameters match across ranks) but run NO forward and NO backward;
    leave every ``param.grad`` as None. The ``materialize_zeros=True``
    inside the collective then zero-fills its grads in place.

    Post-collective every rank dumps its full flattened grad tensor
    to a file. The parent process (a) computes the expected train-
    rank average independently from the same per-rank seeds, (b)
    asserts rank 0's dump matches that expected average within fp32
    tolerance, and (c) asserts all 4 ranks' dumps are bit-identical.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # Same model init seed on every rank so parameters and gradient
        # shapes agree — broadcast_params is the runner's normal path,
        # but using a deterministic seed here is equivalent and avoids
        # an unrelated collective.
        torch.manual_seed(11)
        model = _build_multi_param_model()

        n_train = world_size - 1
        is_episodic_rank = (rank == world_size - 1)

        # All ranks must call ``new_group`` — it is itself a WORLD
        # collective on gloo/nccl, exactly mirroring the runner's
        # behavior at init.
        all_group = dist.new_group(list(range(world_size)))

        if not is_episodic_rank:
            # Train rank: real forward+backward against per-rank input,
            # then caller-side pre-scale by 1/N_train so the SUM
            # collective + zero-fill on the episodic rank reconstructs
            # the train-rank average.
            x, y = _train_rank_input(rank)
            pred = model(x)
            loss = (pred - y).pow(2).mean()
            loss.backward()
            inv_n_train = 1.0 / float(n_train)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(inv_n_train)
        # else: episodic rank — leave param.grad as None for every
        # parameter. ``materialize_zeros=True`` zero-fills inside the
        # collective.

        allreduce_grads(
            model,
            world_size,
            group=all_group,
            op=dist.ReduceOp.SUM,
            materialize_zeros=True,
        )

        # Dump every rank's full flat grad so the parent can compute
        # cross-rank identity AND compare against the independently
        # computed train-rank-average reference.
        flat = torch.cat(
            [p.grad.detach().flatten() for p in model.parameters()]
        )
        torch.save(flat, result_path + f".{rank}.pt")

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------


class TestRunner3Plus1SkipMain(unittest.TestCase):
    """4-rank gloo coverage for the Phase 1 3+1 collective shape."""

    @staticmethod
    def _spawn(worker, world_size: int, tmp: str) -> str:
        """Spawn ``world_size`` ranks running ``worker`` and return the
        per-rank result file prefix.
        """
        result_path = os.path.join(tmp, "result")
        master_port = _pick_free_port()
        mp.spawn(
            worker,
            args=(world_size, master_port, result_path),
            nprocs=world_size,
            join=True,
        )
        return result_path

    def test_3plus1_skip_main_yields_main_avg_only(self) -> None:
        """3 train ranks + 1 episodic rank: SUM collective with caller
        pre-scale + zero-fill must equal the train-rank average on every
        rank.

        Independently rebuilds the model from the same init seed, runs
        a forward+backward against each train rank's distinct batch,
        averages the per-rank grads, and asserts (a) rank 0's
        post-collective grad equals that average within fp32 tolerance
        AND (b) all 4 ranks hold bit-identical grads after the call.
        """
        world_size = 4
        with tempfile.TemporaryDirectory() as tmp:
            result_path = self._spawn(
                _worker_3plus1_skip_main, world_size, tmp,
            )

            # Load each rank's post-collective flat-grad dump.
            grads_per_rank = [
                torch.load(f"{result_path}.{rank}.pt", weights_only=True)
                for rank in range(world_size)
            ]

            # Cross-rank bit-identity is the load-bearing invariant of
            # an all-reduce. Any drift implies the collective produced
            # different results on different ranks — a routing or op
            # bug, not a numeric one.
            for rank in range(1, world_size):
                max_drift = (
                    grads_per_rank[0] - grads_per_rank[rank]
                ).abs().max().item()
                self.assertEqual(
                    max_drift,
                    0.0,
                    f"rank {rank}: post-allreduce grads must be "
                    f"bit-identical to rank 0; max abs diff = {max_drift}",
                )

            # Independently compute the expected train-rank-average
            # gradient. Same init seed, same per-rank batch seeds, same
            # loss, mean over the train ranks.
            n_train = world_size - 1
            torch.manual_seed(11)
            ref = _build_multi_param_model()
            sum_grads: list[torch.Tensor] | None = None
            for train_rank in range(n_train):
                # Reset grads then run a fresh forward+backward for
                # this train rank's batch.
                for p in ref.parameters():
                    p.grad = None
                x, y = _train_rank_input(train_rank)
                pred = ref(x)
                loss = (pred - y).pow(2).mean()
                loss.backward()
                if sum_grads is None:
                    sum_grads = [
                        p.grad.detach().clone() for p in ref.parameters()
                    ]
                else:
                    for s, p in zip(sum_grads, ref.parameters(), strict=True):
                        s.add_(p.grad)
            assert sum_grads is not None  # n_train >= 1 by construction
            expected_flat = torch.cat(
                [(s / float(n_train)).flatten() for s in sum_grads]
            )

            # Rank 0's collective output must equal the independently-
            # computed train-rank average. fp32 tolerance is generous
            # here — the only inexactness is the order of additions
            # inside the collective vs the independent reference.
            actual_flat = grads_per_rank[0]
            self.assertEqual(
                actual_flat.shape,
                expected_flat.shape,
                "post-allreduce flat-grad shape must match the reference",
            )
            max_diff = (actual_flat - expected_flat).abs().max().item()
            self.assertLess(
                max_diff,
                1e-5,
                "rank 0 post-allreduce grads must equal the independently-"
                f"computed train-rank average; max abs diff = {max_diff}",
            )


if __name__ == "__main__":
    unittest.main()
