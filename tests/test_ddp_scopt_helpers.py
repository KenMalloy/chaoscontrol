"""DDP correctness test for the coalesced all-reduce helper in
``experiments/23_fast_path/runner_fast_path.py``.

The hot path used by ``ScarcityAwareOptimizer`` replaces N per-tensor
``all_reduce(SUM)+divide`` calls with a single flatten-reduce-unflatten
using ``ReduceOp.AVG``. The two forms are algebraically equivalent, but
the flatten path exercises ``torch._utils._flatten_dense_tensors`` and
an in-place ``copy_`` back into the caller's references — both of which
could silently corrupt gradients if wired wrong. This test runs both
helpers side by side on identical inputs and asserts bitwise (for
integer-averaging) or tolerance-based (for floating) equality.

Runs on CPU with ``gloo`` so no GPU is required; the helper is backend-
agnostic. ``mp.spawn`` launches two ranks locally.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load_runner_helpers():
    """Import the two helpers from the fast-path runner file.

    The runner lives under ``experiments/`` and is not importable as a
    package; load it by path so the test works regardless of cwd.
    """
    repo_root = Path(__file__).resolve().parents[1]
    runner_path = repo_root / "experiments" / "23_fast_path" / "runner_fast_path.py"
    spec = importlib.util.spec_from_file_location("_runner_fast_path", runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._average_dense_tensors, module._average_dense_tensors_coalesced


def _worker(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        per_tensor, coalesced = _load_runner_helpers()

        # Identical random base on every rank (same seed), then add a
        # rank-varying offset. This makes the analytic AVG exactly
        # ``base + (world_size-1)/2`` across ranks — bug-visible if flatten
        # or unflatten transposes/reorders anything. Shapes differ on
        # purpose to catch flatten/unflatten size mismatches.
        torch.manual_seed(1234)
        shapes = [(4, 8), (7,), (3, 3, 2)]
        base_tensors = [torch.randn(*shape, dtype=torch.float32) for shape in shapes]
        per_tensor_inputs = [base + float(rank) for base in base_tensors]
        coalesced_inputs = [t.clone() for t in per_tensor_inputs]

        per_tensor(per_tensor_inputs, world_size=world_size)
        coalesced(coalesced_inputs, world_size=world_size)

        # Both helpers must land on the same values on a single backend.
        max_abs = max(
            (a - b).abs().max().item()
            for a, b in zip(per_tensor_inputs, coalesced_inputs, strict=True)
        )

        # Analytic cross-check: AVG of (base + r) over r=0..W-1 is
        # base + (W-1)/2 on every rank. Checking on both ranks confirms
        # the reduce was a true all_reduce, not a rank-0-only read.
        expected_offset = (world_size - 1) / 2.0
        expected = [base + expected_offset for base in base_tensors]
        max_vs_expected = max(
            (a - e).abs().max().item()
            for a, e in zip(per_tensor_inputs, expected, strict=True)
        )

        with open(result_path + f".{rank}", "w") as fh:
            fh.write(f"{max_abs}\n{max_vs_expected}\n")
    finally:
        dist.destroy_process_group()


class TestCoalescedAllReduce(unittest.TestCase):
    def test_coalesced_matches_per_tensor(self) -> None:
        world_size = 2
        with tempfile.TemporaryDirectory() as tmp:
            result_base = os.path.join(tmp, "result")
            # Pick a port unlikely to clash with other DDP tests in the suite.
            master_port = 29731
            mp.spawn(
                _worker,
                args=(world_size, master_port, result_base),
                nprocs=world_size,
                join=True,
            )
            for rank in range(world_size):
                path = f"{result_base}.{rank}"
                with open(path) as fh:
                    lines = fh.read().strip().splitlines()
                max_abs = float(lines[0])
                max_vs_expected = float(lines[1])
                self.assertLess(
                    max_abs,
                    1e-6,
                    f"rank {rank}: coalesced vs per-tensor diverged by {max_abs}",
                )
                self.assertLess(
                    max_vs_expected,
                    1e-6,
                    f"rank {rank}: reduced value diverged from analytic expectation "
                    f"by {max_vs_expected}",
                )


if __name__ == "__main__":
    unittest.main()
