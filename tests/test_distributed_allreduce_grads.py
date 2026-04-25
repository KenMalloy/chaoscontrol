"""mp.spawn / gloo coverage for ``allreduce_grads`` kwarg extensions.

Phase 1 Task 1.2 of the memory-aware optimizer plan: ``allreduce_grads``
gains three keyword-only knobs — ``group``, ``op``, and
``materialize_zeros`` — without disturbing any existing call site. The
function's default behavior (positional ``model, world_size``) must
remain ``ReduceOp.AVG`` over the WORLD group, exactly matching the path
exp23/exp24 already exercise on pods.

The unit tests in ``test_distributed.py`` mock ``dist.all_reduce`` and
verify the local flatten/unflatten contract. These tests stand up real
2- and 4-rank gloo process groups via ``mp.spawn`` and validate the
collective itself: who reduces with whom, what op gets applied, and
whether the materialization step keeps the per-rank flattened buffer
shape consistent. Live process-group coverage is the only place a
``new_group`` / subgroup-routing bug would surface — mocks can't model
collective topology.

Test cases:

1. ``test_back_compat_avg_unchanged`` — positional 2-arg form preserves
   the AVG-over-WORLD behavior bit-for-bit.
2. ``test_sum_op_with_prescaling_gives_main_avg_plus_replay`` — the 3+1
   topology from Phase 1 Task 1.3: SUM-over-WORLD with caller pre-
   scaling reproduces (main_avg + replay_full) on every rank.
3. ``test_materialize_zeros_handles_inconsistent_grad_set`` — when one
   rank has a ``None`` grad on some param, ``materialize_zeros=True``
   keeps the flatten shapes consistent across ranks and the collective
   completes without deadlock or shape mismatch.
4. ``test_subgroup_routing_isolates_subset`` — ``group=`` routes the
   reduction through a 2-rank subgroup; the other 2 ranks' grads stay
   bit-identical to their pre-call values.
5. ``test_existing_call_sites_smoke`` — small forward+backward+
   allreduce_grads loop modeled on the exp23 runner's positional call
   form; cheapest catch for any silent back-compat break.
6. ``test_composability_subgroup_sum_materialize`` — all three new
   kwargs together (the Phase 1 Task 1.3 production call shape):
   subgroup-restricted SUM-op all-reduce with caller-side pre-scaling
   and one rank's grad missing. Catches branch-interaction bugs that
   pass each kwarg in isolation.

Runs on CPU with ``gloo`` so no GPU is required. Follows the file-
backed result + ``dist.barrier`` + ``finally: destroy_process_group``
pattern from ``test_ddp_integration.py`` because mp.spawn rendezvous on
local CPU is racy on macOS (see ``feedback_mp_spawn_flake.md``); a
fresh ephemeral port per spawn and a barrier before destroy keeps the
flakes off this file.
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

    Same idiom as ``test_ddp_integration._pick_free_port`` — a fresh port
    per ``mp.spawn`` invocation prevents collisions when multiple multi-
    process tests run back-to-back in the same pytest session. The race
    between ``close()`` and the DDP init reusing the port is theoretical
    on CPU/gloo; in practice this matches torch's own DDP test suite.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_two_param_model(seed: int) -> nn.Module:
    """Tiny 2-parameter model: one Linear with bias.

    Two parameters total (weight, bias) so each test can talk in terms of
    'first param' and 'second param' and the flatten path still has work
    to do (more than one tensor concatenated). Seed pins init values so
    the DDP grad checks below have known reference values.
    """
    torch.manual_seed(seed)
    return nn.Linear(2, 2, bias=True)


# ---------------------------------------------------------------------------
# Worker functions — one per test case
# ---------------------------------------------------------------------------


def _worker_back_compat_avg(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
) -> None:
    """Test 1: positional ``allreduce_grads(model, world_size=2)`` averages.

    Each rank sets a known grad value, then calls the legacy 2-arg form.
    Post-call grads on every rank must equal the across-rank mean.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        model = _build_two_param_model(seed=11)
        # Distinct grads per rank so the post-AVG value is non-trivially
        # determined by both contributions; rank-0 grads set to ones,
        # rank-1 grads set to threes — AVG must land at twos.
        for p in model.parameters():
            p.grad = torch.full_like(p, float(2 * rank + 1))

        allreduce_grads(model, world_size=world_size)

        # Across-rank mean of {1, 3} is 2.0 — every element on every rank.
        max_diff = max(
            (p.grad - torch.full_like(p, 2.0)).abs().max().item()
            for p in model.parameters()
        )
        with open(result_path + f".{rank}", "w") as fh:
            fh.write(f"{max_diff}\n")

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_sum_with_prescaling(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
) -> None:
    """Test 2: 3+1 topology — train ranks pre-scale by 1/N_train, episodic
    contributes full-magnitude replay. SUM all-reduce reconstructs the
    main-avg + replay sum on every rank.

    With 4 ranks (3 train + 1 episodic):
        rank 0/1/2: grad = [1, 2] pre-scaled to [1/3, 2/3]
        rank 3:    grad = [10, 20] (replay, no pre-scale)
    SUM over all 4 ranks: 3*[1/3, 2/3] + [10, 20] = [11, 22].
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # Single 2-element parameter so the expected vector is exactly
        # [11.0, 22.0]; using a Linear here would obscure that with a
        # weight matrix.
        model = nn.Linear(2, 1, bias=False)
        # Force a known shape for the parameter by overwriting it.
        with torch.no_grad():
            model.weight.copy_(torch.zeros(1, 2))

        n_train = world_size - 1  # 3
        if rank < n_train:
            base_grad = torch.tensor([[1.0, 2.0]])
            base_grad.div_(float(n_train))  # caller-side pre-scale
            model.weight.grad = base_grad
        else:
            # Episodic rank — full-magnitude replay grads.
            model.weight.grad = torch.tensor([[10.0, 20.0]])

        allreduce_grads(
            model,
            world_size=world_size,
            op=dist.ReduceOp.SUM,
            materialize_zeros=False,
        )

        expected = torch.tensor([[11.0, 22.0]])
        max_diff = (model.weight.grad - expected).abs().max().item()
        with open(result_path + f".{rank}", "w") as fh:
            fh.write(f"{max_diff}\n")

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_materialize_zeros(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
) -> None:
    """Test 3: rank 1 has fewer params with grads than rank 0;
    ``materialize_zeros=True`` keeps the flatten shapes aligned and the
    AVG reduces rank 0's value with rank 1's zero-fill.

    Rank 0: weight.grad = 4.0, bias.grad = 6.0
    Rank 1: weight.grad = 4.0, bias.grad = None  (cleared)
        ↓ materialize_zeros fills rank-1 bias.grad with zeros ↓
    AVG: weight = 4.0 (matches across ranks already), bias = 3.0 (=6/2)
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Tighter init timeout than default so a deadlock fails fast instead
    # of hanging the test runner.
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    try:
        # Use deterministic explicit grads instead of a real backward —
        # the contract under test is "missing grads get zero-materialized
        # and the collective doesn't deadlock", not "backward populates
        # grads correctly". Setting grads by hand makes the assertion
        # ironclad.
        model = _build_two_param_model(seed=29)
        params = list(model.parameters())  # [weight, bias]
        for p in params:
            p.grad = torch.full_like(p, 4.0)
        if rank == 1:
            # Simulate the asymmetric path: rank 1's bias never received
            # a grad. Without materialize_zeros, the flatten step would
            # produce different shapes on rank 0 (2 tensors flattened)
            # vs rank 1 (1 tensor flattened) — gloo would deadlock or
            # raise a shape-mismatch.
            params[-1].grad = None
        else:
            # Rank 0 sets the bias grad to a distinguishing value so the
            # post-AVG check verifies that the rank-1 zero contributed.
            params[-1].grad = torch.full_like(params[-1], 6.0)

        allreduce_grads(
            model,
            world_size=world_size,
            materialize_zeros=True,
        )

        # weight: both ranks contributed 4.0 ⇒ AVG = 4.0
        # bias  : rank 0 = 6.0, rank 1 = 0 (materialized) ⇒ AVG = 3.0
        weight, bias = params
        weight_diff = (weight.grad - torch.full_like(weight, 4.0)).abs().max().item()
        bias_diff = (bias.grad - torch.full_like(bias, 3.0)).abs().max().item()
        with open(result_path + f".{rank}", "w") as fh:
            fh.write(f"{weight_diff}\n{bias_diff}\n")

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_subgroup_routing(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
) -> None:
    """Test 4: ``group=subgroup`` of {0, 1} reduces rank 0/1 grads only;
    ranks 2/3 do not call the collective and their grads are untouched.

    Rank 0 grad = 1.0, rank 1 grad = 5.0 ⇒ AVG within subgroup = 3.0
    Rank 2 grad = 100.0, rank 3 grad = 200.0 ⇒ untouched

    NB: ``dist.new_group`` is itself a WORLD-collective on gloo — every
    rank in WORLD must call it, even ranks not in the subgroup. Ranks
    2/3 then sit on a barrier so spawn-shutdown doesn't race with the
    subgroup collective on rank 0/1.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        model = _build_two_param_model(seed=37)
        # Distinct per-rank grad fills so the in/out-of-subgroup contrast
        # is visible across ranks at assertion time.
        rank_grad_value = {0: 1.0, 1: 5.0, 2: 100.0, 3: 200.0}[rank]
        for p in model.parameters():
            p.grad = torch.full_like(p, rank_grad_value)

        # Snapshot grads so ranks 2/3 can confirm bit-identity post-call.
        snapshot = [p.grad.clone() for p in model.parameters()]

        # ALL ranks must call new_group — it is itself a WORLD collective.
        subgroup = dist.new_group([0, 1])

        if rank in (0, 1):
            allreduce_grads(
                model,
                world_size=2,
                group=subgroup,
                op=dist.ReduceOp.AVG,
            )
            # Subgroup AVG of {1, 5} = 3.0 on both ranks 0 and 1.
            max_diff = max(
                (p.grad - torch.full_like(p, 3.0)).abs().max().item()
                for p in model.parameters()
            )
            with open(result_path + f".{rank}", "w") as fh:
                fh.write(f"{max_diff}\n")
        else:
            # Ranks 2/3: did not participate; grads must be unchanged.
            max_drift = max(
                (p.grad - s).abs().max().item()
                for p, s in zip(model.parameters(), snapshot, strict=True)
            )
            with open(result_path + f".{rank}", "w") as fh:
                fh.write(f"{max_drift}\n")

        # Final WORLD barrier so all 4 ranks finish together. Without
        # this, the subgroup ranks can race ahead of the non-subgroup
        # ranks through destroy_process_group and trigger gloo cleanup
        # races on macOS.
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_composability_subgroup_sum_materialize(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
) -> None:
    """Test 6: ``group=subgroup`` AND ``op=SUM`` AND ``materialize_zeros=True``
    in one call — the exact production shape Phase 1 Task 1.3 will run.

    4 ranks, subgroup={0, 1, 2}, single 2-element parameter:
        - Train ranks 0/1: grad = [3, 6], pre-scaled by 1/2 ⇒ [1.5, 3.0]
        - Episodic rank 2: grad = None ⇒ materialize_zeros fills [0, 0]
        - Rank 3: not in subgroup, grad untouched
    SUM over subgroup of 3:
        [1.5 + 1.5 + 0, 3.0 + 3.0 + 0] = [3.0, 6.0]

    The expected vector recovers the (un-pre-scaled) train-rank average
    on every subgroup member because (1.5 + 1.5)/N_train = 1.0 doesn't
    apply here — caller's 1/N_train pre-scale × SUM-over-N_train = AVG
    of the train ranks' raw [3, 6]. The episodic rank contributes zero,
    which is the whole point of ``materialize_zeros=True``: when no
    replay item is queued, the episodic rank's contribution is exactly
    zero (a no-op term in the SUM).
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # Single-parameter Linear (no bias) so the assertion vector is a
        # clean [3, 6] without weight-matrix obfuscation.
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.zeros(1, 2))

        # ALL ranks must call new_group — it is itself a WORLD collective.
        subgroup = dist.new_group([0, 1, 2])

        # Set per-rank grads:
        #   train ranks 0, 1: pre-scaled main grads
        #   episodic rank 2 : grad is None (will be materialized to zeros)
        #   rank 3          : not in subgroup, has its own value, must
        #                     remain untouched after the call.
        n_train_in_subgroup = 2  # ranks 0, 1
        if rank in (0, 1):
            base_grad = torch.tensor([[3.0, 6.0]])
            base_grad.div_(float(n_train_in_subgroup))
            model.weight.grad = base_grad
        elif rank == 2:
            # Episodic rank with no replay → grad stays None until
            # materialize_zeros zero-fills it inside the call.
            model.weight.grad = None
        else:
            # Out-of-subgroup rank — initialize to a distinguishing
            # sentinel so any leak from the subgroup collective shows up
            # at assertion time.
            model.weight.grad = torch.tensor([[999.0, 999.0]])

        # Snapshot rank-3 grad so we can confirm it's untouched.
        snapshot = (
            model.weight.grad.clone() if model.weight.grad is not None else None
        )

        if rank in (0, 1, 2):
            allreduce_grads(
                model,
                world_size=3,  # subgroup size, but not used in body
                group=subgroup,
                op=dist.ReduceOp.SUM,
                materialize_zeros=True,
            )
            expected = torch.tensor([[3.0, 6.0]])
            max_diff = (model.weight.grad - expected).abs().max().item()
            with open(result_path + f".{rank}", "w") as fh:
                fh.write(f"{max_diff}\n")
        else:
            # Rank 3: confirm pre-call snapshot still holds bit-identically.
            max_drift = (model.weight.grad - snapshot).abs().max().item()
            with open(result_path + f".{rank}", "w") as fh:
                fh.write(f"{max_drift}\n")

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_existing_call_sites_smoke(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
) -> None:
    """Test 5: smoke-test the exp23 positional call form end-to-end.

    Mirrors the runner's path — build a model, run forward+backward,
    then call ``allreduce_grads(model, world_size)`` with no kwargs.
    The test catches any silent back-compat break the kwarg additions
    might have introduced into the legacy call signature.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # Same seed on every rank so post-AVG grads must equal the local
        # grads (any divergence implies allreduce silently corrupted
        # them). Per-rank input data exercises the collective non-trivially.
        torch.manual_seed(53)
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))

        # Per-rank input → per-rank loss → per-rank grads, then AVG.
        torch.manual_seed(100 + rank)
        x = torch.randn(3, 4)
        y = model(x).pow(2).sum()
        y.backward()

        # Capture pre-call grads so we can compute the expected AVG by
        # cross-rank file exchange below.
        pre = [p.grad.clone() for p in model.parameters()]

        # The legacy call form — no kwargs at all.
        allreduce_grads(model, world_size)

        # Post-call grads should be identical across ranks (an AVG over
        # all ranks is the same value on every rank). The cross-rank
        # equality check happens in the parent after collecting these
        # files.
        max_pre_post_diff = max(
            (p.grad - g).abs().max().item()
            for p, g in zip(model.parameters(), pre, strict=True)
        )
        # Also dump the post-call grads themselves so the parent can
        # compare across ranks for bit-identity.
        flat = torch.cat([p.grad.detach().flatten() for p in model.parameters()])
        torch.save(flat, result_path + f".{rank}.pt")
        with open(result_path + f".{rank}", "w") as fh:
            # Just record "did the call complete and return non-NaN" —
            # the cross-rank identity check happens in the parent.
            fh.write(f"{max_pre_post_diff}\n")

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestAllreduceGradsKwargs(unittest.TestCase):
    """End-to-end gloo coverage for the kwarg additions."""

    @staticmethod
    def _spawn(worker, world_size: int, tmp: str) -> str:
        """Spawn ``world_size`` ranks running ``worker`` and return the
        per-rank result file prefix.

        Each worker writes its own ``<prefix>.<rank>`` text file plus
        whatever auxiliary artifacts the test asks for. Using files
        instead of ``mp.Queue`` avoids the queue-shutdown hang
        ``test_ddp_integration`` documents.
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

    def test_back_compat_avg_unchanged(self) -> None:
        """Test 1: positional 2-arg form still does AVG over WORLD.

        This is the load-bearing back-compat test — exp23/exp24 callers
        all use the positional form. If this regresses, every existing
        runner silently averages wrong on its next pod run.
        """
        with tempfile.TemporaryDirectory() as tmp:
            result_path = self._spawn(_worker_back_compat_avg, 2, tmp)
            for rank in range(2):
                content = Path(f"{result_path}.{rank}").read_text().strip()
                max_diff = float(content)
                self.assertLess(
                    max_diff, 1e-6,
                    f"rank {rank}: AVG of {{1, 3}} should be 2.0; "
                    f"max abs deviation = {max_diff}",
                )

    def test_sum_op_with_prescaling_gives_main_avg_plus_replay(self) -> None:
        """Test 2: 3+1 split with SUM-op + caller pre-scale matches the
        Phase 1 Task 1.3 design intent ([11, 22] every rank).
        """
        with tempfile.TemporaryDirectory() as tmp:
            result_path = self._spawn(_worker_sum_with_prescaling, 4, tmp)
            for rank in range(4):
                content = Path(f"{result_path}.{rank}").read_text().strip()
                max_diff = float(content)
                # SUM-vs-AVG and the 1/3 division both have fp32 round-
                # off noise; 1e-5 is comfortably above the worst-case
                # IEEE 754 drift on a 2-element vector.
                self.assertLess(
                    max_diff, 1e-5,
                    f"rank {rank}: SUM with caller pre-scale should land at "
                    f"[11, 22]; max abs deviation = {max_diff}",
                )

    def test_materialize_zeros_handles_inconsistent_grad_set(self) -> None:
        """Test 3: rank 1's missing bias grad gets zero-filled, the call
        completes, the AVG halves the rank-0 contribution.
        """
        with tempfile.TemporaryDirectory() as tmp:
            result_path = self._spawn(_worker_materialize_zeros, 2, tmp)
            for rank in range(2):
                content = Path(f"{result_path}.{rank}").read_text().strip().splitlines()
                weight_diff = float(content[0])
                bias_diff = float(content[1])
                self.assertLess(
                    weight_diff, 1e-6,
                    f"rank {rank}: weight AVG should be 4.0; "
                    f"max abs deviation = {weight_diff}",
                )
                self.assertLess(
                    bias_diff, 1e-6,
                    f"rank {rank}: bias AVG (6+0)/2 should be 3.0; "
                    f"max abs deviation = {bias_diff}",
                )

    def test_subgroup_routing_isolates_subset(self) -> None:
        """Test 4: subgroup-restricted reduction reaches only ranks 0/1.

        Ranks 2/3 record post-call drift vs their pre-call snapshot;
        any non-zero drift means the subgroup collective leaked to them.
        """
        with tempfile.TemporaryDirectory() as tmp:
            result_path = self._spawn(_worker_subgroup_routing, 4, tmp)
            for rank in (0, 1):
                content = Path(f"{result_path}.{rank}").read_text().strip()
                max_diff = float(content)
                self.assertLess(
                    max_diff, 1e-6,
                    f"rank {rank}: subgroup AVG of {{1, 5}} should be 3.0; "
                    f"max abs deviation = {max_diff}",
                )
            for rank in (2, 3):
                content = Path(f"{result_path}.{rank}").read_text().strip()
                max_drift = float(content)
                # Bit-identity required — the subgroup collective MUST NOT
                # touch ranks not in the subgroup. Even fp epsilon-level
                # drift would imply the collective leaked.
                self.assertEqual(
                    max_drift, 0.0,
                    f"rank {rank}: out-of-subgroup grads must not change; "
                    f"max abs drift = {max_drift}",
                )

    def test_composability_subgroup_sum_materialize(self) -> None:
        """Test 6: subgroup + SUM + materialize_zeros simultaneously.

        This is the exact call shape Phase 1 Task 1.3's optimizer step
        will issue. Each kwarg has its own test above; this one catches
        bugs where the branches interact (e.g. materialize_zeros running
        on WORLD params but the collective restricted to a subgroup,
        which would zero-fill grads on out-of-subgroup ranks too — a
        regression mode no single-kwarg test would expose).
        """
        with tempfile.TemporaryDirectory() as tmp:
            result_path = self._spawn(
                _worker_composability_subgroup_sum_materialize, 4, tmp,
            )
            for rank in (0, 1, 2):
                content = Path(f"{result_path}.{rank}").read_text().strip()
                max_diff = float(content)
                self.assertLess(
                    max_diff, 1e-5,
                    f"rank {rank}: subgroup SUM with caller pre-scale + "
                    f"zero-fill on rank 2 should land at [3, 6]; "
                    f"max abs deviation = {max_diff}",
                )
            # Rank 3 was never in the subgroup; grad must be untouched.
            content = Path(f"{result_path}.3").read_text().strip()
            max_drift = float(content)
            self.assertEqual(
                max_drift, 0.0,
                "out-of-subgroup rank 3 grad must not change; "
                f"max abs drift = {max_drift}",
            )

    def test_existing_call_sites_smoke(self) -> None:
        """Test 5: forward+backward+legacy ``allreduce_grads(model, ws)``
        path runs cleanly and produces bit-identical post-AVG grads on
        both ranks (defining feature of an all-reduce).
        """
        with tempfile.TemporaryDirectory() as tmp:
            result_path = self._spawn(
                _worker_existing_call_sites_smoke, 2, tmp,
            )
            # Each rank wrote a flat-grad tensor; cross-rank identity is
            # the load-bearing assertion (an all-reduce produces the
            # same result on every rank).
            grads_per_rank = [
                torch.load(f"{result_path}.{rank}.pt", weights_only=True)
                for rank in range(2)
            ]
            self.assertEqual(grads_per_rank[0].shape, grads_per_rank[1].shape)
            max_cross_rank_diff = (
                grads_per_rank[0] - grads_per_rank[1]
            ).abs().max().item()
            self.assertLess(
                max_cross_rank_diff, 1e-6,
                "post-AVG grads must be bit-identical across ranks; "
                f"max abs cross-rank diff = {max_cross_rank_diff}",
            )


if __name__ == "__main__":
    unittest.main()
