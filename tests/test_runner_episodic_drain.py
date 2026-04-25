"""Tests for the Phase 1 Task 1.5 episodic-rank consumer drain.

Phase 1 Task 1.5 of the memory-aware optimizer plan: when the runner is
launched with ``episodic_enabled=True`` and ``world_size = N``, rank
``N - 1`` is the episodic rank. At init it constructs an
``EpisodicCache`` and ``ShmRing.attach``-es to each train rank's
``episodic_write_ring_rank{R}`` ring (R in 0..N-2). On every train step
that rank drains its attached rings into the cache before issuing the
shared SUM all-reduce that keeps the 3+1 collective in lockstep.

Tests in this file:

1. ``test_runner_creates_cache_on_episodic_rank`` — when
   ``episodic_enabled=True`` and the helper is called from the episodic
   rank, a real ``EpisodicCache`` materializes with the config-derived
   kwargs (capacity, span_length, key_rep_dim).
2. ``test_runner_does_not_create_cache_on_train_rank`` — when
   ``episodic_enabled=True`` but the rank is a train rank, no cache is
   built (the helper returns ``None`` for the cache slot).
3. ``test_runner_does_not_create_cache_when_episodic_disabled`` —
   back-compat: ``episodic_enabled=False`` produces no cache regardless
   of rank.
4. ``test_episodic_rank_attaches_to_all_train_rank_write_rings`` — the
   attach loop pairs with the contract names; the count is
   ``world_size - 1`` and the names match
   ``episodic_write_ring_rank{R}`` exactly.
5. ``test_episodic_rank_drains_write_ring_into_cache`` — multi-process
   ``mp.spawn`` smoke (gloo, world_size=4): faux train ranks create
   their write rings and push synthetic payloads via ``ShmRing.try_write``
   directly (no full Task 1.4 producer logic — this test pins the
   consumer's drain). The episodic rank attaches, runs through
   ``_run_train_step`` once, and asserts ``len(cache)`` matches the
   number of payloads pushed.
6. ``test_episodic_rank_heartbeat_advances`` — after N
   ``_run_train_step`` calls on the episodic-rank branch the heartbeat
   counter equals N.

Tests 1-3 use the importlib pattern from ``test_cd_config_threading``
to load ``runner_fast_path.py`` without monkey-patching ``sys.path``.

Test 5 follows the ``mp.spawn + gloo + free port`` pattern from
``test_distributed_allreduce_grads.py`` because mp.spawn rendezvous on
local CPU is racy on macOS (see ``feedback_mp_spawn_flake.md``).
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import socket
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp

from chaoscontrol.episodic.ipc import ShmRing
from chaoscontrol.episodic.payload_dtypes import make_write_payload_dtype


# ---------------------------------------------------------------------------
# Module loader (matches tests/test_cd_config_threading.py)
# ---------------------------------------------------------------------------


def _load_runner_module():
    path = (
        Path(__file__).resolve().parent.parent
        / "experiments" / "23_fast_path" / "runner_fast_path.py"
    )
    spec = importlib.util.spec_from_file_location("runner_fast_path", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helper-call tests (1-4): no real distributed bring-up needed
# ---------------------------------------------------------------------------


def _unique_ring_name(rank: int) -> str:
    """Per-test unique name for an in-test ring on this PID.

    POSIX shm names on darwin cap at ~30 chars (PSHMNAMLEN). The
    ``_attach_episodic_consumer`` helper itself uses the contract's
    ``episodic_write_ring_rank{R}`` pattern; tests that need to first
    create rings before calling attach use unique names per their own
    PID + uuid suffix and pass them through. The runner's contract name
    is the only one that matters at the integration layer; tests that
    only exercise the helper-creation path supply their own names.
    """
    return f"cc_t15_{os.getpid() & 0xFFFF:04x}_{uuid.uuid4().hex[:6]}_{rank}"


class TestEpisodicConsumerHelper(unittest.TestCase):
    """Direct calls into the consumer-init helper.

    Tests 1-4 exercise ``_attach_episodic_consumer`` (or whatever name
    the runner exposes) without spinning a real ``ProcessGroup`` — the
    helper's contract is to produce a cache + ring list keyed off the
    ``(episodic_enabled, is_episodic_rank, world_size)`` tuple, with
    config-derived dimensions.
    """

    def setUp(self) -> None:
        # Belt-and-suspenders: clean any dangling shm with the contract
        # name from a previous failed run. World size in this file is
        # bounded above by 4; clean a wider range to be safe.
        _cleanup_contract_shms(world_size=8)

    def tearDown(self) -> None:
        _cleanup_contract_shms(world_size=8)

    def _create_train_write_rings(self, world_size: int, *, span_length: int, key_rep_dim: int):
        """Create the N-1 producer rings that the helper expects to attach
        to. The test owns ``close_and_unlink`` of each (Task 1.4
        ownership). Returns the list of created rings so the test can
        unlink them in tearDown.
        """
        dtype = make_write_payload_dtype(
            span_length=span_length, key_rep_dim=key_rep_dim,
        )
        created = []
        for r in range(world_size - 1):
            ring = ShmRing.create(
                name=f"episodic_write_ring_rank{r}",
                slot_shape=(),
                dtype=dtype,
                capacity=8,
            )
            created.append(ring)
        return created

    def _cleanup_rings(self, rings):
        for r in rings:
            try:
                r.close_and_unlink()
            except Exception:
                pass

    def test_runner_creates_cache_on_episodic_rank(self):
        """Test 1: the helper produces a real EpisodicCache on the
        episodic rank with config-derived kwargs.
        """
        mod = _load_runner_module()
        from chaoscontrol.optim.episodic_cache import EpisodicCache

        world_size = 4
        span_length = 4
        key_rep_dim = 16
        capacity = 32
        # Producer rings must exist BEFORE the helper attaches.
        train_rings = self._create_train_write_rings(
            world_size, span_length=span_length, key_rep_dim=key_rep_dim,
        )
        try:
            consumer = mod._attach_episodic_consumer(
                episodic_enabled=True,
                is_episodic_rank=True,
                world_size=world_size,
                config={
                    "episodic_capacity": capacity,
                    "episodic_span_length": span_length,
                    "episodic_key_rep_dim": key_rep_dim,
                    "episodic_grace_steps": 50,
                    "episodic_utility_ema_decay": 0.95,
                    "episodic_write_ring_capacity": 8,
                },
                model_dim=key_rep_dim,
                all_group=None,  # tests 1-4 do not exercise the barrier
            )
            self.assertIsNotNone(consumer.cache)
            self.assertIsInstance(consumer.cache, EpisodicCache)
            self.assertEqual(consumer.cache.capacity, capacity)
            self.assertEqual(consumer.cache.span_length, span_length)
            self.assertEqual(consumer.cache.key_rep_dim, key_rep_dim)
            self.assertEqual(consumer.cache.grace_steps, 50)
            self.assertAlmostEqual(consumer.cache.utility_ema_decay, 0.95)
        finally:
            self._cleanup_rings(train_rings)

    def test_runner_does_not_create_cache_on_train_rank(self):
        """Test 2: when ``is_episodic_rank=False`` the helper produces a
        no-op consumer — no cache, no attached rings.
        """
        mod = _load_runner_module()

        consumer = mod._attach_episodic_consumer(
            episodic_enabled=True,
            is_episodic_rank=False,
            world_size=4,
            config={"episodic_capacity": 16, "episodic_span_length": 4},
            model_dim=16,
            all_group=None,
        )
        self.assertIsNone(consumer.cache)
        self.assertEqual(consumer.write_rings, [])

    def test_runner_does_not_create_cache_when_episodic_disabled(self):
        """Test 3: ``episodic_enabled=False`` is the back-compat path.

        Bit-identical to pre-Task-1.5: the helper returns a no-op
        consumer regardless of rank.
        """
        mod = _load_runner_module()

        for is_epr in (False, True):
            consumer = mod._attach_episodic_consumer(
                episodic_enabled=False,
                is_episodic_rank=is_epr,
                world_size=4,
                config={},
                model_dim=16,
                all_group=None,
            )
            self.assertIsNone(consumer.cache)
            self.assertEqual(consumer.write_rings, [])

    def test_episodic_rank_attaches_to_all_train_rank_write_rings(self):
        """Test 4: the helper attaches once per train rank, names match
        the contract pattern, and the resulting list has length
        ``world_size - 1``.
        """
        mod = _load_runner_module()
        world_size = 4
        span_length = 4
        key_rep_dim = 8
        train_rings = self._create_train_write_rings(
            world_size, span_length=span_length, key_rep_dim=key_rep_dim,
        )
        try:
            consumer = mod._attach_episodic_consumer(
                episodic_enabled=True,
                is_episodic_rank=True,
                world_size=world_size,
                config={
                    "episodic_capacity": 16,
                    "episodic_span_length": span_length,
                    "episodic_key_rep_dim": key_rep_dim,
                    "episodic_write_ring_capacity": 8,
                },
                model_dim=key_rep_dim,
                all_group=None,
            )
            self.assertEqual(len(consumer.write_rings), world_size - 1)
            # Round-trip a payload through one of the per-rank rings to
            # confirm the attach is bound to the producer's shm — the
            # producer pushes via train_rings[r], the consumer reads via
            # consumer.write_rings[r].
            dtype = make_write_payload_dtype(
                span_length=span_length, key_rep_dim=key_rep_dim,
            )
            for r in range(world_size - 1):
                slot = np.zeros((), dtype=dtype)
                slot["key_fp"] = 100 + r
                train_rings[r].try_write(slot)
                got = consumer.write_rings[r].try_read()
                self.assertIsNotNone(got)
                self.assertEqual(int(got["key_fp"]), 100 + r)
            # Heartbeat counter starts at 0.
            self.assertEqual(consumer.heartbeat[0], 0)
        finally:
            self._cleanup_rings(train_rings)

    def test_runner_helper_is_no_op_on_non_episodic_world_size_1(self):
        """Defensive: ``episodic_enabled=True`` is incompatible with
        ``world_size=1`` (the runner's own guard rejects it earlier),
        but if the helper is asked anyway it should not crash on a
        train-rank role at world_size=1; it returns the no-op consumer.
        """
        mod = _load_runner_module()
        consumer = mod._attach_episodic_consumer(
            episodic_enabled=False,
            is_episodic_rank=False,
            world_size=1,
            config={},
            model_dim=8,
            all_group=None,
        )
        self.assertIsNone(consumer.cache)
        self.assertEqual(consumer.write_rings, [])


# ---------------------------------------------------------------------------
# Multi-process drain test (test 5) and heartbeat test (test 6)
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Same idiom as ``test_distributed_allreduce_grads._pick_free_port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _cleanup_contract_shms(world_size: int) -> None:
    """Proactively unlink any leftover ``episodic_write_ring_rank{R}`` shm.

    The tests in this file all use the contract's fixed name pattern
    (Task 1.5's helper attaches by exact name; the test producers
    create by exact name). If a previous test failed mid-run and left
    a shm dangling, the next ``ShmRing.create(create=True)`` raises
    ``FileExistsError``. This belt-and-suspenders cleanup walks both
    the slot shm and its ``_c`` counter shm and silently swallows
    ``FileNotFoundError`` (the common case — nothing to clean).
    """
    from multiprocessing.shared_memory import SharedMemory

    for r in range(world_size):
        for suffix in ("", "_c"):
            name = f"episodic_write_ring_rank{r}{suffix}"
            try:
                shm = SharedMemory(name=name, create=False)
            except FileNotFoundError:
                continue
            except Exception:
                # Defensive: any other access error means the shm is in
                # a state we can't act on; leave it for the OS to GC.
                continue
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def _drain_test_worker(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
    span_length: int,
    key_rep_dim: int,
    payloads_per_train_rank: int,
) -> None:
    """4-rank gloo: 3 train + 1 episodic, exercise the consumer drain.

    Train ranks 0/1/2 act as faux producers — they don't run
    forward/backward; they just create their per-rank write ring (the
    Task 1.4 producer side), push ``payloads_per_train_rank`` synthetic
    payloads, then issue the shared barrier and the SUM all-reduce so
    the episodic rank's collective doesn't hang.

    Episodic rank 3 runs through ``_run_train_step`` once with
    ``is_episodic_rank=True`` and the cache + attached rings the runner
    helper builds. After the call, it dumps ``len(cache)`` and
    ``heartbeat[0]`` to a file for the parent to assert.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # Load the runner module the same way the helper-tests do.
        spec = importlib.util.spec_from_file_location(
            "runner_fast_path",
            str(
                Path(__file__).resolve().parent.parent
                / "experiments" / "23_fast_path" / "runner_fast_path.py"
            ),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        is_episodic_rank = (rank == world_size - 1)
        # All ranks must call new_group — it is itself a WORLD collective.
        all_group = dist.new_group(list(range(world_size)))

        write_payload_dtype = make_write_payload_dtype(
            span_length=span_length, key_rep_dim=key_rep_dim,
        )
        # A tiny model so allreduce_grads has something to operate on.
        # Using torch.nn.Linear keeps the param shapes generic and
        # exercises the materialize_zeros path on the episodic rank.
        model = torch.nn.Linear(2, 2, bias=False)

        if not is_episodic_rank:
            # Faux producer: create our write ring, push synthetic
            # payloads, then participate in the rendezvous + collective.
            ring = ShmRing.create(
                name=f"episodic_write_ring_rank{rank}",
                slot_shape=(),
                dtype=write_payload_dtype,
                capacity=16,
            )
            try:
                for i in range(payloads_per_train_rank):
                    slot = np.zeros((), dtype=write_payload_dtype)
                    # Distinct fingerprints per (rank, payload) so the
                    # episodic rank's cache.append does not collide
                    # (duplicate fingerprints replace the older slot via
                    # the hash index — a separate test pins that
                    # behavior; here we want every payload to add a
                    # fresh entry).
                    slot["key_fp"] = 1000 * (rank + 1) + i
                    slot["key_rep"] = np.full(
                        (key_rep_dim,), float(rank + 1), dtype=np.float32,
                    )
                    slot["value_tok_ids"] = np.arange(
                        i * span_length, i * span_length + span_length,
                        dtype=np.int64,
                    )
                    slot["value_anchor_id"] = i * span_length
                    ring.try_write(slot)

                # Producer-side rendezvous: barrier AFTER create. Both
                # sides converge on this barrier (Task 1.4 + Task 1.5
                # contract).
                dist.barrier(group=all_group)

                # Train rank also issues the all-reduce so the episodic
                # rank's SUM collective inside _run_train_step doesn't
                # deadlock. Train ranks contribute zero grads here — the
                # collective shape mirrors the runner's pre-scale path.
                for p in model.parameters():
                    p.grad = torch.zeros_like(p)
                from chaoscontrol.distributed import allreduce_grads
                allreduce_grads(
                    model,
                    world_size,
                    group=all_group,
                    op=dist.ReduceOp.SUM,
                    materialize_zeros=True,
                )

                # Cleanup is the producer's responsibility (Task 1.4
                # ownership). Wait until all ranks finish so the
                # consumer doesn't trip over an unlinked shm.
                dist.barrier()
            finally:
                ring.close_and_unlink()

            with open(result_path + f".{rank}", "w") as fh:
                fh.write("ok\n")
        else:
            # Episodic rank: build cache + attach rings via the helper.
            consumer = mod._attach_episodic_consumer(
                episodic_enabled=True,
                is_episodic_rank=True,
                world_size=world_size,
                config={
                    "episodic_capacity": 64,
                    "episodic_span_length": span_length,
                    "episodic_key_rep_dim": key_rep_dim,
                    "episodic_grace_steps": 100,
                    "episodic_utility_ema_decay": 0.99,
                    "episodic_write_ring_capacity": 16,
                },
                model_dim=key_rep_dim,
                all_group=all_group,
            )

            # The helper itself issues the post-attach barrier.
            # Run the actual episodic-rank branch of _run_train_step so
            # the test exercises the production drain code, not a
            # separate helper.
            inputs = torch.zeros((1, 1), dtype=torch.long)
            targets = torch.zeros((1, 1), dtype=torch.long)
            loss = mod._run_train_step(
                model=model,
                inputs=inputs,
                targets=targets,
                chunk_size=1,
                precision="fp32",
                ddp_active=True,
                world_size=world_size,
                lm_head_backward_mode="fused",
                grad_allreduce_mode="bulk",
                is_episodic_rank=True,
                all_group=all_group,
                episodic_cache=consumer.cache,
                episodic_write_rings=consumer.write_rings,
                episodic_heartbeat=consumer.heartbeat,
                current_step=0,
                embedding_version=0,
            )

            # Wait for producers to finish unlinking.
            dist.barrier()

            # Close (consumer side, no unlink) — Task 1.5 ownership.
            for r in consumer.write_rings:
                r.close()

            with open(result_path + f".{rank}", "w") as fh:
                fh.write(
                    f"cache_len={len(consumer.cache)}\n"
                    f"heartbeat={consumer.heartbeat[0]}\n"
                    f"loss_dtype={loss.dtype}\n"
                )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _heartbeat_test_worker(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
    n_steps: int,
) -> None:
    """4-rank gloo: episodic rank runs N steps, asserts heartbeat == N.

    Same dance as the drain worker but with no payloads pushed; we just
    care that ``consumer.heartbeat[0]`` advances once per step.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        spec = importlib.util.spec_from_file_location(
            "runner_fast_path",
            str(
                Path(__file__).resolve().parent.parent
                / "experiments" / "23_fast_path" / "runner_fast_path.py"
            ),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        is_episodic_rank = (rank == world_size - 1)
        all_group = dist.new_group(list(range(world_size)))

        span_length = 2
        key_rep_dim = 4
        write_payload_dtype = make_write_payload_dtype(
            span_length=span_length, key_rep_dim=key_rep_dim,
        )
        model = torch.nn.Linear(2, 2, bias=False)

        if not is_episodic_rank:
            ring = ShmRing.create(
                name=f"episodic_write_ring_rank{rank}",
                slot_shape=(),
                dtype=write_payload_dtype,
                capacity=8,
            )
            try:
                dist.barrier(group=all_group)
                from chaoscontrol.distributed import allreduce_grads
                # N step-shaped collectives so the episodic rank's
                # in-step all-reduce can find a peer each step.
                for _ in range(n_steps):
                    for p in model.parameters():
                        p.grad = torch.zeros_like(p)
                    allreduce_grads(
                        model,
                        world_size,
                        group=all_group,
                        op=dist.ReduceOp.SUM,
                        materialize_zeros=True,
                    )
                dist.barrier()
            finally:
                ring.close_and_unlink()
            with open(result_path + f".{rank}", "w") as fh:
                fh.write("ok\n")
        else:
            consumer = mod._attach_episodic_consumer(
                episodic_enabled=True,
                is_episodic_rank=True,
                world_size=world_size,
                config={
                    "episodic_capacity": 16,
                    "episodic_span_length": span_length,
                    "episodic_key_rep_dim": key_rep_dim,
                    "episodic_write_ring_capacity": 8,
                },
                model_dim=key_rep_dim,
                all_group=all_group,
            )

            inputs = torch.zeros((1, 1), dtype=torch.long)
            targets = torch.zeros((1, 1), dtype=torch.long)
            for step in range(n_steps):
                mod._run_train_step(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    chunk_size=1,
                    precision="fp32",
                    ddp_active=True,
                    world_size=world_size,
                    lm_head_backward_mode="fused",
                    grad_allreduce_mode="bulk",
                    is_episodic_rank=True,
                    all_group=all_group,
                    episodic_cache=consumer.cache,
                    episodic_write_rings=consumer.write_rings,
                    episodic_heartbeat=consumer.heartbeat,
                    current_step=step,
                    embedding_version=0,
                )

            dist.barrier()
            for r in consumer.write_rings:
                r.close()
            with open(result_path + f".{rank}", "w") as fh:
                fh.write(f"heartbeat={consumer.heartbeat[0]}\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestEpisodicRankDrainSpawn(unittest.TestCase):
    """4-rank gloo coverage for the in-step drain + heartbeat advance."""

    def setUp(self) -> None:
        _cleanup_contract_shms(world_size=8)

    def tearDown(self) -> None:
        _cleanup_contract_shms(world_size=8)

    @staticmethod
    def _spawn(worker, world_size: int, tmp: str, *worker_args) -> str:
        result_path = os.path.join(tmp, "result")
        master_port = _pick_free_port()
        torch_mp.spawn(
            worker,
            args=(world_size, master_port, result_path, *worker_args),
            nprocs=world_size,
            join=True,
        )
        return result_path

    def test_episodic_rank_drains_write_ring_into_cache(self):
        """Test 5: payloads pushed to N-1 train-rank rings end up in
        the cache after one episodic step.

        With world_size=4, 3 producers each push 5 payloads → cache
        length should be 15 after the drain. Distinct fingerprints per
        producer guarantee no hash-index collisions during the drain.
        """
        world_size = 4
        span_length = 4
        key_rep_dim = 8
        payloads_per_train_rank = 5
        with tempfile.TemporaryDirectory() as tmp:
            result_path = self._spawn(
                _drain_test_worker,
                world_size,
                tmp,
                span_length,
                key_rep_dim,
                payloads_per_train_rank,
            )
            # Episodic-rank result is on rank world_size - 1.
            content = (
                Path(f"{result_path}.{world_size - 1}").read_text().strip()
            )
            kv = dict(line.split("=") for line in content.splitlines())
            self.assertEqual(
                int(kv["cache_len"]),
                payloads_per_train_rank * (world_size - 1),
                f"cache should hold every payload pushed by every train "
                f"rank; got {kv['cache_len']} (expected "
                f"{payloads_per_train_rank * (world_size - 1)})",
            )
            self.assertEqual(int(kv["heartbeat"]), 1)

    def test_episodic_rank_heartbeat_advances(self):
        """Test 6: N _run_train_step calls advance the heartbeat by N."""
        world_size = 4
        n_steps = 5
        with tempfile.TemporaryDirectory() as tmp:
            result_path = self._spawn(
                _heartbeat_test_worker, world_size, tmp, n_steps,
            )
            content = (
                Path(f"{result_path}.{world_size - 1}").read_text().strip()
            )
            self.assertEqual(content, f"heartbeat={n_steps}")


if __name__ == "__main__":
    unittest.main()
