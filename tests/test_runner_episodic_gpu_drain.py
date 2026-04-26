"""Tests for the episodic-rank GPU drain path (Perf Pass C.3 + C.4).

Replaces the POSIX-shm Phase 1 Task 1.5 consumer tests
(``test_runner_episodic_drain.py`` pre-Pass-C). The new path receives
``[K_max, slot_dim]`` fp32 tensors via ``dist.gather``, filters by
``valid_mask``, and routes each surviving row to (a) ``cache.append``
for the write side and (b) ``controller_query_queue`` for Phase 2.

Tests:

  1. ``test_attach_episodic_consumer_creates_cache_on_episodic_rank`` —
     helper builds an ``EpisodicCache`` with config-derived dimensions.
  2. ``test_attach_episodic_consumer_no_op_on_train_rank`` —
     train-rank invocation produces an empty consumer (no cache, empty
     queue).
  3. ``test_attach_episodic_consumer_no_op_when_disabled`` — back-compat:
     ``episodic_enabled=False`` is a skip regardless of rank.
  4. ``test_consumer_state_no_longer_has_write_rings`` — Pass C dropped
     the ``write_rings`` field; pin the rename so any code reaching for
     it surfaces immediately.
  5. ``test_drain_filters_invalid_rows_and_appends_valid_ones`` —
     direct call to ``_drain_episodic_payloads_gpu`` with a hand-built
     gather_list verifies that valid_mask=1 rows fill the cache and
     valid_mask=0 rows are skipped.
  6. ``test_drain_pushes_to_controller_query_queue`` — same drain
     populates the ``controller_query_queue`` Python list with one
     entry per valid slot (Phase 2 prerequisite).
  7. ``test_4rank_gloo_end_to_end`` (mp.spawn) — 3 train + 1 episodic
     gloo workers run a single ``_run_train_step`` each, train ranks
     emit packed slot tensors, the episodic rank drains via
     ``dist.gather``. Asserts cache_len matches the number of valid
     rows produced and the controller_query_queue has the same length.

Tests 1-6 run single-process. Test 7 uses ``mp.spawn`` + gloo because
that's the only way to exercise the ``dist.gather`` collective without
real CUDA — and it's the load-bearing test for the IPC end-to-end.
"""
from __future__ import annotations

import importlib.util
import os
import socket
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
import torch.nn as nn

from chaoscontrol.episodic.gpu_slot import (
    make_slot_tensor,
    pack_payload,
    slot_dim,
)


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
# Tests 1-4: helper invocations and consumer-state shape pins
# ---------------------------------------------------------------------------


class TestEpisodicConsumerHelper(unittest.TestCase):
    """Direct calls to ``_attach_episodic_consumer`` without a process group."""

    def test_attach_episodic_consumer_creates_cache_on_episodic_rank(self):
        mod = _load_runner_module()
        from chaoscontrol.optim.episodic_cache import EpisodicCache

        consumer = mod._attach_episodic_consumer(
            episodic_enabled=True,
            is_episodic_rank=True,
            world_size=4,
            config={
                "episodic_capacity": 32,
                "episodic_span_length": 4,
                "episodic_key_rep_dim": 16,
                "episodic_grace_steps": 50,
                "episodic_utility_ema_decay": 0.95,
            },
            model_dim=16,
            all_group=None,
        )
        self.assertIsNotNone(consumer.cache)
        self.assertIsInstance(consumer.cache, EpisodicCache)
        self.assertEqual(consumer.cache.capacity, 32)
        self.assertEqual(consumer.cache.span_length, 4)
        self.assertEqual(consumer.cache.key_rep_dim, 16)
        self.assertEqual(consumer.cache.grace_steps, 50)
        self.assertAlmostEqual(consumer.cache.utility_ema_decay, 0.95)
        self.assertEqual(consumer.heartbeat[0], 0)
        # Pass C: controller_query_queue starts empty.
        self.assertEqual(consumer.controller_query_queue, [])

    def test_attach_episodic_consumer_no_op_on_train_rank(self):
        """``is_episodic_rank=False`` produces an empty consumer."""
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
        self.assertEqual(consumer.controller_query_queue, [])

    def test_attach_episodic_consumer_no_op_when_disabled(self):
        """``episodic_enabled=False`` is a skip regardless of rank."""
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
            self.assertEqual(consumer.controller_query_queue, [])

    def test_consumer_state_no_longer_has_write_rings(self):
        """Pass C dropped the ``write_rings`` field. Pin the rename so
        any code reaching for the old attribute fails loudly.
        """
        mod = _load_runner_module()
        consumer = mod._attach_episodic_consumer(
            episodic_enabled=False,
            is_episodic_rank=False,
            world_size=2,
            config={},
            model_dim=4,
            all_group=None,
        )
        self.assertFalse(hasattr(consumer, "write_rings"))
        self.assertTrue(hasattr(consumer, "controller_query_queue"))
        self.assertTrue(hasattr(consumer, "cache"))
        self.assertTrue(hasattr(consumer, "heartbeat"))


# ---------------------------------------------------------------------------
# Tests 5-6: direct call to _drain_episodic_payloads_gpu
# ---------------------------------------------------------------------------


def _build_consumer_state(
    mod, *, span_length: int, key_rep_dim: int,
    controller_query_enabled: bool = True,
):
    """Helper: build an episodic-rank consumer state with a real cache.

    ``controller_query_enabled`` defaults to True for the existing
    test 6 path that asserts queue grows. Phase 1 production default is
    False (slow-OOM guard); see ``test_controller_query_queue_default_off``.
    """
    return mod._attach_episodic_consumer(
        episodic_enabled=True,
        is_episodic_rank=True,
        world_size=2,
        config={
            "episodic_capacity": 64,
            "episodic_span_length": span_length,
            "episodic_key_rep_dim": key_rep_dim,
            "episodic_grace_steps": 100,
            "episodic_utility_ema_decay": 0.99,
            "controller_query_enabled": controller_query_enabled,
        },
        model_dim=key_rep_dim,
        all_group=None,
    )


def test_controller_query_queue_default_off_no_growth() -> None:
    """Phase 1 production default: controller_query_enabled is False, so
    the drain still appends to ``EpisodicCache`` (write side) but does
    NOT push to ``controller_query_queue``. Without this gate, the queue
    retains GPU residual tensors unboundedly across a 600s run (~1.25 GB
    at world=8, D=256). Phase 2 flips the flag True at the same time it
    wires the consumer that drains the queue.
    """
    mod = _load_runner_module()
    span_length = 2
    key_rep_dim = 3
    k_max = 2
    consumer = _build_consumer_state(
        mod, span_length=span_length, key_rep_dim=key_rep_dim,
        controller_query_enabled=False,
    )
    # Hand-build a 1-rank gather_list with one valid row.
    t = make_slot_tensor(
        k_max=k_max, span_length=span_length, key_rep_dim=key_rep_dim,
        device=torch.device("cpu"),
    )
    pack_payload(
        t[0],
        valid_mask=1.0, pressure=0.5, key_fp=99,
        value_anchor_id=0,
        value_tok_ids=torch.zeros(span_length, dtype=torch.int64),
        key_rep=torch.zeros(key_rep_dim),
        residual=torch.ones(key_rep_dim, dtype=torch.float32),
        span_length=span_length, key_rep_dim=key_rep_dim,
    )
    mod._drain_episodic_payloads_gpu(
        consumer=consumer, gather_list=[t],
        span_length=span_length, key_rep_dim=key_rep_dim,
        k_max=k_max, current_step=1, embedding_version=0,
    )
    # Cache append happened (write side is unconditional).
    assert len(consumer.cache) == 1
    # Queue stayed empty (gated by controller_query_enabled=False).
    assert consumer.controller_query_queue == []


def test_drain_filters_invalid_rows_and_appends_valid_ones() -> None:
    """Hand-build a 4-rank gather_list with mixed valid/invalid rows;
    the drain appends only the valid ones to the cache.
    """
    mod = _load_runner_module()
    span_length = 3
    key_rep_dim = 4
    k_max = 4
    consumer = _build_consumer_state(
        mod, span_length=span_length, key_rep_dim=key_rep_dim,
    )
    # 4 ranks, k_max=4 each. Mark some rows valid with distinct fingerprints.
    gather_list: list[torch.Tensor] = []
    expected_valid = 0
    for r in range(4):
        t = make_slot_tensor(
            k_max=k_max,
            span_length=span_length,
            key_rep_dim=key_rep_dim,
            device=torch.device("cpu"),
        )
        for k in range(k_max):
            # 50% valid, 50% invalid (driven by k%2). Distinct
            # fingerprints per (rank, k) to avoid cache hash collisions.
            if k % 2 == 0:
                pack_payload(
                    t[k],
                    valid_mask=1.0,
                    pressure=0.5,
                    key_fp=1000 * (r + 1) + k,
                    value_anchor_id=k * span_length,
                    value_tok_ids=torch.tensor(
                        [k * span_length + i for i in range(span_length)],
                        dtype=torch.int64,
                    ),
                    key_rep=torch.full(
                        (key_rep_dim,), float(r + 1), dtype=torch.float32,
                    ),
                    residual=torch.full(
                        (key_rep_dim,), float(r + 1), dtype=torch.float32,
                    ),
                    span_length=span_length,
                    key_rep_dim=key_rep_dim,
                )
                expected_valid += 1
            # else: leave row at zeros → valid_mask=0 → drain skips
        gather_list.append(t)
    mod._drain_episodic_payloads_gpu(
        consumer=consumer,
        gather_list=gather_list,
        span_length=span_length,
        key_rep_dim=key_rep_dim,
        k_max=k_max,
        current_step=42,
        embedding_version=1,
    )
    assert len(consumer.cache) == expected_valid


def test_drain_pushes_to_controller_query_queue() -> None:
    """The Phase 2 controller_query_queue grows by exactly the number
    of valid rows in the gather_list, each entry carrying rank + k +
    pressure + residual.
    """
    mod = _load_runner_module()
    span_length = 2
    key_rep_dim = 3
    k_max = 2
    consumer = _build_consumer_state(
        mod, span_length=span_length, key_rep_dim=key_rep_dim,
    )
    gather_list: list[torch.Tensor] = []
    # Rank 0: 1 valid, 1 invalid. Rank 1: 2 valid. Rank 2: 0 valid.
    for r, valid_per_rank in [(0, 1), (1, 2), (2, 0)]:
        t = make_slot_tensor(
            k_max=k_max,
            span_length=span_length,
            key_rep_dim=key_rep_dim,
            device=torch.device("cpu"),
        )
        for k in range(valid_per_rank):
            pack_payload(
                t[k],
                valid_mask=1.0,
                pressure=0.1 * (k + 1),
                key_fp=10 * r + k,
                value_anchor_id=k,
                value_tok_ids=torch.zeros(span_length, dtype=torch.int64),
                key_rep=torch.zeros(key_rep_dim),
                residual=torch.tensor(
                    [float(r + k + 1)] * key_rep_dim, dtype=torch.float32,
                ),
                span_length=span_length,
                key_rep_dim=key_rep_dim,
            )
        gather_list.append(t)
    mod._drain_episodic_payloads_gpu(
        consumer=consumer,
        gather_list=gather_list,
        span_length=span_length,
        key_rep_dim=key_rep_dim,
        k_max=k_max,
        current_step=7,
        embedding_version=0,
    )
    # 1 + 2 + 0 = 3 valid rows.
    assert len(consumer.controller_query_queue) == 3
    # Each entry has the documented fields.
    for entry in consumer.controller_query_queue:
        assert "step" in entry and entry["step"] == 7
        assert "rank" in entry
        assert "k" in entry
        assert "pressure" in entry
        assert "residual" in entry
        assert isinstance(entry["residual"], torch.Tensor)
        assert entry["residual"].shape == (key_rep_dim,)
    # Rank routing: rank 0 contributes 1 entry, rank 1 contributes 2,
    # rank 2 contributes 0 — preserved through the drain.
    rank_counts: dict[int, int] = {}
    for entry in consumer.controller_query_queue:
        rank_counts[entry["rank"]] = rank_counts.get(entry["rank"], 0) + 1
    assert rank_counts.get(0, 0) == 1
    assert rank_counts.get(1, 0) == 2
    assert rank_counts.get(2, 0) == 0


# ---------------------------------------------------------------------------
# Test 7: 4-rank mp.spawn gloo end-to-end
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Same idiom as ``test_distributed_allreduce_grads._pick_free_port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _TinyTokenTrainModel(nn.Module):
    """Mirror of the train-step test model. Tiny vocab + hidden so we
    can run a real forward+backward in the gloo workers without compile.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(6, 4)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, 6, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embed(inputs)


def _gpu_drain_test_worker(
    rank: int,
    world_size: int,
    master_port: int,
    result_path: str,
    span_length: int,
    key_rep_dim: int,
    k_max: int,
) -> None:
    """4-rank gloo end-to-end smoke for the GPU IPC path.

    Each train rank (0/1/2) runs one ``_run_train_step`` with a real
    forward+backward; the episodic rank (3) skips main, calls
    ``dist.gather`` as the destination, drains into the cache + queue,
    and participates in the SUM all-reduce. Asserts both consumer-side
    counters end up positive (cache filled, queue grew).
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

        torch.manual_seed(31 + rank)
        is_episodic_rank = (rank == world_size - 1)
        all_group = dist.new_group(list(range(world_size)))

        model = _TinyTokenTrainModel()
        # Episodic rank's emit handle stays all-zeros; train ranks pack
        # into theirs. Same K_max + slot_dim across all ranks → gather
        # symmetric shape.
        emit_config = {
            "episodic_enabled": True,
            "episodic_span_length": span_length,
            "episodic_fingerprint_window": 1,
            "episodic_key_rep_dim": key_rep_dim,
            "episodic_top_p": 0.5,
            "episodic_k_max": k_max,
            "model_dim": key_rep_dim,
        }
        emit_handle = mod._create_episodic_emit(
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
            config=emit_config,
        )
        consumer = mod._attach_episodic_consumer(
            episodic_enabled=True,
            is_episodic_rank=is_episodic_rank,
            world_size=world_size,
            config={
                "episodic_capacity": 64,
                "episodic_span_length": span_length,
                "episodic_key_rep_dim": key_rep_dim,
                "episodic_grace_steps": 100,
                "episodic_utility_ema_decay": 0.99,
                # End-to-end test asserts queue grows to match cache_len,
                # so the queue gate must be on. Phase 1 production default
                # is False; see test_controller_query_queue_default_off.
                "controller_query_enabled": True,
            },
            model_dim=key_rep_dim,
            all_group=all_group,
        )
        # Inputs sized so positions can pass the boundary check
        # (window=1, span=2 → need t >= 1 and t + 2 <= T).
        inputs = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
        targets = torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.int64)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        # Train + episodic both call ``_run_train_step``; the function
        # branches on ``is_episodic_rank``.
        n_steps = 3
        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            mod._run_train_step(
                model=model,
                inputs=inputs,
                targets=targets,
                chunk_size=4,
                precision="fp32",
                ddp_active=True,
                world_size=world_size,
                rank=rank,
                lm_head_backward_mode="fused",
                grad_allreduce_mode="bulk",
                is_episodic_rank=is_episodic_rank,
                all_group=all_group,
                episodic_emit=emit_handle,
                episodic_consumer=consumer,
                current_step=step,
                embedding_version=0,
            )

        dist.barrier(group=all_group)

        if is_episodic_rank:
            with open(result_path + f".{rank}", "w") as fh:
                fh.write(
                    f"cache_len={len(consumer.cache)}\n"
                    f"heartbeat={consumer.heartbeat[0]}\n"
                    f"query_queue_len={len(consumer.controller_query_queue)}\n"
                )
        else:
            with open(result_path + f".{rank}", "w") as fh:
                fh.write("ok\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestEpisodicGpuDrainEndToEnd(unittest.TestCase):
    """4-rank gloo coverage for the gather-based episodic IPC end-to-end."""

    def test_4rank_gloo_end_to_end(self):
        """3 train ranks emit packed slot tensors; episodic rank drains
        into the cache + queue. After 3 steps:

          - cache_len > 0 (some valid slots survived the gather)
          - heartbeat == n_steps (drain ran every step)
          - query_queue_len == cache_len (one queue entry per cache append)
        """
        world_size = 4
        span_length = 2
        key_rep_dim = 4
        k_max = 4
        n_steps = 3
        with tempfile.TemporaryDirectory() as tmp:
            result_path = os.path.join(tmp, "result")
            master_port = _pick_free_port()
            torch_mp.spawn(
                _gpu_drain_test_worker,
                args=(
                    world_size, master_port, result_path,
                    span_length, key_rep_dim, k_max,
                ),
                nprocs=world_size,
                join=True,
            )
            content = (
                Path(f"{result_path}.{world_size - 1}").read_text().strip()
            )
            kv = dict(line.split("=") for line in content.splitlines())
            cache_len = int(kv["cache_len"])
            heartbeat = int(kv["heartbeat"])
            queue_len = int(kv["query_queue_len"])
            self.assertEqual(
                heartbeat, n_steps,
                f"episodic-rank heartbeat advances once per step; got "
                f"{heartbeat} after {n_steps} steps",
            )
            self.assertGreater(
                cache_len, 0,
                "cache should hold at least one valid slot end-to-end; "
                f"got {cache_len}",
            )
            # Phase 1 contract: every cache append also pushes a queue
            # entry. Pin them to be equal so a missing routing branch
            # would surface here.
            self.assertEqual(
                cache_len, queue_len,
                f"controller_query_queue length must match cache_len; "
                f"got cache={cache_len}, queue={queue_len}",
            )


if __name__ == "__main__":
    unittest.main()
