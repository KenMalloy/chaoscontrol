"""CRCT integration coverage for the Exp23/24 fast runner.

These tests pin the seams that can silently invalidate CRCT:

* Exp24's imported model builder must stop hardcoding a bare SSM when
  ``crct_enabled=True``.
* The fast runner must compose weighted LM CE plus residual packet injection
  from a dense teacher payload without putting slot reads on the trunk path.
* The oracle path must append real memory so ``force_on`` can differ from
  ``off`` after one scored batch.
"""
from __future__ import annotations

import importlib.util
import copy
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from chaoscontrol.model import CareStudentLM
from chaoscontrol.replay_eviction import ReplayEvictionLoop
from chaoscontrol.wake_cache_txn import TransactionalWakeCache


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"
RUNNER21_PATH = REPO / "experiments" / "21_sgns_tokenizer" / "runner_exp21.py"


@pytest.fixture(autouse=True)
def _restore_fast_runner_backend_env():
    keys = ("CHAOSCONTROL_DIAG_SCAN_BACKEND", "CHAOSCONTROL_POST_SCAN_BACKEND")
    old = {key: os.environ.get(key) for key in keys}
    os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] = "chunked"
    os.environ["CHAOSCONTROL_POST_SCAN_BACKEND"] = "eager"
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_crct_model() -> CareStudentLM:
    torch.manual_seed(123)
    model = CareStudentLM(
        vocab_size=32,
        dim=8,
        num_layers=1,
        ff_mult=2,
        a_mode="diag",
        outer_model_dim=4,
        outer_model_type="multislot",
        outer_max_slots=64,
        buffer_mode="append_only",
    )
    model.train()
    return model


def _wait_for_metric(
    transport,
    key: str,
    expected: int,
    timeout_s: float = 5.0,
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if int(transport.diagnostics().get(key, 0)) >= int(expected):
            return
        time.sleep(0.01)
    assert int(transport.diagnostics().get(key, 0)) >= int(expected)


def test_crct_rank_topology_splits_packet_and_maintenance_only_at_8x() -> None:
    mod = _load_module("runner_fast_path_crct_rank_topology", RUNNER_PATH)

    top4 = mod._crct_rank_topology(world_size=4, replay_eviction_enabled=True)
    assert top4["train_ranks"] == [0, 1, 2]
    assert top4["packet_rank"] == 3
    assert top4["maintenance_rank"] == 3
    assert top4["memory_ranks"] == [3]
    assert top4["split_memory_ranks"] is False

    top8 = mod._crct_rank_topology(world_size=8, replay_eviction_enabled=True)
    assert top8["train_ranks"] == [0, 1, 2, 3, 4, 5]
    assert top8["packet_rank"] == 6
    assert top8["maintenance_rank"] == 7
    assert top8["memory_ranks"] == [6, 7]
    assert top8["split_memory_ranks"] is True


def test_crct_online_eval_state_merges_packet_cache_and_maintenance_state() -> None:
    mod = _load_module("runner_fast_path_crct_online_eval_state", RUNNER_PATH)
    packet_model = _tiny_crct_model()
    records = packet_model.append_memory_from_hidden(
        torch.randn(1, 4, packet_model.dim),
        max_tokens=3,
    )
    assert len(records) == 3

    packet_state = mod._crct_packet_cache_eval_state(packet_model)
    merged = mod._merge_online_eval_state_payloads(
        [
            {"packet_cache": packet_state},
            {"replay_eviction": {"schema_version": 1, "updates_total": 7}},
            None,
        ]
    )

    assert set(merged) == {"packet_cache", "replay_eviction"}
    assert merged["packet_cache"]["slot_count"] == 3
    assert merged["replay_eviction"]["updates_total"] == 7

    eval_model = _tiny_crct_model()
    assert len(eval_model.outer_model.table) == 0
    applied = mod._apply_crct_packet_cache_eval_state(
        eval_model,
        merged["packet_cache"],
    )
    assert applied == 3
    assert len(eval_model.outer_model.table) == 3


def test_crct_maintenance_mailbox_has_own_request_ring_and_no_result_packets(
    tmp_path,
) -> None:
    mod = _load_module("runner_fast_path_crct_maintenance_mailbox", RUNNER_PATH)
    model = _tiny_crct_model()
    cache = TransactionalWakeCache(max_moments=0, max_hidden_buffer=0)
    loop = ReplayEvictionLoop()
    kwargs = {
        "world_size": 8,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
        "memory_rank": 7,
        "memory_role": "maintenance",
        "produce_results": False,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank7 = mod._CrctMailboxTeacherTransport(rank=7, **kwargs)
    try:
        assert rank0.diagnostics()["produce_results"] is False
        assert rank0._teacher_result_ring is None
        assert rank0._teacher_result_payload is None
        assert rank0._teacher_request_ring_name != mod._teacher_shm_name(
            Path(tmp_path), "tq"
        )

        inputs = torch.arange(10, dtype=torch.long).reshape(2, 5)
        targets = torch.arange(1, 11, dtype=torch.long).reshape(2, 5)
        rank0.begin_step(inputs=inputs, targets=targets, step=3)
        _wait_for_metric(rank0, "request_broadcasts_completed", 1)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not rank7.pending_input_requests:
            rank7.begin_step(inputs=inputs, targets=targets, step=3)
            time.sleep(0.01)
        assert len(rank7.pending_input_requests) == 1
        assert rank7.pending_input_requests[0]["step"] == 3
        assert rank7.diagnostics()["teacher_shm_request_events_popped"] == 1
        rank7.after_optimizer_step(
            model=model,
            cache=cache,
            scarcity_optimizer=None,
            step=3,
            total_steps=4,
            tau=0.1,
            strength=0.1,
            w_max=1.2,
            alpha_max=0.15,
            memory_write_tokens=4,
            replay_eviction_loop=loop,
        )
        diag7 = rank7.diagnostics()
        assert diag7["payloads_scored"] == 0
        assert diag7["maintenance_score_path_skips"] == 1
        assert diag7["maintenance_request_frames_ingested"] == 1
        assert diag7["pending_input_requests"] == 0
        assert loop.has_probe()
        assert loop._probe_step == 3
        torch.testing.assert_close(
            loop._probe_input_ids,
            mod._crct_full_input_ids(inputs, targets).to(torch.int32),
        )
    finally:
        rank7.close()
        rank0.close()


def test_crct_packet_builder_matches_force_on_pre_recurrence_lane() -> None:
    model = _tiny_crct_model()
    model.eval()
    torch.manual_seed(7)
    model.append_memory_from_hidden(torch.randn(2, 4, model.dim), max_tokens=4)
    inputs = torch.arange(10, dtype=torch.long).reshape(2, 5) % 32

    packet = model.build_episodic_packet(inputs)

    assert packet["packet_source_count"] > 0
    assert packet["memory_residual"].shape == (2, 1, model.dim)
    assert packet["memory_gate"].shape == inputs.shape
    assert torch.equal(packet["memory_gate"], torch.ones_like(packet["memory_gate"]))

    with torch.no_grad():
        h_force = model.encode(inputs, memory_mode="force_on")
        h_packet = model.encode(
            inputs,
            memory_mode="packet",
            episodic_residual=packet["memory_residual"],
            episodic_gate=packet["memory_gate"],
        )

    torch.testing.assert_close(h_packet, h_force, rtol=1e-5, atol=1e-5)


def test_crct_fast_path_allows_bucket_prototypes_as_sidecar_prior() -> None:
    mod = _load_module("runner_fast_path_crct_bucket_prototype_gate", RUNNER_PATH)
    model = CareStudentLM(
        vocab_size=32,
        dim=8,
        num_layers=1,
        ff_mult=2,
        a_mode="diag",
        outer_model_dim=4,
        outer_model_type="multislot",
        outer_max_slots=64,
        buffer_mode="append_only",
        bucket_prototypes=True,
        prototype_dim=4,
    )

    assert model.bucket_prototypes_module is not None
    mod._reject_unsupported_fast_step(model, crct_enabled=True)


def test_replay_eviction_probe_tracks_latest_rank3_teacher_batch() -> None:
    mod = _load_module("runner_fast_path_crct_probe_test", RUNNER_PATH)
    model = _tiny_crct_model()
    cache = TransactionalWakeCache(max_moments=0, max_hidden_buffer=0)
    loop = ReplayEvictionLoop()

    inputs_1 = (torch.arange(16).reshape(2, 8) % 32).to(torch.int32)
    targets_1 = ((torch.arange(16).reshape(2, 8) + 1) % 32).to(torch.long)
    mod._crct_score_payload_inline(
        model=model,
        cache=cache,
        scarcity_optimizer=None,
        inputs=inputs_1,
        targets=targets_1,
        step=10,
        total_steps=100,
        tau=0.10,
        strength=0.10,
        w_max=1.20,
        alpha_max=0.15,
        memory_write_tokens=4,
        update_model_memory_after=True,
    )
    assert mod._crct_replay_cache_probe(loop, model, 10) is True
    assert loop._probe_step == 10
    assert loop._probe_cue is not None
    first_probe = loop._probe_input_ids.clone()
    first_cue = loop._probe_cue.clone()
    assert mod._crct_replay_cache_probe(loop, model, 10) is False

    inputs_2 = ((torch.arange(16).reshape(2, 8) + 7) % 32).to(torch.int32)
    targets_2 = ((torch.arange(16).reshape(2, 8) + 11) % 32).to(torch.long)
    mod._crct_score_payload_inline(
        model=model,
        cache=cache,
        scarcity_optimizer=None,
        inputs=inputs_2,
        targets=targets_2,
        step=20,
        total_steps=100,
        tau=0.10,
        strength=0.10,
        w_max=1.20,
        alpha_max=0.15,
        memory_write_tokens=4,
        update_model_memory_after=True,
    )
    assert mod._crct_replay_cache_probe(loop, model, 20) is True
    assert loop._probe_step == 20
    assert not torch.equal(loop._probe_input_ids, first_probe)
    assert loop._probe_cue is not None
    assert not torch.equal(loop._probe_cue, first_cue)


def test_replay_eviction_tick_uses_teacher_step_not_memory_rank_spin_step() -> None:
    mod = _load_module("runner_fast_path_crct_replay_clock_test", RUNNER_PATH)
    model = _tiny_crct_model()
    cache = TransactionalWakeCache(max_moments=0, max_hidden_buffer=0)
    loop = ReplayEvictionLoop(
        min_slot_age_steps=0,
        max_seconds_per_tick=999.0,
        frame_ttl_steps=256,
        slot_work_chunk_size=4,
    )

    inputs = (torch.arange(16).reshape(2, 8) % 32).to(torch.int32)
    targets = ((torch.arange(16).reshape(2, 8) + 1) % 32).to(torch.long)
    mod._crct_score_payload_inline(
        model=model,
        cache=cache,
        scarcity_optimizer=None,
        inputs=inputs,
        targets=targets,
        step=10,
        total_steps=100,
        tau=0.10,
        strength=0.10,
        w_max=1.20,
        alpha_max=0.15,
        memory_write_tokens=4,
        update_model_memory_after=True,
    )
    assert mod._crct_replay_cache_probe(loop, model, 10) is True
    assert loop.has_probe()

    # Rank 3 can execute many local maintenance iterations while train ranks
    # advance only a few steps.  The replay frame should age by the teacher's
    # train step, not by this local spin counter.
    replay_step = mod._crct_replay_tick_step(loop, model, fallback_step=10_000)
    assert replay_step == 10
    result = loop.tick(model=model, step=replay_step)
    diag = loop.diagnostics()
    assert result is not None
    assert diag["probe_frames_dropped_stale"] == 0
    assert diag["replays_total"] == 1
    assert diag["slots_scored_total"] > 0


def _pick_free_port_or_skip() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", 0))
        except PermissionError as exc:
            pytest.skip(f"localhost socket bind unavailable in sandbox: {exc}")
        return int(sock.getsockname()[1])


def _worker_collect_crct_payload(
    rank: int,
    world_size: int,
    port: int,
    result_dir: str,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mod = _load_module(f"runner_fast_path_crct_dist_{rank}", RUNNER_PATH)
        model = _tiny_crct_model()
        cache = TransactionalWakeCache(max_moments=0, max_hidden_buffer=0)
        all_group = dist.new_group(list(range(world_size)))
        base = torch.arange(10, dtype=torch.long).reshape(2, 5)
        inputs = ((base + rank) % 32).to(dtype=torch.int32)
        targets = ((base + rank + 1) % 32).to(dtype=torch.long)

        payload = mod._collect_crct_teacher_payload(
            model=model,
            cache=cache,
            scarcity_optimizer=None,
            inputs=inputs,
            targets=targets,
            rank=rank,
            world_size=world_size,
            all_group=all_group,
            step=3,
            total_steps=10,
            tau=0.1,
            strength=0.1,
            w_max=1.2,
            alpha_max=0.15,
            memory_write_tokens=4,
        )
        record = {
            "rank": int(rank),
            "has_payload": payload is not None,
            "slots": int(len(model.outer_model._slots)),
        }
        if payload is not None:
            record.update(
                {
                    "target_shape": list(payload["target"].shape),
                    "confidence_shape": list(payload["confidence"].shape),
                    "loss_weight_shape": list(payload["loss_weight"].shape),
                    "utility_shape": list(payload["utility"].shape),
                    "finite": bool(
                        torch.isfinite(payload["loss_weight"]).all().item()
                    ),
                }
            )
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps(record))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _worker_async_crct_transport(
    rank: int,
    world_size: int,
    port: int,
    result_dir: str,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mod = _load_module(f"runner_fast_path_crct_async_{rank}", RUNNER_PATH)
        model = _tiny_crct_model()
        cache = TransactionalWakeCache(max_moments=0, max_hidden_buffer=0)
        teacher_group = dist.new_group([0, world_size - 1])
        transport = None
        if rank in {0, world_size - 1}:
            transport = mod._CrctAsyncTeacherTransport(
                rank=rank,
                world_size=world_size,
                teacher_group=teacher_group,
                payload_shape=(1, 2, 5),
                full_ids_shape=(2, 6),
                device=torch.device("cpu"),
                payload_dtype=torch.float32,
                max_local_batches=8,
                max_payload_lag_steps=8,
            )
        expected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        matched: list[dict[str, object]] = []
        base = torch.arange(10, dtype=torch.long).reshape(2, 5)
        for step in range(6):
            inputs = ((base + rank * 7 + step * 3) % 32).to(dtype=torch.int32)
            targets = ((base + rank * 7 + step * 3 + 1) % 32).to(dtype=torch.long)
            if rank == 0:
                expected[step] = (inputs.clone(), targets.clone())
            ready = (
                transport.begin_step(inputs=inputs, targets=targets, step=step)
                if transport is not None
                else None
            )

            # Make the unit test deterministic without changing production
            # semantics: production polls; this test waits so the next step can
            # reap the previous broadcast and prove the batch join is exact.
            if transport is not None:
                for slot in list(transport.pending_result_broadcasts):
                    for work in slot["works"]:
                        work.wait()
                for slot in list(transport.pending_input_requests):
                    wait = getattr(slot["work"], "wait", None)
                    if wait is not None:
                        wait()

            if ready is not None and rank == 0:
                payload, train_inputs, train_targets = ready
                request_step = int(payload["step_id"].item())
                exp_inputs, exp_targets = expected[request_step]
                matched.append(
                    {
                        "request_step": request_step,
                        "matched_inputs": bool(torch.equal(train_inputs, exp_inputs)),
                        "matched_targets": bool(
                            torch.equal(train_targets, exp_targets)
                        ),
                        "finite": bool(
                            torch.isfinite(payload["loss_weight"]).all().item()
                        ),
                    }
                )

            if transport is not None:
                transport.after_optimizer_step(
                    model=model,
                    cache=cache,
                    scarcity_optimizer=None,
                    step=step,
                    total_steps=8,
                    tau=0.1,
                    strength=0.1,
                    w_max=1.2,
                    alpha_max=0.15,
                    memory_write_tokens=4,
                )
            # Test-only synchronization: production deliberately polls the
            # side channel, but this unit test wants deterministic proof that
            # each scored rank0 request can be joined back to its originating
            # local batch.
            dist.barrier()

        if transport is not None:
            transport.close()
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "rank": int(rank),
                    "matched": matched,
                    "slots": int(len(model.outer_model._slots)),
                    "diagnostics": (
                        transport.diagnostics()
                        if transport is not None
                        else {
                            "mode": "async_rank0_memory_broadcast",
                            "participant": False,
                            "requests_started": 0,
                            "payloads_used": 0,
                            "errors": 0,
                        }
                    ),
                }
            )
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _worker_async_crct_train_loop(
    rank: int,
    world_size: int,
    port: int,
    result_dir: str,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mod = _load_module(f"runner_fast_path_crct_loop_{rank}", RUNNER_PATH)
        model = _tiny_crct_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokens = torch.arange(256, dtype=torch.long) % 32
        result = mod.train_fast_for_budget(
            model,
            train_tokens=tokens,
            train_num_tokens=int(tokens.numel()),
            stride=5,
            seq_len=5,
            batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer,
            budget_seconds=5.0,
            chunk_size=8,
            grad_clip_norm=1.0,
            fused_grad_clip=False,
            rank=rank,
            world_size=world_size,
            seed=123,
            precision="fp32",
            stop_check_interval=1,
            stop_margin_seconds=0.0,
            vocab_size=32,
            max_steps=5,
            lm_head_backward_mode="single",
            grad_allreduce_mode="bulk",
            train_sampling_mode="random",
            prefetch_batches=False,
            crct_enabled=True,
            crct_async_teacher_transport=True,
            crct_async_teacher_pending_batches=8,
            crct_async_teacher_max_lag_steps=8,
            crct_async_teacher_payload_dtype="fp32",
            crct_teacher_score_interval_steps=1,
            crct_memory_write_tokens_per_step=4,
            crct_gradient_conflict_enabled=True,
        )
        if rank == 0:
            Path(result_dir, "rank0.json").write_text(
                json.dumps(result["mechanisms"]["crct"])
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _worker_slot_commit_transport(
    rank: int,
    world_size: int,
    port: int,
    result_dir: str,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mod = _load_module(f"runner_fast_path_slot_commit_{rank}", RUNNER_PATH)
        group = dist.new_group([0, 1])
        model = _tiny_crct_model()
        assert model.outer_model is not None
        initial = torch.zeros(1, model.outer_model.outer_dim)
        model.outer_model.append_kv_batch(initial, torch.zeros(1, dtype=torch.long))
        transport = mod._CrctSlotCommitPeerTransport(
            rank=rank,
            packet_rank=0,
            maintenance_rank=1,
            group=group,
            device=torch.device("cpu"),
        )
        replacement = torch.full_like(initial, 0.25)
        appended = torch.full_like(initial, -0.75)
        if rank == 0:
            assert transport.submit_peer(
                mod.SlotCommit(
                    slot_id=7,
                    action="APPEND",
                    step=5,
                    base_generation=None,
                    new_generation=0,
                    bucket_id=3,
                    event_id=700,
                    tensor=appended,
                )
            )
        if rank == 1:
            assert transport.submit_peer(
                mod.SlotCommit(
                    slot_id=0,
                    action="REFRESH",
                    step=5,
                    base_generation=0,
                    new_generation=1,
                    tensor=replacement,
                )
            )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            transport.poll(model=model)
            diag = transport.diagnostics()
            if (
                rank == 0
                and int(diag.get("maintenance_commits_applied", 0)) >= 1
                and int(diag.get("send_completed", 0)) >= 1
            ):
                break
            if (
                rank == 1
                and int(diag.get("append_commits_applied", 0)) >= 1
                and int(diag.get("send_completed", 0)) >= 1
            ):
                break
            time.sleep(0.005)
        diag = transport.diagnostics()
        tensor = model.outer_model.table.get_tensor(0)
        append_tensor = model.outer_model.table.get_tensor(7)
        append_record = model.outer_model.table.record(7)
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "diagnostics": diag,
                    "slot": tensor.tolist() if tensor is not None else None,
                    "append_slot": (
                        append_tensor.tolist() if append_tensor is not None else None
                    ),
                    "append_event_id": (
                        append_record.event_id if append_record is not None else None
                    ),
                    "generation": model.outer_model.table.record(0).write_generation,
                }
            )
        )
        transport.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_slot_commit_peer_transport_updates_packet_cache_over_p2p(tmp_path) -> None:
    port = _pick_free_port_or_skip()
    world_size = 2
    mp.spawn(
        _worker_slot_commit_transport,
        args=(world_size, port, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    rank0 = json.loads(Path(tmp_path, "rank0.json").read_text())
    rank1 = json.loads(Path(tmp_path, "rank1.json").read_text())
    assert rank0["diagnostics"]["maintenance_commits_applied"] == 1
    assert rank0["generation"] == 1
    assert rank0["slot"] == [[0.25, 0.25, 0.25, 0.25]]
    assert rank1["diagnostics"]["append_commits_applied"] == 1
    assert rank1["append_slot"] == [[-0.75, -0.75, -0.75, -0.75]]
    assert rank1["append_event_id"] == 700


def test_exp24_model_builder_threads_crct_memory_without_trunk_controller() -> None:
    mod = _load_module("runner_exp21_crct_test", RUNNER21_PATH)
    cfg = {
        "vocab_size": 32,
        "model_dim": 8,
        "num_layers": 1,
        "ff_mult": 2,
        "crct_enabled": True,
        "outer_model_dim": 4,
        "outer_max_slots": 128,
    }

    model = mod.build_model(cfg, torch.device("cpu"), torch.float32)

    assert not hasattr(model, "memory_controller")
    assert model.outer_model is not None
    assert model.outer_model.max_slots == 128
    assert model.buffer_mode == "append_only"


def test_crct_train_step_uses_payload_packet_without_controller_hot_path() -> None:
    mod = _load_module("runner_fast_path_crct_train_step", RUNNER_PATH)
    model = _tiny_crct_model()
    inputs = torch.randint(0, 32, (2, 5), dtype=torch.int32)
    targets = torch.randint(0, 32, (2, 5), dtype=torch.long)
    payload = {
        "target": torch.full((2, 5), 0.85),
        "confidence": torch.ones(2, 5),
        "loss_weight": torch.linspace(0.9, 1.1, 10).reshape(2, 5),
        "memory_residual": torch.randn(2, 1, 8),
        "memory_gate": torch.ones(2, 5),
    }

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=8,
        precision="fp32",
        ddp_active=False,
        world_size=1,
        lm_head_backward_mode="single",
        crct_enabled=True,
        crct_payload=payload,
    )

    assert torch.isfinite(loss)
    assert not hasattr(model, "memory_controller")
    # Rank 3 is the only CRCT memory writer. Train ranks consume a residual
    # packet from async payloads, but must not mutate local memory.
    assert len(model.outer_model._slots) == 0


def test_crct_train_step_does_not_read_slots_on_trunk_path() -> None:
    mod = _load_module("runner_fast_path_crct_no_trunk_memory_read", RUNNER_PATH)
    model = _tiny_crct_model()
    assert model.outer_model is not None
    model.outer_model.append_kv_batch(
        torch.randn(8, model.outer_model.outer_dim),
        torch.zeros(8, dtype=torch.long),
    )

    def _forbidden_read(*_args, **_kwargs):
        raise AssertionError("CRCT train-rank trunk path read outer memory")

    model.outer_model.read = _forbidden_read  # type: ignore[method-assign]
    model.outer_model.read_bucket = _forbidden_read  # type: ignore[method-assign]

    inputs = torch.randint(0, 32, (2, 5), dtype=torch.int32)
    targets = torch.randint(0, 32, (2, 5), dtype=torch.long)
    payload = {
        "target": torch.full((2, 5), 0.85),
        "confidence": torch.ones(2, 5),
        "loss_weight": torch.ones(2, 5),
        "memory_residual": torch.randn(2, 1, 8),
        "memory_gate": torch.ones(2, 5),
    }

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=8,
        precision="fp32",
        ddp_active=False,
        world_size=1,
        lm_head_backward_mode="single",
        crct_enabled=True,
        crct_payload=payload,
    )

    assert torch.isfinite(loss)
    assert not hasattr(model, "memory_controller")


def test_crct_plasticity_budget_payload_installs_on_optimizer() -> None:
    mod = _load_module("runner_fast_path_crct_plasticity_apply", RUNNER_PATH)
    from chaoscontrol.optim.muon import Muon

    p = torch.nn.Parameter(torch.zeros(2))
    opt = Muon([p], lr=0.01)
    opt.bind_param_names([("layers.0.core.log_a", p)])

    applied = mod._apply_plasticity_budget_payload(
        optimizer=opt,
        payload={
            "step_id": torch.tensor(11),
            "plasticity_budget": torch.tensor([0.0, 1.0]),
            "plasticity_confidence": torch.ones(2),
        },
        strength=0.5,
    )

    assert applied is True
    trace = opt.plasticity_budget_trace()
    assert trace["enabled"] is True
    assert trace["step"] == 11
    assert trace["lr_multiplier_max"] == pytest.approx(1.5)


def test_crct_mailbox_transport_matches_stored_batch(tmp_path) -> None:
    os.environ["CHAOSCONTROL_DIAG_SCAN_BACKEND"] = "chunked"
    mod = _load_module("runner_fast_path_crct_mailbox", RUNNER_PATH)
    model = _tiny_crct_model()
    cache = TransactionalWakeCache(max_moments=0, max_hidden_buffer=0)
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
        "score_stage_timing_enabled": True,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)
    base = torch.arange(10, dtype=torch.long).reshape(2, 5)
    inputs = (base % 32).to(dtype=torch.int32)
    targets = ((base + 1) % 32).to(dtype=torch.long)

    assert rank0.begin_step(inputs=inputs, targets=targets, step=0) is None
    assert rank0._request_write_thread is not None
    _wait_for_metric(rank0, "teacher_shm_request_events_pushed", 1)
    assert rank3.begin_step(inputs=inputs, targets=targets, step=0) is None
    rank3.after_optimizer_step(
        model=model,
        cache=cache,
        scarcity_optimizer=None,
        step=0,
        total_steps=4,
        tau=0.1,
        strength=0.1,
        w_max=1.2,
        alpha_max=0.15,
        memory_write_tokens=4,
    )

    ready = rank0.begin_step(inputs=inputs + 3, targets=targets + 3, step=1)
    assert rank0._request_write_thread is not None
    assert ready is not None
    _wait_for_metric(rank0, "teacher_shm_request_events_pushed", 2)
    payload, train_inputs, train_targets = ready
    assert int(payload["step_id"].item()) == 0
    assert torch.equal(train_inputs, inputs)
    assert torch.equal(train_targets, targets)
    assert torch.isfinite(payload["loss_weight"]).all()
    diag0 = rank0.diagnostics()
    diag3 = rank3.diagnostics()
    assert diag0["mode"] == "async_rank0_memory_shm"
    assert diag0["payloads_used"] == 1
    assert diag0["request_stage_started"] == 2
    assert diag0["request_writer_cpu_copy_seconds_max"] >= 0.0
    assert diag0["request_submit_seconds_max"] >= 0.0
    assert diag3["score_stage_timing_enabled"] is True
    assert diag3["score_stage_samples"] == 0
    assert diag0["request_host_stage_bytes"] == (
        inputs.shape[0] * (inputs.shape[1] + 1) * 4
    )
    assert isinstance(diag0["request_host_pinned"], bool)
    assert diag0["local_batch_gpu_clones"] == 0
    assert diag3["payloads_scored"] == 0
    assert diag3["payloads_served"] == 1
    assert diag3["payloads_served_approximate"] == 1
    assert diag3["packet_service_seconds_max"] >= 0.0
    assert diag3["packet_service_approx_write_records"] == 4
    assert torch.count_nonzero(payload["target"]) == 0
    assert torch.count_nonzero(payload["confidence"]) == 0
    assert torch.isfinite(payload["memory_gate"]).all()


def test_crct_mailbox_gpu3_packet_score_ignores_fastslow_readiness(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_no_fastslow", RUNNER_PATH)
    model = _tiny_crct_model()
    cache = TransactionalWakeCache(max_moments=0, max_hidden_buffer=0)
    fast_slow = mod.FastSlowConsolidator.from_config(
        model,
        {
            "fast_slow_enabled": True,
            "fast_slow_interval": 0,
            "fast_slow_alpha": 0.25,
        },
    )
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)
    base = torch.arange(10, dtype=torch.long).reshape(2, 5)
    inputs = (base % 32).to(dtype=torch.int32)
    targets = ((base + 1) % 32).to(dtype=torch.long)

    rank0.begin_step(inputs=inputs, targets=targets, step=0)
    _wait_for_metric(rank0, "teacher_shm_request_events_pushed", 1)
    rank3.begin_step(inputs=inputs, targets=targets, step=0)
    rank3.after_optimizer_step(
        model=model,
        cache=cache,
        scarcity_optimizer=None,
        step=0,
        total_steps=4,
        tau=0.1,
        strength=0.1,
        w_max=1.2,
        alpha_max=0.15,
        memory_write_tokens=4,
        fast_slow=fast_slow,
    )

    diag3 = rank3.diagnostics()
    assert diag3["payloads_scored"] == 0
    assert diag3["payloads_served"] == 1
    assert diag3["payloads_served_approximate"] == 1
    assert diag3["fast_slow_readiness_scores"] == 0
    assert diag3["fast_slow_readiness_skips_gpu3_mirror"] == 1
    assert diag3["fast_slow_decisions_issued"] == 0


def test_crct_mailbox_preserves_burst_results_fifo(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_result_fifo", RUNNER_PATH)
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)
    inputs0 = torch.full((2, 5), 3, dtype=torch.int32)
    targets0 = torch.full((2, 5), 4, dtype=torch.long)
    inputs1 = torch.full((2, 5), 7, dtype=torch.int32)
    targets1 = torch.full((2, 5), 8, dtype=torch.long)
    rank0.local_batches_by_step[0] = (inputs0, targets0)
    rank0.local_batch_order.append(0)
    rank0.local_batches_by_step[1] = (inputs1, targets1)
    rank0.local_batch_order.append(1)

    for step, value in ((0, 0.25), (1, 0.75)):
        rank3._write_result(
            request_step=step,
            scored={
                "target": torch.full((2, 5), value),
                "confidence": torch.ones(2, 5),
                "loss_weight": torch.ones(2, 5),
                "utility": torch.zeros(2, 5),
            },
        )

    ready0 = rank0._poll_results(current_step=2)
    ready1 = rank0._poll_results(current_step=3)

    assert ready0 is not None
    assert ready1 is not None
    payload0, train_inputs0, train_targets0 = ready0
    payload1, train_inputs1, train_targets1 = ready1
    assert int(payload0["step_id"].item()) == 0
    assert int(payload1["step_id"].item()) == 1
    assert torch.equal(train_inputs0, inputs0)
    assert torch.equal(train_targets0, targets0)
    assert torch.equal(train_inputs1, inputs1)
    assert torch.equal(train_targets1, targets1)
    diag0 = rank0.diagnostics()
    assert diag0["payloads_received"] == 2
    assert diag0["payloads_used"] == 2
    assert diag0["superseded_payloads_dropped"] == 0
    assert diag0["ready_result_queue_max"] == 2


def test_crct_mailbox_request_ingest_is_latest_only(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_request_latest", RUNNER_PATH)
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)
    inputs0 = torch.full((2, 5), 3, dtype=torch.int32)
    targets0 = torch.full((2, 5), 4, dtype=torch.long)
    inputs1 = torch.full((2, 5), 7, dtype=torch.int32)
    targets1 = torch.full((2, 5), 8, dtype=torch.long)

    rank0.begin_step(inputs=inputs0, targets=targets0, step=0)
    _wait_for_metric(rank0, "teacher_shm_request_events_pushed", 1)
    rank0.begin_step(inputs=inputs1, targets=targets1, step=1)
    _wait_for_metric(rank0, "teacher_shm_request_events_pushed", 2)

    assert rank3.begin_step(inputs=inputs0, targets=targets0, step=1) is None

    assert len(rank3.pending_input_requests) == 1
    latest = rank3.pending_input_requests.popleft()
    assert int(latest["step"]) == 1
    torch.testing.assert_close(
        latest["buffer"],
        mod._crct_full_input_ids(inputs1, targets1).to(torch.int32),
    )
    diag3 = rank3.diagnostics()
    assert diag3["completed_requests_dropped"] == 1
    assert diag3["memory_rank_request_events_superseded"] == 1
    assert diag3["memory_rank_pump_request_pops"] == 2
    assert diag3["max_pending_input_requests"] == 1


def test_crct_mailbox_transport_round_trips_memory_packet(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_packet", RUNNER_PATH)
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)
    inputs = torch.randint(0, 32, (2, 5), dtype=torch.int32)
    targets = torch.randint(0, 32, (2, 5), dtype=torch.long)
    rank0.local_batches_by_step[0] = (inputs, targets)
    rank0.local_batch_order.append(0)
    residual = torch.randn(2, 1, 8)
    gate = torch.full((2, 5), 0.75)

    rank3._write_result(
        request_step=0,
        scored={
            "target": torch.full((2, 5), 0.85),
            "confidence": torch.ones(2, 5),
            "loss_weight": torch.ones(2, 5),
            "utility": torch.zeros(2, 5),
            "memory_residual": residual,
            "memory_gate": gate,
        },
    )

    ready = rank0._poll_results(current_step=1)
    assert ready is not None
    payload, _inputs, _targets = ready
    assert torch.allclose(payload["memory_residual"], residual)
    assert torch.allclose(payload["memory_gate"], gate)
    assert rank3.diagnostics()["memory_packets_sent"] == 1
    assert rank0.diagnostics()["memory_packets_received"] == 1


def test_crct_mailbox_aliases_memory_gate_to_target_for_compact_packet(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_packet_alias", RUNNER_PATH)
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)
    inputs = torch.randint(0, 32, (2, 5), dtype=torch.int32)
    targets = torch.randint(0, 32, (2, 5), dtype=torch.long)
    rank0.local_batches_by_step[0] = (inputs, targets)
    rank0.local_batch_order.append(0)
    residual = torch.randn(2, 1, 8)
    target = torch.full((2, 5), 0.85)

    rank3._write_result(
        request_step=0,
        scored={
            "target": target,
            "confidence": torch.ones(2, 5),
            "loss_weight": torch.ones(2, 5),
            "utility": torch.zeros(2, 5),
            "memory_residual": residual,
            "memory_gate": target,
            "memory_gate_alias_target": True,
        },
    )

    ready = rank0._poll_results(current_step=1)
    assert ready is not None
    payload, _inputs, _targets = ready
    assert torch.allclose(payload["memory_residual"], residual)
    assert torch.allclose(payload["memory_gate"], target)
    diag3 = rank3.diagnostics()
    diag0 = rank0.diagnostics()
    assert diag3["memory_packet_gate_alias_target_sent"] == 1
    assert diag0["memory_packet_gate_alias_target_received"] == 1
    assert diag3["memory_packet_bytes_sent"] == residual.numel() * residual.element_size()


def test_crct_mailbox_round_trips_plasticity_budget_packet(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_plasticity", RUNNER_PATH)
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
        "plasticity_ema_beta": 0.5,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)
    inputs = torch.randint(0, 32, (2, 5), dtype=torch.int32)
    targets = torch.randint(0, 32, (2, 5), dtype=torch.long)
    rank0.local_batches_by_step[0] = (inputs, targets)
    rank0.local_batch_order.append(0)
    coverage = torch.tensor([1.0, -0.5, 0.25])
    confidence = torch.tensor([0.8, 0.7, 0.6])
    budget = torch.tensor([0.8, 0.0, 0.15])

    rank3._write_result(
        request_step=0,
        scored={
            "target": torch.full((2, 5), 0.85),
            "confidence": torch.ones(2, 5),
            "loss_weight": torch.ones(2, 5),
            "utility": torch.zeros(2, 5),
            "plasticity_coverage": coverage,
            "plasticity_confidence": confidence,
            "plasticity_budget": budget,
        },
    )

    ready = rank0._poll_results(current_step=2)
    assert ready is not None
    payload, _inputs, _targets = ready
    torch.testing.assert_close(payload["plasticity_coverage"], coverage)
    torch.testing.assert_close(payload["plasticity_confidence"], confidence)
    torch.testing.assert_close(payload["plasticity_budget"], budget)
    diag3 = rank3.diagnostics()
    diag0 = rank0.diagnostics()
    assert diag3["plasticity_packets_sent"] == 1
    assert diag0["plasticity_packets_received"] == 1
    assert diag0["plasticity_lag_steps_max"] == 2
    assert diag0["plasticity_budget_max_received"] == pytest.approx(0.8)


def test_crct_mailbox_rejects_sequence_memory_packet(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_packet_rejects", RUNNER_PATH)
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)

    with pytest.raises(ValueError, match="compact"):
        rank3._write_result(
            request_step=0,
            scored={
                "target": torch.full((2, 5), 0.85),
                "confidence": torch.ones(2, 5),
                "loss_weight": torch.ones(2, 5),
                "utility": torch.zeros(2, 5),
                "memory_residual": torch.randn(2, 5, 8),
                "memory_gate": torch.ones(2, 5),
            },
        )
    assert rank3.diagnostics()["memory_packet_sequence_residual_rejections"] == 1


def test_crct_mailbox_defers_low_priority_maintenance_only_at_ring_capacity(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_packet_priority", RUNNER_PATH)
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)

    assert rank3.should_defer_low_priority_maintenance() is False
    rank3.pending_input_requests.append(
        {"step": 7, "buffer": torch.zeros((2, 6), dtype=torch.int32)}
    )
    assert rank3.should_defer_low_priority_maintenance() is False
    rank3.pending_input_requests.clear()
    assert rank0._teacher_request_ring is not None
    for i in range(int(rank3._teacher_ring_capacity) - 1):
        assert rank0._teacher_request_ring.push(
            {
                "event_type": 6,
                "source_rank": 0,
                "status": 0,
                "flags": 0,
                "slice_count": 1,
                "request_id": i,
                "step": i,
                "weight_snapshot_version": 0,
                "full_ids": mod._teacher_empty_slice(),
            }
        )
    assert rank3.should_defer_low_priority_maintenance() is True

    diag = rank3.diagnostics()
    assert diag["low_priority_maintenance_allows"] == 2
    assert diag["low_priority_maintenance_defers"] == 1
    assert diag["low_priority_maintenance_defer_pending_requests"] == 0
    assert diag["low_priority_maintenance_defer_request_mailbox"] == 1
    assert diag["low_priority_maintenance_last_reason"] == "request_ring_capacity"


def test_crct_mailbox_weight_snapshot_applies_latest_model(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_weights", RUNNER_PATH)
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)
    model0 = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    model3 = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    with torch.no_grad():
        for param in model0.parameters():
            param.fill_(0.25)
        for param in model3.parameters():
            param.fill_(-0.5)

    before = {
        name: tensor.detach().clone()
        for name, tensor in model3.state_dict().items()
    }
    rank0.maybe_publish_weight_snapshot(model=model0, step=7)
    staged_expected = {
        name: tensor.detach().clone()
        for name, tensor in model0.state_dict().items()
    }
    with torch.no_grad():
        for param in model0.parameters():
            param.fill_(0.75)
    assert rank0._weight_publish_thread is not None
    _wait_for_metric(rank0, "weight_snapshot_published", 1)

    rank3.poll_weight_snapshot(model=model3, step=11)

    for name, tensor in model3.state_dict().items():
        assert not torch.equal(tensor, before[name])
        assert torch.equal(tensor, staged_expected[name])
    diag0 = rank0.diagnostics()
    diag3 = rank3.diagnostics()
    assert diag0["weight_snapshot_published"] == 1
    assert diag0["weight_snapshot_stage_started"] == 1
    assert diag0["weight_snapshot_hotpath_cpu_copies"] == 0
    assert diag0["weight_snapshot_stage_gpu_seconds_max"] >= 0.0
    assert diag0["weight_snapshot_writer_cpu_copy_seconds_max"] >= 0.0
    host_bytes = (
        diag0["weight_snapshot_host_pinned_bytes"]
        + diag0["weight_snapshot_host_pageable_bytes"]
    )
    assert host_bytes == diag0["weight_snapshot_stage_bytes"]
    assert diag0["mailbox_write_seconds_max"] == 0.0
    assert diag0["weight_snapshot_shm_writes"] == 1
    assert diag0["weight_snapshot_pickle_writes"] == 0
    assert diag3["weight_snapshot_applied"] == 1
    assert diag3["weight_snapshot_last_applied_step"] == 7
    assert diag3["weight_snapshot_version_lag_steps"] == 4
    assert diag3["weight_snapshot_shm_reads"] == 1
    assert diag3["weight_snapshot_pickle_reads"] == 0
    rank3.poll_weight_snapshot(model=model3, step=12)
    diag3_after = rank3.diagnostics()
    assert diag3_after["weight_snapshot_applied"] == 1
    assert diag3_after["weight_snapshot_stat_skips"] == 1


def test_crct_mailbox_weight_snapshot_applies_fastslow_decision(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_fastslow", RUNNER_PATH)
    kwargs = {
        "world_size": 4,
        "mailbox_dir": str(tmp_path),
        "payload_shape": (1, 2, 5),
        "full_ids_shape": (2, 6),
        "device": torch.device("cpu"),
        "payload_dtype": torch.float32,
        "max_local_batches": 8,
        "max_payload_lag_steps": 8,
        "score_interval_steps": 1,
    }
    rank0 = mod._CrctMailboxTeacherTransport(rank=0, **kwargs)
    rank3 = mod._CrctMailboxTeacherTransport(rank=3, **kwargs)
    model0 = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    model3 = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    with torch.no_grad():
        model0[0].weight.fill_(0.25)
        model3[0].weight.fill_(-0.5)
    fast_slow = mod.FastSlowConsolidator.from_config(
        model3,
        {
            "fast_slow_enabled": True,
            "fast_slow_interval": 0,
            "fast_slow_alpha": 0.25,
        },
    )
    decision = mod.FastSlowDecision(
        mode="learned",
        accepted=True,
        alpha=0.5,
        gate=1.0,
        effective_alpha=0.5,
        step=8,
        reason="test",
    )

    rank0.maybe_publish_weight_snapshot(
        model=model0,
        step=7,
        fast_slow_decision=decision,
    )
    assert rank0._weight_publish_thread is not None
    _wait_for_metric(rank0, "weight_snapshot_published", 1)

    rank3.poll_weight_snapshot(model=model3, step=11, fast_slow=fast_slow)

    assert torch.equal(model3.state_dict()["0.weight"], model0.state_dict()["0.weight"])
    expected_slow = torch.full_like(model3[0].weight, -0.125)
    torch.testing.assert_close(fast_slow.slow_state["0.weight"], expected_slow)
    assert rank0.diagnostics()["fast_slow_snapshot_decisions_published"] == 1
    assert rank3.diagnostics()["fast_slow_snapshot_decisions_applied"] == 1


def test_fastslow_result_payload_applies_memory_rank_decision() -> None:
    mod = _load_module("runner_fast_path_fastslow_result_decision", RUNNER_PATH)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    fast_slow = mod.FastSlowConsolidator.from_config(
        model,
        {
            "fast_slow_enabled": True,
            "fast_slow_interval": 0,
            "fast_slow_alpha": 0.25,
        },
    )
    with torch.no_grad():
        model[0].weight.fill_(3.0)
    decision = mod.FastSlowDecision(
        mode="learned",
        accepted=True,
        alpha=0.5,
        gate=1.0,
        effective_alpha=0.5,
        step=12,
        reason="memory_rank_oracle",
    )

    metrics = {}
    applied = mod._apply_fast_slow_result_payload(
        model=model,
        fast_slow=fast_slow,
        payload={"fast_slow_decision": mod.fast_slow_decision_to_dict(decision)},
        metrics=metrics,
    )

    assert applied is not None
    assert applied.reason == "memory_rank_oracle"
    assert metrics["fast_slow_result_decisions_applied"] == 1
    torch.testing.assert_close(
        fast_slow.slow_state["0.weight"],
        torch.full_like(model[0].weight, 2.0),
    )


def test_crct_mailbox_weight_snapshot_ignores_stale_writer_thread_state(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_weights_no_busy_gate", RUNNER_PATH)
    rank0 = mod._CrctMailboxTeacherTransport(
        rank=0,
        world_size=4,
        mailbox_dir=str(tmp_path),
        payload_shape=(1, 2, 5),
        full_ids_shape=(2, 6),
        device=torch.device("cpu"),
        payload_dtype=torch.float32,
        max_local_batches=8,
        max_payload_lag_steps=8,
        score_interval_steps=1,
    )
    blocker_started = threading.Event()
    blocker_release = threading.Event()

    def _block_writer() -> None:
        blocker_started.set()
        blocker_release.wait(timeout=5.0)

    stale_thread = threading.Thread(target=_block_writer)
    rank0._weight_publish_thread = stale_thread
    stale_thread.start()
    assert blocker_started.wait(timeout=1.0)
    rank0.maybe_publish_weight_snapshot(model=torch.nn.Linear(2, 2), step=1)
    _wait_for_metric(rank0, "weight_snapshot_published", 1)
    blocker_release.set()
    stale_thread.join(timeout=1.0)

    diag = rank0.diagnostics()
    assert diag["weight_snapshot_attempts"] == 1
    assert diag["weight_snapshot_publish_skipped_busy"] == 0
    assert diag["weight_snapshot_published"] == 1
    assert diag["weight_snapshot_pickle_writes"] == 0


def test_crct_mailbox_request_ignores_stale_writer_thread_state(tmp_path) -> None:
    mod = _load_module("runner_fast_path_crct_mailbox_request_no_busy_gate", RUNNER_PATH)
    rank0 = mod._CrctMailboxTeacherTransport(
        rank=0,
        world_size=4,
        mailbox_dir=str(tmp_path),
        payload_shape=(1, 2, 5),
        full_ids_shape=(2, 6),
        device=torch.device("cpu"),
        payload_dtype=torch.float32,
        max_local_batches=8,
        max_payload_lag_steps=8,
        score_interval_steps=1,
    )
    blocker_started = threading.Event()
    blocker_release = threading.Event()

    def _block_writer() -> None:
        blocker_started.set()
        blocker_release.wait(timeout=5.0)

    rank0._request_write_thread = threading.Thread(target=_block_writer)
    rank0._request_write_thread.start()
    assert blocker_started.wait(timeout=1.0)
    base = torch.arange(10, dtype=torch.long).reshape(2, 5)
    rank0.begin_step(
        inputs=(base % 32).to(dtype=torch.int32),
        targets=((base + 1) % 32).to(dtype=torch.long),
        step=0,
    )
    blocker_release.set()
    rank0._request_write_thread.join(timeout=1.0)

    diag = rank0.diagnostics()
    assert diag["request_write_skipped_busy"] == 0
    assert diag["requests_started"] == 1
    assert diag["teacher_shm_request_events_pushed"] == 1


def test_fastslow_readiness_oracle_scores_against_slow_mirror_without_mutating_fast_model() -> None:
    mod = _load_module("runner_fast_path_fastslow_readiness", RUNNER_PATH)
    model = _tiny_crct_model()
    fast_slow = mod.FastSlowConsolidator.from_config(
        model,
        {
            "fast_slow_enabled": True,
            "fast_slow_interval": 0,
            "fast_slow_alpha": 0.25,
        },
    )
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    slow_model = copy.deepcopy(model)
    fast_slow.copy_slow_to_model(slow_model)
    fast_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if torch.is_tensor(tensor)
    }
    full_ids = (torch.arange(24).reshape(3, 8) % 32).to(torch.int32)

    evidence = mod._score_fast_slow_readiness_inline(
        model=model,
        slow_model=slow_model,
        fast_slow=fast_slow,
        input_ids=full_ids,
        step=9,
        chunk_size=4,
    )

    assert evidence is not None
    assert evidence["credit_key"] == -1
    assert evidence["valid_tokens"] > 0
    assert evidence["score_seconds"] >= 0.0
    assert torch.isfinite(torch.tensor(float(evidence["delta_nll"])))
    for name, tensor in model.state_dict().items():
        if torch.is_tensor(tensor):
            torch.testing.assert_close(tensor, fast_state[name])


def test_crct_teacher_payload_appends_memory_after_scoring() -> None:
    mod = _load_module("runner_fast_path_crct_payload", RUNNER_PATH)
    model = _tiny_crct_model()
    cache = TransactionalWakeCache(max_moments=0, max_hidden_buffer=0)
    inputs = torch.randint(0, 32, (2, 5), dtype=torch.int32)
    targets = torch.randint(0, 32, (2, 5), dtype=torch.long)

    payload = mod._crct_score_payload_inline(
        model=model,
        cache=cache,
        scarcity_optimizer=None,
        inputs=inputs,
        targets=targets,
        step=0,
        total_steps=10,
        tau=0.1,
        strength=0.1,
        w_max=1.2,
        alpha_max=0.15,
        memory_write_tokens=3,
        update_model_memory_after=True,
    )

    assert payload["target"].shape == targets.shape
    assert payload["loss_weight"].shape == targets.shape
    assert len(model.outer_model._slots) == 3
    assert min(model.outer_model._slot_event_ids) > 0

    with torch.no_grad():
        h_off = model.encode(inputs, memory_mode="off")
        h_mem = model.encode(inputs, memory_mode="force_on")
    assert not torch.allclose(h_off, h_mem)

    second = mod._crct_score_payload_inline(
        model=model,
        cache=cache,
        scarcity_optimizer=None,
        inputs=inputs,
        targets=targets,
        step=1,
        total_steps=10,
        tau=0.1,
        strength=0.1,
        w_max=1.2,
        alpha_max=0.15,
        memory_write_tokens=3,
        update_model_memory_after=True,
    )
    assert second["memory_residual"].shape == (2, 1, model.dim)
    assert second["memory_gate"].shape == targets.shape


def test_crct_fail_open_skips_controller_hot_path() -> None:
    mod = _load_module("runner_fast_path_crct_fail_open", RUNNER_PATH)
    model = _tiny_crct_model()
    inputs = torch.randint(0, 32, (2, 5), dtype=torch.int32)
    targets = torch.randint(0, 32, (2, 5), dtype=torch.long)

    mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=8,
        precision="fp32",
        ddp_active=False,
        world_size=1,
        lm_head_backward_mode="single",
        crct_enabled=True,
        crct_payload=None,
    )

    assert not hasattr(model, "memory_controller")


def test_crct_rejects_async_grad_reducer_before_collectives() -> None:
    mod = _load_module("runner_fast_path_crct_async_guard", RUNNER_PATH)
    model = _tiny_crct_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, 32, (256,), dtype=torch.long)

    with pytest.raises(ValueError, match="grad_allreduce_mode='bulk'"):
        mod.train_fast_for_budget(
            model,
            train_tokens=tokens,
            train_num_tokens=int(tokens.numel()),
            stride=8,
            seq_len=8,
            batch_size=2,
            device=torch.device("cpu"),
            optimizer=optimizer,
            budget_seconds=1.0,
            chunk_size=8,
            grad_clip_norm=1.0,
            fused_grad_clip=False,
            rank=0,
            world_size=4,
            seed=123,
            precision="fp32",
            stop_check_interval=1,
            stop_margin_seconds=0.0,
            vocab_size=32,
            max_steps=1,
            lm_head_backward_mode="single",
            grad_allreduce_mode="async_param",
            train_sampling_mode="random",
            crct_enabled=True,
        )


def test_crct_optimizer_routes_aux_matrices_to_adamw_fallback() -> None:
    mod = _load_module("runner_fast_path_crct_optimizer", RUNNER_PATH)
    model = _tiny_crct_model()

    opt = mod._build_optimizer(
        {
            "optimizer": "muon",
            "base_lr": 1e-3,
            "weight_decay": 0.01,
            "fused_muon": False,
            "crct_enabled": True,
        },
        model,
    )

    matrix_names = opt._matrix_param_names
    assert matrix_names is not None
    assert "embed.weight" not in matrix_names
    assert "lm_head.weight" not in matrix_names
    assert "outer_model.encoder.weight" not in matrix_names
    assert any(name.startswith("layers.") for name in matrix_names)


def test_crct_teacher_payload_distributed_gather_score_broadcast() -> None:
    world_size = 4
    port = _pick_free_port_or_skip()
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _worker_collect_crct_payload,
            args=(world_size, port, tmp),
            nprocs=world_size,
            join=True,
        )
        records = [
            json.loads(Path(tmp, f"rank{rank}.json").read_text())
            for rank in range(world_size)
        ]

    for rec in records[:3]:
        assert rec["has_payload"] is True
        assert rec["target_shape"] == [2, 5]
        assert rec["confidence_shape"] == [2, 5]
        assert rec["loss_weight_shape"] == [2, 5]
        assert rec["utility_shape"] == [2, 5]
        assert rec["finite"] is True
    assert records[3]["has_payload"] is False
    assert records[3]["slots"] == 4


def test_crct_async_teacher_transport_matches_stored_batch() -> None:
    world_size = 4
    port = _pick_free_port_or_skip()
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _worker_async_crct_transport,
            args=(world_size, port, tmp),
            nprocs=world_size,
            join=True,
        )
        records = [
            json.loads(Path(tmp, f"rank{rank}.json").read_text())
            for rank in range(world_size)
        ]

    rank0 = records[0]
    assert len(rank0["matched"]) >= 2
    assert all(row["matched_inputs"] for row in rank0["matched"])
    assert all(row["matched_targets"] for row in rank0["matched"])
    assert all(row["finite"] for row in rank0["matched"])
    diag = rank0["diagnostics"]
    assert diag["mode"] == "async_rank0_memory_broadcast"
    assert diag["transport_group"] == "rank0_memory"
    assert diag["participant"] is True
    assert diag["coordinator_rank"] == 0
    assert diag["memory_rank"] == 3
    assert diag["requests_started"] == 6
    assert diag["request_broadcasts_started"] == 6
    assert diag["payloads_used"] == len(rank0["matched"])
    assert diag["sentinels_received"] >= 1
    assert diag["payload_lag_steps_max"] >= 1
    assert diag["orphan_payloads_dropped"] == 0
    assert diag["stale_payloads_dropped"] == 0
    assert diag["errors"] == 0

    for rec in records[1:3]:
        assert rec["matched"] == []
        assert rec["diagnostics"]["participant"] is False
        assert rec["diagnostics"]["requests_started"] == 0

    memory_diag = records[3]["diagnostics"]
    assert memory_diag["payloads_scored"] >= 3
    assert memory_diag["payloads_sent"] >= 2
    assert records[3]["slots"] > 0


def test_crct_async_teacher_transport_wired_into_train_loop() -> None:
    world_size = 4
    port = _pick_free_port_or_skip()
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _worker_async_crct_train_loop,
            args=(world_size, port, tmp),
            nprocs=world_size,
            join=True,
        )
        crct = json.loads(Path(tmp, "rank0.json").read_text())

    assert crct["teacher_transport_mode"] == "async_rank0_memory_broadcast"
    assert crct["async_teacher_transport"] is True
    assert crct["teacher_requests"] == 5
    assert crct["teacher_payloads"] >= 1
    assert crct["teacher_fail_open"] >= 1
    assert crct["teacher_coordinator_rank"] == 0
    assert crct["teacher_memory_rank"] == 3
    assert crct["teacher_bypass_steps"] == 0
    assert crct["teacher_param_sync_interval_steps"] == 1
    assert crct["teacher_param_syncs"] >= 1
    assert crct["memory_owner"] == "packet_and_maintenance_shared"
    assert crct["trunk_memory_mode"] == "packet"
    assert crct["grad_sync_group"] == "train_ranks"
    assert crct["memory_rank_joins_grad"] is False
    assert crct["stop_sync_group"] == "train_ranks"
    assert crct["train_rank_slot_reads"] == 0
    assert crct["train_rank_slot_writes"] == 0
    assert crct["teacher_memory_slots"] > 0
    assert crct["gradient_conflict"]["enabled"] is True
    assert crct["gradient_conflict"]["calls"] >= 1
    ranks = crct["rank_diagnostics"]
    assert len(ranks) == 4
    train0 = ranks[0]["transport"]
    memory = ranks[3]["transport"]
    memory_conflict = ranks[3]["gradient_conflict"]
    assert ranks[0]["memory_slots"] == 0
    assert ranks[0]["grad_sync_group"] == "train_ranks"
    assert ranks[0]["memory_rank_joins_grad"] is False
    assert ranks[3]["stop_sync_group"] == "local"
    assert ranks[1]["memory_slots"] == 0
    assert ranks[2]["memory_slots"] == 0
    assert ranks[1]["teacher_transport_participant"] is False
    assert ranks[2]["teacher_transport_participant"] is False
    assert ranks[1]["teacher_bypass_steps"] == 5
    assert ranks[2]["teacher_bypass_steps"] == 5
    assert ranks[1]["transport"]["participant"] is False
    assert ranks[2]["transport"]["participant"] is False
    assert ranks[1]["transport"]["requests_started"] == 0
    assert ranks[2]["transport"]["requests_started"] == 0
    assert ranks[3]["memory_slots"] == crct["teacher_memory_slots"]
    assert train0["requests_started"] == 5
    assert train0["request_broadcasts_started"] == 5
    assert train0["pre_sync_waits"] >= 0
    assert train0["payloads_used"] == crct["teacher_payloads"]
    assert train0["errors"] == 0
    assert memory["payloads_scored"] >= 2
    assert memory["payloads_sent"] >= 1
    assert memory_conflict["calls"] == crct["gradient_conflict"]["calls"]
    assert memory_conflict["candidates_seen"] > 0
