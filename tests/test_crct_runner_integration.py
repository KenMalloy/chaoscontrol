"""CRCT integration coverage for the Exp23/24 fast runner.

These tests pin the seams that can silently invalidate CRCT:

* Exp24's imported model builder must stop hardcoding a bare SSM when
  ``crct_enabled=True``.
* The fast runner must compose weighted LM CE plus controller BCE from a
  dense teacher payload.
* The oracle path must append real memory so ``force_on`` can differ from
  ``off`` after one scored batch.
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.wake_cache_txn import TransactionalWakeCache


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"
RUNNER21_PATH = REPO / "experiments" / "21_sgns_tokenizer" / "runner_exp21.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_crct_model() -> ChaosStudentLM:
    torch.manual_seed(123)
    model = ChaosStudentLM(
        vocab_size=32,
        dim=8,
        num_layers=1,
        ff_mult=2,
        a_mode="diag",
        outer_model_dim=4,
        outer_model_type="multislot",
        outer_max_slots=64,
        buffer_mode="append_only",
        enable_controller=True,
        controller_hidden_dim=4,
    )
    model.train()
    return model


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
        all_group = dist.new_group(list(range(world_size)))
        transport = mod._CrctAsyncTeacherTransport(
            rank=rank,
            world_size=world_size,
            all_group=all_group,
            payload_shape=(world_size - 1, 2, 5),
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
            if rank < world_size - 1:
                expected[step] = (inputs.clone(), targets.clone())
            ready = transport.begin_step(inputs=inputs, targets=targets, step=step)

            # Make the unit test deterministic without changing production
            # semantics: production polls; this test waits so the next step can
            # reap the previous broadcast and prove the batch join is exact.
            for slot in list(transport.pending_result_broadcasts):
                for work in slot["works"]:
                    work.wait()
            for slot in list(transport.pending_input_gathers):
                slot["work"].wait()

            if ready is not None and rank < world_size - 1:
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

        transport.close()
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "rank": int(rank),
                    "matched": matched,
                    "slots": int(len(model.outer_model._slots)),
                    "diagnostics": transport.diagnostics(),
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
            crct_memory_write_tokens_per_step=4,
        )
        if rank == 0:
            Path(result_dir, "rank0.json").write_text(
                json.dumps(result["mechanisms"]["crct"])
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_exp24_model_builder_threads_crct_memory_and_controller() -> None:
    mod = _load_module("runner_exp21_crct_test", RUNNER21_PATH)
    cfg = {
        "vocab_size": 32,
        "model_dim": 8,
        "num_layers": 1,
        "ff_mult": 2,
        "crct_enabled": True,
        "outer_model_dim": 4,
        "outer_max_slots": 128,
        "controller_hidden_dim": 4,
    }

    model = mod.build_model(cfg, torch.device("cpu"), torch.float32)

    assert model.memory_controller is not None
    assert model.outer_model is not None
    assert model.outer_model.max_slots == 128
    assert model.buffer_mode == "append_only"


def test_crct_train_step_uses_payload_and_trains_controller() -> None:
    mod = _load_module("runner_fast_path_crct_train_step", RUNNER_PATH)
    model = _tiny_crct_model()
    inputs = torch.randint(0, 32, (2, 5), dtype=torch.int32)
    targets = torch.randint(0, 32, (2, 5), dtype=torch.long)
    payload = {
        "target": torch.full((2, 5), 0.85),
        "confidence": torch.ones(2, 5),
        "loss_weight": torch.linspace(0.9, 1.1, 10).reshape(2, 5),
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
        crct_lambda_controller=0.1,
    )

    assert torch.isfinite(loss)
    assert model.memory_controller is not None
    grad = model.memory_controller.net[0].weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0
    assert len(model.outer_model._slots) == inputs.numel()


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


def test_crct_fail_open_keeps_controller_grad_zero_but_present() -> None:
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

    assert model.memory_controller is not None
    grad = model.memory_controller.net[0].weight.grad
    assert grad is not None
    assert torch.equal(grad, torch.zeros_like(grad))


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
    assert "memory_controller.net.0.weight" not in matrix_names
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

    for rec in records[:3]:
        assert len(rec["matched"]) >= 2
        assert all(row["matched_inputs"] for row in rec["matched"])
        assert all(row["matched_targets"] for row in rec["matched"])
        assert all(row["finite"] for row in rec["matched"])
        diag = rec["diagnostics"]
        assert diag["mode"] == "async_allgather_broadcast"
        assert diag["requests_started"] == 6
        assert diag["payloads_used"] == len(rec["matched"])
        assert diag["sentinels_received"] >= 1
        assert diag["payload_lag_steps_max"] >= 1
        assert diag["orphan_payloads_dropped"] == 0
        assert diag["stale_payloads_dropped"] == 0
        assert diag["errors"] == 0

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

    assert crct["teacher_transport_mode"] == "async_allgather_broadcast"
    assert crct["async_teacher_transport"] is True
    assert crct["teacher_requests"] == 5
    assert crct["teacher_payloads"] >= 1
    assert crct["teacher_fail_open"] >= 1
    ranks = crct["rank_diagnostics"]
    assert len(ranks) == 4
    train0 = ranks[0]["transport"]
    memory = ranks[3]["transport"]
    assert train0["requests_started"] == 5
    assert train0["payloads_used"] == crct["teacher_payloads"]
    assert train0["errors"] == 0
    assert memory["payloads_scored"] >= 2
    assert memory["payloads_sent"] >= 1
