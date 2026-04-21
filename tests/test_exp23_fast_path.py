"""Tests for Exp 23 fastest-path training helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from chaoscontrol.data import batch_from_starts
from chaoscontrol import train_ssm


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "experiments" / "23_fast_path" / "fast_path.py"
RUNNER_PATH = REPO / "experiments" / "23_fast_path" / "runner_fast_path.py"
LAUNCH_PATH = REPO / "experiments" / "23_fast_path" / "launch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("exp23_fast_path", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("exp23_runner_fast_path", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _TinyTrainStepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4, bias=False)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, 6, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)


class _TinyTokenTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(6, 4)
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(4, 6, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embed(inputs)


def _load_launch_module():
    spec = importlib.util.spec_from_file_location("exp23_launch", LAUNCH_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vectorized_batch_matches_reference_batcher():
    mod = _load_module()
    tokens = torch.arange(40, dtype=torch.int16)
    starts = torch.tensor([0, 5, 13, 21], dtype=torch.long)

    got_inputs, got_targets = mod.batch_from_start_tensor(
        tokens=tokens,
        starts=starts,
        seq_len=6,
        device=torch.device("cpu"),
    )
    ref_inputs, ref_targets = batch_from_starts(
        tokens=tokens,
        starts=[int(x) for x in starts.tolist()],
        seq_len=6,
        device=torch.device("cpu"),
    )

    assert torch.equal(got_inputs, ref_inputs)
    assert torch.equal(got_targets, ref_targets)
    assert got_inputs.dtype == torch.long
    assert got_targets.dtype == torch.long


def test_vectorized_batch_can_clamp_header_contamination():
    mod = _load_module()
    tokens = torch.tensor([-7, 1, 2, 999, 4, 5, 6, 7], dtype=torch.int16)
    starts = torch.tensor([0, 3], dtype=torch.long)

    got_inputs, got_targets = mod.batch_from_start_tensor(
        tokens=tokens,
        starts=starts,
        seq_len=3,
        device=torch.device("cpu"),
        vocab_size=8,
    )

    assert got_inputs.tolist() == [[0, 1, 2], [7, 4, 5]]
    assert got_targets.tolist() == [[1, 2, 7], [4, 5, 6]]


def test_lazy_lm_start_sampling_matches_sharded_range():
    mod = _load_module()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)

    starts = mod.sample_sharded_lm_starts(
        num_tokens=10_000,
        seq_len=32,
        stride=16,
        batch_size=128,
        rank=2,
        world_size=4,
        generator=generator,
    )

    total_starts = len(range(0, 10_000 - 32 - 1, 16))
    valid_rank_starts = {
        global_idx * 16
        for global_idx in range(total_starts)
        if global_idx % 4 == 2
    }
    assert starts.shape == (128,)
    assert starts.dtype == torch.long
    assert all(int(start) in valid_rank_starts for start in starts.tolist())


def test_lazy_eval_start_selection_matches_eager_shape():
    mod = _load_module()

    starts = mod.choose_lm_starts_lazy(
        num_tokens=10_000,
        seq_len=32,
        stride=16,
        batch_size=7,
        eval_batches=3,
        seed=123,
    )

    assert len(starts) == 21
    assert len(set(starts)) == 21
    assert all(start % 16 == 0 for start in starts)
    assert all(0 <= start < 10_000 - 32 - 1 for start in starts)


def test_stage_a_matrix_names_and_fast_defaults():
    mod = _load_module()
    entries = mod.build_stage_a_matrix(
        seeds=[1337],
        vocab_sizes=[16384],
        batch_sizes=[1024, 2048],
        chunk_sizes=[64, 256],
        activation_checkpoints=[True, False],
        world_size=8,
    )

    assert [entry["name"] for entry in entries] == [
        "stageA_v16384_b1024_c64_ckpt_s1337",
        "stageA_v16384_b1024_c64_nockpt_s1337",
        "stageA_v16384_b1024_c256_ckpt_s1337",
        "stageA_v16384_b1024_c256_nockpt_s1337",
        "stageA_v16384_b2048_c64_ckpt_s1337",
        "stageA_v16384_b2048_c64_nockpt_s1337",
        "stageA_v16384_b2048_c256_ckpt_s1337",
        "stageA_v16384_b2048_c256_nockpt_s1337",
    ]
    for entry in entries:
        assert entry["mode"] == "speed_sweep"
        assert entry["world_size"] == 8
        assert entry["model_type"] == "ssm"
        assert entry["precision"] == "bf16"
        assert entry["fused_grad_clip"] is True
        assert entry["fused_muon"] is True
        assert entry["compile_full_path"] is False
        assert entry["lm_head_backward_mode"] == "single"
        assert entry["prefetch_batches"] is True
        assert entry["eval_batches"] == 0
        assert entry["warmup_steps"] == 5
        assert entry["stop_check_interval"] == 4


def test_stage_b_matrix_crosses_vocab_and_embedding_inits():
    mod = _load_module()
    speed_cfg = {
        "batch_size": 2048,
        "chunk_size": 256,
        "activation_checkpoint": True,
        "base_lr": 0.2,
        "eval_batches": 0,
    }
    init_paths = {
        8192: {
            "meanstd": "artifacts/v8192_meanstd.pt",
            "fullcov": "artifacts/v8192_fullcov.pt",
        },
        16384: {
            "meanstd": "artifacts/v16384_meanstd.pt",
            "fullcov": "artifacts/v16384_fullcov.pt",
        },
    }

    entries = mod.build_stage_b_matrix(
        speed_config=speed_cfg,
        seeds=[1337, 2674],
        vocab_sizes=[8192, 16384],
        init_paths=init_paths,
        world_size=8,
    )

    assert len(entries) == 12  # 2 vocabs x 3 init arms x 2 seeds
    names = [entry["name"] for entry in entries]
    assert "stageB_v8192_random_s1337" in names
    assert "stageB_v16384_meanstd_s2674" in names
    assert "stageB_v16384_fullcov_s1337" in names

    random_entries = [entry for entry in entries if entry["embed_init"] == "random"]
    assert random_entries
    assert all(entry.get("embed_init_path") is None for entry in random_entries)

    meanstd_16384 = next(
        entry for entry in entries
        if entry["vocab_size"] == 16384
        and entry["embed_init"] == "meanstd"
        and entry["seed"] == 2674
    )
    assert meanstd_16384["embed_init_path"] == "artifacts/v16384_meanstd.pt"
    assert meanstd_16384["batch_size"] == 2048
    assert meanstd_16384["chunk_size"] == 256
    assert meanstd_16384["activation_checkpoint"] is True
    assert meanstd_16384["prefetch_batches"] is True
    assert meanstd_16384["eval_batches"] == 16
    assert meanstd_16384["budget_seconds"] == 600.0


def test_token_accounting_summary_uses_global_tokens():
    mod = _load_module()
    summary = mod.summarize_train_timing(
        steps=10,
        elapsed_s=2.0,
        batch_size=4,
        seq_len=8,
        world_size=3,
    )

    assert summary["tokens_per_step"] == 96
    assert summary["aggregate_tokens_per_sec"] == 480.0
    assert summary["per_gpu_tokens_per_sec"] == 160.0


def test_run_train_step_uses_compiled_encoder_when_enabled(monkeypatch):
    mod = _load_runner_module()
    if hasattr(train_ssm._compiled_step_fn, "cache_clear"):
        train_ssm._compiled_step_fn.cache_clear()

    compile_calls: list[tuple[str, bool, bool]] = []

    def fake_compile(fn, *, fullgraph, dynamic):
        compile_calls.append((fn.__name__, fullgraph, dynamic))

        def compiled(model, inputs):
            compile_calls.append(("compiled", True, True))
            return fn(model, inputs)

        return compiled

    monkeypatch.setattr(mod.torch, "compile", fake_compile)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    try:
        loss = mod._run_train_step(
            model=model,
            inputs=inputs,
            targets=targets,
            chunk_size=2,
            precision="bf16",
            ddp_active=False,
            world_size=1,
            compile_full_path=True,
        )

        assert loss.ndim == 0
        assert compile_calls == [
            ("_encode_only", True, False),
            ("compiled", True, True),
        ]
    finally:
        if hasattr(train_ssm._compiled_step_fn, "cache_clear"):
            train_ssm._compiled_step_fn.cache_clear()


def test_run_train_step_leaves_encoder_eager_when_compile_disabled(monkeypatch):
    mod = _load_runner_module()

    compile_calls: list[tuple[str, bool, bool]] = []

    def fake_compile(fn, *, fullgraph, dynamic):
        compile_calls.append((fn.__name__, fullgraph, dynamic))
        return fn

    monkeypatch.setattr(mod.torch, "compile", fake_compile)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        compile_full_path=False,
    )

    assert loss.ndim == 0
    assert compile_calls == []


def test_run_train_step_can_use_single_backward(monkeypatch):
    mod = _load_runner_module()

    chunked_calls = 0

    def fail_if_chunked(**kwargs):
        nonlocal chunked_calls
        chunked_calls += 1
        raise AssertionError("single-backward mode must not call chunked CE")

    monkeypatch.setattr(mod, "chunked_lm_head_backward", fail_if_chunked)

    model = _TinyTrainStepModel()
    inputs = torch.randn(2, 3, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)

    loss = mod._run_train_step(
        model=model,
        inputs=inputs,
        targets=targets,
        chunk_size=2,
        precision="bf16",
        ddp_active=False,
        world_size=1,
        compile_full_path=False,
        lm_head_backward_mode="single",
    )

    assert loss.ndim == 0
    assert chunked_calls == 0
    assert model.encoder.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_train_fast_for_budget_can_use_batch_prefetcher(monkeypatch):
    mod = _load_runner_module()
    instances = []

    class FakePrefetcher:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.prime_calls = 0
            self.next_calls = 0
            self.close_calls = 0
            instances.append(self)

        def prime(self):
            self.prime_calls += 1
            return self

        def next(self):
            self.next_calls += 1
            inputs = torch.zeros(2, 3, dtype=torch.long)
            targets = torch.zeros(2, 3, dtype=torch.long)
            return inputs, targets

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(mod, "Exp23BatchPrefetcher", FakePrefetcher)

    model = _TinyTokenTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = mod.train_fast_for_budget(
        model,
        train_tokens=torch.arange(128, dtype=torch.int16),
        train_num_tokens=128,
        stride=4,
        seq_len=3,
        batch_size=2,
        device=torch.device("cpu"),
        optimizer=optimizer,
        budget_seconds=300.0,
        chunk_size=2,
        grad_clip_norm=0.0,
        fused_grad_clip=False,
        rank=0,
        world_size=1,
        seed=123,
        precision="bf16",
        stop_check_interval=1,
        stop_margin_seconds=0.0,
        vocab_size=6,
        max_steps=2,
        prefetch_batches=True,
    )

    assert result["steps"] == 2
    assert len(instances) == 1
    prefetcher = instances[0]
    assert prefetcher.prime_calls == 1
    assert prefetcher.next_calls == 2
    assert prefetcher.close_calls == 1
    assert prefetcher.kwargs["seq_len"] == 3
    assert prefetcher.kwargs["batch_size"] == 2
    assert prefetcher.kwargs["vocab_size"] == 6


def test_training_stop_predicate_can_stop_on_fixed_warmup_steps():
    mod = _load_module()

    assert not mod.should_stop_training_loop(
        steps=0,
        elapsed_s=999.0,
        budget_seconds=300.0,
        stop_margin_seconds=0.0,
        max_steps=5,
    )
    assert not mod.should_stop_training_loop(
        steps=4,
        elapsed_s=0.0,
        budget_seconds=300.0,
        stop_margin_seconds=0.0,
        max_steps=5,
    )
    assert mod.should_stop_training_loop(
        steps=5,
        elapsed_s=0.0,
        budget_seconds=300.0,
        stop_margin_seconds=0.0,
        max_steps=5,
    )
    assert mod.should_stop_training_loop(
        steps=1,
        elapsed_s=298.0,
        budget_seconds=300.0,
        stop_margin_seconds=2.0,
        max_steps=None,
    )


def test_read_speed_config_accepts_yaml(tmp_path):
    mod = _load_module()
    cfg = tmp_path / "speed.yaml"
    cfg.write_text("batch_size: 2048\nchunk_size: 256\n")

    assert mod.read_speed_config(cfg) == {
        "batch_size": 2048,
        "chunk_size": 256,
    }


def test_torchrun_command_uses_all_requested_gpus(tmp_path):
    mod = _load_launch_module()
    cfg = tmp_path / "cfg.yaml"
    out = tmp_path / "out.json"
    cmd = mod.build_torchrun_cmd(
        runner_path=Path("experiments/23_fast_path/runner_fast_path.py"),
        config_path=cfg,
        data_path="/data/fineweb",
        sp_model_path="/data/sp16384.model",
        output_json=out,
        world_size=8,
        rdzv_port=23456,
    )

    assert cmd[:3] == [mod.sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=8" in cmd
    assert "--rdzv-endpoint=localhost:23456" in cmd
    assert "--config" in cmd and str(cfg) in cmd
    assert "--output-json" in cmd and str(out) in cmd


def test_summarize_results_ranks_successes_and_records_errors(tmp_path):
    mod = _load_launch_module()
    results = tmp_path / "results"
    results.mkdir()
    (results / "slow.json").write_text(
        '{"config":{"name":"slow"},"train":{"aggregate_tokens_per_sec":10.0}}'
    )
    (results / "fast.json").write_text(
        '{"config":{"name":"fast"},"train":{"aggregate_tokens_per_sec":20.0}}'
    )
    (results / "bad.json").write_text(
        '{"config":{"name":"bad"},"error":"oom"}'
    )
    (results / "matrix.json").write_text('[{"name":"not-a-result"}]')

    summary = mod.summarize_result_dir(results)

    assert [row["name"] for row in summary["ranked"]] == ["fast", "slow"]
    assert summary["errors"] == [{"name": "bad", "error": "oom"}]
