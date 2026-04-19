import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from chaoscontrol.eval_stream.temporal_heads import TemporalHeadChunkResult


RAW_DOC_TEXTS = ("hello world this is a doc", "another small doc")


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_exp22_temporal_heads.py"
    spec = importlib.util.spec_from_file_location("run_exp22_temporal_heads_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_fixture(tmp_path, *, condition, horizon_shifts):
    import sentencepiece as spm
    from chaoscontrol.model import ChaosStudentLM

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["alpha beta gamma", "delta epsilon"] * 50))
    sp_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(sp_prefix),
        vocab_size=64,
        character_coverage=1.0,
        model_type="bpe",
    )

    jsonl = tmp_path / "docs.jsonl"
    with jsonl.open("w") as fh:
        for text in RAW_DOC_TEXTS:
            fh.write(json.dumps({"text": text}) + "\n")

    model = ChaosStudentLM(
        vocab_size=64,
        dim=16,
        num_layers=2,
        block_type="ssm",
        a_mode="diag",
    )
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "vocab_size": 64,
                "dim": 16,
                "num_layers": 2,
                "block_type": "ssm",
                "a_mode": "diag",
            },
        },
        ckpt_path,
    )

    out_path = tmp_path / "metrics.jsonl"
    summary_path = tmp_path / "summary.json"
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(
        json.dumps(
            {
                "condition": condition,
                "horizon_shifts": horizon_shifts,
                "chunk_size": 32,
                "max_docs": 2,
                "seed": 0,
                "jsonl_paths": [str(jsonl)],
                "sp_model_path": f"{sp_prefix}.model",
                "checkpoint_path": str(ckpt_path),
                "output_path": str(out_path),
                "summary_path": str(summary_path),
            }
        )
    )
    return cfg_path, out_path, summary_path


def _run_exp22_config(cfg_path):
    return subprocess.run(
        [sys.executable, "scripts/run_exp22_temporal_heads.py", "--config", str(cfg_path)],
        capture_output=True,
        text=True,
    )


def test_exp22_runner_writes_metrics_and_summary(tmp_path):
    cfg_path, out_path, summary_path = _write_tiny_fixture(
        tmp_path,
        condition="temporal_heads",
        horizon_shifts=[-0.5, 0.0, 0.5],
    )

    result = _run_exp22_config(cfg_path)

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(records) == 2
    assert all(record["condition"] == "temporal_heads" for record in records)
    assert all(record["horizon_shifts"] == [-0.5, 0.0, 0.5] for record in records)

    summary = json.loads(summary_path.read_text())
    expected_bpb = (
        sum(record["loss_nats"] for record in records)
        / sum(len(text.encode("utf-8")) for text in RAW_DOC_TEXTS)
        / math.log(2.0)
    )
    assert summary["condition"] == "temporal_heads"
    assert summary["evidence_label"] == "exploratory"
    assert summary["temporal_head_count"] == 3
    assert summary["horizon_shifts"] == [-0.5, 0.0, 0.5]
    assert summary["docs_scored"] == 2
    assert summary["aggregate_bpb"] == pytest.approx(expected_bpb)


def test_exp22_runner_supports_single_horizon_pilot(tmp_path):
    cfg_path, out_path, summary_path = _write_tiny_fixture(
        tmp_path,
        condition="single_horizon",
        horizon_shifts=[-0.5],
    )

    result = _run_exp22_config(cfg_path)

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(records) == 2
    assert all(record["condition"] == "single_horizon" for record in records)
    assert all(record["horizon_shifts"] == [-0.5] for record in records)
    assert all(list(record["per_head_bpb"]) == ["-0.5"] for record in records)

    summary = json.loads(summary_path.read_text())
    assert summary["condition"] == "single_horizon"
    assert summary["temporal_head_count"] == 1
    assert summary["horizon_shifts"] == [-0.5]


def test_exp22_runner_rejects_unwired_gated_condition(tmp_path):
    cfg_path, _, _ = _write_tiny_fixture(
        tmp_path,
        condition="gated_temporal_heads",
        horizon_shifts=[-0.5, 0.0, 0.5],
    )

    result = _run_exp22_config(cfg_path)

    assert result.returncode != 0
    assert "gated_temporal_heads is not wired" in result.stderr


def test_exp22_runner_rejects_unknown_config_keys(tmp_path):
    cfg_path, _, _ = _write_tiny_fixture(
        tmp_path,
        condition="temporal_heads",
        horizon_shifts=[-0.5, 0.0, 0.5],
    )
    raw = json.loads(cfg_path.read_text())
    raw["horizon_knob"] = "delta_scale"
    cfg_path.write_text(json.dumps(raw))

    result = _run_exp22_config(cfg_path)

    assert result.returncode != 0
    assert "unknown Exp 22 config key(s): horizon_knob" in result.stderr


def test_exp22_parameter_free_run_does_not_mutate_checkpoint(tmp_path):
    cfg_path, _, _ = _write_tiny_fixture(
        tmp_path,
        condition="temporal_heads",
        horizon_shifts=[-0.5, 0.0, 0.5],
    )
    cfg = json.loads(cfg_path.read_text())
    ckpt_path = tmp_path / "ckpt.pt"
    before = ckpt_path.read_bytes()

    result = _run_exp22_config(cfg_path)

    assert result.returncode == 0, result.stderr
    assert ckpt_path == tmp_path / cfg["checkpoint_path"]
    assert ckpt_path.read_bytes() == before


def test_same_horizon_virtual_depth_threads_one_direct_state_bundle(monkeypatch, tmp_path):
    runner = _load_runner_module()

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class FakeStreamer:
        def __init__(self, **_kwargs):
            pass

        def __iter__(self):
            yield SimpleNamespace(doc_id=0, tokens=[0, 1, 2, 3, 4, 5, 6], raw_bytes=7)

    seen_states = []

    def fake_build_model(_ckpt_path, _cfg):
        return FakeModel(), {"vocab_size": 8, "num_layers": 1}

    def fake_score_direct_chunk(_model, _chunk, states):
        seen_states.append(None if states is None else [state.clone() for state in states])
        next_state = torch.full((1, 1), float(len(seen_states)))
        return 1.0, [next_state]

    monkeypatch.setattr(runner, "_build_model", fake_build_model)
    monkeypatch.setattr(runner, "DocStreamer", FakeStreamer)
    monkeypatch.setattr(runner, "_score_direct_chunk", fake_score_direct_chunk)

    cfg = runner.Exp22RunConfig(
        condition="same_horizon_virtual_depth",
        checkpoint_path=str(tmp_path / "unused.pt"),
        output_path=str(tmp_path / "metrics.jsonl"),
        summary_path="",
        chunk_size=3,
    )
    runner.run(cfg, jsonl_paths=["unused.jsonl"], sp_model_path="unused.model")

    assert seen_states[0] is None
    assert torch.equal(seen_states[1][0], torch.ones(1, 1))


def test_temporal_head_budget_time_includes_all_head_work(monkeypatch, tmp_path):
    runner = _load_runner_module()

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class FakeStreamer:
        def __init__(self, **_kwargs):
            pass

        def __iter__(self):
            yield SimpleNamespace(doc_id=0, tokens=[0, 1, 2, 3, 4, 5, 6], raw_bytes=7)

    clock = {"now": 0.0}
    forwards = {"count": 0}

    def fake_monotonic():
        return clock["now"]

    def fake_build_model(_ckpt_path, _cfg):
        return FakeModel(), {"vocab_size": 8, "num_layers": 1}

    def fake_score_temporal_heads_chunk(_model, chunk, *, states_by_shift, cfg):
        for _shift in cfg.horizon_shifts:
            forwards["count"] += 1
            clock["now"] += 0.25
        return TemporalHeadChunkResult(
            loss_nats=1.0,
            tokens_scored=chunk.size(1) - 1,
            mixed_log_probs=torch.empty(0),
            final_states_by_shift={shift: [] for shift in cfg.horizon_shifts},
            per_head_loss_nats={shift: 1.0 for shift in cfg.horizon_shifts},
        )

    monkeypatch.setattr(runner.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(runner, "_build_model", fake_build_model)
    monkeypatch.setattr(runner, "DocStreamer", FakeStreamer)
    monkeypatch.setattr(runner, "score_temporal_heads_chunk", fake_score_temporal_heads_chunk)

    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.write_bytes(b"checkpoint")
    summary_path = tmp_path / "summary.json"
    cfg = runner.Exp22RunConfig(
        condition="temporal_heads",
        checkpoint_path=str(ckpt_path),
        output_path=str(tmp_path / "metrics.jsonl"),
        summary_path=str(summary_path),
        chunk_size=3,
        horizon_shifts=(-0.5, 0.0, 0.5),
        budget_seconds=999.0,
    )
    runner.run(cfg, jsonl_paths=["unused.jsonl"], sp_model_path="unused.model")

    summary = json.loads(summary_path.read_text())
    assert forwards["count"] == 6
    assert summary["score_wall_seconds"] == pytest.approx(1.5)
