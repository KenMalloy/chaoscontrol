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


def _write_tiny_fixture(tmp_path, *, condition, horizon_shifts, extra_config=None):
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
    raw_config = {
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
    if extra_config:
        raw_config.update(extra_config)
    cfg_path.write_text(json.dumps(raw_config))
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


def test_exp22_runner_writes_analysis_sidecar_and_mixer_metadata(tmp_path):
    analysis_path = tmp_path / "analysis.jsonl"
    cfg_path, out_path, summary_path = _write_tiny_fixture(
        tmp_path,
        condition="temporal_heads",
        horizon_shifts=[-0.5, 0.0, 0.5],
        extra_config={
            "analysis_path": str(analysis_path),
            "mixer": "base_prior_logprob",
            "mixer_weights": [0.1, 0.8, 0.1],
        },
    )

    result = _run_exp22_config(cfg_path)

    assert result.returncode == 0, result.stderr
    metric_records = [json.loads(line) for line in out_path.read_text().splitlines()]
    analysis_records = [json.loads(line) for line in analysis_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    assert summary["mixer"] == "base_prior_logprob"
    assert summary["mixer_weights"] == [0.1, 0.8, 0.1]
    assert len(analysis_records) == len(metric_records)
    first = analysis_records[0]
    assert first["analysis_only"] is True
    assert first["mixer"] == "base_prior_logprob"
    assert set(first["winner_counts_by_shift"]) == {"-0.5", "0.0", "0.5"}
    assert sum(first["winner_counts_by_shift"].values()) == first["tokens"]
    assert set(first["half_life_stats_by_shift"]) == {"-0.5", "0.0", "0.5"}
    assert "separated_fraction_vs_base" in first["half_life_stats_by_shift"]["-0.5"][0]
    assert set(first["state_divergence_by_shift"]) == {"-0.5", "0.5"}
    assert "cosine_vs_base" in first["state_divergence_by_shift"]["-0.5"][0]


def test_exp22_runner_identical_heads_uniform_matches_score_only(tmp_path):
    head_ids = ["same_a", "same_b", "same_c"]
    cfg_path, out_path, summary_path = _write_tiny_fixture(
        tmp_path,
        condition="identical_heads_uniform",
        horizon_shifts=[0.0, 0.0, 0.0],
        extra_config={"head_ids": head_ids},
    )
    raw = json.loads(cfg_path.read_text())
    score_out_path = tmp_path / "score_metrics.jsonl"
    score_summary_path = tmp_path / "score_summary.json"
    score_cfg_path = tmp_path / "score_run.json"
    score_raw = {
        **raw,
        "condition": "score_only",
        "horizon_shifts": [0.0],
        "output_path": str(score_out_path),
        "summary_path": str(score_summary_path),
    }
    score_raw.pop("head_ids")
    score_cfg_path.write_text(json.dumps(score_raw))

    identical_result = _run_exp22_config(cfg_path)
    score_result = _run_exp22_config(score_cfg_path)

    assert identical_result.returncode == 0, identical_result.stderr
    assert score_result.returncode == 0, score_result.stderr
    identical_records = [json.loads(line) for line in out_path.read_text().splitlines()]
    score_records = [json.loads(line) for line in score_out_path.read_text().splitlines()]
    assert len(identical_records) == len(score_records) == 2
    for identical, score in zip(identical_records, score_records, strict=True):
        assert identical["head_ids"] == head_ids
        assert identical["horizon_shifts"] == [0.0, 0.0, 0.0]
        assert set(identical["per_head_bpb"]) == set(head_ids)
        assert identical["bpb"] == pytest.approx(score["bpb"])
        assert identical["loss_nats"] == pytest.approx(score["loss_nats"])

    summary = json.loads(summary_path.read_text())
    assert summary["condition"] == "identical_heads_uniform"
    assert summary["head_ids"] == head_ids
    assert summary["temporal_head_count"] == 3


def test_exp22_runner_accepts_online_exp_weights_mixer(tmp_path):
    cfg_path, _, summary_path = _write_tiny_fixture(
        tmp_path,
        condition="temporal_heads",
        horizon_shifts=[-0.5, 0.0, 0.5],
        extra_config={
            "mixer": "online_exp_weights_logprob",
            "online_eta": 0.75,
            "online_initial_weights": [0.2, 0.6, 0.2],
        },
    )

    result = _run_exp22_config(cfg_path)

    assert result.returncode == 0, result.stderr
    summary = json.loads(summary_path.read_text())
    assert summary["mixer"] == "online_exp_weights_logprob"
    assert summary["online_eta"] == 0.75
    assert summary["online_initial_weights"] == [0.2, 0.6, 0.2]


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
            winner_counts_by_shift={shift: 0 for shift in cfg.horizon_shifts},
            half_life_stats_by_shift={shift: [] for shift in cfg.horizon_shifts},
            state_divergence_by_shift={},
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
