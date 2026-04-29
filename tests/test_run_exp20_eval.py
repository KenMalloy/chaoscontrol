import subprocess
import sys
import json
import numpy as np
from pathlib import Path
import pytest
import torch

from scripts.run_exp20_eval import _build_model, _build_optimizer


def test_build_optimizer_uses_fused_muon_for_eval_ttt():
    param = torch.nn.Parameter(torch.randn(4, 4))
    optimizers = _build_optimizer([param], lr=0.016)

    assert len(optimizers) == 1
    assert getattr(optimizers[0], "_fused") is True


def test_eval_loader_rejects_online_replay_eviction_checkpoint(tmp_path):
    from chaoscontrol.model import ChaosStudentLM

    model = ChaosStudentLM(
        vocab_size=32,
        dim=8,
        num_layers=1,
        block_type="ssm",
        a_mode="diag",
    )
    ckpt_path = tmp_path / "online.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "vocab_size": 32,
                "dim": 8,
                "num_layers": 1,
                "block_type": "ssm",
                "a_mode": "diag",
                "replay_eviction_enabled": True,
            },
            "online_eval_state": {"replay_eviction": {"schema_version": 1}},
        },
        ckpt_path,
    )

    with pytest.raises(RuntimeError, match="CPU control plane"):
        _build_model(ckpt_path)


def test_driver_runs_tiny_stream(tmp_path):
    # Tiny SP model + JSONL doc file — matches the DocStreamer retrofit
    # (JSONL + on-the-fly SP tokenization).
    import sentencepiece as spm
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["alpha beta gamma", "delta epsilon"] * 50))
    sp_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(sp_prefix),
        vocab_size=64, character_coverage=1.0, model_type="bpe",
    )
    sp_model_path = f"{sp_prefix}.model"

    jsonl = tmp_path / "docs.jsonl"
    with jsonl.open("w") as fh:
        for t in ["hello world this is a doc", "another small doc", "and a third"]:
            fh.write(json.dumps({"text": t}) + "\n")

    # Tiny checkpoint — vocab_size must match the SP model's piece count.
    from chaoscontrol.model import ChaosStudentLM
    m = ChaosStudentLM(vocab_size=64, dim=16, num_layers=2, block_type="ssm", a_mode="diag")
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(),
                "config": {"vocab_size": 64, "dim": 16, "num_layers": 2,
                           "block_type": "ssm", "a_mode": "diag"}}, ckpt_path)

    out_path = tmp_path / "metrics.jsonl"
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps({
        "adapt_set": "none", "persistence_mode": "reset",
        "chunk_size": 32, "steps_per_chunk": 0,
        "max_docs": 3, "seed": 0,
        "jsonl_paths": [str(jsonl)],
        "sp_model_path": sp_model_path,
        "checkpoint_path": str(ckpt_path),
        "output_path": str(out_path),
    }))
    result = subprocess.run(
        [sys.executable, "scripts/run_exp20_eval.py", "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 3  # 3 docs


def test_driver_writes_score_floor_summary_for_no_ttt_run(tmp_path):
    import sentencepiece as spm
    from chaoscontrol.model import ChaosStudentLM

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["alpha beta gamma", "delta epsilon"] * 50))
    sp_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(sp_prefix),
        vocab_size=64, character_coverage=1.0, model_type="bpe",
    )
    jsonl = tmp_path / "docs.jsonl"
    with jsonl.open("w") as fh:
        fh.write(json.dumps({"text": "hello world this is a doc"}) + "\n")
        fh.write(json.dumps({"text": "another small doc"}) + "\n")

    m = ChaosStudentLM(vocab_size=64, dim=16, num_layers=2, block_type="ssm", a_mode="diag")
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(),
                "config": {"vocab_size": 64, "dim": 16, "num_layers": 2,
                           "block_type": "ssm", "a_mode": "diag"}}, ckpt_path)

    out_path = tmp_path / "metrics.jsonl"
    summary_path = tmp_path / "summary.json"
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps({
        "adapt_set": "none", "persistence_mode": "reset",
        "chunk_size": 32, "steps_per_chunk": 0,
        "max_docs": 2, "seed": 0, "budget_seconds": 600.0,
        "jsonl_paths": [str(jsonl)],
        "sp_model_path": f"{sp_prefix}.model",
        "checkpoint_path": str(ckpt_path),
        "output_path": str(out_path),
        "summary_path": str(summary_path),
        "safety_margin_seconds": 30.0,
    }))

    result = subprocess.run(
        [sys.executable, "scripts/run_exp20_eval.py", "--config", str(cfg_path)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(summary_path.read_text())
    assert summary["score_only_mode"] is True
    assert summary["score_floor_seconds"] == summary["elapsed_seconds"]
    assert summary["score_wall_seconds"] <= summary["score_floor_seconds"]
    assert summary["usable_ttt_budget_seconds"] <= 570.0
    assert summary["docs_scored"] == 2
    assert summary["adapt_steps"] == 0


def test_driver_skips_weight_ttt_when_floor_leaves_no_slack(tmp_path):
    import sentencepiece as spm
    from chaoscontrol.model import ChaosStudentLM

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["alpha beta gamma", "delta epsilon"] * 50))
    sp_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus), model_prefix=str(sp_prefix),
        vocab_size=64, character_coverage=1.0, model_type="bpe",
    )
    jsonl = tmp_path / "docs.jsonl"
    with jsonl.open("w") as fh:
        fh.write(json.dumps({"text": "hello world this is a doc"}) + "\n")
        fh.write(json.dumps({"text": "another small doc"}) + "\n")

    m = ChaosStudentLM(vocab_size=64, dim=16, num_layers=2, block_type="ssm", a_mode="diag")
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(),
                "config": {"vocab_size": 64, "dim": 16, "num_layers": 2,
                           "block_type": "ssm", "a_mode": "diag"}}, ckpt_path)

    out_path = tmp_path / "metrics.jsonl"
    summary_path = tmp_path / "summary.json"
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps({
        "adapt_set": "lm_head", "persistence_mode": "reset",
        "chunk_size": 32, "steps_per_chunk": 1,
        "max_docs": 2, "seed": 0, "budget_seconds": 600.0,
        "score_floor_seconds": 600.0,
        "jsonl_paths": [str(jsonl)],
        "sp_model_path": f"{sp_prefix}.model",
        "checkpoint_path": str(ckpt_path),
        "output_path": str(out_path),
        "summary_path": str(summary_path),
    }))

    result = subprocess.run(
        [sys.executable, "scripts/run_exp20_eval.py", "--config", str(cfg_path)],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert records
    assert all(rec["step_count"] == 0 for rec in records)
    summary = json.loads(summary_path.read_text())
    assert summary["usable_ttt_budget_seconds"] == 0.0
    assert summary["adapt_steps"] == 0
