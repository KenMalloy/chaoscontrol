import subprocess
import sys
import json
import numpy as np
from pathlib import Path
import torch


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
