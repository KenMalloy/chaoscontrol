import json
import subprocess
import sys

import torch


def test_exp22_runner_writes_metrics_and_summary(tmp_path):
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
        fh.write(json.dumps({"text": "hello world this is a doc"}) + "\n")
        fh.write(json.dumps({"text": "another small doc"}) + "\n")

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
                "condition": "temporal_heads",
                "horizon_shifts": [-0.5, 0.0, 0.5],
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

    result = subprocess.run(
        [sys.executable, "scripts/run_exp22_temporal_heads.py", "--config", str(cfg_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(records) == 2
    assert all(record["condition"] == "temporal_heads" for record in records)
    assert all(record["horizon_shifts"] == [-0.5, 0.0, 0.5] for record in records)

    summary = json.loads(summary_path.read_text())
    assert summary["condition"] == "temporal_heads"
    assert summary["evidence_label"] == "exploratory"
    assert summary["temporal_head_count"] == 3
    assert summary["horizon_shifts"] == [-0.5, 0.0, 0.5]
    assert summary["docs_scored"] == 2
