from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import sentencepiece as spm

from chaoscontrol.eval_stream.val_cache import load_val_cache


@pytest.fixture
def sp_model(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join([
        "the model scores validation documents quickly",
        "sentencepiece encodes repeated fineweb-like text",
        "cache builders should be deterministic and idempotent",
    ] * 80))
    prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=64,
        character_coverage=1.0,
        model_type="bpe",
    )
    return Path(f"{prefix}.model")


def _write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for text in texts:
            fh.write(json.dumps({"text": text}) + "\n")


def test_build_exp20_val_cache_cli_writes_loadable_cache(tmp_path: Path, sp_model: Path) -> None:
    jsonl = tmp_path / "docs_selected.jsonl"
    _write_jsonl(jsonl, ["alpha beta", "gamma delta", "epsilon zeta"])
    cache_dir = tmp_path / "cache"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_exp20_val_cache.py",
            "--jsonl-path",
            str(jsonl),
            "--sp-model-path",
            str(sp_model),
            "--cache-dir",
            str(cache_dir),
            "--max-docs",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "num_docs=2" in result.stdout
    assert (cache_dir / "manifest.json").is_file()
    assert (cache_dir / "tokens.npy").is_file()
    assert (cache_dir / "docs.npy").is_file()
    cache = load_val_cache(cache_dir)
    assert cache.num_docs == 2


def test_build_exp20_val_cache_cli_skips_matching_cache(tmp_path: Path, sp_model: Path) -> None:
    jsonl = tmp_path / "docs_selected.jsonl"
    _write_jsonl(jsonl, ["alpha beta", "gamma delta"])
    cache_dir = tmp_path / "cache"
    cmd = [
        sys.executable,
        "scripts/build_exp20_val_cache.py",
        "--jsonl-path",
        str(jsonl),
        "--sp-model-path",
        str(sp_model),
        "--cache-dir",
        str(cache_dir),
        "--max-docs",
        "2",
    ]

    first = subprocess.run(cmd, capture_output=True, text=True)
    second = subprocess.run(cmd, capture_output=True, text=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "cache_status=existing" in second.stdout
