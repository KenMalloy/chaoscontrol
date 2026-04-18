import json
from pathlib import Path

import pytest
import sentencepiece as spm

from chaoscontrol.eval_stream.doc_stream import DocStreamer


@pytest.fixture
def sp_model(tmp_path: Path) -> Path:
    """Train a tiny SP model for hermetic testing."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join([
        "the quick brown fox jumps over the lazy dog",
        "sphinx of black quartz judge my vow",
        "pack my box with five dozen liquor jugs",
    ] * 50))
    model_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=64,
        character_coverage=1.0,
        model_type="bpe",
    )
    return Path(f"{model_prefix}.model")


def _write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t}) + "\n")


def test_iterates_docs_in_order(tmp_path, sp_model):
    jsonl = tmp_path / "docs.jsonl"
    _write_jsonl(jsonl, ["alpha beta gamma", "delta epsilon", "zeta"])
    docs = list(DocStreamer(jsonl_paths=[jsonl], sp_model_path=sp_model, max_docs=10))
    assert len(docs) == 3
    assert [d.doc_id for d in docs] == [0, 1, 2]
    assert all(len(d.tokens) > 0 for d in docs)


def test_respects_max_docs(tmp_path, sp_model):
    jsonl = tmp_path / "docs.jsonl"
    _write_jsonl(jsonl, [f"doc number {i}" for i in range(8)])
    docs = list(DocStreamer(jsonl_paths=[jsonl], sp_model_path=sp_model, max_docs=2))
    assert len(docs) == 2


def test_raw_bytes_equal_utf8_length(tmp_path, sp_model):
    jsonl = tmp_path / "docs.jsonl"
    text = "hello"
    _write_jsonl(jsonl, [text])
    docs = list(DocStreamer(jsonl_paths=[jsonl], sp_model_path=sp_model, max_docs=1))
    assert docs[0].raw_bytes == len(text.encode("utf-8"))


def test_doc_id_continues_across_jsonl_files(tmp_path, sp_model):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(a, ["one", "two"])
    _write_jsonl(b, ["three"])
    docs = list(DocStreamer(jsonl_paths=[a, b], sp_model_path=sp_model, max_docs=10))
    assert [d.doc_id for d in docs] == [0, 1, 2]


def test_empty_text_is_skipped(tmp_path, sp_model):
    jsonl = tmp_path / "docs.jsonl"
    _write_jsonl(jsonl, ["", "real doc"])
    docs = list(DocStreamer(jsonl_paths=[jsonl], sp_model_path=sp_model, max_docs=10))
    assert len(docs) == 1
    assert docs[0].doc_id == 0
