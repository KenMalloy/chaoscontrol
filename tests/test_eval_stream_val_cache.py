from __future__ import annotations

import json
from pathlib import Path

import pytest
import sentencepiece as spm

from chaoscontrol.eval_stream.doc_stream import DocStreamer
from chaoscontrol.eval_stream.val_cache import load_val_cache, write_val_cache


@pytest.fixture
def sp_model(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join([
        "the quick brown fox jumps over the lazy dog",
        "sphinx of black quartz judge my vow",
        "pack my box with five dozen liquor jugs",
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


def test_val_cache_matches_doc_stream_tokens_and_metadata(tmp_path: Path, sp_model: Path) -> None:
    jsonl = tmp_path / "docs.jsonl"
    texts = [
        "alpha beta gamma",
        "delta epsilon",
        "unicode cafe \u2603",
        "zeta eta theta",
    ]
    _write_jsonl(jsonl, texts)
    cache_dir = tmp_path / "cache"

    manifest = write_val_cache(
        jsonl_paths=[jsonl],
        sp_model_path=sp_model,
        cache_dir=cache_dir,
        max_docs=3,
    )
    cache = load_val_cache(cache_dir)
    expected = list(DocStreamer(jsonl_paths=[jsonl], sp_model_path=sp_model, max_docs=3))

    assert manifest["num_docs"] == 3
    assert cache.num_docs == 3
    assert cache.total_tokens == sum(len(doc.tokens) for doc in expected)
    assert cache.total_raw_bytes == sum(doc.raw_bytes for doc in expected)
    assert cache.manifest["schema_version"] == 1
    assert cache.manifest["max_docs"] == 3

    offset = 0
    for cached, streamed in zip(cache.iter_docs(), expected):
        assert cached.doc_id == streamed.doc_id
        assert cached.token_start == offset
        assert cached.token_len == len(streamed.tokens)
        assert cached.raw_bytes == streamed.raw_bytes
        assert cache.tokens_for_doc(cached).tolist() == streamed.tokens
        offset += cached.token_len


def test_val_cache_doc_ids_continue_across_files(tmp_path: Path, sp_model: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, ["one", "two"])
    _write_jsonl(second, ["three"])
    cache_dir = tmp_path / "cache"

    write_val_cache(
        jsonl_paths=[first, second],
        sp_model_path=sp_model,
        cache_dir=cache_dir,
        max_docs=50_000,
    )
    cache = load_val_cache(cache_dir)

    assert [doc.doc_id for doc in cache.iter_docs()] == [0, 1, 2]


def test_val_cache_rejects_stale_manifest_without_force(tmp_path: Path, sp_model: Path) -> None:
    jsonl = tmp_path / "docs.jsonl"
    _write_jsonl(jsonl, ["one", "two", "three"])
    cache_dir = tmp_path / "cache"

    write_val_cache(
        jsonl_paths=[jsonl],
        sp_model_path=sp_model,
        cache_dir=cache_dir,
        max_docs=2,
    )

    with pytest.raises(ValueError, match="does not match requested inputs"):
        write_val_cache(
            jsonl_paths=[jsonl],
            sp_model_path=sp_model,
            cache_dir=cache_dir,
            max_docs=3,
        )


def test_val_cache_manifest_uses_source_stat_not_full_jsonl_hash(tmp_path: Path, sp_model: Path) -> None:
    jsonl = tmp_path / "docs.jsonl"
    _write_jsonl(jsonl, ["one", "two"])
    cache_dir = tmp_path / "cache"

    manifest = write_val_cache(
        jsonl_paths=[jsonl],
        sp_model_path=sp_model,
        cache_dir=cache_dir,
        max_docs=2,
    )

    request = manifest["request"]
    assert "jsonl_sha256" not in request
    assert request["jsonl_stats"][0]["path"] == str(jsonl)
    assert request["jsonl_stats"][0]["size_bytes"] == jsonl.stat().st_size
    assert manifest["cache_content_sha256"]
