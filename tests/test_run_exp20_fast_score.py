from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import sentencepiece as spm
import torch

from chaoscontrol.eval_stream.val_cache import CachedDoc, load_val_cache, write_val_cache


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_exp20_fast_score.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_exp20_fast_score", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fast_score = _load_module()
doc_range_for_rank = fast_score.doc_range_for_rank
prepare_doc_work = fast_score.prepare_doc_work
resolve_doc_batch_size = fast_score.resolve_doc_batch_size
resolve_distributed_context = fast_score.resolve_distributed_context


@pytest.fixture
def tiny_fixture(tmp_path: Path) -> dict[str, Path]:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join([
        "alpha beta gamma delta epsilon zeta eta theta",
        "the model should score cached validation docs",
        "chunk boundaries carry state inside each document",
    ] * 80))
    sp_prefix = tmp_path / "sp"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(sp_prefix),
        vocab_size=64,
        character_coverage=1.0,
        model_type="bpe",
    )
    sp_model = Path(f"{sp_prefix}.model")

    jsonl = tmp_path / "docs.jsonl"
    docs = [
        "alpha beta gamma delta epsilon zeta eta theta",
        "the model scores cached validation documents quickly",
        "chunk boundaries must carry recurrent state within the doc",
    ]
    with jsonl.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps({"text": doc}) + "\n")

    from chaoscontrol.model import ChaosStudentLM

    model = ChaosStudentLM(
        vocab_size=64,
        dim=16,
        num_layers=2,
        block_type="ssm",
        a_mode="diag",
    )
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {
            "vocab_size": 64,
            "dim": 16,
            "num_layers": 2,
            "block_type": "ssm",
            "a_mode": "diag",
        },
    }, ckpt_path)

    cache_dir = tmp_path / "cache"
    write_val_cache(
        jsonl_paths=[jsonl],
        sp_model_path=sp_model,
        cache_dir=cache_dir,
        max_docs=3,
    )
    return {
        "jsonl": jsonl,
        "sp_model": sp_model,
        "ckpt": ckpt_path,
        "cache_dir": cache_dir,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_fast_score_matches_exp20_correctness_harness(tiny_fixture: dict[str, Path], tmp_path: Path) -> None:
    canonical_out = tmp_path / "canonical.jsonl"
    canonical_summary = tmp_path / "canonical_summary.json"
    canonical_cfg = tmp_path / "canonical_cfg.json"
    canonical_cfg.write_text(json.dumps({
        "adapt_set": "none",
        "persistence_mode": "reset",
        "chunk_size": 8,
        "steps_per_chunk": 0,
        "max_docs": 3,
        "seed": 0,
        "budget_seconds": 600.0,
        "jsonl_paths": [str(tiny_fixture["jsonl"])],
        "sp_model_path": str(tiny_fixture["sp_model"]),
        "checkpoint_path": str(tiny_fixture["ckpt"]),
        "output_path": str(canonical_out),
        "summary_path": str(canonical_summary),
    }))
    canonical = subprocess.run(
        [sys.executable, "scripts/run_exp20_eval.py", "--config", str(canonical_cfg)],
        capture_output=True,
        text=True,
    )
    assert canonical.returncode == 0, canonical.stderr

    fast_out = tmp_path / "fast.jsonl"
    fast_summary = tmp_path / "fast_summary.json"
    fast = subprocess.run(
        [
            sys.executable,
            "scripts/run_exp20_fast_score.py",
            "--cache-dir",
            str(tiny_fixture["cache_dir"]),
            "--checkpoint-path",
            str(tiny_fixture["ckpt"]),
            "--output-path",
            str(fast_out),
            "--summary-path",
            str(fast_summary),
            "--chunk-size",
            "8",
            "--device",
            "cpu",
            "--doc-batch-size",
            "2",
            "--no-score-boundary-targets",
        ],
        capture_output=True,
        text=True,
    )
    assert fast.returncode == 0, fast.stderr

    canonical_records = _read_jsonl(canonical_out)
    fast_records = _read_jsonl(fast_out)
    assert [r["doc_id"] for r in fast_records] == [r["doc_id"] for r in canonical_records]
    assert [r["tokens"] for r in fast_records] == [r["tokens"] for r in canonical_records]
    for fast_rec, canonical_rec in zip(fast_records, canonical_records):
        assert fast_rec["bpb"] == pytest.approx(canonical_rec["bpb"], rel=0.0, abs=1e-6)
        assert fast_rec["step_count"] == 0

    summary = json.loads(fast_summary.read_text())
    canonical_summary_data = json.loads(canonical_summary.read_text())
    assert summary["docs_scored"] == canonical_summary_data["docs_scored"] == 3
    assert summary["tokens_scored"] == canonical_summary_data["tokens_scored"]
    assert summary["chunks_scored"] == canonical_summary_data["chunks_scored"]
    assert summary["score_only_mode"] is True
    assert summary["requested_docs_complete"] is True
    assert summary["result_status"] == "exploratory_prefix_complete"
    assert summary["score_boundary_targets"] is False
    assert summary["aggregate_bpb"] > 0


def test_chunk_boundary_targets_match_whole_doc_score(
    tiny_fixture: dict[str, Path],
    tmp_path: Path,
) -> None:
    whole_out = tmp_path / "whole.jsonl"
    whole_summary = tmp_path / "whole_summary.json"
    whole = subprocess.run(
        [
            sys.executable,
            "scripts/run_exp20_fast_score.py",
            "--cache-dir",
            str(tiny_fixture["cache_dir"]),
            "--checkpoint-path",
            str(tiny_fixture["ckpt"]),
            "--output-path",
            str(whole_out),
            "--summary-path",
            str(whole_summary),
            "--chunk-size",
            "-1",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert whole.returncode == 0, whole.stderr

    chunked_out = tmp_path / "chunked.jsonl"
    chunked_summary = tmp_path / "chunked_summary.json"
    chunked = subprocess.run(
        [
            sys.executable,
            "scripts/run_exp20_fast_score.py",
            "--cache-dir",
            str(tiny_fixture["cache_dir"]),
            "--checkpoint-path",
            str(tiny_fixture["ckpt"]),
            "--output-path",
            str(chunked_out),
            "--summary-path",
            str(chunked_summary),
            "--chunk-size",
            "8",
            "--device",
            "cpu",
            "--doc-batch-size",
            "3",
        ],
        capture_output=True,
        text=True,
    )
    assert chunked.returncode == 0, chunked.stderr

    cache = load_val_cache(tiny_fixture["cache_dir"])
    expected_targets = sum(max(doc.token_len - 1, 0) for doc in cache.iter_docs())
    whole_records = _read_jsonl(whole_out)
    chunked_records = _read_jsonl(chunked_out)
    assert [r["doc_id"] for r in chunked_records] == [r["doc_id"] for r in whole_records]
    assert sum(r["tokens"] for r in whole_records) == expected_targets
    assert sum(r["tokens"] for r in chunked_records) == expected_targets
    for whole_rec, chunked_rec in zip(whole_records, chunked_records):
        assert chunked_rec["tokens"] == whole_rec["tokens"]
        assert chunked_rec["bpb"] == pytest.approx(whole_rec["bpb"], rel=0.0, abs=1e-6)

    summary = json.loads(chunked_summary.read_text())
    assert summary["tokens_scored"] == expected_targets
    assert summary["score_boundary_targets"] is True
    assert summary["doc_ordering"] == "token_len_desc"
    assert summary["device_tokens_staged"] is True
    assert summary["torch_compile_mode"] == "none"

    source_out = tmp_path / "source.jsonl"
    source_summary = tmp_path / "source_summary.json"
    source_order = subprocess.run(
        [
            sys.executable,
            "scripts/run_exp20_fast_score.py",
            "--cache-dir",
            str(tiny_fixture["cache_dir"]),
            "--checkpoint-path",
            str(tiny_fixture["ckpt"]),
            "--output-path",
            str(source_out),
            "--summary-path",
            str(source_summary),
            "--chunk-size",
            "8",
            "--device",
            "cpu",
            "--doc-batch-size",
            "3",
            "--no-sort-docs-by-length",
        ],
        capture_output=True,
        text=True,
    )
    assert source_order.returncode == 0, source_order.stderr
    source_records = _read_jsonl(source_out)
    assert [r["doc_id"] for r in source_records] == [r["doc_id"] for r in whole_records]
    assert sum(r["tokens"] for r in source_records) == expected_targets
    for whole_rec, source_rec in zip(whole_records, source_records):
        assert source_rec["tokens"] == whole_rec["tokens"]
        assert source_rec["bpb"] == pytest.approx(whole_rec["bpb"], rel=0.0, abs=1e-6)
    assert json.loads(source_summary.read_text())["doc_ordering"] == "source_order"


def test_prepare_doc_work_sorts_by_length_and_remembers_output_order() -> None:
    docs = [
        CachedDoc(doc_id=0, token_start=0, token_len=4, raw_bytes=4),
        CachedDoc(doc_id=1, token_start=4, token_len=12, raw_bytes=12),
        CachedDoc(doc_id=2, token_start=16, token_len=7, raw_bytes=7),
    ]

    sorted_work = prepare_doc_work(docs, sort_by_length=True)
    original_work = prepare_doc_work(docs, sort_by_length=False)

    assert [work.doc.doc_id for work in sorted_work] == [1, 2, 0]
    assert [work.output_index for work in sorted_work] == [1, 2, 0]
    assert [work.doc.doc_id for work in original_work] == [0, 1, 2]
    assert [work.output_index for work in original_work] == [0, 1, 2]


def test_resolve_doc_batch_size_caps_by_token_budget() -> None:
    assert resolve_doc_batch_size(
        requested_doc_batch_size=4096,
        chunk_size=256,
        max_batch_tokens=524_288,
    ) == 2048
    assert resolve_doc_batch_size(
        requested_doc_batch_size=512,
        chunk_size=256,
        max_batch_tokens=524_288,
    ) == 512
    assert resolve_doc_batch_size(
        requested_doc_batch_size=4096,
        chunk_size=-1,
        max_batch_tokens=524_288,
    ) == 4096


def test_doc_range_for_rank_partitions_docs_exactly_once() -> None:
    ranges = [doc_range_for_rank(num_docs=10, rank=rank, world_size=4) for rank in range(4)]

    assert ranges == [(0, 2), (2, 5), (5, 7), (7, 10)]
    covered = [doc_id for start, end in ranges for doc_id in range(start, end)]
    assert covered == list(range(10))


def test_doc_range_for_rank_handles_empty_tail_ranks() -> None:
    ranges = [doc_range_for_rank(num_docs=2, rank=rank, world_size=4) for rank in range(4)]

    assert ranges == [(0, 0), (0, 1), (1, 1), (1, 2)]
    covered = [doc_id for start, end in ranges for doc_id in range(start, end)]
    assert covered == [0, 1]


def test_resolve_distributed_context_defaults_to_single_process() -> None:
    ctx = resolve_distributed_context({})

    assert ctx == {"rank": 0, "world_size": 1, "local_rank": 0, "distributed": False}


def test_resolve_distributed_context_reads_torchrun_env() -> None:
    ctx = resolve_distributed_context({
        "RANK": "2",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "1",
    })

    assert ctx == {"rank": 2, "world_size": 4, "local_rank": 1, "distributed": True}
