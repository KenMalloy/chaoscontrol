from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import sentencepiece as spm
import torch
import torch.nn.functional as F

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
expected_scored_tokens = fast_score.expected_scored_tokens
prepare_doc_work = fast_score.prepare_doc_work
prepare_rank_doc_work = fast_score.prepare_rank_doc_work
padded_token_work = fast_score.padded_token_work
record_order_safe_reason = fast_score.record_order_safe_reason
resolve_doc_batch_size = fast_score.resolve_doc_batch_size
resolve_max_forward_tokens = fast_score.resolve_max_forward_tokens
resolve_distributed_context = fast_score.resolve_distributed_context
validate_doc_score_coverage = fast_score.validate_doc_score_coverage


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
    assert summary["doc_ordering"] == "chunk_count_tail"
    assert summary["doc_packing"] == "chunk_count_tail"
    assert summary["record_order_safe"] is True
    assert summary["record_order_safe_reason"] == "reset_score_only_commutative_ce_reduction"
    assert summary["score_reduction_order_invariant"] is True
    assert summary["device_tokens_staged"] is True
    assert summary["device_token_dtype"] == "torch.int32"
    assert summary["torch_compile_mode"] == "none"
    assert summary["score_warmup_steps"] == 0
    assert summary["score_graph_mode"] == "none"
    assert summary["graph_replay_count"] == 0
    assert summary["graph_fallback_count"] == 0

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


def test_batched_boundary_targets_match_direct_chunked_model_score(
    tiny_fixture: dict[str, Path],
) -> None:
    cache = load_val_cache(tiny_fixture["cache_dir"])
    model, _cfg = fast_score._build_model(tiny_fixture["ckpt"])
    model.eval()
    device = torch.device("cpu")
    chunk_size = 8
    docs = list(cache.iter_docs())
    work = prepare_doc_work(docs, doc_packing="source_order", chunk_size=chunk_size)
    device_tokens = torch.tensor(cache.tokens, dtype=torch.int32, device=device)

    def direct_doc_score(doc: CachedDoc) -> tuple[float, int]:
        prev_states = None
        ce_nats = 0.0
        tokens_scored = 0
        doc_tokens = cache.tokens_for_doc(doc)
        with torch.inference_mode():
            for start, end in fast_score._token_chunk_ranges(int(doc.token_len), chunk_size):
                if end - start < 2:
                    continue
                chunk = torch.tensor(doc_tokens[start:end], dtype=torch.long, device=device).unsqueeze(0)
                kwargs = {"initial_states": prev_states} if prev_states is not None else {}
                out = model(chunk, **kwargs)
                logits = fast_score._model_logits(out)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    chunk[:, 1:].reshape(-1),
                    reduction="sum",
                )
                tokens_scored += int(chunk.size(1) - 1)
                if end < int(doc.token_len):
                    target = torch.tensor([int(doc_tokens[end])], dtype=torch.long, device=device)
                    loss = loss + F.cross_entropy(logits[:, -1, :], target, reduction="sum")
                    tokens_scored += 1
                ce_nats += float(loss.item())
                prev_states = fast_score._model_final_states(out)
        return ce_nats, tokens_scored

    with torch.inference_mode():
        scores = fast_score._score_docs_reset_batch(
            model=model,
            work_items=work,
            device_tokens=device_tokens,
            chunk_size=chunk_size,
            device=device,
            score_boundary_targets=True,
        )

    scores_by_output = sorted(scores, key=lambda score: score.output_index)
    for score, doc in zip(scores_by_output, docs):
        direct_ce, direct_tokens = direct_doc_score(doc)
        assert score.doc.doc_id == doc.doc_id
        assert score.tokens_scored == direct_tokens == max(int(doc.token_len) - 1, 0)
        assert score.ce_nats == pytest.approx(direct_ce, rel=0.0, abs=1e-4)


def test_fast_score_rejects_online_replay_eviction_checkpoint(tmp_path: Path) -> None:
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

    with pytest.raises(RuntimeError, match="GPU3 memory oracle"):
        fast_score._build_model_with_blob(ckpt_path)


def test_fake_graph_path_matches_eager_and_replays_zero_initial_chunks(
    tiny_fixture: dict[str, Path],
) -> None:
    cache = load_val_cache(tiny_fixture["cache_dir"])
    model, _cfg = fast_score._build_model(tiny_fixture["ckpt"])
    model.eval()
    device = torch.device("cpu")
    chunk_size = 8
    docs = list(cache.iter_docs())
    work = prepare_doc_work(docs, doc_packing="source_order", chunk_size=chunk_size)
    device_tokens = torch.tensor(cache.tokens, dtype=torch.int32, device=device)

    class FakeGraphRunner:
        batch_size = len(work)
        seq_len = chunk_size

        def __init__(self) -> None:
            self.zero_replay_count = 0
            self.carried_replay_count = 0

        def _score(
            self,
            *,
            chunk: torch.Tensor,
            initial_states: list[torch.Tensor] | None,
            boundary_targets: torch.Tensor,
            boundary_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            kwargs = {"initial_states": initial_states} if initial_states is not None else {}
            out = model(chunk, **kwargs)
            logits = fast_score._model_logits(out)
            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1).to(torch.long),
                reduction="none",
            ).reshape(chunk.size(0), chunk.size(1) - 1).sum(dim=1)
            boundary_losses = F.cross_entropy(
                logits[:, -1, :],
                boundary_targets.to(torch.long),
                reduction="none",
            )
            losses = losses + boundary_losses * boundary_mask.to(boundary_losses.dtype)
            return losses, fast_score._model_final_states(out)

        def replay_zero_initial(
            self,
            *,
            chunk: torch.Tensor,
            boundary_targets: torch.Tensor,
            boundary_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            self.zero_replay_count += 1
            return self._score(
                chunk=chunk,
                initial_states=None,
                boundary_targets=boundary_targets,
                boundary_mask=boundary_mask,
            )

        def replay(
            self,
            *,
            chunk: torch.Tensor,
            initial_states: list[torch.Tensor],
            boundary_targets: torch.Tensor,
            boundary_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            self.carried_replay_count += 1
            return self._score(
                chunk=chunk,
                initial_states=initial_states,
                boundary_targets=boundary_targets,
                boundary_mask=boundary_mask,
            )

    with torch.inference_mode():
        eager_scores = fast_score._score_docs_reset_batch(
            model=model,
            work_items=work,
            device_tokens=device_tokens,
            chunk_size=chunk_size,
            device=device,
            score_boundary_targets=True,
        )
        fake_graph = FakeGraphRunner()
        graph_stats = fast_score._ScoreGraphStats()
        graph_scores = fast_score._score_docs_reset_batch(
            model=model,
            work_items=work,
            device_tokens=device_tokens,
            chunk_size=chunk_size,
            device=device,
            score_boundary_targets=True,
            score_graph_runner=fake_graph,
            graph_stats=graph_stats,
        )

    eager_by_output = sorted(eager_scores, key=lambda score: score.output_index)
    graph_by_output = sorted(graph_scores, key=lambda score: score.output_index)
    assert fake_graph.zero_replay_count >= 1
    assert fake_graph.carried_replay_count >= 1
    assert graph_stats.replay_count == fake_graph.zero_replay_count + fake_graph.carried_replay_count
    for eager_score, graph_score in zip(eager_by_output, graph_by_output):
        assert graph_score.doc.doc_id == eager_score.doc.doc_id
        assert graph_score.tokens_scored == eager_score.tokens_scored
        assert graph_score.ce_nats == pytest.approx(eager_score.ce_nats, rel=0.0, abs=0.0)


def test_record_order_safe_reason_documents_packing_contract() -> None:
    assert record_order_safe_reason(
        persistence_mode="reset",
        score_only_mode=True,
        doc_packing="chunk_count_tail",
    ) == "reset_score_only_commutative_ce_reduction"
    assert record_order_safe_reason(
        persistence_mode="carry_state",
        score_only_mode=True,
        doc_packing="chunk_count_tail",
    ) == "not_order_safe_for_stateful_or_adaptive_eval"
    assert record_order_safe_reason(
        persistence_mode="reset",
        score_only_mode=False,
        doc_packing="chunk_count_tail",
    ) == "not_order_safe_for_stateful_or_adaptive_eval"


def test_expected_scored_tokens_matches_boundary_modes() -> None:
    assert expected_scored_tokens(token_len=17, chunk_size=-1, score_boundary_targets=True) == 16
    assert expected_scored_tokens(token_len=17, chunk_size=8, score_boundary_targets=True) == 16
    assert expected_scored_tokens(token_len=17, chunk_size=8, score_boundary_targets=False) == 14
    assert expected_scored_tokens(token_len=1, chunk_size=8, score_boundary_targets=True) == 0


def test_validate_doc_score_coverage_detects_skipped_chunks() -> None:
    doc = CachedDoc(doc_id=7, token_start=100, token_len=17, raw_bytes=17)
    score = fast_score._DocScore(
        doc=doc,
        output_index=0,
        ce_nats=1.0,
        tokens_scored=15,
        chunk_count=2,
        loss_before=0.5,
        wall_ms=1.0,
        state_norm=0.0,
    )

    with pytest.raises(RuntimeError, match="token coverage mismatch"):
        validate_doc_score_coverage(
            score,
            chunk_size=8,
            score_boundary_targets=True,
        )


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


def test_prepare_doc_work_sorts_by_chunk_count_tail() -> None:
    docs = [
        CachedDoc(doc_id=0, token_start=0, token_len=17, raw_bytes=17),
        CachedDoc(doc_id=1, token_start=17, token_len=33, raw_bytes=33),
        CachedDoc(doc_id=2, token_start=50, token_len=31, raw_bytes=31),
        CachedDoc(doc_id=3, token_start=81, token_len=32, raw_bytes=32),
    ]

    work = prepare_doc_work(docs, doc_packing="chunk_count_tail", chunk_size=16)

    assert [item.doc.doc_id for item in work] == [1, 3, 2, 0]
    assert [item.output_index for item in work] == [1, 3, 2, 0]


def test_prepare_rank_doc_work_lpt_balances_padded_batches() -> None:
    docs = [
        CachedDoc(doc_id=0, token_start=0, token_len=65, raw_bytes=65),
        CachedDoc(doc_id=1, token_start=65, token_len=64, raw_bytes=64),
        CachedDoc(doc_id=2, token_start=129, token_len=33, raw_bytes=33),
        CachedDoc(doc_id=3, token_start=162, token_len=32, raw_bytes=32),
        CachedDoc(doc_id=4, token_start=194, token_len=17, raw_bytes=17),
        CachedDoc(doc_id=5, token_start=211, token_len=16, raw_bytes=16),
    ]

    rank0 = prepare_rank_doc_work(
        docs,
        rank=0,
        world_size=2,
        doc_packing="chunk_count_tail",
        chunk_size=16,
        doc_batch_size=2,
    )
    rank1 = prepare_rank_doc_work(
        docs,
        rank=1,
        world_size=2,
        doc_packing="chunk_count_tail",
        chunk_size=16,
        doc_batch_size=2,
    )

    covered = sorted(item.output_index for item in rank0 + rank1)
    assert covered == list(range(len(docs)))
    loads = [
        sum(padded_token_work(item.doc, chunk_size=16) for item in rank_work)
        for rank_work in (rank0, rank1)
    ]
    assert max(loads) - min(loads) <= 32


def test_resolve_doc_batch_size_caps_by_token_budget() -> None:
    assert resolve_doc_batch_size(
        requested_doc_batch_size=4096,
        chunk_size=256,
        max_forward_tokens=524_288,
    ) == 2048
    assert resolve_doc_batch_size(
        requested_doc_batch_size=512,
        chunk_size=256,
        max_forward_tokens=524_288,
    ) == 512
    assert resolve_doc_batch_size(
        requested_doc_batch_size=4096,
        chunk_size=-1,
        max_forward_tokens=524_288,
    ) == 4096


def test_resolve_max_forward_tokens_auto_uses_probe_for_cuda() -> None:
    calls = []

    def fake_probe(limit: int) -> int:
        calls.append(limit)
        return 786_432

    assert resolve_max_forward_tokens(
        max_forward_tokens="auto",
        requested_doc_batch_size=4096,
        chunk_size=256,
        device=torch.device("cuda"),
        probe_fn=fake_probe,
    ) == 786_432
    assert calls == [1_048_576]


def test_resolve_max_forward_tokens_auto_uses_requested_shape_on_cpu() -> None:
    assert resolve_max_forward_tokens(
        max_forward_tokens="auto",
        requested_doc_batch_size=8,
        chunk_size=32,
        device=torch.device("cpu"),
    ) == 256


def test_resolve_max_forward_tokens_accepts_explicit_integer() -> None:
    assert resolve_max_forward_tokens(
        max_forward_tokens="131072",
        requested_doc_batch_size=4096,
        chunk_size=256,
        device=torch.device("cuda"),
        probe_fn=lambda _limit: 1,
    ) == 131_072


def test_cuda_graph_mode_requires_cuda(tiny_fixture: dict[str, Path], tmp_path: Path) -> None:
    run = subprocess.run(
        [
            sys.executable,
            "scripts/run_exp20_fast_score.py",
            "--cache-dir",
            str(tiny_fixture["cache_dir"]),
            "--checkpoint-path",
            str(tiny_fixture["ckpt"]),
            "--output-path",
            str(tmp_path / "out.jsonl"),
            "--summary-path",
            str(tmp_path / "summary.json"),
            "--chunk-size",
            "8",
            "--doc-batch-size",
            "3",
            "--score-graph-mode",
            "cuda",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )

    assert run.returncode != 0
    assert "--score-graph-mode cuda requires CUDA" in run.stderr


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
