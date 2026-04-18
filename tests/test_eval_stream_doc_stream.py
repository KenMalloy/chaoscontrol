import numpy as np
import pytest
from chaoscontrol.eval_stream.doc_stream import DocStreamer


def test_iterates_docs_in_order(tmp_path):
    # Build a synthetic shard with 3 docs separated by EOS token (0)
    toks = np.array([10, 11, 12, 0,  20, 21, 0,  30, 31, 32, 33, 0], dtype=np.int32)
    shard = tmp_path / "eval.bin"
    toks.tofile(shard)

    streamer = DocStreamer(shard_paths=[shard], eos_token=0, max_docs=10)
    docs = list(streamer)

    assert len(docs) == 3
    assert docs[0].tokens == [10, 11, 12]
    assert docs[1].tokens == [20, 21]
    assert docs[2].tokens == [30, 31, 32, 33]
    assert all(d.doc_id == i for i, d in enumerate(docs))


def test_respects_max_docs(tmp_path):
    toks = np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype=np.int32)
    shard = tmp_path / "eval.bin"
    toks.tofile(shard)
    docs = list(DocStreamer(shard_paths=[shard], eos_token=0, max_docs=2))
    assert len(docs) == 2


def test_raw_bytes_recorded(tmp_path):
    toks = np.array([100, 101, 0], dtype=np.int32)
    shard = tmp_path / "eval.bin"
    toks.tofile(shard)
    # Raw bytes computed from detokenizer; in this test we stub it via constant
    docs = list(DocStreamer(shard_paths=[shard], eos_token=0, max_docs=1,
                            bytes_per_token_estimate=4.0))
    # 2 tokens × 4 bytes/token = 8
    assert docs[0].raw_bytes == 8
