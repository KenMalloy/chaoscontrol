from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

import sentencepiece as spm

from chaoscontrol.eval_stream.types import DocRecord


class DocStreamer:
    """Iterates docs from FineWeb JSONL files. Tokenizes each line on the fly
    with a persistent SP handle.

    Canonical SP shards are built with `append_eos=False` (see
    scripts/build_sp_shards.py) — there is no EOS sentinel inside them, so we
    must source from the raw JSONL that feeds the shard builder rather than
    the .bin shards themselves. doc_id is zero-based, counted across the
    provided JSONL files in given order.

    Eval-split disjointness vs Exp 19 train is enforced by the caller choosing
    non-overlapping JSONL paths.
    """

    def __init__(
        self,
        *,
        jsonl_paths: list[Path],
        sp_model_path: Path,
        max_docs: int = 50_000,
    ) -> None:
        self.jsonl_paths = [Path(p) for p in jsonl_paths]
        self.sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        self.max_docs = max_docs

    def __iter__(self) -> Iterator[DocRecord]:
        doc_id = 0
        for p in self.jsonl_paths:
            with open(p) as fh:
                for line in fh:
                    text = json.loads(line)["text"]
                    tokens = self.sp.encode(text, out_type=int)
                    if not tokens:
                        continue
                    yield DocRecord(
                        doc_id=doc_id,
                        tokens=tokens,
                        raw_bytes=len(text.encode("utf-8")),
                    )
                    doc_id += 1
                    if doc_id >= self.max_docs:
                        return
