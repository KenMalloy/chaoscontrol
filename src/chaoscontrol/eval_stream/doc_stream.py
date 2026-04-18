from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np

from chaoscontrol.eval_stream.types import DocRecord


class DocStreamer:
    """Iterates docs from tokenized eval shards. Splits on EOS token.

    Doc order is deterministic (shard order, then position). Eval-split disjointness
    vs Exp 19 train stream is enforced by caller via shard_paths choice.
    """

    def __init__(
        self,
        *,
        shard_paths: list[Path],
        eos_token: int,
        max_docs: int = 50_000,
        bytes_per_token_estimate: float = 4.0,  # FineWeb avg, ~4 bytes/subword for SP8192
    ) -> None:
        self.shard_paths = [Path(p) for p in shard_paths]
        self.eos_token = eos_token
        self.max_docs = max_docs
        self.bytes_per_token_estimate = bytes_per_token_estimate

    def __iter__(self) -> Iterator[DocRecord]:
        doc_id = 0
        for shard in self.shard_paths:
            arr = np.fromfile(str(shard), dtype=np.int32)
            buf: list[int] = []
            for t in arr:
                t_int = int(t)
                if t_int == self.eos_token:
                    if buf:
                        yield DocRecord(
                            doc_id=doc_id,
                            tokens=buf,
                            raw_bytes=int(len(buf) * self.bytes_per_token_estimate),
                        )
                        doc_id += 1
                        buf = []
                        if doc_id >= self.max_docs:
                            return
                else:
                    buf.append(t_int)
            if buf:
                yield DocRecord(
                    doc_id=doc_id,
                    tokens=buf,
                    raw_bytes=int(len(buf) * self.bytes_per_token_estimate),
                )
                doc_id += 1
                if doc_id >= self.max_docs:
                    return
