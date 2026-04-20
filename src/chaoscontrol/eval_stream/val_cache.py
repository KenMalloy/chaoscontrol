from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from chaoscontrol.eval_stream.doc_stream import DocStreamer


SCHEMA_VERSION = 1
TOKEN_DTYPE = np.uint16
DOC_DTYPE = np.dtype([
    ("doc_id", np.int64),
    ("token_start", np.int64),
    ("token_len", np.int64),
    ("raw_bytes", np.int64),
])


@dataclass(frozen=True)
class CachedDoc:
    doc_id: int
    token_start: int
    token_len: int
    raw_bytes: int


@dataclass
class ValCache:
    cache_dir: Path
    manifest: dict
    tokens: np.ndarray
    docs: np.ndarray

    @property
    def num_docs(self) -> int:
        return int(self.docs.shape[0])

    @property
    def total_tokens(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def total_raw_bytes(self) -> int:
        if self.docs.shape[0] == 0:
            return 0
        return int(self.docs["raw_bytes"].sum())

    def iter_docs(self) -> Iterator[CachedDoc]:
        for row in self.docs:
            yield CachedDoc(
                doc_id=int(row["doc_id"]),
                token_start=int(row["token_start"]),
                token_len=int(row["token_len"]),
                raw_bytes=int(row["raw_bytes"]),
            )

    def tokens_for_doc(self, doc: CachedDoc) -> np.ndarray:
        start = doc.token_start
        end = start + doc.token_len
        return self.tokens[start:end]


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _request_manifest(
    *,
    jsonl_paths: Iterable[Path],
    sp_model_path: Path,
    max_docs: int,
) -> dict:
    paths = [Path(p) for p in jsonl_paths]
    jsonl_stats = []
    for path in paths:
        stat = path.stat()
        jsonl_stats.append({
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "jsonl_stats": jsonl_stats,
        "sp_model_path": str(sp_model_path),
        "sp_model_sha256": _sha256_file(sp_model_path),
        "max_docs": int(max_docs),
        "token_dtype": np.dtype(TOKEN_DTYPE).name,
        "doc_dtype": [list(item) for item in DOC_DTYPE.descr],
    }


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _tokens_path(cache_dir: Path) -> Path:
    return cache_dir / "tokens.npy"


def _docs_path(cache_dir: Path) -> Path:
    return cache_dir / "docs.npy"


def _load_manifest(cache_dir: Path) -> dict | None:
    path = _manifest_path(cache_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_files_exist(cache_dir: Path) -> bool:
    return _tokens_path(cache_dir).is_file() and _docs_path(cache_dir).is_file()


def _cache_content_sha256(tokens: np.ndarray, docs: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tokens).view(np.uint8))
    h.update(np.ascontiguousarray(docs).view(np.uint8))
    return h.hexdigest()


def write_val_cache(
    *,
    jsonl_paths: list[Path],
    sp_model_path: Path,
    cache_dir: Path,
    max_docs: int = 50_000,
    force: bool = False,
) -> dict:
    """Build a generated validation cache for fast Exp20 scoring.

    The cache is intentionally a generated artifact: callers choose a pod/local
    cache directory and source control should track this builder, not its output.
    """
    cache_dir = Path(cache_dir)
    sp_model_path = Path(sp_model_path)
    jsonl_paths = [Path(p) for p in jsonl_paths]
    request = _request_manifest(
        jsonl_paths=jsonl_paths,
        sp_model_path=sp_model_path,
        max_docs=max_docs,
    )
    existing = _load_manifest(cache_dir)
    if existing is not None and existing.get("request") == request and _cache_files_exist(cache_dir):
        return existing
    if existing is not None and existing.get("request") != request and not force:
        raise ValueError(
            f"existing validation cache at {cache_dir} does not match requested inputs; "
            "pass force=True to rebuild"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    for path in (_tokens_path(cache_dir), _docs_path(cache_dir), _manifest_path(cache_dir)):
        if force:
            path.unlink(missing_ok=True)

    token_values: list[int] = []
    doc_rows: list[tuple[int, int, int, int]] = []
    max_token_id = np.iinfo(TOKEN_DTYPE).max
    offset = 0
    for doc in DocStreamer(
        jsonl_paths=jsonl_paths,
        sp_model_path=sp_model_path,
        max_docs=max_docs,
    ):
        if doc.tokens and max(doc.tokens) > max_token_id:
            raise ValueError(
                f"token id {max(doc.tokens)} exceeds {np.dtype(TOKEN_DTYPE).name} capacity"
            )
        token_len = len(doc.tokens)
        doc_rows.append((doc.doc_id, offset, token_len, doc.raw_bytes))
        token_values.extend(doc.tokens)
        offset += token_len

    tokens = np.asarray(token_values, dtype=TOKEN_DTYPE)
    docs = np.asarray(doc_rows, dtype=DOC_DTYPE)
    np.save(_tokens_path(cache_dir), tokens)
    np.save(_docs_path(cache_dir), docs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "request": request,
        "num_docs": int(docs.shape[0]),
        "total_tokens": int(tokens.shape[0]),
        "total_raw_bytes": int(docs["raw_bytes"].sum()) if docs.shape[0] else 0,
        "cache_content_sha256": _cache_content_sha256(tokens, docs),
        "tokens_file": _tokens_path(cache_dir).name,
        "docs_file": _docs_path(cache_dir).name,
        "max_docs": int(max_docs),
    }
    _manifest_path(cache_dir).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_val_cache(cache_dir: Path) -> ValCache:
    cache_dir = Path(cache_dir)
    manifest = _load_manifest(cache_dir)
    if manifest is None:
        raise FileNotFoundError(f"validation cache manifest not found: {_manifest_path(cache_dir)}")
    tokens = np.load(_tokens_path(cache_dir), mmap_mode="r")
    docs = np.load(_docs_path(cache_dir), mmap_mode="r")
    if tokens.dtype != np.dtype(TOKEN_DTYPE):
        raise ValueError(f"unexpected token dtype {tokens.dtype}; expected {np.dtype(TOKEN_DTYPE)}")
    if docs.dtype != DOC_DTYPE:
        raise ValueError(f"unexpected docs dtype {docs.dtype}; expected {DOC_DTYPE}")
    return ValCache(
        cache_dir=cache_dir,
        manifest=manifest,
        tokens=tokens,
        docs=docs,
    )


__all__ = [
    "CachedDoc",
    "ValCache",
    "load_val_cache",
    "write_val_cache",
]
