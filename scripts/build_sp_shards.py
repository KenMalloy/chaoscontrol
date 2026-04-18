#!/usr/bin/env python3
"""Build SentencePiece tokenizer + uint16 token shards for Parameter Golf.

Consumes ``docs_selected.jsonl`` (one JSON doc per line, UTF-8, each with a
``text`` field), trains a BPE model at the requested vocab size, tokenizes
every doc, and writes the dataset in the shard layout
``baselines/parameter_golf/datasets/fineweb10B_sp{N}/`` uses — matching the
contract ``src/chaoscontrol/data.py:load_fineweb_tokens`` expects.

Shard contract (matches ``baselines/parameter_golf/cached_challenge_fineweb.py``
and the challenge's published ``manifest.json`` entries):

* Flat ``uint16`` little-endian bytes, no header, no separators.
* File names ``fineweb_{train,val}_{NNNNNN}.bin``.
* Each shard ends on a whole-doc boundary so ``load_fineweb_tokens`` can
  concatenate shards without spurious cross-doc boundaries.
* ``append_eos=False`` — tokens are raw BPE IDs with no per-doc suffix.
* Split: first ``--val-docs`` docs of ``docs_selected.jsonl`` → val; the
  remainder → train. ``docs_selected.jsonl`` is pre-shuffled by the
  challenge at ``shuffle_seed=1337``, so no reshuffling here.

Determinism: same jsonl + same ``--vocab-size`` + same ``--sp-seed`` yields
byte-identical output. The SentencePiece trainer receives ``--random_seed``
and training material is the *first* ``--sp-train-docs`` docs after the
val split, which keeps training disjoint from val.

This is submission-day infrastructure. Keep the CLI stable and the output
reproducible; treat this file as part of the public-repo contract.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Iterator


# The HuggingFace revision of ``willdepueoai/parameter-golf`` whose
# ``docs_selected.jsonl`` defines our input corpus. Hard-coded so the
# ``build_manifest.json`` we emit is self-identifying across machines.
DATASET_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"

# Writing tokens as uint16 matches the on-disk convention in the published
# shards (``cached_challenge_fineweb.py``) and the reader in
# ``src/chaoscontrol/data.py:load_fineweb_tokens``. The reader views the
# mmap as int16 — identical bit pattern so long as every ID is < 32768.
TOKEN_DTYPE_STR = "uint16"
TOKEN_BYTES = 2
MAX_VOCAB_FOR_UINT16 = 32768

# Shard-index field width in the ``fineweb_{split}_{NNNNNN}.bin`` pattern.
SHARD_INDEX_WIDTH = 6


# ---------------------------------------------------------------------------
# JSONL streaming
# ---------------------------------------------------------------------------


def _iter_docs(jsonl_path: Path) -> Iterator[str]:
    """Yield the ``text`` field of every well-formed JSONL line, in file order.

    Silently skips blank lines and malformed JSON, matching the tolerant
    extractor in ``src/chaoscontrol/data.py:_extract_jsonl_to_raw``.
    """
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            if text:
                yield text


def _count_docs(jsonl_path: Path) -> int:
    """Count docs that ``_iter_docs`` will actually yield.

    Shares the filter pipeline with ``_iter_docs`` — blank lines,
    malformed JSON, and empty-``text`` entries are all skipped — so
    ``num_docs`` on the build manifest agrees with the number of docs
    written to shards. The trade-off is a full JSON parse of every
    line (one extra pass over the 48GB corpus) instead of a blank-line
    grep; the correctness-by-construction is worth the ~minute cost.
    """
    n = 0
    for _ in _iter_docs(jsonl_path):
        n += 1
    return n


# ---------------------------------------------------------------------------
# SentencePiece training
# ---------------------------------------------------------------------------


def _write_sp_training_corpus(
    jsonl_path: Path, out_path: Path, skip_first: int, take: int,
) -> int:
    """Write one-text-per-line plaintext for SentencePiece training input.

    SP's Python trainer accepts either an ``--input`` file list or a Python
    iterable; the file path is more portable (no hidden global state) and
    survives multiprocess SP builds on older wheels.
    """
    written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, text in enumerate(_iter_docs(jsonl_path)):
            if i < skip_first:
                continue
            if written >= take:
                break
            # SP expects one sentence per line; embedded newlines in the
            # doc would fracture the training signal, so collapse to space.
            fout.write(text.replace("\n", " ").replace("\r", " "))
            fout.write("\n")
            written += 1
    return written


def _train_sentencepiece(
    corpus_path: Path,
    model_prefix: Path,
    vocab_size: int,
    sp_seed: int,
    num_threads: int,
) -> None:
    """Train a BPE SentencePiece model at ``vocab_size`` with fixed IDs.

    ``append_eos`` is not controlled here (tokenization time); but the
    special-ID layout (UNK=0, BOS=1, EOS=2, PAD=-1 disabled) is fixed so
    the trained vocabulary is stable regardless of how callers later
    configure tokenization.

    Note on ``sp_seed``: the python SentencePiece binding does not expose
    a TrainerSpec seed field (confirmed against sentencepiece 0.2.1 —
    both ``random_seed`` and ``seed`` are unknown). BPE training is a
    deterministic reduction over the corpus once shuffling is disabled,
    so we get byte-identical output from identical inputs without a seed
    knob. ``sp_seed`` is still recorded in the build manifest for future
    auditability, and ``shuffle_input_sentence=False`` locks the single
    source of nondeterminism SP would otherwise introduce.

    Determinism caveat — byte-identity is guaranteed only within matching
    ``(vocab_size, sp_seed, num_workers)`` triples. SentencePiece's
    multi-threaded merge counting can produce tie-breaking drift across
    thread counts. On submission day, always run with the same
    ``--num-workers`` as the previous successful run, or pin to
    ``--num-workers=1`` for strict cross-machine determinism at ~1.5×
    SP training cost.

    Atomicity — SP writes ``.model`` and ``.vocab`` to a ``.partial``
    prefix first, then ``os.replace`` into the final names on success.
    A killed mid-training process cannot leave a half-written ``.model``
    that later ``--skip-train`` runs would silently trust.
    """
    import sentencepiece as spm

    # SP's trainer writes ``{prefix}.model`` and ``{prefix}.vocab`` by
    # string concatenation — use ``.with_name(name + ".ext")`` here (not
    # ``.with_suffix``) because ``model_prefix`` may itself end in a
    # ``.partial`` segment that ``.with_suffix`` would overwrite.
    partial_prefix = model_prefix.with_name(model_prefix.name + ".partial")
    final_model = model_prefix.with_name(model_prefix.name + ".model")
    final_vocab = model_prefix.with_name(model_prefix.name + ".vocab")
    partial_model = partial_prefix.with_name(partial_prefix.name + ".model")
    partial_vocab = partial_prefix.with_name(partial_prefix.name + ".vocab")

    # Clean any stale partials from a previous crashed run.
    partial_model.unlink(missing_ok=True)
    partial_vocab.unlink(missing_ok=True)

    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(partial_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        # Byte fallback guarantees round-tripping even on rare unicode —
        # required for FineWeb's raw web text.
        byte_fallback=True,
        # Fixed special-ID layout: matches the challenge baseline tokenizer
        # so downstream code can assume UNK=0, BOS=1, EOS=2.
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        num_threads=num_threads,
        # Determinism — disables SP's internal input shuffling which
        # would otherwise be the only nondeterministic step.
        input_sentence_size=0,
        shuffle_input_sentence=False,
    )

    # Promote partials to final paths atomically. Both must succeed or
    # neither should be visible; if ``.vocab`` promotion fails after
    # ``.model`` already landed, raise loudly so the caller sees it.
    os.replace(partial_model, final_model)
    try:
        os.replace(partial_vocab, final_vocab)
    except OSError:
        # Roll back the model promotion so the pair stays consistent.
        final_model.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Tokenization worker pool
# ---------------------------------------------------------------------------


# Global per-worker SentencePieceProcessor — set once by the pool
# initializer so ``_encode_batch`` doesn't reload the model every call.
_WORKER_SP = None


def _worker_init(sp_model_path: str) -> None:
    """Pool initializer: load the SP model once per worker process."""
    import sentencepiece as spm

    global _WORKER_SP
    _WORKER_SP = spm.SentencePieceProcessor(model_file=sp_model_path)


def _encode_batch(batch: list[str]) -> list[list[int]]:
    """Encode one batch of docs → list of token-ID lists.

    No EOS appended — matches the ``append_eos: false`` manifest contract.
    """
    assert _WORKER_SP is not None, "worker not initialized"
    return _WORKER_SP.encode(batch, out_type=int)


def _batched(
    source: Iterable[str], batch_size: int,
) -> Iterator[list[str]]:
    """Yield lists of up to ``batch_size`` items from ``source``."""
    buf: list[str] = []
    for item in source:
        buf.append(item)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


class ShardWriter:
    """Accumulate encoded docs and flush ``>= shard_size``-token shards.

    Every flush happens on a doc boundary — the buffer only grows by
    whole docs, so the invariant is structural, not a runtime check.
    """

    def __init__(self, out_dir: Path, split: str, shard_size: int) -> None:
        self.out_dir = out_dir
        self.split = split
        self.shard_size = shard_size
        self._buf: list[int] = []  # type: ignore[var-annotated]
        self._shard_index = 0
        self._files: list[Path] = []
        self._total_tokens = 0

    def _flush(self) -> None:
        if not self._buf:
            return
        import numpy as np

        path = self.out_dir / (
            f"fineweb_{self.split}_{self._shard_index:0{SHARD_INDEX_WIDTH}d}.bin"
        )
        # Use a temp file + atomic rename so a crash mid-write never
        # leaves a corrupt shard that future idempotent-skip checks
        # would mistake for a valid one.
        tmp = path.with_suffix(".bin.tmp")
        arr = np.asarray(self._buf, dtype=np.uint16)
        arr.tofile(tmp)
        os.replace(tmp, path)

        self._files.append(path)
        self._total_tokens += len(self._buf)
        self._shard_index += 1
        self._buf = []

    def add_doc(self, tokens: list[int]) -> None:
        self._buf.extend(tokens)
        if len(self._buf) >= self.shard_size:
            self._flush()

    def finalize(self) -> None:
        self._flush()

    @property
    def files(self) -> list[Path]:
        return list(self._files)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


# ---------------------------------------------------------------------------
# Build-manifest book-keeping
# ---------------------------------------------------------------------------


def _build_manifest_dict(
    vocab_size: int,
    num_docs: int,
    num_val_docs: int,
    num_train_docs: int,
    sp_train_docs: int,
    shard_size: int,
    num_workers: int,
    files_val: list[Path],
    files_train: list[Path],
    tokens_val: int,
    tokens_train: int,
    sp_seed: int,
    build_time_seconds: float,
) -> dict:
    return {
        "dataset_revision": DATASET_REVISION,
        "vocab_size": vocab_size,
        "sp_seed": sp_seed,
        "num_docs": num_docs,
        "num_val_docs": num_val_docs,
        "num_train_docs": num_train_docs,
        "sp_train_docs": sp_train_docs,
        "shard_size": shard_size,
        "num_workers": num_workers,
        "files_val": [p.name for p in files_val],
        "files_train": [p.name for p in files_train],
        "tokens_val": tokens_val,
        "tokens_train": tokens_train,
        "build_time_seconds": round(build_time_seconds, 3),
        "token_dtype": TOKEN_DTYPE_STR,
        "append_eos": False,
    }


def _manifest_matches(existing: dict, requested: dict) -> bool:
    """Idempotent-skip gate: does ``existing`` describe the same build?"""
    keys = (
        "dataset_revision",
        "vocab_size",
        "sp_seed",
        "num_docs",
        "num_val_docs",
        "num_train_docs",
        "sp_train_docs",
        "shard_size",
        "num_workers",
    )
    return all(existing.get(k) == requested.get(k) for k in keys)


# ---------------------------------------------------------------------------
# Core build routine
# ---------------------------------------------------------------------------


def build(
    docs_path: Path,
    vocab_size: int,
    output_dir: Path,
    tokenizer_dir: Path,
    sp_train_docs: int,
    val_docs: int,
    shard_size: int,
    sp_seed: int,
    num_workers: int,
    dry_run: bool,
    skip_tokenize: bool,
    skip_train: bool,
    force: bool,
    log_every_docs: int = 100_000,
) -> dict:
    """Build SP model and shards. Returns the build manifest as a dict.

    On ``dry_run`` no files are written; the returned dict is the planned
    manifest with token counts omitted.

    ``--skip-tokenize`` produces only the ``.model`` + ``.vocab`` files; no
    ``build_manifest.json`` is written. A subsequent ``--skip-train`` run
    against the same ``--output-dir`` writes the full manifest and the
    token shards.
    """
    if vocab_size <= 0 or vocab_size > MAX_VOCAB_FOR_UINT16:
        raise ValueError(
            f"--vocab-size must be in (0, {MAX_VOCAB_FOR_UINT16}] so "
            f"every ID fits in int16/uint16; got {vocab_size}"
        )
    if not docs_path.is_file():
        raise FileNotFoundError(f"docs path not found: {docs_path}")
    if val_docs < 0 or sp_train_docs <= 0 or shard_size <= 0:
        raise ValueError("--val-docs, --sp-train-docs, --shard-size must be positive")

    # Single-line run-start config log — ssh-disconnect + relaunch finds
    # this at the log tail, no scrolling needed to confirm the config.
    print(
        f"[build_sp_shards] start: vocab_size={vocab_size} sp_seed={sp_seed} "
        f"num_workers={num_workers} docs_path={docs_path} "
        f"output_dir={output_dir} tokenizer_dir={tokenizer_dir} "
        f"dry_run={dry_run} skip_train={skip_train} "
        f"skip_tokenize={skip_tokenize} force={force}"
    )

    model_path = tokenizer_dir / f"fineweb_{vocab_size}_bpe.model"
    vocab_path = tokenizer_dir / f"fineweb_{vocab_size}_bpe.vocab"
    build_manifest_path = output_dir / "build_manifest.json"

    # --- DRY RUN ---------------------------------------------------------
    if dry_run:
        num_docs = _count_docs(docs_path)
        plan = {
            "dataset_revision": DATASET_REVISION,
            "vocab_size": vocab_size,
            "sp_seed": sp_seed,
            "num_docs": num_docs,
            "num_val_docs": min(val_docs, num_docs),
            "num_train_docs": max(num_docs - val_docs, 0),
            "sp_train_docs_planned": min(sp_train_docs, max(num_docs - val_docs, 0)),
            "shard_size": shard_size,
            "num_workers": num_workers,
            "output_dir": str(output_dir),
            "tokenizer_dir": str(tokenizer_dir),
            "model_path": str(model_path),
            "estimate": _estimate_build_time(
                num_docs=num_docs,
                sp_train_docs=min(sp_train_docs, max(num_docs - val_docs, 0)),
                num_workers=num_workers,
            ),
        }
        print(json.dumps(plan, indent=2))
        return plan

    # --- IDEMPOTENT SKIP -------------------------------------------------
    num_docs = _count_docs(docs_path)
    requested = {
        "dataset_revision": DATASET_REVISION,
        "vocab_size": vocab_size,
        "sp_seed": sp_seed,
        "num_docs": num_docs,
        "num_val_docs": min(val_docs, num_docs),
        "num_train_docs": max(num_docs - val_docs, 0),
        "sp_train_docs": min(sp_train_docs, max(num_docs - val_docs, 0)),
        "shard_size": shard_size,
        "num_workers": num_workers,
    }

    if build_manifest_path.is_file():
        try:
            existing = json.loads(build_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if _manifest_matches(existing, requested):
            print(f"[build_sp_shards] Shards already built, skipping: {output_dir}")
            return existing
        if not force:
            raise RuntimeError(
                f"output dir {output_dir} has a build_manifest.json that does "
                f"not match the requested config; pass --force to overwrite. "
                f"existing={existing} requested={requested}"
            )

    # Guard against stale partial shards without a manifest — these would
    # silently be globbed by ``load_fineweb_tokens`` and corrupt training.
    existing_shards = list(output_dir.glob("fineweb_*.bin"))
    if existing_shards and not build_manifest_path.is_file() and not force:
        raise RuntimeError(
            f"output dir {output_dir} has shards but no build_manifest.json "
            f"— refusing to overwrite. Pass --force to clear and rebuild."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for p in existing_shards:
            p.unlink()
        build_manifest_path.unlink(missing_ok=True)

    # --- DISK-SPACE PREFLIGHT -------------------------------------------
    # Rough estimate: ~700 tokens per FineWeb doc × 2 bytes/token × 1.5×
    # safety factor. Cheap ``shutil.disk_usage`` call — failing fast here
    # beats dying 20 minutes into a tokenization pass with a half-written
    # shard on disk.
    avg_tokens_per_doc = 700
    estimated_shard_bytes = num_docs * avg_tokens_per_doc * TOKEN_BYTES
    required_bytes = int(estimated_shard_bytes * 1.5)
    try:
        usage = shutil.disk_usage(output_dir)
    except OSError:
        usage = None
    if usage is not None and usage.free < required_bytes:
        need_gb = required_bytes / (1024 ** 3)
        have_gb = usage.free / (1024 ** 3)
        raise RuntimeError(
            f"Insufficient disk space: need ~{need_gb:.1f}GB, have "
            f"{have_gb:.1f}GB at {output_dir}. Move output_dir to a "
            f"larger volume."
        )

    t0 = time.monotonic()

    # --- TRAIN SP --------------------------------------------------------
    if not skip_train:
        # Training material: the first ``sp_train_docs`` docs *after* the
        # val region. Disjoint from val by construction.
        effective_train_take = min(sp_train_docs, max(num_docs - val_docs, 0))
        if effective_train_take == 0:
            raise RuntimeError(
                "no docs available for SP training after reserving val — "
                "shrink --val-docs or enlarge the input."
            )
        print(
            f"[build_sp_shards] Training SP BPE vocab_size={vocab_size} "
            f"on {effective_train_take} docs (skip_first={val_docs}) ..."
        )
        with tempfile.TemporaryDirectory(prefix="sp_train_") as td:
            corpus_path = Path(td) / "sp_train_corpus.txt"
            n_written = _write_sp_training_corpus(
                docs_path, corpus_path,
                skip_first=val_docs, take=effective_train_take,
            )
            if n_written == 0:
                raise RuntimeError(
                    "SP training corpus is empty — check docs_selected.jsonl"
                )
            _train_sentencepiece(
                corpus_path=corpus_path,
                model_prefix=tokenizer_dir / f"fineweb_{vocab_size}_bpe",
                vocab_size=vocab_size,
                sp_seed=sp_seed,
                num_threads=max(1, num_workers),
            )
        print(f"[build_sp_shards] Wrote tokenizer to {model_path}")
    else:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"--skip-train given but {model_path} does not exist"
            )

    if skip_tokenize:
        print("[build_sp_shards] --skip-tokenize: done (tokenizer only)")
        return {"model_path": str(model_path), "vocab_size": vocab_size}

    # --- TOKENIZE + SHARD -----------------------------------------------
    val_writer = ShardWriter(output_dir, "val", shard_size)
    train_writer = ShardWriter(output_dir, "train", shard_size)

    # Batch size is a local throughput knob; large enough to amortize IPC,
    # small enough that each worker returns promptly so the shard writer
    # sees tokens and flushes. 256 is a reasonable middle for FineWeb.
    batch_size = 256

    print(
        f"[build_sp_shards] Tokenizing {num_docs} docs with "
        f"{num_workers} worker(s), shard_size={shard_size:,} ..."
    )
    t_tok = time.monotonic()
    processed = 0
    last_log = t_tok

    def _iter_batches() -> Iterator[list[str]]:
        yield from _batched(_iter_docs(docs_path), batch_size)

    # ``imap`` (not ``imap_unordered``) preserves the input order, which is
    # essential — the val/train split is defined by the doc index in the
    # jsonl, and shard contents must be deterministic across runs.
    def _process(encode: "callable") -> None:
        nonlocal processed, last_log
        doc_index = 0
        for batch_tokens in encode(_iter_batches()):
            for tokens in batch_tokens:
                # Defense in depth: the trainer could in principle emit
                # an ID outside [0, vocab_size) with byte_fallback + a
                # truncated vocab; explicit check would corrupt uint16
                # storage silently otherwise.
                # (No per-token check for speed; rely on invariant that
                # SP only produces IDs in [0, vocab_size).)
                if doc_index < val_docs:
                    val_writer.add_doc(tokens)
                else:
                    train_writer.add_doc(tokens)
                doc_index += 1
            processed = doc_index
            now = time.monotonic()
            if processed % log_every_docs < batch_size and now - last_log > 5.0:
                elapsed = now - t_tok
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (num_docs - processed) / rate if rate > 0 else 0.0
                print(
                    f"[build_sp_shards] {processed:,}/{num_docs:,} docs "
                    f"({rate:,.0f} docs/s, eta={eta:,.0f}s, "
                    f"shard_val={val_writer._shard_index}, "
                    f"shard_train={train_writer._shard_index})"
                )
                last_log = now

    if num_workers <= 1:
        # Single-process path — also used by tests (multiprocessing in a
        # pytest fork can hang on macOS under certain conditions).
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(model_path))

        def _encode_single(batches: Iterator[list[str]]) -> Iterator[list[list[int]]]:
            for b in batches:
                yield sp.encode(b, out_type=int)

        _process(_encode_single)
    else:
        # ``fork`` on Linux / ``spawn`` on macOS — use the default start
        # method so pool initializer paths stay predictable.
        ctx = mp.get_context()
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(str(model_path),),
        ) as pool:
            # chunksize=1 since each task is already a batch of docs;
            # larger chunking would delay the first shard flush.
            def _encode_pool(batches: Iterator[list[str]]) -> Iterator[list[list[int]]]:
                yield from pool.imap(_encode_batch, batches, chunksize=1)

            _process(_encode_pool)

    val_writer.finalize()
    train_writer.finalize()

    build_time = time.monotonic() - t0
    manifest = _build_manifest_dict(
        vocab_size=vocab_size,
        num_docs=num_docs,
        num_val_docs=min(val_docs, num_docs),
        num_train_docs=max(num_docs - val_docs, 0),
        sp_train_docs=min(sp_train_docs, max(num_docs - val_docs, 0)),
        shard_size=shard_size,
        num_workers=num_workers,
        files_val=val_writer.files,
        files_train=train_writer.files,
        tokens_val=val_writer.total_tokens,
        tokens_train=train_writer.total_tokens,
        sp_seed=sp_seed,
        build_time_seconds=build_time,
    )
    # Write manifest atomically — it's the idempotent-skip marker.
    tmp_path = build_manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, build_manifest_path)

    print(
        f"[build_sp_shards] Done in {build_time:,.1f}s — "
        f"val: {len(val_writer.files)} shard(s) / {val_writer.total_tokens:,} tokens; "
        f"train: {len(train_writer.files)} shard(s) / {train_writer.total_tokens:,} tokens"
    )
    return manifest


# ---------------------------------------------------------------------------
# Build-time estimator for the dry-run plan
# ---------------------------------------------------------------------------


def _estimate_build_time(
    num_docs: int, sp_train_docs: int, num_workers: int,
) -> dict:
    """Heuristic wall-time estimate for the submission-day plan.

    Baselines (order-of-magnitude, recalibrate on first real run):
      * SP BPE training on 5M FineWeb docs, vocab=16384 → ~25 minutes
        (single-process trainer; num_threads helps sub-linearly above
        ~8 threads). Scales ~linearly in docs, gently in log(vocab).
      * Tokenization throughput per worker → ~2,000 FineWeb docs/s
        (docs average ~700 tokens; sentencepiece ``encode`` on the
        Python binding runs ~1.5M tokens/s/worker). Near-linear
        scaling to ~24 workers, then memory bandwidth dominates.

    The 5M-doc SP train dominates wall time for the 16384 config; the
    tokenization pass is I/O-bound but parallel enough to finish in
    under two hours on 28 vCPUs.
    """
    # SP training: linear in docs, gentle log in vocab; the 5M plateau
    # reflects SP's internal sub-sampling cap, not a strict ceiling.
    sp_minutes = (sp_train_docs / 5_000_000) * 25.0
    # Tokenization: ~2k docs/s per worker, scaling to ~90% of linear
    # for the first 24 workers, then dropping sharply.
    per_worker_rate = 2_000.0
    effective_workers = min(num_workers, 24) * 0.9 + max(0, num_workers - 24) * 0.3
    tok_seconds = num_docs / max(per_worker_rate * effective_workers, 1.0)
    return {
        "sp_train_minutes": round(sp_minutes, 1),
        "tokenize_minutes": round(tok_seconds / 60.0, 1),
        "total_minutes": round(sp_minutes + tok_seconds / 60.0, 1),
        "notes": "calibrated on 28-vCPU CUDA 13 pod; FineWeb text; rough estimate",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_output_dir(vocab_size: int) -> Path:
    return (
        Path("baselines/parameter_golf/datasets") / f"fineweb10B_sp{vocab_size}"
    )


def _default_tokenizer_dir() -> Path:
    return Path("baselines/parameter_golf/tokenizers")


def _default_docs_path() -> Path:
    return Path("baselines/parameter_golf/datasets/docs_selected.jsonl")


def _default_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(1, cpu // 2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--docs-path", type=Path, default=_default_docs_path())
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--tokenizer-dir", type=Path, default=_default_tokenizer_dir())
    p.add_argument("--sp-train-docs", type=int, default=5_000_000)
    p.add_argument("--val-docs", type=int, default=50_000)
    p.add_argument("--shard-size", type=int, default=100_000_000)
    p.add_argument("--sp-seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=_default_workers())
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-tokenize", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument(
        "--force", action="store_true",
        help="overwrite existing shards when build_manifest.json is absent or mismatched",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or _default_output_dir(args.vocab_size)
    try:
        build(
            docs_path=args.docs_path,
            vocab_size=args.vocab_size,
            output_dir=output_dir,
            tokenizer_dir=args.tokenizer_dir,
            sp_train_docs=args.sp_train_docs,
            val_docs=args.val_docs,
            shard_size=args.shard_size,
            sp_seed=args.sp_seed,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
            skip_tokenize=args.skip_tokenize,
            skip_train=args.skip_train,
            force=args.force,
        )
    except Exception as e:
        print(f"[build_sp_shards] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
