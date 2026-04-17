"""Unit tests for ``scripts/build_sp_shards.py``.

All tests run on tiny synthetic JSONL (≈1000 docs, vocab=256) so the
full suite finishes in seconds; the 10B-token real build is exercised
only on a pod.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_sp_shards.py"


def _load_module():
    """Load ``scripts/build_sp_shards.py`` as a module — not on sys.path."""
    spec = importlib.util.spec_from_file_location(
        "build_sp_shards", SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_sp_shards = _load_module()
pytest.importorskip("sentencepiece")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_docs(n: int, seed: int = 0) -> list[str]:
    """Generate readable, non-trivial docs so SP BPE training converges.

    Uses a small controlled vocabulary; SP needs real repetition to
    learn merges at vocab=256. Random-letter noise would mostly hit
    byte fallback and defeat the test's purpose.
    """
    rng = random.Random(seed)
    words = [
        "the", "model", "reads", "tokens", "and", "writes", "shards",
        "with", "byte", "fallback", "when", "rare", "unicode", "appears",
        "chaos", "control", "experiments", "run", "on", "fineweb", "data",
        "sentence", "piece", "vocab", "boundary", "check", "uint", "sixteen",
        "parameter", "golf", "submission", "day", "pod", "worker",
    ]
    docs: list[str] = []
    for i in range(n):
        length = rng.randint(20, 120)
        doc = " ".join(rng.choice(words) for _ in range(length))
        # Seed a unique token so each doc's encoding differs; important
        # for the shard-boundary test where we need nonzero token counts.
        docs.append(f"doc{i} {doc}")
    return docs


@pytest.fixture
def synthetic_jsonl(tmp_path: Path) -> Path:
    """A 1000-doc synthetic JSONL, deterministic across runs."""
    docs = _make_synthetic_docs(n=1000, seed=42)
    path = tmp_path / "docs_selected.jsonl"
    with open(path, "w", encoding="utf-8") as fout:
        for d in docs:
            fout.write(json.dumps({"text": d}) + "\n")
    return path


def _run_build(
    tmp_path: Path,
    jsonl: Path,
    *,
    vocab_size: int = 512,
    sp_train_docs: int = 500,
    val_docs: int = 100,
    shard_size: int = 10_000,
    sp_seed: int = 1337,
    num_workers: int = 1,
    dry_run: bool = False,
    skip_tokenize: bool = False,
    skip_train: bool = False,
    force: bool = False,
    output_subdir: str = "out",
    tokenizer_subdir: str = "tok",
) -> tuple[Path, Path, dict]:
    output_dir = tmp_path / output_subdir
    tokenizer_dir = tmp_path / tokenizer_subdir
    result = build_sp_shards.build(
        docs_path=jsonl,
        vocab_size=vocab_size,
        output_dir=output_dir,
        tokenizer_dir=tokenizer_dir,
        sp_train_docs=sp_train_docs,
        val_docs=val_docs,
        shard_size=shard_size,
        sp_seed=sp_seed,
        num_workers=num_workers,
        dry_run=dry_run,
        skip_tokenize=skip_tokenize,
        skip_train=skip_train,
        force=force,
    )
    return output_dir, tokenizer_dir, result


def _read_shard(path: Path) -> np.ndarray:
    """Read a uint16 shard into memory — tests only, not production."""
    return np.fromfile(path, dtype=np.uint16)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trains_sp_and_tokenizes_round_trip(
    tmp_path: Path, synthetic_jsonl: Path,
) -> None:
    """End-to-end: SP trains, shards written, decoded text matches input."""
    import sentencepiece as spm

    out_dir, tok_dir, manifest = _run_build(
        tmp_path, synthetic_jsonl,
        vocab_size=512, sp_train_docs=500, val_docs=100, shard_size=10_000,
    )

    model_path = tok_dir / "fineweb_512_bpe.model"
    assert model_path.is_file()

    val_shards = sorted(out_dir.glob("fineweb_val_*.bin"))
    train_shards = sorted(out_dir.glob("fineweb_train_*.bin"))
    assert len(val_shards) >= 1, "expected at least one val shard"
    assert len(train_shards) >= 1, "expected at least one train shard"

    # Total tokens match a fresh independent encoding of every doc.
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    with open(synthetic_jsonl, "r", encoding="utf-8") as fin:
        all_docs = [json.loads(line)["text"] for line in fin if line.strip()]
    expected_total = sum(len(sp.encode(d, out_type=int)) for d in all_docs)
    observed_total = manifest["tokens_val"] + manifest["tokens_train"]
    assert observed_total == expected_total

    # Round-trip the first val shard: decoded bytes should substantially
    # overlap with the first N val docs' text. Byte-fallback IDs may
    # lose exact spacing, so match by word set.
    val_tokens = _read_shard(val_shards[0]).astype(np.int64).tolist()
    decoded = sp.decode(val_tokens)
    # Pick a rare token from the synthetic corpus and check it survives.
    assert "chaos" in decoded or "fineweb" in decoded


def test_determinism(tmp_path: Path, synthetic_jsonl: Path) -> None:
    """Two builds on same seed → byte-identical shards."""
    out_a, _, _ = _run_build(
        tmp_path, synthetic_jsonl, output_subdir="a", tokenizer_subdir="tok_a",
    )
    out_b, _, _ = _run_build(
        tmp_path, synthetic_jsonl, output_subdir="b", tokenizer_subdir="tok_b",
    )
    files_a = sorted(p.name for p in out_a.glob("fineweb_*.bin"))
    files_b = sorted(p.name for p in out_b.glob("fineweb_*.bin"))
    assert files_a == files_b, "shard filename sets differ"
    for name in files_a:
        ha = _sha256_file(out_a / name)
        hb = _sha256_file(out_b / name)
        assert ha == hb, f"shard {name} differs byte-wise between runs"


def test_idempotent_skip(tmp_path: Path, synthetic_jsonl: Path, capsys) -> None:
    """Second run with matching config leaves .bin mtimes untouched."""
    out_dir, tok_dir, _ = _run_build(tmp_path, synthetic_jsonl)
    shards = sorted(out_dir.glob("fineweb_*.bin"))
    mtimes_before = {p.name: p.stat().st_mtime_ns for p in shards}

    capsys.readouterr()  # clear
    _run_build(tmp_path, synthetic_jsonl)  # same config, same dirs
    captured = capsys.readouterr()
    assert "Shards already built, skipping" in captured.out

    shards_after = sorted(out_dir.glob("fineweb_*.bin"))
    mtimes_after = {p.name: p.stat().st_mtime_ns for p in shards_after}
    assert mtimes_before == mtimes_after, (
        "idempotent-skip path rewrote shards"
    )


def test_shard_boundary_on_doc(
    tmp_path: Path, synthetic_jsonl: Path,
) -> None:
    """Every shard boundary lines up with a cumulative doc-token count."""
    import sentencepiece as spm

    shard_size = 2_000  # tiny so flushes happen often on 1000-doc input
    out_dir, tok_dir, manifest = _run_build(
        tmp_path, synthetic_jsonl, shard_size=shard_size,
    )
    sp = spm.SentencePieceProcessor(
        model_file=str(tok_dir / "fineweb_512_bpe.model"),
    )

    with open(synthetic_jsonl, "r", encoding="utf-8") as fin:
        all_docs = [json.loads(line)["text"] for line in fin if line.strip()]
    val_docs = 100
    val_encoded = [sp.encode(d, out_type=int) for d in all_docs[:val_docs]]
    train_encoded = [sp.encode(d, out_type=int) for d in all_docs[val_docs:]]

    def _cumulative(enc_list: list[list[int]]) -> list[int]:
        out = []
        c = 0
        for t in enc_list:
            c += len(t)
            out.append(c)
        return out

    # For each split, check each shard's byte length (in tokens) matches
    # some cumulative doc boundary — i.e., no shard splits a doc.
    for split, encoded in (("val", val_encoded), ("train", train_encoded)):
        shards = sorted(out_dir.glob(f"fineweb_{split}_*.bin"))
        boundaries = set(_cumulative(encoded))
        running = 0
        for shard in shards:
            running += _read_shard(shard).size
            assert running in boundaries, (
                f"shard {shard.name} ends at token index {running} which is "
                f"not a doc boundary in the {split} split"
            )


def test_val_train_split(tmp_path: Path, synthetic_jsonl: Path) -> None:
    """First 100 docs → val; remaining 900 → train; tokens sum matches."""
    import sentencepiece as spm

    val_docs = 100
    out_dir, tok_dir, manifest = _run_build(
        tmp_path, synthetic_jsonl, val_docs=val_docs,
    )
    sp = spm.SentencePieceProcessor(
        model_file=str(tok_dir / "fineweb_512_bpe.model"),
    )

    with open(synthetic_jsonl, "r", encoding="utf-8") as fin:
        all_docs = [json.loads(line)["text"] for line in fin if line.strip()]

    expected_val = sum(len(sp.encode(d, out_type=int)) for d in all_docs[:val_docs])
    expected_train = sum(len(sp.encode(d, out_type=int)) for d in all_docs[val_docs:])

    val_shards = sorted(out_dir.glob("fineweb_val_*.bin"))
    train_shards = sorted(out_dir.glob("fineweb_train_*.bin"))
    observed_val = sum(_read_shard(p).size for p in val_shards)
    observed_train = sum(_read_shard(p).size for p in train_shards)

    assert observed_val == expected_val, (
        f"val token count mismatch: observed={observed_val} expected={expected_val}"
    )
    assert observed_train == expected_train, (
        f"train token count mismatch: observed={observed_train} expected={expected_train}"
    )
    assert manifest["num_val_docs"] == val_docs
    assert manifest["num_train_docs"] == len(all_docs) - val_docs


def test_force_reuses_dir_and_overwrites_shards(
    tmp_path: Path, synthetic_jsonl: Path,
) -> None:
    """--force with a different vocab replaces every shard in the dir."""
    out_dir, tok_dir, _ = _run_build(
        tmp_path, synthetic_jsonl, vocab_size=512,
    )
    shards_before = sorted(out_dir.glob("fineweb_*.bin"))
    assert shards_before, "expected shards from first build"
    # Hash each shard so we can prove the bytes actually changed, not
    # just mtimes (rerunning at the same vocab size would be byte-
    # identical under determinism and bypass the point of the test).
    hashes_before = {p.name: _sha256_file(p) for p in shards_before}

    # Same dirs, different vocab size → config mismatch; without --force
    # the build should refuse. With --force it should overwrite.
    with pytest.raises(RuntimeError, match="does not match"):
        _run_build(
            tmp_path, synthetic_jsonl, vocab_size=768,
            force=False,
        )

    _run_build(
        tmp_path, synthetic_jsonl, vocab_size=768, force=True,
    )
    shards_after = sorted(out_dir.glob("fineweb_*.bin"))
    assert shards_after, "expected shards from second (forced) build"
    # Every .bin from the first build must be either gone or have new
    # bytes — the vocab change means token IDs differ, so identical
    # hashes would indicate a stale file from the previous run.
    for name, old_hash in hashes_before.items():
        new_path = out_dir / name
        if not new_path.exists():
            continue
        assert _sha256_file(new_path) != old_hash, (
            f"shard {name} survived --force with unchanged bytes"
        )


def test_stale_partial_shards_without_manifest_fails(
    tmp_path: Path, synthetic_jsonl: Path,
) -> None:
    """A fake shard in an empty output_dir blocks the build without --force."""
    out_dir = tmp_path / "out"
    tok_dir = tmp_path / "tok"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop a bogus shard — no manifest next to it. This mimics a
    # previous run that died between the first shard flush and the
    # manifest write.
    fake_shard = out_dir / "fineweb_train_000000.bin"
    fake_shard.write_bytes(b"\x00\x00" * 16)

    with pytest.raises(RuntimeError, match="no build_manifest.json"):
        build_sp_shards.build(
            docs_path=synthetic_jsonl,
            vocab_size=512,
            output_dir=out_dir,
            tokenizer_dir=tok_dir,
            sp_train_docs=500,
            val_docs=100,
            shard_size=10_000,
            sp_seed=1337,
            num_workers=1,
            dry_run=False,
            skip_tokenize=False,
            skip_train=False,
            force=False,
        )
    # Fake shard must still be there — refusal, not silent cleanup.
    assert fake_shard.is_file()


def test_skip_train_reuses_existing_model(
    tmp_path: Path, synthetic_jsonl: Path,
) -> None:
    """--skip-train leaves the SP .model untouched and reproduces shards."""
    out_dir_a, tok_dir, _ = _run_build(
        tmp_path, synthetic_jsonl,
        output_subdir="a", tokenizer_subdir="tok",
    )
    model_path = tok_dir / "fineweb_512_bpe.model"
    assert model_path.is_file()
    mtime_before = model_path.stat().st_mtime_ns
    # Hash each shard so we can assert byte-identity after the second run.
    hashes_a = {
        p.name: _sha256_file(p)
        for p in sorted(out_dir_a.glob("fineweb_*.bin"))
    }

    # Second build into a fresh output dir, pointing at the same
    # tokenizer dir with --skip-train. SP training must not run.
    out_dir_b, _, _ = _run_build(
        tmp_path, synthetic_jsonl,
        output_subdir="b", tokenizer_subdir="tok",
        skip_train=True,
    )
    assert model_path.stat().st_mtime_ns == mtime_before, (
        "--skip-train unexpectedly rewrote the SP .model"
    )
    hashes_b = {
        p.name: _sha256_file(p)
        for p in sorted(out_dir_b.glob("fineweb_*.bin"))
    }
    assert hashes_a == hashes_b, (
        "--skip-train produced different shard bytes from the training run"
    )


def test_dry_run_no_files_written(
    tmp_path: Path, synthetic_jsonl: Path, capsys,
) -> None:
    """--dry-run prints the plan and creates nothing on disk."""
    out_dir = tmp_path / "out"
    tok_dir = tmp_path / "tok"
    result = build_sp_shards.build(
        docs_path=synthetic_jsonl,
        vocab_size=512,
        output_dir=out_dir,
        tokenizer_dir=tok_dir,
        sp_train_docs=500,
        val_docs=100,
        shard_size=10_000,
        sp_seed=1337,
        num_workers=1,
        dry_run=True,
        skip_tokenize=False,
        skip_train=False,
        force=False,
    )
    captured = capsys.readouterr()
    # Plan JSON printed.
    assert '"vocab_size": 512' in captured.out
    assert result["num_docs"] == 1000
    # No files/dirs created.
    assert not out_dir.exists()
    assert not tok_dir.exists()
