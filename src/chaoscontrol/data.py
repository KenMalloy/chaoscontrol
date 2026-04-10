"""Self-contained data utilities for ChaosControl experiments.

Extracted from parameter-golf/tools/evolutionary_benchmark.py and
parameter-golf/spectral_flood_walk_v2a.py.
"""
from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_param_dtype(name: str, device: torch.device) -> torch.dtype:
    requested = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]
    if device.type != "cuda":
        return torch.float32
    return requested


def maybe_autocast(device: torch.device, dtype: torch.dtype) -> Any:
    enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def maybe_cache_tokens_on_device(tokens: torch.Tensor, *, device: torch.device, enabled: bool) -> torch.Tensor:
    if enabled and device.type == "cuda" and tokens.device != device:
        return tokens.to(device=device, dtype=torch.long)
    return tokens


def maybe_sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _concat_shards_mmap(shard_paths: list[Path], cache_path: Path) -> np.ndarray:
    """Concatenate binary shards into a single memory-mapped file.

    On first call, reads all shards and writes a flat binary cache.
    On subsequent calls (including from parallel processes), returns an
    mmap view of the cache — all processes share the same physical pages.

    Race-safe: uses per-PID temp files and retries if another process
    finishes the cache first.
    """
    import os
    import time

    if cache_path.exists():
        return np.memmap(str(cache_path), dtype=np.uint16, mode="c")

    # Use per-PID temp file to avoid collisions between parallel processes
    tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
    try:
        arrays = [np.fromfile(str(s), dtype=np.uint16) for s in shard_paths]
        combined = np.concatenate(arrays)
        combined.tofile(str(tmp_path))
        del combined, arrays
        # Atomic rename — if another process beat us, that's fine
        try:
            tmp_path.rename(cache_path)
        except OSError:
            # Another process already created the cache; clean up our temp
            tmp_path.unlink(missing_ok=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # Wait briefly if cache doesn't exist yet (another process still writing)
    for _ in range(30):
        if cache_path.exists():
            break
        time.sleep(0.5)

    return np.memmap(str(cache_path), dtype=np.uint16, mode="c")


def load_fineweb_tokens(data_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load FineWeb binary shards (uint16 SentencePiece token IDs).

    Returns (train_tokens, val_tokens) as int64 tensors backed by
    memory-mapped files. Multiple processes sharing the same data_dir
    will share physical memory pages via the OS page cache.
    """
    data_path = Path(data_dir)
    train_shards = sorted(data_path.glob("fineweb_train_*.bin"))
    val_shards = sorted(data_path.glob("fineweb_val_*.bin"))
    if not train_shards:
        raise FileNotFoundError(f"No training shards found in {data_dir}")
    if not val_shards:
        raise FileNotFoundError(f"No validation shards found in {data_dir}")

    train_mmap = _concat_shards_mmap(train_shards, data_path / ".train_cache.bin")
    val_mmap = _concat_shards_mmap(val_shards, data_path / ".val_cache.bin")

    # Zero-copy: view uint16 mmap as int16 (identical bit patterns for
    # values < 32768; sp1024 vocab max is 1023). torch.from_numpy shares
    # the mmap backing. Multiple processes reading the same cache file
    # share OS page cache — no per-process duplication.
    # batch_from_starts does .to(dtype=torch.long) per-batch.
    train_tokens = torch.from_numpy(train_mmap.view(np.int16))
    val_tokens = torch.from_numpy(val_mmap.view(np.int16))
    return train_tokens, val_tokens


def _extract_jsonl_to_raw(jsonl_path: Path, raw_path: Path) -> None:
    """Extract raw UTF-8 text from docs_selected.jsonl to a flat bytes file.

    Each line in the JSONL has a "text" field. We concatenate all text
    with newline separators and write as raw bytes.
    """
    import json as _json
    tmp = raw_path.with_suffix(".tmp")
    skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(tmp, "wb") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                doc = _json.loads(line)
            except _json.JSONDecodeError:
                skipped += 1
                continue
            text = doc.get("text", "")
            fout.write(text.encode("utf-8"))
            fout.write(b"\n")
    if skipped:
        print(f"  Skipped {skipped} malformed JSONL lines")
    tmp.rename(raw_path)


# Competition validation split: first 50k documents
_COMPETITION_VAL_DOCS = 50_000


def _extract_jsonl_split(
    jsonl_path: Path, val_path: Path, train_path: Path,
    val_docs: int = _COMPETITION_VAL_DOCS,
) -> None:
    """Split docs_selected.jsonl into val and train raw text files.

    The competition defines validation as "the fixed first-50k-document set."
    First val_docs lines → val, remainder → train.
    """
    import json as _json
    val_tmp = val_path.with_suffix(".tmp")
    train_tmp = train_path.with_suffix(".tmp")
    skipped = 0
    doc_count = 0
    with (
        open(jsonl_path, "r", encoding="utf-8") as fin,
        open(val_tmp, "wb") as fval,
        open(train_tmp, "wb") as ftrain,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                doc = _json.loads(line)
            except _json.JSONDecodeError:
                skipped += 1
                continue
            text = doc.get("text", "")
            out = fval if doc_count < val_docs else ftrain
            out.write(text.encode("utf-8"))
            out.write(b"\n")
            doc_count += 1
    if skipped:
        print(f"  Skipped {skipped} malformed JSONL lines")
    val_tmp.rename(val_path)
    train_tmp.rename(train_path)
    print(f"  Split: {val_docs} val docs, {doc_count - val_docs} train docs")


def load_fineweb_raw_bytes(text_path: str) -> torch.Tensor:
    """Load a raw UTF-8 text file as a byte tensor via mmap.

    Returns torch.uint8 tensor (values 0-255) backed by mmap.
    batch_from_starts converts to int64 per-batch via .to(dtype=torch.long).
    """
    p = Path(text_path)
    if not p.exists() or p.stat().st_size == 0:
        raise ValueError(f"empty or missing text file: {text_path}")
    # mode="c" = copy-on-write: appears writable (no torch warning)
    # but shares physical pages via mmap. No actual copy unless written to.
    mmap = np.memmap(str(p), dtype=np.uint8, mode="c")
    return torch.from_numpy(mmap)


def prepare_fineweb_splits(
    data_dir: str,
    *,
    device: torch.device,
    cache_on_device: bool = True,
    train_fraction: float = 0.90,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load FineWeb data as raw UTF-8 bytes, returning (train, val, test) tensors.

    Priority:
      1. docs_val_raw.txt + docs_train_raw.txt — document-aware split matching
         the competition's "first 50k documents = validation" definition.
      2. docs_selected.jsonl — extract and split by document boundary, then mmap.
      3. docs_raw.txt — legacy single-file fallback with percentage split.
      4. Raise error.

    All tensors stay on CPU. batch_from_starts handles per-batch GPU transfer.
    """
    data_path = Path(data_dir)
    val_raw = data_path / "docs_val_raw.txt"
    train_raw = data_path / "docs_train_raw.txt"
    jsonl = data_path / "docs_selected.jsonl"

    # Document-aware split: first 50k docs = val (matches competition definition)
    if not val_raw.exists() and jsonl.exists():
        print(f"  Splitting {jsonl} by document boundary (first {_COMPETITION_VAL_DOCS} = val)...")
        _extract_jsonl_split(jsonl, val_raw, train_raw)
        print(f"  val:   {val_raw} ({val_raw.stat().st_size / 1e9:.2f} GB)")
        print(f"  train: {train_raw} ({train_raw.stat().st_size / 1e9:.2f} GB)")

    if val_raw.exists() and train_raw.exists():
        val_tokens = load_fineweb_raw_bytes(str(val_raw))
        train_all = load_fineweb_raw_bytes(str(train_raw))
        # Reserve last 5% of train as held-out test (for warming curve eval)
        test_boundary = int(train_all.numel() * 0.95)
        train_tokens = train_all[:test_boundary]
        test_tokens = train_all[test_boundary:]
        print(f"  Loaded: train={train_tokens.numel():,} val={val_tokens.numel():,} test={test_tokens.numel():,} bytes")
        return train_tokens, val_tokens, test_tokens

    # Legacy fallback: single docs_raw.txt with percentage split
    raw_text = data_path / "docs_raw.txt"
    if not raw_text.exists() and jsonl.exists():
        print(f"  Extracting raw text from {jsonl} (legacy single-file mode)...")
        _extract_jsonl_to_raw(jsonl, raw_text)
        print(f"  Wrote {raw_text} ({raw_text.stat().st_size / 1e9:.2f} GB)")

    if not raw_text.exists():
        raise FileNotFoundError(
            f"No raw text found in {data_dir}. Need docs_selected.jsonl (preferred) "
            f"or docs_raw.txt. Download with: python cached_challenge_fineweb.py --with-docs"
        )

    print("  WARNING: Using legacy percentage split — validation does not match competition's 50k-document set.")
    all_tokens = load_fineweb_raw_bytes(str(raw_text))
    train_end = int(all_tokens.numel() * train_fraction)
    val_end = int(all_tokens.numel() * (train_fraction + 0.05))
    train_tokens = all_tokens[:train_end]
    val_tokens = all_tokens[train_end:val_end]
    test_tokens = all_tokens[val_end:]
    return train_tokens, val_tokens, test_tokens


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------

def build_lm_starts(num_tokens: int, seq_len: int, stride: int) -> list[int]:
    stop = num_tokens - seq_len - 1
    if stop <= 0:
        return []
    return list(range(0, stop, stride))


def batch_from_starts(
    tokens: torch.Tensor,
    starts: list[int],
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    windows = [tokens[start : start + seq_len + 1] for start in starts]
    batch = torch.stack(windows).to(device=device, dtype=torch.long)
    return batch[:, :-1], batch[:, 1:]


def choose_eval_starts(
    starts: list[int],
    *,
    batch_size: int,
    eval_batches: int,
    seed: int,
) -> list[int]:
    needed = batch_size * eval_batches
    if not starts:
        return []
    if len(starts) <= needed:
        return starts[:needed]
    rng = random.Random(seed)
    return rng.sample(starts, needed)
