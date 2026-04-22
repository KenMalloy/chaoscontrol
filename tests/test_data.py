import tempfile
from pathlib import Path

import numpy as np
import torch
from chaoscontrol.data import (
    batch_from_starts, build_lm_starts, resolve_device, resolve_param_dtype,
    maybe_autocast, maybe_sync_cuda, choose_eval_starts,
    load_fineweb_raw_bytes,
)
from chaoscontrol.data import _concat_shards_mmap

def test_build_lm_starts():
    starts = build_lm_starts(100, seq_len=10, stride=5)
    assert len(starts) > 0
    assert all(s + 10 + 1 <= 100 for s in starts)

def test_batch_from_starts():
    tokens = torch.arange(100)
    starts = [0, 10, 20]
    inputs, targets = batch_from_starts(tokens, starts, seq_len=8, device=torch.device("cpu"))
    assert inputs.shape == (3, 8)
    assert targets.shape == (3, 8)

def test_resolve_device():
    d = resolve_device("cpu")
    assert d == torch.device("cpu")

def test_resolve_param_dtype():
    d = resolve_param_dtype("fp32", torch.device("cpu"))
    assert d == torch.float32

def test_choose_eval_starts():
    starts = list(range(0, 100, 5))
    selected = choose_eval_starts(starts, batch_size=4, eval_batches=2, seed=42)
    assert len(selected) == 8


# ---------------------------------------------------------------------------
# FineWeb raw bytes tests
# ---------------------------------------------------------------------------

def test_load_fineweb_raw_bytes():
    """load_fineweb_raw_bytes should return mmap-backed byte tensor."""
    content = b"Hello, world!\n"
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        tokens = load_fineweb_raw_bytes(f.name)
    assert tokens.dtype == torch.uint8
    assert tokens.numel() == len(content)
    assert tokens[0].item() == ord("H")
    assert tokens[-1].item() == ord("\n")
    # All values in [0, 255]
    assert tokens.min().item() >= 0
    assert tokens.max().item() <= 255


def test_load_fineweb_raw_bytes_multibyte_utf8():
    """Multi-byte UTF-8 characters should produce multiple byte tokens."""
    # U+00E9 (e-acute) is 2 bytes in UTF-8: 0xC3 0xA9
    content = "\u00e9".encode("utf-8")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        tokens = load_fineweb_raw_bytes(f.name)
    assert tokens.numel() == 2
    assert tokens[0].item() == 0xC3
    assert tokens[1].item() == 0xA9


def test_concat_shards_mmap_matches_sequential(tmp_path):
    """Parallel shard read must produce identical bytes to sequential concat."""
    rng = np.random.default_rng(0)
    shards = []
    expected_parts = []
    for i in range(5):
        shard_path = tmp_path / f"shard_{i}.bin"
        data = rng.integers(0, 65535, size=1000 + i * 123, dtype=np.uint16)
        data.tofile(str(shard_path))
        shards.append(shard_path)
        expected_parts.append(data)
    expected = np.concatenate(expected_parts)

    cache_path = tmp_path / ".test_cache.bin"
    result = _concat_shards_mmap(shards, cache_path)

    assert cache_path.exists()
    assert result.shape == expected.shape
    assert np.array_equal(np.asarray(result), expected)


def test_concat_shards_mmap_reuses_cache(tmp_path):
    """Second call with existing cache must return mmap without rebuilding."""
    shard_path = tmp_path / "shard_0.bin"
    data = np.arange(100, dtype=np.uint16)
    data.tofile(str(shard_path))
    cache_path = tmp_path / ".test_cache.bin"

    first = _concat_shards_mmap([shard_path], cache_path)
    mtime_first = cache_path.stat().st_mtime_ns

    # Replace the shard on disk — if the cache is rebuilt, the mtime changes.
    other = np.arange(100, 200, dtype=np.uint16)
    other.tofile(str(shard_path))

    second = _concat_shards_mmap([shard_path], cache_path)
    assert cache_path.stat().st_mtime_ns == mtime_first
    assert np.array_equal(np.asarray(second), np.asarray(first))


def test_batch_from_starts_with_raw_bytes():
    """batch_from_starts should work with tensors from load_fineweb_raw_bytes."""
    content = b"The quick brown fox jumps over the lazy dog."
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        tokens = load_fineweb_raw_bytes(f.name)
    starts = [0, 5, 10]
    inputs, targets = batch_from_starts(tokens, starts, seq_len=8, device=torch.device("cpu"))
    assert inputs.shape == (3, 8)
    assert targets.shape == (3, 8)
    # Targets are one-step-ahead of inputs
    assert inputs[0, 1].item() == targets[0, 0].item()
