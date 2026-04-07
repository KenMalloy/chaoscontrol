import tempfile
from pathlib import Path

import torch
from chaoscontrol.data import (
    batch_from_starts, build_lm_starts, resolve_device, resolve_param_dtype,
    maybe_autocast, maybe_sync_cuda, choose_eval_starts,
    load_fineweb_raw_bytes,
)

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
    """load_fineweb_raw_bytes should convert a UTF-8 file to int64 byte values."""
    content = b"Hello, world!\n"
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        tokens = load_fineweb_raw_bytes(f.name)
    assert tokens.dtype == torch.int64
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
