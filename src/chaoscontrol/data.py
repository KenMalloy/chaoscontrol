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

def load_fineweb_tokens(data_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load FineWeb binary shards (uint16 SentencePiece token IDs).

    Returns (train_tokens, val_tokens) as int64 tensors.
    The competition format stores shards as flat binary files of uint16.
    """
    data_path = Path(data_dir)
    train_shards = sorted(data_path.glob("fineweb_train_*.bin"))
    val_shards = sorted(data_path.glob("fineweb_val_*.bin"))
    if not train_shards:
        raise FileNotFoundError(f"No training shards found in {data_dir}")
    if not val_shards:
        raise FileNotFoundError(f"No validation shards found in {data_dir}")

    train_arrays = [np.fromfile(str(s), dtype=np.uint16) for s in train_shards]
    val_arrays = [np.fromfile(str(s), dtype=np.uint16) for s in val_shards]

    train_tokens = torch.from_numpy(np.concatenate(train_arrays).astype(np.int64, copy=False))
    val_tokens = torch.from_numpy(np.concatenate(val_arrays).astype(np.int64, copy=False))
    return train_tokens, val_tokens


def load_fineweb_raw_bytes(text_path: str) -> torch.Tensor:
    """Load a raw UTF-8 text file as a uint8 byte tensor (int64 for LM use).

    This is the raw-bytes approach: each byte is a token in [0, 255].
    """
    p = Path(text_path)
    raw = p.read_bytes()
    if len(raw) == 0:
        raise ValueError(f"empty text file: {text_path}")
    return torch.tensor(list(raw), dtype=torch.int64)


def prepare_fineweb_splits(
    data_dir: str,
    *,
    device: torch.device,
    cache_on_device: bool = True,
    train_fraction: float = 0.90,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load FineWeb data, returning (train, val, test) tensors.

    If a raw text file (docs_raw.txt) exists in data_dir, uses raw bytes.
    Otherwise falls back to the uint16 shard format (SentencePiece tokens).
    """
    data_path = Path(data_dir)
    raw_text = data_path / "docs_raw.txt"

    if raw_text.exists():
        # Raw bytes path
        all_tokens = load_fineweb_raw_bytes(str(raw_text))
        train_end = int(all_tokens.numel() * train_fraction)
        val_end = int(all_tokens.numel() * (train_fraction + 0.05))
        train_tokens = all_tokens[:train_end]
        val_tokens = all_tokens[train_end:val_end]
        test_tokens = all_tokens[val_end:]
    else:
        # SentencePiece uint16 shard path — train/val already split by shards
        train_tokens, val_tokens = load_fineweb_tokens(data_dir)
        # No separate test set in competition format; carve 10% off val
        split_at = int(val_tokens.numel() * 0.5)
        test_tokens = val_tokens[split_at:]
        val_tokens = val_tokens[:split_at]

    return (
        maybe_cache_tokens_on_device(train_tokens, device=device, enabled=cache_on_device),
        maybe_cache_tokens_on_device(val_tokens, device=device, enabled=cache_on_device),
        maybe_cache_tokens_on_device(test_tokens, device=device, enabled=cache_on_device),
    )


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
