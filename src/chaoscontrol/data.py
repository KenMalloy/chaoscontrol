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
# enwik8 loading
# ---------------------------------------------------------------------------

def load_enwik8_splits(
    path: Path,
    *,
    train_fraction: float = 0.90,
    val_fraction: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"empty enwik8 file: {path}")
    tokens = torch.from_numpy(raw.astype(np.int64, copy=False))
    train_end = int(tokens.numel() * train_fraction)
    val_end = int(tokens.numel() * (train_fraction + val_fraction))
    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    test_tokens = tokens[val_end:]
    if min(train_tokens.numel(), val_tokens.numel(), test_tokens.numel()) <= 0:
        raise ValueError("enwik8 split produced an empty partition")
    return train_tokens, val_tokens, test_tokens


def prepare_tokenized_enwik8_splits(path, *, device, cache_on_device=True):
    train, val, test = load_enwik8_splits(path)
    return (
        maybe_cache_tokens_on_device(train, device=device, enabled=cache_on_device),
        maybe_cache_tokens_on_device(val, device=device, enabled=cache_on_device),
        maybe_cache_tokens_on_device(test, device=device, enabled=cache_on_device),
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
