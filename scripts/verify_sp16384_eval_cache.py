#!/usr/bin/env python3
"""Verify Exp27 ValCache tokens match the prepared SP16384 val shard.

The final TTT scorer needs doc boundaries and raw byte counts, so it consumes
``ValCache`` built from the first 50k docs of ``docs_selected.jsonl``. The
canonical token stream lives in Natooka/parameter-golf-sp-tokenizers as
``shards/fineweb_val_000000.bin``. This preflight proves those two views are
the same token sequence before spending H100 time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chaoscontrol.eval_stream.val_cache import load_val_cache  # noqa: E402


DEFAULT_DATA_DIR = (
    REPO_ROOT / "baselines" / "parameter_golf" / "datasets" / "fineweb10B_sp16384"
)
DEFAULT_VAL_SHARD = DEFAULT_DATA_DIR / "fineweb_val_000000.bin"
DEFAULT_VAL_CACHE_DIR = Path(
    os.environ.get(
        "VAL_CACHE_DIR",
        str(REPO_ROOT / "experiments" / "27_ttt_headline" / "val_cache"),
    )
)
EXPECTED_SP16384_VAL_TOKENS = 42_266_034


def _first_mismatch(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    chunk_tokens: int,
) -> tuple[int, int, int] | None:
    n = int(lhs.shape[0])
    for start in range(0, n, int(chunk_tokens)):
        end = min(n, start + int(chunk_tokens))
        a = lhs[start:end]
        b = rhs[start:end]
        if np.array_equal(a, b):
            continue
        rel = np.nonzero(a != b)[0]
        idx = start + int(rel[0])
        return idx, int(lhs[idx]), int(rhs[idx])
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-shard", type=Path, default=DEFAULT_VAL_SHARD)
    parser.add_argument("--val-cache-dir", type=Path, default=DEFAULT_VAL_CACHE_DIR)
    parser.add_argument("--expected-tokens", type=int, default=EXPECTED_SP16384_VAL_TOKENS)
    parser.add_argument("--chunk-tokens", type=int, default=1 << 22)
    args = parser.parse_args(argv)

    if not args.val_shard.is_file():
        raise FileNotFoundError(f"SP16384 val shard missing: {args.val_shard}")
    if not args.val_cache_dir.is_dir():
        raise FileNotFoundError(f"ValCache directory missing: {args.val_cache_dir}")

    shard = np.memmap(args.val_shard, dtype=np.uint16, mode="r")
    cache = load_val_cache(args.val_cache_dir)
    tokens = cache.tokens

    if int(shard.shape[0]) != int(args.expected_tokens):
        raise ValueError(
            f"val shard token count {int(shard.shape[0])} != "
            f"expected {int(args.expected_tokens)}"
        )
    if int(tokens.shape[0]) != int(shard.shape[0]):
        raise ValueError(
            f"ValCache token count {int(tokens.shape[0])} != "
            f"SP16384 val shard token count {int(shard.shape[0])}"
        )
    mismatch = _first_mismatch(
        shard,
        tokens,
        chunk_tokens=max(1, int(args.chunk_tokens)),
    )
    if mismatch is not None:
        idx, shard_value, cache_value = mismatch
        raise ValueError(
            "ValCache token stream differs from SP16384 val shard at "
            f"index={idx}: shard={shard_value} cache={cache_value}"
        )

    out = {
        "ok": True,
        "val_shard": str(args.val_shard),
        "val_cache_dir": str(args.val_cache_dir),
        "tokens": int(tokens.shape[0]),
        "docs": int(cache.num_docs),
        "raw_bytes": int(cache.total_raw_bytes),
        "cache_content_sha256": str(cache.manifest.get("cache_content_sha256", "")),
    }
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
