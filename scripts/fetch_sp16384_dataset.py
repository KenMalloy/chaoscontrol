"""Fetch SP16384 train+val shards + tokenizer from Natooka/parameter-golf-sp-tokenizers.

This is the canonical training-side dataset for the SP16384-based runs.
docs_selected.jsonl (the source for val cache / BPB) lives in a DIFFERENT
repo (willdepueoai/parameter-golf) and is fetched separately by
``scripts/stream_docs_selected.py``.

Idempotent: each file is size-checked and skipped if already on disk.
Hardlinks from HF cache to destination so disk isn't doubled.

Usage:
    python scripts/fetch_sp16384_dataset.py
    HF_TOKEN=... python scripts/fetch_sp16384_dataset.py  # only needed for rate limits
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Natooka/parameter-golf-sp-tokenizers"
REVISION = "e9d696d1592d884dbb97e754efb2a7203aca3080"
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("HF_HOME", "/workspace/hf_cache_natooka"))
DATA_DIR = REPO_ROOT / "baselines/parameter_golf/datasets/fineweb10B_sp16384"
TOK_DIR = REPO_ROOT / "baselines/parameter_golf/tokenizers"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        s.rfilename
        for s in HfApi().repo_info(REPO_ID, repo_type="dataset").siblings
    )
    wanted = [
        f
        for f in files
        if f in {"fineweb_16384_bpe.model", "fineweb_16384_bpe.vocab"}
        or (f.startswith("shards/fineweb_train_") and f.endswith(".bin"))
        or (f.startswith("shards/fineweb_val_") and f.endswith(".bin"))
    ]
    print(f"plan: {len(wanted)} files from {REPO_ID}@{REVISION[:12]}", flush=True)

    def dest_for(remote_name: str) -> Path:
        name = Path(remote_name).name
        return DATA_DIR / name if remote_name.startswith("shards/") else TOK_DIR / name

    def fetch(remote_name: str) -> tuple[str, int]:
        dest = dest_for(remote_name)
        if dest.exists() and dest.stat().st_size > 0:
            return remote_name, dest.stat().st_size
        cached = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=remote_name,
                revision=REVISION,
                cache_dir=str(CACHE_DIR),
            )
        ).resolve(strict=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.link(cached, tmp)
        except OSError:
            import shutil
            shutil.copy2(cached, tmp)
        os.replace(tmp, dest)
        return remote_name, dest.stat().st_size

    start = time.monotonic()
    done_bytes = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, f) for f in wanted]
        for i, future in enumerate(as_completed(futures), start=1):
            name, size = future.result()
            done_bytes += size
            if i <= 5 or i % 20 == 0 or i == len(futures):
                rate = done_bytes / 1024**2 / max(time.monotonic() - start, 1e-9)
                print(
                    f"{i}/{len(futures)} {name} total={done_bytes / 1024**3:.2f}GiB rate={rate:.1f}MiB/s",
                    flush=True,
                )

    train = list(DATA_DIR.glob("fineweb_train_*.bin"))
    val = list(DATA_DIR.glob("fineweb_val_*.bin"))
    assert len(train) == 133, f"expected 133 train shards, got {len(train)}"
    assert len(val) == 1, f"expected 1 val shard, got {len(val)}"
    assert (TOK_DIR / "fineweb_16384_bpe.model").is_file()
    print("SP16384 ready", flush=True)


if __name__ == "__main__":
    main()
