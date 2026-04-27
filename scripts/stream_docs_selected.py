"""Stream-download the first N docs of docs_selected.jsonl from HF.

The full docs_selected.jsonl in willdepueoai/parameter-golf is 48 GB —
too big to keep on a 100 GB pod volume alongside the train shards. The
val cache builder only consumes ``--max-docs N`` (default 50_000) docs,
so streaming exactly N docs via HTTP Range and stopping at the Nth
newline gives us identical val cache contents at <1% of the disk cost.

Discovered the hard way on 2026-04-27: cached_challenge_fineweb.py
crashed twice with disk-full errors during HF cache writes.

Usage:
    HF_TOKEN=... python scripts/stream_docs_selected.py
    HF_TOKEN=... python scripts/stream_docs_selected.py --target-docs 100000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

DEFAULT_REPO = "willdepueoai/parameter-golf"
DEFAULT_PATH = "datasets/docs_selected.jsonl"
DEFAULT_REVISION = "main"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "baselines/parameter_golf/datasets/docs_selected.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--path", default=DEFAULT_PATH)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--target-docs", type=int, default=50_000,
                   help="Stop after the Nth newline (default 50000, matches "
                        "build_exp20_val_cache.py default --max-docs)")
    p.add_argument("--chunk-bytes", type=int, default=1 << 20,
                   help="HTTP read chunk size in bytes (default 1 MiB)")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var required (the willdepueoai/parameter-golf "
              "repo requires auth)", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://huggingface.co/datasets/{args.repo}/resolve/{args.revision}/{args.path}"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"streaming first {args.target_docs} docs from {url}", flush=True)

    doc_count = 0
    bytes_total = 0
    buf = b""
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        print(f"status={r.status_code} content-length={r.headers.get('content-length')}",
              flush=True)
        with args.out.open("wb") as out:
            for chunk in r.iter_content(chunk_size=args.chunk_bytes):
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf and doc_count < args.target_docs:
                    line, _, rest = buf.partition(b"\n")
                    out.write(line + b"\n")
                    bytes_total += len(line) + 1
                    buf = rest
                    doc_count += 1
                    if doc_count % 5000 == 0:
                        print(f"docs={doc_count} bytes={bytes_total/1024**2:.1f} MiB",
                              flush=True)
                if doc_count >= args.target_docs:
                    break

    print(f"DONE: docs={doc_count} bytes={bytes_total/1024**2:.1f} MiB at {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
