#!/usr/bin/env python3
"""Build the generated validation cache used by Exp20 fast eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chaoscontrol.eval_stream.val_cache import write_val_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-path", type=Path, nargs="+", required=True)
    parser.add_argument("--sp-model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=50_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    had_manifest = (args.cache_dir / "manifest.json").is_file()
    manifest = write_val_cache(
        jsonl_paths=args.jsonl_path,
        sp_model_path=args.sp_model_path,
        cache_dir=args.cache_dir,
        max_docs=args.max_docs,
        force=args.force,
    )
    status = "existing" if had_manifest and not args.force else "built"
    print(
        " ".join([
            f"cache_status={status}",
            f"cache_dir={args.cache_dir}",
            f"num_docs={manifest['num_docs']}",
            f"total_tokens={manifest['total_tokens']}",
            f"total_raw_bytes={manifest['total_raw_bytes']}",
        ])
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
