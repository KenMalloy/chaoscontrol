#!/usr/bin/env python3
"""Run the packet-online-cache seeded-vs-empty compare and persist a summary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Match the standalone script pattern used by the rest of the repo so
# ``chaoscontrol`` imports resolve without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chaoscontrol.eval.packet_online_cache_compare import (
    load_and_run_packet_online_cache_compare,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-path", required=True, type=Path)
    p.add_argument("--val-cache-dir", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--write-tokens-per-chunk", type=int, default=16)
    p.add_argument("--gate-value", type=float, default=1.0)
    p.add_argument("--decay", type=float, default=1.0)
    args = p.parse_args(argv)

    result = load_and_run_packet_online_cache_compare(
        artifact_path=args.artifact_path,
        val_cache_dir=args.val_cache_dir,
        device=args.device,
        compare_config={
            "chunk_tokens": int(args.chunk_tokens),
            "write_tokens_per_chunk": int(args.write_tokens_per_chunk),
            "gate_value": float(args.gate_value),
            "decay": float(args.decay),
        },
        output_json=args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
