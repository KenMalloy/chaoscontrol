"""Validate CRCT dim=384 artifact headroom before pod sweeps.

The fast local path estimates compressed size from the repo's measured
dim=256 int6+LZMA artifact.  The pod path can add a real quantized pack check
later; this script already exposes the same JSON contract for both.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chaoscontrol.model import ChaosStudentLM


MEASURED_DIM256_COMPRESSED_BYTES = 7_375_816
ARTIFACT_LIMIT_BYTES = 16_000_000
DEFAULT_OVERHEAD_BYTES = 500_000


@dataclass
class ArtifactHeadroom:
    dim: int
    vocab_size: int
    num_layers: int
    raw_bf16_bytes: int
    baseline_dim256_raw_bf16_bytes: int
    estimated_compressed_model_bytes: int
    overhead_bytes: int
    estimated_total_bytes: int
    margin_bytes: int
    controller_params: int
    under_budget: bool


def build_crct_model(
    *,
    dim: int,
    vocab_size: int,
    num_layers: int,
    ff_mult: int,
) -> ChaosStudentLM:
    return ChaosStudentLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        ff_mult=ff_mult,
        a_mode="diag",
        rich_b_mode="none",
        outer_model_dim=0,
    )


def estimate_artifact_headroom(
    *,
    dim: int = 384,
    vocab_size: int = 16_384,
    num_layers: int = 4,
    ff_mult: int = 2,
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES,
) -> ArtifactHeadroom:
    model = build_crct_model(
        dim=dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        ff_mult=ff_mult,
    )
    baseline = build_crct_model(
        dim=256,
        vocab_size=vocab_size,
        num_layers=num_layers,
        ff_mult=ff_mult,
    )
    raw = int(model.artifact_bytes())
    baseline_raw = int(baseline.artifact_bytes())
    ratio = MEASURED_DIM256_COMPRESSED_BYTES / float(baseline_raw)
    compressed = int(round(raw * ratio))
    total = compressed + int(overhead_bytes)
    return ArtifactHeadroom(
        dim=int(dim),
        vocab_size=int(vocab_size),
        num_layers=int(num_layers),
        raw_bf16_bytes=raw,
        baseline_dim256_raw_bf16_bytes=baseline_raw,
        estimated_compressed_model_bytes=compressed,
        overhead_bytes=int(overhead_bytes),
        estimated_total_bytes=total,
        margin_bytes=ARTIFACT_LIMIT_BYTES - total,
        controller_params=0,
        under_budget=total < ARTIFACT_LIMIT_BYTES,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--overhead-bytes", type=int, default=DEFAULT_OVERHEAD_BYTES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    result = estimate_artifact_headroom(
        dim=args.dim,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        ff_mult=args.ff_mult,
        overhead_bytes=args.overhead_bytes,
    )
    text = json.dumps(asdict(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0 if result.under_budget else 2


if __name__ == "__main__":
    raise SystemExit(main())
