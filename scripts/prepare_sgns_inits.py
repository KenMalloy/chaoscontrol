#!/usr/bin/env python3
"""Derive Exp 21 embedding-init tensors from a trained SGNS artifact.

Produces three ``(vocab_size, dim)`` tensors from one SGNS training run:
  - ``meanstd``:   row-norm mean+std matched to a reference Gaussian init.
                   Isolates "semantic geometry" from "distributional scale."
  - ``fullcov``:   full row-covariance matched to reference (Cholesky
                   whitening + re-coloring). Stronger control — distributions
                   match in all moments captured by covariance, so any remaining
                   delta is directional (semantic).
  - ``shuffled``:  meanstd tensor with rows randomly permuted. Preserves
                   the marginal distribution but destroys token-ID → vector
                   mapping. Positive control: any measured "SGNS helps" effect
                   must vanish here, else the signal is distributional, not
                   semantic.

Reference tensor is a standard Gaussian matched to the model's init scale
(nn.Embedding default: ``std=1.0``). Passing a different reference is
supported for comparability with alternate init schemes.

Usage:
    python scripts/prepare_sgns_inits.py \\
        --sgns artifacts/sgns_v8192_d256.pt \\
        --out-dir artifacts/ \\
        --shuffled-seed 42

Outputs:
    artifacts/sgns_init_meanstd.pt
    artifacts/sgns_init_fullcov.pt
    artifacts/sgns_init_shuffled.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.sgns.moment_match import (  # noqa: E402
    match_full_covariance,
    match_row_norm_moments,
    shuffle_rows,
)


def _reference_init(vocab_size: int, dim: int, seed: int) -> torch.Tensor:
    """Reference Gaussian matching nn.Embedding's default (std=1)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(vocab_size, dim, generator=g)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive meanstd + fullcov + shuffled init variants from SGNS output"
    )
    parser.add_argument("--sgns", required=True, help="Path to sgns_vN_dD.pt")
    parser.add_argument("--out-dir", required=True, help="Directory for init_*.pt outputs")
    parser.add_argument("--reference-seed", type=int, default=0,
                        help="Seed for reference Gaussian (for moment matching target)")
    parser.add_argument("--shuffled-seed", type=int, default=42,
                        help="Seed for the shuffled-row control permutation")
    args = parser.parse_args(argv)

    sgns_path = Path(args.sgns)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sgns = torch.load(str(sgns_path), map_location="cpu")
    if sgns.dim() != 2:
        raise ValueError(
            f"Expected SGNS tensor of shape (V, D); got {tuple(sgns.shape)}"
        )
    vocab_size, dim = sgns.shape
    sgns = sgns.to(dtype=torch.float32)

    reference = _reference_init(vocab_size, dim, args.reference_seed)

    meanstd = match_row_norm_moments(sgns, reference)
    fullcov = match_full_covariance(sgns, reference)
    shuffled = shuffle_rows(meanstd, args.shuffled_seed)

    meanstd_path = out_dir / "sgns_init_meanstd.pt"
    fullcov_path = out_dir / "sgns_init_fullcov.pt"
    shuffled_path = out_dir / "sgns_init_shuffled.pt"
    torch.save(meanstd, str(meanstd_path))
    torch.save(fullcov, str(fullcov_path))
    torch.save(shuffled, str(shuffled_path))

    print(f"Wrote {meanstd_path}  shape={tuple(meanstd.shape)}")
    print(f"Wrote {fullcov_path}  shape={tuple(fullcov.shape)}")
    print(f"Wrote {shuffled_path} shape={tuple(shuffled.shape)}  (seed={args.shuffled_seed})")

    # Quick sanity: meanstd should have row-norm distribution close to reference.
    ref_norms = reference.norm(dim=-1)
    ms_norms = meanstd.norm(dim=-1)
    print(
        f"  ref row-norm    mean={ref_norms.mean():.4f}  std={ref_norms.std():.4f}"
    )
    print(
        f"  meanstd norm    mean={ms_norms.mean():.4f}  std={ms_norms.std():.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
