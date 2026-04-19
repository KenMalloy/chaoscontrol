#!/usr/bin/env python3
"""Derive Exp 21 embedding-init tensors from a trained SGNS artifact.

Produces several ``(vocab_size, dim)`` tensors from one SGNS training run:
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
  - ``zero``:      deterministic all-zero floor used by runner_controls.py.
  - ``norm_only``: random directions with each token's SGNS/meanstd row norm.
                   Tests whether per-token scale/frequency explains the gain.
  - ``norm_only_shuffled``: same norm multiset, shuffled across token IDs.
  - ``fullcov_shuffled``: row-shuffled full-cov control.
  - ``random_fullcov``: synthetic Gaussian cloud matched to fullcov mean/cov.
  - ``freq_bucket_shuffle``: optional meanstd row shuffle within frequency
                   rank buckets when ``--token-counts`` is provided.
  - ``class_bucket_shuffle``: optional meanstd row shuffle within coarse
                   SentencePiece token classes when ``--sp-model`` is provided.

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
    artifacts/sgns_init_zero.pt
    artifacts/sgns_init_norm_only.pt
    artifacts/sgns_init_norm_only_shuffled.pt
    artifacts/sgns_init_fullcov_shuffled.pt
    artifacts/sgns_init_random_fullcov.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from chaoscontrol.sgns.moment_match import (  # noqa: E402
    frequency_bucket_ids,
    match_full_covariance,
    match_row_norm_moments,
    sample_full_covariance,
    sample_with_row_norms,
    shuffle_rows,
    shuffle_rows_within_buckets,
    token_class_bucket_ids,
)


def _reference_init(vocab_size: int, dim: int, seed: int) -> torch.Tensor:
    """Reference Gaussian matching nn.Embedding's default (std=1)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(vocab_size, dim, generator=g)


def _load_token_counts(path: Path, vocab_size: int) -> torch.Tensor:
    raw = torch.load(str(path), map_location="cpu")
    if isinstance(raw, dict):
        if "counts" not in raw:
            raise ValueError(f"{path} is a dict but has no 'counts' key")
        raw = raw["counts"]
    counts = torch.as_tensor(raw, dtype=torch.float32).flatten()
    if counts.shape[0] != vocab_size:
        raise ValueError(
            f"token-counts length mismatch: got {counts.shape[0]}, expected {vocab_size}"
        )
    return counts


def _count_data_dir_tokens(
    data_dir: Path,
    vocab_size: int,
    max_tokens: int | None,
) -> torch.Tensor:
    from chaoscontrol.data import load_fineweb_tokens

    train_tokens_mmap, _val = load_fineweb_tokens(str(data_dir))
    n = (
        len(train_tokens_mmap)
        if max_tokens is None
        else min(len(train_tokens_mmap), max_tokens)
    )
    stream = train_tokens_mmap[:n].to(torch.long)
    oor_mask = (stream < 0) | (stream >= vocab_size)
    n_oor = int(oor_mask.sum().item())
    if n_oor > 0:
        oor_frac = n_oor / max(stream.numel(), 1)
        if oor_frac > 0.001:
            raise ValueError(
                f"prepare_sgns_inits: tokenizer/data mismatch at {data_dir}. "
                f"More than 0.1% of tokens outside [0, {vocab_size}). "
                f"oor={n_oor}/{stream.numel()} ({oor_frac:.4%})."
            )
        print(f"clamping {n_oor} out-of-range tokens ({oor_frac:.4%}) to 0")
        stream = stream.clamp(0, vocab_size - 1)
    return torch.bincount(stream, minlength=vocab_size).float()


def _sentencepiece_pieces(path: Path, vocab_size: int) -> list[str]:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    loaded = sp.Load(str(path))
    if loaded is False:
        raise RuntimeError(f"failed to load SentencePiece model {path}")
    sp_vocab_size = int(sp.vocab_size())
    if sp_vocab_size < vocab_size:
        raise ValueError(
            f"SentencePiece vocab_size={sp_vocab_size} smaller than requested {vocab_size}"
        )
    return [sp.IdToPiece(i) for i in range(vocab_size)]


def _save(out_dir: Path, name: str, tensor: torch.Tensor) -> Path:
    path = out_dir / f"sgns_init_{name}.pt"
    torch.save(tensor, str(path))
    print(f"Wrote {path} shape={tuple(tensor.shape)}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive Exp 21/21b embedding-init variants from SGNS output"
    )
    parser.add_argument("--sgns", required=True, help="Path to sgns_vN_dD.pt")
    parser.add_argument("--out-dir", required=True, help="Directory for init_*.pt outputs")
    parser.add_argument("--reference-seed", type=int, default=0,
                        help="Seed for reference Gaussian (for moment matching target)")
    parser.add_argument("--shuffled-seed", type=int, default=42,
                        help="Seed for the shuffled-row control permutation")
    parser.add_argument("--norm-seed", type=int, default=101,
                        help="Seed for norm-only random directions")
    parser.add_argument("--norm-shuffled-seed", type=int, default=102,
                        help="Seed for norm-only token-ID norm permutation")
    parser.add_argument("--fullcov-shuffled-seed", type=int, default=103,
                        help="Seed for fullcov row-shuffle permutation")
    parser.add_argument("--random-fullcov-seed", type=int, default=104,
                        help="Seed for synthetic random full-covariance sample")
    parser.add_argument("--token-counts", type=Path, default=None,
                        help="Optional per-token counts tensor for frequency bucket shuffle")
    parser.add_argument("--data-dir-for-counts", type=Path, default=None,
                        help="Optional FineWeb shard dir used to compute token counts")
    parser.add_argument("--counts-max-tokens", type=int, default=50_000_000,
                        help="Cap count computation to first N train tokens")
    parser.add_argument("--freq-buckets", type=int, default=8,
                        help="Number of frequency rank buckets")
    parser.add_argument("--freq-shuffled-seed", type=int, default=105,
                        help="Seed for within-frequency-bucket row shuffle")
    parser.add_argument("--sp-model", type=Path, default=None,
                        help="Optional SentencePiece model for token-class bucket shuffle")
    parser.add_argument("--class-shuffled-seed", type=int, default=106,
                        help="Seed for within-token-class row shuffle")
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
    zero = torch.zeros_like(meanstd)
    norm_only = sample_with_row_norms(meanstd, seed=args.norm_seed)
    norm_only_shuffled = sample_with_row_norms(
        meanstd,
        seed=args.norm_seed,
        shuffle_norms_seed=args.norm_shuffled_seed,
    )
    fullcov_shuffled = shuffle_rows(fullcov, args.fullcov_shuffled_seed)
    random_fullcov = sample_full_covariance(fullcov, seed=args.random_fullcov_seed)

    _save(out_dir, "meanstd", meanstd)
    _save(out_dir, "fullcov", fullcov)
    _save(out_dir, "shuffled", shuffled)
    _save(out_dir, "zero", zero)
    _save(out_dir, "norm_only", norm_only)
    _save(out_dir, "norm_only_shuffled", norm_only_shuffled)
    _save(out_dir, "fullcov_shuffled", fullcov_shuffled)
    _save(out_dir, "random_fullcov", random_fullcov)

    counts: torch.Tensor | None = None
    if args.token_counts is not None and args.data_dir_for_counts is not None:
        raise ValueError("pass only one of --token-counts or --data-dir-for-counts")
    if args.token_counts is not None:
        counts = _load_token_counts(args.token_counts, vocab_size)
    elif args.data_dir_for_counts is not None:
        counts = _count_data_dir_tokens(
            args.data_dir_for_counts, vocab_size, args.counts_max_tokens
        )

    if counts is not None:
        freq_ids = frequency_bucket_ids(counts, num_buckets=args.freq_buckets)
        freq_bucket_shuffle = shuffle_rows_within_buckets(
            meanstd, freq_ids, seed=args.freq_shuffled_seed
        )
        _save(out_dir, "freq_bucket_shuffle", freq_bucket_shuffle)
    else:
        print(
            "Skipped sgns_init_freq_bucket_shuffle.pt "
            "(pass --token-counts or --data-dir-for-counts)"
        )

    if args.sp_model is not None:
        class_ids = token_class_bucket_ids(
            _sentencepiece_pieces(args.sp_model, vocab_size)
        )
        class_bucket_shuffle = shuffle_rows_within_buckets(
            meanstd, class_ids, seed=args.class_shuffled_seed
        )
        _save(out_dir, "class_bucket_shuffle", class_bucket_shuffle)
    else:
        print("Skipped sgns_init_class_bucket_shuffle.pt (pass --sp-model)")

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
