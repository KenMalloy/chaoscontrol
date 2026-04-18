"""Offline SGNS training on FineWeb train. Produces per-subword W_in tensor.

Example (on GPU pod with shards in place):
    python scripts/train_sgns.py \\
        --data-dir baselines/parameter_golf/datasets/fineweb10B_sp8192 \\
        --vocab-size 8192 \\
        --dim 256 \\
        --window 5 \\
        --k 10 \\
        --epochs 3 \\
        --max-tokens 50000000 \\
        --subsample-threshold 1e-5 \\
        --out artifacts/sgns_v8192_d256.pt
"""
import argparse
from pathlib import Path

import torch

from chaoscontrol.data import load_fineweb_tokens
from chaoscontrol.sgns.model import SGNSModel
from chaoscontrol.sgns.sampler import NegativeSampler, unigram_probs_from_counts
from chaoscontrol.sgns.train import train_one_epoch


def _subsample(stream: torch.Tensor, counts: torch.Tensor, threshold: float) -> torch.Tensor:
    """word2vec subsampling: drop frequent tokens with prob 1 - sqrt(t / f)."""
    total = counts.sum().item()
    freqs = counts / total
    keep_prob = torch.minimum(
        torch.sqrt(threshold / freqs.clamp(min=1e-12)),
        torch.ones_like(freqs),
    )
    mask = torch.rand(len(stream)) < keep_prob[stream]
    return stream[mask]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Shard dir — e.g. baselines/parameter_golf/datasets/fineweb10B_sp8192")
    parser.add_argument("--vocab-size", type=int, required=True,
                        help="SGNS embedding table size — must match tokenizer V")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--subsample-threshold", type=float, default=1e-5)
    parser.add_argument("--max-tokens", type=int, default=50_000_000,
                        help="Cap training stream to first N tokens; bigger = more RAM/GPU needed")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load tokens — int16 memmap → int64 slice we can use for embedding lookup
    train_tokens_mmap, _val = load_fineweb_tokens(args.data_dir)
    print(f"train shards total length: {len(train_tokens_mmap):,}")
    n = min(len(train_tokens_mmap), args.max_tokens)
    stream = train_tokens_mmap[:n].to(torch.long)
    print(f"using first {n:,} tokens")

    # Unigram counts + subsampling
    counts = torch.bincount(stream, minlength=args.vocab_size).float()
    stream = _subsample(stream, counts, args.subsample_threshold)
    print(f"after subsampling: {len(stream):,} tokens")

    probs = unigram_probs_from_counts(counts, power=0.75)

    sampler = NegativeSampler(probs.to(args.device))
    model = SGNSModel(vocab_size=args.vocab_size, dim=args.dim).to(args.device)
    stream = stream.to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model, stream, sampler, args.window, args.k, args.batch_size, opt
        )
        print(f"epoch {epoch}: mean_loss = {loss:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.input_embed.weight.detach().cpu(), args.out)
    print(f"saved W_in to {args.out}")


if __name__ == "__main__":
    main()
