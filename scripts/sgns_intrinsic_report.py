"""Intrinsic validation report for a saved SGNS embedding table.

Usage: python scripts/sgns_intrinsic_report.py --embed artifacts/sgns_v8192_d256.pt \\
           --sp-model baselines/parameter_golf/tokenizers/fineweb_8192_bpe.model

When ``--sp-model`` is provided, token IDs are decoded to their subword
pieces so the kill-criterion ("do NN look semantically coherent?") can be
mechanically evaluated. A quantitative coherence proxy is always
printed — mean-cosine of top-k NN vs mean-cosine of random pairs. A
ratio ≲ 1.5 is a red flag that SGNS did not learn structure.
"""
import argparse
from collections import Counter
from pathlib import Path

import torch

from chaoscontrol.sgns.intrinsic import nearest_neighbors, kmeans_clusters


def _load_sp(path: Path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(path))
    return sp


def _decode(sp, token_id: int) -> str:
    try:
        return sp.IdToPiece(int(token_id))
    except Exception:
        return f"<id={token_id}>"


def _random_pair_mean_cos(embed: torch.Tensor, n_pairs: int, seed: int) -> float:
    g = torch.Generator().manual_seed(seed)
    V = embed.shape[0]
    normed = torch.nn.functional.normalize(embed, dim=-1)
    ia = torch.randint(0, V, (n_pairs,), generator=g)
    ib = torch.randint(0, V, (n_pairs,), generator=g)
    mask = ia != ib
    ia, ib = ia[mask], ib[mask]
    sims = (normed[ia] * normed[ib]).sum(dim=-1)
    return float(sims.mean().item())


def _nn_mean_cos(embed: torch.Tensor, queries: list[int], k: int) -> float:
    normed = torch.nn.functional.normalize(embed, dim=-1)
    totals = []
    for q in queries:
        sims = normed @ normed[q]
        sims[q] = float("-inf")
        top_vals = torch.topk(sims, k).values
        totals.append(float(top_vals.mean().item()))
    return sum(totals) / max(len(totals), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embed", type=Path, required=True)
    p.add_argument("--sp-model", type=Path, default=None,
                   help="Optional SP model — decodes token IDs to pieces if provided")
    p.add_argument("--n-common", type=int, default=50)
    p.add_argument("--k-clusters", type=int, default=20)
    p.add_argument("--k-nn", type=int, default=5)
    p.add_argument("--n-random-pairs", type=int, default=20000)
    args = p.parse_args()

    embed = torch.load(args.embed)
    V, D = embed.shape
    print(f"Loaded {args.embed}: V={V}, D={D}")

    sp = _load_sp(args.sp_model) if args.sp_model else None
    if sp is None:
        print("(no --sp-model given — NN printed as raw IDs)")

    # NN on first N_common token IDs (convention: low IDs are common under BPE)
    queries = list(range(args.n_common))
    nn = nearest_neighbors(embed, query_ids=queries, k=args.k_nn)
    print(f"\n== Nearest neighbors for first {args.n_common} token IDs ==")
    if sp is not None:
        for q, nbrs in list(nn.items())[:args.n_common]:
            q_piece = _decode(sp, q)
            nbr_pieces = [_decode(sp, n) for n in nbrs]
            print(f"  {q:5d} {q_piece!r:>18s}  ->  {nbr_pieces}")
    else:
        for q, nbrs in list(nn.items())[:args.n_common]:
            print(f"  {q:5d} -> {nbrs}")

    # Quantitative coherence proxy — NN cosine vs random-pair cosine.
    nn_cos = _nn_mean_cos(embed, queries, k=args.k_nn)
    rand_cos = _random_pair_mean_cos(embed, args.n_random_pairs, seed=0)
    ratio = nn_cos / rand_cos if abs(rand_cos) > 1e-8 else float("inf")
    print(
        f"\n== Coherence proxy ==\n"
        f"  mean top-{args.k_nn} NN cosine (first {args.n_common} IDs): {nn_cos:.4f}\n"
        f"  mean random-pair cosine (n={args.n_random_pairs}): {rand_cos:.4f}\n"
        f"  ratio (NN/random): {ratio:.2f}  "
        f"(>~1.5 = learned structure; ≲1.0 = no signal)"
    )

    # k-means
    labels = kmeans_clusters(embed, k=args.k_clusters)
    counts = Counter(labels.tolist())
    print(f"\n== k-means (k={args.k_clusters}) cluster sizes ==")
    for c, n in sorted(counts.items()):
        print(f"  cluster {c:2d}: {n:6d} tokens")


if __name__ == "__main__":
    main()
