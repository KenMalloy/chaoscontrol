"""Intrinsic validation report for a saved SGNS embedding table.

Usage: python scripts/sgns_intrinsic_report.py --embed artifacts/sgns_v8192_d256.pt
"""
import argparse
from collections import Counter
from pathlib import Path

import torch

from chaoscontrol.sgns.intrinsic import nearest_neighbors, kmeans_clusters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embed", type=Path, required=True)
    p.add_argument("--n-common", type=int, default=50)
    p.add_argument("--k-clusters", type=int, default=20)
    args = p.parse_args()

    embed = torch.load(args.embed)
    V, D = embed.shape
    print(f"Loaded {args.embed}: V={V}, D={D}")

    # NN on first N_common token IDs (convention: low IDs are common under BPE)
    queries = list(range(args.n_common))
    nn = nearest_neighbors(embed, query_ids=queries, k=5)
    print(f"\n== Nearest neighbors for first {args.n_common} token IDs ==")
    for q, nbrs in list(nn.items())[:args.n_common]:
        print(f"  {q:5d} -> {nbrs}")

    # k-means
    labels = kmeans_clusters(embed, k=args.k_clusters)
    counts = Counter(labels.tolist())
    print(f"\n== k-means (k={args.k_clusters}) cluster sizes ==")
    for c, n in sorted(counts.items()):
        print(f"  cluster {c:2d}: {n:6d} tokens")


if __name__ == "__main__":
    main()
