import torch


def nearest_neighbors(
    embed: torch.Tensor, query_ids: list[int], k: int = 5
) -> dict[int, list[int]]:
    """Return top-k nearest neighbors (excluding self) for each query_id, by cosine."""
    normed = torch.nn.functional.normalize(embed, dim=-1)
    out: dict[int, list[int]] = {}
    for q in query_ids:
        sims = normed @ normed[q]
        sims[q] = float("-inf")
        top = torch.topk(sims, k).indices.tolist()
        out[q] = top
    return out


def kmeans_clusters(embed: torch.Tensor, k: int, n_iter: int = 20) -> torch.Tensor:
    """Simple k-means over embedding rows. Returns cluster label per row."""
    n, d = embed.shape
    torch.manual_seed(0)
    centroids = embed[torch.randperm(n)[:k]].clone()
    for _ in range(n_iter):
        dists = torch.cdist(embed, centroids)
        labels = dists.argmin(dim=-1)
        for j in range(k):
            members = embed[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(dim=0)
    return labels
