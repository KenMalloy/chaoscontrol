import torch
from chaoscontrol.sgns.intrinsic import nearest_neighbors, kmeans_clusters


def test_nearest_neighbors_returns_topk_by_cosine():
    # 3 tokens: 0 and 1 are nearly identical, 2 is orthogonal
    embed = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    nn_result = nearest_neighbors(embed, query_ids=[0], k=2)
    assert nn_result[0][0] == 1  # closest non-self


def test_kmeans_clusters_shapes():
    torch.manual_seed(0)
    embed = torch.randn(50, 8)
    labels = kmeans_clusters(embed, k=5)
    assert labels.shape == (50,)
    assert labels.max().item() < 5
