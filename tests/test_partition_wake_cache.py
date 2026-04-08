# tests/test_partition_wake_cache.py
import torch
from chaoscontrol.wake_cache import WakeCache
from chaoscontrol.partition import SemanticPartition


def test_filter_moments_by_partition():
    cache = WakeCache(max_moments=16, max_hidden_buffer=8)
    for bucket_dominant in [0, 1, 0, 2, 1, 0]:
        bids = torch.full((1, 4), bucket_dominant, dtype=torch.long)
        cache.record_moment(
            surprise=1.0,
            inputs=torch.zeros(1, 4),
            targets=torch.zeros(1, 4),
            hidden=torch.randn(1, 4, 8),
            bucket_ids=bids,
        )
    p0 = SemanticPartition(partition_id=0, bucket_ids={0})
    filtered = cache.filter_moments_by_partition(p0)
    assert len(filtered) == 3


def test_filter_empty_cache():
    cache = WakeCache(max_moments=8, max_hidden_buffer=8)
    p = SemanticPartition(partition_id=0, bucket_ids={0})
    assert cache.filter_moments_by_partition(p) == []


def test_filter_no_bucket_ids():
    cache = WakeCache(max_moments=8, max_hidden_buffer=8)
    cache.record_moment(
        surprise=1.0,
        inputs=torch.zeros(1, 4),
        targets=torch.zeros(1, 4),
        hidden=torch.randn(1, 4, 8),
    )
    p = SemanticPartition(partition_id=0, bucket_ids={0})
    assert cache.filter_moments_by_partition(p) == []
