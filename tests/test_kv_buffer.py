"""Tests for append-only KV buffer and within-bucket retrieval."""
import torch
from chaoscontrol.memory import MultiSlotOuterModel


def test_append_kv_stores_entry():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    kv = torch.randn(1, 64)
    model.append_kv(kv, bucket_id=3)
    assert len(model._slots) == 1
    assert model._slot_buckets[0] == 3


def test_append_kv_unconditional():
    """Append should always store, no surprise gating."""
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    for i in range(100):
        kv = torch.randn(1, 64)
        model.append_kv(kv, bucket_id=i % 4)
    assert len(model._slots) == 100


def test_append_kv_no_compression_when_unlimited():
    """max_slots=0 means unlimited -- no compression should fire."""
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    for i in range(500):
        kv = torch.randn(1, 64)
        model.append_kv(kv, bucket_id=i % 8)
    assert len(model._slots) == 500


def test_append_kv_bucket_tracking():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    model.append_kv(torch.randn(1, 64), bucket_id=0)
    model.append_kv(torch.randn(1, 64), bucket_id=0)
    model.append_kv(torch.randn(1, 64), bucket_id=5)
    bucket_0 = [i for i, b in enumerate(model._slot_buckets) if b == 0]
    bucket_5 = [i for i, b in enumerate(model._slot_buckets) if b == 5]
    assert len(bucket_0) == 2
    assert len(bucket_5) == 1


def test_append_kv_compresses_when_capped():
    """When max_slots > 0, compression fires past capacity."""
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=10)
    for i in range(20):
        model.append_kv(torch.randn(1, 64), bucket_id=0)
    assert len(model._slots) <= 10
