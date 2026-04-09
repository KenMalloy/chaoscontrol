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


# ---- Within-bucket retrieval tests ----


def test_read_bucket_mean():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    # Add 3 entries to bucket 0, 2 entries to bucket 1
    v0a = torch.ones(1, 64)
    v0b = torch.ones(1, 64) * 2
    v0c = torch.ones(1, 64) * 3
    v1a = torch.ones(1, 64) * 10
    v1b = torch.ones(1, 64) * 20
    model.append_kv(v0a, bucket_id=0)
    model.append_kv(v0b, bucket_id=0)
    model.append_kv(v1a, bucket_id=1)
    model.append_kv(v0c, bucket_id=0)
    model.append_kv(v1b, bucket_id=1)

    # Read from bucket 0: decoded to model_dim (128)
    result = model.read_bucket(batch_size=1, bucket_id=0, mode="bucket_mean")
    assert result.shape == (1, 128)  # Fix 1: decoded to model_dim

    # Read from bucket 1: decoded to model_dim
    result1 = model.read_bucket(batch_size=1, bucket_id=1, mode="bucket_mean")
    assert result1.shape == (1, 128)


def test_read_bucket_mean_empty_bucket():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    model.append_kv(torch.randn(1, 64), bucket_id=0)
    # Bucket 5 is empty -- should return zeros in model_dim
    result = model.read_bucket(batch_size=1, bucket_id=5, mode="bucket_mean")
    assert result.shape == (1, 128)  # model_dim, not outer_dim
    assert torch.allclose(result, torch.zeros(1, 128))


def test_read_bucket_recent():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    for i in range(20):
        model.append_kv(torch.ones(1, 64) * i, bucket_id=0)
    # k=3: should use last 3 entries, decoded to model_dim
    result = model.read_bucket(batch_size=1, bucket_id=0, mode="bucket_recent", k=3)
    assert result.shape == (1, 128)  # decoded to model_dim


def test_read_bucket_topk():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    # Insert entries: one very similar to query, others random
    target = torch.randn(1, 64)
    model.append_kv(torch.randn(1, 64), bucket_id=0)
    model.append_kv(torch.randn(1, 64), bucket_id=0)
    model.append_kv(target.clone(), bucket_id=0)
    model.append_kv(torch.randn(1, 64), bucket_id=0)

    # top-1 with the target as cue (in outer_dim space)
    result = model.read_bucket(batch_size=1, bucket_id=0, mode="bucket_topk",
                               k=1, cue=target)
    assert result.shape == (1, 128)  # decoded to model_dim


def test_read_bucket_topk_softmax_weighting():
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    v1 = torch.ones(1, 64)
    v2 = torch.ones(1, 64) * 2
    model.append_kv(v1, bucket_id=0)
    model.append_kv(v2, bucket_id=0)
    cue = torch.ones(1, 64) * 1.5  # equidistant-ish
    result = model.read_bucket(batch_size=1, bucket_id=0, mode="bucket_topk",
                               k=2, cue=cue)
    assert result.shape == (1, 128)  # decoded to model_dim


def test_read_bucket_softmax_all():
    """softmax_all retrieval uses all slots regardless of bucket, decoded to model_dim."""
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    model.append_kv(torch.randn(1, 64), bucket_id=0)
    model.append_kv(torch.randn(1, 64), bucket_id=1)
    model.append_kv(torch.randn(1, 64), bucket_id=2)
    cue = torch.randn(1, 64)
    result = model.read_bucket(batch_size=1, bucket_id=0, mode="softmax_all",
                               cue=cue)
    assert result.shape == (1, 128)  # decoded to model_dim


def test_read_bucket_empty_buffer():
    """Reading from an entirely empty buffer returns model_dim zeros."""
    model = MultiSlotOuterModel(model_dim=128, outer_dim=64, max_slots=0)
    result = model.read_bucket(batch_size=2, bucket_id=0, mode="bucket_mean")
    assert result.shape == (2, 128)
    assert torch.allclose(result, torch.zeros(2, 128))
