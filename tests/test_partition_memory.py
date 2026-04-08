# tests/test_partition_memory.py
import torch
from chaoscontrol.memory import MultiSlotOuterModel
from chaoscontrol.partition import SemanticPartition


def test_get_partition_slot_indices():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    for bucket in [0, 1, 0, 2]:
        om.write(torch.randn(1, 1, 16), bucket_id=bucket)

    p0 = SemanticPartition(partition_id=0, bucket_ids={0})
    p1 = SemanticPartition(partition_id=1, bucket_ids={1, 2})

    assert om.get_partition_slot_indices(p0) == [0, 2]
    assert om.get_partition_slot_indices(p1) == [1, 3]


def test_partition_slot_count():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    for bucket in [0, 0, 1, 1, 2]:
        om.write(torch.randn(1, 1, 16), bucket_id=bucket)

    p = SemanticPartition(partition_id=0, bucket_ids={0})
    assert om.partition_slot_count(p) == 2


def test_is_write_allowed():
    awake = SemanticPartition(partition_id=0, bucket_ids={0}, mode="awake")
    sleeping = SemanticPartition(partition_id=1, bucket_ids={1}, mode="sleeping")

    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    assert om.is_write_allowed(bucket_id=0, partitions=[awake, sleeping])
    assert not om.is_write_allowed(bucket_id=1, partitions=[awake, sleeping])


def test_empty_partition():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    om.write(torch.randn(1, 1, 16), bucket_id=5)

    p = SemanticPartition(partition_id=0, bucket_ids={0})
    assert om.get_partition_slot_indices(p) == []
    assert om.partition_slot_count(p) == 0
