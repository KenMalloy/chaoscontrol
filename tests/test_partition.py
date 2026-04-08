"""Tests for SemanticPartition and PartitionTopology."""

import pytest
from chaoscontrol.partition import SemanticPartition, PartitionTopology


def test_partition_init():
    """partition_id, bucket_ids, mode, is_awake, is_sleeping."""
    p = SemanticPartition(partition_id=0)
    assert p.partition_id == 0
    assert p.bucket_ids == set()
    assert p.mode == "awake"
    assert p.is_awake is True
    assert p.is_sleeping is False

    p2 = SemanticPartition(partition_id=3, bucket_ids={1, 5, 9}, mode="sleeping")
    assert p2.partition_id == 3
    assert p2.bucket_ids == {1, 5, 9}
    assert p2.mode == "sleeping"
    assert p2.is_awake is False
    assert p2.is_sleeping is True


def test_partition_mode_toggle():
    """Change mode, verify properties update."""
    p = SemanticPartition(partition_id=1)
    assert p.is_awake is True
    assert p.is_sleeping is False

    p.mode = "sleeping"
    assert p.is_awake is False
    assert p.is_sleeping is True

    p.mode = "awake"
    assert p.is_awake is True
    assert p.is_sleeping is False


def test_partition_owns_bucket():
    """owns_bucket returns True/False correctly."""
    p = SemanticPartition(partition_id=0, bucket_ids={0, 1, 2, 3})
    assert p.owns_bucket(0) is True
    assert p.owns_bucket(3) is True
    assert p.owns_bucket(4) is False
    assert p.owns_bucket(15) is False


def test_slot_striped_topology():
    """n_partitions=4, total_slots=64 — slot_owner_map assigns round-robin."""
    topo = PartitionTopology.slot_striped(n_partitions=4)

    assert len(topo.partitions) == 4
    # All partitions should have empty bucket_ids (slot-based, not bucket-based)
    for p in topo.partitions:
        assert p.bucket_ids == set()

    owner_map = topo.slot_owner_map(total_slots=64)
    assert len(owner_map) == 64

    # Round-robin: slot 0 -> partition 0, slot 1 -> partition 1, ...
    assert owner_map[0] == 0
    assert owner_map[1] == 1
    assert owner_map[2] == 2
    assert owner_map[3] == 3
    assert owner_map[4] == 0
    assert owner_map[63] == 3  # 63 % 4 == 3

    # Each partition owns exactly 16 slots
    for pid in range(4):
        assert owner_map.count(pid) == 16


def test_bucket_owned_topology():
    """n_partitions=4, k_max=16 — each partition gets 4 exclusive buckets."""
    topo = PartitionTopology.bucket_owned(n_partitions=4, k_max=16)

    assert len(topo.partitions) == 4
    assert topo.partitions[0].bucket_ids == {0, 1, 2, 3}
    assert topo.partitions[1].bucket_ids == {4, 5, 6, 7}
    assert topo.partitions[2].bucket_ids == {8, 9, 10, 11}
    assert topo.partitions[3].bucket_ids == {12, 13, 14, 15}

    # No overlap
    all_buckets = set()
    for p in topo.partitions:
        assert len(all_buckets & p.bucket_ids) == 0
        all_buckets |= p.bucket_ids
    assert all_buckets == set(range(16))


def test_bucket_striped_topology():
    """n_partitions=4, k_max=16, group_size=2 — partitions in same group share bucket sets."""
    topo = PartitionTopology.bucket_striped(n_partitions=4, k_max=16, group_size=2)

    assert len(topo.partitions) == 4

    # Group 0: partitions 0, 1 share buckets 0-7
    assert topo.partitions[0].bucket_ids == set(range(0, 8))
    assert topo.partitions[1].bucket_ids == set(range(0, 8))

    # Group 1: partitions 2, 3 share buckets 8-15
    assert topo.partitions[2].bucket_ids == set(range(8, 16))
    assert topo.partitions[3].bucket_ids == set(range(8, 16))


def test_awake_sleeping_helpers():
    """Set some partitions sleeping, verify helpers return correct lists."""
    topo = PartitionTopology.bucket_owned(n_partitions=4, k_max=16)

    # Initially all awake
    assert len(topo.awake_partitions()) == 4
    assert len(topo.sleeping_partitions()) == 0
    assert topo.awake_bucket_ids() == set(range(16))

    # Put partitions 1 and 3 to sleep
    topo.partitions[1].mode = "sleeping"
    topo.partitions[3].mode = "sleeping"

    awake = topo.awake_partitions()
    sleeping = topo.sleeping_partitions()

    assert len(awake) == 2
    assert len(sleeping) == 2

    awake_ids = {p.partition_id for p in awake}
    sleeping_ids = {p.partition_id for p in sleeping}
    assert awake_ids == {0, 2}
    assert sleeping_ids == {1, 3}

    # awake_bucket_ids should be buckets from partitions 0 and 2
    assert topo.awake_bucket_ids() == {0, 1, 2, 3, 8, 9, 10, 11}
