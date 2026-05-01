import pytest
from chaoscontrol.public.engine_entry import init_arm_topology


def test_8gpu_splits_6_2():
    role6 = init_arm_topology(rank=6, world_size=8)
    role7 = init_arm_topology(rank=7, world_size=8)
    role0 = init_arm_topology(rank=0, world_size=8)
    assert role6.is_packet_rank
    assert not role6.is_maintenance_rank
    assert role7.is_maintenance_rank
    assert not role7.is_packet_rank
    assert role0.is_train_rank
    assert role6.split_memory_ranks
    assert role7.split_memory_ranks


def test_4gpu_shares_memory():
    role3 = init_arm_topology(rank=3, world_size=4)
    role0 = init_arm_topology(rank=0, world_size=4)
    assert role3.is_packet_rank
    assert role3.is_maintenance_rank
    assert not role3.is_train_rank
    assert role0.is_train_rank
    assert not role3.split_memory_ranks


def test_1gpu_all_train():
    role = init_arm_topology(rank=0, world_size=1)
    assert role.is_train_rank
    assert not role.is_packet_rank
    assert not role.is_maintenance_rank


def test_packet_rank_value_8gpu():
    role = init_arm_topology(rank=0, world_size=8)
    assert role.packet_rank == 6
    assert role.maintenance_rank == 7
