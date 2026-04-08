# tests/test_polyphasic_integration.py
import torch
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.sleep import SleepCycle, SleepConfig
from chaoscontrol.wake_cache import WakeCache
from chaoscontrol.partition import (
    PartitionTopology, PolyphasicScheduler,
)


def test_full_polyphasic_cycle():
    """End-to-end: create model, write slots, schedule sleep, verify isolation."""
    model = ChaosStudentLM(
        vocab_size=256, dim=16, num_layers=2, ff_mult=2,
        outer_model_dim=8, outer_max_slots=32,
        outer_model_type="multislot",
        wernicke_enabled=True, wernicke_router="moe", wernicke_k_max=4,
        typed_storage=True,
    )
    om = model.outer_model
    cache = WakeCache(max_moments=16, max_hidden_buffer=8)

    # Create topology and scheduler (4 partitions, 3 awake, bucket_owned)
    topo = PartitionTopology.bucket_owned(n_partitions=4, k_max=4)
    sched = PolyphasicScheduler(topo, k_awake=3, swap_interval=2)

    # Write some slots with bucket assignments via model forward pass
    for step in range(8):
        x = torch.randint(0, 256, (1, 16))
        out = model(x)
        bucket_id = step % 4
        # hidden is (batch, seq, dim) — mean-pool to (batch, dim) for write
        h = out["hidden"].mean(dim=1)
        om.write(h, bucket_id=bucket_id)

    assert len(om._slots) == 8

    # Verify partition slot ownership
    for p in topo.partitions:
        indices = om.get_partition_slot_indices(p)
        assert len(indices) == 2  # 8 slots / 4 partitions = 2 each

    # Run sleep on the sleeping partition
    sleeping = sched.sleeping()
    assert len(sleeping) == 1
    sleeping_p = sleeping[0]

    cfg = SleepConfig(stages="n3_only", budget=64, survival_floor=0.5)
    cycle = SleepCycle(cfg)

    # Record non-partition survival values
    awake_indices = set()
    for p in sched.awake():
        awake_indices |= set(om.get_partition_slot_indices(p))
    awake_survivals = {i: om._survival[i] for i in awake_indices}

    diag = cycle.run(model, cache, device="cpu", partition=sleeping_p)
    assert isinstance(diag, dict)
    assert "n3" in diag

    # Verify awake partition slots were NOT touched
    for idx, surv in awake_survivals.items():
        if idx < len(om._survival):
            assert om._survival[idx] == surv, (
                f"Awake slot {idx} was modified during sleeping partition's sleep"
            )

    # Test scheduler rotation
    swapped = sched.step()  # step 1: no swap (swap_interval=2)
    assert not swapped
    swapped = sched.step()  # step 2: swap
    assert swapped

    # After swap, a different partition should be sleeping
    new_sleeping = sched.sleeping()[0].partition_id
    assert new_sleeping != sleeping_p.partition_id

    # Write gating: sleeping partition's bucket should be blocked
    new_sleeping_p = sched.sleeping()[0]
    blocked_bucket = min(new_sleeping_p.bucket_ids) if new_sleeping_p.bucket_ids else -1
    assert not om.is_write_allowed(blocked_bucket, topo.partitions)

    # Awake partition's bucket should be allowed
    awake_p = sched.awake()[0]
    allowed_bucket = min(awake_p.bucket_ids) if awake_p.bucket_ids else -1
    assert om.is_write_allowed(allowed_bucket, topo.partitions)


def test_polyphasic_slot_striped():
    """Verify slot_striped topology works with scheduler."""
    topo = PartitionTopology.slot_striped(n_partitions=4)
    sched = PolyphasicScheduler(topo, k_awake=3, swap_interval=1)

    # With slot_striped, buckets are assigned round-robin (uniform, not semantic)
    slot_map = topo.slot_owner_map(total_slots=16)
    assert slot_map[0] == 0
    assert slot_map[1] == 1
    assert slot_map[4] == 0

    # Scheduler rotation should still work
    for _ in range(8):
        sched.step()
        assert len(sched.awake()) == 3
        assert len(sched.sleeping()) == 1
