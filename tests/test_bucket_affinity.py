"""Tests for bucket affinity matrix and affinity-based clustering."""
import torch
from chaoscontrol.memory import MultiSlotOuterModel


def test_affinity_starts_identity():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    assert om.bucket_affinity(0, 0) == 1.0
    assert om.bucket_affinity(0, 1) == 0.0
    assert om.bucket_affinity(5, 5) == 1.0


def test_affinity_update():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    om._ensure_affinity(4)

    # Positive update: buckets 0 and 1 become friends
    om.update_affinity(0, 1, delta=1.0, lr=0.1)
    assert abs(om.bucket_affinity(0, 1) - 0.1) < 1e-6
    assert abs(om.bucket_affinity(1, 0) - 0.1) < 1e-6  # symmetric

    # Same-bucket update is a no-op
    om.update_affinity(0, 0, delta=1.0, lr=0.1)
    assert om.bucket_affinity(0, 0) == 1.0  # unchanged

    # Negative update: decrease affinity
    om.update_affinity(0, 1, delta=-1.0, lr=0.1)
    assert abs(om.bucket_affinity(0, 1)) < 1e-6  # back to 0

    # Affinity clamps to [0, 1]
    om.update_affinity(2, 3, delta=-10.0, lr=1.0)
    assert om.bucket_affinity(2, 3) == 0.0


def test_affinity_clusters_isolated():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    om._ensure_affinity(4)
    # No cross-bucket affinity → 4 singleton clusters
    clusters = om.affinity_clusters(threshold=0.1)
    assert len(clusters) == 4


def test_affinity_clusters_merged():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    om._ensure_affinity(4)
    # Make buckets 0-1 friends and 2-3 friends
    om.update_affinity(0, 1, delta=1.0, lr=0.5)
    om.update_affinity(2, 3, delta=1.0, lr=0.5)

    clusters = om.affinity_clusters(threshold=0.3)
    assert len(clusters) == 2
    cluster_sizes = sorted([len(c) for c in clusters])
    assert cluster_sizes == [2, 2]


def test_affinity_clusters_chain():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    om._ensure_affinity(4)
    # Chain: 0-1, 1-2 → all three in one cluster
    om.update_affinity(0, 1, delta=1.0, lr=0.5)
    om.update_affinity(1, 2, delta=1.0, lr=0.5)

    clusters = om.affinity_clusters(threshold=0.3)
    # Buckets 0,1,2 should be in one cluster, bucket 3 alone
    assert len(clusters) == 2
    big_cluster = max(clusters, key=len)
    assert big_cluster == {0, 1, 2}


def test_cross_bucket_bootstrap():
    """Two identical slots in different buckets should produce a cross-bucket proposal."""
    from chaoscontrol.sleep import SleepCycle, SleepConfig

    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    # Write two identical slots to different buckets
    slot_val = torch.randn(1, 1, 16)
    om.write(slot_val.clone(), bucket_id=0)
    om.write(slot_val.clone(), bucket_id=1)

    cfg = SleepConfig(stages="n3_only", merge_sim_threshold=0.85)
    cycle = SleepCycle(cfg)

    # The slots are identical (sim=1.0) so even at exploration threshold (0.85)
    # they should be proposed for cross-bucket merge
    proposals = cycle._propose_typed_merges(om, cfg)
    cross = [p for p in proposals if p["bucket_a"] != p["bucket_b"]]
    assert len(cross) >= 1, f"Expected cross-bucket proposal, got {proposals}"


def test_slot_one_proposal_only():
    """Each slot participates in at most one merge proposal."""
    from chaoscontrol.sleep import SleepCycle, SleepConfig

    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    slot_val = torch.randn(1, 1, 16)
    # 4 identical slots across 2 buckets
    for bucket in [0, 0, 1, 1]:
        om.write(slot_val.clone(), bucket_id=bucket)

    cfg = SleepConfig(stages="n3_only", merge_sim_threshold=0.85)
    cycle = SleepCycle(cfg)

    proposals = cycle._propose_typed_merges(om, cfg)
    used_slots = set()
    for p in proposals:
        assert p["idx_a"] not in used_slots, f"Slot {p['idx_a']} used twice"
        assert p["idx_b"] not in used_slots, f"Slot {p['idx_b']} used twice"
        used_slots.add(p["idx_a"])
        used_slots.add(p["idx_b"])


def test_cross_bucket_competes_with_same_bucket():
    """With high affinity, cross-bucket merge can win over weaker same-bucket pair."""
    from chaoscontrol.sleep import SleepCycle, SleepConfig

    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    om._ensure_affinity(2)
    om.update_affinity(0, 1, delta=1.0, lr=0.9)  # affinity 0.9

    # Slot 0 (bucket 0): unique vector
    om.write(torch.randn(1, 1, 16), bucket_id=0)
    # Slot 1 (bucket 0): different random vector
    om.write(torch.randn(1, 1, 16), bucket_id=0)
    # Slot 2 (bucket 1): clone of slot 0's encoded value (identical, cross-bucket)
    om._slots.append(om._slots[0].clone())
    om._survival.append(0.5)
    om._slot_buckets.append(1)

    cfg = SleepConfig(stages="n3_only", merge_sim_threshold=0.85)
    cycle = SleepCycle(cfg)
    proposals = cycle._propose_typed_merges(om, cfg)

    # Slot 0 and slot 2 are identical (sim=1.0) with affinity=0.9 → score=0.9
    # Slot 0 and slot 1 are random (sim likely < 0.85) → probably not proposed
    # So the cross-bucket pair (0,2) should appear
    cross = [p for p in proposals if p["bucket_a"] != p["bucket_b"]]
    assert len(cross) >= 1, f"Expected cross-bucket proposal with high affinity, got {proposals}"


def test_affinity_serialization():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    om._ensure_affinity(4)
    om.update_affinity(0, 1, delta=1.0, lr=0.3)

    # Save
    state = om.get_extra_state()
    assert "bucket_affinity" in state

    # Restore into a fresh model
    om2 = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    om2.set_extra_state(state)
    assert abs(om2.bucket_affinity(0, 1) - 0.3) < 1e-6
