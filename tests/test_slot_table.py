"""Tests for SlotTable: persistent identity and lifecycle management."""
from __future__ import annotations

import torch
import pytest

from chaoscontrol.slot_table import (
    SlotTable, SlotRecord, SlotId,
    SLOT_WARMING, SLOT_ACTIVE, SLOT_QUARANTINED, SLOT_RETIRED,
)


class TestSlotId:
    def test_monotonic(self):
        t = SlotTable()
        ids = [t.append(torch.randn(1, 4), step=i) for i in range(5)]
        assert ids == [0, 1, 2, 3, 4]

    def test_never_reused(self):
        t = SlotTable()
        s0 = t.append(torch.randn(1, 4), step=0)
        s1 = t.append(torch.randn(1, 4), step=1)
        t.retire(s0, reason="test")
        s2 = t.append(torch.randn(1, 4), step=2)
        assert s2 == 2  # not 0
        assert s0 not in t.active_slot_ids()


class TestAppendRetire:
    def test_append(self):
        t = SlotTable()
        sid = t.append(torch.randn(1, 8), bucket_id=3, step=10)
        assert len(t) == 1
        rec = t.record(sid)
        assert rec.state == SLOT_WARMING
        assert rec.bucket_id == 3
        assert rec.created_step == 10

    def test_retire_single(self):
        t = SlotTable()
        s0 = t.append(torch.randn(1, 4))
        s1 = t.append(torch.randn(1, 4))
        t.retire(s0, reason="test")
        assert len(t) == 1
        assert t.record(s0).state == SLOT_RETIRED
        assert t.slot_id_to_physical(s1) == 0

    def test_retire_many(self):
        t = SlotTable()
        ids = [t.append(torch.randn(1, 4)) for _ in range(6)]
        t.retire_many([ids[1], ids[3], ids[4]], reason="batch")
        assert len(t) == 3
        assert t.active_slot_ids() == [ids[0], ids[2], ids[5]]

    def test_retire_nonexistent(self):
        t = SlotTable()
        assert t.retire(999, reason="nope") is False

    def test_physical_reindex_after_retire(self):
        t = SlotTable()
        s0 = t.append(torch.randn(1, 4))
        s1 = t.append(torch.randn(1, 4))
        s2 = t.append(torch.randn(1, 4))
        s3 = t.append(torch.randn(1, 4))
        t.retire(s1, reason="test")
        assert t.slot_id_to_physical(s0) == 0
        assert t.slot_id_to_physical(s2) == 1
        assert t.slot_id_to_physical(s3) == 2
        assert t.physical_to_slot_id(0) == s0
        assert t.physical_to_slot_id(1) == s2
        assert t.physical_to_slot_id(2) == s3


class TestQuarantineMask:
    def test_quarantine_hides_from_visible(self):
        t = SlotTable()
        s0 = t.append(torch.randn(1, 4))
        s1 = t.append(torch.randn(1, 4))
        assert len(t.visible_indices()) == 2
        t.quarantine(s0)
        assert t.visible_indices() == [1]

    def test_release_restores_visibility(self):
        t = SlotTable()
        s0 = t.append(torch.randn(1, 4))
        t.quarantine(s0)
        assert t.visible_indices() == []
        t.release(s0)
        assert t.visible_indices() == [0]

    def test_quarantine_updates_state(self):
        t = SlotTable()
        s0 = t.append(torch.randn(1, 4))
        t.quarantine(s0)
        assert t.record(s0).state == SLOT_QUARANTINED
        assert t.record(s0).quarantine_count == 1


class TestPriorityVector:
    def test_default_priority(self):
        t = SlotTable()
        t.append(torch.randn(1, 4))
        t.append(torch.randn(1, 4))
        pv = t.priority_vector()
        assert pv.tolist() == [1.0, 1.0]

    def test_quarantine_zeroes_priority(self):
        t = SlotTable()
        s0 = t.append(torch.randn(1, 4))
        s1 = t.append(torch.randn(1, 4))
        t.quarantine(s0)
        pv = t.priority_vector()
        assert pv[0].item() == 0.0
        assert pv[1].item() == 1.0


class TestSlotMatrix:
    def test_shape(self):
        t = SlotTable()
        for _ in range(5):
            t.append(torch.randn(1, 8))
        mat = t.slot_matrix()
        assert mat.shape == (5, 8)

    def test_subset(self):
        t = SlotTable()
        for _ in range(5):
            t.append(torch.randn(1, 8))
        mat = t.slot_matrix([1, 3])
        assert mat.shape == (2, 8)


class TestStateDict:
    def test_roundtrip(self):
        t = SlotTable()
        s0 = t.append(torch.randn(1, 4), bucket_id=2, step=5)
        s1 = t.append(torch.randn(1, 4), bucket_id=3, step=10)
        t.quarantine(s0)
        rec = t.record(s1)
        rec.utility_ema = 0.42
        rec.peak_utility = 0.99

        sd = t.state_dict()
        t2 = SlotTable()
        t2.load_state_dict(sd)

        assert len(t2) == 2
        assert t2.record(s0).state == SLOT_QUARANTINED
        assert t2.record(s1).utility_ema == pytest.approx(0.42)
        assert t2.record(s1).peak_utility == pytest.approx(0.99)

    def test_retired_excluded_from_state_dict(self):
        t = SlotTable()
        s0 = t.append(torch.randn(1, 4))
        t.retire(s0, reason="test")
        sd = t.state_dict()
        assert len(sd["slots"]) == 0


class TestReplaceTensor:
    def test_replace(self):
        t = SlotTable()
        s0 = t.append(torch.zeros(1, 4))
        new = torch.ones(1, 4)
        t.replace_tensor(s0, new)
        got = t.get_tensor(s0)
        assert (got == new).all()

    def test_replace_nonexistent(self):
        t = SlotTable()
        assert t.replace_tensor(999, torch.randn(1, 4)) is False


class TestPurgeRetired:
    def test_purge(self):
        t = SlotTable()
        s0 = t.append(torch.randn(1, 4))
        s1 = t.append(torch.randn(1, 4))
        t.retire(s0, reason="test")
        assert t.record(s0).state == SLOT_RETIRED
        n = t.purge_retired()
        assert n == 1
        assert t.record(s0) is None
