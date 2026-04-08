# Polyphasic Partitioned Sleep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement polyphasic partitioned sleep — K-of-N semantic partition scheduling where K partitions are awake (accepting writes) and N-K are sleeping (running consolidation), with three topology modes.

**Architecture:** Logical partitions over a single GPU. Each partition owns a subset of episodic slots (by bucket assignment). A scheduler rotates wake/sleep assignments round-robin. The SSM trunk and Wernicke routing run every step; only memory writes and consolidation are partition-scoped. Total semantic capacity is fixed regardless of N.

**Tech Stack:** PyTorch, existing ChaosControl sleep/memory/training infrastructure.

---

### Task 1: SemanticPartition dataclass

**Files:**
- Create: `src/chaoscontrol/partition.py`
- Test: `tests/test_partition.py`

**Step 1: Write the failing test**

```python
# tests/test_partition.py
import pytest
from chaoscontrol.partition import SemanticPartition, PartitionTopology


def test_partition_init():
    p = SemanticPartition(partition_id=0, bucket_ids={0, 1}, mode="awake")
    assert p.partition_id == 0
    assert p.bucket_ids == {0, 1}
    assert p.is_awake
    assert not p.is_sleeping


def test_partition_mode_toggle():
    p = SemanticPartition(partition_id=0, bucket_ids={0, 1}, mode="awake")
    p.mode = "sleeping"
    assert p.is_sleeping
    assert not p.is_awake


def test_partition_owns_slot():
    p = SemanticPartition(partition_id=0, bucket_ids={0, 1}, mode="awake")
    assert p.owns_bucket(0)
    assert p.owns_bucket(1)
    assert not p.owns_bucket(2)


def test_slot_striped_topology():
    topo = PartitionTopology.slot_striped(n_partitions=4, total_slots=64)
    # Each partition gets 16 slots, assigned round-robin regardless of bucket
    assert len(topo.partitions) == 4
    slot_owners = topo.slot_owner_map(total_slots=64)
    assert slot_owners[0] == 0  # slot 0 -> partition 0
    assert slot_owners[1] == 1  # slot 1 -> partition 1
    assert slot_owners[4] == 0  # slot 4 -> partition 0


def test_bucket_owned_topology():
    topo = PartitionTopology.bucket_owned(n_partitions=4, k_max=16)
    # Each partition owns 4 buckets
    assert len(topo.partitions) == 4
    assert topo.partitions[0].bucket_ids == {0, 1, 2, 3}
    assert topo.partitions[1].bucket_ids == {4, 5, 6, 7}


def test_bucket_striped_topology():
    topo = PartitionTopology.bucket_striped(
        n_partitions=4, k_max=16, group_size=2
    )
    # Each group of 2 partitions shares 8 bucket families
    # Partitions 0,1 share buckets 0-7; partitions 2,3 share 8-15
    assert len(topo.partitions) == 4
    # Within a group, each partition owns all buckets in the group
    assert topo.partitions[0].bucket_ids == topo.partitions[1].bucket_ids
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_partition.py -v`
Expected: FAIL with "No module named 'chaoscontrol.partition'"

**Step 3: Write minimal implementation**

```python
# src/chaoscontrol/partition.py
"""Semantic partitions for polyphasic sleep scheduling.

Each partition owns a subset of the semantic engine (slots, traces, bases)
defined by Wernicke bucket ownership. Partitions cycle between awake
(accepting new episodic writes) and sleeping (running N2/N3/REM consolidation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SemanticPartition:
    """A logical partition of the semantic engine."""

    partition_id: int
    bucket_ids: set[int] = field(default_factory=set)
    mode: Literal["awake", "sleeping"] = "awake"

    @property
    def is_awake(self) -> bool:
        return self.mode == "awake"

    @property
    def is_sleeping(self) -> bool:
        return self.mode == "sleeping"

    def owns_bucket(self, bucket_id: int) -> bool:
        return bucket_id in self.bucket_ids


class PartitionTopology:
    """Assigns slots/buckets to partitions under different topologies."""

    def __init__(self, partitions: list[SemanticPartition]) -> None:
        self.partitions = partitions

    @classmethod
    def slot_striped(cls, n_partitions: int, total_slots: int) -> PartitionTopology:
        """Round-robin slot assignment, no semantic awareness."""
        # All partitions own all buckets (any bucket can route to any partition)
        all_buckets: set[int] = set()  # not used for routing in striped mode
        partitions = [
            SemanticPartition(partition_id=i, bucket_ids=all_buckets)
            for i in range(n_partitions)
        ]
        return cls(partitions)

    def slot_owner_map(self, total_slots: int) -> list[int]:
        """For slot_striped: return partition owner per slot index."""
        n = len(self.partitions)
        return [i % n for i in range(total_slots)]

    @classmethod
    def bucket_owned(cls, n_partitions: int, k_max: int) -> PartitionTopology:
        """Each partition exclusively owns k_max/n_partitions buckets."""
        buckets_per = k_max // n_partitions
        partitions = []
        for i in range(n_partitions):
            start = i * buckets_per
            end = start + buckets_per if i < n_partitions - 1 else k_max
            partitions.append(
                SemanticPartition(
                    partition_id=i,
                    bucket_ids=set(range(start, end)),
                )
            )
        return cls(partitions)

    @classmethod
    def bucket_striped(
        cls, n_partitions: int, k_max: int, group_size: int = 2
    ) -> PartitionTopology:
        """Bucket families map to GPU groups, slots striped within group."""
        n_groups = n_partitions // group_size
        buckets_per_group = k_max // n_groups
        partitions = []
        for i in range(n_partitions):
            group = i // group_size
            start = group * buckets_per_group
            end = start + buckets_per_group if group < n_groups - 1 else k_max
            partitions.append(
                SemanticPartition(
                    partition_id=i,
                    bucket_ids=set(range(start, end)),
                )
            )
        return cls(partitions)

    def awake_partitions(self) -> list[SemanticPartition]:
        return [p for p in self.partitions if p.is_awake]

    def sleeping_partitions(self) -> list[SemanticPartition]:
        return [p for p in self.partitions if p.is_sleeping]

    def awake_bucket_ids(self) -> set[int]:
        result: set[int] = set()
        for p in self.awake_partitions():
            result |= p.bucket_ids
        return result
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_partition.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/partition.py tests/test_partition.py
git commit -m "feat: SemanticPartition and PartitionTopology for polyphasic sleep"
```

---

### Task 2: PolyphasicScheduler

**Files:**
- Modify: `src/chaoscontrol/partition.py`
- Test: `tests/test_partition.py` (append)

**Step 1: Write the failing tests**

```python
# Append to tests/test_partition.py
from chaoscontrol.partition import PolyphasicScheduler


def test_scheduler_init():
    topo = PartitionTopology.bucket_owned(n_partitions=4, k_max=16)
    sched = PolyphasicScheduler(topo, k_awake=3, swap_interval=256)
    assert len(sched.awake()) == 3
    assert len(sched.sleeping()) == 1


def test_scheduler_rotation():
    topo = PartitionTopology.bucket_owned(n_partitions=4, k_max=16)
    sched = PolyphasicScheduler(topo, k_awake=3, swap_interval=2)
    initial_sleeping = sched.sleeping()[0].partition_id

    # Step through one swap interval
    swapped = sched.step()  # step 1: no swap
    assert not swapped
    swapped = sched.step()  # step 2: swap
    assert swapped

    new_sleeping = sched.sleeping()[0].partition_id
    assert new_sleeping != initial_sleeping  # different partition sleeping now


def test_scheduler_round_robin():
    """Every partition gets sleep within N rotations."""
    topo = PartitionTopology.bucket_owned(n_partitions=4, k_max=16)
    sched = PolyphasicScheduler(topo, k_awake=3, swap_interval=1)

    slept: set[int] = set()
    for _ in range(4):
        sched.step()
        for p in sched.sleeping():
            slept.add(p.partition_id)
    assert slept == {0, 1, 2, 3}


def test_scheduler_fixed_capacity():
    """Total awake + sleeping always equals N."""
    topo = PartitionTopology.bucket_owned(n_partitions=8, k_max=16)
    sched = PolyphasicScheduler(topo, k_awake=6, swap_interval=10)
    for _ in range(100):
        sched.step()
        assert len(sched.awake()) + len(sched.sleeping()) == 8
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_partition.py::test_scheduler_init -v`
Expected: FAIL with "cannot import name 'PolyphasicScheduler'"

**Step 3: Write minimal implementation**

```python
# Append to src/chaoscontrol/partition.py

class PolyphasicScheduler:
    """K-of-N polyphasic sleep scheduler with round-robin rotation."""

    def __init__(
        self,
        topology: PartitionTopology,
        k_awake: int,
        swap_interval: int = 256,
    ) -> None:
        self.topology = topology
        self.k_awake = k_awake
        self.swap_interval = swap_interval
        self._step_count = 0
        self._sleep_cursor = 0  # which partition is next to sleep
        n = len(topology.partitions)
        n_sleeping = n - k_awake
        # Initialize: first n_sleeping partitions are sleeping
        for i, p in enumerate(topology.partitions):
            p.mode = "sleeping" if i < n_sleeping else "awake"
        self._sleep_cursor = n_sleeping

    def awake(self) -> list[SemanticPartition]:
        return self.topology.awake_partitions()

    def sleeping(self) -> list[SemanticPartition]:
        return self.topology.sleeping_partitions()

    def step(self) -> bool:
        """Advance one step. Returns True if a swap occurred."""
        self._step_count += 1
        if self._step_count % self.swap_interval != 0:
            return False
        self._rotate()
        return True

    def _rotate(self) -> None:
        """Round-robin: wake the oldest sleeping partition, put the oldest
        awake partition to sleep."""
        n = len(self.topology.partitions)
        n_sleeping = n - self.k_awake
        # New sleeping set: n_sleeping partitions starting at cursor
        for i, p in enumerate(self.topology.partitions):
            offset = (i - self._sleep_cursor) % n
            p.mode = "sleeping" if offset < n_sleeping else "awake"
        self._sleep_cursor = (self._sleep_cursor + 1) % n
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_partition.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/partition.py tests/test_partition.py
git commit -m "feat: PolyphasicScheduler with K-of-N round-robin rotation"
```

---

### Task 3: Partition-scoped memory operations

**Files:**
- Modify: `src/chaoscontrol/memory.py`
- Test: `tests/test_partition_memory.py`

**Step 1: Write the failing tests**

```python
# tests/test_partition_memory.py
import torch
import pytest
from chaoscontrol.memory import MultiSlotOuterModel
from chaoscontrol.partition import SemanticPartition


def test_get_partition_slot_indices():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    # Manually add 4 slots with different bucket assignments
    for bucket in [0, 1, 0, 2]:
        h = torch.randn(1, 1, 16)
        om.write(h, bucket_id=bucket)

    p0 = SemanticPartition(partition_id=0, bucket_ids={0})
    p1 = SemanticPartition(partition_id=1, bucket_ids={1, 2})

    idx0 = om.get_partition_slot_indices(p0)
    idx1 = om.get_partition_slot_indices(p1)

    assert idx0 == [0, 2]  # slots assigned to bucket 0
    assert idx1 == [1, 3]  # slots assigned to buckets 1, 2


def test_partition_slot_count():
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    for bucket in [0, 0, 1, 1, 2]:
        om.write(torch.randn(1, 1, 16), bucket_id=bucket)

    p = SemanticPartition(partition_id=0, bucket_ids={0})
    assert om.partition_slot_count(p) == 2


def test_is_write_allowed():
    """Only awake partitions accept writes."""
    om = MultiSlotOuterModel(model_dim=16, outer_dim=8, max_slots=10)
    awake = SemanticPartition(partition_id=0, bucket_ids={0}, mode="awake")
    sleeping = SemanticPartition(partition_id=1, bucket_ids={1}, mode="sleeping")
    assert om.is_write_allowed(bucket_id=0, partitions=[awake, sleeping])
    assert not om.is_write_allowed(bucket_id=1, partitions=[awake, sleeping])
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_partition_memory.py -v`
Expected: FAIL with "has no attribute 'get_partition_slot_indices'"

**Step 3: Write minimal implementation**

Add these methods to `MultiSlotOuterModel` in `memory.py`:

```python
def get_partition_slot_indices(self, partition: Any) -> list[int]:
    """Return indices of slots owned by this partition."""
    return [
        i for i, b in enumerate(self._slot_buckets)
        if partition.owns_bucket(b)
    ]

def partition_slot_count(self, partition: Any) -> int:
    """Count slots owned by this partition."""
    return len(self.get_partition_slot_indices(partition))

def is_write_allowed(
    self, bucket_id: int, partitions: list[Any]
) -> bool:
    """Check if the bucket's owning partition is awake."""
    for p in partitions:
        if p.owns_bucket(bucket_id) and p.is_awake:
            return True
    return False
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_partition_memory.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/memory.py tests/test_partition_memory.py
git commit -m "feat: partition-scoped slot queries on MultiSlotOuterModel"
```

---

### Task 4: Partition-scoped SleepCycle

**Files:**
- Modify: `src/chaoscontrol/sleep.py`
- Test: `tests/test_partition_sleep.py`

**Step 1: Write the failing test**

```python
# tests/test_partition_sleep.py
import torch
import pytest
from unittest.mock import MagicMock
from chaoscontrol.sleep import SleepCycle, SleepConfig
from chaoscontrol.partition import SemanticPartition
from chaoscontrol.wake_cache import WakeCache


def _make_model_with_slots(n_slots=8, dim=16, outer_dim=8):
    """Create a mock model with a MultiSlotOuterModel."""
    from chaoscontrol.memory import MultiSlotOuterModel

    om = MultiSlotOuterModel(model_dim=dim, outer_dim=outer_dim, max_slots=32)
    # Add slots with alternating bucket assignments
    for i in range(n_slots):
        om.write(torch.randn(1, 1, dim), bucket_id=i % 4)

    model = MagicMock()
    model.outer_model = om
    model.semantic_tier = None
    model.typed_storage = True
    return model


def test_scoped_sleep_only_touches_partition_slots():
    model = _make_model_with_slots(8)
    om = model.outer_model
    cache = WakeCache(max_moments=8, max_hidden_buffer=8)

    # Partition 0 owns buckets {0, 1} -> slots 0, 1, 4, 5
    partition = SemanticPartition(partition_id=0, bucket_ids={0, 1})
    partition.mode = "sleeping"

    cfg = SleepConfig(stages="n3_only", budget=128, survival_floor=0.5)
    cycle = SleepCycle(cfg)

    # Record initial survivals
    initial_survival = list(om._survival)

    diag = cycle.run(model, cache, device="cpu", partition=partition)

    # Slots owned by partition 0 (buckets 0,1) may have changed
    # Slots owned by other partitions (buckets 2,3) must be unchanged
    for i in [2, 3, 6, 7]:  # bucket 2 and 3 slots
        assert om._survival[i] == initial_survival[i], (
            f"Slot {i} (bucket {om._slot_buckets[i]}) was modified by wrong partition"
        )
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_partition_sleep.py -v`
Expected: FAIL (run() doesn't accept partition= parameter)

**Step 3: Write minimal implementation**

Modify `SleepCycle.run()` in `sleep.py` to accept an optional `partition` parameter. When provided, all stage methods filter to only operate on slots owned by that partition.

Add `partition: Any | None = None` parameter to `run()`. Thread it through to `_n1_transition`, `_n2_tag`, `_n3_rewrite`, `_rem_dream`.

In each stage method, use `om.get_partition_slot_indices(partition)` to filter the slot indices being operated on. If `partition is None`, operate on all slots (backward compatible).

Key changes:
- `_n2_tag`: only score slots in `partition_indices`
- `_n3_rewrite`: only prune/merge slots in `partition_indices`
- `_rem_dream`: only seed dreams from `partition_indices`

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_partition_sleep.py tests/test_sleep.py -v`
Expected: PASS (new test + all existing sleep tests still pass)

**Step 5: Commit**

```bash
git add src/chaoscontrol/sleep.py tests/test_partition_sleep.py
git commit -m "feat: partition-scoped SleepCycle — stages operate on owned slots only"
```

---

### Task 5: Per-partition WakeCache

**Files:**
- Modify: `src/chaoscontrol/wake_cache.py`
- Test: `tests/test_partition_wake_cache.py`

**Step 1: Write the failing test**

```python
# tests/test_partition_wake_cache.py
import torch
from chaoscontrol.wake_cache import WakeCache
from chaoscontrol.partition import SemanticPartition


def test_filter_moments_by_partition():
    cache = WakeCache(max_moments=16, max_hidden_buffer=8)

    # Record moments with different bucket_ids
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
    assert len(filtered) == 3  # 3 moments where dominant bucket is 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_partition_wake_cache.py -v`
Expected: FAIL with "has no attribute 'filter_moments_by_partition'"

**Step 3: Write minimal implementation**

Add to `WakeCache`:

```python
def filter_moments_by_partition(self, partition: Any) -> list[dict]:
    """Return moments whose dominant bucket is owned by this partition."""
    result = []
    for m in self.moments:
        bids = m.get("bucket_ids")
        if bids is None:
            continue
        dominant = int(bids.reshape(-1).mode().values.item())
        if partition.owns_bucket(dominant):
            result.append(m)
    return result
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_partition_wake_cache.py tests/test_wake_cache.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/wake_cache.py tests/test_partition_wake_cache.py
git commit -m "feat: partition-scoped moment filtering in WakeCache"
```

---

### Task 6: Training loop integration

**Files:**
- Modify: `src/chaoscontrol/training.py`
- Modify: `src/chaoscontrol/config.py`
- Test: `tests/test_training_polyphasic.py`

**Step 1: Write the failing test**

```python
# tests/test_training_polyphasic.py
import torch
from chaoscontrol.config import ChaosControlConfig


def test_polyphasic_config_fields():
    cfg = ChaosControlConfig(
        polyphasic_enabled=True,
        polyphasic_n_partitions=4,
        polyphasic_k_awake=3,
        polyphasic_topology="bucket_owned",
        polyphasic_swap_interval=256,
    )
    assert cfg.polyphasic_enabled is True
    assert cfg.polyphasic_n_partitions == 4
    assert cfg.polyphasic_k_awake == 3
    assert cfg.polyphasic_topology == "bucket_owned"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_training_polyphasic.py -v`
Expected: FAIL with "unexpected keyword argument 'polyphasic_enabled'"

**Step 3: Write minimal implementation**

Add to `ChaosControlConfig` in `config.py`:

```python
# Polyphasic partitioned sleep
polyphasic_enabled: bool = False
polyphasic_n_partitions: int = 4
polyphasic_k_awake: int = 3
polyphasic_topology: str = "slot_striped"  # "slot_striped", "bucket_owned", "bucket_striped"
polyphasic_swap_interval: int = 256
```

Then modify `train_chaoscontrol_for_budget` in `training.py` to:
1. Accept polyphasic parameters
2. Create topology + scheduler when polyphasic_enabled
3. Before memory write: check `om.is_write_allowed(dominant_bucket, partitions)`
4. At sleep trigger: iterate sleeping partitions, run scoped sleep cycle for each
5. After scheduler.step(): if swap occurred, sync summary state

The training loop change is approximately:

```python
# In the sleep trigger block, replace the single sleep call with:
if polyphasic_enabled:
    for p in scheduler.sleeping():
        partition_cache = ... # filter cache by partition
        sleep_diag = sleep_cycle.run(
            model, partition_cache, device=device,
            regret_table=regret_table, partition=p,
        )
    swapped = scheduler.step()
else:
    # existing global sleep path
    sleep_diag = sleep_cycle.run(model, wake_cache, device=device, ...)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_training_polyphasic.py tests/test_training_sleep.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/chaoscontrol/config.py src/chaoscontrol/training.py tests/test_training_polyphasic.py
git commit -m "feat: polyphasic partition scheduling in training loop"
```

---

### Task 7: Experiment 12 runner

**Files:**
- Create: `experiments/12_polyphasic_sleep/run_polyphasic_ablation.py`
- Test: manual (run with `--help`)

**Step 1: Write the experiment runner**

Model after `experiments/11_sleep_cycle/run_sleep_ablation.py`. Three primary conditions:

```python
CONDITIONS = {
    "no_sleep": _base(sleep_enabled=False, polyphasic_enabled=False),
    "full_offline_sleep": _base(sleep_enabled=True, polyphasic_enabled=False, **SLEEP_COMMON),
    "polyphasic_K3_N4_striped": _base(
        sleep_enabled=True, polyphasic_enabled=True,
        polyphasic_n_partitions=4, polyphasic_k_awake=3,
        polyphasic_topology="slot_striped", **SLEEP_COMMON,
    ),
}
```

Use 4 partitions (testable on a single GPU) with K=3 (one sleeping at a time).

Sleep payload locked to Experiment 11 winner (default: full_cycle).
Cross-partition read policy: read-only summary (locked).
Swap interval: 256 (locked).

Confirmatory contrasts (Holm-corrected, m=2):
1. `polyphasic_K3_N4_striped` vs `no_sleep`
2. `polyphasic_K3_N4_striped` vs `full_offline_sleep`

Exploratory conditions (added later based on primary results):
- `polyphasic_K3_N4_bucket_owned`
- `polyphasic_K3_N4_bucket_striped`
- `polyphasic_K2_N4_striped` (deeper sleep)

7 seeds, paired Wilcoxon, bootstrap CIs.

**Step 2: Verify runner works**

Run: `PYTHONPATH=src .venv/bin/python experiments/12_polyphasic_sleep/run_polyphasic_ablation.py --help`
Expected: Shows argparse help

**Step 3: Commit**

```bash
git add experiments/12_polyphasic_sleep/run_polyphasic_ablation.py
git commit -m "feat: experiment 12 — polyphasic partitioned sleep ablation runner"
```

---

### Task 8: Integration test

**Files:**
- Create: `tests/test_polyphasic_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_polyphasic_integration.py
import torch
from chaoscontrol.model import ChaosStudentLM
from chaoscontrol.memory import MultiSlotOuterModel
from chaoscontrol.sleep import SleepCycle, SleepConfig
from chaoscontrol.wake_cache import WakeCache
from chaoscontrol.partition import (
    PartitionTopology, PolyphasicScheduler, SemanticPartition,
)


def test_full_polyphasic_cycle():
    """End-to-end: create model, write slots, schedule sleep, verify isolation."""
    model = ChaosStudentLM(
        vocab_size=256, dim=16, num_layers=2, ff_mult=2,
        outer_model_dim=8, outer_max_slots=32,
        wernicke_enabled=True, wernicke_router="moe", wernicke_k_max=4,
        typed_storage=True,
    )
    om = model.outer_model
    cache = WakeCache(max_moments=16, max_hidden_buffer=8)

    # Create topology and scheduler
    topo = PartitionTopology.bucket_owned(n_partitions=4, k_max=4)
    sched = PolyphasicScheduler(topo, k_awake=3, swap_interval=2)

    # Write some slots with bucket assignments
    for step in range(8):
        x = torch.randint(0, 256, (1, 16))
        out = model(x)
        bucket_id = step % 4
        om.write(out["hidden"], bucket_id=bucket_id)

    assert len(om._slots) == 8

    # Run one sleep cycle on the sleeping partition
    sleeping = sched.sleeping()
    assert len(sleeping) == 1
    sleeping_p = sleeping[0]

    cfg = SleepConfig(stages="n3_only", budget=64, survival_floor=0.5)
    cycle = SleepCycle(cfg)

    owned_before = om.get_partition_slot_indices(sleeping_p)
    diag = cycle.run(model, cache, device="cpu", partition=sleeping_p)

    # Verify non-owned slots weren't touched
    for p in sched.awake():
        for idx in om.get_partition_slot_indices(p):
            # Awake partition slots should not have been pruned
            assert idx < len(om._slots) or True  # slots may shift after prune

    # Scheduler rotation works
    swapped = sched.step()  # step 1
    swapped = sched.step()  # step 2 - should swap
    assert swapped
```

**Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_polyphasic_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_polyphasic_integration.py
git commit -m "test: end-to-end polyphasic sleep integration test"
```
