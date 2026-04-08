"""Polyphasic partitioned sleep — semantic partitions and topology layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SemanticPartition:
    """A logical partition of the semantic engine that cycles between awake and sleeping."""

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
        """Return True if this partition owns the given bucket."""
        return bucket_id in self.bucket_ids


class PartitionTopology:
    """Factory for partition layouts and helpers for querying partition state."""

    def __init__(self, partitions: list[SemanticPartition]) -> None:
        self.partitions = partitions

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def slot_striped(cls, n_partitions: int, k_max: int = 16) -> PartitionTopology:
        """Round-robin bucket assignment across partitions.

        Each partition gets every Nth bucket (round-robin), giving uniform
        coverage without semantic grouping. This is the control topology
        for testing polyphasic economics independent of typed ownership.
        """
        bucket_sets: list[set[int]] = [set() for _ in range(n_partitions)]
        for b in range(k_max):
            bucket_sets[b % n_partitions].add(b)
        partitions = [
            SemanticPartition(partition_id=i, bucket_ids=bucket_sets[i])
            for i in range(n_partitions)
        ]
        return cls(partitions)

    def slot_owner_map(self, total_slots: int) -> list[int]:
        """Return a list where index *i* is the partition that owns slot *i*."""
        n = len(self.partitions)
        return [i % n for i in range(total_slots)]

    @classmethod
    def bucket_owned(cls, n_partitions: int, k_max: int) -> PartitionTopology:
        """Each partition exclusively owns ``k_max / n_partitions`` contiguous buckets."""
        buckets_per_partition = k_max // n_partitions
        partitions = []
        for i in range(n_partitions):
            start = i * buckets_per_partition
            end = start + buckets_per_partition
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
        """Bucket families mapped to GPU groups.

        Partitions within the same group share the same bucket assignments.
        With *n_partitions=4*, *k_max=16*, *group_size=2*:
        partitions 0,1 share buckets {0..7}; partitions 2,3 share {8..15}.
        """
        n_groups = n_partitions // group_size
        buckets_per_group = k_max // n_groups
        partitions = []
        for i in range(n_partitions):
            group_idx = i // group_size
            start = group_idx * buckets_per_group
            end = start + buckets_per_group
            partitions.append(
                SemanticPartition(
                    partition_id=i,
                    bucket_ids=set(range(start, end)),
                )
            )
        return cls(partitions)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def awake_partitions(self) -> list[SemanticPartition]:
        """Return all partitions currently in awake mode."""
        return [p for p in self.partitions if p.is_awake]

    def sleeping_partitions(self) -> list[SemanticPartition]:
        """Return all partitions currently in sleeping mode."""
        return [p for p in self.partitions if p.is_sleeping]

    def awake_bucket_ids(self) -> set[int]:
        """Return the union of bucket_ids across all awake partitions."""
        result: set[int] = set()
        for p in self.awake_partitions():
            result |= p.bucket_ids
        return result


class PolyphasicScheduler:
    """K-of-N polyphasic sleep scheduler with round-robin rotation."""

    def __init__(
        self,
        topology: PartitionTopology,
        k_awake: int,
        swap_interval: int = 256,
    ) -> None:
        n = len(topology.partitions)
        if not (0 < k_awake < n):
            raise ValueError(f"k_awake must be in (0, {n}), got {k_awake}")
        self.topology = topology
        self.k_awake = k_awake
        self.swap_interval = swap_interval
        self._step_count = 0
        self._sleep_cursor = 0

        n = len(self.topology.partitions)
        n_sleeping = n - k_awake

        # Set initial assignment: first n_sleeping partitions sleeping, rest awake
        for i, p in enumerate(self.topology.partitions):
            p.mode = "sleeping" if i < n_sleeping else "awake"

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
        """Round-robin: shift the sleeping window by 1 position.

        N partitions form a ring.  The sleeping window is N-K consecutive
        partitions starting at ``_sleep_cursor``.  After rotation the cursor
        advances by 1 (mod N).
        """
        n = len(self.topology.partitions)
        n_sleeping = n - self.k_awake

        # Wake the partition that is leaving the sleeping window
        leaving = self._sleep_cursor
        self.topology.partitions[leaving].mode = "awake"

        # Advance cursor
        self._sleep_cursor = (self._sleep_cursor + 1) % n

        # Put the partition entering the sleeping window to sleep
        entering = (self._sleep_cursor + n_sleeping - 1) % n
        self.topology.partitions[entering].mode = "sleeping"
