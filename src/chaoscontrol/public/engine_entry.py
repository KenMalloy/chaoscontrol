# src/chaoscontrol/public/engine_entry.py
"""Public entry points for the SemanticEngine ARM submission.

This module is the stable public-facing interface for train_gpt.py.
It provides:
  - init_arm_topology(): GPU role routing (train / packet-serving / maintenance)
  - build_arm_config(): hyperparams -> run_condition config dict
  - run_arm_submission(): delegates to runner_fast_path.run_condition()

The heavy ARM training loop (~14,850 lines) lives in
experiments/23_fast_path/runner_fast_path.py and is called via
CHAOSCONTROL_ROOT-based sys.path injection. This keeps engine_entry.py thin
and reuses the existing production implementation.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RoleInfo:
    rank: int
    world_size: int
    packet_rank: int
    maintenance_rank: int
    is_train_rank: bool
    is_packet_rank: bool
    is_maintenance_rank: bool
    split_memory_ranks: bool


def init_arm_topology(rank: int, world_size: int) -> RoleInfo:
    """Assign GPU role based on rank and world_size.

    8+ GPU (split=True):  ranks 0..N-3 train, rank N-2 packet-serving, rank N-1 maintenance
    4 GPU (split=False):  ranks 0..2 train, rank 3 owns both memory roles
    1 GPU:                rank 0 is train only (no dedicated memory GPU)
    """
    world = int(world_size)
    if world > 0:
        assert 0 <= rank < world, f"rank {rank} out of range for world_size {world}"
    split = world >= 8
    packet_rank = world - (2 if split else 1)
    maintenance_rank = world - 1
    if world <= 1:
        # Single GPU: no dedicated memory ranks; rank 0 is train-only.
        is_packet = False
        is_maintenance = False
        is_train = True
    else:
        is_packet = rank == packet_rank
        is_maintenance = rank == maintenance_rank
        # On 4 GPU: packet_rank == maintenance_rank == 3, so that rank is both.
        is_train = not is_packet and not is_maintenance
    return RoleInfo(
        rank=rank,
        world_size=world,
        packet_rank=packet_rank,
        maintenance_rank=maintenance_rank,
        is_train_rank=is_train,
        is_packet_rank=is_packet,
        is_maintenance_rank=is_maintenance,
        split_memory_ranks=split,
    )


def build_arm_config(hyperparams: Any) -> dict[str, Any]:
    raise NotImplementedError


def run_arm_submission(
    config: dict[str, Any],
    *,
    data_path: str,
    sp_model_path: str,
    budget_seconds: float,
    output_json: str | None,
    val_cache_dir: str | None,
    world_size_override: int | None = None,
) -> dict[str, Any]:
    raise NotImplementedError
