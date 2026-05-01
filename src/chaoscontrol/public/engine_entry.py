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
    raise NotImplementedError


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
