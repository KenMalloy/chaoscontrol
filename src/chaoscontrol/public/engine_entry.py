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
    """Build an ARM runner config dict from training hyperparams.

    Merges exp26 lock functions, telemetry-tuned overrides, eval routing, and
    any relevant attributes from *hyperparams*.  Telemetry overrides win over
    whatever the lock functions return.
    """
    # Locate and import exp26 at call time so the module-level import never
    # fails if CHAOSCONTROL_ROOT is not set during library import.
    _root = os.environ.get("CHAOSCONTROL_ROOT", "/workspace/chaoscontrol")
    _exp26_dir = os.path.join(_root, "experiments", "26_arm")
    if _exp26_dir not in sys.path:
        sys.path.insert(0, _exp26_dir)
    import exp26  # noqa: PLC0415

    # --- 1. Baseline lock dicts (fast_slow → crct → pipeline) ---
    cfg: dict[str, Any] = {}
    cfg.update(exp26._fast_slow_lock())
    cfg.update(exp26._crct_lock())
    cfg.update(exp26._replay_eviction_pipeline_lock())

    # --- 2. Telemetry-tuned overrides (must win over lock functions) ---
    cfg["crct_memory_write_tokens_per_step"] = 256
    cfg["online_episodic_write_tokens_per_chunk"] = 64
    cfg["crct_target_write_rate"] = 0.25

    # --- 3. Eval routing ---
    cfg["calc_types"] = ["packet_online_cache"]
    cfg["headline_calc_type"] = "packet_online_cache"

    # --- 4. Forward hyperparams attributes ---
    _MISSING = object()
    _FORWARD_KEYS = (
        "model_dim",
        "budget_seconds",
        "data_path",
        "val_cache_dir",
        "sp_model_path",
    )
    for key in _FORWARD_KEYS:
        val = getattr(hyperparams, key, _MISSING)
        if val is not _MISSING:
            cfg[key] = val

    return cfg


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
