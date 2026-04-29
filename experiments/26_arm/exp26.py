#!/usr/bin/env python3
"""Exp26 Adaptive Residual Memory validation matrix builder.

Exp26 is no longer a headline ablation matrix. It is a fixed systems canary:
one locked fast/slow control and one full Adaptive Residual Memory cell. The
ARM cell includes CRCT evidence, GPU3 oracle scoring, streaming maintenance,
learned Full-A commit authority, traces, and the native CPU/GPU3 maintenance
runtime. There is intentionally no CRCT-only or shadow-mode switch here.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EXP24 = REPO / "experiments" / "24_training_time_bundle"
sys.path.insert(0, str(EXP24))

from exp24 import (  # noqa: E402
    ARTIFACT_CHANGES_WEIGHTS_ONLY,
    _base_entry,
)


EXP26_DIR = Path(__file__).resolve().parent
DEFAULT_VALIDATION_DIR = EXP26_DIR / "validation"
DEFAULT_VALIDATION_TRACE_DIR = DEFAULT_VALIDATION_DIR / "traces"

VALIDATION_ARMS: tuple[str, ...] = (
    "validation_fastslow_control",
    "validation_adaptive_residual_memory",
)

EXP26_MODEL_DIM = 384


def _named_entry(
    *,
    base: dict[str, Any],
    phase: str,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    entry = copy.deepcopy(base)
    name = f"exp26_{phase}_{arm}_s{int(seed)}"
    entry.update(
        {
            "name": name,
            "seed": int(seed),
            "exp26_phase": phase,
            "exp26_mechanism": "arm_v1",
        }
    )
    if entry.get("replay_eviction_arm_runtime_enabled"):
        entry["replay_eviction_arm_runtime_namespace"] = name
    return entry


def _artifact_size_lock() -> dict[str, Any]:
    """Artifact-safe trunk size for the 16k-vocab Exp26 validation run.

    Local artifact-pipeline sizing on the CRCT+bucket-prototype shape:
    dim=384 -> 13.71 MB int6/LZMA-preset0; dim=416 -> 15.19 MB; dim=448
    -> 16.73 MB; dim=512 -> 20.16 MB. 384 is the largest comfortable lock
    with enough room for trained-weight entropy and artifact metadata drift.
    """
    return {"model_dim": EXP26_MODEL_DIM}


def _fast_slow_lock() -> dict[str, Any]:
    """Locked Phase-0 fast/slow trunk recipe inherited from exp24."""
    return {
        "fast_slow_enabled": True,
        "fast_slow_interval": 64,
        "fast_slow_alpha": 0.25,
        "fast_slow_eval_copy": "slow",
        "dreamworld_enabled": False,
        "dreamworld_cache_interval": 0,
        "dreamworld_interval": 0,
        "dreamworld_weight": 0.0,
        "dreamworld_replay_batch_size": 0,
        "train_sampling_mode": "random",
    }


def _crct_lock() -> dict[str, Any]:
    """Locked CRCT evidence/oracle substrate configuration."""
    return {
        "crct_enabled": True,
        "crct_lambda_controller": 0.01,
        "crct_lm_weight_alpha_max": 0.15,
        "crct_lm_weight_strength": 0.10,
        "crct_lm_weight_w_max": 1.20,
        "crct_lm_weight_tau": 0.10,
        "crct_target_read_rate": 0.25,
        "crct_target_write_rate": 0.10,
        "crct_dual_lr": 0.01,
        "crct_ema_beta": 0.95,
        "crct_max_price": 0.50,
        "crct_memory_write_tokens_per_step": 128,
        "crct_async_teacher_transport": True,
        "crct_async_teacher_transport_backend": "mailbox",
        "crct_async_teacher_pending_batches": 64,
        "crct_async_teacher_max_lag_steps": 128,
        "crct_async_teacher_payload_dtype": "auto",
        "crct_teacher_score_interval_steps": 64,
        "crct_teacher_param_sync_interval_steps": 0,
        "outer_model_dim": 64,
        "outer_model_type": "multislot",
        "outer_max_slots": 4096,
        "outer_compress_ratio": 2,
        "buffer_mode": "append_only",
        "retrieval_mode": "softmax_all",
        "retrieval_k": 16,
        "enable_controller": True,
        "controller_hidden_dim": 64,
        "train_sampling_mode": "random",
        "compile_full_path": False,
        "cuda_graph_mode": "none",
        "crct_gradient_conflict_enabled": True,
        "crct_gradient_conflict_soft_gate_strength": 0.0,
        "crct_gradient_conflict_trace_path": str(
            DEFAULT_VALIDATION_TRACE_DIR / "crct_conflict.ndjson"
        ),
        "crct_gradient_conflict_trace_max_rows": 200000,
    }


def _replay_eviction_pipeline_lock() -> dict[str, Any]:
    """Streaming ARM pipeline knobs for the fixed validation cell.

    Threshold priors and shadow/headline splits are intentionally absent. The
    learned controller owns commit authority and GPU3 supplies physics
    confirmation.
    """
    return {
        "replay_eviction_enabled": True,
        "bucket_prototypes": True,
        "prototype_dim": 64,
        "replay_eviction_memory_streams": 8,
        "replay_eviction_arm_runtime_enabled": True,
        # Rank-3 maintenance runs off the trunk critical path. The generic
        # 0.5s default was a scaffold-era placeholder and suppresses action
        # telemetry at Exp26's real 16k-vocab probe cost.
        "replay_eviction_max_seconds": 8.0,
        "replay_eviction_scoring_mode": "oracle",
        "replay_eviction_oracle_confirm_top_k": 32,
        "replay_eviction_oracle_variant_chunk_size": 1,
        "replay_eviction_refresh_candidate_count": 16,
        "replay_eviction_refresh_proposal_rank": 8,
        "replay_eviction_refresh_proposal_noise_scale": 0.04,
        "replay_eviction_refresh_proposal_momentum": 0.9,
        "replay_eviction_refresh_proposal_weight_sync_interval_steps": 64,
        "replay_eviction_refresh_candidate_variant_chunk_size": 16,
        "replay_eviction_refresh_proposal_seed": 1729,
        "replay_eviction_controller_state_dim": 32,
        "replay_eviction_controller_rank": 8,
        "replay_eviction_controller_dt": 1.0,
        "replay_eviction_controller_gamma": 0.08,
        "replay_eviction_controller_target_log_sv": -0.05,
        "replay_eviction_controller_max_state_norm": 8.0,
        "replay_eviction_controller_perturbation_scale": 0.25,
        "replay_eviction_controller_feedback_lr": 0.05,
        "replay_eviction_commit_policy": "learned",
        "replay_eviction_commit_online_lr": 0.05,
        "replay_eviction_commit_temperature": 0.75,
        "replay_eviction_evidence_engine_enabled": True,
        "replay_eviction_evidence_engine_d_model": EXP26_MODEL_DIM,
        "replay_eviction_probe_buffer_size": 32,
        "replay_eviction_frame_ttl_steps": 256,
        "replay_eviction_slot_work_chunk_size": 16,
        "replay_eviction_trace_max_rows": 200000,
        "replay_eviction_trace_flush_rows": 256,
    }


def build_validation_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 45.0,
    seed: int = 1337,
) -> list[dict[str, Any]]:
    """Short runtime validation for the architecture we actually mean.

    The second cell is not "CRCT only"; it is the full Adaptive Residual
    Memory path. CRCT is the evidence/oracle substrate inside ARM, not a
    standalone architecture switch.
    """
    size_lock = _artifact_size_lock()
    fast_slow = _fast_slow_lock()
    crct = _crct_lock()
    pipeline = _replay_eviction_pipeline_lock()
    arm_specs: list[tuple[str, dict[str, Any]]] = [
        ("validation_fastslow_control", {}),
        (
            "validation_adaptive_residual_memory",
            {
                **crct,
                **pipeline,
                "replay_eviction_mode": "active",
                "replay_eviction_action_agreement_count": 1,
                "replay_eviction_trace_path": str(
                    DEFAULT_VALIDATION_TRACE_DIR
                    / f"arm_validation_adaptive_residual_memory_s{int(seed)}.ndjson"
                ),
                "crct_gradient_conflict_trace_path": str(
                    DEFAULT_VALIDATION_TRACE_DIR / "crct_conflict.ndjson"
                ),
            },
        ),
    ]
    entries: list[dict[str, Any]] = []
    for arm_name, arm_overrides in arm_specs:
        arm = {
            "arm": arm_name,
            "exp26_mechanism": "arm_v1_validation",
            "artifact_impact": ARTIFACT_CHANGES_WEIGHTS_ONLY,
            **size_lock,
            **fast_slow,
            **arm_overrides,
        }
        entry = _base_entry(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
        )
        entry.update(arm)
        entries.append(
            _named_entry(
                base=entry,
                phase="validation",
                arm=arm_name,
                seed=int(seed),
            )
        )
    return entries
