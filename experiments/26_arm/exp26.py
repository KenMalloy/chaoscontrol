#!/usr/bin/env python3
"""Exp26 Adaptive Residual Memory headline matrix builders.

Two-stage discipline. Stage 1 can run a shadow-mode cell to observe the
actual distributions of ``utility_ema``, ``peak_utility``,
``peak_sharpness``, ``contradiction_ema``, and drift signals at our scale.
Stage 2 can still write a manifest for post-hoc threshold counterfactuals.
Stage 3 launches the headline matrix with the learned Full-A action simplex
owning the active commit decision.

The architecture under test is CRCT + streaming Adaptive Residual Memory.
The same fast/slow trunk and CRCT contract land from exp24 unchanged; the
active ARM cell adds learned maintenance authority, not a threshold ablation.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EXP24 = REPO / "experiments" / "24_training_time_bundle"
sys.path.insert(0, str(EXP24))

from exp24 import (  # noqa: E402
    ARTIFACT_CHANGES_WEIGHTS_ONLY,
    DEFAULT_CONTROL_SEEDS,
    _base_entry,
)


EXP26_DIR = Path(__file__).resolve().parent
DEFAULT_TRACE_DIR = EXP26_DIR / "results" / "traces"
DEFAULT_CALIBRATION_DIR = EXP26_DIR / "calibration"
DEFAULT_SMOKE_DIR = EXP26_DIR / "smoke"
DEFAULT_SMOKE_TRACE_DIR = DEFAULT_SMOKE_DIR / "traces"
DEFAULT_MANIFEST_PATH = DEFAULT_CALIBRATION_DIR / "manifest.json"
DEFAULT_CALIBRATION_TRACE = DEFAULT_CALIBRATION_DIR / "trace.ndjson"

ARM_V1_ARMS: tuple[str, ...] = (
    "arm_a_fastslow_control",
    "arm_b_crct_controller",
    "arm_c_crct_replay_shadow",
    "arm_d_crct_replay_active_learned",
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
    """Artifact-safe trunk size for the 16k-vocab Exp26 headline.

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
    """Locked CRCT v1 configuration. Identical to exp24's headline lock."""
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
            DEFAULT_TRACE_DIR / "crct_conflict.ndjson"
        ),
        "crct_gradient_conflict_trace_max_rows": 200000,
    }


def _replay_eviction_pipeline_lock() -> dict[str, Any]:
    """Streaming pipeline knobs. Same across calibration and headline arms.

    Threshold knobs are NOT here. Calibration/shadow can still emit
    threshold counterfactual telemetry, but the headline learned arm owns
    its commit decision without percentile-threshold priors.
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
        "replay_eviction_probe_buffer_size": 32,
        "replay_eviction_frame_ttl_steps": 256,
        "replay_eviction_slot_work_chunk_size": 16,
        "replay_eviction_trace_max_rows": 200000,
        "replay_eviction_trace_flush_rows": 256,
    }


def _calibration_thresholds() -> dict[str, Any]:
    """Permissive thresholds for calibration. Shadow mode means no
    mutations regardless; these values just drive *which* action the
    policy would have proposed. The trace records the EMAs themselves
    (utility, peak, sharpness, drift, contradiction), and those are what
    the analyzer percentile-anchors.
    """
    return {
        "replay_eviction_threshold": 0.001,
        "replay_eviction_useful_threshold": 0.0005,
        "replay_eviction_drift_threshold": 0.05,
        "replay_eviction_repr_drift_threshold": 0.05,
        "replay_eviction_quarantine_threshold": -0.001,
        "replay_eviction_distill_peak_threshold": 0.005,
        "replay_eviction_peak_preserve_utility_threshold": 0.005,
        "replay_eviction_peak_preserve_sharpness_threshold": 0.005,
        "replay_eviction_action_agreement_count": 1,
        "replay_eviction_min_age_steps": 32,
        "replay_eviction_min_score_count": 2,
    }


def build_smoke_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 30.0,
    seed: int = 1337,
) -> list[dict[str, Any]]:
    """Phase 0: short runtime smoke before calibration/headline spend.

    This is deliberately not a dry-run: it launches the real runner briefly
    to catch DDP, tokenizer/data, CRCT mailbox, replay-maintenance, prototype,
    and trace-writing breakage. Outputs are isolated under ``smoke/`` so a
    sanity check cannot contaminate calibration thresholds or headline results.
    """
    size_lock = _artifact_size_lock()
    fast_slow = _fast_slow_lock()
    crct = _crct_lock()
    pipeline = _replay_eviction_pipeline_lock()
    smoke_thresholds = _calibration_thresholds()
    arm_specs: list[tuple[str, dict[str, Any]]] = [
        ("smoke_fastslow_control", {}),
        (
            "smoke_crct_replay_active",
            {
                **crct,
                **pipeline,
                **smoke_thresholds,
                "replay_eviction_mode": "active",
                "replay_eviction_action_agreement_count": 1,
                "replay_eviction_trace_path": str(
                    DEFAULT_SMOKE_TRACE_DIR / f"arm_smoke_crct_replay_active_s{int(seed)}.ndjson"
                ),
                "crct_gradient_conflict_trace_path": str(
                    DEFAULT_SMOKE_TRACE_DIR / "crct_conflict.ndjson"
                ),
            },
        ),
    ]
    entries: list[dict[str, Any]] = []
    for arm_name, arm_overrides in arm_specs:
        arm = {
            "arm": arm_name,
            "exp26_mechanism": "arm_v1_smoke",
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
                phase="smoke",
                arm=arm_name,
                seed=int(seed),
            )
        )
    return entries


def build_calibration_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 180.0,
    seed: int = 1337,
) -> list[dict[str, Any]]:
    """Stage 1: single shadow-mode cell. Observes signal distributions.

    Runs the full ARM streaming pipeline (CRCT + maintenance) but with
    ``replay_eviction_mode='shadow'`` so no slot mutations fire. The
    per-decision trace captures EMAs at every shadow-policy decision; the
    analyzer reads those rows and percentile-anchors the headline
    thresholds.
    """
    size_lock = _artifact_size_lock()
    fast_slow = _fast_slow_lock()
    crct = _crct_lock()
    pipeline = _replay_eviction_pipeline_lock()
    permissive = _calibration_thresholds()
    arm_overrides: dict[str, Any] = {
        "arm": "calibration",
        "exp26_mechanism": "arm_v1_calibration",
        "artifact_impact": ARTIFACT_CHANGES_WEIGHTS_ONLY,
        **size_lock,
        **fast_slow,
        **crct,
        **pipeline,
        **permissive,
        "replay_eviction_mode": "shadow",
        "replay_eviction_trace_path": str(DEFAULT_CALIBRATION_TRACE),
    }
    base = _base_entry(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
    )
    base.update(arm_overrides)
    return [
        _named_entry(
            base=base,
            phase="calibration",
            arm="shadow",
            seed=int(seed),
        )
    ]


def load_manifest(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load a calibration manifest. Raises if missing or malformed."""
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(
            f"calibration manifest missing at {p}; "
            f"run stage 1 (build_calibration_matrix) and stage 2 "
            f"(calibrate.analyze) first"
        )
    try:
        manifest = json.loads(p.read_text())
    except Exception as exc:
        raise ValueError(f"manifest at {p} is malformed: {exc!r}") from exc
    required = {"thresholds_balanced"}
    missing = required - set(manifest.keys())
    if missing:
        raise ValueError(
            f"manifest at {p} missing required keys: {sorted(missing)}"
        )
    return manifest


def _learned_active_arm_overrides() -> dict[str, Any]:
    """arm_d: learned Full-A action-simplex commit authority.

    Threshold calibration is intentionally not folded into the active arm.
    The GPU3 oracle supplies physics confirmation, and the Full-A controller
    learns commit authority from that feedback.
    """
    return {
        "replay_eviction_mode": "active",
        "replay_eviction_commit_policy": "learned",
        "replay_eviction_action_agreement_count": 1,
    }


def build_arm_v1_matrix(
    *,
    speed_config: dict[str, Any],
    calibration_manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = DEFAULT_CONTROL_SEEDS,
    arms: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Stage 3: headline 4-arm × N-seed matrix.

    arm_a: locked fast/slow control (no CRCT, no maintenance).
    arm_b: CRCT only (no maintenance).
    arm_c: CRCT + maintenance shadow (telemetry, no mutation).
    arm_d: CRCT + maintenance active with learned Full-A action-simplex
        commit authority.

    ``calibration_manifest_path`` is accepted for CLI compatibility with the
    analyze/headline flow, but the headline learned arm does not consume
    threshold manifests. Calibration output is for post-hoc counterfactual
    analysis, not runtime commit authority.
    """
    _ = calibration_manifest_path
    size_lock = _artifact_size_lock()
    fast_slow = _fast_slow_lock()
    crct = _crct_lock()
    pipeline = _replay_eviction_pipeline_lock()
    shadow_overrides = {
        **crct,
        **pipeline,
        **_calibration_thresholds(),
        "replay_eviction_mode": "shadow",
    }
    arm_specs: list[tuple[str, dict[str, Any]]] = [
        ("arm_a_fastslow_control", {}),
        ("arm_b_crct_controller", crct),
        ("arm_c_crct_replay_shadow", shadow_overrides),
        (
            "arm_d_crct_replay_active_learned",
            {**crct, **pipeline, **_learned_active_arm_overrides()},
        ),
    ]
    if arms is not None:
        allowed = set(arms)
        unknown = allowed - set(ARM_V1_ARMS)
        if unknown:
            raise ValueError(
                f"unknown arm(s) {sorted(unknown)}; allowed: {ARM_V1_ARMS}"
            )
        arm_specs = [
            (name, overrides) for name, overrides in arm_specs if name in allowed
        ]
    entries: list[dict[str, Any]] = []
    for arm_name, arm_overrides in arm_specs:
        arm = {
            "arm": arm_name,
            "exp26_mechanism": "arm_v1",
            "artifact_impact": ARTIFACT_CHANGES_WEIGHTS_ONLY,
            **size_lock,
            **fast_slow,
            **arm_overrides,
        }
        for seed in seed_values:
            entry = _base_entry(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
            )
            entry.update(arm)
            named = _named_entry(
                base=entry,
                phase="phase3",
                arm=arm_name,
                seed=int(seed),
            )
            if named.get("replay_eviction_enabled"):
                named["replay_eviction_trace_path"] = str(
                    DEFAULT_TRACE_DIR
                    / f"arm_v1_{arm_name}_s{int(seed)}.ndjson"
                )
            entries.append(named)
    return entries
