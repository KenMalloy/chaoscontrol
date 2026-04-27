#!/usr/bin/env python3
"""Exp24 training-time bundle matrices and summaries."""

from __future__ import annotations

import copy
import os
import statistics
from collections.abc import Sequence
from typing import Any


DEFAULT_CONTROL_SEEDS = (1337, 2674, 4011)
ARTIFACT_CHANGES_WEIGHTS_ONLY = "artifact_changes_weights_only"
ARTIFACT_TRAINING_ONLY = "artifact_training_only"
PHASE0_FASTSLOW_DW_CANDIDATES = (
    (16, 16, 0.10),
    (16, 16, 0.25),
    (8, 8, 0.10),
)
PHASE0_CONFIRM_CONFIGS = (
    ("A", 32, 0.25, 16, 16, 0.10),
    ("B", 64, 0.25, 16, 16, 0.10),
)


def _base_entry(
    *,
    speed_config: dict[str, Any],
    world_size: int,
    budget_seconds: float,
) -> dict[str, Any]:
    entry = copy.deepcopy(speed_config)
    entry.update(
        {
            "mode": "exp24_training_time_bundle",
            "world_size": int(world_size),
            "budget_seconds": float(budget_seconds),
            "eval_batches": int(entry.get("eval_batches", 0)),
            "train_sampling_mode": str(entry.get("train_sampling_mode", "random")),
            "artifact_impact": str(
                entry.get("artifact_impact", ARTIFACT_CHANGES_WEIGHTS_ONLY)
            ),
            "submit_valid": bool(entry.get("submit_valid", True)),
            "fast_slow_enabled": bool(entry.get("fast_slow_enabled", False)),
            "fast_slow_interval": int(entry.get("fast_slow_interval", 0)),
            "fast_slow_alpha": float(entry.get("fast_slow_alpha", 0.0)),
            "fast_slow_eval_copy": str(entry.get("fast_slow_eval_copy", "fast")),
            "spectral_reg_lambda_dead": float(
                entry.get("spectral_reg_lambda_dead", 0.0)
            ),
            "spectral_reg_lambda_sticky": float(
                entry.get("spectral_reg_lambda_sticky", 0.0)
            ),
            "spectral_reg_min_a": float(entry.get("spectral_reg_min_a", 0.05)),
            "spectral_reg_max_a": float(entry.get("spectral_reg_max_a", 0.98)),
            "predictive_aux_weight": float(
                entry.get("predictive_aux_weight", 0.0)
            ),
            "predictive_aux_horizon": int(
                entry.get("predictive_aux_horizon", 0)
            ),
            "predictive_aux_dim": int(entry.get("predictive_aux_dim", 0)),
            "dreamworld_enabled": bool(entry.get("dreamworld_enabled", False)),
            "dreamworld_cache_interval": int(
                entry.get("dreamworld_cache_interval", 0)
            ),
            "dreamworld_interval": int(entry.get("dreamworld_interval", 0)),
            "dreamworld_weight": float(entry.get("dreamworld_weight", 0.0)),
            "dreamworld_prefix_tokens": int(
                entry.get("dreamworld_prefix_tokens", 128)
            ),
            "dreamworld_replay_tokens": int(
                entry.get("dreamworld_replay_tokens", 64)
            ),
            "dreamworld_replay_batch_size": int(
                entry.get("dreamworld_replay_batch_size", 0)
            ),
            "dreamworld_buffer_size": int(entry.get("dreamworld_buffer_size", 16)),
            "dreamworld_min_size": int(entry.get("dreamworld_min_size", 2)),
            "dreamworld_max_age_steps": int(
                entry.get("dreamworld_max_age_steps", 256)
            ),
            "event_sleep_enabled": bool(entry.get("event_sleep_enabled", False)),
            "event_sleep_loss_ratio": float(
                entry.get("event_sleep_loss_ratio", 1.10)
            ),
            "event_sleep_pressure_threshold": float(
                entry.get("event_sleep_pressure_threshold", 0.05)
            ),
            "event_sleep_ema_decay": float(
                entry.get("event_sleep_ema_decay", 0.99)
            ),
            "event_sleep_warmup_steps": int(
                entry.get("event_sleep_warmup_steps", 32)
            ),
            "event_sleep_min_interval": int(
                entry.get("event_sleep_min_interval", 8)
            ),
            "event_sleep_weight": float(entry.get("event_sleep_weight", 0.0)),
            "embed_freeze_steps": int(entry.get("embed_freeze_steps", 0)),
            "semantic_layer_index": int(entry.get("semantic_layer_index", 0)),
            "semantic_momentum_min": float(
                entry.get("semantic_momentum_min", 0.5)
            ),
            "semantic_overhead_gate": float(
                entry.get("semantic_overhead_gate", 0.08)
            ),
            "optimizer_param_grouping": str(
                entry.get("optimizer_param_grouping", "flat")
            ),
            "optimizer_dynamics_lr_mul": float(
                entry.get("optimizer_dynamics_lr_mul", 0.1)
            ),
        }
    )
    return entry


def _named_entry(
    *,
    base: dict[str, Any],
    phase: str | None,
    mechanism: str,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    entry = copy.deepcopy(base)
    if phase:
        name = f"exp24_{phase}_{arm}_s{int(seed)}"
    else:
        name = f"exp24_{arm}_s{int(seed)}"
    entry.update(
        {
            "name": name,
            "seed": int(seed),
            "exp24_phase": phase if phase is not None else "first_wave",
            "exp24_mechanism": mechanism,
        }
    )
    return entry


def build_ring0_control_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 8,
    budget_seconds: float = 600.0,
    seeds: Sequence[int] = DEFAULT_CONTROL_SEEDS,
) -> list[dict[str, Any]]:
    base = _base_entry(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
    )
    base["train_sampling_mode"] = "random"
    return [
        _named_entry(
            base=base,
            phase="ring0",
            mechanism="control",
            arm="control",
            seed=int(seed),
        )
        for seed in seeds
    ]


def build_phase_a_sampling_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 8,
    budget_seconds: float = 600.0,
    seeds: Sequence[int] = (1337,),
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for policy in ("random", "sequential_epoch", "shuffled_epoch"):
        base = _base_entry(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
        )
        base["train_sampling_mode"] = policy
        for seed in seeds:
            entries.append(
                _named_entry(
                    base=base,
                    phase="phaseA",
                    mechanism="sampling_policy",
                    arm=policy,
                    seed=int(seed),
                )
            )
    return entries


def build_semantic_overhead_gate_matrix(
    *,
    speed_config: dict[str, Any],
    seed: int = 1337,
    world_size: int = 1,
    budget_seconds: float = 90.0,
) -> list[dict[str, Any]]:
    base = _base_entry(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
    )
    base["artifact_impact"] = ARTIFACT_CHANGES_WEIGHTS_ONLY
    entries: list[dict[str, Any]] = []
    for optimizer_name in ("muon", "semantic"):
        opt_base = copy.deepcopy(base)
        opt_base["optimizer"] = optimizer_name
        entries.append(
            _named_entry(
                base=opt_base,
                phase="smoke",
                mechanism="semantic_optimizer_gate",
                arm=f"semantic_gate_{optimizer_name}",
                seed=seed,
            )
        )

    return entries


def build_scopt_overhead_gate_matrix(
    *,
    speed_config: dict[str, Any],
    seed: int = 1337,
    world_size: int = 1,
    budget_seconds: float = 180.0,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """1×H100 Muon-vs-ScOpt smoke at a VRAM-safe batch size.

    ScOpt's split-step path uses unfused ``F.cross_entropy`` with
    ``retain_graph=True`` and materializes the full fp32 ``(B, T, V)``
    logits tensor. At submission bs=1024 / V=16384 that forward + the
    subsequent ``logits.grad`` overflow H100 80 GB. bs=512 fits (~60 GiB
    peak) and, at ~11 steps/s, runs well past ScOpt's default
    ``warmup_steps=200`` inside ``budget_seconds=180``, so
    ``scopt_probes.evaluate_tier0_gates`` has a non-warmup trace to read.

    Both entries use ``optimizer_param_grouping='ssm_three_group'`` — the
    comparison measures ScOpt's mechanism on top of an S4/S5/HOPE-aware
    baseline rather than the legacy uniform-WD path that leaves ``log_a``
    fighting ``wd=0.01`` while ScOpt's recurrence scarcity tries to move
    it.
    """
    base = _base_entry(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
    )
    base["artifact_impact"] = ARTIFACT_CHANGES_WEIGHTS_ONLY
    base["batch_size"] = int(batch_size)
    base["optimizer_param_grouping"] = "ssm_three_group"
    base["optimizer_dynamics_lr_mul"] = 0.1

    entries: list[dict[str, Any]] = []
    for optimizer_name in ("muon", "scopt"):
        opt_base = copy.deepcopy(base)
        opt_base["optimizer"] = optimizer_name
        if optimizer_name == "scopt":
            opt_base["scopt_warmup_steps"] = 200
            opt_base["scopt_split_interval"] = 4
            opt_base["scopt_trace_interval_steps"] = 64
            opt_base["scopt_rare_ema_decay"] = 0.9
            opt_base["scopt_rare_orthogonal_weight"] = 1.0
            # Macro Gerber on rare-orthogonal contribution: caps
            # orth_norm at c * common_norm per parameter, preventing the
            # rare-gradient-dominates-common-gradient feedback loop that
            # drove the 2026-04-24 long-smoke to final_loss 158.
            opt_base["scopt_rare_macro_c"] = 0.5
            # Gerber upper clamp on pressure: winsorized-std threshold
            # to prevent exploded-CE tokens from poisoning the whole
            # rare-gradient accumulation.
            opt_base["scopt_pressure_upper_c"] = 3.0
            opt_base["scopt_pressure_upper_floor"] = 1.0
            opt_base["scopt_row_scarcity_power"] = 0.5
            opt_base["scopt_tau_std_scale"] = 0.5
            opt_base["scopt_layer_index"] = 0
            opt_base["scopt_baseline_buckets"] = 16
            opt_base["scopt_baseline_decay"] = 0.99
        entries.append(
            _named_entry(
                base=opt_base,
                phase="smoke",
                mechanism="scopt_overhead_gate",
                arm=f"scopt_gate_{optimizer_name}",
                seed=seed,
            )
        )

    return entries


def build_scopt_calibration_sweep_matrix(
    *,
    speed_config: dict[str, Any],
    seed: int = 1337,
    world_size: int = 1,
    budget_seconds: float = 1200.0,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    """ScOpt calibration probe: three single-knob variants vs the step-1 base.

    Step 1 (results_scopt_step1_20260424T0500Z) showed
    rare_macro_cap_fired.median=1.0 and pressure_fraction_positive=0.51,
    meaning the macro-Gerber cap was clipping every step and more than
    half of all tokens still looked "rare" under the CE baseline. Each
    cell here perturbs one knob relative to scopt_overhead_gate's ScOpt
    entry so the diagnostic map is isolated per-lever.
    """
    base = _base_entry(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
    )
    base["artifact_impact"] = ARTIFACT_CHANGES_WEIGHTS_ONLY
    base["batch_size"] = int(batch_size)
    base["optimizer_param_grouping"] = "ssm_three_group"
    base["optimizer_dynamics_lr_mul"] = 0.1
    base["optimizer"] = "scopt"
    base["scopt_warmup_steps"] = 200
    base["scopt_split_interval"] = 4
    base["scopt_trace_interval_steps"] = 64
    base["scopt_rare_ema_decay"] = 0.9
    base["scopt_rare_orthogonal_weight"] = 1.0
    base["scopt_rare_macro_c"] = 0.5
    base["scopt_pressure_upper_c"] = 3.0
    base["scopt_pressure_upper_floor"] = 1.0
    base["scopt_row_scarcity_power"] = 0.5
    base["scopt_tau_std_scale"] = 0.5
    base["scopt_layer_index"] = 0
    base["scopt_baseline_buckets"] = 16
    base["scopt_baseline_decay"] = 0.99

    variants: list[tuple[str, dict[str, Any]]] = [
        ("baseline_d95", {"scopt_baseline_decay": 0.95}),
        ("pressure_c15", {"scopt_pressure_upper_c": 1.5}),
        ("rare_ema_095", {"scopt_rare_ema_decay": 0.95}),
    ]

    entries: list[dict[str, Any]] = []
    for arm_suffix, overrides in variants:
        entry_base = copy.deepcopy(base)
        entry_base.update(overrides)
        entries.append(
            _named_entry(
                base=entry_base,
                phase="smoke",
                mechanism="scopt_calibration_sweep",
                arm=f"scopt_cal_{arm_suffix}",
                seed=seed,
            )
        )

    return entries


def build_first_wave_mechanism_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 8,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = DEFAULT_CONTROL_SEEDS,
) -> list[dict[str, Any]]:
    arms = [
        {
            "name_arm": "fastslow_i32_a050",
            "exp24_mechanism": "fast_slow",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            "fast_slow_enabled": True,
            "fast_slow_interval": 32,
            "fast_slow_alpha": 0.50,
            "fast_slow_eval_copy": "slow",
        },
        {
            "name_arm": "spectral_dead1e-04_sticky1e-04",
            "exp24_mechanism": "spectral",
            "artifact_impact": ARTIFACT_CHANGES_WEIGHTS_ONLY,
            "spectral_reg_lambda_dead": 1e-4,
            "spectral_reg_lambda_sticky": 1e-4,
        },
        {
            "name_arm": "predictive_h4_w010",
            "exp24_mechanism": "predictive_aux",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            "predictive_aux_weight": 0.10,
            "predictive_aux_horizon": 4,
        },
        {
            "name_arm": "dreamworld_c4_i4_w025",
            "exp24_mechanism": "dreamworld",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            "dreamworld_enabled": True,
            "dreamworld_cache_interval": 4,
            "dreamworld_interval": 4,
            "dreamworld_weight": 0.25,
            "dreamworld_prefix_tokens": 128,
            "dreamworld_replay_tokens": 64,
            "dreamworld_buffer_size": 16,
            "dreamworld_min_size": 2,
            "dreamworld_max_age_steps": 256,
        },
        {
            "name_arm": "fastslow_i32_a050_dreamworld_c8_i8_w025_sub128",
            "exp24_mechanism": "fast_slow_dreamworld",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            "fast_slow_enabled": True,
            "fast_slow_interval": 32,
            "fast_slow_alpha": 0.50,
            "fast_slow_eval_copy": "slow",
            "dreamworld_enabled": True,
            "dreamworld_cache_interval": 8,
            "dreamworld_interval": 8,
            "dreamworld_weight": 0.25,
            "dreamworld_prefix_tokens": 128,
            "dreamworld_replay_tokens": 64,
            "dreamworld_replay_batch_size": 128,
            "dreamworld_buffer_size": 16,
            "dreamworld_min_size": 2,
            "dreamworld_max_age_steps": 256,
        },
        {
            "name_arm": "fastslow_i32_a050_dreamworld_eventsleep_r110_p005_sub128",
            "exp24_mechanism": "fast_slow_dreamworld_event_sleep",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            "fast_slow_enabled": True,
            "fast_slow_interval": 32,
            "fast_slow_alpha": 0.50,
            "fast_slow_eval_copy": "slow",
            "dreamworld_enabled": True,
            "dreamworld_cache_interval": 8,
            "dreamworld_interval": 8,
            "dreamworld_weight": 0.25,
            "dreamworld_prefix_tokens": 128,
            "dreamworld_replay_tokens": 64,
            "dreamworld_replay_batch_size": 128,
            "dreamworld_buffer_size": 16,
            "dreamworld_min_size": 2,
            "dreamworld_max_age_steps": 256,
            "event_sleep_enabled": True,
            "event_sleep_loss_ratio": 1.10,
            "event_sleep_pressure_threshold": 0.05,
            "event_sleep_ema_decay": 0.99,
            "event_sleep_warmup_steps": 32,
            "event_sleep_min_interval": 8,
        },
    ]

    entries: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seed_values:
            entry = _base_entry(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
            )
            entry.update(arm)
            name_arm = str(entry.pop("name_arm"))
            entries.append(
                _named_entry(
                    base=entry,
                    phase=None,
                    mechanism=str(entry["exp24_mechanism"]),
                    arm=name_arm,
                    seed=int(seed),
                )
            )
    return entries


def build_fastslow_dreamworld_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 8,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = DEFAULT_CONTROL_SEEDS,
) -> list[dict[str, Any]]:
    arm = {
        "name_arm": "fastslow_i32_a050_dreamworld_c8_i8_w025_sub128",
        "exp24_mechanism": "fast_slow_dreamworld",
        "artifact_impact": ARTIFACT_TRAINING_ONLY,
        "fast_slow_enabled": True,
        "fast_slow_interval": 32,
        "fast_slow_alpha": 0.50,
        "fast_slow_eval_copy": "slow",
        "dreamworld_enabled": True,
        "dreamworld_cache_interval": 8,
        "dreamworld_interval": 8,
        "dreamworld_weight": 0.25,
        "dreamworld_prefix_tokens": 128,
        "dreamworld_replay_tokens": 64,
        "dreamworld_replay_batch_size": 128,
        "dreamworld_buffer_size": 16,
        "dreamworld_min_size": 2,
        "dreamworld_max_age_steps": 256,
    }
    entries: list[dict[str, Any]] = []
    for seed in seed_values:
        entry = _base_entry(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
        )
        entry.update(arm)
        name_arm = str(entry.pop("name_arm"))
        entries.append(
            _named_entry(
                base=entry,
                phase=None,
                mechanism=str(entry["exp24_mechanism"]),
                arm=name_arm,
                seed=int(seed),
            )
        )
    return entries


def build_phase0_dreamworld_sweep(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = (1337,),
) -> list[dict[str, Any]]:
    """Phase 0 rung 1: sweep DW interval x weight with FS pinned at anchor."""
    intervals = [4, 8, 16]
    weights = [0.10, 0.25, 0.50]
    entries: list[dict[str, Any]] = []
    for interval in intervals:
        for weight in weights:
            arm = {
                "name_arm": f"fs_i32a050_dw_c{interval}i{interval}_w{int(weight * 100):03d}",
                "exp24_mechanism": "fast_slow_dreamworld",
                "artifact_impact": ARTIFACT_TRAINING_ONLY,
                "fast_slow_enabled": True,
                "fast_slow_interval": 32,
                "fast_slow_alpha": 0.50,
                "fast_slow_eval_copy": "slow",
                "dreamworld_enabled": True,
                "dreamworld_cache_interval": interval,
                "dreamworld_interval": interval,
                "dreamworld_weight": weight,
                "dreamworld_prefix_tokens": 128,
                "dreamworld_replay_tokens": 64,
                "dreamworld_replay_batch_size": 128,
                "dreamworld_buffer_size": 16,
                "dreamworld_min_size": 2,
                "dreamworld_max_age_steps": 256,
            }
            for seed in seed_values:
                entry = _base_entry(
                    speed_config=speed_config,
                    world_size=world_size,
                    budget_seconds=budget_seconds,
                )
                entry.update(arm)
                name_arm = str(entry.pop("name_arm"))
                entries.append(
                    _named_entry(
                        base=entry,
                        phase="phase0",
                        mechanism=str(entry["exp24_mechanism"]),
                        arm=name_arm,
                        seed=int(seed),
                    )
                )
    return entries


def build_phase0_fastslow_sweep(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = (1337,),
) -> list[dict[str, Any]]:
    """Phase 0 rung 2: sweep FS interval x alpha around Task 5 DW candidates.

    DW settings below must match PHASE0_DW_WINNER.md. If changed, update that
    doc and re-run Task 7 from scratch.
    """
    fs_intervals = [16, 32, 64]
    fs_alphas = [0.25, 0.50]
    entries: list[dict[str, Any]] = []
    for dw_cache_interval, dw_interval, dw_weight in PHASE0_FASTSLOW_DW_CANDIDATES:
        for fs_interval in fs_intervals:
            for fs_alpha in fs_alphas:
                arm = {
                    "name_arm": (
                        f"fs_i{fs_interval}a{int(fs_alpha * 100):03d}_"
                        f"dw_c{dw_cache_interval}i{dw_interval}_"
                        f"w{int(dw_weight * 100):03d}"
                    ),
                    "exp24_mechanism": "fast_slow_dreamworld",
                    "artifact_impact": ARTIFACT_TRAINING_ONLY,
                    "fast_slow_enabled": True,
                    "fast_slow_interval": fs_interval,
                    "fast_slow_alpha": fs_alpha,
                    "fast_slow_eval_copy": "slow",
                    "dreamworld_enabled": True,
                    "dreamworld_cache_interval": dw_cache_interval,
                    "dreamworld_interval": dw_interval,
                    "dreamworld_weight": dw_weight,
                    "dreamworld_prefix_tokens": 128,
                    "dreamworld_replay_tokens": 64,
                    "dreamworld_replay_batch_size": 128,
                    "dreamworld_buffer_size": 16,
                    "dreamworld_min_size": 2,
                    "dreamworld_max_age_steps": 256,
                }
                for seed in seed_values:
                    entry = _base_entry(
                        speed_config=speed_config,
                        world_size=world_size,
                        budget_seconds=budget_seconds,
                    )
                    entry.update(arm)
                    name_arm = str(entry.pop("name_arm"))
                    entries.append(
                        _named_entry(
                            base=entry,
                            phase="phase0",
                            mechanism=str(entry["exp24_mechanism"]),
                            arm=name_arm,
                            seed=int(seed),
                        )
                    )
    return entries


def build_phase0_confirm(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = DEFAULT_CONTROL_SEEDS,
) -> list[dict[str, Any]]:
    """Phase 0 rung 3: confirm top-2 configs x 3 seeds with full validation."""
    entries: list[dict[str, Any]] = []
    for label, fs_interval, fs_alpha, dw_cache_interval, dw_interval, dw_weight in (
        PHASE0_CONFIRM_CONFIGS
    ):
        arm = {
            "name_arm": (
                f"confirm_{label}_fs_i{fs_interval}a{int(fs_alpha * 100):03d}_"
                f"dw_c{dw_cache_interval}i{dw_interval}_w{int(dw_weight * 100):03d}"
            ),
            "exp24_mechanism": "fast_slow_dreamworld",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            "fast_slow_enabled": True,
            "fast_slow_interval": fs_interval,
            "fast_slow_alpha": fs_alpha,
            "fast_slow_eval_copy": "slow",
            "dreamworld_enabled": True,
            "dreamworld_cache_interval": dw_cache_interval,
            "dreamworld_interval": dw_interval,
            "dreamworld_weight": dw_weight,
            "dreamworld_prefix_tokens": 128,
            "dreamworld_replay_tokens": 64,
            "dreamworld_replay_batch_size": 128,
            "dreamworld_buffer_size": 16,
            "dreamworld_min_size": 2,
            "dreamworld_max_age_steps": 256,
        }
        for seed in seed_values:
            entry = _base_entry(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
            )
            entry.update(arm)
            name_arm = str(entry.pop("name_arm"))
            entries.append(
                _named_entry(
                    base=entry,
                    phase="phase0",
                    mechanism=str(entry["exp24_mechanism"]),
                    arm=name_arm,
                    seed=int(seed),
                )
            )
    return entries


def build_phase0_fastslow_only_control(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = DEFAULT_CONTROL_SEEDS,
) -> list[dict[str, Any]]:
    """Matched 4x fast/slow-only control for the locked Phase 0 stack."""
    arm = {
        "name_arm": "control_fastslow_only_i64a025",
        "exp24_mechanism": "fast_slow",
        "artifact_impact": ARTIFACT_TRAINING_ONLY,
        "fast_slow_enabled": True,
        "fast_slow_interval": 64,
        "fast_slow_alpha": 0.25,
        "fast_slow_eval_copy": "slow",
        "dreamworld_enabled": False,
        "dreamworld_cache_interval": 0,
        "dreamworld_interval": 0,
        "dreamworld_weight": 0.0,
        "dreamworld_replay_batch_size": 0,
    }
    entries: list[dict[str, Any]] = []
    for seed in seed_values:
        entry = _base_entry(
            speed_config=speed_config,
            world_size=world_size,
            budget_seconds=budget_seconds,
        )
        entry.update(arm)
        name_arm = str(entry.pop("name_arm"))
        entries.append(
            _named_entry(
                base=entry,
                phase="phase0",
                mechanism=str(entry["exp24_mechanism"]),
                arm=name_arm,
                seed=int(seed),
            )
        )
    return entries


EPISODIC_DW_CURATION_V1_ARMS: tuple[str, ...] = (
    "arm_a_uncurated",
    "arm_b_cosine_utility",
    "arm_bp_pressure_only",
    "arm_c_no_dw",
)
# Decision 0.5 specifies escalation runs A/B/B' only — Arm C is the
# topology-baseline reference and doesn't need additional seeds when the
# σ rule fires on the curated arms.
EPISODIC_DW_CURATION_V1_ESCALATION_ARMS: tuple[str, ...] = (
    "arm_a_uncurated",
    "arm_b_cosine_utility",
    "arm_bp_pressure_only",
)
EPISODIC_DW_CURATION_V1_ESCALATION_SEEDS: tuple[int, ...] = (5012, 7331, 9183)


def build_episodic_dw_curation_v1_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = DEFAULT_CONTROL_SEEDS,
    arms: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """[DEPRECATED] Training-only Phase 3 falsifier matrix.

    Superseded by ``build_episodic_ttt_v1_matrix`` (the TTT-shaped successor
    after the architecture pivot — the cache must be live at eval time, not
    only during training). Kept available so deprecated runs remain
    reproducible; new work should target the TTT matrix.

    Phase 3 falsifier matrix for ``episodic_dw_curation_v1``.

    Four arms x N seeds. **All four arms are topologically identical**: 3+1
    rank layout (``world_size=4``), episodic rank present, same all-reduce
    path, same Dreamworld replay knobs (cache_interval, interval, replay
    batch size, prefix/replay tokens, buffer/min/max-age), same fast/slow
    recipe (anchored to ``phase0_fastslow_only_control``), same wall budget.
    The ONLY differences allowed between arms are the replay-candidate-
    selection mechanism (``episodic_enabled`` + ``controller_query_mode``)
    and Arm C's ``dreamworld_weight=0.0`` zeroing of the replay signal.

    Per Decision 0.5 (memory-aware-optimizer-plan): if ``sigma`` of the
    Arm B rare-bucket delta across the default 3 seeds exceeds 0.008 bpb,
    re-invoke this builder with::

        build_episodic_dw_curation_v1_matrix(
            speed_config=...,
            seed_values=EPISODIC_DW_CURATION_V1_ESCALATION_SEEDS,
            arms=EPISODIC_DW_CURATION_V1_ESCALATION_ARMS,
        )

    to escalate. The ``arms`` filter restricts to A/B/B' (skipping C, the
    topology-baseline reference) per the spec; downstream analysis pools
    the two runs across the seed set.

    Arms:
      - ``arm_a_uncurated``: ``episodic_enabled=False``; replay reads the
        existing online buffer in ``dreamworld.py``. No cache writes,
        queries, or controller-driven selection.
      - ``arm_b_cosine_utility``: ``episodic_enabled=True`` +
        ``controller_query_enabled=True`` with
        ``controller_query_mode="cosine_utility_weighted"`` (Decision 0.2).
      - ``arm_bp_pressure_only``: ``episodic_enabled=True`` +
        ``controller_query_enabled=True`` with
        ``controller_query_mode="pressure_only"`` — the mechanism-
        specificity lesion. Cache fills the same way as Arm B; only the
        retrieval policy changes.
      - ``arm_c_no_dw``: ``episodic_enabled=True`` with
        ``dreamworld_weight=0.0``. Replay backward fires (so the rank
        layout pays the same overhead) but contributes zero gradient.

    ``controller_query_mode`` is a forward-looking config knob the future
    Phase 2 controller will read; the runner ignores unknown config keys
    today. ``controller_query_enabled`` is the gate the runner already
    wires (Pass C, default False); B and B' set it True so the queue
    populates with retrieval candidates that the controller drains. Arms
    A and C omit both knobs (Arm A doesn't query the cache; Arm C's
    replay grads are zeroed before they land in any param.grad).
    """
    # Locked Dreamworld replay knobs — match phase0_dw_sweep so the
    # uncurated control reuses a known, replayed-then-failed combo.
    dw_replay_lock = {
        "dreamworld_enabled": True,
        "dreamworld_cache_interval": 16,
        "dreamworld_interval": 16,
        "dreamworld_prefix_tokens": 128,
        "dreamworld_replay_tokens": 64,
        "dreamworld_replay_batch_size": 128,
        "dreamworld_buffer_size": 16,
        "dreamworld_min_size": 2,
        "dreamworld_max_age_steps": 256,
    }
    # Locked fast/slow recipe — match phase0_fastslow_only_control.
    fast_slow_lock = {
        "fast_slow_enabled": True,
        "fast_slow_interval": 64,
        "fast_slow_alpha": 0.25,
        "fast_slow_eval_copy": "slow",
    }
    arm_specs: list[tuple[str, dict[str, Any]]] = [
        (
            "arm_a_uncurated",
            {
                "episodic_enabled": False,
                "dreamworld_weight": 0.10,
            },
        ),
        (
            "arm_b_cosine_utility",
            {
                "episodic_enabled": True,
                "controller_query_enabled": True,
                "dreamworld_weight": 0.10,
                "controller_query_mode": "cosine_utility_weighted",
            },
        ),
        (
            "arm_bp_pressure_only",
            {
                "episodic_enabled": True,
                "controller_query_enabled": True,
                "dreamworld_weight": 0.10,
                "controller_query_mode": "pressure_only",
            },
        ),
        (
            "arm_c_no_dw",
            {
                "episodic_enabled": True,
                "dreamworld_weight": 0.0,
            },
        ),
    ]
    if arms is not None:
        allowed = set(arms)
        unknown = allowed - set(EPISODIC_DW_CURATION_V1_ARMS)
        if unknown:
            raise ValueError(
                f"unknown arm(s) {sorted(unknown)}; "
                f"allowed: {EPISODIC_DW_CURATION_V1_ARMS}"
            )
        arm_specs = [(n, o) for n, o in arm_specs if n in allowed]
    entries: list[dict[str, Any]] = []
    for arm_name, arm_overrides in arm_specs:
        arm = {
            "exp24_mechanism": "episodic_dw_curation_v1",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            **fast_slow_lock,
            **dw_replay_lock,
            **arm_overrides,
        }
        for seed in seed_values:
            entry = _base_entry(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
            )
            entry.update(arm)
            entries.append(
                _named_entry(
                    base=entry,
                    phase="phase3",
                    mechanism="episodic_dw_curation_v1",
                    arm=f"episodic_dw_curation_v1_{arm_name}",
                    seed=int(seed),
                )
            )
    return entries


EPISODIC_TTT_V1_ARMS: tuple[str, ...] = (
    "arm_a_no_cache_no_ttt",
    "arm_b_cache_train_ttt_with_cache",
    "arm_c_cache_train_no_ttt",
    "arm_d_no_cache_train_ttt_only",
)


EPISODIC_CONTROLLER_V1_ARMS: tuple[str, ...] = (
    "arm_a_control",
    "arm_b_heuristic",
    "arm_c_simplex_frozen",
    "arm_d_simplex_online",
    "arm_e_simplex_warm_online",
    "arm_f_simplex_sharp_online",
)
# Initial-temperature override for arm_f. The CSWG-loaded simplex policy
# starts near-uniform (max p ≈ 0.067 vs uniform 0.0625 per the
# 2026-04-27 v2 trace inspection); REINFORCE on near-uniform behavior
# can't bootstrap because the gradient direction cancels across events.
# Sharpening to T=0.2 multiplies the effective logit magnitude by 5
# without re-pretraining. Validated by scripts/simplex_bootstrap_microbench.py
# (T=1.0 produces zero policy drift; T=0.2 produces 2x-uniform p[favored]
# in 19 SGD steps).
ARM_F_INITIAL_TEMPERATURE: float = 0.2
EPISODIC_CONTROLLER_V1_WEIGHTS_PATH = (
    "TO_BE_FILLED/episodic_controller_v1_weights.pt"
)
EPISODIC_CONTROLLER_V1_WEIGHTS_PATH_ENV = "EPISODIC_CONTROLLER_V1_WEIGHTS_PATH"

EPISODIC_CONTROLLER_V1_TRACE_DIR = (
    "experiments/24_training_time_bundle/results/traces"
)
EPISODIC_CONTROLLER_V1_TRACE_DIR_ENV = "EPISODIC_CONTROLLER_V1_TRACE_DIR"


def _resolve_episodic_controller_v1_weights_path() -> str:
    # Pod runbook substitutes the real artifact via the env var so this
    # file stays canonical. Default placeholder is a sentinel; the runner's
    # _build_controller_runtime_from_config rejects it via FileNotFoundError
    # at attach time, before any cell starts training.
    return os.environ.get(
        EPISODIC_CONTROLLER_V1_WEIGHTS_PATH_ENV,
        EPISODIC_CONTROLLER_V1_WEIGHTS_PATH,
    )


def _resolve_episodic_controller_v1_trace_dir() -> str:
    return os.environ.get(
        EPISODIC_CONTROLLER_V1_TRACE_DIR_ENV,
        EPISODIC_CONTROLLER_V1_TRACE_DIR,
    )


def build_episodic_controller_v1_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = DEFAULT_CONTROL_SEEDS,
) -> list[dict[str, Any]]:
    """Phase 3 CPU SSM simplex controller V1 falsifier matrix.

    Five arms x N seeds:
      - ``arm_a_control``: no episodic cache, no controller.
      - ``arm_b_heuristic``: heuristic controller (utility argmax over
        cache-side top-K). The simplex baseline this falsifier needs to
        beat — same candidate-generator semantics as the trained arms,
        but no learned policy on top.
      - ``arm_c_simplex_frozen``: trained simplex controller, fresh
        cache, TRAIN-side online controller training disabled (weights
        stay at the BC-pretrain init for the full 600s window).
      - ``arm_d_simplex_online``: trained simplex controller, fresh
        cache, TRAIN-side online REINFORCE enabled (SGD + EMA on the
        simplex policy parameters W_vp, b_vp, W_lh, b_lh, W_sb, alpha).
      - ``arm_e_simplex_warm_online``: trained simplex controller,
        checkpoint-warm cache, TRAIN-side online REINFORCE enabled. The
        warm cache means the candidate simplex starts richer — vertex
        features (utility, age, cosine) carry more signal at step 0
        than they would for a fresh-cache run.

    Three pairwise contrasts the matrix is designed to expose:
      - heuristic vs trained: arm_b vs arm_c (frozen-trained vs
        heuristic, same retrieval semantics, controller policy is
        the only difference).
      - frozen vs online: arm_c vs arm_d (trained controller, online
        SGD is the only difference).
      - cold vs warm cache: arm_d vs arm_e (online controller held
        constant, cache init is the only difference).

    ``episodic_controller_weights_path`` is the path to the simplex
    policy CSWG v2 file (S4's pretrain output). The default placeholder
    ``TO_BE_FILLED/...`` is sentinel; the runner's
    _build_controller_runtime_from_config rejects it via FileNotFoundError
    at attach time. Pod runbook substitutes via the
    ``EPISODIC_CONTROLLER_V1_WEIGHTS_PATH`` env var.

    Remaining caveat:

    ``eval_episodic_cache_mode`` (cold/warm) is loaded by
       run_exp20_full_val_score.py but DOES NOT affect scored CE — the
       optimized fast-score loop bypasses the cache-aware
       LegalityController path. arm_d vs arm_e currently differs only
       in the train-side checkpoint payload (warm = cache serialized in
       ckpt); downstream eval CE is the same. Cache-aware eval lives in
       run_exp20_eval.py; wiring it into the fast scorer is a separate
       follow-up. Until that lands, treat the cold/warm contrast as a
       train-side simplex-quality check, not eval-time TTT.
    """
    fast_slow_lock = {
        "fast_slow_enabled": True,
        "fast_slow_interval": 64,
        "fast_slow_alpha": 0.25,
        "fast_slow_eval_copy": "slow",
        "dreamworld_enabled": False,
        "dreamworld_cache_interval": 0,
        "dreamworld_interval": 0,
        "dreamworld_weight": 0.0,
        "dreamworld_replay_batch_size": 0,
    }
    eval_cache_schema_lock = {
        "eval_episodic_cache_capacity": 4096,
        "eval_episodic_span_length": 4,
        "eval_episodic_key_rep_dim": -1,
        "eval_episodic_grace_steps": 1000,
        "eval_episodic_fingerprint_window": 8,
        "eval_episodic_cache_reset_per_doc": False,
    }
    no_eval_cache = {
        **eval_cache_schema_lock,
        "eval_episodic_cache_enabled": False,
        "eval_steps_per_chunk": 0,
        "eval_adapt_set": "none",
        "eval_episodic_cache_mode": "none",
    }
    cold_eval_cache = {
        **eval_cache_schema_lock,
        "eval_episodic_cache_enabled": True,
        "eval_steps_per_chunk": 0,
        "eval_adapt_set": "none",
        "eval_episodic_cache_mode": "cold",
    }
    warm_eval_cache = {
        **cold_eval_cache,
        "eval_episodic_cache_mode": "warm",
        "eval_episodic_cache_source": "checkpoint",
    }
    heuristic_controller = {
        "episodic_controller_runtime": "heuristic",
        "controller_train_online": False,
    }
    simplex_controller_frozen = {
        "episodic_controller_runtime": "simplex_v1",
        "episodic_controller_weights_path": _resolve_episodic_controller_v1_weights_path(),
        # The simplex thesis is a policy over all 16 candidates, not a
        # hard rerank. Sample from p so replay credit can reinforce any
        # vertex that pays off.
        "episodic_controller_selection_mode": "sample",
        "controller_train_online": False,
        # Without the post-step CE pair the simplex policy's reward
        # signal is NaN (B5 gates the second forward on this flag).
        # Always-on for trained arms; the heuristic arm doesn't need it.
        "episodic_compute_replay_ce_pair": True,
    }
    simplex_controller_online = {
        **simplex_controller_frozen,
        "controller_train_online": True,
        # Anti-collapse regularizer for BC-saturated policies: small
        # enough for CE-delta REINFORCE to dominate once signal appears,
        # but large enough to keep the 16-way simplex exploratory.
        "episodic_controller_entropy_beta": 0.05,
    }
    arm_specs: list[tuple[str, dict[str, Any]]] = [
        (
            "arm_a_control",
            {
                "episodic_enabled": False,
                "controller_query_enabled": False,
                "episodic_event_log_enabled": False,
                **heuristic_controller,
                **no_eval_cache,
            },
        ),
        (
            "arm_b_heuristic",
            {
                "episodic_enabled": True,
                "controller_query_enabled": True,
                "episodic_event_log_enabled": False,
                "episodic_controller_score_mode": "cosine_utility_weighted",
                **heuristic_controller,
                **cold_eval_cache,
            },
        ),
        (
            "arm_c_simplex_frozen",
            {
                "episodic_enabled": True,
                "controller_query_enabled": True,
                "episodic_event_log_enabled": True,
                **simplex_controller_frozen,
                **cold_eval_cache,
            },
        ),
        (
            "arm_d_simplex_online",
            {
                "episodic_enabled": True,
                "controller_query_enabled": True,
                "episodic_event_log_enabled": True,
                **simplex_controller_online,
                **cold_eval_cache,
            },
        ),
        (
            "arm_e_simplex_warm_online",
            {
                "episodic_enabled": True,
                "controller_query_enabled": True,
                "episodic_event_log_enabled": True,
                **simplex_controller_online,
                **warm_eval_cache,
            },
        ),
        # arm_f isolates the initial-temperature override from the cold-vs-warm
        # cache contrast: same controller as arm_d (simplex_v1, online,
        # cold cache) but with episodic_controller_initial_temperature=0.2
        # to break the bootstrap pathology. arm_d vs arm_f directly tests
        # whether a non-uniform initial sampling policy unblocks REINFORCE.
        (
            "arm_f_simplex_sharp_online",
            {
                "episodic_enabled": True,
                "controller_query_enabled": True,
                "episodic_event_log_enabled": True,
                **simplex_controller_online,
                "episodic_controller_initial_temperature": ARM_F_INITIAL_TEMPERATURE,
                **cold_eval_cache,
            },
        ),
    ]
    entries: list[dict[str, Any]] = []
    trace_dir = _resolve_episodic_controller_v1_trace_dir()
    for arm_name, arm_overrides in arm_specs:
        arm = {
            "arm": arm_name,
            "exp24_mechanism": "episodic_controller_v1",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            **fast_slow_lock,
            **arm_overrides,
        }
        for seed in seed_values:
            entry = _base_entry(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
            )
            entry.update(arm)
            # The runner rejects episodic_enabled=True with
            # train_sampling_mode='sequential_epoch' because the
            # episodic-shard skip-main flow drops 1/N of the dataset
            # per epoch. Per the runner's own ValueError remediation,
            # use 'random' for every cell that turns the cache on.
            if arm.get("episodic_enabled"):
                entry["train_sampling_mode"] = "random"
            # Per-cell NDJSON trace for simplex arms. Operationalizes
            # docs/plans/2026-04-26-learned-controller-action-space.md:
            # "A stop that is not logged is a hidden experimental
            # confound." The runner mkdirs the parent before launch.
            if "simplex" in arm_name:
                entry["episodic_controller_simplex_trace_path"] = (
                    f"{trace_dir}/episodic_controller_v1_{arm_name}_s{int(seed)}.ndjson"
                )
            entries.append(
                _named_entry(
                    base=entry,
                    phase="phase3",
                    mechanism="episodic_controller_v1",
                    arm=f"episodic_controller_v1_{arm_name}",
                    seed=int(seed),
                )
            )
    return entries


def build_episodic_ttt_v1_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = DEFAULT_CONTROL_SEEDS,
    arms: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Phase 3 TTT-shaped falsifier matrix for the memory-aware optimizer.

    Supersedes ``build_episodic_dw_curation_v1_matrix`` after the
    architecture pivot — the cache must be live at eval time too, not
    only during training. Four-arm 2x2 over (train_uses_cache,
    eval_uses_cache+TTT)::

      Arm | Train side                | Eval side
      ----|---------------------------|------------------------------
      A   | Standard (no cache)       | No TTT, no cache (SOTA shape)
      B   | Cache-curated DW          | TTT with loaded cache (the bet)
      C   | Cache-curated DW          | No TTT (cache vs TTT control)
      D   | Standard (no cache)       | TTT, fresh empty cache

    Topology-locking discipline within each axis:
      - Train-side: A and D share the SOTA control train shape (mirrors
        ``phase0_fastslow_only_control``); B and C share the cache-curated
        DW train shape with locked Dreamworld replay knobs.
      - Eval-side: A and C share the score-only / no-TTT eval shape; B
        and D share the cache-aware TTT eval shape (1 step per chunk on
        the lm_head adapt set).
      - Across all arms: same world_size, budget_seconds, fast/slow
        recipe (interval=64, alpha=0.25, eval_copy=slow), seeds.

    The eval-side fields are RECORDED on the matrix entry for downstream
    analysis but are not yet plumbed into ``run_exp20_full_val_score.py`` (the
    path used by ``run_exp24 --full-val-score``). Wiring them through is
    a separate task. ``run_exp20_eval.py`` already consumes them via
    ``RunConfig.episodic_cache_enabled`` and friends.

    NOTE: Arm B's ``loaded cache`` path requires the trainer to serialize
    ``ckpt['episodic_cache']`` alongside the model weights — until that
    save path lands, Arm B falls back to the fresh-cache path and reduces
    to Arm D. That's a falsifier failure (or a downstream wiring task),
    not a code bug in this builder. The matrix encodes the intended
    contrast; the trainer pivot lands separately.

    Per Decision 0.5 (memory-aware-optimizer-plan): if the σ rule fires
    on the headline contrast, re-invoke this builder with::

        build_episodic_ttt_v1_matrix(
            speed_config=...,
            seed_values=(5012, 7331, 9183),
            arms=("arm_a_no_cache_no_ttt",
                  "arm_b_cache_train_ttt_with_cache",
                  "arm_d_no_cache_train_ttt_only"),
        )

    to escalate. (Arm C, the within-train-side cache-only baseline, is
    omitted from escalation by convention — the topology-equivalence
    reading doesn't tighten with extra seeds.)
    """
    # Train-side locks shared by Arms B and C — the cache-curated DW shape.
    # Lifted from build_episodic_dw_curation_v1_matrix's arm_b_cosine_utility
    # so the new TTT matrix replays the same train recipe; the eval split
    # is the new axis the architecture pivot introduces.
    cache_train_dw_lock = {
        "dreamworld_enabled": True,
        "dreamworld_cache_interval": 16,
        "dreamworld_interval": 16,
        "dreamworld_prefix_tokens": 128,
        "dreamworld_replay_tokens": 64,
        "dreamworld_replay_batch_size": 128,
        "dreamworld_buffer_size": 16,
        "dreamworld_min_size": 2,
        "dreamworld_max_age_steps": 256,
        "episodic_enabled": True,
        "controller_query_enabled": True,
        "controller_query_mode": "cosine_utility_weighted",
        "dreamworld_weight": 0.10,
    }
    # Train-side locks shared by Arms A and D — the SOTA control shape.
    # Mirrors phase0_fastslow_only_control's train-side knobs exactly so
    # Arm A stands as a true control vs the cache-train treatment.
    no_cache_train_lock = {
        "dreamworld_enabled": False,
        "dreamworld_cache_interval": 0,
        "dreamworld_interval": 0,
        "dreamworld_weight": 0.0,
        "dreamworld_replay_batch_size": 0,
    }
    # Locked fast/slow recipe — match phase0_fastslow_only_control.
    fast_slow_lock = {
        "fast_slow_enabled": True,
        "fast_slow_interval": 64,
        "fast_slow_alpha": 0.25,
        "fast_slow_eval_copy": "slow",
    }
    # Eval-side shapes — A == C share no-TTT, B == D share cache-aware TTT.
    #
    # Cache schema fields are pinned identically across ALL FOUR arms (not
    # just the TTT arms) so that if the contrast moves, it's the cache
    # CONTENT (Arm B's loaded entries vs Arm D's empty cache) doing the
    # work, not a cache SHAPE difference. ``eval_episodic_key_rep_dim=-1``
    # is the sentinel that resolves to the trainer's model_dim at cache
    # construction; defaults mirror runner_fast_path.py's
    # _construct_episodic_cache.
    eval_cache_schema_lock = {
        "eval_episodic_cache_capacity": 4096,
        "eval_episodic_span_length": 4,
        "eval_episodic_key_rep_dim": -1,
        "eval_episodic_grace_steps": 1000,
        # W must match the trainer's episodic_fingerprint_window or the
        # cache's stored rolling-hash fingerprints don't align with the
        # controller's queries — silent zero-hit-rate failure. Pinned
        # explicitly on every arm so a future config drift on the train
        # side surfaces as an Arm B vs Arm D shape mismatch (loud) rather
        # than a hit-rate collapse (silent).
        "eval_episodic_fingerprint_window": 8,
    }
    eval_no_ttt_lock = {
        **eval_cache_schema_lock,
        "eval_episodic_cache_enabled": False,
        "eval_steps_per_chunk": 0,
        "eval_adapt_set": "none",
        "eval_episodic_cache_reset_per_doc": False,
    }
    eval_ttt_with_cache_lock = {
        **eval_cache_schema_lock,
        "eval_episodic_cache_enabled": True,
        "eval_steps_per_chunk": 1,
        "eval_adapt_set": "lm_head",
        "eval_episodic_cache_reset_per_doc": False,
    }
    arm_specs: list[tuple[str, dict[str, Any]]] = [
        (
            "arm_a_no_cache_no_ttt",
            {**no_cache_train_lock, **eval_no_ttt_lock},
        ),
        (
            "arm_b_cache_train_ttt_with_cache",
            {**cache_train_dw_lock, **eval_ttt_with_cache_lock},
        ),
        (
            "arm_c_cache_train_no_ttt",
            {**cache_train_dw_lock, **eval_no_ttt_lock},
        ),
        (
            "arm_d_no_cache_train_ttt_only",
            {**no_cache_train_lock, **eval_ttt_with_cache_lock},
        ),
    ]
    if arms is not None:
        allowed = set(arms)
        unknown = allowed - set(EPISODIC_TTT_V1_ARMS)
        if unknown:
            raise ValueError(
                f"unknown arm(s) {sorted(unknown)}; "
                f"allowed: {EPISODIC_TTT_V1_ARMS}"
            )
        arm_specs = [(n, o) for n, o in arm_specs if n in allowed]
    entries: list[dict[str, Any]] = []
    for arm_name, arm_overrides in arm_specs:
        arm = {
            "exp24_mechanism": "episodic_ttt_v1",
            "artifact_impact": ARTIFACT_TRAINING_ONLY,
            **fast_slow_lock,
            **arm_overrides,
        }
        for seed in seed_values:
            entry = _base_entry(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
            )
            entry.update(arm)
            entries.append(
                _named_entry(
                    base=entry,
                    phase="phase3",
                    mechanism="episodic_ttt_v1",
                    arm=f"episodic_ttt_v1_{arm_name}",
                    seed=int(seed),
                )
            )
    return entries


def build_criticality_distillation_first_smoke_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 1,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = DEFAULT_CONTROL_SEEDS,
) -> list[dict[str, Any]]:
    """Eight cells x N seeds on the locked control_fastslow_only_i64a025 base.

    Cell types (4 falsifier + 4 sensitivity, all ride the same fast/slow base):
      - treatment, telemetry, shuffled, budget_only
      - hl_short, hl_long, H_short, H_long

    Every entry carries the locked base knobs (fast/slow_enabled=True,
    interval=64, alpha=0.25, eval_copy=slow, Dreamworld off) plus the
    fused-entropy LM-head flags and a full CD config. Delta across cells
    isolates CD's effect from base noise.
    """
    locked_base = {
        "exp24_mechanism": "fast_slow",
        "artifact_impact": ARTIFACT_TRAINING_ONLY,
        "fast_slow_enabled": True,
        "fast_slow_interval": 64,
        "fast_slow_alpha": 0.25,
        "fast_slow_eval_copy": "slow",
        "dreamworld_enabled": False,
        "dreamworld_cache_interval": 0,
        "dreamworld_interval": 0,
        "dreamworld_weight": 0.0,
        "dreamworld_replay_batch_size": 0,
    }
    cd_defaults = {
        "lm_head_backward_mode": "fused_streaming_cached",
        "lm_head_emit_entropy": True,
        "criticality_distill_enabled": True,
        "criticality_distill_budget_frac": 0.15,
        "criticality_distill_critical_value": 0.95,
        "criticality_distill_trace_half_life_steps": 256.0,
        "criticality_distill_trace_ttl_steps": 1024,
        "criticality_distill_horizon_H": 16,
        "criticality_distill_event_frac": 0.05,
        "criticality_distill_seat_refresh_interval": 64,
        "criticality_distill_min_weighted_events_per_layer": 256.0,
        "criticality_distill_weight": 1e-3,
        "criticality_distill_uniform_pressure": False,
        "criticality_distill_score_permute_before_topk": False,
        "criticality_distill_fixed_random_seats": False,
        "rare_bucket_ce_enabled": True,
        "rare_bucket_ce_num_buckets": 4,
    }
    cells = [
        ("treatment", {}),
        ("telemetry", {"criticality_distill_weight": 0.0}),
        ("shuffled", {"criticality_distill_score_permute_before_topk": True}),
        ("budget_only", {"criticality_distill_fixed_random_seats": True}),
        ("hl_short", {"criticality_distill_trace_half_life_steps": 128.0}),
        ("hl_long", {"criticality_distill_trace_half_life_steps": 512.0}),
        ("H_short", {"criticality_distill_horizon_H": 8}),
        ("H_long", {"criticality_distill_horizon_H": 32}),
    ]
    entries: list[dict[str, Any]] = []
    for arm_name, overrides in cells:
        arm = {**locked_base, **cd_defaults, **overrides}
        for seed in seed_values:
            entry = _base_entry(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
            )
            entry.update(arm)
            entries.append(
                _named_entry(
                    base=entry,
                    phase="cd_first_smoke",
                    mechanism="fast_slow",
                    arm=arm_name,
                    seed=int(seed),
                )
            )
    return entries


def build_criticality_distillation_multiseed_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 1,
    budget_seconds: float = 600.0,
    seed_values: Sequence[int] = (1337, 17, 42, 1234, 2024),
) -> list[dict[str, Any]]:
    """Five arms x N seeds on the same locked base as cd_first_smoke.

    Reduced arm set: keep treatment + 3 falsifiers + H_short (the one
    sensitivity cell that survived the first smoke). Drop hl_short/hl_long/
    H_long — first smoke showed them either collapsing the rare-bucket
    win (hl) or converging on shuffled (H_long). Multi-seed gives paired
    bootstrap CIs on b0/b1 so the -0.10 nat rare-CE gap is bounded by
    real noise, not by a single-seed point estimate.
    """
    locked_base = {
        "exp24_mechanism": "fast_slow",
        "artifact_impact": ARTIFACT_TRAINING_ONLY,
        "fast_slow_enabled": True,
        "fast_slow_interval": 64,
        "fast_slow_alpha": 0.25,
        "fast_slow_eval_copy": "slow",
        "dreamworld_enabled": False,
        "dreamworld_cache_interval": 0,
        "dreamworld_interval": 0,
        "dreamworld_weight": 0.0,
        "dreamworld_replay_batch_size": 0,
    }
    cd_defaults = {
        "lm_head_backward_mode": "fused_streaming_cached",
        "lm_head_emit_entropy": True,
        "criticality_distill_enabled": True,
        "criticality_distill_budget_frac": 0.15,
        "criticality_distill_critical_value": 0.95,
        "criticality_distill_trace_half_life_steps": 256.0,
        "criticality_distill_trace_ttl_steps": 1024,
        "criticality_distill_horizon_H": 16,
        "criticality_distill_event_frac": 0.05,
        "criticality_distill_seat_refresh_interval": 64,
        "criticality_distill_min_weighted_events_per_layer": 256.0,
        "criticality_distill_weight": 1e-3,
        "criticality_distill_uniform_pressure": False,
        "criticality_distill_score_permute_before_topk": False,
        "criticality_distill_fixed_random_seats": False,
        "rare_bucket_ce_enabled": True,
        "rare_bucket_ce_num_buckets": 4,
    }
    cells = [
        ("treatment", {}),
        ("telemetry", {"criticality_distill_weight": 0.0}),
        ("shuffled", {"criticality_distill_score_permute_before_topk": True}),
        ("budget_only", {"criticality_distill_fixed_random_seats": True}),
        ("H_short", {"criticality_distill_horizon_H": 8}),
    ]
    entries: list[dict[str, Any]] = []
    for arm_name, overrides in cells:
        arm = {**locked_base, **cd_defaults, **overrides}
        for seed in seed_values:
            entry = _base_entry(
                speed_config=speed_config,
                world_size=world_size,
                budget_seconds=budget_seconds,
            )
            entry.update(arm)
            entries.append(
                _named_entry(
                    base=entry,
                    phase="cd_multiseed",
                    mechanism="fast_slow",
                    arm=arm_name,
                    seed=int(seed),
                )
            )
    return entries


def build_first_wave_matrix(
    *,
    speed_config: dict[str, Any],
    world_size: int = 8,
    budget_seconds: float = 600.0,
    seeds: Sequence[int] = DEFAULT_CONTROL_SEEDS,
) -> list[dict[str, Any]]:
    return build_first_wave_mechanism_matrix(
        speed_config=speed_config,
        world_size=world_size,
        budget_seconds=budget_seconds,
        seed_values=seeds,
    )


def summarize_control_noise(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "seed": int(row.get("config", {}).get("seed")),
            "bpb": float(row.get("eval", {}).get("bpb")),
            "elapsed_s": float(row.get("train", {}).get("elapsed_s")),
            "tokens_per_sec": float(
                row.get("train", {}).get("aggregate_tokens_per_sec")
            ),
        }
        for row in results
    ]
    bpbs = [row["bpb"] for row in rows]
    tokens_per_sec_values = [row["tokens_per_sec"] for row in rows]
    return {
        "count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "bpb_mean": float(statistics.fmean(bpbs)),
        "bpb_sample_std": float(statistics.stdev(bpbs)) if len(bpbs) > 1 else 0.0,
        "bpb_min": float(min(bpbs)),
        "bpb_max": float(max(bpbs)),
        "elapsed_s_mean": float(statistics.fmean(row["elapsed_s"] for row in rows)),
        "tokens_per_sec_mean": float(statistics.fmean(tokens_per_sec_values)),
    }
