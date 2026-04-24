#!/usr/bin/env python3
"""Exp24 training-time bundle matrices and summaries."""

from __future__ import annotations

import copy
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
