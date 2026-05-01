#!/usr/bin/env python3
"""Exp27 TTT headline matrix builder.

Single trunk (the winning trunk from exp26) under a compact test-time-training
selector. The default run keeps the legal floor and the strongest
gradient-free TTT candidate:

  1. ``score_only_reset``  — reset SSM state per doc; floor.
  2. ``adaptive_carry``    — source-ordered state carry plus causal online
                             horizon mixing on the packet-clean encode path.

Legacy exploratory calc_types remain registered and can be requested
explicitly:

  * ``carry_state``       — raw SSM state continues across docs.
  * ``dreamworld_eval``   — per-doc dream rollout + backward + SGD.

A single headline entry trains (or loads) one trunk and runs every
requested calc_type as a serial eval pass. The registrations live
in ``chaoscontrol.eval.calc_types``; this builder only references their
names and forwards calibrated hyperparameters.

A fourth calc_type, ``state_replay_within_doc``, was prototyped and
removed: re-passing a doc with state carried across the full pass leaks
future tokens into past positions (pass r=2's prediction at position t
sees the state produced by pass r=1's encode of t+1..T-1). Same
causality break as ``carry_state`` would have without source ordering,
just with the "doc" as the window. No clean SSM analog of depth
recurrence exists without per-position iterative refinement, which is
its own design conversation.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
EXP24 = REPO / "experiments" / "24_training_time_bundle"
EXP26 = REPO / "experiments" / "26_arm"
SRC = REPO / "src"
sys.path.insert(0, str(EXP26))
sys.path.insert(0, str(EXP24))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exp26 import build_validation_matrix as _build_exp26_validation_matrix  # noqa: E402


EXP27_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_DIR = EXP27_DIR / "calibration"
DEFAULT_RESULTS_DIR = EXP27_DIR / "results"
DEFAULT_MANIFEST_PATH = DEFAULT_CALIBRATION_DIR / "manifest.json"

CALC_TYPES_DEFAULT: tuple[str, ...] = (
    "score_only_reset",
    "adaptive_carry",
)

DEFAULT_HEADLINE_SEEDS: tuple[int, ...] = (1337, 2674, 4011)
EXP26_ARM_NAME = "validation_adaptive_residual_memory"


def _registered_calc_types() -> list[str]:
    """Return the names registered by ``chaoscontrol.eval.calc_types``.

    Imported lazily so that callers who only want ``DEFAULT_MANIFEST_PATH``
    do not pay the torch import cost.
    """
    # Importing the package eagerly registers every calc_type module.
    import chaoscontrol.eval.calc_types  # noqa: F401
    from chaoscontrol.eval.ttt_eval import list_registered_calc_types

    return list_registered_calc_types()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a calc_type calibration manifest. Raises if missing or malformed."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"calibration manifest missing: {p}")
    try:
        manifest = json.loads(p.read_text())
    except Exception as exc:
        raise ValueError(f"manifest at {p} is malformed: {exc!r}") from exc
    required = {"calc_type_hyperparams", "calibrated_at"}
    missing = required - set(manifest.keys())
    if missing:
        raise ValueError(
            f"manifest at {p} missing required keys: {sorted(missing)}"
        )
    return manifest


def build_ttt_headline_matrix(
    *,
    speed_config: dict[str, Any],
    calibration_manifest_path: Path,
    checkpoint_path: Path | None = None,
    world_size: int = 4,
    budget_seconds: float = 600.0,
    seed_values: list[int] | None = None,
    calc_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """One entry per seed. Each entry runs every requested calc_type at eval.

    The single eval session per entry is by design: training the trunk
    once per seed is the expensive part, and every calc_type rides on the
    same trained weights. Calibrated hyperparameters for each calc_type
    are read from the manifest and embedded into the entry config.
    """
    seeds = list(seed_values) if seed_values is not None else list(
        DEFAULT_HEADLINE_SEEDS
    )
    requested = list(calc_types) if calc_types is not None else list(
        CALC_TYPES_DEFAULT
    )

    registered = set(_registered_calc_types())
    for name in requested:
        if name not in registered:
            raise ValueError(f"unknown calc_type: {name!r}")

    manifest = load_manifest(Path(calibration_manifest_path))
    hyperparams_by_calc_type: dict[str, dict[str, Any]] = dict(
        manifest.get("calc_type_hyperparams", {})
    )
    calc_type_configs: dict[str, dict[str, Any]] = {
        name: dict(hyperparams_by_calc_type.get(name, {})) for name in requested
    }
    headline_calc_type = (
        "adaptive_carry" if "adaptive_carry" in requested
        else ("score_only_reset" if "score_only_reset" in requested else requested[0])
    )

    ckpt_value = str(checkpoint_path) if checkpoint_path is not None else None
    entries: list[dict[str, Any]] = []
    for seed in seeds:
        exp26_entries = _build_exp26_validation_matrix(
            speed_config=speed_config,
            world_size=int(world_size),
            budget_seconds=float(budget_seconds),
            seed=int(seed),
        )
        arm_entries = [
            e for e in exp26_entries
            if str(e.get("arm", "")) == EXP26_ARM_NAME
        ]
        if len(arm_entries) != 1:
            raise RuntimeError(
                f"expected one Exp26 {EXP26_ARM_NAME!r} entry, "
                f"got {len(arm_entries)}"
            )
        entry = copy.deepcopy(arm_entries[0])
        entry.update(
            {
                "name": f"exp27_ttt_headline_s{int(seed)}",
                "seed": int(seed),
                "exp27_mechanism": "ttt_headline_v1",
                "calc_types": list(requested),
                "calc_type_configs": copy.deepcopy(calc_type_configs),
                "headline_calc_type": headline_calc_type,
                "checkpoint_path": ckpt_value,
            }
        )
        if ckpt_value is not None:
            entry.update(
                {
                    "eval_only": True,
                    "warmup_steps": 0,
                    "max_steps": 0,
                    "restore_after_warmup": False,
                }
            )
        entries.append(entry)
    return entries
