#!/usr/bin/env python3
"""Exp27 TTT headline matrix builder.

Single trunk (the winning trunk from exp26) under three test-time-training
strategies, each its own 600s eval session:

  1. ``score_only_reset``  — reset SSM state per doc; floor.
  2. ``carry_state``       — SSM state continues across docs (source-order).
  3. ``dreamworld_eval``   — per-doc dream rollout + backward + SGD.

A single headline entry trains (or loads) one trunk and runs every
requested calc_type as a serial eval pass. The three registrations live
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
SRC = REPO / "src"
sys.path.insert(0, str(EXP24))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exp24 import (  # noqa: E402
    ARTIFACT_CHANGES_WEIGHTS_ONLY,
    _base_entry,
)


EXP27_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_DIR = EXP27_DIR / "calibration"
DEFAULT_RESULTS_DIR = EXP27_DIR / "results"
DEFAULT_MANIFEST_PATH = DEFAULT_CALIBRATION_DIR / "manifest.json"

CALC_TYPES_DEFAULT: tuple[str, ...] = (
    "score_only_reset",
    "carry_state",
    "dreamworld_eval",
)

DEFAULT_HEADLINE_SEEDS: tuple[int, ...] = (1337, 2674, 4011)


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

    ckpt_value = str(checkpoint_path) if checkpoint_path is not None else None
    entries: list[dict[str, Any]] = []
    for seed in seeds:
        entry = _base_entry(
            speed_config=speed_config,
            world_size=int(world_size),
            budget_seconds=float(budget_seconds),
        )
        entry.update(
            {
                "name": f"exp27_ttt_headline_s{int(seed)}",
                "seed": int(seed),
                "exp27_mechanism": "ttt_headline_v1",
                "artifact_impact": ARTIFACT_CHANGES_WEIGHTS_ONLY,
                "calc_types": list(requested),
                "calc_type_configs": copy.deepcopy(calc_type_configs),
                "checkpoint_path": ckpt_value,
            }
        )
        entries.append(entry)
    return entries
