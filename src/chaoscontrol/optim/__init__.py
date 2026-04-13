"""Optimizers for ChaosControl training.

Currently provides:
    LAMB — large-batch Adam with per-layer trust ratio (You et al. 2019).

Additional optimizers may be added alongside LAMB when new ablations are
stood up; keep this subpackage the single source of truth for custom
optimizer code.
"""
from __future__ import annotations

from chaoscontrol.optim.lamb import LAMB

__all__ = ["LAMB"]
