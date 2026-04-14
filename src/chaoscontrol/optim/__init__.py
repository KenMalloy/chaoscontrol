"""Optimizers for ChaosControl training.

Currently provides:
    Muon — Newton-Schulz orthogonalized momentum for matrix params, with
        an inline decoupled-AdamW fallback for everything else.
    LAMB — large-batch Adam with per-layer trust ratio (You et al. 2019).

Additional optimizers may be added alongside Muon and LAMB when new
ablations are stood up; keep this subpackage the single source of truth
for custom optimizer code.
"""
from __future__ import annotations

from chaoscontrol.optim.lamb import LAMB
from chaoscontrol.optim.muon import Muon

__all__ = ["LAMB", "Muon"]
