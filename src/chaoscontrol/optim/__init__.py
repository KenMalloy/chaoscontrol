"""Optimizers for ChaosControl training.

Currently provides:
    Muon — Newton-Schulz orthogonalized momentum for matrix params, with
        an inline decoupled-AdamW fallback for everything else.

Keep this subpackage the single source of truth for custom optimizer code.
"""
from __future__ import annotations

from chaoscontrol.optim.muon import Muon

__all__ = ["Muon"]
