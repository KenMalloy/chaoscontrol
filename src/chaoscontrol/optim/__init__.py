"""Optimizers for ChaosControl training.

Currently provides:
    Muon — Newton-Schulz orthogonalized momentum for matrix params, with
        an inline decoupled-AdamW fallback for everything else.
    LAMB — large-batch Adam with per-layer trust ratio (You et al. 2019).
    SemanticOptimizer — Muon with SSM-channel-coupled momentum time constants.
    ScarcityAwareOptimizer — Muon-compatible rare-event optimizer.

Additional optimizers may be added alongside Muon and LAMB when new
ablations are stood up; keep this subpackage the single source of truth
for custom optimizer code.
"""
from __future__ import annotations

from chaoscontrol.optim.lamb import LAMB
from chaoscontrol.optim.muon import Muon
from chaoscontrol.optim.param_groups import (
    SPECTRAL_SUFFIXES,
    build_optimizer_params,
    classify_param,
    ssm_three_group_params,
)
from chaoscontrol.optim.scopt import ScarcityAwareOptimizer
from chaoscontrol.optim.semantic import SemanticOptimizer

__all__ = [
    "LAMB",
    "Muon",
    "ScarcityAwareOptimizer",
    "SemanticOptimizer",
    "SPECTRAL_SUFFIXES",
    "build_optimizer_params",
    "classify_param",
    "ssm_three_group_params",
]
