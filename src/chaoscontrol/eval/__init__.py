"""Evaluation harnesses for ChaosControl experiments.

Public surface used by experiments today:

- ``ttt_eval`` — registry-driven multi-calc_type eval over ``ValCache``.
  Each calc_type is a separate test-time-training (TTT) strategy that
  runs against the same trained checkpoint and emits its own BPB.
"""
from __future__ import annotations
