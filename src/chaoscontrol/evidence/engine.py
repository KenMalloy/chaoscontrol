"""Fixed-shape CPU evidence engine wrapper.

The CPU evidence engine is intentionally not a general scorer.  It accepts
only bounded cue frames, owns controller-plane diagnostics, and leaves exact
memory oracle physics on GPU3.
"""
from __future__ import annotations

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


CpuEvidenceEngine = _ext.CpuEvidenceEngine


__all__ = ["CpuEvidenceEngine"]

