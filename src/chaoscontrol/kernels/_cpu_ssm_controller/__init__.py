"""Optional C++ reference runtime for the CPU SSM controller."""
from __future__ import annotations

from typing import Any

_C: Any
try:
    from . import _C  # type: ignore[attr-defined]
except ImportError:
    _C = None

__all__ = ["_C"]
