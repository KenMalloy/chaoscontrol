"""Optional C++ reference runtime for the CPU SSM controller."""
from __future__ import annotations

from typing import Any

# Import torch first so dyld picks up libc10/libtorch_cpu before the
# extension's `.so` resolves @rpath references on macOS. Without this,
# `from . import _C` raises an opaque "Library not loaded: @rpath/libc10.dylib"
# even when the build succeeded.
try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - torch missing means extension unusable anyway
    pass

_C: Any
try:
    from . import _C  # type: ignore[attr-defined]
except ImportError:
    _C = None


def _missing_extension(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "chaoscontrol.kernels._cpu_ssm_controller._C is not built; "
        "run `python setup_ext.py build_ext --inplace` from the kernel "
        "package directory."
    )


# Wire-event introspection helpers (Phase A1). Re-exported at package
# level so callers can `from chaoscontrol.kernels import _cpu_ssm_controller`
# and call `wire_event_sizes()` directly without reaching into `_C`.
wire_event_sizes = (
    _C.wire_event_sizes if _C is not None else _missing_extension
)
wire_event_alignments = (
    _C.wire_event_alignments if _C is not None else _missing_extension
)
wire_event_constants = (
    _C.wire_event_constants if _C is not None else _missing_extension
)

__all__ = [
    "_C",
    "wire_event_sizes",
    "wire_event_alignments",
    "wire_event_constants",
]
