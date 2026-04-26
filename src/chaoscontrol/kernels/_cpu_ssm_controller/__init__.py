"""Optional C++ reference runtime for the CPU SSM controller."""
from __future__ import annotations

from typing import Any

# Import torch first so dyld picks up libc10/libtorch_cpu before the
# extension's `.so` resolves @rpath references on macOS. Without this,
# `from . import _C` raises an opaque "Library not loaded: @rpath/libc10.dylib"
# even when the build succeeded. Sibling kernels (`_cublaslt`,
# `_lm_head_loss`) use this same plain top-level import.
# TODO(setup_ext): the long-term fix is to add
#   extra_link_args=["-Wl,-rpath,@loader_path/../../torch/lib"]
# to the CppExtension in `setup_ext.py` so the dylib resolves @rpath
# without depending on import order. Out of scope for Phase A1.
import torch  # noqa: F401

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
wire_event_min_slot_alignment = (
    _C.wire_event_min_slot_alignment if _C is not None else _missing_extension
)
wire_event_constants = (
    _C.wire_event_constants if _C is not None else _missing_extension
)

# SpscRing test fixture (Phase A2). Exposes the SpscRing<uint64_t, 1024>
# instantiation bound in cpu_ssm_controller.cpp so tests/test_spsc_ring.py
# can drive it without reaching into `_C`. The real wire-event ring
# instantiations land in Phase A4 (ShmRing).
SpscRingU64x1024 = (
    _C.SpscRingU64x1024 if _C is not None else _missing_extension
)

# POSIX shm RAII wrapper (Phase A3). Exposes the C++ class binding from
# cpu_ssm_controller.cpp so tests/test_posix_shm.py can drive it
# without reaching into `_C`. Phase A4's ShmRing composes this with the
# A2 SpscRing template.
PosixShm = (
    _C.PosixShm if _C is not None else _missing_extension
)

# ShmRing test fixture (Phase A4). Exposes the
# ShmRing<uint64_t, 1024> instantiation bound in cpu_ssm_controller.cpp
# so tests/test_shm_ring.py can drive it without reaching into `_C`.
# The real wire-event ring instantiations are below (Phase A5); the
# per-rank lifecycle that allocates them lands in B4.
ShmRingU64x1024 = (
    _C.ShmRingU64x1024 if _C is not None else _missing_extension
)

# Real wire-event ShmRing instantiations (Phase A5). Capacities chosen
# per the design doc's per-rank throughput estimates:
#   WriteEvent     × 16384 ≈ 9.5MB region per rank (5.7MB/s at 2M tok/s)
#   QueryEvent     × 16384 ≈ 9MB
#   ReplayOutcome  × 8192  ≈ 770KB (640KB/s replay traffic)
# All powers of 2 to satisfy SpscRing's mask-based-modulo static_assert.
# Each class accepts/returns Python dicts whose keys match the non-pad
# fields of the corresponding wire-event struct in src/wire_events.h.
ShmRingWriteEvent = (
    _C.ShmRingWriteEvent if _C is not None else _missing_extension
)
ShmRingQueryEvent = (
    _C.ShmRingQueryEvent if _C is not None else _missing_extension
)
ShmRingReplayOutcome = (
    _C.ShmRingReplayOutcome if _C is not None else _missing_extension
)

__all__ = [
    "_C",
    "wire_event_sizes",
    "wire_event_min_slot_alignment",
    "wire_event_constants",
    "SpscRingU64x1024",
    "PosixShm",
    "ShmRingU64x1024",
    "ShmRingWriteEvent",
    "ShmRingQueryEvent",
    "ShmRingReplayOutcome",
]
