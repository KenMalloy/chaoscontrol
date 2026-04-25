"""SPSC ring buffer over POSIX shared memory (memory-aware optimizer, Task 1.1).

A bounded, single-producer-single-consumer ring with fixed-shape slots over
POSIX shared memory. Producer writes are non-blocking — on full, drops the
OLDEST slot (overwrites head) and bumps a `dropped_count` counter. Consumer
reads in FIFO order.

Built on `multiprocessing.shared_memory.SharedMemory`:
- One large shm holds the slot table as a contiguous numpy buffer of shape
  `(capacity, *slot_shape)` with the configured dtype.
- One small shm holds three int64 counters: `head`, `tail`, `dropped`.
  Counters are monotonic (never modulo); slot index = `counter % capacity`.

SPSC contract:
- Exactly one writer process touches `head` and `dropped`.
- Exactly one reader process touches `tail`.
- On overflow, the writer also bumps `tail` (drops oldest). This is the only
  shared-word write path; the spec calls this "atomic-by-convention". Phase-2
  callers don't depend on read consistency under overflow.
- Multi-writer is a non-feature. Multiple producers should use N separate
  rings, one per producer.

POSIX shm names on darwin are limited (`PSHMNAMLEN` ~30 chars). The counter
shm gets a short suffix (`_c`); keep user-supplied names well under 28 chars.
"""
from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory
from typing import Sequence

import numpy as np


_COUNTER_DTYPE = np.dtype(np.int64)
# Live counters (the writer/reader hot path touches only these three cells).
_COUNTER_HEAD = 0
_COUNTER_TAIL = 1
_COUNTER_DROPPED = 2
_COUNTER_COUNT = 3
# Metadata header (written once by create(), read once by attach() to validate
# the caller's claimed shape/dtype/capacity against the underlying shm).
_META_CAPACITY = 3
_META_DTYPE_NUM = 4
_META_NDIM = 5
_META_SHAPE_BASE = 6
MAX_NDIM = 8
# Total int64 cells in the counter shm: 3 live counters + 3 scalar metadata
# fields + MAX_NDIM shape cells. With MAX_NDIM=8 this is 14 cells = 112 bytes.
_COUNTER_SHM_CELLS = _META_SHAPE_BASE + MAX_NDIM


def _counter_shm_name(name: str) -> str:
    return f"{name}_c"


def _write_metadata(
    counter_shm: SharedMemory,
    *,
    slot_shape: tuple[int, ...],
    dtype: np.dtype,
    capacity: int,
) -> None:
    """Stash (capacity, dtype.num, ndim, shape...) into the counter shm header.

    Caller must have zeroed the buffer first (the unused shape tail stays 0).
    """
    cells = np.ndarray(
        (_COUNTER_SHM_CELLS,), dtype=_COUNTER_DTYPE, buffer=counter_shm.buf,
    )
    cells[_META_CAPACITY] = int(capacity)
    cells[_META_DTYPE_NUM] = int(dtype.num)
    cells[_META_NDIM] = len(slot_shape)
    for i, dim in enumerate(slot_shape):
        cells[_META_SHAPE_BASE + i] = int(dim)


def _dtype_from_num(num: int) -> np.dtype | None:
    """Best-effort reverse lookup of `numpy.dtype.num` for error-message text.

    Returns None if no common scalar dtype matches; callers should fall back
    to printing the numeric `num` in that case. Used only for human-readable
    error messages — equality is established by comparing nums directly.
    """
    candidates = (
        np.float16, np.float32, np.float64,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.bool_, np.complex64, np.complex128,
    )
    for cand in candidates:
        if np.dtype(cand).num == num:
            return np.dtype(cand)
    return None


def _validate_metadata(
    counter_shm: SharedMemory,
    *,
    claimed_slot_shape: tuple[int, ...],
    claimed_dtype: np.dtype,
    claimed_capacity: int,
) -> None:
    """Read metadata from counter shm and compare to caller's claim.

    Raises ValueError naming both the expected (shm) and actual (claimed)
    values for whichever field mismatches first.
    """
    cells = np.ndarray(
        (_COUNTER_SHM_CELLS,), dtype=_COUNTER_DTYPE, buffer=counter_shm.buf,
    )
    actual_capacity = int(cells[_META_CAPACITY])
    actual_dtype_num = int(cells[_META_DTYPE_NUM])
    actual_ndim = int(cells[_META_NDIM])
    actual_shape = tuple(
        int(cells[_META_SHAPE_BASE + i]) for i in range(actual_ndim)
    )

    if actual_capacity != claimed_capacity:
        raise ValueError(
            f"capacity mismatch: shm has {actual_capacity}, "
            f"attach claimed {claimed_capacity}",
        )
    if int(claimed_dtype.num) != actual_dtype_num:
        actual_dtype = _dtype_from_num(actual_dtype_num)
        actual_str = (
            f"{actual_dtype} (num={actual_dtype_num})"
            if actual_dtype is not None
            else f"num={actual_dtype_num}"
        )
        raise ValueError(
            f"dtype mismatch: shm has {actual_str}, "
            f"attach claimed {claimed_dtype} (num={int(claimed_dtype.num)})",
        )
    if actual_shape != claimed_slot_shape:
        raise ValueError(
            f"shape mismatch: shm has {actual_shape}, "
            f"attach claimed {claimed_slot_shape}",
        )


class ShmRing:
    """Bounded SPSC ring over POSIX shared memory; drops oldest on overflow."""

    def __init__(
        self,
        *,
        slot_shape: tuple[int, ...],
        dtype: np.dtype,
        capacity: int,
        slot_shm: SharedMemory,
        counter_shm: SharedMemory,
        owns_unlink: bool,
    ) -> None:
        self._slot_shape = tuple(int(d) for d in slot_shape)
        self._dtype = np.dtype(dtype)
        self._capacity = int(capacity)
        self._slot_shm = slot_shm
        self._counter_shm = counter_shm
        self._owns_unlink = bool(owns_unlink)
        self._closed = False
        self._unlinked = False

        slot_buf_shape = (self._capacity,) + self._slot_shape
        self._slots = np.ndarray(
            slot_buf_shape, dtype=self._dtype, buffer=self._slot_shm.buf,
        )
        self._counters = np.ndarray(
            (_COUNTER_COUNT,), dtype=_COUNTER_DTYPE, buffer=self._counter_shm.buf,
        )

    # ---- factories -----------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        name: str,
        slot_shape: Sequence[int],
        dtype: np.dtype,
        capacity: int,
    ) -> "ShmRing":
        """Producer side. Allocates new shm with `name` (fails if it exists).

        Stashes (capacity, dtype.num, ndim, slot_shape) into a metadata header
        in the counter shm so consumers can validate their `attach()` claim
        against the underlying shm — without that, a typo at the call site
        silently corrupts payloads (POSIX shm rounds up to a page boundary, so
        small mismatched views fit and read garbage).

        Raises ValueError if `len(slot_shape) > MAX_NDIM` (=8). Use cases are
        1-D and 2-D slot shapes; 8 dims is room enough.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive; got {capacity}")
        slot_shape = tuple(int(d) for d in slot_shape)
        if len(slot_shape) > MAX_NDIM:
            raise ValueError(
                f"slot_shape has {len(slot_shape)} dims; MAX_NDIM={MAX_NDIM}",
            )
        dtype = np.dtype(dtype)

        slot_nbytes = int(np.prod((capacity,) + slot_shape)) * dtype.itemsize
        slot_shm = SharedMemory(name=name, create=True, size=slot_nbytes)
        try:
            counter_shm = SharedMemory(
                name=_counter_shm_name(name),
                create=True,
                size=_COUNTER_SHM_CELLS * _COUNTER_DTYPE.itemsize,
            )
        except Exception:
            slot_shm.close()
            slot_shm.unlink()
            raise

        # Zero the slot buffer and the full counter region (live counters +
        # metadata). SharedMemory does not guarantee zeroed contents on every
        # platform. Zero BEFORE writing metadata so the unused shape tail
        # stays 0 and the freshly written metadata isn't clobbered.
        np.ndarray(
            (slot_nbytes,), dtype=np.uint8, buffer=slot_shm.buf,
        )[:] = 0
        np.ndarray(
            (_COUNTER_SHM_CELLS,),
            dtype=_COUNTER_DTYPE,
            buffer=counter_shm.buf,
        )[:] = 0
        _write_metadata(
            counter_shm,
            slot_shape=slot_shape,
            dtype=dtype,
            capacity=capacity,
        )

        return cls(
            slot_shape=slot_shape,
            dtype=dtype,
            capacity=capacity,
            slot_shm=slot_shm,
            counter_shm=counter_shm,
            owns_unlink=True,
        )

    @classmethod
    def attach(
        cls,
        *,
        name: str,
        slot_shape: Sequence[int],
        dtype: np.dtype,
        capacity: int,
    ) -> "ShmRing":
        """Consumer side. Attaches to an existing shm with `name`.

        Validates the caller's claimed (slot_shape, dtype, capacity) against
        metadata stashed by `create()`. Mismatches raise ValueError BEFORE any
        numpy view is constructed — a mismatched view often silently fits
        because POSIX shm rounds up to a page boundary, so we cannot rely on
        ndarray construction to flag the bug.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive; got {capacity}")
        slot_shape = tuple(int(d) for d in slot_shape)
        dtype = np.dtype(dtype)

        slot_shm = SharedMemory(name=name, create=False)
        try:
            counter_shm = SharedMemory(
                name=_counter_shm_name(name), create=False,
            )
        except Exception:
            slot_shm.close()
            raise
        try:
            _validate_metadata(
                counter_shm,
                claimed_slot_shape=slot_shape,
                claimed_dtype=dtype,
                claimed_capacity=capacity,
            )
        except Exception:
            counter_shm.close()
            slot_shm.close()
            raise
        return cls(
            slot_shape=slot_shape,
            dtype=dtype,
            capacity=capacity,
            slot_shm=slot_shm,
            counter_shm=counter_shm,
            owns_unlink=False,
        )

    # ---- producer ------------------------------------------------------------

    def try_write(self, item: np.ndarray) -> None:
        """Non-blocking. On full, drops oldest slot and bumps dropped_count."""
        if not isinstance(item, np.ndarray):
            raise ValueError(
                f"try_write requires numpy.ndarray; got {type(item).__name__}",
            )
        if item.shape != self._slot_shape:
            raise ValueError(
                f"shape mismatch: ring expects {self._slot_shape}, "
                f"got {tuple(item.shape)}",
            )
        if item.dtype != self._dtype:
            raise ValueError(
                f"dtype mismatch: ring expects {self._dtype}, got {item.dtype}",
            )

        head = int(self._counters[_COUNTER_HEAD])
        tail = int(self._counters[_COUNTER_TAIL])
        if head - tail >= self._capacity:
            # Full: drop oldest by advancing tail, bump dropped counter.
            self._counters[_COUNTER_TAIL] = tail + 1
            self._counters[_COUNTER_DROPPED] = (
                int(self._counters[_COUNTER_DROPPED]) + 1
            )

        idx = head % self._capacity
        self._slots[idx] = item
        # Counter writes use Python-int + numpy assignment so an INT64_MAX
        # overflow raises OverflowError loudly. Switching to numpy in-place
        # arithmetic (`+= np.int64(1)`) would silently wraparound — load-
        # bearing semantics; do not "optimize" the int() conversion away.
        self._counters[_COUNTER_HEAD] = head + 1

    # ---- consumer ------------------------------------------------------------

    def try_read(self) -> np.ndarray | None:
        """Non-blocking. Returns oldest unread item or None if empty.

        Returns a COPY (caller-owned), not a view into shared memory.
        """
        head = int(self._counters[_COUNTER_HEAD])
        tail = int(self._counters[_COUNTER_TAIL])
        if head == tail:
            return None
        idx = tail % self._capacity
        out = self._slots[idx].copy()
        self._counters[_COUNTER_TAIL] = tail + 1
        return out

    # ---- introspection -------------------------------------------------------

    def dropped_count(self) -> int:
        """Total number of slots dropped due to overflow since creation."""
        return int(self._counters[_COUNTER_DROPPED])

    # ---- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close handles only (do NOT unlink)."""
        if self._closed:
            return
        # Drop numpy views before closing the underlying buffers — on some
        # platforms SharedMemory.close() raises if there are live mmaps.
        self._slots = None  # type: ignore[assignment]
        self._counters = None  # type: ignore[assignment]
        try:
            self._slot_shm.close()
        finally:
            self._counter_shm.close()
        self._closed = True

    def close_and_unlink(self) -> None:
        """Producer side: close handles AND unlink shm names. Idempotent."""
        if not self._closed:
            self.close()
        if self._unlinked:
            return
        if not self._owns_unlink:
            self._unlinked = True
            return
        try:
            self._slot_shm.unlink()
        except FileNotFoundError:
            pass
        try:
            self._counter_shm.unlink()
        except FileNotFoundError:
            pass
        self._unlinked = True
