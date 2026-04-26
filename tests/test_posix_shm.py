"""POSIX shm RAII wrapper (Phase A3 of CPU SSM controller plan).

Pins single-process round-trip and two-process cross-visibility via
os.fork. The Phase A4 ShmRing will compose this wrapper with the
A2 SpscRing template.

Spec: docs/plans/2026-04-26-cpu-ssm-controller.md (Task A3).
"""
from __future__ import annotations

import os
import time

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


SHM_NAME = "/cc_test_posix_shm"  # leading slash per POSIX; <30 chars for macOS


def _force_unlink(name: str) -> None:
    """Idempotent unlink — a previous failed run can leave the region
    persistent on the kernel; a stale region with `O_CREAT` (no
    `O_EXCL`) will succeed but with the previous size and contents,
    masking real bugs. Safe to call when the region does not exist."""
    try:
        _ext.PosixShm.unlink(name)
    except Exception:
        pass


def test_posix_shm_single_process_round_trip():
    """Create, write, read, unlink — same process. Verifies the basic
    RAII path: shm_open(O_CREAT|O_RDWR) + ftruncate + mmap on
    construction, munmap on destruction, shm_unlink on explicit call."""
    _force_unlink(SHM_NAME)
    shm = _ext.PosixShm(SHM_NAME, 4096, True)
    try:
        assert shm.size() == 4096, f"expected size 4096, got {shm.size()}"
        assert shm.name() == SHM_NAME, f"expected name {SHM_NAME!r}, got {shm.name()!r}"
        shm.write_bytes(0, b"hello shm world")
        got = shm.read_bytes(0, 15)
        assert got == b"hello shm world", f"round-trip mismatch: got {got!r}"
    finally:
        _ext.PosixShm.unlink(SHM_NAME)


def test_posix_shm_cross_process_visibility():
    """Producer process writes; consumer process reads. Verify the
    consumer sees the producer's bytes (proves shared mapping, not
    just per-process anonymous mmap masquerading as shm)."""
    _force_unlink(SHM_NAME)
    pid = os.fork()
    if pid == 0:  # child / consumer
        # Wait for producer to write. The 0.5s sleep is the standard
        # band-aid for fork+torch on macOS; if this goes flaky we
        # switch to a SIGUSR1 handshake rather than chasing the
        # symptom with longer sleeps.
        time.sleep(0.5)
        try:
            shm = _ext.PosixShm(SHM_NAME, 4096, False)
            data = shm.read_bytes(0, 13)
            os._exit(0 if data == b"cross-process" else 1)
        except Exception:
            os._exit(2)
    else:  # parent / producer
        try:
            shm = _ext.PosixShm(SHM_NAME, 4096, True)
            shm.write_bytes(0, b"cross-process")
            _, status = os.waitpid(pid, 0)
            assert os.WIFEXITED(status), f"child did not exit cleanly: status={status}"
            assert os.WEXITSTATUS(status) == 0, (
                f"child exit status {os.WEXITSTATUS(status)} "
                "(1=byte mismatch, 2=exception in child)"
            )
        finally:
            _ext.PosixShm.unlink(SHM_NAME)
