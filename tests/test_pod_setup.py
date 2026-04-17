"""Structural tests for ``scripts/pod_setup_cuda13.sh``.

The script's idempotent fast-path used to probe only ``transformer_engine``
before declaring "Pod ready" — a pod with working TE but missing
sentencepiece / pytest / editable chaoscontrol install would skip the
full install, then fail at first test import, wasting pod time and
obscuring which dep went missing. Fixed 2026-04-17.

These tests are structural rather than executable: the script is
pod-specific (CUDA 13, NVIDIA index URLs, /etc/ld.so.conf.d writes)
and can't be sanely run on a developer laptop. What we CAN guarantee
statically is that every dep the full install produces also appears
in the fast-path probe — a grep-based check that future edits can't
silently drop.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POD_SETUP = REPO / "scripts" / "pod_setup_cuda13.sh"


# Every Python package the full install produces that the pod is
# expected to import at runtime. Keys are the import names; values are
# a human-readable reason to help future maintainers understand why the
# dep is on the probe. Don't add packages not imported by the runtime
# training path.
REQUIRED_IMPORTS = {
    "torch": "training + DDP",
    "transformer_engine": "fp8 path (Test 10, Exp 19 Phase 1 parallel track)",
    "sentencepiece": "SP16384 tokenizer (Test 2 winner)",
    "numpy": "RNG + dataset plumbing",
    "pytest": "in-session test runs on the pod",
    "chaoscontrol": "editable install must be present",
}


def _fast_path_block() -> str:
    """Extract the fast-path probe from the pod setup script.

    The probe is a heredoc between ``python3 - <<'PROBE'`` and ``PROBE``.
    Any test based on the string is brittle to unrelated edits above /
    below; scoping to the probe block keeps the assertion targeted.
    """
    source = POD_SETUP.read_text()
    start_marker = "python3 - <<'PROBE'"
    end_marker = "\nPROBE\n"
    start = source.index(start_marker) + len(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


class TestPodSetupFastPathProbe:
    def test_probe_covers_every_required_import(self) -> None:
        """Fast-path probe must import every dep the full install installs."""
        probe = _fast_path_block()
        missing = [
            name for name in REQUIRED_IMPORTS
            if f"import {name}" not in probe
        ]
        assert not missing, (
            f"fast-path probe at {POD_SETUP} is missing required imports "
            f"(will skip reinstall with these deps absent): {missing}"
        )

    def test_probe_constructs_a_te_linear(self) -> None:
        """``import transformer_engine`` alone is not enough — the NVIDIA
        cublas ABI mismatch that motivated this entire script passes
        import and fails construction. Keep the Linear smoke in the
        probe so a future edit simplifying it back to "import-only"
        re-introduces the original bug.
        """
        probe = _fast_path_block()
        assert "te.Linear(" in probe, (
            "fast-path probe must construct a TE Linear; import alone "
            "cannot catch cublas ABI mismatches"
        )
