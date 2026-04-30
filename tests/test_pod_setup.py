"""Structural tests for pod setup scripts.

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
POD_BOOTSTRAP = REPO / "scripts" / "pod_bootstrap.sh"
POD_BUILD_NATIVE_EXTENSIONS = REPO / "scripts" / "pod_build_native_extensions.sh"
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


class TestPodNativeExtensionBootstrap:
    def test_one_command_bootstrap_builds_native_extensions(self) -> None:
        """The active pod bootstrap must not rely on manual shell history
        for the native extension build. This catches the exact failure mode
        where a pod has CUDA-visible torch but `_C` modules are missing.
        """
        source = POD_BOOTSTRAP.read_text()
        assert "scripts/pod_build_native_extensions.sh" in source
        assert "from chaoscontrol.kernels._lm_head_loss import _C" in source
        assert "from chaoscontrol.kernels._ssm_scan import _C" in source
        assert "write_event_cuda_pack_available()" in source

    def test_one_command_bootstrap_keeps_val_cache_optional(self) -> None:
        """Exp26 setup needs SP16384 shards, not the full scorer val cache.
        The HF-token-gated docs stream should only run when explicitly
        requested. Exp27 calc_types do need that ValCache, so the enabled
        branch must verify it against Natooka's prepared val shard.
        """
        source = POD_BOOTSTRAP.read_text()
        assert "CHAOSCONTROL_BUILD_VAL_CACHE=${CHAOSCONTROL_BUILD_VAL_CACHE:-0}" in source
        assert 'if [ "$CHAOSCONTROL_BUILD_VAL_CACHE" = "1" ]; then' in source
        assert "stream_docs_selected.py" in source
        assert "build_exp20_val_cache.py" in source
        assert "verify_sp16384_eval_cache.py" in source
        assert "not required for Exp26" in source

    def test_native_extension_helper_pins_cuda_home_and_builds_all_required_extensions(
        self,
    ) -> None:
        """The helper encodes the H100 pod lesson: nvcc may exist under
        /usr/local/cuda-12.8 without being on PATH, and the CPU SSM
        controller must build whether the CUDA write-event pack is
        auto-enabled or explicitly unavailable.
        """
        source = POD_BUILD_NATIVE_EXTENSIONS.read_text()
        assert "/usr/local/cuda-12.8" in source
        assert 'export PATH="$WORKSPACE_VENV/bin:$CUDA_HOME/bin:$PATH"' in source
        assert "CHAOSCONTROL_CUDA_ARCH_LIST" in source
        assert "TORCH_CUDA_ARCH_LIST" in source
        assert "CHAOSCONTROL_CPU_SSM_X86_ACCEL" in source
        assert "CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT" in source
        assert "src/chaoscontrol/kernels/_lm_head_loss/setup_ext.py" in source
        assert "src/chaoscontrol/kernels/_cpu_ssm_controller/setup_ext.py" in source
        assert "src/chaoscontrol/kernels/_ssm_scan/setup_ext.py" in source
        assert "write_event_cuda_pack_available()" in source

    def test_native_extension_helper_does_not_force_write_event_pack(self) -> None:
        """Default pod setup should not force a CUDA pack build against a
        mismatched toolkit. setup_ext.py owns the ABI decision: matched
        torch/nvcc versions build the pack, mismatches build the explicit
        CPU-only controller.
        """
        source = POD_BUILD_NATIVE_EXTENSIONS.read_text()
        assert "CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT=${" not in source
        assert "write pack:  ${CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT:-auto}" in source

    def test_cuda13_setup_does_not_force_write_event_pack(self) -> None:
        """The full CUDA13 setup path must stay compatible with torch cu130
        plus a non-13.0 toolkit by leaving the write-event pack on auto.
        """
        source = POD_SETUP.read_text()
        assert "\n    CHAOSCONTROL_CPU_SSM_CUDA_WRITE_EVENT=1 \\\n" not in source
        assert "write_event_cuda_pack_available()" in source
        assert "auto-enables write_event_pack.cu" in source

    def test_cuda13_setup_discovers_venv_site_packages(self) -> None:
        """RunPod templates do not all ship the same Python minor version.
        The CUDA13 wheel path must come from the active venv, not a
        hard-coded ``python3.12`` directory.
        """
        source = POD_SETUP.read_text()
        assert "site.getsitepackages()" in source
        assert "lib/python3.12/site-packages" not in source

    def test_cuda13_setup_exports_wheel_cuda_headers_for_te(self) -> None:
        """TE's source build must use CUDA/CUDNN headers from the venv
        wheels, not the base image's /usr/local/cuda tree.
        """
        source = POD_SETUP.read_text()
        assert "CUDNN_HOME=$PY_SITEPKG/nvidia/cudnn" in source
        assert 'export CUDA_HOME="$CU13_HOME"' in source
        assert 'export CPATH="$CUDNN_HOME/include:$CU13_HOME/include:${CPATH:-}"' in source
        assert "pip install \"${PIP_FLAGS[@]}\" numpy ninja packaging pyyaml" in source

    def test_cuda13_setup_puts_nvcc_on_path_before_te(self) -> None:
        """If TE falls back to source build, its setup hook needs nvcc
        before the TransformerEngine pip install begins.
        """
        source = POD_SETUP.read_text()
        nvcc_pos = source.index("==> 3c/5 installing nvcc/runtime pins before TE build")
        te_pos = source.index("==> 4/5 installing TransformerEngine")
        assert nvcc_pos < te_pos
        assert 'export PATH="$CU13_HOME/bin:$PATH"' in source

    def test_bootstrap_defaults_to_exp27_val_cache_path(self) -> None:
        """The final scorer path is Exp27; the default cache name should
        not send fresh pods into the older Exp23 cache directory.
        """
        source = POD_BOOTSTRAP.read_text()
        assert "VAL_CACHE_DIR=${VAL_CACHE_DIR:-/workspace/cache/exp27_val_16384}" in source
