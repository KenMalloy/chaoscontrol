"""Runtime CPU capability detection for the CPU SSM controller extension."""
from __future__ import annotations

import platform

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


EXPECTED_KEYS = {
    "is_x86",
    "has_avx512f",
    "has_amx_tile",
    "has_amx_bf16",
    "os_avx512_enabled",
    "os_amx_enabled",
}


def test_cpu_features_schema_is_boolean_dict():
    features = _ext.cpu_features()

    assert EXPECTED_KEYS <= features.keys()
    for key in EXPECTED_KEYS:
        assert isinstance(features[key], bool), key


def test_non_x86_hosts_report_runtime_helpers_false():
    features = _ext.cpu_features()
    machine = platform.machine().lower()
    expected_x86 = machine in {"x86_64", "amd64", "i386", "i686"}

    assert features["is_x86"] is expected_x86
    if not expected_x86:
        assert features["is_x86"] is False
        assert features["has_avx512f"] is False
        assert features["has_amx_tile"] is False
        assert features["has_amx_bf16"] is False
        assert features["os_avx512_enabled"] is False
        assert features["os_amx_enabled"] is False
        assert _ext.has_avx512f() is False
        assert _ext.has_amx_bf16() is False


def test_x86_runtime_helpers_match_hardware_and_os_state():
    features = _ext.cpu_features()
    if not features["is_x86"]:
        return

    assert _ext.has_avx512f() is (
        features["has_avx512f"] and features["os_avx512_enabled"]
    )
    assert _ext.has_amx_bf16() is (
        features["has_amx_tile"]
        and features["has_amx_bf16"]
        and features["os_amx_enabled"]
    )
