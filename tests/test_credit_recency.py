"""Recency-decayed credit factor for CPU SSM controller C4."""
from __future__ import annotations

import math

import pytest

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_recency_decay_scales_reward_by_gamma_lag():
    assert _ext.recency_decay(1.0, 1000, 860, 0.995) == pytest.approx(
        0.995**140
    )


def test_recency_decay_returns_original_reward_when_steps_match():
    assert _ext.recency_decay(3.25, 17, 17, 0.995) == 3.25


def test_recency_decay_defensively_ignores_negative_lag():
    assert _ext.recency_decay(2.5, 10, 11, 0.995) == 2.5


def test_recency_decay_preserves_negative_reward_sign():
    expected = -4.0 * (0.995**25)

    assert _ext.recency_decay(-4.0, 125, 100, 0.995) == pytest.approx(expected)


def test_recency_decay_large_lag_matches_python_pow_without_underflow():
    actual = _ext.recency_decay(1.0, 12_000, 0, 0.9999)
    expected = 0.9999**12_000

    assert actual > 0.0
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=5e-4, abs=0.0)
