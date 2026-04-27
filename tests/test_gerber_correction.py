"""Gerber-statistic off-policy correction for CPU SSM controller C5."""
from __future__ import annotations

import pytest

from chaoscontrol.kernels import _cpu_ssm_controller as _ext


def test_gerber_weight_accepts_positive_agreement_beyond_threshold():
    assert _ext.gerber_weight(1.25, 0.75, 0.5) == 1.0


def test_gerber_weight_accepts_negative_agreement_beyond_threshold():
    assert _ext.gerber_weight(-1.25, -0.75, 0.5) == 1.0


def test_gerber_weight_rejects_sign_disagreement_beyond_threshold():
    assert _ext.gerber_weight(1.25, -0.75, 0.5) == 0.0


def test_gerber_weight_rejects_when_either_logit_is_below_threshold():
    assert _ext.gerber_weight(1.25, 0.49, 0.5) == 0.0


def test_gerber_weight_accepts_exact_on_policy_uniform_margin():
    """A uniform on-policy simplex still needs reward credit to break symmetry."""
    assert _ext.gerber_weight(0.0, 0.0, 0.0) == 1.0


def test_gerber_weight_rejects_inactive_off_policy_margin():
    assert _ext.gerber_weight(0.0, 0.01, 0.5) == 0.0


def test_gerber_weight_treats_equality_at_threshold_as_inactive():
    assert _ext.gerber_weight(0.5, 0.75, 0.5) == 0.0


def test_gerber_weight_clamps_negative_threshold_to_zero():
    assert _ext.gerber_weight(0.01, 0.02, -1.0) == 1.0


def test_rolling_stddev_estimates_alternating_sequence_stddev():
    stats = _ext.RollingStddev(0.99)

    for i in range(1000):
        stats.update(-2.0 if i % 2 == 0 else 2.0)

    assert stats.count == 1000
    assert stats.stddev() == pytest.approx(2.0, rel=0.02, abs=0.0)


@pytest.mark.parametrize("decay", [-0.01, 1.0, 1.01])
def test_rolling_stddev_rejects_invalid_decay(decay):
    with pytest.raises((ValueError, RuntimeError), match="decay"):
        _ext.RollingStddev(decay)
