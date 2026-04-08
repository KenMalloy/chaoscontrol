"""Tests for chaoscontrol.fatigue — dynamic fatigue tracking system."""

import pytest

from chaoscontrol.fatigue import FatigueTracker


class TestFatigueTrackerInit:
    """Fatigue starts at zero and has sensible defaults."""

    def test_starts_at_zero(self):
        ft = FatigueTracker()
        assert ft.score == 0.0

    def test_default_params(self):
        ft = FatigueTracker()
        assert ft.accumulation_rate == 0.02
        assert ft.wake_decay_rate == 0.005
        assert ft.sleep_decay_rate == 0.3
        assert ft.surprise_suppression == 1.0


class TestAccumulationUnderPressure:
    """Low surprise + low improvement + high memory pressure -> fatigue rises."""

    def test_accumulates_under_stagnation(self):
        ft = FatigueTracker()
        for _ in range(200):
            ft.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
        assert ft.score > 0.3, f"Expected significant fatigue, got {ft.score}"

    def test_accumulates_faster_with_higher_pressure(self):
        ft_low = FatigueTracker()
        ft_high = FatigueTracker()
        for _ in range(100):
            ft_low.step(surprise=0.0, improvement_rate=0.0, memory_pressure=0.3)
            ft_high.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
        assert ft_high.score > ft_low.score

    def test_improvement_reduces_pressure(self):
        ft_stagnant = FatigueTracker()
        ft_improving = FatigueTracker()
        for _ in range(100):
            ft_stagnant.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
            ft_improving.step(surprise=0.0, improvement_rate=0.8, memory_pressure=1.0)
        assert ft_stagnant.score > ft_improving.score


class TestSurpriseSuppression:
    """High surprise keeps fatigue LOW -- the model is alert."""

    def test_high_surprise_suppresses_fatigue(self):
        ft = FatigueTracker()
        for _ in range(200):
            ft.step(surprise=2.0, improvement_rate=0.0, memory_pressure=1.0)
        assert ft.score < 0.1, f"Expected low fatigue under high surprise, got {ft.score}"

    def test_surprise_drives_fatigue_down(self):
        ft = FatigueTracker()
        # First accumulate fatigue
        for _ in range(200):
            ft.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
        peak = ft.score
        assert peak > 0.2, "Need meaningful fatigue to test suppression"
        # Now apply high surprise
        for _ in range(50):
            ft.step(surprise=3.0, improvement_rate=0.0, memory_pressure=1.0)
        assert ft.score < peak, "Surprise should drive fatigue down"


class TestClamping:
    """Fatigue stays in [0, 1] regardless of inputs."""

    def test_never_exceeds_one(self):
        ft = FatigueTracker(accumulation_rate=1.0)
        for _ in range(1000):
            ft.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
        assert ft.score <= 1.0

    def test_never_goes_below_zero(self):
        ft = FatigueTracker()
        # Start at zero and apply massive surprise
        for _ in range(100):
            ft.step(surprise=10.0, improvement_rate=1.0, memory_pressure=0.0)
        assert ft.score >= 0.0

    def test_clamped_after_sleep_recovery(self):
        ft = FatigueTracker()
        ft.apply_sleep_recovery(sleep_quality=1.0)
        assert ft.score >= 0.0


class TestSleepRecovery:
    """apply_sleep_recovery reduces fatigue proportional to quality."""

    def test_full_quality_recovery(self):
        ft = FatigueTracker()
        # Accumulate fatigue
        for _ in range(200):
            ft.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
        before = ft.score
        ft.apply_sleep_recovery(sleep_quality=1.0)
        assert ft.score < before

    def test_zero_quality_no_recovery(self):
        ft = FatigueTracker()
        for _ in range(200):
            ft.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
        before = ft.score
        ft.apply_sleep_recovery(sleep_quality=0.0)
        assert ft.score == before

    def test_higher_quality_more_recovery(self):
        ft1 = FatigueTracker()
        ft2 = FatigueTracker()
        for _ in range(200):
            ft1.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
            ft2.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
        ft1.apply_sleep_recovery(sleep_quality=0.3)
        ft2.apply_sleep_recovery(sleep_quality=1.0)
        assert ft2.score < ft1.score


class TestSleepSteps:
    """sleep_steps scales duration by fatigue level."""

    def test_zero_fatigue_half_base(self):
        ft = FatigueTracker()
        assert ft.sleep_steps(base_sleep=128) == 64  # 0.5 * 128

    def test_full_fatigue_double_base(self):
        ft = FatigueTracker(accumulation_rate=1.0)
        for _ in range(1000):
            ft.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
        assert ft.score > 0.99, "Need near-max fatigue"
        duration = ft.sleep_steps(base_sleep=128)
        assert duration >= 250  # ~2.0 * 128 = 256, allow int rounding

    def test_moderate_fatigue_scales_linearly(self):
        ft = FatigueTracker()
        # Manually set fatigue for deterministic test
        ft._fatigue = 0.5
        duration = ft.sleep_steps(base_sleep=128)
        expected = int(128 * (0.5 + 1.5 * 0.5))  # 128 * 1.25 = 160
        assert duration == expected


class TestDynamicBehavior:
    """The system is a dynamic ODE, not a weighted sum."""

    def test_soft_ceiling_slows_accumulation(self):
        """As fatigue approaches 1, accumulation slows due to (1-f) term."""
        ft = FatigueTracker(accumulation_rate=0.1)
        deltas = []
        for _ in range(50):
            before = ft.score
            ft.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
            deltas.append(ft.score - before)
        # Early deltas should be larger than late deltas
        early = sum(deltas[:10])
        late = sum(deltas[40:50])
        assert early > late, "Accumulation should slow near ceiling"

    def test_returns_updated_score(self):
        ft = FatigueTracker()
        result = ft.step(surprise=0.0, improvement_rate=0.0, memory_pressure=1.0)
        assert result == ft.score
