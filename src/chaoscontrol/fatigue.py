"""Fatigue tracker: dynamic system governing when the model should sleep.

Fatigue accumulates when the model is stagnating (low surprise, low
improvement, high memory pressure) and decays when the model is alert
(high surprise) or during sleep.  This is a differential-equation-style
system, not a weighted sum -- the rate of fatigue change depends on the
current fatigue level itself, creating nonlinear dynamics.
"""

from __future__ import annotations


class FatigueTracker:
    """Dynamic fatigue accumulator that drives sleep timing decisions.

    The core dynamics per step:

        pressure = (1 - improvement_rate) * memory_pressure
        suppression = surprise * surprise_suppression
        df/dt = accumulation_rate * pressure * (1 - f) - suppression * f - wake_decay_rate * f

    High surprise drives ``suppression * f`` which actively reduces fatigue.
    The ``(1 - f)`` term on accumulation creates a soft ceiling (logistic-like).
    Wake decay provides a slow baseline recovery even without surprise.

    Args:
        accumulation_rate: How fast fatigue grows under pressure.
        wake_decay_rate: Passive fatigue decay per step while awake.
        sleep_decay_rate: Fatigue decay multiplier during sleep recovery.
        surprise_suppression: How strongly surprise counteracts fatigue.
    """

    def __init__(
        self,
        accumulation_rate: float = 0.02,
        wake_decay_rate: float = 0.005,
        sleep_decay_rate: float = 0.3,
        surprise_suppression: float = 1.0,
    ) -> None:
        self.accumulation_rate = accumulation_rate
        self.wake_decay_rate = wake_decay_rate
        self.sleep_decay_rate = sleep_decay_rate
        self.surprise_suppression = surprise_suppression
        self._fatigue: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def score(self) -> float:
        """Current fatigue level in [0, 1]."""
        return self._fatigue

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(
        self,
        surprise: float,
        improvement_rate: float,
        memory_pressure: float,
    ) -> float:
        """Advance the fatigue system by one step.

        Args:
            surprise: Current surprise signal (>=0). High surprise
                suppresses fatigue -- the model is alert and should not sleep.
            improvement_rate: How fast the model is improving (0-1).
                Low improvement means stagnation.
            memory_pressure: Memory subsystem load (0-1).
                High pressure when buffers are full / consolidation is needed.

        Returns:
            Updated fatigue score.
        """
        f = self._fatigue

        # Pressure: stagnation scaled by memory load
        pressure = (1.0 - improvement_rate) * memory_pressure

        # Accumulation with soft ceiling via (1 - f)
        accumulation = self.accumulation_rate * pressure * (1.0 - f)

        # Surprise-driven suppression: actively pulls fatigue down
        suppression = self.surprise_suppression * surprise * f

        # Passive wake decay
        decay = self.wake_decay_rate * f

        # Update
        f = f + accumulation - suppression - decay

        # Clamp to valid range
        self._fatigue = max(0.0, min(1.0, f))
        return self._fatigue

    # ------------------------------------------------------------------
    # Sleep interface
    # ------------------------------------------------------------------

    def apply_sleep_recovery(self, sleep_quality: float) -> float:
        """Reduce fatigue after a sleep phase.

        Args:
            sleep_quality: Quality of the sleep phase (0-1).
                Higher quality means more recovery.

        Returns:
            Updated fatigue score after recovery.
        """
        recovery = self.sleep_decay_rate * sleep_quality
        self._fatigue = max(0.0, self._fatigue - recovery)
        return self._fatigue

    def sleep_steps(self, base_sleep: int = 128) -> int:
        """Compute sleep duration scaled by current fatigue.

        Light fatigue (~0.0) yields ~0.5x base duration.
        Heavy fatigue (~1.0) yields ~2.0x base duration.
        Linear interpolation between these extremes.

        Args:
            base_sleep: Baseline sleep duration in steps.

        Returns:
            Scaled sleep duration (integer steps).
        """
        # Linear map: fatigue 0 -> 0.5x, fatigue 1 -> 2.0x
        scale = 0.5 + 1.5 * self._fatigue
        return int(base_sleep * scale)
