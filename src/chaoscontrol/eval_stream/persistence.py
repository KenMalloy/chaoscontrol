from __future__ import annotations
import torch
import torch.nn as nn
from chaoscontrol.core import ChaosSSMCore


class StateManager:
    """Manages per-block recurrence state across chunks and doc boundaries.

    Modes:
      reset:             zero state at each doc boundary.
      carry_state:       preserve state across doc boundaries.
      carry_weights:     reset state, but keep weight deltas (no-op here; handled by not reverting weights).
      carry_both:        preserve state AND weight deltas.
      trainable_h0:      init state from trainable h0 param at doc boundary; reset at chunk boundary.
      trainable_h0+carry: init first doc from h0, carry thereafter.
    """

    def __init__(self, model: nn.Module, *, persistence_mode: str):
        self.model = model
        self.mode = persistence_mode
        self._cores = [m for m in model.modules() if isinstance(m, ChaosSSMCore)]
        self._state: list[torch.Tensor] = []
        self._doc_idx = -1

    def start_doc(self, *, doc_id: int, batch_size: int) -> None:
        self._doc_idx += 1
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        if self.mode in ("reset", "carry_weights"):
            self._state = [torch.zeros(batch_size, c.dim, device=device, dtype=dtype) for c in self._cores]
        elif self.mode in ("carry_state", "carry_both"):
            if not self._state:
                self._state = [torch.zeros(batch_size, c.dim, device=device, dtype=dtype) for c in self._cores]
            # else keep prior state
        elif self.mode == "trainable_h0":
            # `.clone()` on a view of a Parameter breaks autograd flow back to
            # the Parameter — we want gradients on _trainable_h0 to accumulate,
            # so keep the graph edge intact. `.expand()` is a view and can be
            # fed to the model directly (task 3.5 initial_states kwarg takes
            # ownership of batching).
            self._state = [c._trainable_h0.expand(batch_size, -1) for c in self._cores]
        elif self.mode == "trainable_h0+carry":
            if self._doc_idx == 0:
                self._state = [c._trainable_h0.expand(batch_size, -1) for c in self._cores]
            # else keep prior state
        else:
            raise ValueError(f"unknown persistence_mode: {self.mode}")

    def get_state(self) -> list[torch.Tensor]:
        return self._state

    def set_state(self, state: list[torch.Tensor]) -> None:
        self._state = state


def attach_trainable_h0(model: nn.Module) -> None:
    """Add a trainable h0 vector to each ChaosSSMCore. Eval-time only.

    Placed on the core's own device+dtype so subsequent `initial_states`
    threading doesn't trigger implicit copies. `nn.Parameter(...)` is registered
    via attribute assignment because ChaosSSMCore inherits from nn.Module.
    """
    for core in model.modules():
        if isinstance(core, ChaosSSMCore):
            if not hasattr(core, "_trainable_h0"):
                # Use an existing core parameter to pin device+dtype.
                ref = next(core.parameters())
                core._trainable_h0 = nn.Parameter(
                    torch.zeros(1, core.dim, device=ref.device, dtype=ref.dtype)
                )


def detach_trainable_h0(model: nn.Module) -> None:
    """Remove the _trainable_h0 parameter from every core. After this call,
    `model.state_dict()` must contain no `_trainable_h0` keys.
    """
    for core in model.modules():
        if isinstance(core, ChaosSSMCore):
            if hasattr(core, "_trainable_h0"):
                # Attribute delete also removes from _parameters.
                del core._trainable_h0
