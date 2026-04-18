from __future__ import annotations
import torch
import torch.nn as nn


class DeltaModulator:
    """Context manager that attaches forward hooks to every ChaosSSMCore in a model
    to rescale delta_proj output and shift log_a at eval. No gradients involved.
    """

    def __init__(self, module: nn.Module, *, delta_scale: float = 1.0,
                 log_a_shift: float = 0.0, adapt_set_hint: str | None = None):
        self.module = module
        self.delta_scale = float(delta_scale)
        self.log_a_shift = float(log_a_shift)
        # Optional hint so we can fail loud on Axis 1 × Axis 3 log_a overlap.
        self._adapt_set_hint = adapt_set_hint
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._log_a_originals: list[tuple[nn.Parameter, torch.Tensor]] = []

    def _find_cores(self) -> list[nn.Module]:
        from chaoscontrol.core import ChaosSSMCore
        return [m for m in self.module.modules() if isinstance(m, ChaosSSMCore)]

    def __enter__(self):
        # Axis 3 (log_a_shift) is designed ⊥ Axis 1 (log_a adaptation). If the
        # caller is adapting log_a AND shifting it, DeltaModulator will
        # restore the pre-shift value on exit, wiping the adaptation. Caller
        # must avoid this combination — enforced by the driver entry check in
        # Task 8 as well. We assert here as a backstop.
        if self.log_a_shift != 0.0 and getattr(self, "_adapt_set_hint", None) in (
            "log_a", "log_a+delta_proj", "all",
        ):
            raise ValueError(
                "DeltaModulator.log_a_shift is incompatible with adapting log_a "
                "(log_a_shift reverts log_a on exit; Axis 3 is ⊥ Axis 1)."
            )
        scale = self.delta_scale
        for core in self._find_cores():
            if scale != 1.0:
                h = core.delta_proj.register_forward_hook(
                    lambda mod, inp, out, s=scale: out * s
                )
                self._handles.append(h)
            if self.log_a_shift != 0.0:
                # log_a is a Parameter read in forward; we must mutate then restore
                self._log_a_originals.append((core.log_a, core.log_a.detach().clone()))
                with torch.no_grad():
                    core.log_a.add_(self.log_a_shift)
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for param, orig in self._log_a_originals:
            with torch.no_grad():
                param.copy_(orig)
        self._log_a_originals.clear()
