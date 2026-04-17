"""Bespoke fp8 Linear — drop-in replacement for ``te.Linear``.

Scaffold stage (Task 1B-1): ``FusedFP8Linear`` is a thin wrapper that
INTERNALLY CALLS ``te.Linear`` under a ``te.fp8_autocast`` context. The
public Module API is frozen here so the parity harness in
``tests/test_fp8_linear.py`` can be written and run against a trivially-
correct implementation. When the real fused kernel lands in Task 1B-2,
only the body of :meth:`FusedFP8Linear.forward` changes; the Module API,
parameter layout, and test harness carry over unchanged.

Design choices worth flagging:

* The ``weight`` (and optional ``bias``) live on the ``FusedFP8Linear``
  module itself, NOT on the internal ``te.Linear`` delegate. This keeps
  the optimizer param list stable across the 1B-1 → 1B-2 swap: the
  tensor identity that the optimizer tracks is the one on this wrapper,
  and 1B-2 will replace the TE delegate with a direct
  ``torch.ops.aten._scaled_mm`` call path operating on these same
  tensors. At scaffold stage we ferry the storage into the TE delegate
  via ``.weight.data = self.weight.data`` (same underlying tensor).
* ``transformer_engine`` is imported LAZILY inside ``__init__`` and
  ``forward``, not at module top. This lets the class be imported on
  TE-less hosts (dev macs) and lets tests that don't construct an
  instance (for example, CPU-only parameter-shape checks that go
  through a different path) run without TE installed. Module-level
  import of this file does not touch TE.
* ``fp8_autocast`` management is INTERNAL to ``forward``. Callers do
  not need to wrap invocations in ``with te.fp8_autocast(...)``.

TODO(1B-2): when the fused kernel lands, the following change here:
  1. ``__init__`` stops constructing the internal ``te.Linear`` delegate
     (``self._te_delegate``). The raw ``weight`` / ``bias`` Parameters
     stay exactly as they are.
  2. ``forward`` stops entering ``te.fp8_autocast`` and stops calling
     ``self._te_delegate(x)``. It calls a new ``_fused_forward`` helper
     that dispatches ``torch.ops.aten._scaled_mm`` with explicit scale
     tensors computed from on-device amax history buffers. See the
     plan's Task 1B-2 pseudocode for the call shape.
  3. New module-level state (amax history ring buffers) gets registered
     in ``__init__`` via ``register_buffer(..., persistent=False)``.
  4. :meth:`from_nn_linear` is unchanged — still copies weight/bias
     from a stock ``nn.Linear``.
  5. Tests in ``tests/test_fp8_linear.py`` remain the same; they become
     meaningful parity checks (not trivial) because the bespoke path
     will no longer delegate to TE.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FusedFP8Linear(nn.Module):
    """fp8-path ``nn.Linear`` that matches the ``te.Linear`` output contract.

    At scaffold stage (Task 1B-1), the ``forward`` path delegates to
    ``te.Linear`` inside a ``te.fp8_autocast(enabled=True)`` block, so
    the output is byte-equal to stock TE. Task 1B-2 will replace the
    delegate with a direct ``torch.ops.aten._scaled_mm`` call.

    Args:
        in_features: inner dim of the matmul.
        out_features: outer dim of the matmul.
        bias: whether to allocate a bias parameter.
        device: device for ``weight`` / ``bias``. ``None`` means default
            (usually CPU on import, moved later via ``.to(...)``).
        dtype: master-weight dtype. bf16 by default to match the rest
            of the training loop; the fp8 cast happens inside forward
            via TE (scaffold) or ``_scaled_mm`` (fused).

    Raises:
        ImportError: if ``transformer_engine`` is not installed. Raised
            lazily in ``__init__`` so merely importing the module on a
            TE-less host does not crash.
    """

    in_features: int
    out_features: int
    weight: nn.Parameter
    # bias is Optional[nn.Parameter]; declared via ``register_parameter``
    # below so static attribute access is ``self.bias`` either way.

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Allocate master weight on this module so the optimizer sees a
        # stable param identity across the 1B-1 → 1B-2 swap.
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((out_features,), device=device, dtype=dtype),
            )
        else:
            # ``register_parameter(name, None)`` is the idiomatic way to
            # declare an optional parameter as absent; ``self.bias`` then
            # returns None and is not included in ``.parameters()``.
            self.register_parameter("bias", None)

        # Default-initialize to match ``nn.Linear``'s kaiming-uniform
        # scheme. Not critical at scaffold stage (tests that need known
        # weights use ``from_nn_linear`` or explicit copy_), but keeps
        # bare construction usable.
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Lazy TE import — raises ImportError on TE-less hosts. Keeping
        # the import inside __init__ (not at module top) means the
        # module itself imports cleanly on dev macs; only constructing
        # an instance requires TE.
        # TODO(1B-2): drop this delegate entirely once _scaled_mm path
        # lands. The raw weight/bias above become the sole state.
        try:
            import transformer_engine.pytorch as te  # type: ignore[import-not-found]
        except Exception as exc:
            raise ImportError(
                "FusedFP8Linear requires transformer_engine at scaffold "
                "stage (Task 1B-1). Install with "
                "`pip install transformer-engine[pytorch]` on a CUDA host, "
                "or skip FusedFP8Linear construction on TE-less hosts."
            ) from exc

        # Build the delegate with the same shape + bias flag. ``params_dtype``
        # locks the TE master-weight storage dtype to match ours so the
        # ``.data.copy_`` ferry below does not silently widen.
        self._te_delegate = te.Linear(
            in_features,
            out_features,
            bias=bias,
            params_dtype=dtype,
            device=device,
        )
        # Replace the delegate's Parameters with ours so autograd tracks
        # grads on self.weight / self.bias (the ones the optimizer sees).
        # Sharing via ``.data =`` only aliases storage — the TE Parameter
        # is still a separate autograd leaf, and self.weight.grad stays
        # None after backward. Parameter replacement (delete, then
        # re-assign via ``Module.__setattr__`` which re-registers) keeps
        # one autograd leaf. This is a scaffold shortcut; the 1B-2 path
        # reads self.weight directly and drops the delegate entirely.
        del self._te_delegate.weight
        self._te_delegate.weight = self.weight
        if bias:
            assert self.bias is not None
            del self._te_delegate.bias
            self._te_delegate.bias = self.bias

    @classmethod
    def from_nn_linear(cls, m: nn.Linear) -> "FusedFP8Linear":
        """Adopt an existing ``nn.Linear`` — copy its weight + bias in.

        The returned instance lives on the same device/dtype as ``m``.
        Weight and bias values are byte-copied from ``m``, so call sites
        that swap a pre-initialized ``nn.Linear`` for a fused fp8 one
        preserve training state exactly.
        """
        has_bias = m.bias is not None
        out = cls(
            m.in_features,
            m.out_features,
            bias=has_bias,
            device=m.weight.device,
            dtype=m.weight.dtype,
        )
        with torch.no_grad():
            out.weight.data.copy_(m.weight.data)
            if has_bias:
                assert out.bias is not None
                assert m.bias is not None
                out.bias.data.copy_(m.bias.data)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fp8 linear. Enters ``te.fp8_autocast`` internally.

        Scaffold stage: delegates to the internal ``te.Linear``. The
        ``fp8_autocast`` context ensures the delegate emits fp8 matmul
        kernels regardless of whether the caller set up its own context.

        TODO(1B-2): replace the body below with a call to
        ``_fused_forward(x, self.weight, ...)`` that dispatches
        ``torch.ops.aten._scaled_mm`` directly. At that point the
        ``te.fp8_autocast`` context is no longer needed (we own the
        scaling) and ``self._te_delegate`` is removed.
        """
        # Lazy import — same justification as __init__. If we got this
        # far the delegate exists, so TE is available; but keeping the
        # import local avoids a module-level coupling.
        import transformer_engine.pytorch as te  # type: ignore[import-not-found]

        with te.fp8_autocast(enabled=True):
            return self._te_delegate(x)

    def extra_repr(self) -> str:
        # Match ``nn.Linear.extra_repr`` so model print trees stay readable.
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
