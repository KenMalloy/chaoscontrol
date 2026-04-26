"""CPU SSM controller runtime scaffold.

This is the Python-facing ABI for the learned controller. The default path is a
strict fp32 reference implementation; when the optional C++ extension is built,
``prefer_cpp=True`` dispatches to the same contract. AMX/AVX kernels can replace
the C++ inner loops without changing callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from chaoscontrol.kernels._cpu_ssm_controller import _C
except ImportError as e:  # pragma: no cover - extension is optional
    _C = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class CpuSsmControllerWeights:
    """Weights for one diagonal SSM controller step."""

    w_global_in: torch.Tensor
    w_slot_in: torch.Tensor
    decay_global: torch.Tensor
    decay_slot: torch.Tensor
    w_global_out: torch.Tensor
    w_slot_out: torch.Tensor
    bias: torch.Tensor

    def __post_init__(self) -> None:
        self._validate()

    @property
    def feature_dim(self) -> int:
        return int(self.w_global_in.shape[1])

    @property
    def global_dim(self) -> int:
        return int(self.w_global_in.shape[0])

    @property
    def slot_dim(self) -> int:
        return int(self.w_slot_in.shape[0])

    def _validate(self) -> None:
        if self.w_global_in.dim() != 2:
            raise ValueError("w_global_in must be [D_global, F]")
        if self.w_slot_in.dim() != 2:
            raise ValueError("w_slot_in must be [D_slot, F]")
        if self.w_slot_in.shape[1] != self.w_global_in.shape[1]:
            raise ValueError("w_global_in and w_slot_in must share feature dim")
        checks = {
            "decay_global": (self.global_dim,),
            "decay_slot": (self.slot_dim,),
            "w_global_out": (self.global_dim,),
            "w_slot_out": (self.slot_dim,),
            "bias": (),
        }
        for name, shape in checks.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}; "
                    f"got {tuple(value.shape)}"
                )

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "w_global_in": self.w_global_in.detach()
            .to("cpu", torch.float32)
            .contiguous(),
            "w_slot_in": self.w_slot_in.detach()
            .to("cpu", torch.float32)
            .contiguous(),
            "decay_global": self.decay_global.detach()
            .to("cpu", torch.float32)
            .contiguous(),
            "decay_slot": self.decay_slot.detach()
            .to("cpu", torch.float32)
            .contiguous(),
            "w_global_out": self.w_global_out.detach()
            .to("cpu", torch.float32)
            .contiguous(),
            "w_slot_out": self.w_slot_out.detach()
            .to("cpu", torch.float32)
            .contiguous(),
            "bias": self.bias.detach()
            .to("cpu", torch.float32)
            .reshape(())
            .contiguous(),
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "CpuSsmControllerWeights":
        required = (
            "w_global_in",
            "w_slot_in",
            "decay_global",
            "decay_slot",
            "w_global_out",
            "w_slot_out",
            "bias",
        )
        missing = [k for k in required if k not in blob]
        if missing:
            raise KeyError(f"missing CPU SSM controller weight field(s): {missing}")
        return cls(**{
            k: torch.as_tensor(
                blob[k],
                dtype=torch.float32,
                device="cpu",
            ).contiguous()
            for k in required
        })

    def save(self, path: Path | str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), out)

    @classmethod
    def load(cls, path: Path | str) -> "CpuSsmControllerWeights":
        return cls.from_dict(torch.load(path, map_location="cpu", weights_only=False))


@dataclass(frozen=True)
class CpuSsmControllerState:
    global_state: torch.Tensor
    slot_state: torch.Tensor


@dataclass(frozen=True)
class CpuSsmControllerStep:
    global_state: torch.Tensor
    slot_state: torch.Tensor
    logit: torch.Tensor


class CpuSsmControllerRuntime:
    """Mutable controller state for online scoring.

    It owns one global recurrent state and one per-cache-slot recurrent state.
    The current V1 bridge uses it to emit controller logits alongside the
    heuristic-selected replay tags; a later branch can let these logits drive
    ranking directly.
    """

    def __init__(
        self,
        weights: CpuSsmControllerWeights,
        *,
        capacity: int,
        prefer_cpp: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive; got {capacity}")
        self.weights = weights
        self.prefer_cpp = bool(prefer_cpp)
        self.global_state = torch.zeros(weights.global_dim, dtype=torch.float32)
        self.slot_state = torch.zeros(
            int(capacity), weights.slot_dim, dtype=torch.float32,
        )

    def score_slot(self, features: torch.Tensor, *, slot: int) -> torch.Tensor:
        if not 0 <= int(slot) < int(self.slot_state.shape[0]):
            raise IndexError(f"slot {slot} out of range")
        state = CpuSsmControllerState(
            global_state=self.global_state,
            slot_state=self.slot_state[int(slot)],
        )
        out = forward_step(
            features,
            state,
            self.weights,
            prefer_cpp=self.prefer_cpp,
        )
        self.global_state.copy_(out.global_state)
        self.slot_state[int(slot)].copy_(out.slot_state)
        return out.logit


def cpp_available() -> bool:
    return _C is not None


def require_cpp() -> None:
    if _C is None:  # pragma: no cover - depends on local build
        raise ImportError(
            "chaoscontrol.kernels._cpu_ssm_controller._C is not built; rerun "
            "`pip install -e .` or the extension setup hook. "
            f"Original import error: {_IMPORT_ERROR!r}"
        )


def forward_step(
    features: torch.Tensor,
    state: CpuSsmControllerState,
    weights: CpuSsmControllerWeights,
    *,
    prefer_cpp: bool = True,
) -> CpuSsmControllerStep:
    """Run one diagonal SSM controller step."""
    f = _cpu_f32(features, expected_shape=(weights.feature_dim,))
    g = _cpu_f32(state.global_state, expected_shape=(weights.global_dim,))
    s = _cpu_f32(state.slot_state, expected_shape=(weights.slot_dim,))
    w = weights.to_dict()
    if prefer_cpp and _C is not None:
        out_g, out_s, logit = _C.forward_step(
            f,
            g,
            s,
            w["w_global_in"],
            w["w_slot_in"],
            w["decay_global"],
            w["decay_slot"],
            w["w_global_out"],
            w["w_slot_out"],
            float(w["bias"].item()),
        )
        return CpuSsmControllerStep(out_g, out_s, logit.reshape(()))

    out_g = w["decay_global"] * g + torch.mv(w["w_global_in"], f)
    out_s = w["decay_slot"] * s + torch.mv(w["w_slot_in"], f)
    logit = out_g.dot(w["w_global_out"]) + out_s.dot(w["w_slot_out"]) + w["bias"]
    return CpuSsmControllerStep(
        out_g.contiguous(),
        out_s.contiguous(),
        logit.reshape(()),
    )


def _cpu_f32(t: torch.Tensor, *, expected_shape: tuple[int, ...]) -> torch.Tensor:
    out = torch.as_tensor(t, dtype=torch.float32, device="cpu").contiguous()
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"expected shape {expected_shape}; got {tuple(out.shape)}")
    return out
