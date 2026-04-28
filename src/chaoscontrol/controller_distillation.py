"""Controller distillation helpers for CRCT.

The rank-3 oracle produces dense targets that answer "would memory help this
token?"  The model-side controller is intentionally small: it predicts that
target from the pre-memory hidden stream and turns the probability into a
continuous, non-negative memory gate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ControllerMLP(nn.Module):
    """Tiny per-token MLP that predicts one memory-help logit."""

    def __init__(self, model_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim or max(1, model_dim // 4))
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def gate_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Map memory-help logits to a continuous injection gate in [0, 1].

    ``p_help <= 0.5`` means no memory.  ``p_help == 0.75`` is half-strength.
    There is deliberately no negative gate in CRCT v1.
    """

    p_help = torch.sigmoid(logits.float())
    return torch.clamp((p_help - 0.5) * 2.0, min=0.0, max=1.0).to(
        dtype=logits.dtype
    )


def controller_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Confidence-weighted BCE for oracle controller targets."""

    if logits.shape != target.shape:
        raise ValueError(
            f"controller_loss shape mismatch: logits={tuple(logits.shape)} "
            f"target={tuple(target.shape)}"
        )
    loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        target.float(),
        reduction="none",
    )
    weight = torch.ones_like(loss)
    if confidence is not None:
        if confidence.shape != logits.shape:
            raise ValueError(
                "controller_loss confidence shape mismatch: "
                f"{tuple(confidence.shape)} != {tuple(logits.shape)}"
            )
        weight = weight * confidence.float()
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                "controller_loss mask shape mismatch: "
                f"{tuple(mask.shape)} != {tuple(logits.shape)}"
            )
        weight = weight * mask.float()

    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


__all__ = [
    "ControllerMLP",
    "controller_loss",
    "gate_from_logits",
]
