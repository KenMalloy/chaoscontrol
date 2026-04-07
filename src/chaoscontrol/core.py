"""Core SSM components: ChaosSSMCore (diag/paired/full), RMSNorm, FeedForward, criticality_loss."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = F.rms_norm(x, (x.size(-1),), eps=self.eps)
        return normed * self.weight.to(dtype=x.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        hidden = dim * mult
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.fc(x)))


class ChaosSSMCore(nn.Module):
    """SSM recurrence with three A parameterizations: diag, paired, full."""

    def __init__(
        self,
        dim: int,
        *,
        a_mode: str = "diag",
        a_full_rank: int = 8,
        a_full_gamma: float = 0.05,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.a_mode = a_mode
        self.a_full_rank = a_full_rank

        # Shared projections across all modes
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.select_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        if a_mode == "diag":
            self.delta_proj = nn.Linear(dim, dim, bias=False)
            self.log_a = nn.Parameter(torch.zeros((dim,), dtype=torch.float32))

        elif a_mode == "paired":
            assert dim % 2 == 0, f"paired mode requires even dim, got {dim}"
            n_pairs = dim // 2
            self.delta_proj = nn.Linear(dim, dim, bias=False)
            self.log_r = nn.Parameter(torch.zeros((n_pairs,), dtype=torch.float32))
            self.theta = nn.Parameter(
                torch.linspace(0.0, math.pi, n_pairs, dtype=torch.float32)
            )

        elif a_mode == "full":
            # delta_proj outputs scalar step size
            self.delta_proj = nn.Linear(dim, 1, bias=False)
            # Skew-symmetric S: upper triangle params
            n_skew = dim * (dim - 1) // 2
            self.skew_params = nn.Parameter(torch.zeros((n_skew,), dtype=torch.float32))
            # Damping
            self.log_gamma = nn.Parameter(
                torch.full((), math.log(a_full_gamma), dtype=torch.float32)
            )
            # Non-normal low-rank part UV^T
            rank = min(a_full_rank, dim)
            self.U = nn.Parameter(torch.randn(dim, rank) * 0.01)
            self.V = nn.Parameter(torch.randn(dim, rank) * 0.01)

        else:
            raise ValueError(f"unsupported a_mode: {a_mode}")

    def _build_skew_symmetric(self) -> torch.Tensor:
        """Build a dim x dim skew-symmetric matrix from upper-triangle params."""
        S = torch.zeros(self.dim, self.dim, dtype=self.skew_params.dtype,
                        device=self.skew_params.device)
        idx = torch.triu_indices(self.dim, self.dim, offset=1)
        S[idx[0], idx[1]] = self.skew_params
        S = S - S.T
        return S

    def _get_A_full(self) -> torch.Tensor:
        """Build the continuous-time A matrix for full mode: S - gamma*I + UV^T."""
        S = self._build_skew_symmetric()
        gamma = torch.exp(self.log_gamma)
        A_c = S - gamma * torch.eye(self.dim, device=S.device, dtype=S.dtype) + self.U @ self.V.T
        return A_c

    def forward(
        self,
        x: torch.Tensor,
        *,
        rich_b: Any = None,
        return_jacobian_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        batch, seq, dim = x.shape
        state = x.new_zeros((batch, dim))
        outputs = []

        if self.a_mode == "diag":
            a_base = torch.sigmoid(self.log_a).to(dtype=x.dtype)[None, :]
            for idx in range(seq):
                inp = x[:, idx, :]
                delta = F.softplus(self.delta_proj(inp)).clamp_min(1e-4)
                decay = torch.exp(-delta * a_base)
                if rich_b is not None:
                    update = rich_b(inp, state)
                else:
                    select = torch.sigmoid(self.select_proj(inp))
                    candidate = torch.tanh(self.in_proj(inp))
                    update = select * candidate
                state = decay * state + update
                out = torch.sigmoid(self.gate_proj(inp)) * state
                outputs.append(self.out_proj(out))
            y = torch.stack(outputs, dim=1)
            if return_jacobian_stats:
                return y, {"lambda_max": torch.tensor(0.0), "sv_log_var": torch.tensor(0.0)}
            return y

        elif self.a_mode == "paired":
            n_pairs = dim // 2
            for idx in range(seq):
                inp = x[:, idx, :]
                delta = F.softplus(self.delta_proj(inp)).clamp_min(1e-4)
                # Build per-pair rotation+decay
                r = torch.exp(-F.softplus(self.log_r)).to(dtype=x.dtype)  # (n_pairs,)
                cos_t = torch.cos(self.theta).to(dtype=x.dtype)  # (n_pairs,)
                sin_t = torch.sin(self.theta).to(dtype=x.dtype)  # (n_pairs,)
                # Modulate decay by delta (average over pair dims)
                delta_pairs = (delta[:, 0::2] + delta[:, 1::2]) * 0.5  # (batch, n_pairs)
                effective_r = torch.exp(-delta_pairs * (1.0 - r[None, :]))  # (batch, n_pairs)
                # Apply rotation: state is reshaped to (batch, n_pairs, 2)
                s = state.view(batch, n_pairs, 2)
                s0 = s[:, :, 0]  # (batch, n_pairs)
                s1 = s[:, :, 1]  # (batch, n_pairs)
                new_s0 = effective_r * (cos_t[None, :] * s0 - sin_t[None, :] * s1)
                new_s1 = effective_r * (sin_t[None, :] * s0 + cos_t[None, :] * s1)
                decayed = torch.stack([new_s0, new_s1], dim=-1).view(batch, dim)
                if rich_b is not None:
                    update = rich_b(inp, state)
                else:
                    select = torch.sigmoid(self.select_proj(inp))
                    candidate = torch.tanh(self.in_proj(inp))
                    update = select * candidate
                state = decayed + update
                out = torch.sigmoid(self.gate_proj(inp)) * state
                outputs.append(self.out_proj(out))
            y = torch.stack(outputs, dim=1)
            if return_jacobian_stats:
                return y, {"lambda_max": torch.tensor(0.0), "sv_log_var": torch.tensor(0.0)}
            return y

        elif self.a_mode == "full":
            A_c = self._get_A_full().to(dtype=x.dtype)
            sv_log_maxes = []
            sv_log_vars = []
            for idx in range(seq):
                inp = x[:, idx, :]
                delta = F.softplus(self.delta_proj(inp)).clamp(1e-4, 2.0)
                d = delta.mean()
                A_d = torch.matrix_exp(d * A_c)
                proposed = state @ A_d.T
                if rich_b is not None:
                    update = rich_b(inp, state)
                else:
                    select = torch.sigmoid(self.select_proj(inp))
                    candidate = torch.tanh(self.in_proj(inp))
                    update = select * candidate
                state = proposed + update
                out = torch.sigmoid(self.gate_proj(inp)) * state
                outputs.append(self.out_proj(out))
                if return_jacobian_stats:
                    svs = torch.linalg.svdvals(A_d.detach())
                    log_svs = torch.log(svs.clamp_min(1e-8))
                    sv_log_maxes.append(log_svs[0])
                    sv_log_vars.append(log_svs.var())
            y = torch.stack(outputs, dim=1)
            if return_jacobian_stats:
                stats = {
                    "lambda_max": torch.stack(sv_log_maxes).mean(),
                    "sv_log_var": torch.stack(sv_log_vars).mean(),
                }
                return y, stats
            return y

        else:
            raise ValueError(f"unsupported a_mode: {self.a_mode}")


def criticality_loss(
    jacobian_stats: dict[str, torch.Tensor],
    *,
    alpha: float = 0.01,
    beta: float = 0.001,
    target_log_sv: float = -0.13,  # log(0.88) ~ -0.13, slightly subcritical
) -> torch.Tensor:
    """Penalize drift from near-critical regime.

    Targets slightly subcritical dynamics (~0.88 coupling) rather than
    exactly critical (1.0). lambda_max is the mean top log singular value
    of the per-step Jacobian; we penalize its distance from the target.
    L_crit = alpha * (lambda_max - target)^2 + beta * Var(log singular_values)
    """
    lam = jacobian_stats["lambda_max"]
    sv_var = jacobian_stats["sv_log_var"]
    return alpha * (lam - target_log_sv).pow(2) + beta * sv_var
