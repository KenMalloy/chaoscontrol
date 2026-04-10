"""WernickeLayer: typed compositional preprocessing with VQ/MoE routing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chaoscontrol.core import RMSNorm


class WernickeLayer(nn.Module):
    """Typed compositional preprocessing between byte embeddings and SSM recurrence.

    Three stages:
      1. Composition — causal 1D convolution composes raw bytes into higher-level units.
      2. Typing — hard bucket assignment via VQ codebook or MoE top-1 routing
         with straight-through estimator for gradients.
      3. Refinement — per-bucket linear projection applied to each unit.

    Returns (refined_output, bucket_ids, balance_loss).
    """

    def __init__(
        self,
        dim: int,
        k_max: int = 16,
        window: int = 8,
        router_type: str = "vq",
        balance_weight: float = 0.01,
        expert_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.expert_dim = expert_dim or dim

        # Stage 1: composition via causal conv1d
        self.compose_conv = nn.Conv1d(dim, dim, kernel_size=window, padding=window - 1, bias=False)
        self.compose_norm = RMSNorm(dim)

        # Stage 2: hard bucket assignment
        self.router_type = router_type
        if router_type == "vq":
            self.codebook = nn.Parameter(torch.randn(k_max, dim) * 0.01)
        elif router_type == "moe":
            self.router = nn.Linear(dim, k_max, bias=False)
        else:
            raise ValueError(f"unsupported router_type: {router_type}")

        # Stage 3: per-bucket refinement
        # When expert_dim < dim, each expert projects down then back up,
        # keeping k_max * expert_dim * dim roughly constant across sweeps.
        if self.expert_dim == dim:
            self.bucket_projs = nn.ModuleList([
                nn.Linear(dim, dim, bias=False) for _ in range(k_max)
            ])
        else:
            self.bucket_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, self.expert_dim, bias=False),
                    nn.Linear(self.expert_dim, dim, bias=False),
                ) for _ in range(k_max)
            ])

        self.k_max = k_max
        self.window = window
        self.balance_weight = balance_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq, dim) — raw byte embeddings.

        Returns:
            refined: (batch, seq, dim) — composed, typed, refined representations.
            bucket_ids: (batch, seq) — integer bucket assignments.
            balance_loss: scalar — penalizes uneven bucket usage.
        """
        batch, seq, dim = x.shape

        # Stage 1: Composition via causal conv1d
        # Conv1d expects (batch, channels, seq)
        h = x.transpose(1, 2)  # (batch, dim, seq)
        h = self.compose_conv(h)  # (batch, dim, seq + window - 1)
        # Causal: slice off the last (window-1) positions to prevent future leakage
        h = h[:, :, :seq]  # (batch, dim, seq)
        h = h.transpose(1, 2)  # (batch, seq, dim)
        h = self.compose_norm(h)

        # Stage 2: Typing — hard bucket assignment
        if self.router_type == "vq":
            # Distances from each position to each codebook vector
            # h: (batch, seq, dim), codebook: (k_max, dim)
            # dists: (batch, seq, k_max)
            dists = torch.cdist(h, self.codebook.unsqueeze(0).expand(batch, -1, -1))
            # Hard assignment: nearest codebook vector
            bucket_ids = dists.argmin(dim=-1)  # (batch, seq)
            # Straight-through: forward uses hard one-hot, backward uses soft distances
            one_hot = F.one_hot(bucket_ids, self.k_max).to(h.dtype)  # (batch, seq, k_max)
            soft_weights = F.softmax(-dists, dim=-1)  # (batch, seq, k_max)
            # Straight-through estimator
            routing_weights = one_hot + soft_weights - soft_weights.detach()
        elif self.router_type == "moe":
            # Linear projection to k_max logits, top-1 selection
            logits = self.router(h)  # (batch, seq, k_max)
            bucket_ids = logits.argmax(dim=-1)  # (batch, seq)
            one_hot = F.one_hot(bucket_ids, self.k_max).to(h.dtype)  # (batch, seq, k_max)
            soft_weights = F.softmax(logits, dim=-1)  # (batch, seq, k_max)
            # Straight-through estimator
            routing_weights = one_hot + soft_weights - soft_weights.detach()
        else:
            raise ValueError(f"unsupported router_type: {self.router_type}")

        # Balance loss: penalize uneven bucket usage
        # Fraction of tokens assigned to each bucket
        bucket_counts = one_hot.sum(dim=(0, 1))  # (k_max,)
        bucket_frac = bucket_counts / bucket_counts.sum().clamp_min(1e-8)
        # Uniform target: 1/k_max for each bucket
        uniform = torch.ones_like(bucket_frac) / self.k_max
        balance_loss = ((bucket_frac - uniform) ** 2).sum() * self.k_max

        # Stage 3: Refinement — per-bucket projection via routing weights
        # Compute all bucket projections and mix via routing weights
        # Stack all projections: for each bucket, project h
        # refined = sum over buckets of routing_weight_k * bucket_proj_k(h)
        refined = torch.zeros_like(h)
        for k in range(self.k_max):
            proj_k = self.bucket_projs[k](h)  # (batch, seq, dim)
            w_k = routing_weights[:, :, k].unsqueeze(-1)  # (batch, seq, 1)
            refined = refined + w_k * proj_k

        return refined, bucket_ids, balance_loss

    def compression_consequence_update(self, bucket_id: int, quality_delta: float, lr: float = 0.01) -> None:
        """Gradient-free update from memory compression outcomes.

        When a merge in the outer model produces a quality drop (quality_delta < 0),
        nudge the router to be more discriminating for that bucket. The typing
        head learns from the *consequences* of its typing decisions.

        Works for both MoE and VQ routers.
        """
        if quality_delta >= 0:
            return  # only update on bad merges
        with torch.no_grad():
            if self.router_type == "moe":
                self.router.weight.data[bucket_id] *= (1.0 + lr * abs(quality_delta))
            elif self.router_type == "vq":
                # Push codebook vector away from its nearest neighbor
                dists = torch.cdist(
                    self.codebook.data.unsqueeze(0),
                    self.codebook.data.unsqueeze(0),
                ).squeeze(0)
                dists[bucket_id, bucket_id] = float("inf")
                nearest = dists[bucket_id].argmin()
                direction = self.codebook.data[bucket_id] - self.codebook.data[nearest]
                direction = direction / (direction.norm() + 1e-8)
                self.codebook.data[bucket_id] += lr * abs(quality_delta) * direction


class HierarchicalWernicke(nn.Module):
    """Two-level Wernicke routing: coarse type -> fine subtype.

    Total buckets = k_coarse * k_fine. Bucket id = coarse * k_fine + fine.
    Each level is a standard WernickeLayer. Two sequential routing decisions
    with small codebooks (e.g. 16 coarse x 16 fine = 256 buckets) achieve
    the same bucket granularity as a single flat-256 layer but with fewer
    total expert parameters, because each level only needs k experts instead
    of k^2. Measured overhead: ~27% SSM wall-clock for hierarchical-256 vs
    ~25% for flat-256, but ~40% fewer expert parameters.
    """

    def __init__(
        self,
        dim: int,
        k_coarse: int,
        k_fine: int,
        window: int = 8,
        router_type: str = "moe",
        balance_weight: float = 0.01,
        expert_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.k_coarse = k_coarse
        self.k_fine = k_fine
        self.total_buckets = k_coarse * k_fine
        self.balance_weight = balance_weight

        self.coarse = WernickeLayer(
            dim=dim, k_max=k_coarse, window=window,
            router_type=router_type, balance_weight=balance_weight,
            expert_dim=expert_dim,
        )
        self.fine = WernickeLayer(
            dim=dim, k_max=k_fine, window=window,
            router_type=router_type, balance_weight=balance_weight,
            expert_dim=expert_dim,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: coarse routing then fine routing.

        Args:
            x: (batch, seq, dim) -- input representations.

        Returns:
            refined: (batch, seq, dim) -- refined representations.
            bucket_ids: (batch, seq) -- composite bucket assignments.
            balance_loss: scalar -- sum of coarse and fine balance losses.
        """
        # Coarse routing
        x, coarse_ids, balance_coarse = self.coarse(x)
        # Fine routing
        x, fine_ids, balance_fine = self.fine(x)
        # Composite bucket id
        bucket_ids = coarse_ids * self.k_fine + fine_ids
        balance_loss = balance_coarse + balance_fine
        return x, bucket_ids, balance_loss
