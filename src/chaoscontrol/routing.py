"""Routing modules: RichBNN and DistributedB with hub/assembly/hybrid topologies."""
from __future__ import annotations

import torch
import torch.nn as nn

from chaoscontrol.core import RMSNorm


class RichBNN(nn.Module):
    """Single NN input coupling: B(x, h).
    Two-layer MLP with bottleneck taking concat(x, h).
    """

    def __init__(self, dim: int, bottleneck: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, dim, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, h], dim=-1))


class DistributedB(nn.Module):
    """Distributed semantic network input coupling with three topologies.

    K sub-networks each see a different learned view of state and produce
    partial routing signals. The topology determines coordination:

    topology="hub":       no interaction, central MLP aggregates
    topology="assembly":  lateral message passing for N steps, then mean
    topology="hybrid":    message passing + hub sees pre and post settlement
    """

    def __init__(
        self,
        dim: int,
        num_subnets: int = 4,
        bottleneck: int = 32,
        topology: str = "hub",
        settling_steps: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_subnets = num_subnets
        self.topology = topology
        self.settling_steps = settling_steps

        subnet_dim = max(dim // num_subnets, 4)

        self.view_projs = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(num_subnets)]
        )
        self.subnets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim * 2, subnet_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(subnet_dim, dim, bias=False),
                )
                for _ in range(num_subnets)
            ]
        )

        if topology in ("assembly", "hybrid"):
            self.lateral = nn.Linear(dim * num_subnets, dim, bias=False)
            self.settling_norm = RMSNorm(dim)  # stabilize settling loop

        if topology == "hub":
            hub_input_dim = dim * num_subnets
            self.hub = nn.Sequential(
                nn.Linear(hub_input_dim, bottleneck, bias=False),
                nn.SiLU(),
                nn.Linear(bottleneck, dim, bias=False),
            )
        elif topology == "hybrid":
            hub_input_dim = dim * num_subnets * 2  # pre + post settlement
            self.hub = nn.Sequential(
                nn.Linear(hub_input_dim, bottleneck, bias=False),
                nn.SiLU(),
                nn.Linear(bottleneck, dim, bias=False),
            )

        self.out_act = nn.Tanh()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # 1. Compute per-subnet views of state
        views = [proj(h) for proj in self.view_projs]

        # 2. Compute partial signals from each subnet
        partials = [
            subnet(torch.cat([x, view], dim=-1))
            for subnet, view in zip(self.subnets, views)
        ]

        # 3. Topology-specific aggregation
        if self.topology == "hub":
            combined = torch.cat(partials, dim=-1)
            return self.out_act(self.hub(combined))

        elif self.topology == "assembly":
            for _ in range(self.settling_steps):
                all_cat = torch.cat(partials, dim=-1)
                update = self.settling_norm(self.lateral(all_cat))
                partials = [p + update for p in partials]
            mean_out = torch.stack(partials, dim=0).mean(dim=0)
            return self.out_act(mean_out)

        elif self.topology == "hybrid":
            pre_settlement = torch.cat(partials, dim=-1)
            for _ in range(self.settling_steps):
                all_cat = torch.cat(partials, dim=-1)
                update = self.settling_norm(self.lateral(all_cat))
                partials = [p + update for p in partials]
            post_settlement = torch.cat(partials, dim=-1)
            combined = torch.cat([pre_settlement, post_settlement], dim=-1)
            return self.out_act(self.hub(combined))

        else:
            raise ValueError(f"unsupported topology: {self.topology}")
